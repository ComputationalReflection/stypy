
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import os
7: 
8: from matplotlib._pylab_helpers import Gcf
9: from matplotlib.backend_bases import (
10:     _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2,
11:     TimerBase)
12: 
13: from matplotlib.figure import Figure
14: from matplotlib import rcParams
15: 
16: from matplotlib.widgets import SubplotTool
17: 
18: import matplotlib
19: from matplotlib.backends import _macosx
20: 
21: from .backend_agg import RendererAgg, FigureCanvasAgg
22: 
23: 
24: ########################################################################
25: #
26: # The following functions and classes are for pylab and implement
27: # window/figure managers, etc...
28: #
29: ########################################################################
30: 
31: 
32: class TimerMac(_macosx.Timer, TimerBase):
33:     '''
34:     Subclass of :class:`backend_bases.TimerBase` that uses CoreFoundation
35:     run loops for timer events.
36: 
37:     Attributes
38:     ----------
39:     interval : int
40:         The time between timer events in milliseconds. Default is 1000 ms.
41:     single_shot : bool
42:         Boolean flag indicating whether this timer should operate as single
43:         shot (run once and then stop). Defaults to False.
44:     callbacks : list
45:         Stores list of (func, args) tuples that will be called upon timer
46:         events. This list can be manipulated directly, or the functions
47:         `add_callback` and `remove_callback` can be used.
48: 
49:     '''
50:     # completely implemented at the C-level (in _macosx.Timer)
51: 
52: 
53: class FigureCanvasMac(_macosx.FigureCanvas, FigureCanvasAgg):
54:     '''
55:     The canvas the figure renders into.  Calls the draw and print fig
56:     methods, creates the renderers, etc...
57: 
58:     Events such as button presses, mouse movements, and key presses
59:     are handled in the C code and the base class methods
60:     button_press_event, button_release_event, motion_notify_event,
61:     key_press_event, and key_release_event are called from there.
62: 
63:     Attributes
64:     ----------
65:     figure : `matplotlib.figure.Figure`
66:         A high-level Figure instance
67: 
68:     '''
69: 
70:     def __init__(self, figure):
71:         FigureCanvasBase.__init__(self, figure)
72:         width, height = self.get_width_height()
73:         _macosx.FigureCanvas.__init__(self, width, height)
74:         self._device_scale = 1.0
75: 
76:     def _set_device_scale(self, value):
77:         if self._device_scale != value:
78:             self.figure.dpi = self.figure.dpi / self._device_scale * value
79:             self._device_scale = value
80: 
81:     def get_renderer(self, cleared=False):
82:         l, b, w, h = self.figure.bbox.bounds
83:         key = w, h, self.figure.dpi
84:         try:
85:             self._lastKey, self._renderer
86:         except AttributeError:
87:             need_new_renderer = True
88:         else:
89:             need_new_renderer = (self._lastKey != key)
90: 
91:         if need_new_renderer:
92:             self._renderer = RendererAgg(w, h, self.figure.dpi)
93:             self._lastKey = key
94:         elif cleared:
95:             self._renderer.clear()
96: 
97:         return self._renderer
98: 
99:     def _draw(self):
100:         renderer = self.get_renderer()
101: 
102:         if not self.figure.stale:
103:             return renderer
104: 
105:         self.figure.draw(renderer)
106:         return renderer
107: 
108:     def draw(self):
109:         self.invalidate()
110: 
111:     def draw_idle(self, *args, **kwargs):
112:         self.invalidate()
113: 
114:     def blit(self, bbox):
115:         self.invalidate()
116: 
117:     def resize(self, width, height):
118:         dpi = self.figure.dpi
119:         width /= dpi
120:         height /= dpi
121:         self.figure.set_size_inches(width * self._device_scale,
122:                                     height * self._device_scale,
123:                                     forward=False)
124:         FigureCanvasBase.resize_event(self)
125:         self.draw_idle()
126: 
127:     def new_timer(self, *args, **kwargs):
128:         '''
129:         Creates a new backend-specific subclass of :class:`backend_bases.Timer`.
130:         This is useful for getting periodic events through the backend's native
131:         event loop. Implemented only for backends with GUIs.
132: 
133:         Other Parameters
134:         ----------------
135:         interval : scalar
136:             Timer interval in milliseconds
137:         callbacks : list
138:             Sequence of (func, args, kwargs) where ``func(*args, **kwargs)``
139:             will be executed by the timer every *interval*.
140:         '''
141:         return TimerMac(*args, **kwargs)
142: 
143: 
144: class FigureManagerMac(_macosx.FigureManager, FigureManagerBase):
145:     '''
146:     Wrap everything up into a window for the pylab interface
147:     '''
148:     def __init__(self, canvas, num):
149:         FigureManagerBase.__init__(self, canvas, num)
150:         title = "Figure %d" % num
151:         _macosx.FigureManager.__init__(self, canvas, title)
152:         if rcParams['toolbar']=='toolbar2':
153:             self.toolbar = NavigationToolbar2Mac(canvas)
154:         else:
155:             self.toolbar = None
156:         if self.toolbar is not None:
157:             self.toolbar.update()
158: 
159:         def notify_axes_change(fig):
160:             'this will be called whenever the current axes is changed'
161:             if self.toolbar != None: self.toolbar.update()
162:         self.canvas.figure.add_axobserver(notify_axes_change)
163: 
164:         if matplotlib.is_interactive():
165:             self.show()
166:             self.canvas.draw_idle()
167: 
168:     def close(self):
169:         Gcf.destroy(self.num)
170: 
171: 
172: class NavigationToolbar2Mac(_macosx.NavigationToolbar2, NavigationToolbar2):
173: 
174:     def __init__(self, canvas):
175:         NavigationToolbar2.__init__(self, canvas)
176: 
177:     def _init_toolbar(self):
178:         basedir = os.path.join(rcParams['datapath'], "images")
179:         _macosx.NavigationToolbar2.__init__(self, basedir)
180: 
181:     def draw_rubberband(self, event, x0, y0, x1, y1):
182:         self.canvas.set_rubberband(int(x0), int(y0), int(x1), int(y1))
183: 
184:     def release(self, event):
185:         self.canvas.remove_rubberband()
186: 
187:     def set_cursor(self, cursor):
188:         _macosx.set_cursor(cursor)
189: 
190:     def save_figure(self, *args):
191:         filename = _macosx.choose_save_file('Save the figure',
192:                                             self.canvas.get_default_filename())
193:         if filename is None: # Cancel
194:             return
195:         self.canvas.figure.savefig(filename)
196: 
197:     def prepare_configure_subplots(self):
198:         toolfig = Figure(figsize=(6,3))
199:         canvas = FigureCanvasMac(toolfig)
200:         toolfig.subplots_adjust(top=0.9)
201:         tool = SubplotTool(self.canvas.figure, toolfig)
202:         return canvas
203: 
204:     def set_message(self, message):
205:         _macosx.NavigationToolbar2.set_message(self, message.encode('utf-8'))
206: 
207: 
208: ########################################################################
209: #
210: # Now just provide the standard names that backend.__init__ is expecting
211: #
212: ########################################################################
213: 
214: @_Backend.export
215: class _BackendMac(_Backend):
216:     FigureCanvas = FigureCanvasMac
217:     FigureManager = FigureManagerMac
218: 
219:     def trigger_manager_draw(manager):
220:         # For performance reasons, we don't want to redraw the figure after
221:         # each draw command. Instead, we mark the figure as invalid, so that it
222:         # will be redrawn as soon as the event loop resumes via PyOS_InputHook.
223:         # This function should be called after each draw event, even if
224:         # matplotlib is not running interactively.
225:         manager.canvas.invalidate()
226: 
227:     @staticmethod
228:     def mainloop():
229:         _macosx.show()
230: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230323 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_230323) is not StypyTypeError):

    if (import_230323 != 'pyd_module'):
        __import__(import_230323)
        sys_modules_230324 = sys.modules[import_230323]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_230324.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_230323)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import os' statement (line 6)
import os

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from matplotlib._pylab_helpers import Gcf' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230325 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib._pylab_helpers')

if (type(import_230325) is not StypyTypeError):

    if (import_230325 != 'pyd_module'):
        __import__(import_230325)
        sys_modules_230326 = sys.modules[import_230325]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib._pylab_helpers', sys_modules_230326.module_type_store, module_type_store, ['Gcf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_230326, sys_modules_230326.module_type_store, module_type_store)
    else:
        from matplotlib._pylab_helpers import Gcf

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib._pylab_helpers', None, module_type_store, ['Gcf'], [Gcf])

else:
    # Assigning a type to the variable 'matplotlib._pylab_helpers' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib._pylab_helpers', import_230325)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2, TimerBase' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230327 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backend_bases')

if (type(import_230327) is not StypyTypeError):

    if (import_230327 != 'pyd_module'):
        __import__(import_230327)
        sys_modules_230328 = sys.modules[import_230327]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backend_bases', sys_modules_230328.module_type_store, module_type_store, ['_Backend', 'FigureCanvasBase', 'FigureManagerBase', 'NavigationToolbar2', 'TimerBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_230328, sys_modules_230328.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2, TimerBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backend_bases', None, module_type_store, ['_Backend', 'FigureCanvasBase', 'FigureManagerBase', 'NavigationToolbar2', 'TimerBase'], [_Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2, TimerBase])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backend_bases', import_230327)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from matplotlib.figure import Figure' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230329 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.figure')

if (type(import_230329) is not StypyTypeError):

    if (import_230329 != 'pyd_module'):
        __import__(import_230329)
        sys_modules_230330 = sys.modules[import_230329]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.figure', sys_modules_230330.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_230330, sys_modules_230330.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.figure', import_230329)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from matplotlib import rcParams' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230331 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib')

if (type(import_230331) is not StypyTypeError):

    if (import_230331 != 'pyd_module'):
        __import__(import_230331)
        sys_modules_230332 = sys.modules[import_230331]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', sys_modules_230332.module_type_store, module_type_store, ['rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_230332, sys_modules_230332.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', None, module_type_store, ['rcParams'], [rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', import_230331)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from matplotlib.widgets import SubplotTool' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230333 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.widgets')

if (type(import_230333) is not StypyTypeError):

    if (import_230333 != 'pyd_module'):
        __import__(import_230333)
        sys_modules_230334 = sys.modules[import_230333]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.widgets', sys_modules_230334.module_type_store, module_type_store, ['SubplotTool'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_230334, sys_modules_230334.module_type_store, module_type_store)
    else:
        from matplotlib.widgets import SubplotTool

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.widgets', None, module_type_store, ['SubplotTool'], [SubplotTool])

else:
    # Assigning a type to the variable 'matplotlib.widgets' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.widgets', import_230333)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import matplotlib' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230335 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib')

if (type(import_230335) is not StypyTypeError):

    if (import_230335 != 'pyd_module'):
        __import__(import_230335)
        sys_modules_230336 = sys.modules[import_230335]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib', sys_modules_230336.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib', import_230335)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from matplotlib.backends import _macosx' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230337 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.backends')

if (type(import_230337) is not StypyTypeError):

    if (import_230337 != 'pyd_module'):
        __import__(import_230337)
        sys_modules_230338 = sys.modules[import_230337]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.backends', sys_modules_230338.module_type_store, module_type_store, ['_macosx'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_230338, sys_modules_230338.module_type_store, module_type_store)
    else:
        from matplotlib.backends import _macosx

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.backends', None, module_type_store, ['_macosx'], [_macosx])

else:
    # Assigning a type to the variable 'matplotlib.backends' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.backends', import_230337)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from matplotlib.backends.backend_agg import RendererAgg, FigureCanvasAgg' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230339 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.backends.backend_agg')

if (type(import_230339) is not StypyTypeError):

    if (import_230339 != 'pyd_module'):
        __import__(import_230339)
        sys_modules_230340 = sys.modules[import_230339]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.backends.backend_agg', sys_modules_230340.module_type_store, module_type_store, ['RendererAgg', 'FigureCanvasAgg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_230340, sys_modules_230340.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_agg import RendererAgg, FigureCanvasAgg

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.backends.backend_agg', None, module_type_store, ['RendererAgg', 'FigureCanvasAgg'], [RendererAgg, FigureCanvasAgg])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_agg' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.backends.backend_agg', import_230339)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# Declaration of the 'TimerMac' class
# Getting the type of '_macosx' (line 32)
_macosx_230341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), '_macosx')
# Obtaining the member 'Timer' of a type (line 32)
Timer_230342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), _macosx_230341, 'Timer')
# Getting the type of 'TimerBase' (line 32)
TimerBase_230343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 30), 'TimerBase')

class TimerMac(Timer_230342, TimerBase_230343, ):
    unicode_230344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, (-1)), 'unicode', u'\n    Subclass of :class:`backend_bases.TimerBase` that uses CoreFoundation\n    run loops for timer events.\n\n    Attributes\n    ----------\n    interval : int\n        The time between timer events in milliseconds. Default is 1000 ms.\n    single_shot : bool\n        Boolean flag indicating whether this timer should operate as single\n        shot (run once and then stop). Defaults to False.\n    callbacks : list\n        Stores list of (func, args) tuples that will be called upon timer\n        events. This list can be manipulated directly, or the functions\n        `add_callback` and `remove_callback` can be used.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 32, 0, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerMac.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TimerMac' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'TimerMac', TimerMac)
# Declaration of the 'FigureCanvasMac' class
# Getting the type of '_macosx' (line 53)
_macosx_230345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), '_macosx')
# Obtaining the member 'FigureCanvas' of a type (line 53)
FigureCanvas_230346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 22), _macosx_230345, 'FigureCanvas')
# Getting the type of 'FigureCanvasAgg' (line 53)
FigureCanvasAgg_230347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 44), 'FigureCanvasAgg')

class FigureCanvasMac(FigureCanvas_230346, FigureCanvasAgg_230347, ):
    unicode_230348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'unicode', u'\n    The canvas the figure renders into.  Calls the draw and print fig\n    methods, creates the renderers, etc...\n\n    Events such as button presses, mouse movements, and key presses\n    are handled in the C code and the base class methods\n    button_press_event, button_release_event, motion_notify_event,\n    key_press_event, and key_release_event are called from there.\n\n    Attributes\n    ----------\n    figure : `matplotlib.figure.Figure`\n        A high-level Figure instance\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 70, 4, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasMac.__init__', ['figure'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'self' (line 71)
        self_230351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 34), 'self', False)
        # Getting the type of 'figure' (line 71)
        figure_230352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 40), 'figure', False)
        # Processing the call keyword arguments (line 71)
        kwargs_230353 = {}
        # Getting the type of 'FigureCanvasBase' (line 71)
        FigureCanvasBase_230349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'FigureCanvasBase', False)
        # Obtaining the member '__init__' of a type (line 71)
        init___230350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), FigureCanvasBase_230349, '__init__')
        # Calling __init__(args, kwargs) (line 71)
        init___call_result_230354 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), init___230350, *[self_230351, figure_230352], **kwargs_230353)
        
        
        # Assigning a Call to a Tuple (line 72):
        
        # Assigning a Call to a Name:
        
        # Call to get_width_height(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_230357 = {}
        # Getting the type of 'self' (line 72)
        self_230355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'self', False)
        # Obtaining the member 'get_width_height' of a type (line 72)
        get_width_height_230356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 24), self_230355, 'get_width_height')
        # Calling get_width_height(args, kwargs) (line 72)
        get_width_height_call_result_230358 = invoke(stypy.reporting.localization.Localization(__file__, 72, 24), get_width_height_230356, *[], **kwargs_230357)
        
        # Assigning a type to the variable 'call_assignment_230316' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'call_assignment_230316', get_width_height_call_result_230358)
        
        # Assigning a Call to a Name (line 72):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_230361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 8), 'int')
        # Processing the call keyword arguments
        kwargs_230362 = {}
        # Getting the type of 'call_assignment_230316' (line 72)
        call_assignment_230316_230359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'call_assignment_230316', False)
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___230360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), call_assignment_230316_230359, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_230363 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___230360, *[int_230361], **kwargs_230362)
        
        # Assigning a type to the variable 'call_assignment_230317' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'call_assignment_230317', getitem___call_result_230363)
        
        # Assigning a Name to a Name (line 72):
        # Getting the type of 'call_assignment_230317' (line 72)
        call_assignment_230317_230364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'call_assignment_230317')
        # Assigning a type to the variable 'width' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'width', call_assignment_230317_230364)
        
        # Assigning a Call to a Name (line 72):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_230367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 8), 'int')
        # Processing the call keyword arguments
        kwargs_230368 = {}
        # Getting the type of 'call_assignment_230316' (line 72)
        call_assignment_230316_230365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'call_assignment_230316', False)
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___230366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), call_assignment_230316_230365, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_230369 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___230366, *[int_230367], **kwargs_230368)
        
        # Assigning a type to the variable 'call_assignment_230318' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'call_assignment_230318', getitem___call_result_230369)
        
        # Assigning a Name to a Name (line 72):
        # Getting the type of 'call_assignment_230318' (line 72)
        call_assignment_230318_230370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'call_assignment_230318')
        # Assigning a type to the variable 'height' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'height', call_assignment_230318_230370)
        
        # Call to __init__(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'self' (line 73)
        self_230374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 38), 'self', False)
        # Getting the type of 'width' (line 73)
        width_230375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'width', False)
        # Getting the type of 'height' (line 73)
        height_230376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 51), 'height', False)
        # Processing the call keyword arguments (line 73)
        kwargs_230377 = {}
        # Getting the type of '_macosx' (line 73)
        _macosx_230371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), '_macosx', False)
        # Obtaining the member 'FigureCanvas' of a type (line 73)
        FigureCanvas_230372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), _macosx_230371, 'FigureCanvas')
        # Obtaining the member '__init__' of a type (line 73)
        init___230373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), FigureCanvas_230372, '__init__')
        # Calling __init__(args, kwargs) (line 73)
        init___call_result_230378 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), init___230373, *[self_230374, width_230375, height_230376], **kwargs_230377)
        
        
        # Assigning a Num to a Attribute (line 74):
        
        # Assigning a Num to a Attribute (line 74):
        float_230379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 29), 'float')
        # Getting the type of 'self' (line 74)
        self_230380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member '_device_scale' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_230380, '_device_scale', float_230379)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _set_device_scale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_device_scale'
        module_type_store = module_type_store.open_function_context('_set_device_scale', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasMac._set_device_scale.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasMac._set_device_scale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasMac._set_device_scale.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasMac._set_device_scale.__dict__.__setitem__('stypy_function_name', 'FigureCanvasMac._set_device_scale')
        FigureCanvasMac._set_device_scale.__dict__.__setitem__('stypy_param_names_list', ['value'])
        FigureCanvasMac._set_device_scale.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasMac._set_device_scale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasMac._set_device_scale.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasMac._set_device_scale.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasMac._set_device_scale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasMac._set_device_scale.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasMac._set_device_scale', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_device_scale', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_device_scale(...)' code ##################

        
        
        # Getting the type of 'self' (line 77)
        self_230381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'self')
        # Obtaining the member '_device_scale' of a type (line 77)
        _device_scale_230382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 11), self_230381, '_device_scale')
        # Getting the type of 'value' (line 77)
        value_230383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 33), 'value')
        # Applying the binary operator '!=' (line 77)
        result_ne_230384 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 11), '!=', _device_scale_230382, value_230383)
        
        # Testing the type of an if condition (line 77)
        if_condition_230385 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 8), result_ne_230384)
        # Assigning a type to the variable 'if_condition_230385' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'if_condition_230385', if_condition_230385)
        # SSA begins for if statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Attribute (line 78):
        
        # Assigning a BinOp to a Attribute (line 78):
        # Getting the type of 'self' (line 78)
        self_230386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'self')
        # Obtaining the member 'figure' of a type (line 78)
        figure_230387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 30), self_230386, 'figure')
        # Obtaining the member 'dpi' of a type (line 78)
        dpi_230388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 30), figure_230387, 'dpi')
        # Getting the type of 'self' (line 78)
        self_230389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 48), 'self')
        # Obtaining the member '_device_scale' of a type (line 78)
        _device_scale_230390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 48), self_230389, '_device_scale')
        # Applying the binary operator 'div' (line 78)
        result_div_230391 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 30), 'div', dpi_230388, _device_scale_230390)
        
        # Getting the type of 'value' (line 78)
        value_230392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 69), 'value')
        # Applying the binary operator '*' (line 78)
        result_mul_230393 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 67), '*', result_div_230391, value_230392)
        
        # Getting the type of 'self' (line 78)
        self_230394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'self')
        # Obtaining the member 'figure' of a type (line 78)
        figure_230395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), self_230394, 'figure')
        # Setting the type of the member 'dpi' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), figure_230395, 'dpi', result_mul_230393)
        
        # Assigning a Name to a Attribute (line 79):
        
        # Assigning a Name to a Attribute (line 79):
        # Getting the type of 'value' (line 79)
        value_230396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'value')
        # Getting the type of 'self' (line 79)
        self_230397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'self')
        # Setting the type of the member '_device_scale' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), self_230397, '_device_scale', value_230396)
        # SSA join for if statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_set_device_scale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_device_scale' in the type store
        # Getting the type of 'stypy_return_type' (line 76)
        stypy_return_type_230398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230398)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_device_scale'
        return stypy_return_type_230398


    @norecursion
    def get_renderer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 81)
        False_230399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 35), 'False')
        defaults = [False_230399]
        # Create a new context for function 'get_renderer'
        module_type_store = module_type_store.open_function_context('get_renderer', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasMac.get_renderer.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasMac.get_renderer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasMac.get_renderer.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasMac.get_renderer.__dict__.__setitem__('stypy_function_name', 'FigureCanvasMac.get_renderer')
        FigureCanvasMac.get_renderer.__dict__.__setitem__('stypy_param_names_list', ['cleared'])
        FigureCanvasMac.get_renderer.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasMac.get_renderer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasMac.get_renderer.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasMac.get_renderer.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasMac.get_renderer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasMac.get_renderer.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasMac.get_renderer', ['cleared'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Tuple (line 82):
        
        # Assigning a Subscript to a Name (line 82):
        
        # Obtaining the type of the subscript
        int_230400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
        # Getting the type of 'self' (line 82)
        self_230401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'self')
        # Obtaining the member 'figure' of a type (line 82)
        figure_230402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), self_230401, 'figure')
        # Obtaining the member 'bbox' of a type (line 82)
        bbox_230403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), figure_230402, 'bbox')
        # Obtaining the member 'bounds' of a type (line 82)
        bounds_230404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), bbox_230403, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___230405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), bounds_230404, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_230406 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___230405, int_230400)
        
        # Assigning a type to the variable 'tuple_var_assignment_230319' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_230319', subscript_call_result_230406)
        
        # Assigning a Subscript to a Name (line 82):
        
        # Obtaining the type of the subscript
        int_230407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
        # Getting the type of 'self' (line 82)
        self_230408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'self')
        # Obtaining the member 'figure' of a type (line 82)
        figure_230409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), self_230408, 'figure')
        # Obtaining the member 'bbox' of a type (line 82)
        bbox_230410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), figure_230409, 'bbox')
        # Obtaining the member 'bounds' of a type (line 82)
        bounds_230411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), bbox_230410, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___230412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), bounds_230411, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_230413 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___230412, int_230407)
        
        # Assigning a type to the variable 'tuple_var_assignment_230320' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_230320', subscript_call_result_230413)
        
        # Assigning a Subscript to a Name (line 82):
        
        # Obtaining the type of the subscript
        int_230414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
        # Getting the type of 'self' (line 82)
        self_230415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'self')
        # Obtaining the member 'figure' of a type (line 82)
        figure_230416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), self_230415, 'figure')
        # Obtaining the member 'bbox' of a type (line 82)
        bbox_230417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), figure_230416, 'bbox')
        # Obtaining the member 'bounds' of a type (line 82)
        bounds_230418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), bbox_230417, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___230419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), bounds_230418, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_230420 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___230419, int_230414)
        
        # Assigning a type to the variable 'tuple_var_assignment_230321' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_230321', subscript_call_result_230420)
        
        # Assigning a Subscript to a Name (line 82):
        
        # Obtaining the type of the subscript
        int_230421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
        # Getting the type of 'self' (line 82)
        self_230422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'self')
        # Obtaining the member 'figure' of a type (line 82)
        figure_230423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), self_230422, 'figure')
        # Obtaining the member 'bbox' of a type (line 82)
        bbox_230424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), figure_230423, 'bbox')
        # Obtaining the member 'bounds' of a type (line 82)
        bounds_230425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), bbox_230424, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___230426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), bounds_230425, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_230427 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___230426, int_230421)
        
        # Assigning a type to the variable 'tuple_var_assignment_230322' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_230322', subscript_call_result_230427)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'tuple_var_assignment_230319' (line 82)
        tuple_var_assignment_230319_230428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_230319')
        # Assigning a type to the variable 'l' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'l', tuple_var_assignment_230319_230428)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'tuple_var_assignment_230320' (line 82)
        tuple_var_assignment_230320_230429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_230320')
        # Assigning a type to the variable 'b' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'b', tuple_var_assignment_230320_230429)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'tuple_var_assignment_230321' (line 82)
        tuple_var_assignment_230321_230430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_230321')
        # Assigning a type to the variable 'w' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 14), 'w', tuple_var_assignment_230321_230430)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'tuple_var_assignment_230322' (line 82)
        tuple_var_assignment_230322_230431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_230322')
        # Assigning a type to the variable 'h' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'h', tuple_var_assignment_230322_230431)
        
        # Assigning a Tuple to a Name (line 83):
        
        # Assigning a Tuple to a Name (line 83):
        
        # Obtaining an instance of the builtin type 'tuple' (line 83)
        tuple_230432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 83)
        # Adding element type (line 83)
        # Getting the type of 'w' (line 83)
        w_230433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 14), 'w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 14), tuple_230432, w_230433)
        # Adding element type (line 83)
        # Getting the type of 'h' (line 83)
        h_230434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 14), tuple_230432, h_230434)
        # Adding element type (line 83)
        # Getting the type of 'self' (line 83)
        self_230435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'self')
        # Obtaining the member 'figure' of a type (line 83)
        figure_230436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), self_230435, 'figure')
        # Obtaining the member 'dpi' of a type (line 83)
        dpi_230437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), figure_230436, 'dpi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 14), tuple_230432, dpi_230437)
        
        # Assigning a type to the variable 'key' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'key', tuple_230432)
        
        
        # SSA begins for try-except statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining an instance of the builtin type 'tuple' (line 85)
        tuple_230438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 85)
        # Adding element type (line 85)
        # Getting the type of 'self' (line 85)
        self_230439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'self')
        # Obtaining the member '_lastKey' of a type (line 85)
        _lastKey_230440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), self_230439, '_lastKey')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 12), tuple_230438, _lastKey_230440)
        # Adding element type (line 85)
        # Getting the type of 'self' (line 85)
        self_230441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'self')
        # Obtaining the member '_renderer' of a type (line 85)
        _renderer_230442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 27), self_230441, '_renderer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 12), tuple_230438, _renderer_230442)
        
        # SSA branch for the except part of a try statement (line 84)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 84)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 87):
        
        # Assigning a Name to a Name (line 87):
        # Getting the type of 'True' (line 87)
        True_230443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 32), 'True')
        # Assigning a type to the variable 'need_new_renderer' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'need_new_renderer', True_230443)
        # SSA branch for the else branch of a try statement (line 84)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a Compare to a Name (line 89):
        
        # Assigning a Compare to a Name (line 89):
        
        # Getting the type of 'self' (line 89)
        self_230444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 33), 'self')
        # Obtaining the member '_lastKey' of a type (line 89)
        _lastKey_230445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 33), self_230444, '_lastKey')
        # Getting the type of 'key' (line 89)
        key_230446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 50), 'key')
        # Applying the binary operator '!=' (line 89)
        result_ne_230447 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 33), '!=', _lastKey_230445, key_230446)
        
        # Assigning a type to the variable 'need_new_renderer' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'need_new_renderer', result_ne_230447)
        # SSA join for try-except statement (line 84)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'need_new_renderer' (line 91)
        need_new_renderer_230448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'need_new_renderer')
        # Testing the type of an if condition (line 91)
        if_condition_230449 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 8), need_new_renderer_230448)
        # Assigning a type to the variable 'if_condition_230449' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'if_condition_230449', if_condition_230449)
        # SSA begins for if statement (line 91)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 92):
        
        # Assigning a Call to a Attribute (line 92):
        
        # Call to RendererAgg(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'w' (line 92)
        w_230451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'w', False)
        # Getting the type of 'h' (line 92)
        h_230452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 44), 'h', False)
        # Getting the type of 'self' (line 92)
        self_230453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 47), 'self', False)
        # Obtaining the member 'figure' of a type (line 92)
        figure_230454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 47), self_230453, 'figure')
        # Obtaining the member 'dpi' of a type (line 92)
        dpi_230455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 47), figure_230454, 'dpi')
        # Processing the call keyword arguments (line 92)
        kwargs_230456 = {}
        # Getting the type of 'RendererAgg' (line 92)
        RendererAgg_230450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 29), 'RendererAgg', False)
        # Calling RendererAgg(args, kwargs) (line 92)
        RendererAgg_call_result_230457 = invoke(stypy.reporting.localization.Localization(__file__, 92, 29), RendererAgg_230450, *[w_230451, h_230452, dpi_230455], **kwargs_230456)
        
        # Getting the type of 'self' (line 92)
        self_230458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'self')
        # Setting the type of the member '_renderer' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), self_230458, '_renderer', RendererAgg_call_result_230457)
        
        # Assigning a Name to a Attribute (line 93):
        
        # Assigning a Name to a Attribute (line 93):
        # Getting the type of 'key' (line 93)
        key_230459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'key')
        # Getting the type of 'self' (line 93)
        self_230460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'self')
        # Setting the type of the member '_lastKey' of a type (line 93)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), self_230460, '_lastKey', key_230459)
        # SSA branch for the else part of an if statement (line 91)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'cleared' (line 94)
        cleared_230461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 13), 'cleared')
        # Testing the type of an if condition (line 94)
        if_condition_230462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 13), cleared_230461)
        # Assigning a type to the variable 'if_condition_230462' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 13), 'if_condition_230462', if_condition_230462)
        # SSA begins for if statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to clear(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_230466 = {}
        # Getting the type of 'self' (line 95)
        self_230463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'self', False)
        # Obtaining the member '_renderer' of a type (line 95)
        _renderer_230464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), self_230463, '_renderer')
        # Obtaining the member 'clear' of a type (line 95)
        clear_230465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), _renderer_230464, 'clear')
        # Calling clear(args, kwargs) (line 95)
        clear_call_result_230467 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), clear_230465, *[], **kwargs_230466)
        
        # SSA join for if statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 91)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 97)
        self_230468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'self')
        # Obtaining the member '_renderer' of a type (line 97)
        _renderer_230469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), self_230468, '_renderer')
        # Assigning a type to the variable 'stypy_return_type' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'stypy_return_type', _renderer_230469)
        
        # ################# End of 'get_renderer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_renderer' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_230470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230470)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_renderer'
        return stypy_return_type_230470


    @norecursion
    def _draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_draw'
        module_type_store = module_type_store.open_function_context('_draw', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasMac._draw.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasMac._draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasMac._draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasMac._draw.__dict__.__setitem__('stypy_function_name', 'FigureCanvasMac._draw')
        FigureCanvasMac._draw.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasMac._draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasMac._draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasMac._draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasMac._draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasMac._draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasMac._draw.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasMac._draw', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_draw', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_draw(...)' code ##################

        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to get_renderer(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_230473 = {}
        # Getting the type of 'self' (line 100)
        self_230471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), 'self', False)
        # Obtaining the member 'get_renderer' of a type (line 100)
        get_renderer_230472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 19), self_230471, 'get_renderer')
        # Calling get_renderer(args, kwargs) (line 100)
        get_renderer_call_result_230474 = invoke(stypy.reporting.localization.Localization(__file__, 100, 19), get_renderer_230472, *[], **kwargs_230473)
        
        # Assigning a type to the variable 'renderer' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'renderer', get_renderer_call_result_230474)
        
        
        # Getting the type of 'self' (line 102)
        self_230475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'self')
        # Obtaining the member 'figure' of a type (line 102)
        figure_230476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), self_230475, 'figure')
        # Obtaining the member 'stale' of a type (line 102)
        stale_230477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), figure_230476, 'stale')
        # Applying the 'not' unary operator (line 102)
        result_not__230478 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), 'not', stale_230477)
        
        # Testing the type of an if condition (line 102)
        if_condition_230479 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), result_not__230478)
        # Assigning a type to the variable 'if_condition_230479' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'if_condition_230479', if_condition_230479)
        # SSA begins for if statement (line 102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'renderer' (line 103)
        renderer_230480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'renderer')
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'stypy_return_type', renderer_230480)
        # SSA join for if statement (line 102)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'renderer' (line 105)
        renderer_230484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 'renderer', False)
        # Processing the call keyword arguments (line 105)
        kwargs_230485 = {}
        # Getting the type of 'self' (line 105)
        self_230481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 105)
        figure_230482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), self_230481, 'figure')
        # Obtaining the member 'draw' of a type (line 105)
        draw_230483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), figure_230482, 'draw')
        # Calling draw(args, kwargs) (line 105)
        draw_call_result_230486 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), draw_230483, *[renderer_230484], **kwargs_230485)
        
        # Getting the type of 'renderer' (line 106)
        renderer_230487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'renderer')
        # Assigning a type to the variable 'stypy_return_type' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'stypy_return_type', renderer_230487)
        
        # ################# End of '_draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_draw' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_230488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230488)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_draw'
        return stypy_return_type_230488


    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasMac.draw.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasMac.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasMac.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasMac.draw.__dict__.__setitem__('stypy_function_name', 'FigureCanvasMac.draw')
        FigureCanvasMac.draw.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasMac.draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasMac.draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasMac.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasMac.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasMac.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasMac.draw.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasMac.draw', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to invalidate(...): (line 109)
        # Processing the call keyword arguments (line 109)
        kwargs_230491 = {}
        # Getting the type of 'self' (line 109)
        self_230489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self', False)
        # Obtaining the member 'invalidate' of a type (line 109)
        invalidate_230490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_230489, 'invalidate')
        # Calling invalidate(args, kwargs) (line 109)
        invalidate_call_result_230492 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), invalidate_230490, *[], **kwargs_230491)
        
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_230493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230493)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_230493


    @norecursion
    def draw_idle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_idle'
        module_type_store = module_type_store.open_function_context('draw_idle', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasMac.draw_idle.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasMac.draw_idle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasMac.draw_idle.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasMac.draw_idle.__dict__.__setitem__('stypy_function_name', 'FigureCanvasMac.draw_idle')
        FigureCanvasMac.draw_idle.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasMac.draw_idle.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasMac.draw_idle.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasMac.draw_idle.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasMac.draw_idle.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasMac.draw_idle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasMac.draw_idle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasMac.draw_idle', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Call to invalidate(...): (line 112)
        # Processing the call keyword arguments (line 112)
        kwargs_230496 = {}
        # Getting the type of 'self' (line 112)
        self_230494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'self', False)
        # Obtaining the member 'invalidate' of a type (line 112)
        invalidate_230495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), self_230494, 'invalidate')
        # Calling invalidate(args, kwargs) (line 112)
        invalidate_call_result_230497 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), invalidate_230495, *[], **kwargs_230496)
        
        
        # ################# End of 'draw_idle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_idle' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_230498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230498)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_idle'
        return stypy_return_type_230498


    @norecursion
    def blit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'blit'
        module_type_store = module_type_store.open_function_context('blit', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasMac.blit.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasMac.blit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasMac.blit.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasMac.blit.__dict__.__setitem__('stypy_function_name', 'FigureCanvasMac.blit')
        FigureCanvasMac.blit.__dict__.__setitem__('stypy_param_names_list', ['bbox'])
        FigureCanvasMac.blit.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasMac.blit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasMac.blit.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasMac.blit.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasMac.blit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasMac.blit.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasMac.blit', ['bbox'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'blit', localization, ['bbox'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'blit(...)' code ##################

        
        # Call to invalidate(...): (line 115)
        # Processing the call keyword arguments (line 115)
        kwargs_230501 = {}
        # Getting the type of 'self' (line 115)
        self_230499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'self', False)
        # Obtaining the member 'invalidate' of a type (line 115)
        invalidate_230500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), self_230499, 'invalidate')
        # Calling invalidate(args, kwargs) (line 115)
        invalidate_call_result_230502 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), invalidate_230500, *[], **kwargs_230501)
        
        
        # ################# End of 'blit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'blit' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_230503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230503)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'blit'
        return stypy_return_type_230503


    @norecursion
    def resize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'resize'
        module_type_store = module_type_store.open_function_context('resize', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasMac.resize.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasMac.resize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasMac.resize.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasMac.resize.__dict__.__setitem__('stypy_function_name', 'FigureCanvasMac.resize')
        FigureCanvasMac.resize.__dict__.__setitem__('stypy_param_names_list', ['width', 'height'])
        FigureCanvasMac.resize.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasMac.resize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasMac.resize.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasMac.resize.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasMac.resize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasMac.resize.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasMac.resize', ['width', 'height'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Name (line 118):
        
        # Assigning a Attribute to a Name (line 118):
        # Getting the type of 'self' (line 118)
        self_230504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'self')
        # Obtaining the member 'figure' of a type (line 118)
        figure_230505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 14), self_230504, 'figure')
        # Obtaining the member 'dpi' of a type (line 118)
        dpi_230506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 14), figure_230505, 'dpi')
        # Assigning a type to the variable 'dpi' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'dpi', dpi_230506)
        
        # Getting the type of 'width' (line 119)
        width_230507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'width')
        # Getting the type of 'dpi' (line 119)
        dpi_230508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 17), 'dpi')
        # Applying the binary operator 'div=' (line 119)
        result_div_230509 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 8), 'div=', width_230507, dpi_230508)
        # Assigning a type to the variable 'width' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'width', result_div_230509)
        
        
        # Getting the type of 'height' (line 120)
        height_230510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'height')
        # Getting the type of 'dpi' (line 120)
        dpi_230511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 18), 'dpi')
        # Applying the binary operator 'div=' (line 120)
        result_div_230512 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 8), 'div=', height_230510, dpi_230511)
        # Assigning a type to the variable 'height' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'height', result_div_230512)
        
        
        # Call to set_size_inches(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'width' (line 121)
        width_230516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 36), 'width', False)
        # Getting the type of 'self' (line 121)
        self_230517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 44), 'self', False)
        # Obtaining the member '_device_scale' of a type (line 121)
        _device_scale_230518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 44), self_230517, '_device_scale')
        # Applying the binary operator '*' (line 121)
        result_mul_230519 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 36), '*', width_230516, _device_scale_230518)
        
        # Getting the type of 'height' (line 122)
        height_230520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 36), 'height', False)
        # Getting the type of 'self' (line 122)
        self_230521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 45), 'self', False)
        # Obtaining the member '_device_scale' of a type (line 122)
        _device_scale_230522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 45), self_230521, '_device_scale')
        # Applying the binary operator '*' (line 122)
        result_mul_230523 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 36), '*', height_230520, _device_scale_230522)
        
        # Processing the call keyword arguments (line 121)
        # Getting the type of 'False' (line 123)
        False_230524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 44), 'False', False)
        keyword_230525 = False_230524
        kwargs_230526 = {'forward': keyword_230525}
        # Getting the type of 'self' (line 121)
        self_230513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 121)
        figure_230514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_230513, 'figure')
        # Obtaining the member 'set_size_inches' of a type (line 121)
        set_size_inches_230515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), figure_230514, 'set_size_inches')
        # Calling set_size_inches(args, kwargs) (line 121)
        set_size_inches_call_result_230527 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), set_size_inches_230515, *[result_mul_230519, result_mul_230523], **kwargs_230526)
        
        
        # Call to resize_event(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'self' (line 124)
        self_230530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 38), 'self', False)
        # Processing the call keyword arguments (line 124)
        kwargs_230531 = {}
        # Getting the type of 'FigureCanvasBase' (line 124)
        FigureCanvasBase_230528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'resize_event' of a type (line 124)
        resize_event_230529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), FigureCanvasBase_230528, 'resize_event')
        # Calling resize_event(args, kwargs) (line 124)
        resize_event_call_result_230532 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), resize_event_230529, *[self_230530], **kwargs_230531)
        
        
        # Call to draw_idle(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_230535 = {}
        # Getting the type of 'self' (line 125)
        self_230533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self', False)
        # Obtaining the member 'draw_idle' of a type (line 125)
        draw_idle_230534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_230533, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 125)
        draw_idle_call_result_230536 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), draw_idle_230534, *[], **kwargs_230535)
        
        
        # ################# End of 'resize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'resize' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_230537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230537)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'resize'
        return stypy_return_type_230537


    @norecursion
    def new_timer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_timer'
        module_type_store = module_type_store.open_function_context('new_timer', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasMac.new_timer.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasMac.new_timer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasMac.new_timer.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasMac.new_timer.__dict__.__setitem__('stypy_function_name', 'FigureCanvasMac.new_timer')
        FigureCanvasMac.new_timer.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasMac.new_timer.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasMac.new_timer.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasMac.new_timer.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasMac.new_timer.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasMac.new_timer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasMac.new_timer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasMac.new_timer', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        unicode_230538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, (-1)), 'unicode', u"\n        Creates a new backend-specific subclass of :class:`backend_bases.Timer`.\n        This is useful for getting periodic events through the backend's native\n        event loop. Implemented only for backends with GUIs.\n\n        Other Parameters\n        ----------------\n        interval : scalar\n            Timer interval in milliseconds\n        callbacks : list\n            Sequence of (func, args, kwargs) where ``func(*args, **kwargs)``\n            will be executed by the timer every *interval*.\n        ")
        
        # Call to TimerMac(...): (line 141)
        # Getting the type of 'args' (line 141)
        args_230540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 'args', False)
        # Processing the call keyword arguments (line 141)
        # Getting the type of 'kwargs' (line 141)
        kwargs_230541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), 'kwargs', False)
        kwargs_230542 = {'kwargs_230541': kwargs_230541}
        # Getting the type of 'TimerMac' (line 141)
        TimerMac_230539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'TimerMac', False)
        # Calling TimerMac(args, kwargs) (line 141)
        TimerMac_call_result_230543 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), TimerMac_230539, *[args_230540], **kwargs_230542)
        
        # Assigning a type to the variable 'stypy_return_type' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'stypy_return_type', TimerMac_call_result_230543)
        
        # ################# End of 'new_timer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_timer' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_230544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230544)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_timer'
        return stypy_return_type_230544


# Assigning a type to the variable 'FigureCanvasMac' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'FigureCanvasMac', FigureCanvasMac)
# Declaration of the 'FigureManagerMac' class
# Getting the type of '_macosx' (line 144)
_macosx_230545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 23), '_macosx')
# Obtaining the member 'FigureManager' of a type (line 144)
FigureManager_230546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 23), _macosx_230545, 'FigureManager')
# Getting the type of 'FigureManagerBase' (line 144)
FigureManagerBase_230547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 46), 'FigureManagerBase')

class FigureManagerMac(FigureManager_230546, FigureManagerBase_230547, ):
    unicode_230548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, (-1)), 'unicode', u'\n    Wrap everything up into a window for the pylab interface\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerMac.__init__', ['canvas', 'num'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'self' (line 149)
        self_230551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 35), 'self', False)
        # Getting the type of 'canvas' (line 149)
        canvas_230552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 41), 'canvas', False)
        # Getting the type of 'num' (line 149)
        num_230553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 49), 'num', False)
        # Processing the call keyword arguments (line 149)
        kwargs_230554 = {}
        # Getting the type of 'FigureManagerBase' (line 149)
        FigureManagerBase_230549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'FigureManagerBase', False)
        # Obtaining the member '__init__' of a type (line 149)
        init___230550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), FigureManagerBase_230549, '__init__')
        # Calling __init__(args, kwargs) (line 149)
        init___call_result_230555 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), init___230550, *[self_230551, canvas_230552, num_230553], **kwargs_230554)
        
        
        # Assigning a BinOp to a Name (line 150):
        
        # Assigning a BinOp to a Name (line 150):
        unicode_230556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 16), 'unicode', u'Figure %d')
        # Getting the type of 'num' (line 150)
        num_230557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'num')
        # Applying the binary operator '%' (line 150)
        result_mod_230558 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 16), '%', unicode_230556, num_230557)
        
        # Assigning a type to the variable 'title' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'title', result_mod_230558)
        
        # Call to __init__(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'self' (line 151)
        self_230562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 39), 'self', False)
        # Getting the type of 'canvas' (line 151)
        canvas_230563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 45), 'canvas', False)
        # Getting the type of 'title' (line 151)
        title_230564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 53), 'title', False)
        # Processing the call keyword arguments (line 151)
        kwargs_230565 = {}
        # Getting the type of '_macosx' (line 151)
        _macosx_230559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), '_macosx', False)
        # Obtaining the member 'FigureManager' of a type (line 151)
        FigureManager_230560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), _macosx_230559, 'FigureManager')
        # Obtaining the member '__init__' of a type (line 151)
        init___230561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), FigureManager_230560, '__init__')
        # Calling __init__(args, kwargs) (line 151)
        init___call_result_230566 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), init___230561, *[self_230562, canvas_230563, title_230564], **kwargs_230565)
        
        
        
        
        # Obtaining the type of the subscript
        unicode_230567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 20), 'unicode', u'toolbar')
        # Getting the type of 'rcParams' (line 152)
        rcParams_230568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___230569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 11), rcParams_230568, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_230570 = invoke(stypy.reporting.localization.Localization(__file__, 152, 11), getitem___230569, unicode_230567)
        
        unicode_230571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 32), 'unicode', u'toolbar2')
        # Applying the binary operator '==' (line 152)
        result_eq_230572 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 11), '==', subscript_call_result_230570, unicode_230571)
        
        # Testing the type of an if condition (line 152)
        if_condition_230573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 8), result_eq_230572)
        # Assigning a type to the variable 'if_condition_230573' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'if_condition_230573', if_condition_230573)
        # SSA begins for if statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 153):
        
        # Assigning a Call to a Attribute (line 153):
        
        # Call to NavigationToolbar2Mac(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'canvas' (line 153)
        canvas_230575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 49), 'canvas', False)
        # Processing the call keyword arguments (line 153)
        kwargs_230576 = {}
        # Getting the type of 'NavigationToolbar2Mac' (line 153)
        NavigationToolbar2Mac_230574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 27), 'NavigationToolbar2Mac', False)
        # Calling NavigationToolbar2Mac(args, kwargs) (line 153)
        NavigationToolbar2Mac_call_result_230577 = invoke(stypy.reporting.localization.Localization(__file__, 153, 27), NavigationToolbar2Mac_230574, *[canvas_230575], **kwargs_230576)
        
        # Getting the type of 'self' (line 153)
        self_230578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'self')
        # Setting the type of the member 'toolbar' of a type (line 153)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 12), self_230578, 'toolbar', NavigationToolbar2Mac_call_result_230577)
        # SSA branch for the else part of an if statement (line 152)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 155):
        
        # Assigning a Name to a Attribute (line 155):
        # Getting the type of 'None' (line 155)
        None_230579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 27), 'None')
        # Getting the type of 'self' (line 155)
        self_230580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'self')
        # Setting the type of the member 'toolbar' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), self_230580, 'toolbar', None_230579)
        # SSA join for if statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 156)
        self_230581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'self')
        # Obtaining the member 'toolbar' of a type (line 156)
        toolbar_230582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 11), self_230581, 'toolbar')
        # Getting the type of 'None' (line 156)
        None_230583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'None')
        # Applying the binary operator 'isnot' (line 156)
        result_is_not_230584 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 11), 'isnot', toolbar_230582, None_230583)
        
        # Testing the type of an if condition (line 156)
        if_condition_230585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), result_is_not_230584)
        # Assigning a type to the variable 'if_condition_230585' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'if_condition_230585', if_condition_230585)
        # SSA begins for if statement (line 156)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to update(...): (line 157)
        # Processing the call keyword arguments (line 157)
        kwargs_230589 = {}
        # Getting the type of 'self' (line 157)
        self_230586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'self', False)
        # Obtaining the member 'toolbar' of a type (line 157)
        toolbar_230587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), self_230586, 'toolbar')
        # Obtaining the member 'update' of a type (line 157)
        update_230588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), toolbar_230587, 'update')
        # Calling update(args, kwargs) (line 157)
        update_call_result_230590 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), update_230588, *[], **kwargs_230589)
        
        # SSA join for if statement (line 156)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def notify_axes_change(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'notify_axes_change'
            module_type_store = module_type_store.open_function_context('notify_axes_change', 159, 8, False)
            
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

            unicode_230591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 12), 'unicode', u'this will be called whenever the current axes is changed')
            
            
            # Getting the type of 'self' (line 161)
            self_230592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'self')
            # Obtaining the member 'toolbar' of a type (line 161)
            toolbar_230593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 15), self_230592, 'toolbar')
            # Getting the type of 'None' (line 161)
            None_230594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 31), 'None')
            # Applying the binary operator '!=' (line 161)
            result_ne_230595 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 15), '!=', toolbar_230593, None_230594)
            
            # Testing the type of an if condition (line 161)
            if_condition_230596 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 12), result_ne_230595)
            # Assigning a type to the variable 'if_condition_230596' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'if_condition_230596', if_condition_230596)
            # SSA begins for if statement (line 161)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to update(...): (line 161)
            # Processing the call keyword arguments (line 161)
            kwargs_230600 = {}
            # Getting the type of 'self' (line 161)
            self_230597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'self', False)
            # Obtaining the member 'toolbar' of a type (line 161)
            toolbar_230598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 37), self_230597, 'toolbar')
            # Obtaining the member 'update' of a type (line 161)
            update_230599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 37), toolbar_230598, 'update')
            # Calling update(args, kwargs) (line 161)
            update_call_result_230601 = invoke(stypy.reporting.localization.Localization(__file__, 161, 37), update_230599, *[], **kwargs_230600)
            
            # SSA join for if statement (line 161)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'notify_axes_change(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'notify_axes_change' in the type store
            # Getting the type of 'stypy_return_type' (line 159)
            stypy_return_type_230602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_230602)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'notify_axes_change'
            return stypy_return_type_230602

        # Assigning a type to the variable 'notify_axes_change' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'notify_axes_change', notify_axes_change)
        
        # Call to add_axobserver(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'notify_axes_change' (line 162)
        notify_axes_change_230607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 42), 'notify_axes_change', False)
        # Processing the call keyword arguments (line 162)
        kwargs_230608 = {}
        # Getting the type of 'self' (line 162)
        self_230603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 162)
        canvas_230604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_230603, 'canvas')
        # Obtaining the member 'figure' of a type (line 162)
        figure_230605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), canvas_230604, 'figure')
        # Obtaining the member 'add_axobserver' of a type (line 162)
        add_axobserver_230606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), figure_230605, 'add_axobserver')
        # Calling add_axobserver(args, kwargs) (line 162)
        add_axobserver_call_result_230609 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), add_axobserver_230606, *[notify_axes_change_230607], **kwargs_230608)
        
        
        
        # Call to is_interactive(...): (line 164)
        # Processing the call keyword arguments (line 164)
        kwargs_230612 = {}
        # Getting the type of 'matplotlib' (line 164)
        matplotlib_230610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'matplotlib', False)
        # Obtaining the member 'is_interactive' of a type (line 164)
        is_interactive_230611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 11), matplotlib_230610, 'is_interactive')
        # Calling is_interactive(args, kwargs) (line 164)
        is_interactive_call_result_230613 = invoke(stypy.reporting.localization.Localization(__file__, 164, 11), is_interactive_230611, *[], **kwargs_230612)
        
        # Testing the type of an if condition (line 164)
        if_condition_230614 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 8), is_interactive_call_result_230613)
        # Assigning a type to the variable 'if_condition_230614' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'if_condition_230614', if_condition_230614)
        # SSA begins for if statement (line 164)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to show(...): (line 165)
        # Processing the call keyword arguments (line 165)
        kwargs_230617 = {}
        # Getting the type of 'self' (line 165)
        self_230615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'self', False)
        # Obtaining the member 'show' of a type (line 165)
        show_230616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), self_230615, 'show')
        # Calling show(args, kwargs) (line 165)
        show_call_result_230618 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), show_230616, *[], **kwargs_230617)
        
        
        # Call to draw_idle(...): (line 166)
        # Processing the call keyword arguments (line 166)
        kwargs_230622 = {}
        # Getting the type of 'self' (line 166)
        self_230619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'self', False)
        # Obtaining the member 'canvas' of a type (line 166)
        canvas_230620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), self_230619, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 166)
        draw_idle_230621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), canvas_230620, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 166)
        draw_idle_call_result_230623 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), draw_idle_230621, *[], **kwargs_230622)
        
        # SSA join for if statement (line 164)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def close(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'close'
        module_type_store = module_type_store.open_function_context('close', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerMac.close.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerMac.close.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerMac.close.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerMac.close.__dict__.__setitem__('stypy_function_name', 'FigureManagerMac.close')
        FigureManagerMac.close.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerMac.close.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerMac.close.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerMac.close.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerMac.close.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerMac.close.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerMac.close.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerMac.close', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'close', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'close(...)' code ##################

        
        # Call to destroy(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'self' (line 169)
        self_230626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'self', False)
        # Obtaining the member 'num' of a type (line 169)
        num_230627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 20), self_230626, 'num')
        # Processing the call keyword arguments (line 169)
        kwargs_230628 = {}
        # Getting the type of 'Gcf' (line 169)
        Gcf_230624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'Gcf', False)
        # Obtaining the member 'destroy' of a type (line 169)
        destroy_230625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), Gcf_230624, 'destroy')
        # Calling destroy(args, kwargs) (line 169)
        destroy_call_result_230629 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), destroy_230625, *[num_230627], **kwargs_230628)
        
        
        # ################# End of 'close(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'close' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_230630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230630)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'close'
        return stypy_return_type_230630


# Assigning a type to the variable 'FigureManagerMac' (line 144)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'FigureManagerMac', FigureManagerMac)
# Declaration of the 'NavigationToolbar2Mac' class
# Getting the type of '_macosx' (line 172)
_macosx_230631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), '_macosx')
# Obtaining the member 'NavigationToolbar2' of a type (line 172)
NavigationToolbar2_230632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 28), _macosx_230631, 'NavigationToolbar2')
# Getting the type of 'NavigationToolbar2' (line 172)
NavigationToolbar2_230633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 56), 'NavigationToolbar2')

class NavigationToolbar2Mac(NavigationToolbar2_230632, NavigationToolbar2_230633, ):

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2Mac.__init__', ['canvas'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['canvas'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'self' (line 175)
        self_230636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 36), 'self', False)
        # Getting the type of 'canvas' (line 175)
        canvas_230637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 42), 'canvas', False)
        # Processing the call keyword arguments (line 175)
        kwargs_230638 = {}
        # Getting the type of 'NavigationToolbar2' (line 175)
        NavigationToolbar2_230634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'NavigationToolbar2', False)
        # Obtaining the member '__init__' of a type (line 175)
        init___230635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), NavigationToolbar2_230634, '__init__')
        # Calling __init__(args, kwargs) (line 175)
        init___call_result_230639 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), init___230635, *[self_230636, canvas_230637], **kwargs_230638)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _init_toolbar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init_toolbar'
        module_type_store = module_type_store.open_function_context('_init_toolbar', 177, 4, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2Mac._init_toolbar.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2Mac._init_toolbar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2Mac._init_toolbar.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2Mac._init_toolbar.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2Mac._init_toolbar')
        NavigationToolbar2Mac._init_toolbar.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2Mac._init_toolbar.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2Mac._init_toolbar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2Mac._init_toolbar.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2Mac._init_toolbar.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2Mac._init_toolbar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2Mac._init_toolbar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2Mac._init_toolbar', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 178):
        
        # Assigning a Call to a Name (line 178):
        
        # Call to join(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Obtaining the type of the subscript
        unicode_230643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 40), 'unicode', u'datapath')
        # Getting the type of 'rcParams' (line 178)
        rcParams_230644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 31), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___230645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 31), rcParams_230644, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_230646 = invoke(stypy.reporting.localization.Localization(__file__, 178, 31), getitem___230645, unicode_230643)
        
        unicode_230647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 53), 'unicode', u'images')
        # Processing the call keyword arguments (line 178)
        kwargs_230648 = {}
        # Getting the type of 'os' (line 178)
        os_230640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 178)
        path_230641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 18), os_230640, 'path')
        # Obtaining the member 'join' of a type (line 178)
        join_230642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 18), path_230641, 'join')
        # Calling join(args, kwargs) (line 178)
        join_call_result_230649 = invoke(stypy.reporting.localization.Localization(__file__, 178, 18), join_230642, *[subscript_call_result_230646, unicode_230647], **kwargs_230648)
        
        # Assigning a type to the variable 'basedir' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'basedir', join_call_result_230649)
        
        # Call to __init__(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'self' (line 179)
        self_230653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 44), 'self', False)
        # Getting the type of 'basedir' (line 179)
        basedir_230654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 50), 'basedir', False)
        # Processing the call keyword arguments (line 179)
        kwargs_230655 = {}
        # Getting the type of '_macosx' (line 179)
        _macosx_230650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), '_macosx', False)
        # Obtaining the member 'NavigationToolbar2' of a type (line 179)
        NavigationToolbar2_230651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), _macosx_230650, 'NavigationToolbar2')
        # Obtaining the member '__init__' of a type (line 179)
        init___230652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), NavigationToolbar2_230651, '__init__')
        # Calling __init__(args, kwargs) (line 179)
        init___call_result_230656 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), init___230652, *[self_230653, basedir_230654], **kwargs_230655)
        
        
        # ################# End of '_init_toolbar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init_toolbar' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_230657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230657)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init_toolbar'
        return stypy_return_type_230657


    @norecursion
    def draw_rubberband(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_rubberband'
        module_type_store = module_type_store.open_function_context('draw_rubberband', 181, 4, False)
        # Assigning a type to the variable 'self' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2Mac.draw_rubberband.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2Mac.draw_rubberband.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2Mac.draw_rubberband.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2Mac.draw_rubberband.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2Mac.draw_rubberband')
        NavigationToolbar2Mac.draw_rubberband.__dict__.__setitem__('stypy_param_names_list', ['event', 'x0', 'y0', 'x1', 'y1'])
        NavigationToolbar2Mac.draw_rubberband.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2Mac.draw_rubberband.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2Mac.draw_rubberband.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2Mac.draw_rubberband.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2Mac.draw_rubberband.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2Mac.draw_rubberband.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2Mac.draw_rubberband', ['event', 'x0', 'y0', 'x1', 'y1'], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_rubberband(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Call to int(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'x0' (line 182)
        x0_230662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 39), 'x0', False)
        # Processing the call keyword arguments (line 182)
        kwargs_230663 = {}
        # Getting the type of 'int' (line 182)
        int_230661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 35), 'int', False)
        # Calling int(args, kwargs) (line 182)
        int_call_result_230664 = invoke(stypy.reporting.localization.Localization(__file__, 182, 35), int_230661, *[x0_230662], **kwargs_230663)
        
        
        # Call to int(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'y0' (line 182)
        y0_230666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 48), 'y0', False)
        # Processing the call keyword arguments (line 182)
        kwargs_230667 = {}
        # Getting the type of 'int' (line 182)
        int_230665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 44), 'int', False)
        # Calling int(args, kwargs) (line 182)
        int_call_result_230668 = invoke(stypy.reporting.localization.Localization(__file__, 182, 44), int_230665, *[y0_230666], **kwargs_230667)
        
        
        # Call to int(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'x1' (line 182)
        x1_230670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 57), 'x1', False)
        # Processing the call keyword arguments (line 182)
        kwargs_230671 = {}
        # Getting the type of 'int' (line 182)
        int_230669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 53), 'int', False)
        # Calling int(args, kwargs) (line 182)
        int_call_result_230672 = invoke(stypy.reporting.localization.Localization(__file__, 182, 53), int_230669, *[x1_230670], **kwargs_230671)
        
        
        # Call to int(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'y1' (line 182)
        y1_230674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 66), 'y1', False)
        # Processing the call keyword arguments (line 182)
        kwargs_230675 = {}
        # Getting the type of 'int' (line 182)
        int_230673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 62), 'int', False)
        # Calling int(args, kwargs) (line 182)
        int_call_result_230676 = invoke(stypy.reporting.localization.Localization(__file__, 182, 62), int_230673, *[y1_230674], **kwargs_230675)
        
        # Processing the call keyword arguments (line 182)
        kwargs_230677 = {}
        # Getting the type of 'self' (line 182)
        self_230658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 182)
        canvas_230659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), self_230658, 'canvas')
        # Obtaining the member 'set_rubberband' of a type (line 182)
        set_rubberband_230660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), canvas_230659, 'set_rubberband')
        # Calling set_rubberband(args, kwargs) (line 182)
        set_rubberband_call_result_230678 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), set_rubberband_230660, *[int_call_result_230664, int_call_result_230668, int_call_result_230672, int_call_result_230676], **kwargs_230677)
        
        
        # ################# End of 'draw_rubberband(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_rubberband' in the type store
        # Getting the type of 'stypy_return_type' (line 181)
        stypy_return_type_230679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230679)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_rubberband'
        return stypy_return_type_230679


    @norecursion
    def release(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'release'
        module_type_store = module_type_store.open_function_context('release', 184, 4, False)
        # Assigning a type to the variable 'self' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2Mac.release.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2Mac.release.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2Mac.release.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2Mac.release.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2Mac.release')
        NavigationToolbar2Mac.release.__dict__.__setitem__('stypy_param_names_list', ['event'])
        NavigationToolbar2Mac.release.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2Mac.release.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2Mac.release.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2Mac.release.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2Mac.release.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2Mac.release.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2Mac.release', ['event'], None, None, defaults, varargs, kwargs)

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

        
        # Call to remove_rubberband(...): (line 185)
        # Processing the call keyword arguments (line 185)
        kwargs_230683 = {}
        # Getting the type of 'self' (line 185)
        self_230680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 185)
        canvas_230681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), self_230680, 'canvas')
        # Obtaining the member 'remove_rubberband' of a type (line 185)
        remove_rubberband_230682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), canvas_230681, 'remove_rubberband')
        # Calling remove_rubberband(args, kwargs) (line 185)
        remove_rubberband_call_result_230684 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), remove_rubberband_230682, *[], **kwargs_230683)
        
        
        # ################# End of 'release(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'release' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_230685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230685)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'release'
        return stypy_return_type_230685


    @norecursion
    def set_cursor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_cursor'
        module_type_store = module_type_store.open_function_context('set_cursor', 187, 4, False)
        # Assigning a type to the variable 'self' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2Mac.set_cursor.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2Mac.set_cursor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2Mac.set_cursor.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2Mac.set_cursor.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2Mac.set_cursor')
        NavigationToolbar2Mac.set_cursor.__dict__.__setitem__('stypy_param_names_list', ['cursor'])
        NavigationToolbar2Mac.set_cursor.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2Mac.set_cursor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2Mac.set_cursor.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2Mac.set_cursor.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2Mac.set_cursor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2Mac.set_cursor.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2Mac.set_cursor', ['cursor'], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_cursor(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'cursor' (line 188)
        cursor_230688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 27), 'cursor', False)
        # Processing the call keyword arguments (line 188)
        kwargs_230689 = {}
        # Getting the type of '_macosx' (line 188)
        _macosx_230686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), '_macosx', False)
        # Obtaining the member 'set_cursor' of a type (line 188)
        set_cursor_230687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), _macosx_230686, 'set_cursor')
        # Calling set_cursor(args, kwargs) (line 188)
        set_cursor_call_result_230690 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), set_cursor_230687, *[cursor_230688], **kwargs_230689)
        
        
        # ################# End of 'set_cursor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_cursor' in the type store
        # Getting the type of 'stypy_return_type' (line 187)
        stypy_return_type_230691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230691)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_cursor'
        return stypy_return_type_230691


    @norecursion
    def save_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'save_figure'
        module_type_store = module_type_store.open_function_context('save_figure', 190, 4, False)
        # Assigning a type to the variable 'self' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2Mac.save_figure.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2Mac.save_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2Mac.save_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2Mac.save_figure.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2Mac.save_figure')
        NavigationToolbar2Mac.save_figure.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2Mac.save_figure.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        NavigationToolbar2Mac.save_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2Mac.save_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2Mac.save_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2Mac.save_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2Mac.save_figure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2Mac.save_figure', [], 'args', None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 191):
        
        # Assigning a Call to a Name (line 191):
        
        # Call to choose_save_file(...): (line 191)
        # Processing the call arguments (line 191)
        unicode_230694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 44), 'unicode', u'Save the figure')
        
        # Call to get_default_filename(...): (line 192)
        # Processing the call keyword arguments (line 192)
        kwargs_230698 = {}
        # Getting the type of 'self' (line 192)
        self_230695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 44), 'self', False)
        # Obtaining the member 'canvas' of a type (line 192)
        canvas_230696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 44), self_230695, 'canvas')
        # Obtaining the member 'get_default_filename' of a type (line 192)
        get_default_filename_230697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 44), canvas_230696, 'get_default_filename')
        # Calling get_default_filename(args, kwargs) (line 192)
        get_default_filename_call_result_230699 = invoke(stypy.reporting.localization.Localization(__file__, 192, 44), get_default_filename_230697, *[], **kwargs_230698)
        
        # Processing the call keyword arguments (line 191)
        kwargs_230700 = {}
        # Getting the type of '_macosx' (line 191)
        _macosx_230692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 19), '_macosx', False)
        # Obtaining the member 'choose_save_file' of a type (line 191)
        choose_save_file_230693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 19), _macosx_230692, 'choose_save_file')
        # Calling choose_save_file(args, kwargs) (line 191)
        choose_save_file_call_result_230701 = invoke(stypy.reporting.localization.Localization(__file__, 191, 19), choose_save_file_230693, *[unicode_230694, get_default_filename_call_result_230699], **kwargs_230700)
        
        # Assigning a type to the variable 'filename' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'filename', choose_save_file_call_result_230701)
        
        # Type idiom detected: calculating its left and rigth part (line 193)
        # Getting the type of 'filename' (line 193)
        filename_230702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'filename')
        # Getting the type of 'None' (line 193)
        None_230703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'None')
        
        (may_be_230704, more_types_in_union_230705) = may_be_none(filename_230702, None_230703)

        if may_be_230704:

            if more_types_in_union_230705:
                # Runtime conditional SSA (line 193)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_230705:
                # SSA join for if statement (line 193)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to savefig(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'filename' (line 195)
        filename_230710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 35), 'filename', False)
        # Processing the call keyword arguments (line 195)
        kwargs_230711 = {}
        # Getting the type of 'self' (line 195)
        self_230706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 195)
        canvas_230707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), self_230706, 'canvas')
        # Obtaining the member 'figure' of a type (line 195)
        figure_230708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), canvas_230707, 'figure')
        # Obtaining the member 'savefig' of a type (line 195)
        savefig_230709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), figure_230708, 'savefig')
        # Calling savefig(args, kwargs) (line 195)
        savefig_call_result_230712 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), savefig_230709, *[filename_230710], **kwargs_230711)
        
        
        # ################# End of 'save_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'save_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 190)
        stypy_return_type_230713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230713)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'save_figure'
        return stypy_return_type_230713


    @norecursion
    def prepare_configure_subplots(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'prepare_configure_subplots'
        module_type_store = module_type_store.open_function_context('prepare_configure_subplots', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2Mac.prepare_configure_subplots.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2Mac.prepare_configure_subplots.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2Mac.prepare_configure_subplots.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2Mac.prepare_configure_subplots.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2Mac.prepare_configure_subplots')
        NavigationToolbar2Mac.prepare_configure_subplots.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2Mac.prepare_configure_subplots.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2Mac.prepare_configure_subplots.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2Mac.prepare_configure_subplots.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2Mac.prepare_configure_subplots.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2Mac.prepare_configure_subplots.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2Mac.prepare_configure_subplots.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2Mac.prepare_configure_subplots', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'prepare_configure_subplots', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'prepare_configure_subplots(...)' code ##################

        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Call to Figure(...): (line 198)
        # Processing the call keyword arguments (line 198)
        
        # Obtaining an instance of the builtin type 'tuple' (line 198)
        tuple_230715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 198)
        # Adding element type (line 198)
        int_230716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 34), tuple_230715, int_230716)
        # Adding element type (line 198)
        int_230717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 34), tuple_230715, int_230717)
        
        keyword_230718 = tuple_230715
        kwargs_230719 = {'figsize': keyword_230718}
        # Getting the type of 'Figure' (line 198)
        Figure_230714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 18), 'Figure', False)
        # Calling Figure(args, kwargs) (line 198)
        Figure_call_result_230720 = invoke(stypy.reporting.localization.Localization(__file__, 198, 18), Figure_230714, *[], **kwargs_230719)
        
        # Assigning a type to the variable 'toolfig' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'toolfig', Figure_call_result_230720)
        
        # Assigning a Call to a Name (line 199):
        
        # Assigning a Call to a Name (line 199):
        
        # Call to FigureCanvasMac(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'toolfig' (line 199)
        toolfig_230722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 33), 'toolfig', False)
        # Processing the call keyword arguments (line 199)
        kwargs_230723 = {}
        # Getting the type of 'FigureCanvasMac' (line 199)
        FigureCanvasMac_230721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 17), 'FigureCanvasMac', False)
        # Calling FigureCanvasMac(args, kwargs) (line 199)
        FigureCanvasMac_call_result_230724 = invoke(stypy.reporting.localization.Localization(__file__, 199, 17), FigureCanvasMac_230721, *[toolfig_230722], **kwargs_230723)
        
        # Assigning a type to the variable 'canvas' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'canvas', FigureCanvasMac_call_result_230724)
        
        # Call to subplots_adjust(...): (line 200)
        # Processing the call keyword arguments (line 200)
        float_230727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 36), 'float')
        keyword_230728 = float_230727
        kwargs_230729 = {'top': keyword_230728}
        # Getting the type of 'toolfig' (line 200)
        toolfig_230725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'toolfig', False)
        # Obtaining the member 'subplots_adjust' of a type (line 200)
        subplots_adjust_230726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), toolfig_230725, 'subplots_adjust')
        # Calling subplots_adjust(args, kwargs) (line 200)
        subplots_adjust_call_result_230730 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), subplots_adjust_230726, *[], **kwargs_230729)
        
        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to SubplotTool(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'self' (line 201)
        self_230732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 27), 'self', False)
        # Obtaining the member 'canvas' of a type (line 201)
        canvas_230733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 27), self_230732, 'canvas')
        # Obtaining the member 'figure' of a type (line 201)
        figure_230734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 27), canvas_230733, 'figure')
        # Getting the type of 'toolfig' (line 201)
        toolfig_230735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 47), 'toolfig', False)
        # Processing the call keyword arguments (line 201)
        kwargs_230736 = {}
        # Getting the type of 'SubplotTool' (line 201)
        SubplotTool_230731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'SubplotTool', False)
        # Calling SubplotTool(args, kwargs) (line 201)
        SubplotTool_call_result_230737 = invoke(stypy.reporting.localization.Localization(__file__, 201, 15), SubplotTool_230731, *[figure_230734, toolfig_230735], **kwargs_230736)
        
        # Assigning a type to the variable 'tool' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tool', SubplotTool_call_result_230737)
        # Getting the type of 'canvas' (line 202)
        canvas_230738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 15), 'canvas')
        # Assigning a type to the variable 'stypy_return_type' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'stypy_return_type', canvas_230738)
        
        # ################# End of 'prepare_configure_subplots(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'prepare_configure_subplots' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_230739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230739)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'prepare_configure_subplots'
        return stypy_return_type_230739


    @norecursion
    def set_message(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_message'
        module_type_store = module_type_store.open_function_context('set_message', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2Mac.set_message.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2Mac.set_message.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2Mac.set_message.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2Mac.set_message.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2Mac.set_message')
        NavigationToolbar2Mac.set_message.__dict__.__setitem__('stypy_param_names_list', ['message'])
        NavigationToolbar2Mac.set_message.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2Mac.set_message.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2Mac.set_message.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2Mac.set_message.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2Mac.set_message.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2Mac.set_message.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2Mac.set_message', ['message'], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_message(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'self' (line 205)
        self_230743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 47), 'self', False)
        
        # Call to encode(...): (line 205)
        # Processing the call arguments (line 205)
        unicode_230746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 68), 'unicode', u'utf-8')
        # Processing the call keyword arguments (line 205)
        kwargs_230747 = {}
        # Getting the type of 'message' (line 205)
        message_230744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 53), 'message', False)
        # Obtaining the member 'encode' of a type (line 205)
        encode_230745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 53), message_230744, 'encode')
        # Calling encode(args, kwargs) (line 205)
        encode_call_result_230748 = invoke(stypy.reporting.localization.Localization(__file__, 205, 53), encode_230745, *[unicode_230746], **kwargs_230747)
        
        # Processing the call keyword arguments (line 205)
        kwargs_230749 = {}
        # Getting the type of '_macosx' (line 205)
        _macosx_230740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), '_macosx', False)
        # Obtaining the member 'NavigationToolbar2' of a type (line 205)
        NavigationToolbar2_230741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), _macosx_230740, 'NavigationToolbar2')
        # Obtaining the member 'set_message' of a type (line 205)
        set_message_230742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), NavigationToolbar2_230741, 'set_message')
        # Calling set_message(args, kwargs) (line 205)
        set_message_call_result_230750 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), set_message_230742, *[self_230743, encode_call_result_230748], **kwargs_230749)
        
        
        # ################# End of 'set_message(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_message' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_230751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230751)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_message'
        return stypy_return_type_230751


# Assigning a type to the variable 'NavigationToolbar2Mac' (line 172)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'NavigationToolbar2Mac', NavigationToolbar2Mac)
# Declaration of the '_BackendMac' class
# Getting the type of '_Backend' (line 215)
_Backend_230752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 18), '_Backend')

class _BackendMac(_Backend_230752, ):
    
    # Assigning a Name to a Name (line 216):
    
    # Assigning a Name to a Name (line 217):

    @norecursion
    def trigger_manager_draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'trigger_manager_draw'
        module_type_store = module_type_store.open_function_context('trigger_manager_draw', 219, 4, False)
        # Assigning a type to the variable 'self' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _BackendMac.trigger_manager_draw.__dict__.__setitem__('stypy_localization', localization)
        _BackendMac.trigger_manager_draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _BackendMac.trigger_manager_draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        _BackendMac.trigger_manager_draw.__dict__.__setitem__('stypy_function_name', '_BackendMac.trigger_manager_draw')
        _BackendMac.trigger_manager_draw.__dict__.__setitem__('stypy_param_names_list', [])
        _BackendMac.trigger_manager_draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        _BackendMac.trigger_manager_draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _BackendMac.trigger_manager_draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        _BackendMac.trigger_manager_draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        _BackendMac.trigger_manager_draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _BackendMac.trigger_manager_draw.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendMac.trigger_manager_draw', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to invalidate(...): (line 225)
        # Processing the call keyword arguments (line 225)
        kwargs_230756 = {}
        # Getting the type of 'manager' (line 225)
        manager_230753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'manager', False)
        # Obtaining the member 'canvas' of a type (line 225)
        canvas_230754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), manager_230753, 'canvas')
        # Obtaining the member 'invalidate' of a type (line 225)
        invalidate_230755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), canvas_230754, 'invalidate')
        # Calling invalidate(args, kwargs) (line 225)
        invalidate_call_result_230757 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), invalidate_230755, *[], **kwargs_230756)
        
        
        # ################# End of 'trigger_manager_draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger_manager_draw' in the type store
        # Getting the type of 'stypy_return_type' (line 219)
        stypy_return_type_230758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230758)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger_manager_draw'
        return stypy_return_type_230758


    @staticmethod
    @norecursion
    def mainloop(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mainloop'
        module_type_store = module_type_store.open_function_context('mainloop', 227, 4, False)
        
        # Passed parameters checking function
        _BackendMac.mainloop.__dict__.__setitem__('stypy_localization', localization)
        _BackendMac.mainloop.__dict__.__setitem__('stypy_type_of_self', None)
        _BackendMac.mainloop.__dict__.__setitem__('stypy_type_store', module_type_store)
        _BackendMac.mainloop.__dict__.__setitem__('stypy_function_name', 'mainloop')
        _BackendMac.mainloop.__dict__.__setitem__('stypy_param_names_list', [])
        _BackendMac.mainloop.__dict__.__setitem__('stypy_varargs_param_name', None)
        _BackendMac.mainloop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _BackendMac.mainloop.__dict__.__setitem__('stypy_call_defaults', defaults)
        _BackendMac.mainloop.__dict__.__setitem__('stypy_call_varargs', varargs)
        _BackendMac.mainloop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _BackendMac.mainloop.__dict__.__setitem__('stypy_declared_arg_number', 0)
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

        
        # Call to show(...): (line 229)
        # Processing the call keyword arguments (line 229)
        kwargs_230761 = {}
        # Getting the type of '_macosx' (line 229)
        _macosx_230759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), '_macosx', False)
        # Obtaining the member 'show' of a type (line 229)
        show_230760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), _macosx_230759, 'show')
        # Calling show(args, kwargs) (line 229)
        show_call_result_230762 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), show_230760, *[], **kwargs_230761)
        
        
        # ################# End of 'mainloop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mainloop' in the type store
        # Getting the type of 'stypy_return_type' (line 227)
        stypy_return_type_230763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230763)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mainloop'
        return stypy_return_type_230763


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 214, 0, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendMac.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendMac' (line 214)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), '_BackendMac', _BackendMac)

# Assigning a Name to a Name (line 216):
# Getting the type of 'FigureCanvasMac' (line 216)
FigureCanvasMac_230764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 19), 'FigureCanvasMac')
# Getting the type of '_BackendMac'
_BackendMac_230765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendMac')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendMac_230765, 'FigureCanvas', FigureCanvasMac_230764)

# Assigning a Name to a Name (line 217):
# Getting the type of 'FigureManagerMac' (line 217)
FigureManagerMac_230766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 20), 'FigureManagerMac')
# Getting the type of '_BackendMac'
_BackendMac_230767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendMac')
# Setting the type of the member 'FigureManager' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendMac_230767, 'FigureManager', FigureManagerMac_230766)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
