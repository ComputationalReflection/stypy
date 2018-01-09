
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Render to qt from agg
3: '''
4: from __future__ import (absolute_import, division, print_function,
5:                         unicode_literals)
6: 
7: import six
8: 
9: import ctypes
10: import traceback
11: 
12: from matplotlib import cbook
13: from matplotlib.transforms import Bbox
14: 
15: from .backend_agg import FigureCanvasAgg
16: from .backend_qt5 import (
17:     QtCore, QtGui, QtWidgets, _BackendQT5, FigureCanvasQT, FigureManagerQT,
18:     NavigationToolbar2QT, backend_version)
19: from .qt_compat import QT_API
20: 
21: 
22: class FigureCanvasQTAggBase(FigureCanvasAgg):
23:     '''
24:     The canvas the figure renders into.  Calls the draw and print fig
25:     methods, creates the renderers, etc...
26: 
27:     Attributes
28:     ----------
29:     figure : `matplotlib.figure.Figure`
30:         A high-level Figure instance
31: 
32:     '''
33: 
34:     def __init__(self, figure):
35:         super(FigureCanvasQTAggBase, self).__init__(figure=figure)
36:         self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
37:         self._agg_draw_pending = False
38:         self._bbox_queue = []
39:         self._drawRect = None
40: 
41:     def drawRectangle(self, rect):
42:         if rect is not None:
43:             self._drawRect = [pt / self._dpi_ratio for pt in rect]
44:         else:
45:             self._drawRect = None
46:         self.update()
47: 
48:     @property
49:     @cbook.deprecated("2.1")
50:     def blitbox(self):
51:         return self._bbox_queue
52: 
53:     def paintEvent(self, e):
54:         '''Copy the image from the Agg canvas to the qt.drawable.
55: 
56:         In Qt, all drawing should be done inside of here when a widget is
57:         shown onscreen.
58:         '''
59:         # if there is a pending draw, run it now as we need the updated render
60:         # to paint the widget
61:         if self._agg_draw_pending:
62:             self.__draw_idle_agg()
63:         # As described in __init__ above, we need to be careful in cases with
64:         # mixed resolution displays if dpi_ratio is changing between painting
65:         # events.
66:         if self._dpi_ratio != self._dpi_ratio_prev:
67:             # We need to update the figure DPI
68:             self._update_figure_dpi()
69:             self._dpi_ratio_prev = self._dpi_ratio
70:             # The easiest way to resize the canvas is to emit a resizeEvent
71:             # since we implement all the logic for resizing the canvas for
72:             # that event.
73:             event = QtGui.QResizeEvent(self.size(), self.size())
74:             # We use self.resizeEvent here instead of QApplication.postEvent
75:             # since the latter doesn't guarantee that the event will be emitted
76:             # straight away, and this causes visual delays in the changes.
77:             self.resizeEvent(event)
78:             # resizeEvent triggers a paintEvent itself, so we exit this one.
79:             return
80: 
81:         # if the canvas does not have a renderer, then give up and wait for
82:         # FigureCanvasAgg.draw(self) to be called
83:         if not hasattr(self, 'renderer'):
84:             return
85: 
86:         painter = QtGui.QPainter(self)
87: 
88:         if self._bbox_queue:
89:             bbox_queue = self._bbox_queue
90:         else:
91:             painter.eraseRect(self.rect())
92:             bbox_queue = [
93:                 Bbox([[0, 0], [self.renderer.width, self.renderer.height]])]
94:         self._bbox_queue = []
95:         for bbox in bbox_queue:
96:             l, b, r, t = map(int, bbox.extents)
97:             w = r - l
98:             h = t - b
99:             reg = self.copy_from_bbox(bbox)
100:             buf = reg.to_string_argb()
101:             qimage = QtGui.QImage(buf, w, h, QtGui.QImage.Format_ARGB32)
102:             if hasattr(qimage, 'setDevicePixelRatio'):
103:                 # Not available on Qt4 or some older Qt5.
104:                 qimage.setDevicePixelRatio(self._dpi_ratio)
105:             origin = QtCore.QPoint(l, self.renderer.height - t)
106:             painter.drawImage(origin / self._dpi_ratio, qimage)
107:             # Adjust the buf reference count to work around a memory
108:             # leak bug in QImage under PySide on Python 3.
109:             if QT_API == 'PySide' and six.PY3:
110:                 ctypes.c_long.from_address(id(buf)).value = 1
111: 
112:         # draw the zoom rectangle to the QPainter
113:         if self._drawRect is not None:
114:             pen = QtGui.QPen(QtCore.Qt.black, 1 / self._dpi_ratio,
115:                              QtCore.Qt.DotLine)
116:             painter.setPen(pen)
117:             x, y, w, h = self._drawRect
118:             painter.drawRect(x, y, w, h)
119: 
120:         painter.end()
121: 
122:     def draw(self):
123:         '''Draw the figure with Agg, and queue a request for a Qt draw.
124:         '''
125:         # The Agg draw is done here; delaying causes problems with code that
126:         # uses the result of the draw() to update plot elements.
127:         super(FigureCanvasQTAggBase, self).draw()
128:         self.update()
129: 
130:     def draw_idle(self):
131:         '''Queue redraw of the Agg buffer and request Qt paintEvent.
132:         '''
133:         # The Agg draw needs to be handled by the same thread matplotlib
134:         # modifies the scene graph from. Post Agg draw request to the
135:         # current event loop in order to ensure thread affinity and to
136:         # accumulate multiple draw requests from event handling.
137:         # TODO: queued signal connection might be safer than singleShot
138:         if not self._agg_draw_pending:
139:             self._agg_draw_pending = True
140:             QtCore.QTimer.singleShot(0, self.__draw_idle_agg)
141: 
142:     def __draw_idle_agg(self, *args):
143:         if not self._agg_draw_pending:
144:             return
145:         if self.height() < 0 or self.width() < 0:
146:             self._agg_draw_pending = False
147:             return
148:         try:
149:             self.draw()
150:         except Exception:
151:             # Uncaught exceptions are fatal for PyQt5, so catch them instead.
152:             traceback.print_exc()
153:         finally:
154:             self._agg_draw_pending = False
155: 
156:     def blit(self, bbox=None):
157:         '''Blit the region in bbox.
158:         '''
159:         # If bbox is None, blit the entire canvas. Otherwise
160:         # blit only the area defined by the bbox.
161:         if bbox is None and self.figure:
162:             bbox = self.figure.bbox
163: 
164:         self._bbox_queue.append(bbox)
165: 
166:         # repaint uses logical pixels, not physical pixels like the renderer.
167:         l, b, w, h = [pt / self._dpi_ratio for pt in bbox.bounds]
168:         t = b + h
169:         self.repaint(l, self.renderer.height / self._dpi_ratio - t, w, h)
170: 
171:     def print_figure(self, *args, **kwargs):
172:         super(FigureCanvasQTAggBase, self).print_figure(*args, **kwargs)
173:         self.draw()
174: 
175: 
176: class FigureCanvasQTAgg(FigureCanvasQTAggBase, FigureCanvasQT):
177:     '''
178:     The canvas the figure renders into.  Calls the draw and print fig
179:     methods, creates the renderers, etc.
180: 
181:     Modified to import from Qt5 backend for new-style mouse events.
182: 
183:     Attributes
184:     ----------
185:     figure : `matplotlib.figure.Figure`
186:         A high-level Figure instance
187: 
188:     '''
189: 
190: 
191: @_BackendQT5.export
192: class _BackendQT5Agg(_BackendQT5):
193:     FigureCanvas = FigureCanvasQTAgg
194: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_252321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nRender to qt from agg\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import six' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_252322 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six')

if (type(import_252322) is not StypyTypeError):

    if (import_252322 != 'pyd_module'):
        __import__(import_252322)
        sys_modules_252323 = sys.modules[import_252322]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', sys_modules_252323.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', import_252322)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import ctypes' statement (line 9)
import ctypes

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'ctypes', ctypes, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import traceback' statement (line 10)
import traceback

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'traceback', traceback, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from matplotlib import cbook' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_252324 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib')

if (type(import_252324) is not StypyTypeError):

    if (import_252324 != 'pyd_module'):
        __import__(import_252324)
        sys_modules_252325 = sys.modules[import_252324]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib', sys_modules_252325.module_type_store, module_type_store, ['cbook'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_252325, sys_modules_252325.module_type_store, module_type_store)
    else:
        from matplotlib import cbook

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib', None, module_type_store, ['cbook'], [cbook])

else:
    # Assigning a type to the variable 'matplotlib' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib', import_252324)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from matplotlib.transforms import Bbox' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_252326 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.transforms')

if (type(import_252326) is not StypyTypeError):

    if (import_252326 != 'pyd_module'):
        __import__(import_252326)
        sys_modules_252327 = sys.modules[import_252326]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.transforms', sys_modules_252327.module_type_store, module_type_store, ['Bbox'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_252327, sys_modules_252327.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Bbox

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.transforms', None, module_type_store, ['Bbox'], [Bbox])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.transforms', import_252326)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib.backends.backend_agg import FigureCanvasAgg' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_252328 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backends.backend_agg')

if (type(import_252328) is not StypyTypeError):

    if (import_252328 != 'pyd_module'):
        __import__(import_252328)
        sys_modules_252329 = sys.modules[import_252328]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backends.backend_agg', sys_modules_252329.module_type_store, module_type_store, ['FigureCanvasAgg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_252329, sys_modules_252329.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backends.backend_agg', None, module_type_store, ['FigureCanvasAgg'], [FigureCanvasAgg])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_agg' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backends.backend_agg', import_252328)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from matplotlib.backends.backend_qt5 import QtCore, QtGui, QtWidgets, _BackendQT5, FigureCanvasQT, FigureManagerQT, NavigationToolbar2QT, backend_version' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_252330 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.backends.backend_qt5')

if (type(import_252330) is not StypyTypeError):

    if (import_252330 != 'pyd_module'):
        __import__(import_252330)
        sys_modules_252331 = sys.modules[import_252330]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.backends.backend_qt5', sys_modules_252331.module_type_store, module_type_store, ['QtCore', 'QtGui', 'QtWidgets', '_BackendQT5', 'FigureCanvasQT', 'FigureManagerQT', 'NavigationToolbar2QT', 'backend_version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_252331, sys_modules_252331.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_qt5 import QtCore, QtGui, QtWidgets, _BackendQT5, FigureCanvasQT, FigureManagerQT, NavigationToolbar2QT, backend_version

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.backends.backend_qt5', None, module_type_store, ['QtCore', 'QtGui', 'QtWidgets', '_BackendQT5', 'FigureCanvasQT', 'FigureManagerQT', 'NavigationToolbar2QT', 'backend_version'], [QtCore, QtGui, QtWidgets, _BackendQT5, FigureCanvasQT, FigureManagerQT, NavigationToolbar2QT, backend_version])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_qt5' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.backends.backend_qt5', import_252330)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from matplotlib.backends.qt_compat import QT_API' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_252332 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.backends.qt_compat')

if (type(import_252332) is not StypyTypeError):

    if (import_252332 != 'pyd_module'):
        __import__(import_252332)
        sys_modules_252333 = sys.modules[import_252332]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.backends.qt_compat', sys_modules_252333.module_type_store, module_type_store, ['QT_API'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_252333, sys_modules_252333.module_type_store, module_type_store)
    else:
        from matplotlib.backends.qt_compat import QT_API

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.backends.qt_compat', None, module_type_store, ['QT_API'], [QT_API])

else:
    # Assigning a type to the variable 'matplotlib.backends.qt_compat' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.backends.qt_compat', import_252332)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# Declaration of the 'FigureCanvasQTAggBase' class
# Getting the type of 'FigureCanvasAgg' (line 22)
FigureCanvasAgg_252334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 28), 'FigureCanvasAgg')

class FigureCanvasQTAggBase(FigureCanvasAgg_252334, ):
    unicode_252335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'unicode', u'\n    The canvas the figure renders into.  Calls the draw and print fig\n    methods, creates the renderers, etc...\n\n    Attributes\n    ----------\n    figure : `matplotlib.figure.Figure`\n        A high-level Figure instance\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQTAggBase.__init__', ['figure'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 35)
        # Processing the call keyword arguments (line 35)
        # Getting the type of 'figure' (line 35)
        figure_252342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 59), 'figure', False)
        keyword_252343 = figure_252342
        kwargs_252344 = {'figure': keyword_252343}
        
        # Call to super(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'FigureCanvasQTAggBase' (line 35)
        FigureCanvasQTAggBase_252337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'FigureCanvasQTAggBase', False)
        # Getting the type of 'self' (line 35)
        self_252338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 37), 'self', False)
        # Processing the call keyword arguments (line 35)
        kwargs_252339 = {}
        # Getting the type of 'super' (line 35)
        super_252336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'super', False)
        # Calling super(args, kwargs) (line 35)
        super_call_result_252340 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), super_252336, *[FigureCanvasQTAggBase_252337, self_252338], **kwargs_252339)
        
        # Obtaining the member '__init__' of a type (line 35)
        init___252341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), super_call_result_252340, '__init__')
        # Calling __init__(args, kwargs) (line 35)
        init___call_result_252345 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), init___252341, *[], **kwargs_252344)
        
        
        # Call to setAttribute(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'QtCore' (line 36)
        QtCore_252348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 26), 'QtCore', False)
        # Obtaining the member 'Qt' of a type (line 36)
        Qt_252349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 26), QtCore_252348, 'Qt')
        # Obtaining the member 'WA_OpaquePaintEvent' of a type (line 36)
        WA_OpaquePaintEvent_252350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 26), Qt_252349, 'WA_OpaquePaintEvent')
        # Processing the call keyword arguments (line 36)
        kwargs_252351 = {}
        # Getting the type of 'self' (line 36)
        self_252346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self', False)
        # Obtaining the member 'setAttribute' of a type (line 36)
        setAttribute_252347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_252346, 'setAttribute')
        # Calling setAttribute(args, kwargs) (line 36)
        setAttribute_call_result_252352 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), setAttribute_252347, *[WA_OpaquePaintEvent_252350], **kwargs_252351)
        
        
        # Assigning a Name to a Attribute (line 37):
        
        # Assigning a Name to a Attribute (line 37):
        # Getting the type of 'False' (line 37)
        False_252353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 33), 'False')
        # Getting the type of 'self' (line 37)
        self_252354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member '_agg_draw_pending' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_252354, '_agg_draw_pending', False_252353)
        
        # Assigning a List to a Attribute (line 38):
        
        # Assigning a List to a Attribute (line 38):
        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_252355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        
        # Getting the type of 'self' (line 38)
        self_252356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member '_bbox_queue' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_252356, '_bbox_queue', list_252355)
        
        # Assigning a Name to a Attribute (line 39):
        
        # Assigning a Name to a Attribute (line 39):
        # Getting the type of 'None' (line 39)
        None_252357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'None')
        # Getting the type of 'self' (line 39)
        self_252358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member '_drawRect' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_252358, '_drawRect', None_252357)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def drawRectangle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'drawRectangle'
        module_type_store = module_type_store.open_function_context('drawRectangle', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQTAggBase.drawRectangle.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQTAggBase.drawRectangle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQTAggBase.drawRectangle.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQTAggBase.drawRectangle.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQTAggBase.drawRectangle')
        FigureCanvasQTAggBase.drawRectangle.__dict__.__setitem__('stypy_param_names_list', ['rect'])
        FigureCanvasQTAggBase.drawRectangle.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQTAggBase.drawRectangle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQTAggBase.drawRectangle.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQTAggBase.drawRectangle.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQTAggBase.drawRectangle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQTAggBase.drawRectangle.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQTAggBase.drawRectangle', ['rect'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'drawRectangle', localization, ['rect'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'drawRectangle(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 42)
        # Getting the type of 'rect' (line 42)
        rect_252359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'rect')
        # Getting the type of 'None' (line 42)
        None_252360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'None')
        
        (may_be_252361, more_types_in_union_252362) = may_not_be_none(rect_252359, None_252360)

        if may_be_252361:

            if more_types_in_union_252362:
                # Runtime conditional SSA (line 42)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a ListComp to a Attribute (line 43):
            
            # Assigning a ListComp to a Attribute (line 43):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'rect' (line 43)
            rect_252367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 61), 'rect')
            comprehension_252368 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 30), rect_252367)
            # Assigning a type to the variable 'pt' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 30), 'pt', comprehension_252368)
            # Getting the type of 'pt' (line 43)
            pt_252363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 30), 'pt')
            # Getting the type of 'self' (line 43)
            self_252364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 35), 'self')
            # Obtaining the member '_dpi_ratio' of a type (line 43)
            _dpi_ratio_252365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 35), self_252364, '_dpi_ratio')
            # Applying the binary operator 'div' (line 43)
            result_div_252366 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 30), 'div', pt_252363, _dpi_ratio_252365)
            
            list_252369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 30), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 30), list_252369, result_div_252366)
            # Getting the type of 'self' (line 43)
            self_252370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'self')
            # Setting the type of the member '_drawRect' of a type (line 43)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), self_252370, '_drawRect', list_252369)

            if more_types_in_union_252362:
                # Runtime conditional SSA for else branch (line 42)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_252361) or more_types_in_union_252362):
            
            # Assigning a Name to a Attribute (line 45):
            
            # Assigning a Name to a Attribute (line 45):
            # Getting the type of 'None' (line 45)
            None_252371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 29), 'None')
            # Getting the type of 'self' (line 45)
            self_252372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'self')
            # Setting the type of the member '_drawRect' of a type (line 45)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), self_252372, '_drawRect', None_252371)

            if (may_be_252361 and more_types_in_union_252362):
                # SSA join for if statement (line 42)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to update(...): (line 46)
        # Processing the call keyword arguments (line 46)
        kwargs_252375 = {}
        # Getting the type of 'self' (line 46)
        self_252373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self', False)
        # Obtaining the member 'update' of a type (line 46)
        update_252374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_252373, 'update')
        # Calling update(args, kwargs) (line 46)
        update_call_result_252376 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), update_252374, *[], **kwargs_252375)
        
        
        # ################# End of 'drawRectangle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'drawRectangle' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_252377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252377)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'drawRectangle'
        return stypy_return_type_252377


    @norecursion
    def blitbox(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'blitbox'
        module_type_store = module_type_store.open_function_context('blitbox', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQTAggBase.blitbox.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQTAggBase.blitbox.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQTAggBase.blitbox.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQTAggBase.blitbox.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQTAggBase.blitbox')
        FigureCanvasQTAggBase.blitbox.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasQTAggBase.blitbox.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQTAggBase.blitbox.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQTAggBase.blitbox.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQTAggBase.blitbox.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQTAggBase.blitbox.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQTAggBase.blitbox.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQTAggBase.blitbox', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'blitbox', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'blitbox(...)' code ##################

        # Getting the type of 'self' (line 51)
        self_252378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'self')
        # Obtaining the member '_bbox_queue' of a type (line 51)
        _bbox_queue_252379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), self_252378, '_bbox_queue')
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', _bbox_queue_252379)
        
        # ################# End of 'blitbox(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'blitbox' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_252380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252380)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'blitbox'
        return stypy_return_type_252380


    @norecursion
    def paintEvent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'paintEvent'
        module_type_store = module_type_store.open_function_context('paintEvent', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQTAggBase.paintEvent.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQTAggBase.paintEvent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQTAggBase.paintEvent.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQTAggBase.paintEvent.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQTAggBase.paintEvent')
        FigureCanvasQTAggBase.paintEvent.__dict__.__setitem__('stypy_param_names_list', ['e'])
        FigureCanvasQTAggBase.paintEvent.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQTAggBase.paintEvent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQTAggBase.paintEvent.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQTAggBase.paintEvent.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQTAggBase.paintEvent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQTAggBase.paintEvent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQTAggBase.paintEvent', ['e'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'paintEvent', localization, ['e'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'paintEvent(...)' code ##################

        unicode_252381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'unicode', u'Copy the image from the Agg canvas to the qt.drawable.\n\n        In Qt, all drawing should be done inside of here when a widget is\n        shown onscreen.\n        ')
        
        # Getting the type of 'self' (line 61)
        self_252382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'self')
        # Obtaining the member '_agg_draw_pending' of a type (line 61)
        _agg_draw_pending_252383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 11), self_252382, '_agg_draw_pending')
        # Testing the type of an if condition (line 61)
        if_condition_252384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 8), _agg_draw_pending_252383)
        # Assigning a type to the variable 'if_condition_252384' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'if_condition_252384', if_condition_252384)
        # SSA begins for if statement (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __draw_idle_agg(...): (line 62)
        # Processing the call keyword arguments (line 62)
        kwargs_252387 = {}
        # Getting the type of 'self' (line 62)
        self_252385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'self', False)
        # Obtaining the member '__draw_idle_agg' of a type (line 62)
        draw_idle_agg_252386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), self_252385, '__draw_idle_agg')
        # Calling __draw_idle_agg(args, kwargs) (line 62)
        draw_idle_agg_call_result_252388 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), draw_idle_agg_252386, *[], **kwargs_252387)
        
        # SSA join for if statement (line 61)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 66)
        self_252389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'self')
        # Obtaining the member '_dpi_ratio' of a type (line 66)
        _dpi_ratio_252390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 11), self_252389, '_dpi_ratio')
        # Getting the type of 'self' (line 66)
        self_252391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), 'self')
        # Obtaining the member '_dpi_ratio_prev' of a type (line 66)
        _dpi_ratio_prev_252392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 30), self_252391, '_dpi_ratio_prev')
        # Applying the binary operator '!=' (line 66)
        result_ne_252393 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 11), '!=', _dpi_ratio_252390, _dpi_ratio_prev_252392)
        
        # Testing the type of an if condition (line 66)
        if_condition_252394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 8), result_ne_252393)
        # Assigning a type to the variable 'if_condition_252394' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'if_condition_252394', if_condition_252394)
        # SSA begins for if statement (line 66)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _update_figure_dpi(...): (line 68)
        # Processing the call keyword arguments (line 68)
        kwargs_252397 = {}
        # Getting the type of 'self' (line 68)
        self_252395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'self', False)
        # Obtaining the member '_update_figure_dpi' of a type (line 68)
        _update_figure_dpi_252396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), self_252395, '_update_figure_dpi')
        # Calling _update_figure_dpi(args, kwargs) (line 68)
        _update_figure_dpi_call_result_252398 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), _update_figure_dpi_252396, *[], **kwargs_252397)
        
        
        # Assigning a Attribute to a Attribute (line 69):
        
        # Assigning a Attribute to a Attribute (line 69):
        # Getting the type of 'self' (line 69)
        self_252399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 35), 'self')
        # Obtaining the member '_dpi_ratio' of a type (line 69)
        _dpi_ratio_252400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 35), self_252399, '_dpi_ratio')
        # Getting the type of 'self' (line 69)
        self_252401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'self')
        # Setting the type of the member '_dpi_ratio_prev' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), self_252401, '_dpi_ratio_prev', _dpi_ratio_252400)
        
        # Assigning a Call to a Name (line 73):
        
        # Assigning a Call to a Name (line 73):
        
        # Call to QResizeEvent(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to size(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_252406 = {}
        # Getting the type of 'self' (line 73)
        self_252404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 39), 'self', False)
        # Obtaining the member 'size' of a type (line 73)
        size_252405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 39), self_252404, 'size')
        # Calling size(args, kwargs) (line 73)
        size_call_result_252407 = invoke(stypy.reporting.localization.Localization(__file__, 73, 39), size_252405, *[], **kwargs_252406)
        
        
        # Call to size(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_252410 = {}
        # Getting the type of 'self' (line 73)
        self_252408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 52), 'self', False)
        # Obtaining the member 'size' of a type (line 73)
        size_252409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 52), self_252408, 'size')
        # Calling size(args, kwargs) (line 73)
        size_call_result_252411 = invoke(stypy.reporting.localization.Localization(__file__, 73, 52), size_252409, *[], **kwargs_252410)
        
        # Processing the call keyword arguments (line 73)
        kwargs_252412 = {}
        # Getting the type of 'QtGui' (line 73)
        QtGui_252402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'QtGui', False)
        # Obtaining the member 'QResizeEvent' of a type (line 73)
        QResizeEvent_252403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 20), QtGui_252402, 'QResizeEvent')
        # Calling QResizeEvent(args, kwargs) (line 73)
        QResizeEvent_call_result_252413 = invoke(stypy.reporting.localization.Localization(__file__, 73, 20), QResizeEvent_252403, *[size_call_result_252407, size_call_result_252411], **kwargs_252412)
        
        # Assigning a type to the variable 'event' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'event', QResizeEvent_call_result_252413)
        
        # Call to resizeEvent(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'event' (line 77)
        event_252416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'event', False)
        # Processing the call keyword arguments (line 77)
        kwargs_252417 = {}
        # Getting the type of 'self' (line 77)
        self_252414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'self', False)
        # Obtaining the member 'resizeEvent' of a type (line 77)
        resizeEvent_252415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), self_252414, 'resizeEvent')
        # Calling resizeEvent(args, kwargs) (line 77)
        resizeEvent_call_result_252418 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), resizeEvent_252415, *[event_252416], **kwargs_252417)
        
        # Assigning a type to the variable 'stypy_return_type' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 66)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 83)
        unicode_252419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 29), 'unicode', u'renderer')
        # Getting the type of 'self' (line 83)
        self_252420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 23), 'self')
        
        (may_be_252421, more_types_in_union_252422) = may_not_provide_member(unicode_252419, self_252420)

        if may_be_252421:

            if more_types_in_union_252422:
                # Runtime conditional SSA (line 83)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'self' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self', remove_member_provider_from_union(self_252420, u'renderer'))
            # Assigning a type to the variable 'stypy_return_type' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_252422:
                # SSA join for if statement (line 83)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 86):
        
        # Assigning a Call to a Name (line 86):
        
        # Call to QPainter(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'self' (line 86)
        self_252425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 33), 'self', False)
        # Processing the call keyword arguments (line 86)
        kwargs_252426 = {}
        # Getting the type of 'QtGui' (line 86)
        QtGui_252423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'QtGui', False)
        # Obtaining the member 'QPainter' of a type (line 86)
        QPainter_252424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 18), QtGui_252423, 'QPainter')
        # Calling QPainter(args, kwargs) (line 86)
        QPainter_call_result_252427 = invoke(stypy.reporting.localization.Localization(__file__, 86, 18), QPainter_252424, *[self_252425], **kwargs_252426)
        
        # Assigning a type to the variable 'painter' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'painter', QPainter_call_result_252427)
        
        # Getting the type of 'self' (line 88)
        self_252428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'self')
        # Obtaining the member '_bbox_queue' of a type (line 88)
        _bbox_queue_252429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 11), self_252428, '_bbox_queue')
        # Testing the type of an if condition (line 88)
        if_condition_252430 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 8), _bbox_queue_252429)
        # Assigning a type to the variable 'if_condition_252430' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'if_condition_252430', if_condition_252430)
        # SSA begins for if statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 89):
        
        # Assigning a Attribute to a Name (line 89):
        # Getting the type of 'self' (line 89)
        self_252431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'self')
        # Obtaining the member '_bbox_queue' of a type (line 89)
        _bbox_queue_252432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 25), self_252431, '_bbox_queue')
        # Assigning a type to the variable 'bbox_queue' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'bbox_queue', _bbox_queue_252432)
        # SSA branch for the else part of an if statement (line 88)
        module_type_store.open_ssa_branch('else')
        
        # Call to eraseRect(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to rect(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_252437 = {}
        # Getting the type of 'self' (line 91)
        self_252435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 30), 'self', False)
        # Obtaining the member 'rect' of a type (line 91)
        rect_252436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 30), self_252435, 'rect')
        # Calling rect(args, kwargs) (line 91)
        rect_call_result_252438 = invoke(stypy.reporting.localization.Localization(__file__, 91, 30), rect_252436, *[], **kwargs_252437)
        
        # Processing the call keyword arguments (line 91)
        kwargs_252439 = {}
        # Getting the type of 'painter' (line 91)
        painter_252433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'painter', False)
        # Obtaining the member 'eraseRect' of a type (line 91)
        eraseRect_252434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), painter_252433, 'eraseRect')
        # Calling eraseRect(args, kwargs) (line 91)
        eraseRect_call_result_252440 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), eraseRect_252434, *[rect_call_result_252438], **kwargs_252439)
        
        
        # Assigning a List to a Name (line 92):
        
        # Assigning a List to a Name (line 92):
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_252441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        
        # Call to Bbox(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_252443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_252444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        int_252445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 22), list_252444, int_252445)
        # Adding element type (line 93)
        int_252446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 22), list_252444, int_252446)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 21), list_252443, list_252444)
        # Adding element type (line 93)
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_252447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        # Getting the type of 'self' (line 93)
        self_252448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'self', False)
        # Obtaining the member 'renderer' of a type (line 93)
        renderer_252449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 31), self_252448, 'renderer')
        # Obtaining the member 'width' of a type (line 93)
        width_252450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 31), renderer_252449, 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 30), list_252447, width_252450)
        # Adding element type (line 93)
        # Getting the type of 'self' (line 93)
        self_252451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 52), 'self', False)
        # Obtaining the member 'renderer' of a type (line 93)
        renderer_252452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 52), self_252451, 'renderer')
        # Obtaining the member 'height' of a type (line 93)
        height_252453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 52), renderer_252452, 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 30), list_252447, height_252453)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 21), list_252443, list_252447)
        
        # Processing the call keyword arguments (line 93)
        kwargs_252454 = {}
        # Getting the type of 'Bbox' (line 93)
        Bbox_252442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'Bbox', False)
        # Calling Bbox(args, kwargs) (line 93)
        Bbox_call_result_252455 = invoke(stypy.reporting.localization.Localization(__file__, 93, 16), Bbox_252442, *[list_252443], **kwargs_252454)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 25), list_252441, Bbox_call_result_252455)
        
        # Assigning a type to the variable 'bbox_queue' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'bbox_queue', list_252441)
        # SSA join for if statement (line 88)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Attribute (line 94):
        
        # Assigning a List to a Attribute (line 94):
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_252456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        
        # Getting the type of 'self' (line 94)
        self_252457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Setting the type of the member '_bbox_queue' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_252457, '_bbox_queue', list_252456)
        
        # Getting the type of 'bbox_queue' (line 95)
        bbox_queue_252458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 20), 'bbox_queue')
        # Testing the type of a for loop iterable (line 95)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 95, 8), bbox_queue_252458)
        # Getting the type of the for loop variable (line 95)
        for_loop_var_252459 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 95, 8), bbox_queue_252458)
        # Assigning a type to the variable 'bbox' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'bbox', for_loop_var_252459)
        # SSA begins for a for statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 96):
        
        # Assigning a Call to a Name:
        
        # Call to map(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'int' (line 96)
        int_252461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 29), 'int', False)
        # Getting the type of 'bbox' (line 96)
        bbox_252462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 34), 'bbox', False)
        # Obtaining the member 'extents' of a type (line 96)
        extents_252463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 34), bbox_252462, 'extents')
        # Processing the call keyword arguments (line 96)
        kwargs_252464 = {}
        # Getting the type of 'map' (line 96)
        map_252460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'map', False)
        # Calling map(args, kwargs) (line 96)
        map_call_result_252465 = invoke(stypy.reporting.localization.Localization(__file__, 96, 25), map_252460, *[int_252461, extents_252463], **kwargs_252464)
        
        # Assigning a type to the variable 'call_assignment_252308' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'call_assignment_252308', map_call_result_252465)
        
        # Assigning a Call to a Name (line 96):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_252468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'int')
        # Processing the call keyword arguments
        kwargs_252469 = {}
        # Getting the type of 'call_assignment_252308' (line 96)
        call_assignment_252308_252466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'call_assignment_252308', False)
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___252467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), call_assignment_252308_252466, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_252470 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___252467, *[int_252468], **kwargs_252469)
        
        # Assigning a type to the variable 'call_assignment_252309' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'call_assignment_252309', getitem___call_result_252470)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'call_assignment_252309' (line 96)
        call_assignment_252309_252471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'call_assignment_252309')
        # Assigning a type to the variable 'l' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'l', call_assignment_252309_252471)
        
        # Assigning a Call to a Name (line 96):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_252474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'int')
        # Processing the call keyword arguments
        kwargs_252475 = {}
        # Getting the type of 'call_assignment_252308' (line 96)
        call_assignment_252308_252472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'call_assignment_252308', False)
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___252473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), call_assignment_252308_252472, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_252476 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___252473, *[int_252474], **kwargs_252475)
        
        # Assigning a type to the variable 'call_assignment_252310' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'call_assignment_252310', getitem___call_result_252476)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'call_assignment_252310' (line 96)
        call_assignment_252310_252477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'call_assignment_252310')
        # Assigning a type to the variable 'b' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'b', call_assignment_252310_252477)
        
        # Assigning a Call to a Name (line 96):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_252480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'int')
        # Processing the call keyword arguments
        kwargs_252481 = {}
        # Getting the type of 'call_assignment_252308' (line 96)
        call_assignment_252308_252478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'call_assignment_252308', False)
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___252479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), call_assignment_252308_252478, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_252482 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___252479, *[int_252480], **kwargs_252481)
        
        # Assigning a type to the variable 'call_assignment_252311' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'call_assignment_252311', getitem___call_result_252482)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'call_assignment_252311' (line 96)
        call_assignment_252311_252483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'call_assignment_252311')
        # Assigning a type to the variable 'r' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'r', call_assignment_252311_252483)
        
        # Assigning a Call to a Name (line 96):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_252486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'int')
        # Processing the call keyword arguments
        kwargs_252487 = {}
        # Getting the type of 'call_assignment_252308' (line 96)
        call_assignment_252308_252484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'call_assignment_252308', False)
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___252485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), call_assignment_252308_252484, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_252488 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___252485, *[int_252486], **kwargs_252487)
        
        # Assigning a type to the variable 'call_assignment_252312' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'call_assignment_252312', getitem___call_result_252488)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'call_assignment_252312' (line 96)
        call_assignment_252312_252489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'call_assignment_252312')
        # Assigning a type to the variable 't' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 21), 't', call_assignment_252312_252489)
        
        # Assigning a BinOp to a Name (line 97):
        
        # Assigning a BinOp to a Name (line 97):
        # Getting the type of 'r' (line 97)
        r_252490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'r')
        # Getting the type of 'l' (line 97)
        l_252491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'l')
        # Applying the binary operator '-' (line 97)
        result_sub_252492 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 16), '-', r_252490, l_252491)
        
        # Assigning a type to the variable 'w' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'w', result_sub_252492)
        
        # Assigning a BinOp to a Name (line 98):
        
        # Assigning a BinOp to a Name (line 98):
        # Getting the type of 't' (line 98)
        t_252493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 't')
        # Getting the type of 'b' (line 98)
        b_252494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'b')
        # Applying the binary operator '-' (line 98)
        result_sub_252495 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 16), '-', t_252493, b_252494)
        
        # Assigning a type to the variable 'h' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'h', result_sub_252495)
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to copy_from_bbox(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'bbox' (line 99)
        bbox_252498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'bbox', False)
        # Processing the call keyword arguments (line 99)
        kwargs_252499 = {}
        # Getting the type of 'self' (line 99)
        self_252496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'self', False)
        # Obtaining the member 'copy_from_bbox' of a type (line 99)
        copy_from_bbox_252497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 18), self_252496, 'copy_from_bbox')
        # Calling copy_from_bbox(args, kwargs) (line 99)
        copy_from_bbox_call_result_252500 = invoke(stypy.reporting.localization.Localization(__file__, 99, 18), copy_from_bbox_252497, *[bbox_252498], **kwargs_252499)
        
        # Assigning a type to the variable 'reg' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'reg', copy_from_bbox_call_result_252500)
        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to to_string_argb(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_252503 = {}
        # Getting the type of 'reg' (line 100)
        reg_252501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'reg', False)
        # Obtaining the member 'to_string_argb' of a type (line 100)
        to_string_argb_252502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 18), reg_252501, 'to_string_argb')
        # Calling to_string_argb(args, kwargs) (line 100)
        to_string_argb_call_result_252504 = invoke(stypy.reporting.localization.Localization(__file__, 100, 18), to_string_argb_252502, *[], **kwargs_252503)
        
        # Assigning a type to the variable 'buf' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'buf', to_string_argb_call_result_252504)
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to QImage(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'buf' (line 101)
        buf_252507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 34), 'buf', False)
        # Getting the type of 'w' (line 101)
        w_252508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 39), 'w', False)
        # Getting the type of 'h' (line 101)
        h_252509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'h', False)
        # Getting the type of 'QtGui' (line 101)
        QtGui_252510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 45), 'QtGui', False)
        # Obtaining the member 'QImage' of a type (line 101)
        QImage_252511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 45), QtGui_252510, 'QImage')
        # Obtaining the member 'Format_ARGB32' of a type (line 101)
        Format_ARGB32_252512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 45), QImage_252511, 'Format_ARGB32')
        # Processing the call keyword arguments (line 101)
        kwargs_252513 = {}
        # Getting the type of 'QtGui' (line 101)
        QtGui_252505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'QtGui', False)
        # Obtaining the member 'QImage' of a type (line 101)
        QImage_252506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 21), QtGui_252505, 'QImage')
        # Calling QImage(args, kwargs) (line 101)
        QImage_call_result_252514 = invoke(stypy.reporting.localization.Localization(__file__, 101, 21), QImage_252506, *[buf_252507, w_252508, h_252509, Format_ARGB32_252512], **kwargs_252513)
        
        # Assigning a type to the variable 'qimage' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'qimage', QImage_call_result_252514)
        
        # Type idiom detected: calculating its left and rigth part (line 102)
        unicode_252515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 31), 'unicode', u'setDevicePixelRatio')
        # Getting the type of 'qimage' (line 102)
        qimage_252516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'qimage')
        
        (may_be_252517, more_types_in_union_252518) = may_provide_member(unicode_252515, qimage_252516)

        if may_be_252517:

            if more_types_in_union_252518:
                # Runtime conditional SSA (line 102)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'qimage' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'qimage', remove_not_member_provider_from_union(qimage_252516, u'setDevicePixelRatio'))
            
            # Call to setDevicePixelRatio(...): (line 104)
            # Processing the call arguments (line 104)
            # Getting the type of 'self' (line 104)
            self_252521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 43), 'self', False)
            # Obtaining the member '_dpi_ratio' of a type (line 104)
            _dpi_ratio_252522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 43), self_252521, '_dpi_ratio')
            # Processing the call keyword arguments (line 104)
            kwargs_252523 = {}
            # Getting the type of 'qimage' (line 104)
            qimage_252519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'qimage', False)
            # Obtaining the member 'setDevicePixelRatio' of a type (line 104)
            setDevicePixelRatio_252520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 16), qimage_252519, 'setDevicePixelRatio')
            # Calling setDevicePixelRatio(args, kwargs) (line 104)
            setDevicePixelRatio_call_result_252524 = invoke(stypy.reporting.localization.Localization(__file__, 104, 16), setDevicePixelRatio_252520, *[_dpi_ratio_252522], **kwargs_252523)
            

            if more_types_in_union_252518:
                # SSA join for if statement (line 102)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to QPoint(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'l' (line 105)
        l_252527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 35), 'l', False)
        # Getting the type of 'self' (line 105)
        self_252528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 38), 'self', False)
        # Obtaining the member 'renderer' of a type (line 105)
        renderer_252529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 38), self_252528, 'renderer')
        # Obtaining the member 'height' of a type (line 105)
        height_252530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 38), renderer_252529, 'height')
        # Getting the type of 't' (line 105)
        t_252531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 61), 't', False)
        # Applying the binary operator '-' (line 105)
        result_sub_252532 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 38), '-', height_252530, t_252531)
        
        # Processing the call keyword arguments (line 105)
        kwargs_252533 = {}
        # Getting the type of 'QtCore' (line 105)
        QtCore_252525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'QtCore', False)
        # Obtaining the member 'QPoint' of a type (line 105)
        QPoint_252526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 21), QtCore_252525, 'QPoint')
        # Calling QPoint(args, kwargs) (line 105)
        QPoint_call_result_252534 = invoke(stypy.reporting.localization.Localization(__file__, 105, 21), QPoint_252526, *[l_252527, result_sub_252532], **kwargs_252533)
        
        # Assigning a type to the variable 'origin' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'origin', QPoint_call_result_252534)
        
        # Call to drawImage(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'origin' (line 106)
        origin_252537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 30), 'origin', False)
        # Getting the type of 'self' (line 106)
        self_252538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'self', False)
        # Obtaining the member '_dpi_ratio' of a type (line 106)
        _dpi_ratio_252539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 39), self_252538, '_dpi_ratio')
        # Applying the binary operator 'div' (line 106)
        result_div_252540 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 30), 'div', origin_252537, _dpi_ratio_252539)
        
        # Getting the type of 'qimage' (line 106)
        qimage_252541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 56), 'qimage', False)
        # Processing the call keyword arguments (line 106)
        kwargs_252542 = {}
        # Getting the type of 'painter' (line 106)
        painter_252535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'painter', False)
        # Obtaining the member 'drawImage' of a type (line 106)
        drawImage_252536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), painter_252535, 'drawImage')
        # Calling drawImage(args, kwargs) (line 106)
        drawImage_call_result_252543 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), drawImage_252536, *[result_div_252540, qimage_252541], **kwargs_252542)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'QT_API' (line 109)
        QT_API_252544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'QT_API')
        unicode_252545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'unicode', u'PySide')
        # Applying the binary operator '==' (line 109)
        result_eq_252546 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 15), '==', QT_API_252544, unicode_252545)
        
        # Getting the type of 'six' (line 109)
        six_252547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 38), 'six')
        # Obtaining the member 'PY3' of a type (line 109)
        PY3_252548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 38), six_252547, 'PY3')
        # Applying the binary operator 'and' (line 109)
        result_and_keyword_252549 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 15), 'and', result_eq_252546, PY3_252548)
        
        # Testing the type of an if condition (line 109)
        if_condition_252550 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 12), result_and_keyword_252549)
        # Assigning a type to the variable 'if_condition_252550' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'if_condition_252550', if_condition_252550)
        # SSA begins for if statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 110):
        
        # Assigning a Num to a Attribute (line 110):
        int_252551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 60), 'int')
        
        # Call to from_address(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Call to id(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'buf' (line 110)
        buf_252556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 46), 'buf', False)
        # Processing the call keyword arguments (line 110)
        kwargs_252557 = {}
        # Getting the type of 'id' (line 110)
        id_252555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 43), 'id', False)
        # Calling id(args, kwargs) (line 110)
        id_call_result_252558 = invoke(stypy.reporting.localization.Localization(__file__, 110, 43), id_252555, *[buf_252556], **kwargs_252557)
        
        # Processing the call keyword arguments (line 110)
        kwargs_252559 = {}
        # Getting the type of 'ctypes' (line 110)
        ctypes_252552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'ctypes', False)
        # Obtaining the member 'c_long' of a type (line 110)
        c_long_252553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), ctypes_252552, 'c_long')
        # Obtaining the member 'from_address' of a type (line 110)
        from_address_252554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), c_long_252553, 'from_address')
        # Calling from_address(args, kwargs) (line 110)
        from_address_call_result_252560 = invoke(stypy.reporting.localization.Localization(__file__, 110, 16), from_address_252554, *[id_call_result_252558], **kwargs_252559)
        
        # Setting the type of the member 'value' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), from_address_call_result_252560, 'value', int_252551)
        # SSA join for if statement (line 109)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 113)
        self_252561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'self')
        # Obtaining the member '_drawRect' of a type (line 113)
        _drawRect_252562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 11), self_252561, '_drawRect')
        # Getting the type of 'None' (line 113)
        None_252563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'None')
        # Applying the binary operator 'isnot' (line 113)
        result_is_not_252564 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 11), 'isnot', _drawRect_252562, None_252563)
        
        # Testing the type of an if condition (line 113)
        if_condition_252565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 8), result_is_not_252564)
        # Assigning a type to the variable 'if_condition_252565' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'if_condition_252565', if_condition_252565)
        # SSA begins for if statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to QPen(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'QtCore' (line 114)
        QtCore_252568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 29), 'QtCore', False)
        # Obtaining the member 'Qt' of a type (line 114)
        Qt_252569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 29), QtCore_252568, 'Qt')
        # Obtaining the member 'black' of a type (line 114)
        black_252570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 29), Qt_252569, 'black')
        int_252571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 46), 'int')
        # Getting the type of 'self' (line 114)
        self_252572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 50), 'self', False)
        # Obtaining the member '_dpi_ratio' of a type (line 114)
        _dpi_ratio_252573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 50), self_252572, '_dpi_ratio')
        # Applying the binary operator 'div' (line 114)
        result_div_252574 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 46), 'div', int_252571, _dpi_ratio_252573)
        
        # Getting the type of 'QtCore' (line 115)
        QtCore_252575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'QtCore', False)
        # Obtaining the member 'Qt' of a type (line 115)
        Qt_252576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 29), QtCore_252575, 'Qt')
        # Obtaining the member 'DotLine' of a type (line 115)
        DotLine_252577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 29), Qt_252576, 'DotLine')
        # Processing the call keyword arguments (line 114)
        kwargs_252578 = {}
        # Getting the type of 'QtGui' (line 114)
        QtGui_252566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'QtGui', False)
        # Obtaining the member 'QPen' of a type (line 114)
        QPen_252567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 18), QtGui_252566, 'QPen')
        # Calling QPen(args, kwargs) (line 114)
        QPen_call_result_252579 = invoke(stypy.reporting.localization.Localization(__file__, 114, 18), QPen_252567, *[black_252570, result_div_252574, DotLine_252577], **kwargs_252578)
        
        # Assigning a type to the variable 'pen' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'pen', QPen_call_result_252579)
        
        # Call to setPen(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'pen' (line 116)
        pen_252582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 27), 'pen', False)
        # Processing the call keyword arguments (line 116)
        kwargs_252583 = {}
        # Getting the type of 'painter' (line 116)
        painter_252580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'painter', False)
        # Obtaining the member 'setPen' of a type (line 116)
        setPen_252581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), painter_252580, 'setPen')
        # Calling setPen(args, kwargs) (line 116)
        setPen_call_result_252584 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), setPen_252581, *[pen_252582], **kwargs_252583)
        
        
        # Assigning a Attribute to a Tuple (line 117):
        
        # Assigning a Subscript to a Name (line 117):
        
        # Obtaining the type of the subscript
        int_252585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 12), 'int')
        # Getting the type of 'self' (line 117)
        self_252586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'self')
        # Obtaining the member '_drawRect' of a type (line 117)
        _drawRect_252587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 25), self_252586, '_drawRect')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___252588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), _drawRect_252587, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_252589 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), getitem___252588, int_252585)
        
        # Assigning a type to the variable 'tuple_var_assignment_252313' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'tuple_var_assignment_252313', subscript_call_result_252589)
        
        # Assigning a Subscript to a Name (line 117):
        
        # Obtaining the type of the subscript
        int_252590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 12), 'int')
        # Getting the type of 'self' (line 117)
        self_252591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'self')
        # Obtaining the member '_drawRect' of a type (line 117)
        _drawRect_252592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 25), self_252591, '_drawRect')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___252593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), _drawRect_252592, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_252594 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), getitem___252593, int_252590)
        
        # Assigning a type to the variable 'tuple_var_assignment_252314' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'tuple_var_assignment_252314', subscript_call_result_252594)
        
        # Assigning a Subscript to a Name (line 117):
        
        # Obtaining the type of the subscript
        int_252595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 12), 'int')
        # Getting the type of 'self' (line 117)
        self_252596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'self')
        # Obtaining the member '_drawRect' of a type (line 117)
        _drawRect_252597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 25), self_252596, '_drawRect')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___252598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), _drawRect_252597, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_252599 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), getitem___252598, int_252595)
        
        # Assigning a type to the variable 'tuple_var_assignment_252315' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'tuple_var_assignment_252315', subscript_call_result_252599)
        
        # Assigning a Subscript to a Name (line 117):
        
        # Obtaining the type of the subscript
        int_252600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 12), 'int')
        # Getting the type of 'self' (line 117)
        self_252601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'self')
        # Obtaining the member '_drawRect' of a type (line 117)
        _drawRect_252602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 25), self_252601, '_drawRect')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___252603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), _drawRect_252602, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_252604 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), getitem___252603, int_252600)
        
        # Assigning a type to the variable 'tuple_var_assignment_252316' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'tuple_var_assignment_252316', subscript_call_result_252604)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'tuple_var_assignment_252313' (line 117)
        tuple_var_assignment_252313_252605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'tuple_var_assignment_252313')
        # Assigning a type to the variable 'x' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'x', tuple_var_assignment_252313_252605)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'tuple_var_assignment_252314' (line 117)
        tuple_var_assignment_252314_252606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'tuple_var_assignment_252314')
        # Assigning a type to the variable 'y' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'y', tuple_var_assignment_252314_252606)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'tuple_var_assignment_252315' (line 117)
        tuple_var_assignment_252315_252607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'tuple_var_assignment_252315')
        # Assigning a type to the variable 'w' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'w', tuple_var_assignment_252315_252607)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'tuple_var_assignment_252316' (line 117)
        tuple_var_assignment_252316_252608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'tuple_var_assignment_252316')
        # Assigning a type to the variable 'h' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'h', tuple_var_assignment_252316_252608)
        
        # Call to drawRect(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'x' (line 118)
        x_252611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 29), 'x', False)
        # Getting the type of 'y' (line 118)
        y_252612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 32), 'y', False)
        # Getting the type of 'w' (line 118)
        w_252613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 35), 'w', False)
        # Getting the type of 'h' (line 118)
        h_252614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 38), 'h', False)
        # Processing the call keyword arguments (line 118)
        kwargs_252615 = {}
        # Getting the type of 'painter' (line 118)
        painter_252609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'painter', False)
        # Obtaining the member 'drawRect' of a type (line 118)
        drawRect_252610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), painter_252609, 'drawRect')
        # Calling drawRect(args, kwargs) (line 118)
        drawRect_call_result_252616 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), drawRect_252610, *[x_252611, y_252612, w_252613, h_252614], **kwargs_252615)
        
        # SSA join for if statement (line 113)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to end(...): (line 120)
        # Processing the call keyword arguments (line 120)
        kwargs_252619 = {}
        # Getting the type of 'painter' (line 120)
        painter_252617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'painter', False)
        # Obtaining the member 'end' of a type (line 120)
        end_252618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), painter_252617, 'end')
        # Calling end(args, kwargs) (line 120)
        end_call_result_252620 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), end_252618, *[], **kwargs_252619)
        
        
        # ################# End of 'paintEvent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'paintEvent' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_252621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252621)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'paintEvent'
        return stypy_return_type_252621


    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQTAggBase.draw.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQTAggBase.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQTAggBase.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQTAggBase.draw.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQTAggBase.draw')
        FigureCanvasQTAggBase.draw.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasQTAggBase.draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQTAggBase.draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQTAggBase.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQTAggBase.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQTAggBase.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQTAggBase.draw.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQTAggBase.draw', [], None, None, defaults, varargs, kwargs)

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

        unicode_252622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, (-1)), 'unicode', u'Draw the figure with Agg, and queue a request for a Qt draw.\n        ')
        
        # Call to draw(...): (line 127)
        # Processing the call keyword arguments (line 127)
        kwargs_252629 = {}
        
        # Call to super(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'FigureCanvasQTAggBase' (line 127)
        FigureCanvasQTAggBase_252624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 14), 'FigureCanvasQTAggBase', False)
        # Getting the type of 'self' (line 127)
        self_252625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 37), 'self', False)
        # Processing the call keyword arguments (line 127)
        kwargs_252626 = {}
        # Getting the type of 'super' (line 127)
        super_252623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'super', False)
        # Calling super(args, kwargs) (line 127)
        super_call_result_252627 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), super_252623, *[FigureCanvasQTAggBase_252624, self_252625], **kwargs_252626)
        
        # Obtaining the member 'draw' of a type (line 127)
        draw_252628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), super_call_result_252627, 'draw')
        # Calling draw(args, kwargs) (line 127)
        draw_call_result_252630 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), draw_252628, *[], **kwargs_252629)
        
        
        # Call to update(...): (line 128)
        # Processing the call keyword arguments (line 128)
        kwargs_252633 = {}
        # Getting the type of 'self' (line 128)
        self_252631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self', False)
        # Obtaining the member 'update' of a type (line 128)
        update_252632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_252631, 'update')
        # Calling update(args, kwargs) (line 128)
        update_call_result_252634 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), update_252632, *[], **kwargs_252633)
        
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_252635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252635)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_252635


    @norecursion
    def draw_idle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_idle'
        module_type_store = module_type_store.open_function_context('draw_idle', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQTAggBase.draw_idle.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQTAggBase.draw_idle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQTAggBase.draw_idle.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQTAggBase.draw_idle.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQTAggBase.draw_idle')
        FigureCanvasQTAggBase.draw_idle.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasQTAggBase.draw_idle.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQTAggBase.draw_idle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQTAggBase.draw_idle.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQTAggBase.draw_idle.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQTAggBase.draw_idle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQTAggBase.draw_idle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQTAggBase.draw_idle', [], None, None, defaults, varargs, kwargs)

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

        unicode_252636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'unicode', u'Queue redraw of the Agg buffer and request Qt paintEvent.\n        ')
        
        
        # Getting the type of 'self' (line 138)
        self_252637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'self')
        # Obtaining the member '_agg_draw_pending' of a type (line 138)
        _agg_draw_pending_252638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), self_252637, '_agg_draw_pending')
        # Applying the 'not' unary operator (line 138)
        result_not__252639 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 11), 'not', _agg_draw_pending_252638)
        
        # Testing the type of an if condition (line 138)
        if_condition_252640 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 8), result_not__252639)
        # Assigning a type to the variable 'if_condition_252640' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'if_condition_252640', if_condition_252640)
        # SSA begins for if statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 139):
        
        # Assigning a Name to a Attribute (line 139):
        # Getting the type of 'True' (line 139)
        True_252641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 37), 'True')
        # Getting the type of 'self' (line 139)
        self_252642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'self')
        # Setting the type of the member '_agg_draw_pending' of a type (line 139)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), self_252642, '_agg_draw_pending', True_252641)
        
        # Call to singleShot(...): (line 140)
        # Processing the call arguments (line 140)
        int_252646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 37), 'int')
        # Getting the type of 'self' (line 140)
        self_252647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 40), 'self', False)
        # Obtaining the member '__draw_idle_agg' of a type (line 140)
        draw_idle_agg_252648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 40), self_252647, '__draw_idle_agg')
        # Processing the call keyword arguments (line 140)
        kwargs_252649 = {}
        # Getting the type of 'QtCore' (line 140)
        QtCore_252643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'QtCore', False)
        # Obtaining the member 'QTimer' of a type (line 140)
        QTimer_252644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), QtCore_252643, 'QTimer')
        # Obtaining the member 'singleShot' of a type (line 140)
        singleShot_252645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), QTimer_252644, 'singleShot')
        # Calling singleShot(args, kwargs) (line 140)
        singleShot_call_result_252650 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), singleShot_252645, *[int_252646, draw_idle_agg_252648], **kwargs_252649)
        
        # SSA join for if statement (line 138)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'draw_idle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_idle' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_252651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252651)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_idle'
        return stypy_return_type_252651


    @norecursion
    def __draw_idle_agg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__draw_idle_agg'
        module_type_store = module_type_store.open_function_context('__draw_idle_agg', 142, 4, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQTAggBase.__draw_idle_agg.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQTAggBase.__draw_idle_agg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQTAggBase.__draw_idle_agg.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQTAggBase.__draw_idle_agg.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQTAggBase.__draw_idle_agg')
        FigureCanvasQTAggBase.__draw_idle_agg.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasQTAggBase.__draw_idle_agg.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasQTAggBase.__draw_idle_agg.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQTAggBase.__draw_idle_agg.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQTAggBase.__draw_idle_agg.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQTAggBase.__draw_idle_agg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQTAggBase.__draw_idle_agg.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQTAggBase.__draw_idle_agg', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__draw_idle_agg', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__draw_idle_agg(...)' code ##################

        
        
        # Getting the type of 'self' (line 143)
        self_252652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'self')
        # Obtaining the member '_agg_draw_pending' of a type (line 143)
        _agg_draw_pending_252653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 15), self_252652, '_agg_draw_pending')
        # Applying the 'not' unary operator (line 143)
        result_not__252654 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 11), 'not', _agg_draw_pending_252653)
        
        # Testing the type of an if condition (line 143)
        if_condition_252655 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 8), result_not__252654)
        # Assigning a type to the variable 'if_condition_252655' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'if_condition_252655', if_condition_252655)
        # SSA begins for if statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        
        # Call to height(...): (line 145)
        # Processing the call keyword arguments (line 145)
        kwargs_252658 = {}
        # Getting the type of 'self' (line 145)
        self_252656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'self', False)
        # Obtaining the member 'height' of a type (line 145)
        height_252657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 11), self_252656, 'height')
        # Calling height(args, kwargs) (line 145)
        height_call_result_252659 = invoke(stypy.reporting.localization.Localization(__file__, 145, 11), height_252657, *[], **kwargs_252658)
        
        int_252660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 27), 'int')
        # Applying the binary operator '<' (line 145)
        result_lt_252661 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), '<', height_call_result_252659, int_252660)
        
        
        
        # Call to width(...): (line 145)
        # Processing the call keyword arguments (line 145)
        kwargs_252664 = {}
        # Getting the type of 'self' (line 145)
        self_252662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 32), 'self', False)
        # Obtaining the member 'width' of a type (line 145)
        width_252663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 32), self_252662, 'width')
        # Calling width(args, kwargs) (line 145)
        width_call_result_252665 = invoke(stypy.reporting.localization.Localization(__file__, 145, 32), width_252663, *[], **kwargs_252664)
        
        int_252666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 47), 'int')
        # Applying the binary operator '<' (line 145)
        result_lt_252667 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 32), '<', width_call_result_252665, int_252666)
        
        # Applying the binary operator 'or' (line 145)
        result_or_keyword_252668 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), 'or', result_lt_252661, result_lt_252667)
        
        # Testing the type of an if condition (line 145)
        if_condition_252669 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 8), result_or_keyword_252668)
        # Assigning a type to the variable 'if_condition_252669' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'if_condition_252669', if_condition_252669)
        # SSA begins for if statement (line 145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 146):
        
        # Assigning a Name to a Attribute (line 146):
        # Getting the type of 'False' (line 146)
        False_252670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 37), 'False')
        # Getting the type of 'self' (line 146)
        self_252671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'self')
        # Setting the type of the member '_agg_draw_pending' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), self_252671, '_agg_draw_pending', False_252670)
        # Assigning a type to the variable 'stypy_return_type' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 145)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Try-finally block (line 148)
        
        
        # SSA begins for try-except statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to draw(...): (line 149)
        # Processing the call keyword arguments (line 149)
        kwargs_252674 = {}
        # Getting the type of 'self' (line 149)
        self_252672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'self', False)
        # Obtaining the member 'draw' of a type (line 149)
        draw_252673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), self_252672, 'draw')
        # Calling draw(args, kwargs) (line 149)
        draw_call_result_252675 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), draw_252673, *[], **kwargs_252674)
        
        # SSA branch for the except part of a try statement (line 148)
        # SSA branch for the except 'Exception' branch of a try statement (line 148)
        module_type_store.open_ssa_branch('except')
        
        # Call to print_exc(...): (line 152)
        # Processing the call keyword arguments (line 152)
        kwargs_252678 = {}
        # Getting the type of 'traceback' (line 152)
        traceback_252676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'traceback', False)
        # Obtaining the member 'print_exc' of a type (line 152)
        print_exc_252677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), traceback_252676, 'print_exc')
        # Calling print_exc(args, kwargs) (line 152)
        print_exc_call_result_252679 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), print_exc_252677, *[], **kwargs_252678)
        
        # SSA join for try-except statement (line 148)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # finally branch of the try-finally block (line 148)
        
        # Assigning a Name to a Attribute (line 154):
        
        # Assigning a Name to a Attribute (line 154):
        # Getting the type of 'False' (line 154)
        False_252680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 37), 'False')
        # Getting the type of 'self' (line 154)
        self_252681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'self')
        # Setting the type of the member '_agg_draw_pending' of a type (line 154)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), self_252681, '_agg_draw_pending', False_252680)
        
        
        # ################# End of '__draw_idle_agg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__draw_idle_agg' in the type store
        # Getting the type of 'stypy_return_type' (line 142)
        stypy_return_type_252682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252682)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__draw_idle_agg'
        return stypy_return_type_252682


    @norecursion
    def blit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 156)
        None_252683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 24), 'None')
        defaults = [None_252683]
        # Create a new context for function 'blit'
        module_type_store = module_type_store.open_function_context('blit', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQTAggBase.blit.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQTAggBase.blit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQTAggBase.blit.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQTAggBase.blit.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQTAggBase.blit')
        FigureCanvasQTAggBase.blit.__dict__.__setitem__('stypy_param_names_list', ['bbox'])
        FigureCanvasQTAggBase.blit.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQTAggBase.blit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQTAggBase.blit.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQTAggBase.blit.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQTAggBase.blit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQTAggBase.blit.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQTAggBase.blit', ['bbox'], None, None, defaults, varargs, kwargs)

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

        unicode_252684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, (-1)), 'unicode', u'Blit the region in bbox.\n        ')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'bbox' (line 161)
        bbox_252685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'bbox')
        # Getting the type of 'None' (line 161)
        None_252686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'None')
        # Applying the binary operator 'is' (line 161)
        result_is__252687 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 11), 'is', bbox_252685, None_252686)
        
        # Getting the type of 'self' (line 161)
        self_252688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'self')
        # Obtaining the member 'figure' of a type (line 161)
        figure_252689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 28), self_252688, 'figure')
        # Applying the binary operator 'and' (line 161)
        result_and_keyword_252690 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 11), 'and', result_is__252687, figure_252689)
        
        # Testing the type of an if condition (line 161)
        if_condition_252691 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 8), result_and_keyword_252690)
        # Assigning a type to the variable 'if_condition_252691' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'if_condition_252691', if_condition_252691)
        # SSA begins for if statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 162):
        
        # Assigning a Attribute to a Name (line 162):
        # Getting the type of 'self' (line 162)
        self_252692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'self')
        # Obtaining the member 'figure' of a type (line 162)
        figure_252693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 19), self_252692, 'figure')
        # Obtaining the member 'bbox' of a type (line 162)
        bbox_252694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 19), figure_252693, 'bbox')
        # Assigning a type to the variable 'bbox' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'bbox', bbox_252694)
        # SSA join for if statement (line 161)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'bbox' (line 164)
        bbox_252698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 32), 'bbox', False)
        # Processing the call keyword arguments (line 164)
        kwargs_252699 = {}
        # Getting the type of 'self' (line 164)
        self_252695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'self', False)
        # Obtaining the member '_bbox_queue' of a type (line 164)
        _bbox_queue_252696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), self_252695, '_bbox_queue')
        # Obtaining the member 'append' of a type (line 164)
        append_252697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), _bbox_queue_252696, 'append')
        # Calling append(args, kwargs) (line 164)
        append_call_result_252700 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), append_252697, *[bbox_252698], **kwargs_252699)
        
        
        # Assigning a ListComp to a Tuple (line 167):
        
        # Assigning a Subscript to a Name (line 167):
        
        # Obtaining the type of the subscript
        int_252701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 8), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'bbox' (line 167)
        bbox_252706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 53), 'bbox')
        # Obtaining the member 'bounds' of a type (line 167)
        bounds_252707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 53), bbox_252706, 'bounds')
        comprehension_252708 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 22), bounds_252707)
        # Assigning a type to the variable 'pt' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'pt', comprehension_252708)
        # Getting the type of 'pt' (line 167)
        pt_252702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'pt')
        # Getting the type of 'self' (line 167)
        self_252703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'self')
        # Obtaining the member '_dpi_ratio' of a type (line 167)
        _dpi_ratio_252704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), self_252703, '_dpi_ratio')
        # Applying the binary operator 'div' (line 167)
        result_div_252705 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 22), 'div', pt_252702, _dpi_ratio_252704)
        
        list_252709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 22), list_252709, result_div_252705)
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___252710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), list_252709, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_252711 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), getitem___252710, int_252701)
        
        # Assigning a type to the variable 'tuple_var_assignment_252317' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_252317', subscript_call_result_252711)
        
        # Assigning a Subscript to a Name (line 167):
        
        # Obtaining the type of the subscript
        int_252712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 8), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'bbox' (line 167)
        bbox_252717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 53), 'bbox')
        # Obtaining the member 'bounds' of a type (line 167)
        bounds_252718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 53), bbox_252717, 'bounds')
        comprehension_252719 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 22), bounds_252718)
        # Assigning a type to the variable 'pt' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'pt', comprehension_252719)
        # Getting the type of 'pt' (line 167)
        pt_252713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'pt')
        # Getting the type of 'self' (line 167)
        self_252714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'self')
        # Obtaining the member '_dpi_ratio' of a type (line 167)
        _dpi_ratio_252715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), self_252714, '_dpi_ratio')
        # Applying the binary operator 'div' (line 167)
        result_div_252716 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 22), 'div', pt_252713, _dpi_ratio_252715)
        
        list_252720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 22), list_252720, result_div_252716)
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___252721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), list_252720, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_252722 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), getitem___252721, int_252712)
        
        # Assigning a type to the variable 'tuple_var_assignment_252318' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_252318', subscript_call_result_252722)
        
        # Assigning a Subscript to a Name (line 167):
        
        # Obtaining the type of the subscript
        int_252723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 8), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'bbox' (line 167)
        bbox_252728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 53), 'bbox')
        # Obtaining the member 'bounds' of a type (line 167)
        bounds_252729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 53), bbox_252728, 'bounds')
        comprehension_252730 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 22), bounds_252729)
        # Assigning a type to the variable 'pt' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'pt', comprehension_252730)
        # Getting the type of 'pt' (line 167)
        pt_252724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'pt')
        # Getting the type of 'self' (line 167)
        self_252725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'self')
        # Obtaining the member '_dpi_ratio' of a type (line 167)
        _dpi_ratio_252726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), self_252725, '_dpi_ratio')
        # Applying the binary operator 'div' (line 167)
        result_div_252727 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 22), 'div', pt_252724, _dpi_ratio_252726)
        
        list_252731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 22), list_252731, result_div_252727)
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___252732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), list_252731, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_252733 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), getitem___252732, int_252723)
        
        # Assigning a type to the variable 'tuple_var_assignment_252319' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_252319', subscript_call_result_252733)
        
        # Assigning a Subscript to a Name (line 167):
        
        # Obtaining the type of the subscript
        int_252734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 8), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'bbox' (line 167)
        bbox_252739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 53), 'bbox')
        # Obtaining the member 'bounds' of a type (line 167)
        bounds_252740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 53), bbox_252739, 'bounds')
        comprehension_252741 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 22), bounds_252740)
        # Assigning a type to the variable 'pt' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'pt', comprehension_252741)
        # Getting the type of 'pt' (line 167)
        pt_252735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'pt')
        # Getting the type of 'self' (line 167)
        self_252736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'self')
        # Obtaining the member '_dpi_ratio' of a type (line 167)
        _dpi_ratio_252737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), self_252736, '_dpi_ratio')
        # Applying the binary operator 'div' (line 167)
        result_div_252738 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 22), 'div', pt_252735, _dpi_ratio_252737)
        
        list_252742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 22), list_252742, result_div_252738)
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___252743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), list_252742, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_252744 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), getitem___252743, int_252734)
        
        # Assigning a type to the variable 'tuple_var_assignment_252320' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_252320', subscript_call_result_252744)
        
        # Assigning a Name to a Name (line 167):
        # Getting the type of 'tuple_var_assignment_252317' (line 167)
        tuple_var_assignment_252317_252745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_252317')
        # Assigning a type to the variable 'l' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'l', tuple_var_assignment_252317_252745)
        
        # Assigning a Name to a Name (line 167):
        # Getting the type of 'tuple_var_assignment_252318' (line 167)
        tuple_var_assignment_252318_252746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_252318')
        # Assigning a type to the variable 'b' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'b', tuple_var_assignment_252318_252746)
        
        # Assigning a Name to a Name (line 167):
        # Getting the type of 'tuple_var_assignment_252319' (line 167)
        tuple_var_assignment_252319_252747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_252319')
        # Assigning a type to the variable 'w' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 14), 'w', tuple_var_assignment_252319_252747)
        
        # Assigning a Name to a Name (line 167):
        # Getting the type of 'tuple_var_assignment_252320' (line 167)
        tuple_var_assignment_252320_252748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_252320')
        # Assigning a type to the variable 'h' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 17), 'h', tuple_var_assignment_252320_252748)
        
        # Assigning a BinOp to a Name (line 168):
        
        # Assigning a BinOp to a Name (line 168):
        # Getting the type of 'b' (line 168)
        b_252749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'b')
        # Getting the type of 'h' (line 168)
        h_252750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'h')
        # Applying the binary operator '+' (line 168)
        result_add_252751 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 12), '+', b_252749, h_252750)
        
        # Assigning a type to the variable 't' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 't', result_add_252751)
        
        # Call to repaint(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'l' (line 169)
        l_252754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'l', False)
        # Getting the type of 'self' (line 169)
        self_252755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'self', False)
        # Obtaining the member 'renderer' of a type (line 169)
        renderer_252756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 24), self_252755, 'renderer')
        # Obtaining the member 'height' of a type (line 169)
        height_252757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 24), renderer_252756, 'height')
        # Getting the type of 'self' (line 169)
        self_252758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 47), 'self', False)
        # Obtaining the member '_dpi_ratio' of a type (line 169)
        _dpi_ratio_252759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 47), self_252758, '_dpi_ratio')
        # Applying the binary operator 'div' (line 169)
        result_div_252760 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 24), 'div', height_252757, _dpi_ratio_252759)
        
        # Getting the type of 't' (line 169)
        t_252761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 65), 't', False)
        # Applying the binary operator '-' (line 169)
        result_sub_252762 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 24), '-', result_div_252760, t_252761)
        
        # Getting the type of 'w' (line 169)
        w_252763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 68), 'w', False)
        # Getting the type of 'h' (line 169)
        h_252764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 71), 'h', False)
        # Processing the call keyword arguments (line 169)
        kwargs_252765 = {}
        # Getting the type of 'self' (line 169)
        self_252752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'self', False)
        # Obtaining the member 'repaint' of a type (line 169)
        repaint_252753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), self_252752, 'repaint')
        # Calling repaint(args, kwargs) (line 169)
        repaint_call_result_252766 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), repaint_252753, *[l_252754, result_sub_252762, w_252763, h_252764], **kwargs_252765)
        
        
        # ################# End of 'blit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'blit' in the type store
        # Getting the type of 'stypy_return_type' (line 156)
        stypy_return_type_252767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252767)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'blit'
        return stypy_return_type_252767


    @norecursion
    def print_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_figure'
        module_type_store = module_type_store.open_function_context('print_figure', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQTAggBase.print_figure.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQTAggBase.print_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQTAggBase.print_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQTAggBase.print_figure.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQTAggBase.print_figure')
        FigureCanvasQTAggBase.print_figure.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasQTAggBase.print_figure.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasQTAggBase.print_figure.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasQTAggBase.print_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQTAggBase.print_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQTAggBase.print_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQTAggBase.print_figure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQTAggBase.print_figure', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_figure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_figure(...)' code ##################

        
        # Call to print_figure(...): (line 172)
        # Getting the type of 'args' (line 172)
        args_252774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 57), 'args', False)
        # Processing the call keyword arguments (line 172)
        # Getting the type of 'kwargs' (line 172)
        kwargs_252775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 65), 'kwargs', False)
        kwargs_252776 = {'kwargs_252775': kwargs_252775}
        
        # Call to super(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'FigureCanvasQTAggBase' (line 172)
        FigureCanvasQTAggBase_252769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 14), 'FigureCanvasQTAggBase', False)
        # Getting the type of 'self' (line 172)
        self_252770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 37), 'self', False)
        # Processing the call keyword arguments (line 172)
        kwargs_252771 = {}
        # Getting the type of 'super' (line 172)
        super_252768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'super', False)
        # Calling super(args, kwargs) (line 172)
        super_call_result_252772 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), super_252768, *[FigureCanvasQTAggBase_252769, self_252770], **kwargs_252771)
        
        # Obtaining the member 'print_figure' of a type (line 172)
        print_figure_252773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), super_call_result_252772, 'print_figure')
        # Calling print_figure(args, kwargs) (line 172)
        print_figure_call_result_252777 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), print_figure_252773, *[args_252774], **kwargs_252776)
        
        
        # Call to draw(...): (line 173)
        # Processing the call keyword arguments (line 173)
        kwargs_252780 = {}
        # Getting the type of 'self' (line 173)
        self_252778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self', False)
        # Obtaining the member 'draw' of a type (line 173)
        draw_252779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_252778, 'draw')
        # Calling draw(args, kwargs) (line 173)
        draw_call_result_252781 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), draw_252779, *[], **kwargs_252780)
        
        
        # ################# End of 'print_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_252782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252782)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_figure'
        return stypy_return_type_252782


# Assigning a type to the variable 'FigureCanvasQTAggBase' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'FigureCanvasQTAggBase', FigureCanvasQTAggBase)
# Declaration of the 'FigureCanvasQTAgg' class
# Getting the type of 'FigureCanvasQTAggBase' (line 176)
FigureCanvasQTAggBase_252783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'FigureCanvasQTAggBase')
# Getting the type of 'FigureCanvasQT' (line 176)
FigureCanvasQT_252784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 47), 'FigureCanvasQT')

class FigureCanvasQTAgg(FigureCanvasQTAggBase_252783, FigureCanvasQT_252784, ):
    unicode_252785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, (-1)), 'unicode', u'\n    The canvas the figure renders into.  Calls the draw and print fig\n    methods, creates the renderers, etc.\n\n    Modified to import from Qt5 backend for new-style mouse events.\n\n    Attributes\n    ----------\n    figure : `matplotlib.figure.Figure`\n        A high-level Figure instance\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 176, 0, False)
        # Assigning a type to the variable 'self' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQTAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureCanvasQTAgg' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'FigureCanvasQTAgg', FigureCanvasQTAgg)
# Declaration of the '_BackendQT5Agg' class
# Getting the type of '_BackendQT5' (line 192)
_BackendQT5_252786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 21), '_BackendQT5')

class _BackendQT5Agg(_BackendQT5_252786, ):
    
    # Assigning a Name to a Name (line 193):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 191, 0, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendQT5Agg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendQT5Agg' (line 191)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), '_BackendQT5Agg', _BackendQT5Agg)

# Assigning a Name to a Name (line 193):
# Getting the type of 'FigureCanvasQTAgg' (line 193)
FigureCanvasQTAgg_252787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), 'FigureCanvasQTAgg')
# Getting the type of '_BackendQT5Agg'
_BackendQT5Agg_252788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendQT5Agg')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendQT5Agg_252788, 'FigureCanvas', FigureCanvasQTAgg_252787)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
