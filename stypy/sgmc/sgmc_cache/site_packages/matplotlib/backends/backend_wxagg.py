
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import matplotlib
7: from matplotlib.figure import Figure
8: 
9: from .backend_agg import FigureCanvasAgg
10: 
11: from . import wx_compat as wxc
12: from . import backend_wx
13: from .backend_wx import (_BackendWx, FigureManagerWx, FigureCanvasWx,
14:     FigureFrameWx, DEBUG_MSG, NavigationToolbar2Wx, Toolbar)
15: 
16: import wx
17: 
18: 
19: class FigureFrameWxAgg(FigureFrameWx):
20:     def get_canvas(self, fig):
21:         return FigureCanvasWxAgg(self, -1, fig)
22: 
23:     def _get_toolbar(self, statbar):
24:         if matplotlib.rcParams['toolbar'] == 'toolbar2':
25:             toolbar = NavigationToolbar2WxAgg(self.canvas)
26:             toolbar.set_status_bar(statbar)
27:         else:
28:             toolbar = None
29:         return toolbar
30: 
31: 
32: class FigureCanvasWxAgg(FigureCanvasAgg, FigureCanvasWx):
33:     '''
34:     The FigureCanvas contains the figure and does event handling.
35: 
36:     In the wxPython backend, it is derived from wxPanel, and (usually)
37:     lives inside a frame instantiated by a FigureManagerWx. The parent
38:     window probably implements a wxSizer to control the displayed
39:     control size - but we give a hint as to our preferred minimum
40:     size.
41:     '''
42: 
43:     def draw(self, drawDC=None):
44:         '''
45:         Render the figure using agg.
46:         '''
47:         DEBUG_MSG("draw()", 1, self)
48:         FigureCanvasAgg.draw(self)
49: 
50:         self.bitmap = _convert_agg_to_wx_bitmap(self.get_renderer(), None)
51:         self._isDrawn = True
52:         self.gui_repaint(drawDC=drawDC, origin='WXAgg')
53: 
54:     def blit(self, bbox=None):
55:         '''
56:         Transfer the region of the agg buffer defined by bbox to the display.
57:         If bbox is None, the entire buffer is transferred.
58:         '''
59:         if bbox is None:
60:             self.bitmap = _convert_agg_to_wx_bitmap(self.get_renderer(), None)
61:             self.gui_repaint()
62:             return
63: 
64:         l, b, w, h = bbox.bounds
65:         r = l + w
66:         t = b + h
67:         x = int(l)
68:         y = int(self.bitmap.GetHeight() - t)
69: 
70:         srcBmp = _convert_agg_to_wx_bitmap(self.get_renderer(), None)
71:         srcDC = wx.MemoryDC()
72:         srcDC.SelectObject(srcBmp)
73: 
74:         destDC = wx.MemoryDC()
75:         destDC.SelectObject(self.bitmap)
76: 
77:         destDC.Blit(x, y, int(w), int(h), srcDC, x, y)
78: 
79:         destDC.SelectObject(wx.NullBitmap)
80:         srcDC.SelectObject(wx.NullBitmap)
81:         self.gui_repaint()
82: 
83:     filetypes = FigureCanvasAgg.filetypes
84: 
85:     def print_figure(self, filename, *args, **kwargs):
86:         # Use pure Agg renderer to draw
87:         FigureCanvasAgg.print_figure(self, filename, *args, **kwargs)
88:         # Restore the current view; this is needed because the
89:         # artist contains methods rely on particular attributes
90:         # of the rendered figure for determining things like
91:         # bounding boxes.
92:         if self._isDrawn:
93:             self.draw()
94: 
95: 
96: class NavigationToolbar2WxAgg(NavigationToolbar2Wx):
97:     def get_canvas(self, frame, fig):
98:         return FigureCanvasWxAgg(frame, -1, fig)
99: 
100: 
101: # agg/wxPython image conversion functions (wxPython >= 2.8)
102: 
103: 
104: def _convert_agg_to_wx_image(agg, bbox):
105:     '''
106:     Convert the region of the agg buffer bounded by bbox to a wx.Image.  If
107:     bbox is None, the entire buffer is converted.
108: 
109:     Note: agg must be a backend_agg.RendererAgg instance.
110:     '''
111:     if bbox is None:
112:         # agg => rgb -> image
113:         image = wxc.EmptyImage(int(agg.width), int(agg.height))
114:         image.SetData(agg.tostring_rgb())
115:         return image
116:     else:
117:         # agg => rgba buffer -> bitmap => clipped bitmap => image
118:         return wx.ImageFromBitmap(_WX28_clipped_agg_as_bitmap(agg, bbox))
119: 
120: 
121: def _convert_agg_to_wx_bitmap(agg, bbox):
122:     '''
123:     Convert the region of the agg buffer bounded by bbox to a wx.Bitmap.  If
124:     bbox is None, the entire buffer is converted.
125: 
126:     Note: agg must be a backend_agg.RendererAgg instance.
127:     '''
128:     if bbox is None:
129:         # agg => rgba buffer -> bitmap
130:         return wxc.BitmapFromBuffer(int(agg.width), int(agg.height),
131:                                     agg.buffer_rgba())
132:     else:
133:         # agg => rgba buffer -> bitmap => clipped bitmap
134:         return _WX28_clipped_agg_as_bitmap(agg, bbox)
135: 
136: 
137: def _WX28_clipped_agg_as_bitmap(agg, bbox):
138:     '''
139:     Convert the region of a the agg buffer bounded by bbox to a wx.Bitmap.
140: 
141:     Note: agg must be a backend_agg.RendererAgg instance.
142:     '''
143:     l, b, width, height = bbox.bounds
144:     r = l + width
145:     t = b + height
146: 
147:     srcBmp = wxc.BitmapFromBuffer(int(agg.width), int(agg.height),
148:                                   agg.buffer_rgba())
149:     srcDC = wx.MemoryDC()
150:     srcDC.SelectObject(srcBmp)
151: 
152:     destBmp = wxc.EmptyBitmap(int(width), int(height))
153:     destDC = wx.MemoryDC()
154:     destDC.SelectObject(destBmp)
155: 
156:     x = int(l)
157:     y = int(int(agg.height) - t)
158:     destDC.Blit(0, 0, int(width), int(height), srcDC, x, y)
159: 
160:     srcDC.SelectObject(wx.NullBitmap)
161:     destDC.SelectObject(wx.NullBitmap)
162: 
163:     return destBmp
164: 
165: 
166: @_BackendWx.export
167: class _BackendWxAgg(_BackendWx):
168:     FigureCanvas = FigureCanvasWxAgg
169:     _frame_class = FigureFrameWxAgg
170: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_268561 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_268561) is not StypyTypeError):

    if (import_268561 != 'pyd_module'):
        __import__(import_268561)
        sys_modules_268562 = sys.modules[import_268561]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_268562.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_268561)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import matplotlib' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_268563 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib')

if (type(import_268563) is not StypyTypeError):

    if (import_268563 != 'pyd_module'):
        __import__(import_268563)
        sys_modules_268564 = sys.modules[import_268563]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', sys_modules_268564.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', import_268563)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from matplotlib.figure import Figure' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_268565 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.figure')

if (type(import_268565) is not StypyTypeError):

    if (import_268565 != 'pyd_module'):
        __import__(import_268565)
        sys_modules_268566 = sys.modules[import_268565]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.figure', sys_modules_268566.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_268566, sys_modules_268566.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.figure', import_268565)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from matplotlib.backends.backend_agg import FigureCanvasAgg' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_268567 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends.backend_agg')

if (type(import_268567) is not StypyTypeError):

    if (import_268567 != 'pyd_module'):
        __import__(import_268567)
        sys_modules_268568 = sys.modules[import_268567]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends.backend_agg', sys_modules_268568.module_type_store, module_type_store, ['FigureCanvasAgg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_268568, sys_modules_268568.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends.backend_agg', None, module_type_store, ['FigureCanvasAgg'], [FigureCanvasAgg])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_agg' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends.backend_agg', import_268567)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from matplotlib.backends import wxc' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_268569 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.backends')

if (type(import_268569) is not StypyTypeError):

    if (import_268569 != 'pyd_module'):
        __import__(import_268569)
        sys_modules_268570 = sys.modules[import_268569]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.backends', sys_modules_268570.module_type_store, module_type_store, ['wx_compat'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_268570, sys_modules_268570.module_type_store, module_type_store)
    else:
        from matplotlib.backends import wx_compat as wxc

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.backends', None, module_type_store, ['wx_compat'], [wxc])

else:
    # Assigning a type to the variable 'matplotlib.backends' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.backends', import_268569)

# Adding an alias
module_type_store.add_alias('wxc', 'wx_compat')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from matplotlib.backends import backend_wx' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_268571 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backends')

if (type(import_268571) is not StypyTypeError):

    if (import_268571 != 'pyd_module'):
        __import__(import_268571)
        sys_modules_268572 = sys.modules[import_268571]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backends', sys_modules_268572.module_type_store, module_type_store, ['backend_wx'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_268572, sys_modules_268572.module_type_store, module_type_store)
    else:
        from matplotlib.backends import backend_wx

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backends', None, module_type_store, ['backend_wx'], [backend_wx])

else:
    # Assigning a type to the variable 'matplotlib.backends' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.backends', import_268571)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from matplotlib.backends.backend_wx import _BackendWx, FigureManagerWx, FigureCanvasWx, FigureFrameWx, DEBUG_MSG, NavigationToolbar2Wx, Toolbar' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_268573 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.backends.backend_wx')

if (type(import_268573) is not StypyTypeError):

    if (import_268573 != 'pyd_module'):
        __import__(import_268573)
        sys_modules_268574 = sys.modules[import_268573]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.backends.backend_wx', sys_modules_268574.module_type_store, module_type_store, ['_BackendWx', 'FigureManagerWx', 'FigureCanvasWx', 'FigureFrameWx', 'DEBUG_MSG', 'NavigationToolbar2Wx', 'Toolbar'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_268574, sys_modules_268574.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_wx import _BackendWx, FigureManagerWx, FigureCanvasWx, FigureFrameWx, DEBUG_MSG, NavigationToolbar2Wx, Toolbar

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.backends.backend_wx', None, module_type_store, ['_BackendWx', 'FigureManagerWx', 'FigureCanvasWx', 'FigureFrameWx', 'DEBUG_MSG', 'NavigationToolbar2Wx', 'Toolbar'], [_BackendWx, FigureManagerWx, FigureCanvasWx, FigureFrameWx, DEBUG_MSG, NavigationToolbar2Wx, Toolbar])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_wx' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.backends.backend_wx', import_268573)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import wx' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_268575 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'wx')

if (type(import_268575) is not StypyTypeError):

    if (import_268575 != 'pyd_module'):
        __import__(import_268575)
        sys_modules_268576 = sys.modules[import_268575]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'wx', sys_modules_268576.module_type_store, module_type_store)
    else:
        import wx

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'wx', wx, module_type_store)

else:
    # Assigning a type to the variable 'wx' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'wx', import_268575)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# Declaration of the 'FigureFrameWxAgg' class
# Getting the type of 'FigureFrameWx' (line 19)
FigureFrameWx_268577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'FigureFrameWx')

class FigureFrameWxAgg(FigureFrameWx_268577, ):

    @norecursion
    def get_canvas(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_canvas'
        module_type_store = module_type_store.open_function_context('get_canvas', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureFrameWxAgg.get_canvas.__dict__.__setitem__('stypy_localization', localization)
        FigureFrameWxAgg.get_canvas.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureFrameWxAgg.get_canvas.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureFrameWxAgg.get_canvas.__dict__.__setitem__('stypy_function_name', 'FigureFrameWxAgg.get_canvas')
        FigureFrameWxAgg.get_canvas.__dict__.__setitem__('stypy_param_names_list', ['fig'])
        FigureFrameWxAgg.get_canvas.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureFrameWxAgg.get_canvas.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureFrameWxAgg.get_canvas.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureFrameWxAgg.get_canvas.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureFrameWxAgg.get_canvas.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureFrameWxAgg.get_canvas.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureFrameWxAgg.get_canvas', ['fig'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_canvas', localization, ['fig'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_canvas(...)' code ##################

        
        # Call to FigureCanvasWxAgg(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'self' (line 21)
        self_268579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 33), 'self', False)
        int_268580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 39), 'int')
        # Getting the type of 'fig' (line 21)
        fig_268581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 43), 'fig', False)
        # Processing the call keyword arguments (line 21)
        kwargs_268582 = {}
        # Getting the type of 'FigureCanvasWxAgg' (line 21)
        FigureCanvasWxAgg_268578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'FigureCanvasWxAgg', False)
        # Calling FigureCanvasWxAgg(args, kwargs) (line 21)
        FigureCanvasWxAgg_call_result_268583 = invoke(stypy.reporting.localization.Localization(__file__, 21, 15), FigureCanvasWxAgg_268578, *[self_268579, int_268580, fig_268581], **kwargs_268582)
        
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type', FigureCanvasWxAgg_call_result_268583)
        
        # ################# End of 'get_canvas(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_canvas' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_268584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_268584)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_canvas'
        return stypy_return_type_268584


    @norecursion
    def _get_toolbar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_toolbar'
        module_type_store = module_type_store.open_function_context('_get_toolbar', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureFrameWxAgg._get_toolbar.__dict__.__setitem__('stypy_localization', localization)
        FigureFrameWxAgg._get_toolbar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureFrameWxAgg._get_toolbar.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureFrameWxAgg._get_toolbar.__dict__.__setitem__('stypy_function_name', 'FigureFrameWxAgg._get_toolbar')
        FigureFrameWxAgg._get_toolbar.__dict__.__setitem__('stypy_param_names_list', ['statbar'])
        FigureFrameWxAgg._get_toolbar.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureFrameWxAgg._get_toolbar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureFrameWxAgg._get_toolbar.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureFrameWxAgg._get_toolbar.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureFrameWxAgg._get_toolbar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureFrameWxAgg._get_toolbar.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureFrameWxAgg._get_toolbar', ['statbar'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_toolbar', localization, ['statbar'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_toolbar(...)' code ##################

        
        
        
        # Obtaining the type of the subscript
        unicode_268585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 31), 'unicode', u'toolbar')
        # Getting the type of 'matplotlib' (line 24)
        matplotlib_268586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'matplotlib')
        # Obtaining the member 'rcParams' of a type (line 24)
        rcParams_268587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 11), matplotlib_268586, 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 24)
        getitem___268588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 11), rcParams_268587, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 24)
        subscript_call_result_268589 = invoke(stypy.reporting.localization.Localization(__file__, 24, 11), getitem___268588, unicode_268585)
        
        unicode_268590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 45), 'unicode', u'toolbar2')
        # Applying the binary operator '==' (line 24)
        result_eq_268591 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 11), '==', subscript_call_result_268589, unicode_268590)
        
        # Testing the type of an if condition (line 24)
        if_condition_268592 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 8), result_eq_268591)
        # Assigning a type to the variable 'if_condition_268592' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'if_condition_268592', if_condition_268592)
        # SSA begins for if statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 25):
        
        # Assigning a Call to a Name (line 25):
        
        # Call to NavigationToolbar2WxAgg(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'self' (line 25)
        self_268594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 46), 'self', False)
        # Obtaining the member 'canvas' of a type (line 25)
        canvas_268595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 46), self_268594, 'canvas')
        # Processing the call keyword arguments (line 25)
        kwargs_268596 = {}
        # Getting the type of 'NavigationToolbar2WxAgg' (line 25)
        NavigationToolbar2WxAgg_268593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'NavigationToolbar2WxAgg', False)
        # Calling NavigationToolbar2WxAgg(args, kwargs) (line 25)
        NavigationToolbar2WxAgg_call_result_268597 = invoke(stypy.reporting.localization.Localization(__file__, 25, 22), NavigationToolbar2WxAgg_268593, *[canvas_268595], **kwargs_268596)
        
        # Assigning a type to the variable 'toolbar' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'toolbar', NavigationToolbar2WxAgg_call_result_268597)
        
        # Call to set_status_bar(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'statbar' (line 26)
        statbar_268600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 35), 'statbar', False)
        # Processing the call keyword arguments (line 26)
        kwargs_268601 = {}
        # Getting the type of 'toolbar' (line 26)
        toolbar_268598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'toolbar', False)
        # Obtaining the member 'set_status_bar' of a type (line 26)
        set_status_bar_268599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), toolbar_268598, 'set_status_bar')
        # Calling set_status_bar(args, kwargs) (line 26)
        set_status_bar_call_result_268602 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), set_status_bar_268599, *[statbar_268600], **kwargs_268601)
        
        # SSA branch for the else part of an if statement (line 24)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 28):
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'None' (line 28)
        None_268603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'None')
        # Assigning a type to the variable 'toolbar' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'toolbar', None_268603)
        # SSA join for if statement (line 24)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'toolbar' (line 29)
        toolbar_268604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'toolbar')
        # Assigning a type to the variable 'stypy_return_type' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', toolbar_268604)
        
        # ################# End of '_get_toolbar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_toolbar' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_268605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_268605)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_toolbar'
        return stypy_return_type_268605


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 0, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureFrameWxAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureFrameWxAgg' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'FigureFrameWxAgg', FigureFrameWxAgg)
# Declaration of the 'FigureCanvasWxAgg' class
# Getting the type of 'FigureCanvasAgg' (line 32)
FigureCanvasAgg_268606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'FigureCanvasAgg')
# Getting the type of 'FigureCanvasWx' (line 32)
FigureCanvasWx_268607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 41), 'FigureCanvasWx')

class FigureCanvasWxAgg(FigureCanvasAgg_268606, FigureCanvasWx_268607, ):
    unicode_268608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, (-1)), 'unicode', u'\n    The FigureCanvas contains the figure and does event handling.\n\n    In the wxPython backend, it is derived from wxPanel, and (usually)\n    lives inside a frame instantiated by a FigureManagerWx. The parent\n    window probably implements a wxSizer to control the displayed\n    control size - but we give a hint as to our preferred minimum\n    size.\n    ')

    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 43)
        None_268609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 26), 'None')
        defaults = [None_268609]
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWxAgg.draw.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWxAgg.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWxAgg.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWxAgg.draw.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWxAgg.draw')
        FigureCanvasWxAgg.draw.__dict__.__setitem__('stypy_param_names_list', ['drawDC'])
        FigureCanvasWxAgg.draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWxAgg.draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWxAgg.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWxAgg.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWxAgg.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWxAgg.draw.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWxAgg.draw', ['drawDC'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw', localization, ['drawDC'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw(...)' code ##################

        unicode_268610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, (-1)), 'unicode', u'\n        Render the figure using agg.\n        ')
        
        # Call to DEBUG_MSG(...): (line 47)
        # Processing the call arguments (line 47)
        unicode_268612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 18), 'unicode', u'draw()')
        int_268613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 28), 'int')
        # Getting the type of 'self' (line 47)
        self_268614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'self', False)
        # Processing the call keyword arguments (line 47)
        kwargs_268615 = {}
        # Getting the type of 'DEBUG_MSG' (line 47)
        DEBUG_MSG_268611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'DEBUG_MSG', False)
        # Calling DEBUG_MSG(args, kwargs) (line 47)
        DEBUG_MSG_call_result_268616 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), DEBUG_MSG_268611, *[unicode_268612, int_268613, self_268614], **kwargs_268615)
        
        
        # Call to draw(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'self' (line 48)
        self_268619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 29), 'self', False)
        # Processing the call keyword arguments (line 48)
        kwargs_268620 = {}
        # Getting the type of 'FigureCanvasAgg' (line 48)
        FigureCanvasAgg_268617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'FigureCanvasAgg', False)
        # Obtaining the member 'draw' of a type (line 48)
        draw_268618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), FigureCanvasAgg_268617, 'draw')
        # Calling draw(args, kwargs) (line 48)
        draw_call_result_268621 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), draw_268618, *[self_268619], **kwargs_268620)
        
        
        # Assigning a Call to a Attribute (line 50):
        
        # Assigning a Call to a Attribute (line 50):
        
        # Call to _convert_agg_to_wx_bitmap(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Call to get_renderer(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_268625 = {}
        # Getting the type of 'self' (line 50)
        self_268623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 48), 'self', False)
        # Obtaining the member 'get_renderer' of a type (line 50)
        get_renderer_268624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 48), self_268623, 'get_renderer')
        # Calling get_renderer(args, kwargs) (line 50)
        get_renderer_call_result_268626 = invoke(stypy.reporting.localization.Localization(__file__, 50, 48), get_renderer_268624, *[], **kwargs_268625)
        
        # Getting the type of 'None' (line 50)
        None_268627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 69), 'None', False)
        # Processing the call keyword arguments (line 50)
        kwargs_268628 = {}
        # Getting the type of '_convert_agg_to_wx_bitmap' (line 50)
        _convert_agg_to_wx_bitmap_268622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), '_convert_agg_to_wx_bitmap', False)
        # Calling _convert_agg_to_wx_bitmap(args, kwargs) (line 50)
        _convert_agg_to_wx_bitmap_call_result_268629 = invoke(stypy.reporting.localization.Localization(__file__, 50, 22), _convert_agg_to_wx_bitmap_268622, *[get_renderer_call_result_268626, None_268627], **kwargs_268628)
        
        # Getting the type of 'self' (line 50)
        self_268630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'self')
        # Setting the type of the member 'bitmap' of a type (line 50)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), self_268630, 'bitmap', _convert_agg_to_wx_bitmap_call_result_268629)
        
        # Assigning a Name to a Attribute (line 51):
        
        # Assigning a Name to a Attribute (line 51):
        # Getting the type of 'True' (line 51)
        True_268631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 24), 'True')
        # Getting the type of 'self' (line 51)
        self_268632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self')
        # Setting the type of the member '_isDrawn' of a type (line 51)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_268632, '_isDrawn', True_268631)
        
        # Call to gui_repaint(...): (line 52)
        # Processing the call keyword arguments (line 52)
        # Getting the type of 'drawDC' (line 52)
        drawDC_268635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 32), 'drawDC', False)
        keyword_268636 = drawDC_268635
        unicode_268637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 47), 'unicode', u'WXAgg')
        keyword_268638 = unicode_268637
        kwargs_268639 = {'origin': keyword_268638, 'drawDC': keyword_268636}
        # Getting the type of 'self' (line 52)
        self_268633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self', False)
        # Obtaining the member 'gui_repaint' of a type (line 52)
        gui_repaint_268634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_268633, 'gui_repaint')
        # Calling gui_repaint(args, kwargs) (line 52)
        gui_repaint_call_result_268640 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), gui_repaint_268634, *[], **kwargs_268639)
        
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_268641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_268641)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_268641


    @norecursion
    def blit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 54)
        None_268642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'None')
        defaults = [None_268642]
        # Create a new context for function 'blit'
        module_type_store = module_type_store.open_function_context('blit', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWxAgg.blit.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWxAgg.blit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWxAgg.blit.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWxAgg.blit.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWxAgg.blit')
        FigureCanvasWxAgg.blit.__dict__.__setitem__('stypy_param_names_list', ['bbox'])
        FigureCanvasWxAgg.blit.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWxAgg.blit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWxAgg.blit.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWxAgg.blit.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWxAgg.blit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWxAgg.blit.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWxAgg.blit', ['bbox'], None, None, defaults, varargs, kwargs)

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

        unicode_268643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'unicode', u'\n        Transfer the region of the agg buffer defined by bbox to the display.\n        If bbox is None, the entire buffer is transferred.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 59)
        # Getting the type of 'bbox' (line 59)
        bbox_268644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'bbox')
        # Getting the type of 'None' (line 59)
        None_268645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'None')
        
        (may_be_268646, more_types_in_union_268647) = may_be_none(bbox_268644, None_268645)

        if may_be_268646:

            if more_types_in_union_268647:
                # Runtime conditional SSA (line 59)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 60):
            
            # Assigning a Call to a Attribute (line 60):
            
            # Call to _convert_agg_to_wx_bitmap(...): (line 60)
            # Processing the call arguments (line 60)
            
            # Call to get_renderer(...): (line 60)
            # Processing the call keyword arguments (line 60)
            kwargs_268651 = {}
            # Getting the type of 'self' (line 60)
            self_268649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 52), 'self', False)
            # Obtaining the member 'get_renderer' of a type (line 60)
            get_renderer_268650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 52), self_268649, 'get_renderer')
            # Calling get_renderer(args, kwargs) (line 60)
            get_renderer_call_result_268652 = invoke(stypy.reporting.localization.Localization(__file__, 60, 52), get_renderer_268650, *[], **kwargs_268651)
            
            # Getting the type of 'None' (line 60)
            None_268653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 73), 'None', False)
            # Processing the call keyword arguments (line 60)
            kwargs_268654 = {}
            # Getting the type of '_convert_agg_to_wx_bitmap' (line 60)
            _convert_agg_to_wx_bitmap_268648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), '_convert_agg_to_wx_bitmap', False)
            # Calling _convert_agg_to_wx_bitmap(args, kwargs) (line 60)
            _convert_agg_to_wx_bitmap_call_result_268655 = invoke(stypy.reporting.localization.Localization(__file__, 60, 26), _convert_agg_to_wx_bitmap_268648, *[get_renderer_call_result_268652, None_268653], **kwargs_268654)
            
            # Getting the type of 'self' (line 60)
            self_268656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'self')
            # Setting the type of the member 'bitmap' of a type (line 60)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), self_268656, 'bitmap', _convert_agg_to_wx_bitmap_call_result_268655)
            
            # Call to gui_repaint(...): (line 61)
            # Processing the call keyword arguments (line 61)
            kwargs_268659 = {}
            # Getting the type of 'self' (line 61)
            self_268657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'self', False)
            # Obtaining the member 'gui_repaint' of a type (line 61)
            gui_repaint_268658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), self_268657, 'gui_repaint')
            # Calling gui_repaint(args, kwargs) (line 61)
            gui_repaint_call_result_268660 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), gui_repaint_268658, *[], **kwargs_268659)
            
            # Assigning a type to the variable 'stypy_return_type' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_268647:
                # SSA join for if statement (line 59)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Tuple (line 64):
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_268661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        # Getting the type of 'bbox' (line 64)
        bbox_268662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'bbox')
        # Obtaining the member 'bounds' of a type (line 64)
        bounds_268663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 21), bbox_268662, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___268664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), bounds_268663, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_268665 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___268664, int_268661)
        
        # Assigning a type to the variable 'tuple_var_assignment_268553' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_268553', subscript_call_result_268665)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_268666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        # Getting the type of 'bbox' (line 64)
        bbox_268667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'bbox')
        # Obtaining the member 'bounds' of a type (line 64)
        bounds_268668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 21), bbox_268667, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___268669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), bounds_268668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_268670 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___268669, int_268666)
        
        # Assigning a type to the variable 'tuple_var_assignment_268554' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_268554', subscript_call_result_268670)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_268671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        # Getting the type of 'bbox' (line 64)
        bbox_268672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'bbox')
        # Obtaining the member 'bounds' of a type (line 64)
        bounds_268673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 21), bbox_268672, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___268674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), bounds_268673, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_268675 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___268674, int_268671)
        
        # Assigning a type to the variable 'tuple_var_assignment_268555' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_268555', subscript_call_result_268675)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_268676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        # Getting the type of 'bbox' (line 64)
        bbox_268677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'bbox')
        # Obtaining the member 'bounds' of a type (line 64)
        bounds_268678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 21), bbox_268677, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___268679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), bounds_268678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_268680 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___268679, int_268676)
        
        # Assigning a type to the variable 'tuple_var_assignment_268556' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_268556', subscript_call_result_268680)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_268553' (line 64)
        tuple_var_assignment_268553_268681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_268553')
        # Assigning a type to the variable 'l' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'l', tuple_var_assignment_268553_268681)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_268554' (line 64)
        tuple_var_assignment_268554_268682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_268554')
        # Assigning a type to the variable 'b' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'b', tuple_var_assignment_268554_268682)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_268555' (line 64)
        tuple_var_assignment_268555_268683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_268555')
        # Assigning a type to the variable 'w' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 14), 'w', tuple_var_assignment_268555_268683)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_268556' (line 64)
        tuple_var_assignment_268556_268684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_268556')
        # Assigning a type to the variable 'h' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'h', tuple_var_assignment_268556_268684)
        
        # Assigning a BinOp to a Name (line 65):
        
        # Assigning a BinOp to a Name (line 65):
        # Getting the type of 'l' (line 65)
        l_268685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'l')
        # Getting the type of 'w' (line 65)
        w_268686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'w')
        # Applying the binary operator '+' (line 65)
        result_add_268687 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 12), '+', l_268685, w_268686)
        
        # Assigning a type to the variable 'r' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'r', result_add_268687)
        
        # Assigning a BinOp to a Name (line 66):
        
        # Assigning a BinOp to a Name (line 66):
        # Getting the type of 'b' (line 66)
        b_268688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'b')
        # Getting the type of 'h' (line 66)
        h_268689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'h')
        # Applying the binary operator '+' (line 66)
        result_add_268690 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 12), '+', b_268688, h_268689)
        
        # Assigning a type to the variable 't' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 't', result_add_268690)
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to int(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'l' (line 67)
        l_268692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'l', False)
        # Processing the call keyword arguments (line 67)
        kwargs_268693 = {}
        # Getting the type of 'int' (line 67)
        int_268691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'int', False)
        # Calling int(args, kwargs) (line 67)
        int_call_result_268694 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), int_268691, *[l_268692], **kwargs_268693)
        
        # Assigning a type to the variable 'x' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'x', int_call_result_268694)
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to int(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Call to GetHeight(...): (line 68)
        # Processing the call keyword arguments (line 68)
        kwargs_268699 = {}
        # Getting the type of 'self' (line 68)
        self_268696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'self', False)
        # Obtaining the member 'bitmap' of a type (line 68)
        bitmap_268697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), self_268696, 'bitmap')
        # Obtaining the member 'GetHeight' of a type (line 68)
        GetHeight_268698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), bitmap_268697, 'GetHeight')
        # Calling GetHeight(args, kwargs) (line 68)
        GetHeight_call_result_268700 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), GetHeight_268698, *[], **kwargs_268699)
        
        # Getting the type of 't' (line 68)
        t_268701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 42), 't', False)
        # Applying the binary operator '-' (line 68)
        result_sub_268702 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 16), '-', GetHeight_call_result_268700, t_268701)
        
        # Processing the call keyword arguments (line 68)
        kwargs_268703 = {}
        # Getting the type of 'int' (line 68)
        int_268695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'int', False)
        # Calling int(args, kwargs) (line 68)
        int_call_result_268704 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), int_268695, *[result_sub_268702], **kwargs_268703)
        
        # Assigning a type to the variable 'y' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'y', int_call_result_268704)
        
        # Assigning a Call to a Name (line 70):
        
        # Assigning a Call to a Name (line 70):
        
        # Call to _convert_agg_to_wx_bitmap(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Call to get_renderer(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_268708 = {}
        # Getting the type of 'self' (line 70)
        self_268706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 43), 'self', False)
        # Obtaining the member 'get_renderer' of a type (line 70)
        get_renderer_268707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 43), self_268706, 'get_renderer')
        # Calling get_renderer(args, kwargs) (line 70)
        get_renderer_call_result_268709 = invoke(stypy.reporting.localization.Localization(__file__, 70, 43), get_renderer_268707, *[], **kwargs_268708)
        
        # Getting the type of 'None' (line 70)
        None_268710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 64), 'None', False)
        # Processing the call keyword arguments (line 70)
        kwargs_268711 = {}
        # Getting the type of '_convert_agg_to_wx_bitmap' (line 70)
        _convert_agg_to_wx_bitmap_268705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), '_convert_agg_to_wx_bitmap', False)
        # Calling _convert_agg_to_wx_bitmap(args, kwargs) (line 70)
        _convert_agg_to_wx_bitmap_call_result_268712 = invoke(stypy.reporting.localization.Localization(__file__, 70, 17), _convert_agg_to_wx_bitmap_268705, *[get_renderer_call_result_268709, None_268710], **kwargs_268711)
        
        # Assigning a type to the variable 'srcBmp' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'srcBmp', _convert_agg_to_wx_bitmap_call_result_268712)
        
        # Assigning a Call to a Name (line 71):
        
        # Assigning a Call to a Name (line 71):
        
        # Call to MemoryDC(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_268715 = {}
        # Getting the type of 'wx' (line 71)
        wx_268713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'wx', False)
        # Obtaining the member 'MemoryDC' of a type (line 71)
        MemoryDC_268714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 16), wx_268713, 'MemoryDC')
        # Calling MemoryDC(args, kwargs) (line 71)
        MemoryDC_call_result_268716 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), MemoryDC_268714, *[], **kwargs_268715)
        
        # Assigning a type to the variable 'srcDC' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'srcDC', MemoryDC_call_result_268716)
        
        # Call to SelectObject(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'srcBmp' (line 72)
        srcBmp_268719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'srcBmp', False)
        # Processing the call keyword arguments (line 72)
        kwargs_268720 = {}
        # Getting the type of 'srcDC' (line 72)
        srcDC_268717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'srcDC', False)
        # Obtaining the member 'SelectObject' of a type (line 72)
        SelectObject_268718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), srcDC_268717, 'SelectObject')
        # Calling SelectObject(args, kwargs) (line 72)
        SelectObject_call_result_268721 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), SelectObject_268718, *[srcBmp_268719], **kwargs_268720)
        
        
        # Assigning a Call to a Name (line 74):
        
        # Assigning a Call to a Name (line 74):
        
        # Call to MemoryDC(...): (line 74)
        # Processing the call keyword arguments (line 74)
        kwargs_268724 = {}
        # Getting the type of 'wx' (line 74)
        wx_268722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'wx', False)
        # Obtaining the member 'MemoryDC' of a type (line 74)
        MemoryDC_268723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 17), wx_268722, 'MemoryDC')
        # Calling MemoryDC(args, kwargs) (line 74)
        MemoryDC_call_result_268725 = invoke(stypy.reporting.localization.Localization(__file__, 74, 17), MemoryDC_268723, *[], **kwargs_268724)
        
        # Assigning a type to the variable 'destDC' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'destDC', MemoryDC_call_result_268725)
        
        # Call to SelectObject(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'self' (line 75)
        self_268728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 28), 'self', False)
        # Obtaining the member 'bitmap' of a type (line 75)
        bitmap_268729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 28), self_268728, 'bitmap')
        # Processing the call keyword arguments (line 75)
        kwargs_268730 = {}
        # Getting the type of 'destDC' (line 75)
        destDC_268726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'destDC', False)
        # Obtaining the member 'SelectObject' of a type (line 75)
        SelectObject_268727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), destDC_268726, 'SelectObject')
        # Calling SelectObject(args, kwargs) (line 75)
        SelectObject_call_result_268731 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), SelectObject_268727, *[bitmap_268729], **kwargs_268730)
        
        
        # Call to Blit(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'x' (line 77)
        x_268734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'x', False)
        # Getting the type of 'y' (line 77)
        y_268735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 23), 'y', False)
        
        # Call to int(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'w' (line 77)
        w_268737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 30), 'w', False)
        # Processing the call keyword arguments (line 77)
        kwargs_268738 = {}
        # Getting the type of 'int' (line 77)
        int_268736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'int', False)
        # Calling int(args, kwargs) (line 77)
        int_call_result_268739 = invoke(stypy.reporting.localization.Localization(__file__, 77, 26), int_268736, *[w_268737], **kwargs_268738)
        
        
        # Call to int(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'h' (line 77)
        h_268741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'h', False)
        # Processing the call keyword arguments (line 77)
        kwargs_268742 = {}
        # Getting the type of 'int' (line 77)
        int_268740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 34), 'int', False)
        # Calling int(args, kwargs) (line 77)
        int_call_result_268743 = invoke(stypy.reporting.localization.Localization(__file__, 77, 34), int_268740, *[h_268741], **kwargs_268742)
        
        # Getting the type of 'srcDC' (line 77)
        srcDC_268744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 42), 'srcDC', False)
        # Getting the type of 'x' (line 77)
        x_268745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 49), 'x', False)
        # Getting the type of 'y' (line 77)
        y_268746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 52), 'y', False)
        # Processing the call keyword arguments (line 77)
        kwargs_268747 = {}
        # Getting the type of 'destDC' (line 77)
        destDC_268732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'destDC', False)
        # Obtaining the member 'Blit' of a type (line 77)
        Blit_268733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), destDC_268732, 'Blit')
        # Calling Blit(args, kwargs) (line 77)
        Blit_call_result_268748 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), Blit_268733, *[x_268734, y_268735, int_call_result_268739, int_call_result_268743, srcDC_268744, x_268745, y_268746], **kwargs_268747)
        
        
        # Call to SelectObject(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'wx' (line 79)
        wx_268751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 28), 'wx', False)
        # Obtaining the member 'NullBitmap' of a type (line 79)
        NullBitmap_268752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 28), wx_268751, 'NullBitmap')
        # Processing the call keyword arguments (line 79)
        kwargs_268753 = {}
        # Getting the type of 'destDC' (line 79)
        destDC_268749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'destDC', False)
        # Obtaining the member 'SelectObject' of a type (line 79)
        SelectObject_268750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), destDC_268749, 'SelectObject')
        # Calling SelectObject(args, kwargs) (line 79)
        SelectObject_call_result_268754 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), SelectObject_268750, *[NullBitmap_268752], **kwargs_268753)
        
        
        # Call to SelectObject(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'wx' (line 80)
        wx_268757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'wx', False)
        # Obtaining the member 'NullBitmap' of a type (line 80)
        NullBitmap_268758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 27), wx_268757, 'NullBitmap')
        # Processing the call keyword arguments (line 80)
        kwargs_268759 = {}
        # Getting the type of 'srcDC' (line 80)
        srcDC_268755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'srcDC', False)
        # Obtaining the member 'SelectObject' of a type (line 80)
        SelectObject_268756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), srcDC_268755, 'SelectObject')
        # Calling SelectObject(args, kwargs) (line 80)
        SelectObject_call_result_268760 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), SelectObject_268756, *[NullBitmap_268758], **kwargs_268759)
        
        
        # Call to gui_repaint(...): (line 81)
        # Processing the call keyword arguments (line 81)
        kwargs_268763 = {}
        # Getting the type of 'self' (line 81)
        self_268761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self', False)
        # Obtaining the member 'gui_repaint' of a type (line 81)
        gui_repaint_268762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_268761, 'gui_repaint')
        # Calling gui_repaint(args, kwargs) (line 81)
        gui_repaint_call_result_268764 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), gui_repaint_268762, *[], **kwargs_268763)
        
        
        # ################# End of 'blit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'blit' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_268765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_268765)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'blit'
        return stypy_return_type_268765

    
    # Assigning a Attribute to a Name (line 83):

    @norecursion
    def print_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_figure'
        module_type_store = module_type_store.open_function_context('print_figure', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWxAgg.print_figure.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWxAgg.print_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWxAgg.print_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWxAgg.print_figure.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWxAgg.print_figure')
        FigureCanvasWxAgg.print_figure.__dict__.__setitem__('stypy_param_names_list', ['filename'])
        FigureCanvasWxAgg.print_figure.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasWxAgg.print_figure.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasWxAgg.print_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWxAgg.print_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWxAgg.print_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWxAgg.print_figure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWxAgg.print_figure', ['filename'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_figure', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_figure(...)' code ##################

        
        # Call to print_figure(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'self' (line 87)
        self_268768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 37), 'self', False)
        # Getting the type of 'filename' (line 87)
        filename_268769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 43), 'filename', False)
        # Getting the type of 'args' (line 87)
        args_268770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 54), 'args', False)
        # Processing the call keyword arguments (line 87)
        # Getting the type of 'kwargs' (line 87)
        kwargs_268771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 62), 'kwargs', False)
        kwargs_268772 = {'kwargs_268771': kwargs_268771}
        # Getting the type of 'FigureCanvasAgg' (line 87)
        FigureCanvasAgg_268766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'FigureCanvasAgg', False)
        # Obtaining the member 'print_figure' of a type (line 87)
        print_figure_268767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), FigureCanvasAgg_268766, 'print_figure')
        # Calling print_figure(args, kwargs) (line 87)
        print_figure_call_result_268773 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), print_figure_268767, *[self_268768, filename_268769, args_268770], **kwargs_268772)
        
        
        # Getting the type of 'self' (line 92)
        self_268774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'self')
        # Obtaining the member '_isDrawn' of a type (line 92)
        _isDrawn_268775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 11), self_268774, '_isDrawn')
        # Testing the type of an if condition (line 92)
        if_condition_268776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 8), _isDrawn_268775)
        # Assigning a type to the variable 'if_condition_268776' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'if_condition_268776', if_condition_268776)
        # SSA begins for if statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to draw(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_268779 = {}
        # Getting the type of 'self' (line 93)
        self_268777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'self', False)
        # Obtaining the member 'draw' of a type (line 93)
        draw_268778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), self_268777, 'draw')
        # Calling draw(args, kwargs) (line 93)
        draw_call_result_268780 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), draw_268778, *[], **kwargs_268779)
        
        # SSA join for if statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'print_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_268781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_268781)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_figure'
        return stypy_return_type_268781


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWxAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureCanvasWxAgg' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'FigureCanvasWxAgg', FigureCanvasWxAgg)

# Assigning a Attribute to a Name (line 83):
# Getting the type of 'FigureCanvasAgg' (line 83)
FigureCanvasAgg_268782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'FigureCanvasAgg')
# Obtaining the member 'filetypes' of a type (line 83)
filetypes_268783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), FigureCanvasAgg_268782, 'filetypes')
# Getting the type of 'FigureCanvasWxAgg'
FigureCanvasWxAgg_268784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWxAgg')
# Setting the type of the member 'filetypes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWxAgg_268784, 'filetypes', filetypes_268783)
# Declaration of the 'NavigationToolbar2WxAgg' class
# Getting the type of 'NavigationToolbar2Wx' (line 96)
NavigationToolbar2Wx_268785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 30), 'NavigationToolbar2Wx')

class NavigationToolbar2WxAgg(NavigationToolbar2Wx_268785, ):

    @norecursion
    def get_canvas(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_canvas'
        module_type_store = module_type_store.open_function_context('get_canvas', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2WxAgg.get_canvas.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2WxAgg.get_canvas.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2WxAgg.get_canvas.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2WxAgg.get_canvas.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2WxAgg.get_canvas')
        NavigationToolbar2WxAgg.get_canvas.__dict__.__setitem__('stypy_param_names_list', ['frame', 'fig'])
        NavigationToolbar2WxAgg.get_canvas.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2WxAgg.get_canvas.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2WxAgg.get_canvas.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2WxAgg.get_canvas.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2WxAgg.get_canvas.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2WxAgg.get_canvas.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2WxAgg.get_canvas', ['frame', 'fig'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_canvas', localization, ['frame', 'fig'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_canvas(...)' code ##################

        
        # Call to FigureCanvasWxAgg(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'frame' (line 98)
        frame_268787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'frame', False)
        int_268788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 40), 'int')
        # Getting the type of 'fig' (line 98)
        fig_268789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 44), 'fig', False)
        # Processing the call keyword arguments (line 98)
        kwargs_268790 = {}
        # Getting the type of 'FigureCanvasWxAgg' (line 98)
        FigureCanvasWxAgg_268786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'FigureCanvasWxAgg', False)
        # Calling FigureCanvasWxAgg(args, kwargs) (line 98)
        FigureCanvasWxAgg_call_result_268791 = invoke(stypy.reporting.localization.Localization(__file__, 98, 15), FigureCanvasWxAgg_268786, *[frame_268787, int_268788, fig_268789], **kwargs_268790)
        
        # Assigning a type to the variable 'stypy_return_type' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'stypy_return_type', FigureCanvasWxAgg_call_result_268791)
        
        # ################# End of 'get_canvas(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_canvas' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_268792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_268792)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_canvas'
        return stypy_return_type_268792


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 96, 0, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2WxAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NavigationToolbar2WxAgg' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'NavigationToolbar2WxAgg', NavigationToolbar2WxAgg)

@norecursion
def _convert_agg_to_wx_image(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_convert_agg_to_wx_image'
    module_type_store = module_type_store.open_function_context('_convert_agg_to_wx_image', 104, 0, False)
    
    # Passed parameters checking function
    _convert_agg_to_wx_image.stypy_localization = localization
    _convert_agg_to_wx_image.stypy_type_of_self = None
    _convert_agg_to_wx_image.stypy_type_store = module_type_store
    _convert_agg_to_wx_image.stypy_function_name = '_convert_agg_to_wx_image'
    _convert_agg_to_wx_image.stypy_param_names_list = ['agg', 'bbox']
    _convert_agg_to_wx_image.stypy_varargs_param_name = None
    _convert_agg_to_wx_image.stypy_kwargs_param_name = None
    _convert_agg_to_wx_image.stypy_call_defaults = defaults
    _convert_agg_to_wx_image.stypy_call_varargs = varargs
    _convert_agg_to_wx_image.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_convert_agg_to_wx_image', ['agg', 'bbox'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_convert_agg_to_wx_image', localization, ['agg', 'bbox'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_convert_agg_to_wx_image(...)' code ##################

    unicode_268793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, (-1)), 'unicode', u'\n    Convert the region of the agg buffer bounded by bbox to a wx.Image.  If\n    bbox is None, the entire buffer is converted.\n\n    Note: agg must be a backend_agg.RendererAgg instance.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 111)
    # Getting the type of 'bbox' (line 111)
    bbox_268794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 7), 'bbox')
    # Getting the type of 'None' (line 111)
    None_268795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'None')
    
    (may_be_268796, more_types_in_union_268797) = may_be_none(bbox_268794, None_268795)

    if may_be_268796:

        if more_types_in_union_268797:
            # Runtime conditional SSA (line 111)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 113):
        
        # Assigning a Call to a Name (line 113):
        
        # Call to EmptyImage(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to int(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'agg' (line 113)
        agg_268801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 35), 'agg', False)
        # Obtaining the member 'width' of a type (line 113)
        width_268802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 35), agg_268801, 'width')
        # Processing the call keyword arguments (line 113)
        kwargs_268803 = {}
        # Getting the type of 'int' (line 113)
        int_268800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'int', False)
        # Calling int(args, kwargs) (line 113)
        int_call_result_268804 = invoke(stypy.reporting.localization.Localization(__file__, 113, 31), int_268800, *[width_268802], **kwargs_268803)
        
        
        # Call to int(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'agg' (line 113)
        agg_268806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 51), 'agg', False)
        # Obtaining the member 'height' of a type (line 113)
        height_268807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 51), agg_268806, 'height')
        # Processing the call keyword arguments (line 113)
        kwargs_268808 = {}
        # Getting the type of 'int' (line 113)
        int_268805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 47), 'int', False)
        # Calling int(args, kwargs) (line 113)
        int_call_result_268809 = invoke(stypy.reporting.localization.Localization(__file__, 113, 47), int_268805, *[height_268807], **kwargs_268808)
        
        # Processing the call keyword arguments (line 113)
        kwargs_268810 = {}
        # Getting the type of 'wxc' (line 113)
        wxc_268798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'wxc', False)
        # Obtaining the member 'EmptyImage' of a type (line 113)
        EmptyImage_268799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 16), wxc_268798, 'EmptyImage')
        # Calling EmptyImage(args, kwargs) (line 113)
        EmptyImage_call_result_268811 = invoke(stypy.reporting.localization.Localization(__file__, 113, 16), EmptyImage_268799, *[int_call_result_268804, int_call_result_268809], **kwargs_268810)
        
        # Assigning a type to the variable 'image' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'image', EmptyImage_call_result_268811)
        
        # Call to SetData(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to tostring_rgb(...): (line 114)
        # Processing the call keyword arguments (line 114)
        kwargs_268816 = {}
        # Getting the type of 'agg' (line 114)
        agg_268814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'agg', False)
        # Obtaining the member 'tostring_rgb' of a type (line 114)
        tostring_rgb_268815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 22), agg_268814, 'tostring_rgb')
        # Calling tostring_rgb(args, kwargs) (line 114)
        tostring_rgb_call_result_268817 = invoke(stypy.reporting.localization.Localization(__file__, 114, 22), tostring_rgb_268815, *[], **kwargs_268816)
        
        # Processing the call keyword arguments (line 114)
        kwargs_268818 = {}
        # Getting the type of 'image' (line 114)
        image_268812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'image', False)
        # Obtaining the member 'SetData' of a type (line 114)
        SetData_268813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), image_268812, 'SetData')
        # Calling SetData(args, kwargs) (line 114)
        SetData_call_result_268819 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), SetData_268813, *[tostring_rgb_call_result_268817], **kwargs_268818)
        
        # Getting the type of 'image' (line 115)
        image_268820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'image')
        # Assigning a type to the variable 'stypy_return_type' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'stypy_return_type', image_268820)

        if more_types_in_union_268797:
            # Runtime conditional SSA for else branch (line 111)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_268796) or more_types_in_union_268797):
        
        # Call to ImageFromBitmap(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Call to _WX28_clipped_agg_as_bitmap(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'agg' (line 118)
        agg_268824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 62), 'agg', False)
        # Getting the type of 'bbox' (line 118)
        bbox_268825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 67), 'bbox', False)
        # Processing the call keyword arguments (line 118)
        kwargs_268826 = {}
        # Getting the type of '_WX28_clipped_agg_as_bitmap' (line 118)
        _WX28_clipped_agg_as_bitmap_268823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 34), '_WX28_clipped_agg_as_bitmap', False)
        # Calling _WX28_clipped_agg_as_bitmap(args, kwargs) (line 118)
        _WX28_clipped_agg_as_bitmap_call_result_268827 = invoke(stypy.reporting.localization.Localization(__file__, 118, 34), _WX28_clipped_agg_as_bitmap_268823, *[agg_268824, bbox_268825], **kwargs_268826)
        
        # Processing the call keyword arguments (line 118)
        kwargs_268828 = {}
        # Getting the type of 'wx' (line 118)
        wx_268821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'wx', False)
        # Obtaining the member 'ImageFromBitmap' of a type (line 118)
        ImageFromBitmap_268822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 15), wx_268821, 'ImageFromBitmap')
        # Calling ImageFromBitmap(args, kwargs) (line 118)
        ImageFromBitmap_call_result_268829 = invoke(stypy.reporting.localization.Localization(__file__, 118, 15), ImageFromBitmap_268822, *[_WX28_clipped_agg_as_bitmap_call_result_268827], **kwargs_268828)
        
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type', ImageFromBitmap_call_result_268829)

        if (may_be_268796 and more_types_in_union_268797):
            # SSA join for if statement (line 111)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_convert_agg_to_wx_image(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_convert_agg_to_wx_image' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_268830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_268830)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_convert_agg_to_wx_image'
    return stypy_return_type_268830

# Assigning a type to the variable '_convert_agg_to_wx_image' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), '_convert_agg_to_wx_image', _convert_agg_to_wx_image)

@norecursion
def _convert_agg_to_wx_bitmap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_convert_agg_to_wx_bitmap'
    module_type_store = module_type_store.open_function_context('_convert_agg_to_wx_bitmap', 121, 0, False)
    
    # Passed parameters checking function
    _convert_agg_to_wx_bitmap.stypy_localization = localization
    _convert_agg_to_wx_bitmap.stypy_type_of_self = None
    _convert_agg_to_wx_bitmap.stypy_type_store = module_type_store
    _convert_agg_to_wx_bitmap.stypy_function_name = '_convert_agg_to_wx_bitmap'
    _convert_agg_to_wx_bitmap.stypy_param_names_list = ['agg', 'bbox']
    _convert_agg_to_wx_bitmap.stypy_varargs_param_name = None
    _convert_agg_to_wx_bitmap.stypy_kwargs_param_name = None
    _convert_agg_to_wx_bitmap.stypy_call_defaults = defaults
    _convert_agg_to_wx_bitmap.stypy_call_varargs = varargs
    _convert_agg_to_wx_bitmap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_convert_agg_to_wx_bitmap', ['agg', 'bbox'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_convert_agg_to_wx_bitmap', localization, ['agg', 'bbox'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_convert_agg_to_wx_bitmap(...)' code ##################

    unicode_268831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, (-1)), 'unicode', u'\n    Convert the region of the agg buffer bounded by bbox to a wx.Bitmap.  If\n    bbox is None, the entire buffer is converted.\n\n    Note: agg must be a backend_agg.RendererAgg instance.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 128)
    # Getting the type of 'bbox' (line 128)
    bbox_268832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 7), 'bbox')
    # Getting the type of 'None' (line 128)
    None_268833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'None')
    
    (may_be_268834, more_types_in_union_268835) = may_be_none(bbox_268832, None_268833)

    if may_be_268834:

        if more_types_in_union_268835:
            # Runtime conditional SSA (line 128)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to BitmapFromBuffer(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Call to int(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'agg' (line 130)
        agg_268839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 40), 'agg', False)
        # Obtaining the member 'width' of a type (line 130)
        width_268840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 40), agg_268839, 'width')
        # Processing the call keyword arguments (line 130)
        kwargs_268841 = {}
        # Getting the type of 'int' (line 130)
        int_268838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 36), 'int', False)
        # Calling int(args, kwargs) (line 130)
        int_call_result_268842 = invoke(stypy.reporting.localization.Localization(__file__, 130, 36), int_268838, *[width_268840], **kwargs_268841)
        
        
        # Call to int(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'agg' (line 130)
        agg_268844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 56), 'agg', False)
        # Obtaining the member 'height' of a type (line 130)
        height_268845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 56), agg_268844, 'height')
        # Processing the call keyword arguments (line 130)
        kwargs_268846 = {}
        # Getting the type of 'int' (line 130)
        int_268843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 52), 'int', False)
        # Calling int(args, kwargs) (line 130)
        int_call_result_268847 = invoke(stypy.reporting.localization.Localization(__file__, 130, 52), int_268843, *[height_268845], **kwargs_268846)
        
        
        # Call to buffer_rgba(...): (line 131)
        # Processing the call keyword arguments (line 131)
        kwargs_268850 = {}
        # Getting the type of 'agg' (line 131)
        agg_268848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 36), 'agg', False)
        # Obtaining the member 'buffer_rgba' of a type (line 131)
        buffer_rgba_268849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 36), agg_268848, 'buffer_rgba')
        # Calling buffer_rgba(args, kwargs) (line 131)
        buffer_rgba_call_result_268851 = invoke(stypy.reporting.localization.Localization(__file__, 131, 36), buffer_rgba_268849, *[], **kwargs_268850)
        
        # Processing the call keyword arguments (line 130)
        kwargs_268852 = {}
        # Getting the type of 'wxc' (line 130)
        wxc_268836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'wxc', False)
        # Obtaining the member 'BitmapFromBuffer' of a type (line 130)
        BitmapFromBuffer_268837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 15), wxc_268836, 'BitmapFromBuffer')
        # Calling BitmapFromBuffer(args, kwargs) (line 130)
        BitmapFromBuffer_call_result_268853 = invoke(stypy.reporting.localization.Localization(__file__, 130, 15), BitmapFromBuffer_268837, *[int_call_result_268842, int_call_result_268847, buffer_rgba_call_result_268851], **kwargs_268852)
        
        # Assigning a type to the variable 'stypy_return_type' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'stypy_return_type', BitmapFromBuffer_call_result_268853)

        if more_types_in_union_268835:
            # Runtime conditional SSA for else branch (line 128)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_268834) or more_types_in_union_268835):
        
        # Call to _WX28_clipped_agg_as_bitmap(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'agg' (line 134)
        agg_268855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 43), 'agg', False)
        # Getting the type of 'bbox' (line 134)
        bbox_268856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 48), 'bbox', False)
        # Processing the call keyword arguments (line 134)
        kwargs_268857 = {}
        # Getting the type of '_WX28_clipped_agg_as_bitmap' (line 134)
        _WX28_clipped_agg_as_bitmap_268854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), '_WX28_clipped_agg_as_bitmap', False)
        # Calling _WX28_clipped_agg_as_bitmap(args, kwargs) (line 134)
        _WX28_clipped_agg_as_bitmap_call_result_268858 = invoke(stypy.reporting.localization.Localization(__file__, 134, 15), _WX28_clipped_agg_as_bitmap_268854, *[agg_268855, bbox_268856], **kwargs_268857)
        
        # Assigning a type to the variable 'stypy_return_type' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type', _WX28_clipped_agg_as_bitmap_call_result_268858)

        if (may_be_268834 and more_types_in_union_268835):
            # SSA join for if statement (line 128)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_convert_agg_to_wx_bitmap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_convert_agg_to_wx_bitmap' in the type store
    # Getting the type of 'stypy_return_type' (line 121)
    stypy_return_type_268859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_268859)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_convert_agg_to_wx_bitmap'
    return stypy_return_type_268859

# Assigning a type to the variable '_convert_agg_to_wx_bitmap' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), '_convert_agg_to_wx_bitmap', _convert_agg_to_wx_bitmap)

@norecursion
def _WX28_clipped_agg_as_bitmap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_WX28_clipped_agg_as_bitmap'
    module_type_store = module_type_store.open_function_context('_WX28_clipped_agg_as_bitmap', 137, 0, False)
    
    # Passed parameters checking function
    _WX28_clipped_agg_as_bitmap.stypy_localization = localization
    _WX28_clipped_agg_as_bitmap.stypy_type_of_self = None
    _WX28_clipped_agg_as_bitmap.stypy_type_store = module_type_store
    _WX28_clipped_agg_as_bitmap.stypy_function_name = '_WX28_clipped_agg_as_bitmap'
    _WX28_clipped_agg_as_bitmap.stypy_param_names_list = ['agg', 'bbox']
    _WX28_clipped_agg_as_bitmap.stypy_varargs_param_name = None
    _WX28_clipped_agg_as_bitmap.stypy_kwargs_param_name = None
    _WX28_clipped_agg_as_bitmap.stypy_call_defaults = defaults
    _WX28_clipped_agg_as_bitmap.stypy_call_varargs = varargs
    _WX28_clipped_agg_as_bitmap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_WX28_clipped_agg_as_bitmap', ['agg', 'bbox'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_WX28_clipped_agg_as_bitmap', localization, ['agg', 'bbox'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_WX28_clipped_agg_as_bitmap(...)' code ##################

    unicode_268860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, (-1)), 'unicode', u'\n    Convert the region of a the agg buffer bounded by bbox to a wx.Bitmap.\n\n    Note: agg must be a backend_agg.RendererAgg instance.\n    ')
    
    # Assigning a Attribute to a Tuple (line 143):
    
    # Assigning a Subscript to a Name (line 143):
    
    # Obtaining the type of the subscript
    int_268861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 4), 'int')
    # Getting the type of 'bbox' (line 143)
    bbox_268862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 26), 'bbox')
    # Obtaining the member 'bounds' of a type (line 143)
    bounds_268863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 26), bbox_268862, 'bounds')
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___268864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 4), bounds_268863, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_268865 = invoke(stypy.reporting.localization.Localization(__file__, 143, 4), getitem___268864, int_268861)
    
    # Assigning a type to the variable 'tuple_var_assignment_268557' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_268557', subscript_call_result_268865)
    
    # Assigning a Subscript to a Name (line 143):
    
    # Obtaining the type of the subscript
    int_268866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 4), 'int')
    # Getting the type of 'bbox' (line 143)
    bbox_268867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 26), 'bbox')
    # Obtaining the member 'bounds' of a type (line 143)
    bounds_268868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 26), bbox_268867, 'bounds')
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___268869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 4), bounds_268868, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_268870 = invoke(stypy.reporting.localization.Localization(__file__, 143, 4), getitem___268869, int_268866)
    
    # Assigning a type to the variable 'tuple_var_assignment_268558' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_268558', subscript_call_result_268870)
    
    # Assigning a Subscript to a Name (line 143):
    
    # Obtaining the type of the subscript
    int_268871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 4), 'int')
    # Getting the type of 'bbox' (line 143)
    bbox_268872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 26), 'bbox')
    # Obtaining the member 'bounds' of a type (line 143)
    bounds_268873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 26), bbox_268872, 'bounds')
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___268874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 4), bounds_268873, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_268875 = invoke(stypy.reporting.localization.Localization(__file__, 143, 4), getitem___268874, int_268871)
    
    # Assigning a type to the variable 'tuple_var_assignment_268559' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_268559', subscript_call_result_268875)
    
    # Assigning a Subscript to a Name (line 143):
    
    # Obtaining the type of the subscript
    int_268876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 4), 'int')
    # Getting the type of 'bbox' (line 143)
    bbox_268877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 26), 'bbox')
    # Obtaining the member 'bounds' of a type (line 143)
    bounds_268878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 26), bbox_268877, 'bounds')
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___268879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 4), bounds_268878, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_268880 = invoke(stypy.reporting.localization.Localization(__file__, 143, 4), getitem___268879, int_268876)
    
    # Assigning a type to the variable 'tuple_var_assignment_268560' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_268560', subscript_call_result_268880)
    
    # Assigning a Name to a Name (line 143):
    # Getting the type of 'tuple_var_assignment_268557' (line 143)
    tuple_var_assignment_268557_268881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_268557')
    # Assigning a type to the variable 'l' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'l', tuple_var_assignment_268557_268881)
    
    # Assigning a Name to a Name (line 143):
    # Getting the type of 'tuple_var_assignment_268558' (line 143)
    tuple_var_assignment_268558_268882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_268558')
    # Assigning a type to the variable 'b' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 7), 'b', tuple_var_assignment_268558_268882)
    
    # Assigning a Name to a Name (line 143):
    # Getting the type of 'tuple_var_assignment_268559' (line 143)
    tuple_var_assignment_268559_268883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_268559')
    # Assigning a type to the variable 'width' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 10), 'width', tuple_var_assignment_268559_268883)
    
    # Assigning a Name to a Name (line 143):
    # Getting the type of 'tuple_var_assignment_268560' (line 143)
    tuple_var_assignment_268560_268884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_268560')
    # Assigning a type to the variable 'height' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 17), 'height', tuple_var_assignment_268560_268884)
    
    # Assigning a BinOp to a Name (line 144):
    
    # Assigning a BinOp to a Name (line 144):
    # Getting the type of 'l' (line 144)
    l_268885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'l')
    # Getting the type of 'width' (line 144)
    width_268886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'width')
    # Applying the binary operator '+' (line 144)
    result_add_268887 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 8), '+', l_268885, width_268886)
    
    # Assigning a type to the variable 'r' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'r', result_add_268887)
    
    # Assigning a BinOp to a Name (line 145):
    
    # Assigning a BinOp to a Name (line 145):
    # Getting the type of 'b' (line 145)
    b_268888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'b')
    # Getting the type of 'height' (line 145)
    height_268889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'height')
    # Applying the binary operator '+' (line 145)
    result_add_268890 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 8), '+', b_268888, height_268889)
    
    # Assigning a type to the variable 't' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 't', result_add_268890)
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 147):
    
    # Call to BitmapFromBuffer(...): (line 147)
    # Processing the call arguments (line 147)
    
    # Call to int(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'agg' (line 147)
    agg_268894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 38), 'agg', False)
    # Obtaining the member 'width' of a type (line 147)
    width_268895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 38), agg_268894, 'width')
    # Processing the call keyword arguments (line 147)
    kwargs_268896 = {}
    # Getting the type of 'int' (line 147)
    int_268893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 34), 'int', False)
    # Calling int(args, kwargs) (line 147)
    int_call_result_268897 = invoke(stypy.reporting.localization.Localization(__file__, 147, 34), int_268893, *[width_268895], **kwargs_268896)
    
    
    # Call to int(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'agg' (line 147)
    agg_268899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 54), 'agg', False)
    # Obtaining the member 'height' of a type (line 147)
    height_268900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 54), agg_268899, 'height')
    # Processing the call keyword arguments (line 147)
    kwargs_268901 = {}
    # Getting the type of 'int' (line 147)
    int_268898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 50), 'int', False)
    # Calling int(args, kwargs) (line 147)
    int_call_result_268902 = invoke(stypy.reporting.localization.Localization(__file__, 147, 50), int_268898, *[height_268900], **kwargs_268901)
    
    
    # Call to buffer_rgba(...): (line 148)
    # Processing the call keyword arguments (line 148)
    kwargs_268905 = {}
    # Getting the type of 'agg' (line 148)
    agg_268903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 34), 'agg', False)
    # Obtaining the member 'buffer_rgba' of a type (line 148)
    buffer_rgba_268904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 34), agg_268903, 'buffer_rgba')
    # Calling buffer_rgba(args, kwargs) (line 148)
    buffer_rgba_call_result_268906 = invoke(stypy.reporting.localization.Localization(__file__, 148, 34), buffer_rgba_268904, *[], **kwargs_268905)
    
    # Processing the call keyword arguments (line 147)
    kwargs_268907 = {}
    # Getting the type of 'wxc' (line 147)
    wxc_268891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 13), 'wxc', False)
    # Obtaining the member 'BitmapFromBuffer' of a type (line 147)
    BitmapFromBuffer_268892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 13), wxc_268891, 'BitmapFromBuffer')
    # Calling BitmapFromBuffer(args, kwargs) (line 147)
    BitmapFromBuffer_call_result_268908 = invoke(stypy.reporting.localization.Localization(__file__, 147, 13), BitmapFromBuffer_268892, *[int_call_result_268897, int_call_result_268902, buffer_rgba_call_result_268906], **kwargs_268907)
    
    # Assigning a type to the variable 'srcBmp' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'srcBmp', BitmapFromBuffer_call_result_268908)
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to MemoryDC(...): (line 149)
    # Processing the call keyword arguments (line 149)
    kwargs_268911 = {}
    # Getting the type of 'wx' (line 149)
    wx_268909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'wx', False)
    # Obtaining the member 'MemoryDC' of a type (line 149)
    MemoryDC_268910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), wx_268909, 'MemoryDC')
    # Calling MemoryDC(args, kwargs) (line 149)
    MemoryDC_call_result_268912 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), MemoryDC_268910, *[], **kwargs_268911)
    
    # Assigning a type to the variable 'srcDC' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'srcDC', MemoryDC_call_result_268912)
    
    # Call to SelectObject(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'srcBmp' (line 150)
    srcBmp_268915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 23), 'srcBmp', False)
    # Processing the call keyword arguments (line 150)
    kwargs_268916 = {}
    # Getting the type of 'srcDC' (line 150)
    srcDC_268913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'srcDC', False)
    # Obtaining the member 'SelectObject' of a type (line 150)
    SelectObject_268914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 4), srcDC_268913, 'SelectObject')
    # Calling SelectObject(args, kwargs) (line 150)
    SelectObject_call_result_268917 = invoke(stypy.reporting.localization.Localization(__file__, 150, 4), SelectObject_268914, *[srcBmp_268915], **kwargs_268916)
    
    
    # Assigning a Call to a Name (line 152):
    
    # Assigning a Call to a Name (line 152):
    
    # Call to EmptyBitmap(...): (line 152)
    # Processing the call arguments (line 152)
    
    # Call to int(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'width' (line 152)
    width_268921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 34), 'width', False)
    # Processing the call keyword arguments (line 152)
    kwargs_268922 = {}
    # Getting the type of 'int' (line 152)
    int_268920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'int', False)
    # Calling int(args, kwargs) (line 152)
    int_call_result_268923 = invoke(stypy.reporting.localization.Localization(__file__, 152, 30), int_268920, *[width_268921], **kwargs_268922)
    
    
    # Call to int(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'height' (line 152)
    height_268925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 46), 'height', False)
    # Processing the call keyword arguments (line 152)
    kwargs_268926 = {}
    # Getting the type of 'int' (line 152)
    int_268924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 42), 'int', False)
    # Calling int(args, kwargs) (line 152)
    int_call_result_268927 = invoke(stypy.reporting.localization.Localization(__file__, 152, 42), int_268924, *[height_268925], **kwargs_268926)
    
    # Processing the call keyword arguments (line 152)
    kwargs_268928 = {}
    # Getting the type of 'wxc' (line 152)
    wxc_268918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 14), 'wxc', False)
    # Obtaining the member 'EmptyBitmap' of a type (line 152)
    EmptyBitmap_268919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 14), wxc_268918, 'EmptyBitmap')
    # Calling EmptyBitmap(args, kwargs) (line 152)
    EmptyBitmap_call_result_268929 = invoke(stypy.reporting.localization.Localization(__file__, 152, 14), EmptyBitmap_268919, *[int_call_result_268923, int_call_result_268927], **kwargs_268928)
    
    # Assigning a type to the variable 'destBmp' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'destBmp', EmptyBitmap_call_result_268929)
    
    # Assigning a Call to a Name (line 153):
    
    # Assigning a Call to a Name (line 153):
    
    # Call to MemoryDC(...): (line 153)
    # Processing the call keyword arguments (line 153)
    kwargs_268932 = {}
    # Getting the type of 'wx' (line 153)
    wx_268930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 13), 'wx', False)
    # Obtaining the member 'MemoryDC' of a type (line 153)
    MemoryDC_268931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 13), wx_268930, 'MemoryDC')
    # Calling MemoryDC(args, kwargs) (line 153)
    MemoryDC_call_result_268933 = invoke(stypy.reporting.localization.Localization(__file__, 153, 13), MemoryDC_268931, *[], **kwargs_268932)
    
    # Assigning a type to the variable 'destDC' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'destDC', MemoryDC_call_result_268933)
    
    # Call to SelectObject(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'destBmp' (line 154)
    destBmp_268936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 24), 'destBmp', False)
    # Processing the call keyword arguments (line 154)
    kwargs_268937 = {}
    # Getting the type of 'destDC' (line 154)
    destDC_268934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'destDC', False)
    # Obtaining the member 'SelectObject' of a type (line 154)
    SelectObject_268935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 4), destDC_268934, 'SelectObject')
    # Calling SelectObject(args, kwargs) (line 154)
    SelectObject_call_result_268938 = invoke(stypy.reporting.localization.Localization(__file__, 154, 4), SelectObject_268935, *[destBmp_268936], **kwargs_268937)
    
    
    # Assigning a Call to a Name (line 156):
    
    # Assigning a Call to a Name (line 156):
    
    # Call to int(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'l' (line 156)
    l_268940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'l', False)
    # Processing the call keyword arguments (line 156)
    kwargs_268941 = {}
    # Getting the type of 'int' (line 156)
    int_268939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'int', False)
    # Calling int(args, kwargs) (line 156)
    int_call_result_268942 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), int_268939, *[l_268940], **kwargs_268941)
    
    # Assigning a type to the variable 'x' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'x', int_call_result_268942)
    
    # Assigning a Call to a Name (line 157):
    
    # Assigning a Call to a Name (line 157):
    
    # Call to int(...): (line 157)
    # Processing the call arguments (line 157)
    
    # Call to int(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'agg' (line 157)
    agg_268945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'agg', False)
    # Obtaining the member 'height' of a type (line 157)
    height_268946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 16), agg_268945, 'height')
    # Processing the call keyword arguments (line 157)
    kwargs_268947 = {}
    # Getting the type of 'int' (line 157)
    int_268944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'int', False)
    # Calling int(args, kwargs) (line 157)
    int_call_result_268948 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), int_268944, *[height_268946], **kwargs_268947)
    
    # Getting the type of 't' (line 157)
    t_268949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 't', False)
    # Applying the binary operator '-' (line 157)
    result_sub_268950 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 12), '-', int_call_result_268948, t_268949)
    
    # Processing the call keyword arguments (line 157)
    kwargs_268951 = {}
    # Getting the type of 'int' (line 157)
    int_268943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'int', False)
    # Calling int(args, kwargs) (line 157)
    int_call_result_268952 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), int_268943, *[result_sub_268950], **kwargs_268951)
    
    # Assigning a type to the variable 'y' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'y', int_call_result_268952)
    
    # Call to Blit(...): (line 158)
    # Processing the call arguments (line 158)
    int_268955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 16), 'int')
    int_268956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 19), 'int')
    
    # Call to int(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'width' (line 158)
    width_268958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 26), 'width', False)
    # Processing the call keyword arguments (line 158)
    kwargs_268959 = {}
    # Getting the type of 'int' (line 158)
    int_268957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'int', False)
    # Calling int(args, kwargs) (line 158)
    int_call_result_268960 = invoke(stypy.reporting.localization.Localization(__file__, 158, 22), int_268957, *[width_268958], **kwargs_268959)
    
    
    # Call to int(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'height' (line 158)
    height_268962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 38), 'height', False)
    # Processing the call keyword arguments (line 158)
    kwargs_268963 = {}
    # Getting the type of 'int' (line 158)
    int_268961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 34), 'int', False)
    # Calling int(args, kwargs) (line 158)
    int_call_result_268964 = invoke(stypy.reporting.localization.Localization(__file__, 158, 34), int_268961, *[height_268962], **kwargs_268963)
    
    # Getting the type of 'srcDC' (line 158)
    srcDC_268965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 47), 'srcDC', False)
    # Getting the type of 'x' (line 158)
    x_268966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 54), 'x', False)
    # Getting the type of 'y' (line 158)
    y_268967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 57), 'y', False)
    # Processing the call keyword arguments (line 158)
    kwargs_268968 = {}
    # Getting the type of 'destDC' (line 158)
    destDC_268953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'destDC', False)
    # Obtaining the member 'Blit' of a type (line 158)
    Blit_268954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 4), destDC_268953, 'Blit')
    # Calling Blit(args, kwargs) (line 158)
    Blit_call_result_268969 = invoke(stypy.reporting.localization.Localization(__file__, 158, 4), Blit_268954, *[int_268955, int_268956, int_call_result_268960, int_call_result_268964, srcDC_268965, x_268966, y_268967], **kwargs_268968)
    
    
    # Call to SelectObject(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'wx' (line 160)
    wx_268972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 23), 'wx', False)
    # Obtaining the member 'NullBitmap' of a type (line 160)
    NullBitmap_268973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 23), wx_268972, 'NullBitmap')
    # Processing the call keyword arguments (line 160)
    kwargs_268974 = {}
    # Getting the type of 'srcDC' (line 160)
    srcDC_268970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'srcDC', False)
    # Obtaining the member 'SelectObject' of a type (line 160)
    SelectObject_268971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 4), srcDC_268970, 'SelectObject')
    # Calling SelectObject(args, kwargs) (line 160)
    SelectObject_call_result_268975 = invoke(stypy.reporting.localization.Localization(__file__, 160, 4), SelectObject_268971, *[NullBitmap_268973], **kwargs_268974)
    
    
    # Call to SelectObject(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'wx' (line 161)
    wx_268978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'wx', False)
    # Obtaining the member 'NullBitmap' of a type (line 161)
    NullBitmap_268979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 24), wx_268978, 'NullBitmap')
    # Processing the call keyword arguments (line 161)
    kwargs_268980 = {}
    # Getting the type of 'destDC' (line 161)
    destDC_268976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'destDC', False)
    # Obtaining the member 'SelectObject' of a type (line 161)
    SelectObject_268977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), destDC_268976, 'SelectObject')
    # Calling SelectObject(args, kwargs) (line 161)
    SelectObject_call_result_268981 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), SelectObject_268977, *[NullBitmap_268979], **kwargs_268980)
    
    # Getting the type of 'destBmp' (line 163)
    destBmp_268982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'destBmp')
    # Assigning a type to the variable 'stypy_return_type' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type', destBmp_268982)
    
    # ################# End of '_WX28_clipped_agg_as_bitmap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_WX28_clipped_agg_as_bitmap' in the type store
    # Getting the type of 'stypy_return_type' (line 137)
    stypy_return_type_268983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_268983)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_WX28_clipped_agg_as_bitmap'
    return stypy_return_type_268983

# Assigning a type to the variable '_WX28_clipped_agg_as_bitmap' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), '_WX28_clipped_agg_as_bitmap', _WX28_clipped_agg_as_bitmap)
# Declaration of the '_BackendWxAgg' class
# Getting the type of '_BackendWx' (line 167)
_BackendWx_268984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), '_BackendWx')

class _BackendWxAgg(_BackendWx_268984, ):
    
    # Assigning a Name to a Name (line 168):
    
    # Assigning a Name to a Name (line 169):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 166, 0, False)
        # Assigning a type to the variable 'self' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendWxAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendWxAgg' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), '_BackendWxAgg', _BackendWxAgg)

# Assigning a Name to a Name (line 168):
# Getting the type of 'FigureCanvasWxAgg' (line 168)
FigureCanvasWxAgg_268985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'FigureCanvasWxAgg')
# Getting the type of '_BackendWxAgg'
_BackendWxAgg_268986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendWxAgg')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendWxAgg_268986, 'FigureCanvas', FigureCanvasWxAgg_268985)

# Assigning a Name to a Name (line 169):
# Getting the type of 'FigureFrameWxAgg' (line 169)
FigureFrameWxAgg_268987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'FigureFrameWxAgg')
# Getting the type of '_BackendWxAgg'
_BackendWxAgg_268988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendWxAgg')
# Setting the type of the member '_frame_class' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendWxAgg_268988, '_frame_class', FigureFrameWxAgg_268987)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
