
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Render to gtk from agg
3: '''
4: from __future__ import (absolute_import, division, print_function,
5:                         unicode_literals)
6: 
7: import six
8: 
9: import os
10: 
11: import matplotlib
12: from matplotlib.figure import Figure
13: from matplotlib.backends.backend_agg import FigureCanvasAgg
14: from matplotlib.backends.backend_gtk import (
15:     gtk, _BackendGTK, FigureCanvasGTK, FigureManagerGTK, NavigationToolbar2GTK,
16:     backend_version, error_msg_gtk, PIXELS_PER_INCH)
17: from matplotlib.backends._gtkagg import agg_to_gtk_drawable
18: 
19: 
20: DEBUG = False
21: 
22: class NavigationToolbar2GTKAgg(NavigationToolbar2GTK):
23:     def _get_canvas(self, fig):
24:         return FigureCanvasGTKAgg(fig)
25: 
26: 
27: class FigureManagerGTKAgg(FigureManagerGTK):
28:     def _get_toolbar(self, canvas):
29:         # must be inited after the window, drawingArea and figure
30:         # attrs are set
31:         if matplotlib.rcParams['toolbar']=='toolbar2':
32:             toolbar = NavigationToolbar2GTKAgg (canvas, self.window)
33:         else:
34:             toolbar = None
35:         return toolbar
36: 
37: 
38: class FigureCanvasGTKAgg(FigureCanvasGTK, FigureCanvasAgg):
39:     filetypes = FigureCanvasGTK.filetypes.copy()
40:     filetypes.update(FigureCanvasAgg.filetypes)
41: 
42:     def configure_event(self, widget, event=None):
43: 
44:         if DEBUG: print('FigureCanvasGTKAgg.configure_event')
45:         if widget.window is None:
46:             return
47:         try:
48:             del self.renderer
49:         except AttributeError:
50:             pass
51:         w,h = widget.window.get_size()
52:         if w==1 or h==1: return # empty fig
53: 
54:         # compute desired figure size in inches
55:         dpival = self.figure.dpi
56:         winch = w/dpival
57:         hinch = h/dpival
58:         self.figure.set_size_inches(winch, hinch, forward=False)
59:         self._need_redraw = True
60:         self.resize_event()
61:         if DEBUG: print('FigureCanvasGTKAgg.configure_event end')
62:         return True
63: 
64:     def _render_figure(self, pixmap, width, height):
65:         if DEBUG: print('FigureCanvasGTKAgg.render_figure')
66:         FigureCanvasAgg.draw(self)
67:         if DEBUG: print('FigureCanvasGTKAgg.render_figure pixmap', pixmap)
68:         #agg_to_gtk_drawable(pixmap, self.renderer._renderer, None)
69: 
70:         buf = self.buffer_rgba()
71:         ren = self.get_renderer()
72:         w = int(ren.width)
73:         h = int(ren.height)
74: 
75:         pixbuf = gtk.gdk.pixbuf_new_from_data(
76:             buf, gtk.gdk.COLORSPACE_RGB,  True, 8, w, h, w*4)
77:         pixmap.draw_pixbuf(pixmap.new_gc(), pixbuf, 0, 0, 0, 0, w, h,
78:                            gtk.gdk.RGB_DITHER_NONE, 0, 0)
79:         if DEBUG: print('FigureCanvasGTKAgg.render_figure done')
80: 
81:     def blit(self, bbox=None):
82:         if DEBUG: print('FigureCanvasGTKAgg.blit', self._pixmap)
83:         agg_to_gtk_drawable(self._pixmap, self.renderer._renderer, bbox)
84: 
85:         x, y, w, h = self.allocation
86: 
87:         self.window.draw_drawable (self.style.fg_gc[self.state], self._pixmap,
88:                                    0, 0, 0, 0, w, h)
89:         if DEBUG: print('FigureCanvasGTKAgg.done')
90: 
91:     def print_png(self, filename, *args, **kwargs):
92:         # Do this so we can save the resolution of figure in the PNG file
93:         agg = self.switch_backends(FigureCanvasAgg)
94:         return agg.print_png(filename, *args, **kwargs)
95: 
96: 
97: @_BackendGTK.export
98: class _BackendGTKAgg(_BackendGTK):
99:     FigureCanvas = FigureCanvasGTKAgg
100:     FigureManager = FigureManagerGTKAgg
101: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_229914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nRender to gtk from agg\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import six' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229915 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six')

if (type(import_229915) is not StypyTypeError):

    if (import_229915 != 'pyd_module'):
        __import__(import_229915)
        sys_modules_229916 = sys.modules[import_229915]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', sys_modules_229916.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', import_229915)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import os' statement (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import matplotlib' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229917 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib')

if (type(import_229917) is not StypyTypeError):

    if (import_229917 != 'pyd_module'):
        __import__(import_229917)
        sys_modules_229918 = sys.modules[import_229917]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib', sys_modules_229918.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib', import_229917)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from matplotlib.figure import Figure' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229919 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.figure')

if (type(import_229919) is not StypyTypeError):

    if (import_229919 != 'pyd_module'):
        __import__(import_229919)
        sys_modules_229920 = sys.modules[import_229919]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.figure', sys_modules_229920.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_229920, sys_modules_229920.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.figure', import_229919)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from matplotlib.backends.backend_agg import FigureCanvasAgg' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229921 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.backends.backend_agg')

if (type(import_229921) is not StypyTypeError):

    if (import_229921 != 'pyd_module'):
        __import__(import_229921)
        sys_modules_229922 = sys.modules[import_229921]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.backends.backend_agg', sys_modules_229922.module_type_store, module_type_store, ['FigureCanvasAgg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_229922, sys_modules_229922.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.backends.backend_agg', None, module_type_store, ['FigureCanvasAgg'], [FigureCanvasAgg])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_agg' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.backends.backend_agg', import_229921)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from matplotlib.backends.backend_gtk import gtk, _BackendGTK, FigureCanvasGTK, FigureManagerGTK, NavigationToolbar2GTK, backend_version, error_msg_gtk, PIXELS_PER_INCH' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229923 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.backends.backend_gtk')

if (type(import_229923) is not StypyTypeError):

    if (import_229923 != 'pyd_module'):
        __import__(import_229923)
        sys_modules_229924 = sys.modules[import_229923]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.backends.backend_gtk', sys_modules_229924.module_type_store, module_type_store, ['gtk', '_BackendGTK', 'FigureCanvasGTK', 'FigureManagerGTK', 'NavigationToolbar2GTK', 'backend_version', 'error_msg_gtk', 'PIXELS_PER_INCH'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_229924, sys_modules_229924.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_gtk import gtk, _BackendGTK, FigureCanvasGTK, FigureManagerGTK, NavigationToolbar2GTK, backend_version, error_msg_gtk, PIXELS_PER_INCH

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.backends.backend_gtk', None, module_type_store, ['gtk', '_BackendGTK', 'FigureCanvasGTK', 'FigureManagerGTK', 'NavigationToolbar2GTK', 'backend_version', 'error_msg_gtk', 'PIXELS_PER_INCH'], [gtk, _BackendGTK, FigureCanvasGTK, FigureManagerGTK, NavigationToolbar2GTK, backend_version, error_msg_gtk, PIXELS_PER_INCH])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_gtk' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.backends.backend_gtk', import_229923)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from matplotlib.backends._gtkagg import agg_to_gtk_drawable' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229925 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.backends._gtkagg')

if (type(import_229925) is not StypyTypeError):

    if (import_229925 != 'pyd_module'):
        __import__(import_229925)
        sys_modules_229926 = sys.modules[import_229925]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.backends._gtkagg', sys_modules_229926.module_type_store, module_type_store, ['agg_to_gtk_drawable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_229926, sys_modules_229926.module_type_store, module_type_store)
    else:
        from matplotlib.backends._gtkagg import agg_to_gtk_drawable

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.backends._gtkagg', None, module_type_store, ['agg_to_gtk_drawable'], [agg_to_gtk_drawable])

else:
    # Assigning a type to the variable 'matplotlib.backends._gtkagg' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.backends._gtkagg', import_229925)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a Name to a Name (line 20):

# Assigning a Name to a Name (line 20):
# Getting the type of 'False' (line 20)
False_229927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'False')
# Assigning a type to the variable 'DEBUG' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'DEBUG', False_229927)
# Declaration of the 'NavigationToolbar2GTKAgg' class
# Getting the type of 'NavigationToolbar2GTK' (line 22)
NavigationToolbar2GTK_229928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 31), 'NavigationToolbar2GTK')

class NavigationToolbar2GTKAgg(NavigationToolbar2GTK_229928, ):

    @norecursion
    def _get_canvas(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_canvas'
        module_type_store = module_type_store.open_function_context('_get_canvas', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2GTKAgg._get_canvas.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2GTKAgg._get_canvas.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2GTKAgg._get_canvas.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2GTKAgg._get_canvas.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2GTKAgg._get_canvas')
        NavigationToolbar2GTKAgg._get_canvas.__dict__.__setitem__('stypy_param_names_list', ['fig'])
        NavigationToolbar2GTKAgg._get_canvas.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2GTKAgg._get_canvas.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2GTKAgg._get_canvas.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2GTKAgg._get_canvas.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2GTKAgg._get_canvas.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2GTKAgg._get_canvas.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2GTKAgg._get_canvas', ['fig'], None, None, defaults, varargs, kwargs)

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

        
        # Call to FigureCanvasGTKAgg(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'fig' (line 24)
        fig_229930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 34), 'fig', False)
        # Processing the call keyword arguments (line 24)
        kwargs_229931 = {}
        # Getting the type of 'FigureCanvasGTKAgg' (line 24)
        FigureCanvasGTKAgg_229929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 15), 'FigureCanvasGTKAgg', False)
        # Calling FigureCanvasGTKAgg(args, kwargs) (line 24)
        FigureCanvasGTKAgg_call_result_229932 = invoke(stypy.reporting.localization.Localization(__file__, 24, 15), FigureCanvasGTKAgg_229929, *[fig_229930], **kwargs_229931)
        
        # Assigning a type to the variable 'stypy_return_type' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'stypy_return_type', FigureCanvasGTKAgg_call_result_229932)
        
        # ################# End of '_get_canvas(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_canvas' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_229933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229933)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_canvas'
        return stypy_return_type_229933


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 22, 0, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2GTKAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NavigationToolbar2GTKAgg' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'NavigationToolbar2GTKAgg', NavigationToolbar2GTKAgg)
# Declaration of the 'FigureManagerGTKAgg' class
# Getting the type of 'FigureManagerGTK' (line 27)
FigureManagerGTK_229934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 26), 'FigureManagerGTK')

class FigureManagerGTKAgg(FigureManagerGTK_229934, ):

    @norecursion
    def _get_toolbar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_toolbar'
        module_type_store = module_type_store.open_function_context('_get_toolbar', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerGTKAgg._get_toolbar.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerGTKAgg._get_toolbar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerGTKAgg._get_toolbar.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerGTKAgg._get_toolbar.__dict__.__setitem__('stypy_function_name', 'FigureManagerGTKAgg._get_toolbar')
        FigureManagerGTKAgg._get_toolbar.__dict__.__setitem__('stypy_param_names_list', ['canvas'])
        FigureManagerGTKAgg._get_toolbar.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerGTKAgg._get_toolbar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerGTKAgg._get_toolbar.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerGTKAgg._get_toolbar.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerGTKAgg._get_toolbar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerGTKAgg._get_toolbar.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTKAgg._get_toolbar', ['canvas'], None, None, defaults, varargs, kwargs)

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

        
        
        
        # Obtaining the type of the subscript
        unicode_229935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'unicode', u'toolbar')
        # Getting the type of 'matplotlib' (line 31)
        matplotlib_229936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'matplotlib')
        # Obtaining the member 'rcParams' of a type (line 31)
        rcParams_229937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), matplotlib_229936, 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___229938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), rcParams_229937, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_229939 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), getitem___229938, unicode_229935)
        
        unicode_229940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 43), 'unicode', u'toolbar2')
        # Applying the binary operator '==' (line 31)
        result_eq_229941 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 11), '==', subscript_call_result_229939, unicode_229940)
        
        # Testing the type of an if condition (line 31)
        if_condition_229942 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 8), result_eq_229941)
        # Assigning a type to the variable 'if_condition_229942' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'if_condition_229942', if_condition_229942)
        # SSA begins for if statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 32):
        
        # Assigning a Call to a Name (line 32):
        
        # Call to NavigationToolbar2GTKAgg(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'canvas' (line 32)
        canvas_229944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 48), 'canvas', False)
        # Getting the type of 'self' (line 32)
        self_229945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 56), 'self', False)
        # Obtaining the member 'window' of a type (line 32)
        window_229946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 56), self_229945, 'window')
        # Processing the call keyword arguments (line 32)
        kwargs_229947 = {}
        # Getting the type of 'NavigationToolbar2GTKAgg' (line 32)
        NavigationToolbar2GTKAgg_229943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'NavigationToolbar2GTKAgg', False)
        # Calling NavigationToolbar2GTKAgg(args, kwargs) (line 32)
        NavigationToolbar2GTKAgg_call_result_229948 = invoke(stypy.reporting.localization.Localization(__file__, 32, 22), NavigationToolbar2GTKAgg_229943, *[canvas_229944, window_229946], **kwargs_229947)
        
        # Assigning a type to the variable 'toolbar' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'toolbar', NavigationToolbar2GTKAgg_call_result_229948)
        # SSA branch for the else part of an if statement (line 31)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 34):
        
        # Assigning a Name to a Name (line 34):
        # Getting the type of 'None' (line 34)
        None_229949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 22), 'None')
        # Assigning a type to the variable 'toolbar' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'toolbar', None_229949)
        # SSA join for if statement (line 31)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'toolbar' (line 35)
        toolbar_229950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'toolbar')
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', toolbar_229950)
        
        # ################# End of '_get_toolbar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_toolbar' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_229951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229951)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_toolbar'
        return stypy_return_type_229951


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 27, 0, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTKAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureManagerGTKAgg' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'FigureManagerGTKAgg', FigureManagerGTKAgg)
# Declaration of the 'FigureCanvasGTKAgg' class
# Getting the type of 'FigureCanvasGTK' (line 38)
FigureCanvasGTK_229952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'FigureCanvasGTK')
# Getting the type of 'FigureCanvasAgg' (line 38)
FigureCanvasAgg_229953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 42), 'FigureCanvasAgg')

class FigureCanvasGTKAgg(FigureCanvasGTK_229952, FigureCanvasAgg_229953, ):

    @norecursion
    def configure_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 42)
        None_229954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 44), 'None')
        defaults = [None_229954]
        # Create a new context for function 'configure_event'
        module_type_store = module_type_store.open_function_context('configure_event', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTKAgg.configure_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTKAgg.configure_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTKAgg.configure_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTKAgg.configure_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTKAgg.configure_event')
        FigureCanvasGTKAgg.configure_event.__dict__.__setitem__('stypy_param_names_list', ['widget', 'event'])
        FigureCanvasGTKAgg.configure_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTKAgg.configure_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTKAgg.configure_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTKAgg.configure_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTKAgg.configure_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTKAgg.configure_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTKAgg.configure_event', ['widget', 'event'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'DEBUG' (line 44)
        DEBUG_229955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'DEBUG')
        # Testing the type of an if condition (line 44)
        if_condition_229956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 8), DEBUG_229955)
        # Assigning a type to the variable 'if_condition_229956' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'if_condition_229956', if_condition_229956)
        # SSA begins for if statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 44)
        # Processing the call arguments (line 44)
        unicode_229958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'unicode', u'FigureCanvasGTKAgg.configure_event')
        # Processing the call keyword arguments (line 44)
        kwargs_229959 = {}
        # Getting the type of 'print' (line 44)
        print_229957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 18), 'print', False)
        # Calling print(args, kwargs) (line 44)
        print_call_result_229960 = invoke(stypy.reporting.localization.Localization(__file__, 44, 18), print_229957, *[unicode_229958], **kwargs_229959)
        
        # SSA join for if statement (line 44)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 45)
        # Getting the type of 'widget' (line 45)
        widget_229961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'widget')
        # Obtaining the member 'window' of a type (line 45)
        window_229962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 11), widget_229961, 'window')
        # Getting the type of 'None' (line 45)
        None_229963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 28), 'None')
        
        (may_be_229964, more_types_in_union_229965) = may_be_none(window_229962, None_229963)

        if may_be_229964:

            if more_types_in_union_229965:
                # Runtime conditional SSA (line 45)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_229965:
                # SSA join for if statement (line 45)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # SSA begins for try-except statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Deleting a member
        # Getting the type of 'self' (line 48)
        self_229966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'self')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 48, 12), self_229966, 'renderer')
        # SSA branch for the except part of a try statement (line 47)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 47)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 51):
        
        # Assigning a Call to a Name:
        
        # Call to get_size(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_229970 = {}
        # Getting the type of 'widget' (line 51)
        widget_229967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'widget', False)
        # Obtaining the member 'window' of a type (line 51)
        window_229968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 14), widget_229967, 'window')
        # Obtaining the member 'get_size' of a type (line 51)
        get_size_229969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 14), window_229968, 'get_size')
        # Calling get_size(args, kwargs) (line 51)
        get_size_call_result_229971 = invoke(stypy.reporting.localization.Localization(__file__, 51, 14), get_size_229969, *[], **kwargs_229970)
        
        # Assigning a type to the variable 'call_assignment_229907' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'call_assignment_229907', get_size_call_result_229971)
        
        # Assigning a Call to a Name (line 51):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_229974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 8), 'int')
        # Processing the call keyword arguments
        kwargs_229975 = {}
        # Getting the type of 'call_assignment_229907' (line 51)
        call_assignment_229907_229972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'call_assignment_229907', False)
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___229973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), call_assignment_229907_229972, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_229976 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___229973, *[int_229974], **kwargs_229975)
        
        # Assigning a type to the variable 'call_assignment_229908' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'call_assignment_229908', getitem___call_result_229976)
        
        # Assigning a Name to a Name (line 51):
        # Getting the type of 'call_assignment_229908' (line 51)
        call_assignment_229908_229977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'call_assignment_229908')
        # Assigning a type to the variable 'w' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'w', call_assignment_229908_229977)
        
        # Assigning a Call to a Name (line 51):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_229980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 8), 'int')
        # Processing the call keyword arguments
        kwargs_229981 = {}
        # Getting the type of 'call_assignment_229907' (line 51)
        call_assignment_229907_229978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'call_assignment_229907', False)
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___229979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), call_assignment_229907_229978, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_229982 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___229979, *[int_229980], **kwargs_229981)
        
        # Assigning a type to the variable 'call_assignment_229909' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'call_assignment_229909', getitem___call_result_229982)
        
        # Assigning a Name to a Name (line 51):
        # Getting the type of 'call_assignment_229909' (line 51)
        call_assignment_229909_229983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'call_assignment_229909')
        # Assigning a type to the variable 'h' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 10), 'h', call_assignment_229909_229983)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'w' (line 52)
        w_229984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'w')
        int_229985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 14), 'int')
        # Applying the binary operator '==' (line 52)
        result_eq_229986 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 11), '==', w_229984, int_229985)
        
        
        # Getting the type of 'h' (line 52)
        h_229987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'h')
        int_229988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 22), 'int')
        # Applying the binary operator '==' (line 52)
        result_eq_229989 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 19), '==', h_229987, int_229988)
        
        # Applying the binary operator 'or' (line 52)
        result_or_keyword_229990 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 11), 'or', result_eq_229986, result_eq_229989)
        
        # Testing the type of an if condition (line 52)
        if_condition_229991 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 8), result_or_keyword_229990)
        # Assigning a type to the variable 'if_condition_229991' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'if_condition_229991', if_condition_229991)
        # SSA begins for if statement (line 52)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 25), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 52)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 55):
        
        # Assigning a Attribute to a Name (line 55):
        # Getting the type of 'self' (line 55)
        self_229992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'self')
        # Obtaining the member 'figure' of a type (line 55)
        figure_229993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 17), self_229992, 'figure')
        # Obtaining the member 'dpi' of a type (line 55)
        dpi_229994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 17), figure_229993, 'dpi')
        # Assigning a type to the variable 'dpival' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'dpival', dpi_229994)
        
        # Assigning a BinOp to a Name (line 56):
        
        # Assigning a BinOp to a Name (line 56):
        # Getting the type of 'w' (line 56)
        w_229995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'w')
        # Getting the type of 'dpival' (line 56)
        dpival_229996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'dpival')
        # Applying the binary operator 'div' (line 56)
        result_div_229997 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 16), 'div', w_229995, dpival_229996)
        
        # Assigning a type to the variable 'winch' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'winch', result_div_229997)
        
        # Assigning a BinOp to a Name (line 57):
        
        # Assigning a BinOp to a Name (line 57):
        # Getting the type of 'h' (line 57)
        h_229998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'h')
        # Getting the type of 'dpival' (line 57)
        dpival_229999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'dpival')
        # Applying the binary operator 'div' (line 57)
        result_div_230000 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 16), 'div', h_229998, dpival_229999)
        
        # Assigning a type to the variable 'hinch' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'hinch', result_div_230000)
        
        # Call to set_size_inches(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'winch' (line 58)
        winch_230004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 36), 'winch', False)
        # Getting the type of 'hinch' (line 58)
        hinch_230005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 43), 'hinch', False)
        # Processing the call keyword arguments (line 58)
        # Getting the type of 'False' (line 58)
        False_230006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 58), 'False', False)
        keyword_230007 = False_230006
        kwargs_230008 = {'forward': keyword_230007}
        # Getting the type of 'self' (line 58)
        self_230001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 58)
        figure_230002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_230001, 'figure')
        # Obtaining the member 'set_size_inches' of a type (line 58)
        set_size_inches_230003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), figure_230002, 'set_size_inches')
        # Calling set_size_inches(args, kwargs) (line 58)
        set_size_inches_call_result_230009 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), set_size_inches_230003, *[winch_230004, hinch_230005], **kwargs_230008)
        
        
        # Assigning a Name to a Attribute (line 59):
        
        # Assigning a Name to a Attribute (line 59):
        # Getting the type of 'True' (line 59)
        True_230010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 28), 'True')
        # Getting the type of 'self' (line 59)
        self_230011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member '_need_redraw' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_230011, '_need_redraw', True_230010)
        
        # Call to resize_event(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_230014 = {}
        # Getting the type of 'self' (line 60)
        self_230012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self', False)
        # Obtaining the member 'resize_event' of a type (line 60)
        resize_event_230013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_230012, 'resize_event')
        # Calling resize_event(args, kwargs) (line 60)
        resize_event_call_result_230015 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), resize_event_230013, *[], **kwargs_230014)
        
        
        # Getting the type of 'DEBUG' (line 61)
        DEBUG_230016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'DEBUG')
        # Testing the type of an if condition (line 61)
        if_condition_230017 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 8), DEBUG_230016)
        # Assigning a type to the variable 'if_condition_230017' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'if_condition_230017', if_condition_230017)
        # SSA begins for if statement (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 61)
        # Processing the call arguments (line 61)
        unicode_230019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 24), 'unicode', u'FigureCanvasGTKAgg.configure_event end')
        # Processing the call keyword arguments (line 61)
        kwargs_230020 = {}
        # Getting the type of 'print' (line 61)
        print_230018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'print', False)
        # Calling print(args, kwargs) (line 61)
        print_call_result_230021 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), print_230018, *[unicode_230019], **kwargs_230020)
        
        # SSA join for if statement (line 61)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'True' (line 62)
        True_230022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', True_230022)
        
        # ################# End of 'configure_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'configure_event' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_230023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230023)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'configure_event'
        return stypy_return_type_230023


    @norecursion
    def _render_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_render_figure'
        module_type_store = module_type_store.open_function_context('_render_figure', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTKAgg._render_figure.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTKAgg._render_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTKAgg._render_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTKAgg._render_figure.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTKAgg._render_figure')
        FigureCanvasGTKAgg._render_figure.__dict__.__setitem__('stypy_param_names_list', ['pixmap', 'width', 'height'])
        FigureCanvasGTKAgg._render_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTKAgg._render_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTKAgg._render_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTKAgg._render_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTKAgg._render_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTKAgg._render_figure.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTKAgg._render_figure', ['pixmap', 'width', 'height'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_render_figure', localization, ['pixmap', 'width', 'height'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_render_figure(...)' code ##################

        
        # Getting the type of 'DEBUG' (line 65)
        DEBUG_230024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'DEBUG')
        # Testing the type of an if condition (line 65)
        if_condition_230025 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 8), DEBUG_230024)
        # Assigning a type to the variable 'if_condition_230025' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'if_condition_230025', if_condition_230025)
        # SSA begins for if statement (line 65)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 65)
        # Processing the call arguments (line 65)
        unicode_230027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'unicode', u'FigureCanvasGTKAgg.render_figure')
        # Processing the call keyword arguments (line 65)
        kwargs_230028 = {}
        # Getting the type of 'print' (line 65)
        print_230026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 18), 'print', False)
        # Calling print(args, kwargs) (line 65)
        print_call_result_230029 = invoke(stypy.reporting.localization.Localization(__file__, 65, 18), print_230026, *[unicode_230027], **kwargs_230028)
        
        # SSA join for if statement (line 65)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'self' (line 66)
        self_230032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'self', False)
        # Processing the call keyword arguments (line 66)
        kwargs_230033 = {}
        # Getting the type of 'FigureCanvasAgg' (line 66)
        FigureCanvasAgg_230030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'FigureCanvasAgg', False)
        # Obtaining the member 'draw' of a type (line 66)
        draw_230031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), FigureCanvasAgg_230030, 'draw')
        # Calling draw(args, kwargs) (line 66)
        draw_call_result_230034 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), draw_230031, *[self_230032], **kwargs_230033)
        
        
        # Getting the type of 'DEBUG' (line 67)
        DEBUG_230035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'DEBUG')
        # Testing the type of an if condition (line 67)
        if_condition_230036 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), DEBUG_230035)
        # Assigning a type to the variable 'if_condition_230036' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_230036', if_condition_230036)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 67)
        # Processing the call arguments (line 67)
        unicode_230038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 24), 'unicode', u'FigureCanvasGTKAgg.render_figure pixmap')
        # Getting the type of 'pixmap' (line 67)
        pixmap_230039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 67), 'pixmap', False)
        # Processing the call keyword arguments (line 67)
        kwargs_230040 = {}
        # Getting the type of 'print' (line 67)
        print_230037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'print', False)
        # Calling print(args, kwargs) (line 67)
        print_call_result_230041 = invoke(stypy.reporting.localization.Localization(__file__, 67, 18), print_230037, *[unicode_230038, pixmap_230039], **kwargs_230040)
        
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 70):
        
        # Assigning a Call to a Name (line 70):
        
        # Call to buffer_rgba(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_230044 = {}
        # Getting the type of 'self' (line 70)
        self_230042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), 'self', False)
        # Obtaining the member 'buffer_rgba' of a type (line 70)
        buffer_rgba_230043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 14), self_230042, 'buffer_rgba')
        # Calling buffer_rgba(args, kwargs) (line 70)
        buffer_rgba_call_result_230045 = invoke(stypy.reporting.localization.Localization(__file__, 70, 14), buffer_rgba_230043, *[], **kwargs_230044)
        
        # Assigning a type to the variable 'buf' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'buf', buffer_rgba_call_result_230045)
        
        # Assigning a Call to a Name (line 71):
        
        # Assigning a Call to a Name (line 71):
        
        # Call to get_renderer(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_230048 = {}
        # Getting the type of 'self' (line 71)
        self_230046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'self', False)
        # Obtaining the member 'get_renderer' of a type (line 71)
        get_renderer_230047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 14), self_230046, 'get_renderer')
        # Calling get_renderer(args, kwargs) (line 71)
        get_renderer_call_result_230049 = invoke(stypy.reporting.localization.Localization(__file__, 71, 14), get_renderer_230047, *[], **kwargs_230048)
        
        # Assigning a type to the variable 'ren' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'ren', get_renderer_call_result_230049)
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to int(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'ren' (line 72)
        ren_230051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'ren', False)
        # Obtaining the member 'width' of a type (line 72)
        width_230052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 16), ren_230051, 'width')
        # Processing the call keyword arguments (line 72)
        kwargs_230053 = {}
        # Getting the type of 'int' (line 72)
        int_230050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'int', False)
        # Calling int(args, kwargs) (line 72)
        int_call_result_230054 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), int_230050, *[width_230052], **kwargs_230053)
        
        # Assigning a type to the variable 'w' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'w', int_call_result_230054)
        
        # Assigning a Call to a Name (line 73):
        
        # Assigning a Call to a Name (line 73):
        
        # Call to int(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'ren' (line 73)
        ren_230056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'ren', False)
        # Obtaining the member 'height' of a type (line 73)
        height_230057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), ren_230056, 'height')
        # Processing the call keyword arguments (line 73)
        kwargs_230058 = {}
        # Getting the type of 'int' (line 73)
        int_230055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'int', False)
        # Calling int(args, kwargs) (line 73)
        int_call_result_230059 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), int_230055, *[height_230057], **kwargs_230058)
        
        # Assigning a type to the variable 'h' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'h', int_call_result_230059)
        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to pixbuf_new_from_data(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'buf' (line 76)
        buf_230063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'buf', False)
        # Getting the type of 'gtk' (line 76)
        gtk_230064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'gtk', False)
        # Obtaining the member 'gdk' of a type (line 76)
        gdk_230065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 17), gtk_230064, 'gdk')
        # Obtaining the member 'COLORSPACE_RGB' of a type (line 76)
        COLORSPACE_RGB_230066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 17), gdk_230065, 'COLORSPACE_RGB')
        # Getting the type of 'True' (line 76)
        True_230067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 42), 'True', False)
        int_230068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 48), 'int')
        # Getting the type of 'w' (line 76)
        w_230069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 51), 'w', False)
        # Getting the type of 'h' (line 76)
        h_230070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 54), 'h', False)
        # Getting the type of 'w' (line 76)
        w_230071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 57), 'w', False)
        int_230072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 59), 'int')
        # Applying the binary operator '*' (line 76)
        result_mul_230073 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 57), '*', w_230071, int_230072)
        
        # Processing the call keyword arguments (line 75)
        kwargs_230074 = {}
        # Getting the type of 'gtk' (line 75)
        gtk_230060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'gtk', False)
        # Obtaining the member 'gdk' of a type (line 75)
        gdk_230061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 17), gtk_230060, 'gdk')
        # Obtaining the member 'pixbuf_new_from_data' of a type (line 75)
        pixbuf_new_from_data_230062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 17), gdk_230061, 'pixbuf_new_from_data')
        # Calling pixbuf_new_from_data(args, kwargs) (line 75)
        pixbuf_new_from_data_call_result_230075 = invoke(stypy.reporting.localization.Localization(__file__, 75, 17), pixbuf_new_from_data_230062, *[buf_230063, COLORSPACE_RGB_230066, True_230067, int_230068, w_230069, h_230070, result_mul_230073], **kwargs_230074)
        
        # Assigning a type to the variable 'pixbuf' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'pixbuf', pixbuf_new_from_data_call_result_230075)
        
        # Call to draw_pixbuf(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Call to new_gc(...): (line 77)
        # Processing the call keyword arguments (line 77)
        kwargs_230080 = {}
        # Getting the type of 'pixmap' (line 77)
        pixmap_230078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'pixmap', False)
        # Obtaining the member 'new_gc' of a type (line 77)
        new_gc_230079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 27), pixmap_230078, 'new_gc')
        # Calling new_gc(args, kwargs) (line 77)
        new_gc_call_result_230081 = invoke(stypy.reporting.localization.Localization(__file__, 77, 27), new_gc_230079, *[], **kwargs_230080)
        
        # Getting the type of 'pixbuf' (line 77)
        pixbuf_230082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 44), 'pixbuf', False)
        int_230083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 52), 'int')
        int_230084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 55), 'int')
        int_230085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 58), 'int')
        int_230086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 61), 'int')
        # Getting the type of 'w' (line 77)
        w_230087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 64), 'w', False)
        # Getting the type of 'h' (line 77)
        h_230088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 67), 'h', False)
        # Getting the type of 'gtk' (line 78)
        gtk_230089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 27), 'gtk', False)
        # Obtaining the member 'gdk' of a type (line 78)
        gdk_230090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 27), gtk_230089, 'gdk')
        # Obtaining the member 'RGB_DITHER_NONE' of a type (line 78)
        RGB_DITHER_NONE_230091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 27), gdk_230090, 'RGB_DITHER_NONE')
        int_230092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 52), 'int')
        int_230093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 55), 'int')
        # Processing the call keyword arguments (line 77)
        kwargs_230094 = {}
        # Getting the type of 'pixmap' (line 77)
        pixmap_230076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'pixmap', False)
        # Obtaining the member 'draw_pixbuf' of a type (line 77)
        draw_pixbuf_230077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), pixmap_230076, 'draw_pixbuf')
        # Calling draw_pixbuf(args, kwargs) (line 77)
        draw_pixbuf_call_result_230095 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), draw_pixbuf_230077, *[new_gc_call_result_230081, pixbuf_230082, int_230083, int_230084, int_230085, int_230086, w_230087, h_230088, RGB_DITHER_NONE_230091, int_230092, int_230093], **kwargs_230094)
        
        
        # Getting the type of 'DEBUG' (line 79)
        DEBUG_230096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'DEBUG')
        # Testing the type of an if condition (line 79)
        if_condition_230097 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), DEBUG_230096)
        # Assigning a type to the variable 'if_condition_230097' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_230097', if_condition_230097)
        # SSA begins for if statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 79)
        # Processing the call arguments (line 79)
        unicode_230099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 24), 'unicode', u'FigureCanvasGTKAgg.render_figure done')
        # Processing the call keyword arguments (line 79)
        kwargs_230100 = {}
        # Getting the type of 'print' (line 79)
        print_230098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'print', False)
        # Calling print(args, kwargs) (line 79)
        print_call_result_230101 = invoke(stypy.reporting.localization.Localization(__file__, 79, 18), print_230098, *[unicode_230099], **kwargs_230100)
        
        # SSA join for if statement (line 79)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_render_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_render_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_230102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230102)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_render_figure'
        return stypy_return_type_230102


    @norecursion
    def blit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 81)
        None_230103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 24), 'None')
        defaults = [None_230103]
        # Create a new context for function 'blit'
        module_type_store = module_type_store.open_function_context('blit', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTKAgg.blit.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTKAgg.blit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTKAgg.blit.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTKAgg.blit.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTKAgg.blit')
        FigureCanvasGTKAgg.blit.__dict__.__setitem__('stypy_param_names_list', ['bbox'])
        FigureCanvasGTKAgg.blit.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTKAgg.blit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTKAgg.blit.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTKAgg.blit.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTKAgg.blit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTKAgg.blit.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTKAgg.blit', ['bbox'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'DEBUG' (line 82)
        DEBUG_230104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'DEBUG')
        # Testing the type of an if condition (line 82)
        if_condition_230105 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), DEBUG_230104)
        # Assigning a type to the variable 'if_condition_230105' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'if_condition_230105', if_condition_230105)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 82)
        # Processing the call arguments (line 82)
        unicode_230107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 24), 'unicode', u'FigureCanvasGTKAgg.blit')
        # Getting the type of 'self' (line 82)
        self_230108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'self', False)
        # Obtaining the member '_pixmap' of a type (line 82)
        _pixmap_230109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 51), self_230108, '_pixmap')
        # Processing the call keyword arguments (line 82)
        kwargs_230110 = {}
        # Getting the type of 'print' (line 82)
        print_230106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'print', False)
        # Calling print(args, kwargs) (line 82)
        print_call_result_230111 = invoke(stypy.reporting.localization.Localization(__file__, 82, 18), print_230106, *[unicode_230107, _pixmap_230109], **kwargs_230110)
        
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to agg_to_gtk_drawable(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'self' (line 83)
        self_230113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'self', False)
        # Obtaining the member '_pixmap' of a type (line 83)
        _pixmap_230114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 28), self_230113, '_pixmap')
        # Getting the type of 'self' (line 83)
        self_230115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 42), 'self', False)
        # Obtaining the member 'renderer' of a type (line 83)
        renderer_230116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 42), self_230115, 'renderer')
        # Obtaining the member '_renderer' of a type (line 83)
        _renderer_230117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 42), renderer_230116, '_renderer')
        # Getting the type of 'bbox' (line 83)
        bbox_230118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 67), 'bbox', False)
        # Processing the call keyword arguments (line 83)
        kwargs_230119 = {}
        # Getting the type of 'agg_to_gtk_drawable' (line 83)
        agg_to_gtk_drawable_230112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'agg_to_gtk_drawable', False)
        # Calling agg_to_gtk_drawable(args, kwargs) (line 83)
        agg_to_gtk_drawable_call_result_230120 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), agg_to_gtk_drawable_230112, *[_pixmap_230114, _renderer_230117, bbox_230118], **kwargs_230119)
        
        
        # Assigning a Attribute to a Tuple (line 85):
        
        # Assigning a Subscript to a Name (line 85):
        
        # Obtaining the type of the subscript
        int_230121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
        # Getting the type of 'self' (line 85)
        self_230122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'self')
        # Obtaining the member 'allocation' of a type (line 85)
        allocation_230123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 21), self_230122, 'allocation')
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___230124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), allocation_230123, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_230125 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), getitem___230124, int_230121)
        
        # Assigning a type to the variable 'tuple_var_assignment_229910' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_229910', subscript_call_result_230125)
        
        # Assigning a Subscript to a Name (line 85):
        
        # Obtaining the type of the subscript
        int_230126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
        # Getting the type of 'self' (line 85)
        self_230127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'self')
        # Obtaining the member 'allocation' of a type (line 85)
        allocation_230128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 21), self_230127, 'allocation')
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___230129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), allocation_230128, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_230130 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), getitem___230129, int_230126)
        
        # Assigning a type to the variable 'tuple_var_assignment_229911' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_229911', subscript_call_result_230130)
        
        # Assigning a Subscript to a Name (line 85):
        
        # Obtaining the type of the subscript
        int_230131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
        # Getting the type of 'self' (line 85)
        self_230132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'self')
        # Obtaining the member 'allocation' of a type (line 85)
        allocation_230133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 21), self_230132, 'allocation')
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___230134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), allocation_230133, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_230135 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), getitem___230134, int_230131)
        
        # Assigning a type to the variable 'tuple_var_assignment_229912' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_229912', subscript_call_result_230135)
        
        # Assigning a Subscript to a Name (line 85):
        
        # Obtaining the type of the subscript
        int_230136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
        # Getting the type of 'self' (line 85)
        self_230137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'self')
        # Obtaining the member 'allocation' of a type (line 85)
        allocation_230138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 21), self_230137, 'allocation')
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___230139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), allocation_230138, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_230140 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), getitem___230139, int_230136)
        
        # Assigning a type to the variable 'tuple_var_assignment_229913' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_229913', subscript_call_result_230140)
        
        # Assigning a Name to a Name (line 85):
        # Getting the type of 'tuple_var_assignment_229910' (line 85)
        tuple_var_assignment_229910_230141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_229910')
        # Assigning a type to the variable 'x' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'x', tuple_var_assignment_229910_230141)
        
        # Assigning a Name to a Name (line 85):
        # Getting the type of 'tuple_var_assignment_229911' (line 85)
        tuple_var_assignment_229911_230142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_229911')
        # Assigning a type to the variable 'y' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'y', tuple_var_assignment_229911_230142)
        
        # Assigning a Name to a Name (line 85):
        # Getting the type of 'tuple_var_assignment_229912' (line 85)
        tuple_var_assignment_229912_230143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_229912')
        # Assigning a type to the variable 'w' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 14), 'w', tuple_var_assignment_229912_230143)
        
        # Assigning a Name to a Name (line 85):
        # Getting the type of 'tuple_var_assignment_229913' (line 85)
        tuple_var_assignment_229913_230144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_229913')
        # Assigning a type to the variable 'h' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'h', tuple_var_assignment_229913_230144)
        
        # Call to draw_drawable(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 87)
        self_230148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 52), 'self', False)
        # Obtaining the member 'state' of a type (line 87)
        state_230149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 52), self_230148, 'state')
        # Getting the type of 'self' (line 87)
        self_230150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 35), 'self', False)
        # Obtaining the member 'style' of a type (line 87)
        style_230151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 35), self_230150, 'style')
        # Obtaining the member 'fg_gc' of a type (line 87)
        fg_gc_230152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 35), style_230151, 'fg_gc')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___230153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 35), fg_gc_230152, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_230154 = invoke(stypy.reporting.localization.Localization(__file__, 87, 35), getitem___230153, state_230149)
        
        # Getting the type of 'self' (line 87)
        self_230155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 65), 'self', False)
        # Obtaining the member '_pixmap' of a type (line 87)
        _pixmap_230156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 65), self_230155, '_pixmap')
        int_230157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 35), 'int')
        int_230158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 38), 'int')
        int_230159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 41), 'int')
        int_230160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 44), 'int')
        # Getting the type of 'w' (line 88)
        w_230161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 47), 'w', False)
        # Getting the type of 'h' (line 88)
        h_230162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 50), 'h', False)
        # Processing the call keyword arguments (line 87)
        kwargs_230163 = {}
        # Getting the type of 'self' (line 87)
        self_230145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 87)
        window_230146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_230145, 'window')
        # Obtaining the member 'draw_drawable' of a type (line 87)
        draw_drawable_230147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), window_230146, 'draw_drawable')
        # Calling draw_drawable(args, kwargs) (line 87)
        draw_drawable_call_result_230164 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), draw_drawable_230147, *[subscript_call_result_230154, _pixmap_230156, int_230157, int_230158, int_230159, int_230160, w_230161, h_230162], **kwargs_230163)
        
        
        # Getting the type of 'DEBUG' (line 89)
        DEBUG_230165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'DEBUG')
        # Testing the type of an if condition (line 89)
        if_condition_230166 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 8), DEBUG_230165)
        # Assigning a type to the variable 'if_condition_230166' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'if_condition_230166', if_condition_230166)
        # SSA begins for if statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 89)
        # Processing the call arguments (line 89)
        unicode_230168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 24), 'unicode', u'FigureCanvasGTKAgg.done')
        # Processing the call keyword arguments (line 89)
        kwargs_230169 = {}
        # Getting the type of 'print' (line 89)
        print_230167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'print', False)
        # Calling print(args, kwargs) (line 89)
        print_call_result_230170 = invoke(stypy.reporting.localization.Localization(__file__, 89, 18), print_230167, *[unicode_230168], **kwargs_230169)
        
        # SSA join for if statement (line 89)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'blit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'blit' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_230171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230171)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'blit'
        return stypy_return_type_230171


    @norecursion
    def print_png(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_png'
        module_type_store = module_type_store.open_function_context('print_png', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTKAgg.print_png.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTKAgg.print_png.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTKAgg.print_png.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTKAgg.print_png.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTKAgg.print_png')
        FigureCanvasGTKAgg.print_png.__dict__.__setitem__('stypy_param_names_list', ['filename'])
        FigureCanvasGTKAgg.print_png.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasGTKAgg.print_png.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasGTKAgg.print_png.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTKAgg.print_png.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTKAgg.print_png.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTKAgg.print_png.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTKAgg.print_png', ['filename'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_png', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_png(...)' code ##################

        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Call to switch_backends(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'FigureCanvasAgg' (line 93)
        FigureCanvasAgg_230174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 35), 'FigureCanvasAgg', False)
        # Processing the call keyword arguments (line 93)
        kwargs_230175 = {}
        # Getting the type of 'self' (line 93)
        self_230172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 14), 'self', False)
        # Obtaining the member 'switch_backends' of a type (line 93)
        switch_backends_230173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 14), self_230172, 'switch_backends')
        # Calling switch_backends(args, kwargs) (line 93)
        switch_backends_call_result_230176 = invoke(stypy.reporting.localization.Localization(__file__, 93, 14), switch_backends_230173, *[FigureCanvasAgg_230174], **kwargs_230175)
        
        # Assigning a type to the variable 'agg' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'agg', switch_backends_call_result_230176)
        
        # Call to print_png(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'filename' (line 94)
        filename_230179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 29), 'filename', False)
        # Getting the type of 'args' (line 94)
        args_230180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 40), 'args', False)
        # Processing the call keyword arguments (line 94)
        # Getting the type of 'kwargs' (line 94)
        kwargs_230181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 48), 'kwargs', False)
        kwargs_230182 = {'kwargs_230181': kwargs_230181}
        # Getting the type of 'agg' (line 94)
        agg_230177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'agg', False)
        # Obtaining the member 'print_png' of a type (line 94)
        print_png_230178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 15), agg_230177, 'print_png')
        # Calling print_png(args, kwargs) (line 94)
        print_png_call_result_230183 = invoke(stypy.reporting.localization.Localization(__file__, 94, 15), print_png_230178, *[filename_230179, args_230180], **kwargs_230182)
        
        # Assigning a type to the variable 'stypy_return_type' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'stypy_return_type', print_png_call_result_230183)
        
        # ################# End of 'print_png(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_png' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_230184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230184)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_png'
        return stypy_return_type_230184


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 38, 0, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTKAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureCanvasGTKAgg' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'FigureCanvasGTKAgg', FigureCanvasGTKAgg)

# Assigning a Call to a Name (line 39):

# Call to copy(...): (line 39)
# Processing the call keyword arguments (line 39)
kwargs_230188 = {}
# Getting the type of 'FigureCanvasGTK' (line 39)
FigureCanvasGTK_230185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'FigureCanvasGTK', False)
# Obtaining the member 'filetypes' of a type (line 39)
filetypes_230186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 16), FigureCanvasGTK_230185, 'filetypes')
# Obtaining the member 'copy' of a type (line 39)
copy_230187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 16), filetypes_230186, 'copy')
# Calling copy(args, kwargs) (line 39)
copy_call_result_230189 = invoke(stypy.reporting.localization.Localization(__file__, 39, 16), copy_230187, *[], **kwargs_230188)

# Getting the type of 'FigureCanvasGTKAgg'
FigureCanvasGTKAgg_230190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasGTKAgg')
# Setting the type of the member 'filetypes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasGTKAgg_230190, 'filetypes', copy_call_result_230189)

# Assigning a Call to a Name (line 39):

# Call to update(...): (line 40)
# Processing the call arguments (line 40)
# Getting the type of 'FigureCanvasAgg' (line 40)
FigureCanvasAgg_230194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 21), 'FigureCanvasAgg', False)
# Obtaining the member 'filetypes' of a type (line 40)
filetypes_230195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 21), FigureCanvasAgg_230194, 'filetypes')
# Processing the call keyword arguments (line 40)
kwargs_230196 = {}
# Getting the type of 'FigureCanvasGTKAgg'
FigureCanvasGTKAgg_230191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasGTKAgg', False)
# Obtaining the member 'filetypes' of a type
filetypes_230192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasGTKAgg_230191, 'filetypes')
# Obtaining the member 'update' of a type (line 40)
update_230193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 4), filetypes_230192, 'update')
# Calling update(args, kwargs) (line 40)
update_call_result_230197 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), update_230193, *[filetypes_230195], **kwargs_230196)

# Declaration of the '_BackendGTKAgg' class
# Getting the type of '_BackendGTK' (line 98)
_BackendGTK_230198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), '_BackendGTK')

class _BackendGTKAgg(_BackendGTK_230198, ):
    
    # Assigning a Name to a Name (line 99):
    
    # Assigning a Name to a Name (line 100):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 97, 0, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendGTKAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendGTKAgg' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), '_BackendGTKAgg', _BackendGTKAgg)

# Assigning a Name to a Name (line 99):
# Getting the type of 'FigureCanvasGTKAgg' (line 99)
FigureCanvasGTKAgg_230199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'FigureCanvasGTKAgg')
# Getting the type of '_BackendGTKAgg'
_BackendGTKAgg_230200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendGTKAgg')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendGTKAgg_230200, 'FigureCanvas', FigureCanvasGTKAgg_230199)

# Assigning a Name to a Name (line 100):
# Getting the type of 'FigureManagerGTKAgg' (line 100)
FigureManagerGTKAgg_230201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'FigureManagerGTKAgg')
# Getting the type of '_BackendGTKAgg'
_BackendGTKAgg_230202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendGTKAgg')
# Setting the type of the member 'FigureManager' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendGTKAgg_230202, 'FigureManager', FigureManagerGTKAgg_230201)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
