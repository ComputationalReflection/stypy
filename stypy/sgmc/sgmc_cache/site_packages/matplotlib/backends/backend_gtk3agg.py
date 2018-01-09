
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import numpy as np
7: import warnings
8: 
9: from . import backend_agg, backend_gtk3
10: from .backend_cairo import cairo, HAS_CAIRO_CFFI
11: from .backend_gtk3 import _BackendGTK3
12: from matplotlib.figure import Figure
13: from matplotlib import transforms
14: 
15: if six.PY3 and not HAS_CAIRO_CFFI:
16:     warnings.warn(
17:         "The Gtk3Agg backend is known to not work on Python 3.x with pycairo. "
18:         "Try installing cairocffi.")
19: 
20: 
21: class FigureCanvasGTK3Agg(backend_gtk3.FigureCanvasGTK3,
22:                           backend_agg.FigureCanvasAgg):
23:     def __init__(self, figure):
24:         backend_gtk3.FigureCanvasGTK3.__init__(self, figure)
25:         self._bbox_queue = []
26: 
27:     def _renderer_init(self):
28:         pass
29: 
30:     def _render_figure(self, width, height):
31:         backend_agg.FigureCanvasAgg.draw(self)
32: 
33:     def on_draw_event(self, widget, ctx):
34:         ''' GtkDrawable draw event, like expose_event in GTK 2.X
35:         '''
36:         allocation = self.get_allocation()
37:         w, h = allocation.width, allocation.height
38: 
39:         if not len(self._bbox_queue):
40:             self._render_figure(w, h)
41:             bbox_queue = [transforms.Bbox([[0, 0], [w, h]])]
42:         else:
43:             bbox_queue = self._bbox_queue
44: 
45:         if HAS_CAIRO_CFFI:
46:             ctx = cairo.Context._from_pointer(
47:                 cairo.ffi.cast('cairo_t **',
48:                                id(ctx) + object.__basicsize__)[0],
49:                 incref=True)
50: 
51:         for bbox in bbox_queue:
52:             area = self.copy_from_bbox(bbox)
53:             buf = np.fromstring(area.to_string_argb(), dtype='uint8')
54: 
55:             x = int(bbox.x0)
56:             y = h - int(bbox.y1)
57:             width = int(bbox.x1) - int(bbox.x0)
58:             height = int(bbox.y1) - int(bbox.y0)
59: 
60:             if HAS_CAIRO_CFFI:
61:                 image = cairo.ImageSurface.create_for_data(
62:                     buf.data, cairo.FORMAT_ARGB32, width, height)
63:             else:
64:                 image = cairo.ImageSurface.create_for_data(
65:                     buf, cairo.FORMAT_ARGB32, width, height)
66:             ctx.set_source_surface(image, x, y)
67:             ctx.paint()
68: 
69:         if len(self._bbox_queue):
70:             self._bbox_queue = []
71: 
72:         return False
73: 
74:     def blit(self, bbox=None):
75:         # If bbox is None, blit the entire canvas to gtk. Otherwise
76:         # blit only the area defined by the bbox.
77:         if bbox is None:
78:             bbox = self.figure.bbox
79: 
80:         allocation = self.get_allocation()
81:         w, h = allocation.width, allocation.height
82:         x = int(bbox.x0)
83:         y = h - int(bbox.y1)
84:         width = int(bbox.x1) - int(bbox.x0)
85:         height = int(bbox.y1) - int(bbox.y0)
86: 
87:         self._bbox_queue.append(bbox)
88:         self.queue_draw_area(x, y, width, height)
89: 
90:     def print_png(self, filename, *args, **kwargs):
91:         # Do this so we can save the resolution of figure in the PNG file
92:         agg = self.switch_backends(backend_agg.FigureCanvasAgg)
93:         return agg.print_png(filename, *args, **kwargs)
94: 
95: 
96: class FigureManagerGTK3Agg(backend_gtk3.FigureManagerGTK3):
97:     pass
98: 
99: 
100: @_BackendGTK3.export
101: class _BackendGTK3Cairo(_BackendGTK3):
102:     FigureCanvas = FigureCanvasGTK3Agg
103:     FigureManager = FigureManagerGTK3Agg
104: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229484 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_229484) is not StypyTypeError):

    if (import_229484 != 'pyd_module'):
        __import__(import_229484)
        sys_modules_229485 = sys.modules[import_229484]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_229485.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_229484)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229486 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_229486) is not StypyTypeError):

    if (import_229486 != 'pyd_module'):
        __import__(import_229486)
        sys_modules_229487 = sys.modules[import_229486]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_229487.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_229486)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import warnings' statement (line 7)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from matplotlib.backends import backend_agg, backend_gtk3' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229488 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends')

if (type(import_229488) is not StypyTypeError):

    if (import_229488 != 'pyd_module'):
        __import__(import_229488)
        sys_modules_229489 = sys.modules[import_229488]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends', sys_modules_229489.module_type_store, module_type_store, ['backend_agg', 'backend_gtk3'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_229489, sys_modules_229489.module_type_store, module_type_store)
    else:
        from matplotlib.backends import backend_agg, backend_gtk3

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends', None, module_type_store, ['backend_agg', 'backend_gtk3'], [backend_agg, backend_gtk3])

else:
    # Assigning a type to the variable 'matplotlib.backends' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends', import_229488)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from matplotlib.backends.backend_cairo import cairo, HAS_CAIRO_CFFI' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229490 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.backends.backend_cairo')

if (type(import_229490) is not StypyTypeError):

    if (import_229490 != 'pyd_module'):
        __import__(import_229490)
        sys_modules_229491 = sys.modules[import_229490]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.backends.backend_cairo', sys_modules_229491.module_type_store, module_type_store, ['cairo', 'HAS_CAIRO_CFFI'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_229491, sys_modules_229491.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_cairo import cairo, HAS_CAIRO_CFFI

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.backends.backend_cairo', None, module_type_store, ['cairo', 'HAS_CAIRO_CFFI'], [cairo, HAS_CAIRO_CFFI])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_cairo' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.backends.backend_cairo', import_229490)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from matplotlib.backends.backend_gtk3 import _BackendGTK3' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229492 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.backends.backend_gtk3')

if (type(import_229492) is not StypyTypeError):

    if (import_229492 != 'pyd_module'):
        __import__(import_229492)
        sys_modules_229493 = sys.modules[import_229492]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.backends.backend_gtk3', sys_modules_229493.module_type_store, module_type_store, ['_BackendGTK3'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_229493, sys_modules_229493.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_gtk3 import _BackendGTK3

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.backends.backend_gtk3', None, module_type_store, ['_BackendGTK3'], [_BackendGTK3])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_gtk3' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.backends.backend_gtk3', import_229492)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from matplotlib.figure import Figure' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229494 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.figure')

if (type(import_229494) is not StypyTypeError):

    if (import_229494 != 'pyd_module'):
        __import__(import_229494)
        sys_modules_229495 = sys.modules[import_229494]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.figure', sys_modules_229495.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_229495, sys_modules_229495.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.figure', import_229494)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from matplotlib import transforms' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229496 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib')

if (type(import_229496) is not StypyTypeError):

    if (import_229496 != 'pyd_module'):
        __import__(import_229496)
        sys_modules_229497 = sys.modules[import_229496]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib', sys_modules_229497.module_type_store, module_type_store, ['transforms'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_229497, sys_modules_229497.module_type_store, module_type_store)
    else:
        from matplotlib import transforms

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib', None, module_type_store, ['transforms'], [transforms])

else:
    # Assigning a type to the variable 'matplotlib' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib', import_229496)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')



# Evaluating a boolean operation
# Getting the type of 'six' (line 15)
six_229498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 3), 'six')
# Obtaining the member 'PY3' of a type (line 15)
PY3_229499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 3), six_229498, 'PY3')

# Getting the type of 'HAS_CAIRO_CFFI' (line 15)
HAS_CAIRO_CFFI_229500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'HAS_CAIRO_CFFI')
# Applying the 'not' unary operator (line 15)
result_not__229501 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 15), 'not', HAS_CAIRO_CFFI_229500)

# Applying the binary operator 'and' (line 15)
result_and_keyword_229502 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 3), 'and', PY3_229499, result_not__229501)

# Testing the type of an if condition (line 15)
if_condition_229503 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 0), result_and_keyword_229502)
# Assigning a type to the variable 'if_condition_229503' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'if_condition_229503', if_condition_229503)
# SSA begins for if statement (line 15)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to warn(...): (line 16)
# Processing the call arguments (line 16)
unicode_229506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'unicode', u'The Gtk3Agg backend is known to not work on Python 3.x with pycairo. Try installing cairocffi.')
# Processing the call keyword arguments (line 16)
kwargs_229507 = {}
# Getting the type of 'warnings' (line 16)
warnings_229504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'warnings', False)
# Obtaining the member 'warn' of a type (line 16)
warn_229505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), warnings_229504, 'warn')
# Calling warn(args, kwargs) (line 16)
warn_call_result_229508 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), warn_229505, *[unicode_229506], **kwargs_229507)

# SSA join for if statement (line 15)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'FigureCanvasGTK3Agg' class
# Getting the type of 'backend_gtk3' (line 21)
backend_gtk3_229509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 26), 'backend_gtk3')
# Obtaining the member 'FigureCanvasGTK3' of a type (line 21)
FigureCanvasGTK3_229510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 26), backend_gtk3_229509, 'FigureCanvasGTK3')
# Getting the type of 'backend_agg' (line 22)
backend_agg_229511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 26), 'backend_agg')
# Obtaining the member 'FigureCanvasAgg' of a type (line 22)
FigureCanvasAgg_229512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 26), backend_agg_229511, 'FigureCanvasAgg')

class FigureCanvasGTK3Agg(FigureCanvasGTK3_229510, FigureCanvasAgg_229512, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3Agg.__init__', ['figure'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'self' (line 24)
        self_229516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 47), 'self', False)
        # Getting the type of 'figure' (line 24)
        figure_229517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 53), 'figure', False)
        # Processing the call keyword arguments (line 24)
        kwargs_229518 = {}
        # Getting the type of 'backend_gtk3' (line 24)
        backend_gtk3_229513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'backend_gtk3', False)
        # Obtaining the member 'FigureCanvasGTK3' of a type (line 24)
        FigureCanvasGTK3_229514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), backend_gtk3_229513, 'FigureCanvasGTK3')
        # Obtaining the member '__init__' of a type (line 24)
        init___229515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), FigureCanvasGTK3_229514, '__init__')
        # Calling __init__(args, kwargs) (line 24)
        init___call_result_229519 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), init___229515, *[self_229516, figure_229517], **kwargs_229518)
        
        
        # Assigning a List to a Attribute (line 25):
        
        # Assigning a List to a Attribute (line 25):
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_229520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        
        # Getting the type of 'self' (line 25)
        self_229521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member '_bbox_queue' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_229521, '_bbox_queue', list_229520)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _renderer_init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_renderer_init'
        module_type_store = module_type_store.open_function_context('_renderer_init', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3Agg._renderer_init.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3Agg._renderer_init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3Agg._renderer_init.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3Agg._renderer_init.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3Agg._renderer_init')
        FigureCanvasGTK3Agg._renderer_init.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasGTK3Agg._renderer_init.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3Agg._renderer_init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3Agg._renderer_init.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3Agg._renderer_init.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3Agg._renderer_init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3Agg._renderer_init.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3Agg._renderer_init', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_renderer_init', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_renderer_init(...)' code ##################

        pass
        
        # ################# End of '_renderer_init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_renderer_init' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_229522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229522)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_renderer_init'
        return stypy_return_type_229522


    @norecursion
    def _render_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_render_figure'
        module_type_store = module_type_store.open_function_context('_render_figure', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3Agg._render_figure.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3Agg._render_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3Agg._render_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3Agg._render_figure.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3Agg._render_figure')
        FigureCanvasGTK3Agg._render_figure.__dict__.__setitem__('stypy_param_names_list', ['width', 'height'])
        FigureCanvasGTK3Agg._render_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3Agg._render_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3Agg._render_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3Agg._render_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3Agg._render_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3Agg._render_figure.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3Agg._render_figure', ['width', 'height'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_render_figure', localization, ['width', 'height'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_render_figure(...)' code ##################

        
        # Call to draw(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'self' (line 31)
        self_229526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 41), 'self', False)
        # Processing the call keyword arguments (line 31)
        kwargs_229527 = {}
        # Getting the type of 'backend_agg' (line 31)
        backend_agg_229523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'backend_agg', False)
        # Obtaining the member 'FigureCanvasAgg' of a type (line 31)
        FigureCanvasAgg_229524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), backend_agg_229523, 'FigureCanvasAgg')
        # Obtaining the member 'draw' of a type (line 31)
        draw_229525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), FigureCanvasAgg_229524, 'draw')
        # Calling draw(args, kwargs) (line 31)
        draw_call_result_229528 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), draw_229525, *[self_229526], **kwargs_229527)
        
        
        # ################# End of '_render_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_render_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_229529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229529)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_render_figure'
        return stypy_return_type_229529


    @norecursion
    def on_draw_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'on_draw_event'
        module_type_store = module_type_store.open_function_context('on_draw_event', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3Agg.on_draw_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3Agg.on_draw_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3Agg.on_draw_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3Agg.on_draw_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3Agg.on_draw_event')
        FigureCanvasGTK3Agg.on_draw_event.__dict__.__setitem__('stypy_param_names_list', ['widget', 'ctx'])
        FigureCanvasGTK3Agg.on_draw_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3Agg.on_draw_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3Agg.on_draw_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3Agg.on_draw_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3Agg.on_draw_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3Agg.on_draw_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3Agg.on_draw_event', ['widget', 'ctx'], None, None, defaults, varargs, kwargs)

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

        unicode_229530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'unicode', u' GtkDrawable draw event, like expose_event in GTK 2.X\n        ')
        
        # Assigning a Call to a Name (line 36):
        
        # Assigning a Call to a Name (line 36):
        
        # Call to get_allocation(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_229533 = {}
        # Getting the type of 'self' (line 36)
        self_229531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'self', False)
        # Obtaining the member 'get_allocation' of a type (line 36)
        get_allocation_229532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 21), self_229531, 'get_allocation')
        # Calling get_allocation(args, kwargs) (line 36)
        get_allocation_call_result_229534 = invoke(stypy.reporting.localization.Localization(__file__, 36, 21), get_allocation_229532, *[], **kwargs_229533)
        
        # Assigning a type to the variable 'allocation' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'allocation', get_allocation_call_result_229534)
        
        # Assigning a Tuple to a Tuple (line 37):
        
        # Assigning a Attribute to a Name (line 37):
        # Getting the type of 'allocation' (line 37)
        allocation_229535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'allocation')
        # Obtaining the member 'width' of a type (line 37)
        width_229536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), allocation_229535, 'width')
        # Assigning a type to the variable 'tuple_assignment_229480' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'tuple_assignment_229480', width_229536)
        
        # Assigning a Attribute to a Name (line 37):
        # Getting the type of 'allocation' (line 37)
        allocation_229537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 33), 'allocation')
        # Obtaining the member 'height' of a type (line 37)
        height_229538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 33), allocation_229537, 'height')
        # Assigning a type to the variable 'tuple_assignment_229481' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'tuple_assignment_229481', height_229538)
        
        # Assigning a Name to a Name (line 37):
        # Getting the type of 'tuple_assignment_229480' (line 37)
        tuple_assignment_229480_229539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'tuple_assignment_229480')
        # Assigning a type to the variable 'w' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'w', tuple_assignment_229480_229539)
        
        # Assigning a Name to a Name (line 37):
        # Getting the type of 'tuple_assignment_229481' (line 37)
        tuple_assignment_229481_229540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'tuple_assignment_229481')
        # Assigning a type to the variable 'h' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'h', tuple_assignment_229481_229540)
        
        
        
        # Call to len(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'self' (line 39)
        self_229542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'self', False)
        # Obtaining the member '_bbox_queue' of a type (line 39)
        _bbox_queue_229543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 19), self_229542, '_bbox_queue')
        # Processing the call keyword arguments (line 39)
        kwargs_229544 = {}
        # Getting the type of 'len' (line 39)
        len_229541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'len', False)
        # Calling len(args, kwargs) (line 39)
        len_call_result_229545 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), len_229541, *[_bbox_queue_229543], **kwargs_229544)
        
        # Applying the 'not' unary operator (line 39)
        result_not__229546 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 11), 'not', len_call_result_229545)
        
        # Testing the type of an if condition (line 39)
        if_condition_229547 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 8), result_not__229546)
        # Assigning a type to the variable 'if_condition_229547' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'if_condition_229547', if_condition_229547)
        # SSA begins for if statement (line 39)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _render_figure(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'w' (line 40)
        w_229550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 32), 'w', False)
        # Getting the type of 'h' (line 40)
        h_229551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'h', False)
        # Processing the call keyword arguments (line 40)
        kwargs_229552 = {}
        # Getting the type of 'self' (line 40)
        self_229548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'self', False)
        # Obtaining the member '_render_figure' of a type (line 40)
        _render_figure_229549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), self_229548, '_render_figure')
        # Calling _render_figure(args, kwargs) (line 40)
        _render_figure_call_result_229553 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), _render_figure_229549, *[w_229550, h_229551], **kwargs_229552)
        
        
        # Assigning a List to a Name (line 41):
        
        # Assigning a List to a Name (line 41):
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_229554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        
        # Call to Bbox(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_229557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_229558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        int_229559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 43), list_229558, int_229559)
        # Adding element type (line 41)
        int_229560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 43), list_229558, int_229560)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 42), list_229557, list_229558)
        # Adding element type (line 41)
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_229561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        # Getting the type of 'w' (line 41)
        w_229562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 52), 'w', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 51), list_229561, w_229562)
        # Adding element type (line 41)
        # Getting the type of 'h' (line 41)
        h_229563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 55), 'h', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 51), list_229561, h_229563)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 42), list_229557, list_229561)
        
        # Processing the call keyword arguments (line 41)
        kwargs_229564 = {}
        # Getting the type of 'transforms' (line 41)
        transforms_229555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 26), 'transforms', False)
        # Obtaining the member 'Bbox' of a type (line 41)
        Bbox_229556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 26), transforms_229555, 'Bbox')
        # Calling Bbox(args, kwargs) (line 41)
        Bbox_call_result_229565 = invoke(stypy.reporting.localization.Localization(__file__, 41, 26), Bbox_229556, *[list_229557], **kwargs_229564)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 25), list_229554, Bbox_call_result_229565)
        
        # Assigning a type to the variable 'bbox_queue' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'bbox_queue', list_229554)
        # SSA branch for the else part of an if statement (line 39)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 43):
        
        # Assigning a Attribute to a Name (line 43):
        # Getting the type of 'self' (line 43)
        self_229566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 25), 'self')
        # Obtaining the member '_bbox_queue' of a type (line 43)
        _bbox_queue_229567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 25), self_229566, '_bbox_queue')
        # Assigning a type to the variable 'bbox_queue' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'bbox_queue', _bbox_queue_229567)
        # SSA join for if statement (line 39)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'HAS_CAIRO_CFFI' (line 45)
        HAS_CAIRO_CFFI_229568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'HAS_CAIRO_CFFI')
        # Testing the type of an if condition (line 45)
        if_condition_229569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 8), HAS_CAIRO_CFFI_229568)
        # Assigning a type to the variable 'if_condition_229569' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'if_condition_229569', if_condition_229569)
        # SSA begins for if statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 46):
        
        # Assigning a Call to a Name (line 46):
        
        # Call to _from_pointer(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Obtaining the type of the subscript
        int_229573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 63), 'int')
        
        # Call to cast(...): (line 47)
        # Processing the call arguments (line 47)
        unicode_229577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 31), 'unicode', u'cairo_t **')
        
        # Call to id(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'ctx' (line 48)
        ctx_229579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 34), 'ctx', False)
        # Processing the call keyword arguments (line 48)
        kwargs_229580 = {}
        # Getting the type of 'id' (line 48)
        id_229578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'id', False)
        # Calling id(args, kwargs) (line 48)
        id_call_result_229581 = invoke(stypy.reporting.localization.Localization(__file__, 48, 31), id_229578, *[ctx_229579], **kwargs_229580)
        
        # Getting the type of 'object' (line 48)
        object_229582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 41), 'object', False)
        # Obtaining the member '__basicsize__' of a type (line 48)
        basicsize___229583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 41), object_229582, '__basicsize__')
        # Applying the binary operator '+' (line 48)
        result_add_229584 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 31), '+', id_call_result_229581, basicsize___229583)
        
        # Processing the call keyword arguments (line 47)
        kwargs_229585 = {}
        # Getting the type of 'cairo' (line 47)
        cairo_229574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'cairo', False)
        # Obtaining the member 'ffi' of a type (line 47)
        ffi_229575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), cairo_229574, 'ffi')
        # Obtaining the member 'cast' of a type (line 47)
        cast_229576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), ffi_229575, 'cast')
        # Calling cast(args, kwargs) (line 47)
        cast_call_result_229586 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), cast_229576, *[unicode_229577, result_add_229584], **kwargs_229585)
        
        # Obtaining the member '__getitem__' of a type (line 47)
        getitem___229587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), cast_call_result_229586, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 47)
        subscript_call_result_229588 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), getitem___229587, int_229573)
        
        # Processing the call keyword arguments (line 46)
        # Getting the type of 'True' (line 49)
        True_229589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'True', False)
        keyword_229590 = True_229589
        kwargs_229591 = {'incref': keyword_229590}
        # Getting the type of 'cairo' (line 46)
        cairo_229570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'cairo', False)
        # Obtaining the member 'Context' of a type (line 46)
        Context_229571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 18), cairo_229570, 'Context')
        # Obtaining the member '_from_pointer' of a type (line 46)
        _from_pointer_229572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 18), Context_229571, '_from_pointer')
        # Calling _from_pointer(args, kwargs) (line 46)
        _from_pointer_call_result_229592 = invoke(stypy.reporting.localization.Localization(__file__, 46, 18), _from_pointer_229572, *[subscript_call_result_229588], **kwargs_229591)
        
        # Assigning a type to the variable 'ctx' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'ctx', _from_pointer_call_result_229592)
        # SSA join for if statement (line 45)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'bbox_queue' (line 51)
        bbox_queue_229593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'bbox_queue')
        # Testing the type of a for loop iterable (line 51)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 51, 8), bbox_queue_229593)
        # Getting the type of the for loop variable (line 51)
        for_loop_var_229594 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 51, 8), bbox_queue_229593)
        # Assigning a type to the variable 'bbox' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'bbox', for_loop_var_229594)
        # SSA begins for a for statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to copy_from_bbox(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'bbox' (line 52)
        bbox_229597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 39), 'bbox', False)
        # Processing the call keyword arguments (line 52)
        kwargs_229598 = {}
        # Getting the type of 'self' (line 52)
        self_229595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'self', False)
        # Obtaining the member 'copy_from_bbox' of a type (line 52)
        copy_from_bbox_229596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 19), self_229595, 'copy_from_bbox')
        # Calling copy_from_bbox(args, kwargs) (line 52)
        copy_from_bbox_call_result_229599 = invoke(stypy.reporting.localization.Localization(__file__, 52, 19), copy_from_bbox_229596, *[bbox_229597], **kwargs_229598)
        
        # Assigning a type to the variable 'area' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'area', copy_from_bbox_call_result_229599)
        
        # Assigning a Call to a Name (line 53):
        
        # Assigning a Call to a Name (line 53):
        
        # Call to fromstring(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to to_string_argb(...): (line 53)
        # Processing the call keyword arguments (line 53)
        kwargs_229604 = {}
        # Getting the type of 'area' (line 53)
        area_229602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 32), 'area', False)
        # Obtaining the member 'to_string_argb' of a type (line 53)
        to_string_argb_229603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 32), area_229602, 'to_string_argb')
        # Calling to_string_argb(args, kwargs) (line 53)
        to_string_argb_call_result_229605 = invoke(stypy.reporting.localization.Localization(__file__, 53, 32), to_string_argb_229603, *[], **kwargs_229604)
        
        # Processing the call keyword arguments (line 53)
        unicode_229606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 61), 'unicode', u'uint8')
        keyword_229607 = unicode_229606
        kwargs_229608 = {'dtype': keyword_229607}
        # Getting the type of 'np' (line 53)
        np_229600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 18), 'np', False)
        # Obtaining the member 'fromstring' of a type (line 53)
        fromstring_229601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 18), np_229600, 'fromstring')
        # Calling fromstring(args, kwargs) (line 53)
        fromstring_call_result_229609 = invoke(stypy.reporting.localization.Localization(__file__, 53, 18), fromstring_229601, *[to_string_argb_call_result_229605], **kwargs_229608)
        
        # Assigning a type to the variable 'buf' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'buf', fromstring_call_result_229609)
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to int(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'bbox' (line 55)
        bbox_229611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'bbox', False)
        # Obtaining the member 'x0' of a type (line 55)
        x0_229612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 20), bbox_229611, 'x0')
        # Processing the call keyword arguments (line 55)
        kwargs_229613 = {}
        # Getting the type of 'int' (line 55)
        int_229610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'int', False)
        # Calling int(args, kwargs) (line 55)
        int_call_result_229614 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), int_229610, *[x0_229612], **kwargs_229613)
        
        # Assigning a type to the variable 'x' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'x', int_call_result_229614)
        
        # Assigning a BinOp to a Name (line 56):
        
        # Assigning a BinOp to a Name (line 56):
        # Getting the type of 'h' (line 56)
        h_229615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'h')
        
        # Call to int(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'bbox' (line 56)
        bbox_229617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'bbox', False)
        # Obtaining the member 'y1' of a type (line 56)
        y1_229618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 24), bbox_229617, 'y1')
        # Processing the call keyword arguments (line 56)
        kwargs_229619 = {}
        # Getting the type of 'int' (line 56)
        int_229616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'int', False)
        # Calling int(args, kwargs) (line 56)
        int_call_result_229620 = invoke(stypy.reporting.localization.Localization(__file__, 56, 20), int_229616, *[y1_229618], **kwargs_229619)
        
        # Applying the binary operator '-' (line 56)
        result_sub_229621 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 16), '-', h_229615, int_call_result_229620)
        
        # Assigning a type to the variable 'y' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'y', result_sub_229621)
        
        # Assigning a BinOp to a Name (line 57):
        
        # Assigning a BinOp to a Name (line 57):
        
        # Call to int(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'bbox' (line 57)
        bbox_229623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'bbox', False)
        # Obtaining the member 'x1' of a type (line 57)
        x1_229624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 24), bbox_229623, 'x1')
        # Processing the call keyword arguments (line 57)
        kwargs_229625 = {}
        # Getting the type of 'int' (line 57)
        int_229622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'int', False)
        # Calling int(args, kwargs) (line 57)
        int_call_result_229626 = invoke(stypy.reporting.localization.Localization(__file__, 57, 20), int_229622, *[x1_229624], **kwargs_229625)
        
        
        # Call to int(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'bbox' (line 57)
        bbox_229628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 39), 'bbox', False)
        # Obtaining the member 'x0' of a type (line 57)
        x0_229629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 39), bbox_229628, 'x0')
        # Processing the call keyword arguments (line 57)
        kwargs_229630 = {}
        # Getting the type of 'int' (line 57)
        int_229627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 35), 'int', False)
        # Calling int(args, kwargs) (line 57)
        int_call_result_229631 = invoke(stypy.reporting.localization.Localization(__file__, 57, 35), int_229627, *[x0_229629], **kwargs_229630)
        
        # Applying the binary operator '-' (line 57)
        result_sub_229632 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 20), '-', int_call_result_229626, int_call_result_229631)
        
        # Assigning a type to the variable 'width' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'width', result_sub_229632)
        
        # Assigning a BinOp to a Name (line 58):
        
        # Assigning a BinOp to a Name (line 58):
        
        # Call to int(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'bbox' (line 58)
        bbox_229634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 25), 'bbox', False)
        # Obtaining the member 'y1' of a type (line 58)
        y1_229635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 25), bbox_229634, 'y1')
        # Processing the call keyword arguments (line 58)
        kwargs_229636 = {}
        # Getting the type of 'int' (line 58)
        int_229633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'int', False)
        # Calling int(args, kwargs) (line 58)
        int_call_result_229637 = invoke(stypy.reporting.localization.Localization(__file__, 58, 21), int_229633, *[y1_229635], **kwargs_229636)
        
        
        # Call to int(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'bbox' (line 58)
        bbox_229639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 40), 'bbox', False)
        # Obtaining the member 'y0' of a type (line 58)
        y0_229640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 40), bbox_229639, 'y0')
        # Processing the call keyword arguments (line 58)
        kwargs_229641 = {}
        # Getting the type of 'int' (line 58)
        int_229638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 36), 'int', False)
        # Calling int(args, kwargs) (line 58)
        int_call_result_229642 = invoke(stypy.reporting.localization.Localization(__file__, 58, 36), int_229638, *[y0_229640], **kwargs_229641)
        
        # Applying the binary operator '-' (line 58)
        result_sub_229643 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 21), '-', int_call_result_229637, int_call_result_229642)
        
        # Assigning a type to the variable 'height' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'height', result_sub_229643)
        
        # Getting the type of 'HAS_CAIRO_CFFI' (line 60)
        HAS_CAIRO_CFFI_229644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'HAS_CAIRO_CFFI')
        # Testing the type of an if condition (line 60)
        if_condition_229645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 12), HAS_CAIRO_CFFI_229644)
        # Assigning a type to the variable 'if_condition_229645' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'if_condition_229645', if_condition_229645)
        # SSA begins for if statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 61):
        
        # Assigning a Call to a Name (line 61):
        
        # Call to create_for_data(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'buf' (line 62)
        buf_229649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'buf', False)
        # Obtaining the member 'data' of a type (line 62)
        data_229650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 20), buf_229649, 'data')
        # Getting the type of 'cairo' (line 62)
        cairo_229651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 30), 'cairo', False)
        # Obtaining the member 'FORMAT_ARGB32' of a type (line 62)
        FORMAT_ARGB32_229652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 30), cairo_229651, 'FORMAT_ARGB32')
        # Getting the type of 'width' (line 62)
        width_229653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 51), 'width', False)
        # Getting the type of 'height' (line 62)
        height_229654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 58), 'height', False)
        # Processing the call keyword arguments (line 61)
        kwargs_229655 = {}
        # Getting the type of 'cairo' (line 61)
        cairo_229646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'cairo', False)
        # Obtaining the member 'ImageSurface' of a type (line 61)
        ImageSurface_229647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), cairo_229646, 'ImageSurface')
        # Obtaining the member 'create_for_data' of a type (line 61)
        create_for_data_229648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), ImageSurface_229647, 'create_for_data')
        # Calling create_for_data(args, kwargs) (line 61)
        create_for_data_call_result_229656 = invoke(stypy.reporting.localization.Localization(__file__, 61, 24), create_for_data_229648, *[data_229650, FORMAT_ARGB32_229652, width_229653, height_229654], **kwargs_229655)
        
        # Assigning a type to the variable 'image' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'image', create_for_data_call_result_229656)
        # SSA branch for the else part of an if statement (line 60)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 64):
        
        # Assigning a Call to a Name (line 64):
        
        # Call to create_for_data(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'buf' (line 65)
        buf_229660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'buf', False)
        # Getting the type of 'cairo' (line 65)
        cairo_229661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'cairo', False)
        # Obtaining the member 'FORMAT_ARGB32' of a type (line 65)
        FORMAT_ARGB32_229662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 25), cairo_229661, 'FORMAT_ARGB32')
        # Getting the type of 'width' (line 65)
        width_229663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 46), 'width', False)
        # Getting the type of 'height' (line 65)
        height_229664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 53), 'height', False)
        # Processing the call keyword arguments (line 64)
        kwargs_229665 = {}
        # Getting the type of 'cairo' (line 64)
        cairo_229657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'cairo', False)
        # Obtaining the member 'ImageSurface' of a type (line 64)
        ImageSurface_229658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 24), cairo_229657, 'ImageSurface')
        # Obtaining the member 'create_for_data' of a type (line 64)
        create_for_data_229659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 24), ImageSurface_229658, 'create_for_data')
        # Calling create_for_data(args, kwargs) (line 64)
        create_for_data_call_result_229666 = invoke(stypy.reporting.localization.Localization(__file__, 64, 24), create_for_data_229659, *[buf_229660, FORMAT_ARGB32_229662, width_229663, height_229664], **kwargs_229665)
        
        # Assigning a type to the variable 'image' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'image', create_for_data_call_result_229666)
        # SSA join for if statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_source_surface(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'image' (line 66)
        image_229669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 35), 'image', False)
        # Getting the type of 'x' (line 66)
        x_229670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 42), 'x', False)
        # Getting the type of 'y' (line 66)
        y_229671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 45), 'y', False)
        # Processing the call keyword arguments (line 66)
        kwargs_229672 = {}
        # Getting the type of 'ctx' (line 66)
        ctx_229667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'ctx', False)
        # Obtaining the member 'set_source_surface' of a type (line 66)
        set_source_surface_229668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), ctx_229667, 'set_source_surface')
        # Calling set_source_surface(args, kwargs) (line 66)
        set_source_surface_call_result_229673 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), set_source_surface_229668, *[image_229669, x_229670, y_229671], **kwargs_229672)
        
        
        # Call to paint(...): (line 67)
        # Processing the call keyword arguments (line 67)
        kwargs_229676 = {}
        # Getting the type of 'ctx' (line 67)
        ctx_229674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'ctx', False)
        # Obtaining the member 'paint' of a type (line 67)
        paint_229675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), ctx_229674, 'paint')
        # Calling paint(args, kwargs) (line 67)
        paint_call_result_229677 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), paint_229675, *[], **kwargs_229676)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to len(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'self' (line 69)
        self_229679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'self', False)
        # Obtaining the member '_bbox_queue' of a type (line 69)
        _bbox_queue_229680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 15), self_229679, '_bbox_queue')
        # Processing the call keyword arguments (line 69)
        kwargs_229681 = {}
        # Getting the type of 'len' (line 69)
        len_229678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'len', False)
        # Calling len(args, kwargs) (line 69)
        len_call_result_229682 = invoke(stypy.reporting.localization.Localization(__file__, 69, 11), len_229678, *[_bbox_queue_229680], **kwargs_229681)
        
        # Testing the type of an if condition (line 69)
        if_condition_229683 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), len_call_result_229682)
        # Assigning a type to the variable 'if_condition_229683' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_229683', if_condition_229683)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Attribute (line 70):
        
        # Assigning a List to a Attribute (line 70):
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_229684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        
        # Getting the type of 'self' (line 70)
        self_229685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'self')
        # Setting the type of the member '_bbox_queue' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), self_229685, '_bbox_queue', list_229684)
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'False' (line 72)
        False_229686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', False_229686)
        
        # ################# End of 'on_draw_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'on_draw_event' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_229687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229687)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'on_draw_event'
        return stypy_return_type_229687


    @norecursion
    def blit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 74)
        None_229688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 24), 'None')
        defaults = [None_229688]
        # Create a new context for function 'blit'
        module_type_store = module_type_store.open_function_context('blit', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3Agg.blit.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3Agg.blit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3Agg.blit.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3Agg.blit.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3Agg.blit')
        FigureCanvasGTK3Agg.blit.__dict__.__setitem__('stypy_param_names_list', ['bbox'])
        FigureCanvasGTK3Agg.blit.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3Agg.blit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3Agg.blit.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3Agg.blit.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3Agg.blit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3Agg.blit.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3Agg.blit', ['bbox'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 77)
        # Getting the type of 'bbox' (line 77)
        bbox_229689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'bbox')
        # Getting the type of 'None' (line 77)
        None_229690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'None')
        
        (may_be_229691, more_types_in_union_229692) = may_be_none(bbox_229689, None_229690)

        if may_be_229691:

            if more_types_in_union_229692:
                # Runtime conditional SSA (line 77)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 78):
            
            # Assigning a Attribute to a Name (line 78):
            # Getting the type of 'self' (line 78)
            self_229693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'self')
            # Obtaining the member 'figure' of a type (line 78)
            figure_229694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 19), self_229693, 'figure')
            # Obtaining the member 'bbox' of a type (line 78)
            bbox_229695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 19), figure_229694, 'bbox')
            # Assigning a type to the variable 'bbox' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'bbox', bbox_229695)

            if more_types_in_union_229692:
                # SSA join for if statement (line 77)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to get_allocation(...): (line 80)
        # Processing the call keyword arguments (line 80)
        kwargs_229698 = {}
        # Getting the type of 'self' (line 80)
        self_229696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 21), 'self', False)
        # Obtaining the member 'get_allocation' of a type (line 80)
        get_allocation_229697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 21), self_229696, 'get_allocation')
        # Calling get_allocation(args, kwargs) (line 80)
        get_allocation_call_result_229699 = invoke(stypy.reporting.localization.Localization(__file__, 80, 21), get_allocation_229697, *[], **kwargs_229698)
        
        # Assigning a type to the variable 'allocation' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'allocation', get_allocation_call_result_229699)
        
        # Assigning a Tuple to a Tuple (line 81):
        
        # Assigning a Attribute to a Name (line 81):
        # Getting the type of 'allocation' (line 81)
        allocation_229700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'allocation')
        # Obtaining the member 'width' of a type (line 81)
        width_229701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 15), allocation_229700, 'width')
        # Assigning a type to the variable 'tuple_assignment_229482' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'tuple_assignment_229482', width_229701)
        
        # Assigning a Attribute to a Name (line 81):
        # Getting the type of 'allocation' (line 81)
        allocation_229702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 33), 'allocation')
        # Obtaining the member 'height' of a type (line 81)
        height_229703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 33), allocation_229702, 'height')
        # Assigning a type to the variable 'tuple_assignment_229483' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'tuple_assignment_229483', height_229703)
        
        # Assigning a Name to a Name (line 81):
        # Getting the type of 'tuple_assignment_229482' (line 81)
        tuple_assignment_229482_229704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'tuple_assignment_229482')
        # Assigning a type to the variable 'w' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'w', tuple_assignment_229482_229704)
        
        # Assigning a Name to a Name (line 81):
        # Getting the type of 'tuple_assignment_229483' (line 81)
        tuple_assignment_229483_229705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'tuple_assignment_229483')
        # Assigning a type to the variable 'h' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'h', tuple_assignment_229483_229705)
        
        # Assigning a Call to a Name (line 82):
        
        # Assigning a Call to a Name (line 82):
        
        # Call to int(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'bbox' (line 82)
        bbox_229707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'bbox', False)
        # Obtaining the member 'x0' of a type (line 82)
        x0_229708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 16), bbox_229707, 'x0')
        # Processing the call keyword arguments (line 82)
        kwargs_229709 = {}
        # Getting the type of 'int' (line 82)
        int_229706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'int', False)
        # Calling int(args, kwargs) (line 82)
        int_call_result_229710 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), int_229706, *[x0_229708], **kwargs_229709)
        
        # Assigning a type to the variable 'x' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'x', int_call_result_229710)
        
        # Assigning a BinOp to a Name (line 83):
        
        # Assigning a BinOp to a Name (line 83):
        # Getting the type of 'h' (line 83)
        h_229711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'h')
        
        # Call to int(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'bbox' (line 83)
        bbox_229713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'bbox', False)
        # Obtaining the member 'y1' of a type (line 83)
        y1_229714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), bbox_229713, 'y1')
        # Processing the call keyword arguments (line 83)
        kwargs_229715 = {}
        # Getting the type of 'int' (line 83)
        int_229712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'int', False)
        # Calling int(args, kwargs) (line 83)
        int_call_result_229716 = invoke(stypy.reporting.localization.Localization(__file__, 83, 16), int_229712, *[y1_229714], **kwargs_229715)
        
        # Applying the binary operator '-' (line 83)
        result_sub_229717 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 12), '-', h_229711, int_call_result_229716)
        
        # Assigning a type to the variable 'y' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'y', result_sub_229717)
        
        # Assigning a BinOp to a Name (line 84):
        
        # Assigning a BinOp to a Name (line 84):
        
        # Call to int(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'bbox' (line 84)
        bbox_229719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'bbox', False)
        # Obtaining the member 'x1' of a type (line 84)
        x1_229720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 20), bbox_229719, 'x1')
        # Processing the call keyword arguments (line 84)
        kwargs_229721 = {}
        # Getting the type of 'int' (line 84)
        int_229718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'int', False)
        # Calling int(args, kwargs) (line 84)
        int_call_result_229722 = invoke(stypy.reporting.localization.Localization(__file__, 84, 16), int_229718, *[x1_229720], **kwargs_229721)
        
        
        # Call to int(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'bbox' (line 84)
        bbox_229724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 35), 'bbox', False)
        # Obtaining the member 'x0' of a type (line 84)
        x0_229725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 35), bbox_229724, 'x0')
        # Processing the call keyword arguments (line 84)
        kwargs_229726 = {}
        # Getting the type of 'int' (line 84)
        int_229723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'int', False)
        # Calling int(args, kwargs) (line 84)
        int_call_result_229727 = invoke(stypy.reporting.localization.Localization(__file__, 84, 31), int_229723, *[x0_229725], **kwargs_229726)
        
        # Applying the binary operator '-' (line 84)
        result_sub_229728 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 16), '-', int_call_result_229722, int_call_result_229727)
        
        # Assigning a type to the variable 'width' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'width', result_sub_229728)
        
        # Assigning a BinOp to a Name (line 85):
        
        # Assigning a BinOp to a Name (line 85):
        
        # Call to int(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'bbox' (line 85)
        bbox_229730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'bbox', False)
        # Obtaining the member 'y1' of a type (line 85)
        y1_229731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 21), bbox_229730, 'y1')
        # Processing the call keyword arguments (line 85)
        kwargs_229732 = {}
        # Getting the type of 'int' (line 85)
        int_229729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'int', False)
        # Calling int(args, kwargs) (line 85)
        int_call_result_229733 = invoke(stypy.reporting.localization.Localization(__file__, 85, 17), int_229729, *[y1_229731], **kwargs_229732)
        
        
        # Call to int(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'bbox' (line 85)
        bbox_229735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 36), 'bbox', False)
        # Obtaining the member 'y0' of a type (line 85)
        y0_229736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 36), bbox_229735, 'y0')
        # Processing the call keyword arguments (line 85)
        kwargs_229737 = {}
        # Getting the type of 'int' (line 85)
        int_229734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 32), 'int', False)
        # Calling int(args, kwargs) (line 85)
        int_call_result_229738 = invoke(stypy.reporting.localization.Localization(__file__, 85, 32), int_229734, *[y0_229736], **kwargs_229737)
        
        # Applying the binary operator '-' (line 85)
        result_sub_229739 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 17), '-', int_call_result_229733, int_call_result_229738)
        
        # Assigning a type to the variable 'height' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'height', result_sub_229739)
        
        # Call to append(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'bbox' (line 87)
        bbox_229743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 32), 'bbox', False)
        # Processing the call keyword arguments (line 87)
        kwargs_229744 = {}
        # Getting the type of 'self' (line 87)
        self_229740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self', False)
        # Obtaining the member '_bbox_queue' of a type (line 87)
        _bbox_queue_229741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_229740, '_bbox_queue')
        # Obtaining the member 'append' of a type (line 87)
        append_229742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), _bbox_queue_229741, 'append')
        # Calling append(args, kwargs) (line 87)
        append_call_result_229745 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), append_229742, *[bbox_229743], **kwargs_229744)
        
        
        # Call to queue_draw_area(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'x' (line 88)
        x_229748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 29), 'x', False)
        # Getting the type of 'y' (line 88)
        y_229749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 32), 'y', False)
        # Getting the type of 'width' (line 88)
        width_229750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 35), 'width', False)
        # Getting the type of 'height' (line 88)
        height_229751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 42), 'height', False)
        # Processing the call keyword arguments (line 88)
        kwargs_229752 = {}
        # Getting the type of 'self' (line 88)
        self_229746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self', False)
        # Obtaining the member 'queue_draw_area' of a type (line 88)
        queue_draw_area_229747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_229746, 'queue_draw_area')
        # Calling queue_draw_area(args, kwargs) (line 88)
        queue_draw_area_call_result_229753 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), queue_draw_area_229747, *[x_229748, y_229749, width_229750, height_229751], **kwargs_229752)
        
        
        # ################# End of 'blit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'blit' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_229754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229754)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'blit'
        return stypy_return_type_229754


    @norecursion
    def print_png(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_png'
        module_type_store = module_type_store.open_function_context('print_png', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3Agg.print_png.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3Agg.print_png.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3Agg.print_png.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3Agg.print_png.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3Agg.print_png')
        FigureCanvasGTK3Agg.print_png.__dict__.__setitem__('stypy_param_names_list', ['filename'])
        FigureCanvasGTK3Agg.print_png.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasGTK3Agg.print_png.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasGTK3Agg.print_png.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3Agg.print_png.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3Agg.print_png.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3Agg.print_png.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3Agg.print_png', ['filename'], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 92):
        
        # Assigning a Call to a Name (line 92):
        
        # Call to switch_backends(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'backend_agg' (line 92)
        backend_agg_229757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 35), 'backend_agg', False)
        # Obtaining the member 'FigureCanvasAgg' of a type (line 92)
        FigureCanvasAgg_229758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 35), backend_agg_229757, 'FigureCanvasAgg')
        # Processing the call keyword arguments (line 92)
        kwargs_229759 = {}
        # Getting the type of 'self' (line 92)
        self_229755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'self', False)
        # Obtaining the member 'switch_backends' of a type (line 92)
        switch_backends_229756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 14), self_229755, 'switch_backends')
        # Calling switch_backends(args, kwargs) (line 92)
        switch_backends_call_result_229760 = invoke(stypy.reporting.localization.Localization(__file__, 92, 14), switch_backends_229756, *[FigureCanvasAgg_229758], **kwargs_229759)
        
        # Assigning a type to the variable 'agg' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'agg', switch_backends_call_result_229760)
        
        # Call to print_png(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'filename' (line 93)
        filename_229763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 29), 'filename', False)
        # Getting the type of 'args' (line 93)
        args_229764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 40), 'args', False)
        # Processing the call keyword arguments (line 93)
        # Getting the type of 'kwargs' (line 93)
        kwargs_229765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 48), 'kwargs', False)
        kwargs_229766 = {'kwargs_229765': kwargs_229765}
        # Getting the type of 'agg' (line 93)
        agg_229761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'agg', False)
        # Obtaining the member 'print_png' of a type (line 93)
        print_png_229762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 15), agg_229761, 'print_png')
        # Calling print_png(args, kwargs) (line 93)
        print_png_call_result_229767 = invoke(stypy.reporting.localization.Localization(__file__, 93, 15), print_png_229762, *[filename_229763, args_229764], **kwargs_229766)
        
        # Assigning a type to the variable 'stypy_return_type' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'stypy_return_type', print_png_call_result_229767)
        
        # ################# End of 'print_png(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_png' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_229768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229768)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_png'
        return stypy_return_type_229768


# Assigning a type to the variable 'FigureCanvasGTK3Agg' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'FigureCanvasGTK3Agg', FigureCanvasGTK3Agg)
# Declaration of the 'FigureManagerGTK3Agg' class
# Getting the type of 'backend_gtk3' (line 96)
backend_gtk3_229769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'backend_gtk3')
# Obtaining the member 'FigureManagerGTK3' of a type (line 96)
FigureManagerGTK3_229770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 27), backend_gtk3_229769, 'FigureManagerGTK3')

class FigureManagerGTK3Agg(FigureManagerGTK3_229770, ):
    pass

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTK3Agg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureManagerGTK3Agg' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'FigureManagerGTK3Agg', FigureManagerGTK3Agg)
# Declaration of the '_BackendGTK3Cairo' class
# Getting the type of '_BackendGTK3' (line 101)
_BackendGTK3_229771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), '_BackendGTK3')

class _BackendGTK3Cairo(_BackendGTK3_229771, ):
    
    # Assigning a Name to a Name (line 102):
    
    # Assigning a Name to a Name (line 103):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 100, 0, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendGTK3Cairo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendGTK3Cairo' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), '_BackendGTK3Cairo', _BackendGTK3Cairo)

# Assigning a Name to a Name (line 102):
# Getting the type of 'FigureCanvasGTK3Agg' (line 102)
FigureCanvasGTK3Agg_229772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'FigureCanvasGTK3Agg')
# Getting the type of '_BackendGTK3Cairo'
_BackendGTK3Cairo_229773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendGTK3Cairo')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendGTK3Cairo_229773, 'FigureCanvas', FigureCanvasGTK3Agg_229772)

# Assigning a Name to a Name (line 103):
# Getting the type of 'FigureManagerGTK3Agg' (line 103)
FigureManagerGTK3Agg_229774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'FigureManagerGTK3Agg')
# Getting the type of '_BackendGTK3Cairo'
_BackendGTK3Cairo_229775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendGTK3Cairo')
# Setting the type of the member 'FigureManager' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendGTK3Cairo_229775, 'FigureManager', FigureManagerGTK3Agg_229774)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
