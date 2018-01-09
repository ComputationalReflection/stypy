
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: GTK+ Matplotlib interface using cairo (not GDK) drawing operations.
3: Author: Steve Chaplin
4: '''
5: from __future__ import (absolute_import, division, print_function,
6:                         unicode_literals)
7: 
8: import six
9: 
10: import gtk
11: if gtk.pygtk_version < (2, 7, 0):
12:     import cairo.gtk
13: 
14: from matplotlib import cbook
15: from matplotlib.backends import backend_cairo
16: from matplotlib.backends.backend_gtk import *
17: from matplotlib.backends.backend_gtk import _BackendGTK
18: 
19: backend_version = ('PyGTK(%d.%d.%d) ' % gtk.pygtk_version
20:                    + 'Pycairo(%s)' % backend_cairo.backend_version)
21: 
22: 
23: class RendererGTKCairo (backend_cairo.RendererCairo):
24:     if gtk.pygtk_version >= (2,7,0):
25:         def set_pixmap (self, pixmap):
26:             self.gc.ctx = pixmap.cairo_create()
27:     else:
28:         def set_pixmap (self, pixmap):
29:             self.gc.ctx = cairo.gtk.gdk_cairo_create (pixmap)
30: 
31: 
32: class FigureCanvasGTKCairo(backend_cairo.FigureCanvasCairo, FigureCanvasGTK):
33:     filetypes = FigureCanvasGTK.filetypes.copy()
34:     filetypes.update(backend_cairo.FigureCanvasCairo.filetypes)
35: 
36:     def _renderer_init(self):
37:         '''Override to use cairo (rather than GDK) renderer'''
38:         self._renderer = RendererGTKCairo(self.figure.dpi)
39: 
40: 
41: # This class has been unused for a while at least.
42: @cbook.deprecated("2.1")
43: class FigureManagerGTKCairo(FigureManagerGTK):
44:     def _get_toolbar(self, canvas):
45:         # must be inited after the window, drawingArea and figure
46:         # attrs are set
47:         if matplotlib.rcParams['toolbar']=='toolbar2':
48:             toolbar = NavigationToolbar2GTKCairo (canvas, self.window)
49:         else:
50:             toolbar = None
51:         return toolbar
52: 
53: 
54: # This class has been unused for a while at least.
55: @cbook.deprecated("2.1")
56: class NavigationToolbar2Cairo(NavigationToolbar2GTK):
57:     def _get_canvas(self, fig):
58:         return FigureCanvasGTKCairo(fig)
59: 
60: 
61: @_BackendGTK.export
62: class _BackendGTKCairo(_BackendGTK):
63:     FigureCanvas = FigureCanvasGTKCairo
64:     FigureManager = FigureManagerGTK
65: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_230203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'unicode', u'\nGTK+ Matplotlib interface using cairo (not GDK) drawing operations.\nAuthor: Steve Chaplin\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import six' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230204 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six')

if (type(import_230204) is not StypyTypeError):

    if (import_230204 != 'pyd_module'):
        __import__(import_230204)
        sys_modules_230205 = sys.modules[import_230204]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', sys_modules_230205.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', import_230204)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import gtk' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230206 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'gtk')

if (type(import_230206) is not StypyTypeError):

    if (import_230206 != 'pyd_module'):
        __import__(import_230206)
        sys_modules_230207 = sys.modules[import_230206]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'gtk', sys_modules_230207.module_type_store, module_type_store)
    else:
        import gtk

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'gtk', gtk, module_type_store)

else:
    # Assigning a type to the variable 'gtk' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'gtk', import_230206)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')



# Getting the type of 'gtk' (line 11)
gtk_230208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 3), 'gtk')
# Obtaining the member 'pygtk_version' of a type (line 11)
pygtk_version_230209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 3), gtk_230208, 'pygtk_version')

# Obtaining an instance of the builtin type 'tuple' (line 11)
tuple_230210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 11)
# Adding element type (line 11)
int_230211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 24), tuple_230210, int_230211)
# Adding element type (line 11)
int_230212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 24), tuple_230210, int_230212)
# Adding element type (line 11)
int_230213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 24), tuple_230210, int_230213)

# Applying the binary operator '<' (line 11)
result_lt_230214 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 3), '<', pygtk_version_230209, tuple_230210)

# Testing the type of an if condition (line 11)
if_condition_230215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 11, 0), result_lt_230214)
# Assigning a type to the variable 'if_condition_230215' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'if_condition_230215', if_condition_230215)
# SSA begins for if statement (line 11)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 4))

# 'import cairo.gtk' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230216 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'cairo.gtk')

if (type(import_230216) is not StypyTypeError):

    if (import_230216 != 'pyd_module'):
        __import__(import_230216)
        sys_modules_230217 = sys.modules[import_230216]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'cairo.gtk', sys_modules_230217.module_type_store, module_type_store)
    else:
        import cairo.gtk

        import_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'cairo.gtk', cairo.gtk, module_type_store)

else:
    # Assigning a type to the variable 'cairo.gtk' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'cairo.gtk', import_230216)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# SSA join for if statement (line 11)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from matplotlib import cbook' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230218 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib')

if (type(import_230218) is not StypyTypeError):

    if (import_230218 != 'pyd_module'):
        __import__(import_230218)
        sys_modules_230219 = sys.modules[import_230218]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', sys_modules_230219.module_type_store, module_type_store, ['cbook'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_230219, sys_modules_230219.module_type_store, module_type_store)
    else:
        from matplotlib import cbook

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', None, module_type_store, ['cbook'], [cbook])

else:
    # Assigning a type to the variable 'matplotlib' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', import_230218)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib.backends import backend_cairo' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230220 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backends')

if (type(import_230220) is not StypyTypeError):

    if (import_230220 != 'pyd_module'):
        __import__(import_230220)
        sys_modules_230221 = sys.modules[import_230220]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backends', sys_modules_230221.module_type_store, module_type_store, ['backend_cairo'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_230221, sys_modules_230221.module_type_store, module_type_store)
    else:
        from matplotlib.backends import backend_cairo

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backends', None, module_type_store, ['backend_cairo'], [backend_cairo])

else:
    # Assigning a type to the variable 'matplotlib.backends' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backends', import_230220)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from matplotlib.backends.backend_gtk import ' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230222 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.backends.backend_gtk')

if (type(import_230222) is not StypyTypeError):

    if (import_230222 != 'pyd_module'):
        __import__(import_230222)
        sys_modules_230223 = sys.modules[import_230222]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.backends.backend_gtk', sys_modules_230223.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_230223, sys_modules_230223.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_gtk import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.backends.backend_gtk', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_gtk' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.backends.backend_gtk', import_230222)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from matplotlib.backends.backend_gtk import _BackendGTK' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230224 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.backends.backend_gtk')

if (type(import_230224) is not StypyTypeError):

    if (import_230224 != 'pyd_module'):
        __import__(import_230224)
        sys_modules_230225 = sys.modules[import_230224]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.backends.backend_gtk', sys_modules_230225.module_type_store, module_type_store, ['_BackendGTK'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_230225, sys_modules_230225.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_gtk import _BackendGTK

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.backends.backend_gtk', None, module_type_store, ['_BackendGTK'], [_BackendGTK])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_gtk' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.backends.backend_gtk', import_230224)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a BinOp to a Name (line 19):
unicode_230226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 19), 'unicode', u'PyGTK(%d.%d.%d) ')
# Getting the type of 'gtk' (line 19)
gtk_230227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 40), 'gtk')
# Obtaining the member 'pygtk_version' of a type (line 19)
pygtk_version_230228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 40), gtk_230227, 'pygtk_version')
# Applying the binary operator '%' (line 19)
result_mod_230229 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 19), '%', unicode_230226, pygtk_version_230228)

unicode_230230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'unicode', u'Pycairo(%s)')
# Getting the type of 'backend_cairo' (line 20)
backend_cairo_230231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 37), 'backend_cairo')
# Obtaining the member 'backend_version' of a type (line 20)
backend_version_230232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 37), backend_cairo_230231, 'backend_version')
# Applying the binary operator '%' (line 20)
result_mod_230233 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 21), '%', unicode_230230, backend_version_230232)

# Applying the binary operator '+' (line 19)
result_add_230234 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 19), '+', result_mod_230229, result_mod_230233)

# Assigning a type to the variable 'backend_version' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'backend_version', result_add_230234)
# Declaration of the 'RendererGTKCairo' class
# Getting the type of 'backend_cairo' (line 23)
backend_cairo_230235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'backend_cairo')
# Obtaining the member 'RendererCairo' of a type (line 23)
RendererCairo_230236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 24), backend_cairo_230235, 'RendererCairo')

class RendererGTKCairo(RendererCairo_230236, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 23, 0, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGTKCairo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'RendererGTKCairo' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'RendererGTKCairo', RendererGTKCairo)


# Getting the type of 'gtk' (line 24)
gtk_230237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 7), 'gtk')
# Obtaining the member 'pygtk_version' of a type (line 24)
pygtk_version_230238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 7), gtk_230237, 'pygtk_version')

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_230239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 29), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
int_230240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 29), tuple_230239, int_230240)
# Adding element type (line 24)
int_230241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 29), tuple_230239, int_230241)
# Adding element type (line 24)
int_230242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 29), tuple_230239, int_230242)

# Applying the binary operator '>=' (line 24)
result_ge_230243 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 7), '>=', pygtk_version_230238, tuple_230239)

# Testing the type of an if condition (line 24)
if_condition_230244 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 4), result_ge_230243)
# Assigning a type to the variable 'if_condition_230244' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'if_condition_230244', if_condition_230244)
# SSA begins for if statement (line 24)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def set_pixmap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'set_pixmap'
    module_type_store = module_type_store.open_function_context('set_pixmap', 25, 8, False)
    
    # Passed parameters checking function
    set_pixmap.stypy_localization = localization
    set_pixmap.stypy_type_of_self = None
    set_pixmap.stypy_type_store = module_type_store
    set_pixmap.stypy_function_name = 'set_pixmap'
    set_pixmap.stypy_param_names_list = ['self', 'pixmap']
    set_pixmap.stypy_varargs_param_name = None
    set_pixmap.stypy_kwargs_param_name = None
    set_pixmap.stypy_call_defaults = defaults
    set_pixmap.stypy_call_varargs = varargs
    set_pixmap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'set_pixmap', ['self', 'pixmap'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'set_pixmap', localization, ['self', 'pixmap'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'set_pixmap(...)' code ##################

    
    # Assigning a Call to a Attribute (line 26):
    
    # Call to cairo_create(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_230247 = {}
    # Getting the type of 'pixmap' (line 26)
    pixmap_230245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 26), 'pixmap', False)
    # Obtaining the member 'cairo_create' of a type (line 26)
    cairo_create_230246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 26), pixmap_230245, 'cairo_create')
    # Calling cairo_create(args, kwargs) (line 26)
    cairo_create_call_result_230248 = invoke(stypy.reporting.localization.Localization(__file__, 26, 26), cairo_create_230246, *[], **kwargs_230247)
    
    # Getting the type of 'self' (line 26)
    self_230249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'self')
    # Obtaining the member 'gc' of a type (line 26)
    gc_230250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), self_230249, 'gc')
    # Setting the type of the member 'ctx' of a type (line 26)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), gc_230250, 'ctx', cairo_create_call_result_230248)
    
    # ################# End of 'set_pixmap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set_pixmap' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_230251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_230251)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set_pixmap'
    return stypy_return_type_230251

# Assigning a type to the variable 'set_pixmap' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'set_pixmap', set_pixmap)
# SSA branch for the else part of an if statement (line 24)
module_type_store.open_ssa_branch('else')

@norecursion
def set_pixmap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'set_pixmap'
    module_type_store = module_type_store.open_function_context('set_pixmap', 28, 8, False)
    
    # Passed parameters checking function
    set_pixmap.stypy_localization = localization
    set_pixmap.stypy_type_of_self = None
    set_pixmap.stypy_type_store = module_type_store
    set_pixmap.stypy_function_name = 'set_pixmap'
    set_pixmap.stypy_param_names_list = ['self', 'pixmap']
    set_pixmap.stypy_varargs_param_name = None
    set_pixmap.stypy_kwargs_param_name = None
    set_pixmap.stypy_call_defaults = defaults
    set_pixmap.stypy_call_varargs = varargs
    set_pixmap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'set_pixmap', ['self', 'pixmap'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'set_pixmap', localization, ['self', 'pixmap'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'set_pixmap(...)' code ##################

    
    # Assigning a Call to a Attribute (line 29):
    
    # Call to gdk_cairo_create(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'pixmap' (line 29)
    pixmap_230255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 54), 'pixmap', False)
    # Processing the call keyword arguments (line 29)
    kwargs_230256 = {}
    # Getting the type of 'cairo' (line 29)
    cairo_230252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 26), 'cairo', False)
    # Obtaining the member 'gtk' of a type (line 29)
    gtk_230253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 26), cairo_230252, 'gtk')
    # Obtaining the member 'gdk_cairo_create' of a type (line 29)
    gdk_cairo_create_230254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 26), gtk_230253, 'gdk_cairo_create')
    # Calling gdk_cairo_create(args, kwargs) (line 29)
    gdk_cairo_create_call_result_230257 = invoke(stypy.reporting.localization.Localization(__file__, 29, 26), gdk_cairo_create_230254, *[pixmap_230255], **kwargs_230256)
    
    # Getting the type of 'self' (line 29)
    self_230258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'self')
    # Obtaining the member 'gc' of a type (line 29)
    gc_230259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), self_230258, 'gc')
    # Setting the type of the member 'ctx' of a type (line 29)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), gc_230259, 'ctx', gdk_cairo_create_call_result_230257)
    
    # ################# End of 'set_pixmap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set_pixmap' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_230260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_230260)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set_pixmap'
    return stypy_return_type_230260

# Assigning a type to the variable 'set_pixmap' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'set_pixmap', set_pixmap)
# SSA join for if statement (line 24)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'FigureCanvasGTKCairo' class
# Getting the type of 'backend_cairo' (line 32)
backend_cairo_230261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 27), 'backend_cairo')
# Obtaining the member 'FigureCanvasCairo' of a type (line 32)
FigureCanvasCairo_230262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 27), backend_cairo_230261, 'FigureCanvasCairo')
# Getting the type of 'FigureCanvasGTK' (line 32)
FigureCanvasGTK_230263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 60), 'FigureCanvasGTK')

class FigureCanvasGTKCairo(FigureCanvasCairo_230262, FigureCanvasGTK_230263, ):

    @norecursion
    def _renderer_init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_renderer_init'
        module_type_store = module_type_store.open_function_context('_renderer_init', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTKCairo._renderer_init.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTKCairo._renderer_init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTKCairo._renderer_init.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTKCairo._renderer_init.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTKCairo._renderer_init')
        FigureCanvasGTKCairo._renderer_init.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasGTKCairo._renderer_init.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTKCairo._renderer_init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTKCairo._renderer_init.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTKCairo._renderer_init.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTKCairo._renderer_init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTKCairo._renderer_init.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTKCairo._renderer_init', [], None, None, defaults, varargs, kwargs)

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

        unicode_230264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 8), 'unicode', u'Override to use cairo (rather than GDK) renderer')
        
        # Assigning a Call to a Attribute (line 38):
        
        # Call to RendererGTKCairo(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'self' (line 38)
        self_230266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 42), 'self', False)
        # Obtaining the member 'figure' of a type (line 38)
        figure_230267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 42), self_230266, 'figure')
        # Obtaining the member 'dpi' of a type (line 38)
        dpi_230268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 42), figure_230267, 'dpi')
        # Processing the call keyword arguments (line 38)
        kwargs_230269 = {}
        # Getting the type of 'RendererGTKCairo' (line 38)
        RendererGTKCairo_230265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'RendererGTKCairo', False)
        # Calling RendererGTKCairo(args, kwargs) (line 38)
        RendererGTKCairo_call_result_230270 = invoke(stypy.reporting.localization.Localization(__file__, 38, 25), RendererGTKCairo_230265, *[dpi_230268], **kwargs_230269)
        
        # Getting the type of 'self' (line 38)
        self_230271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member '_renderer' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_230271, '_renderer', RendererGTKCairo_call_result_230270)
        
        # ################# End of '_renderer_init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_renderer_init' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_230272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230272)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_renderer_init'
        return stypy_return_type_230272


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTKCairo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureCanvasGTKCairo' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'FigureCanvasGTKCairo', FigureCanvasGTKCairo)

# Assigning a Call to a Name (line 33):

# Call to copy(...): (line 33)
# Processing the call keyword arguments (line 33)
kwargs_230276 = {}
# Getting the type of 'FigureCanvasGTK' (line 33)
FigureCanvasGTK_230273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'FigureCanvasGTK', False)
# Obtaining the member 'filetypes' of a type (line 33)
filetypes_230274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 16), FigureCanvasGTK_230273, 'filetypes')
# Obtaining the member 'copy' of a type (line 33)
copy_230275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 16), filetypes_230274, 'copy')
# Calling copy(args, kwargs) (line 33)
copy_call_result_230277 = invoke(stypy.reporting.localization.Localization(__file__, 33, 16), copy_230275, *[], **kwargs_230276)

# Getting the type of 'FigureCanvasGTKCairo'
FigureCanvasGTKCairo_230278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasGTKCairo')
# Setting the type of the member 'filetypes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasGTKCairo_230278, 'filetypes', copy_call_result_230277)

# Call to update(...): (line 34)
# Processing the call arguments (line 34)
# Getting the type of 'backend_cairo' (line 34)
backend_cairo_230282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'backend_cairo', False)
# Obtaining the member 'FigureCanvasCairo' of a type (line 34)
FigureCanvasCairo_230283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), backend_cairo_230282, 'FigureCanvasCairo')
# Obtaining the member 'filetypes' of a type (line 34)
filetypes_230284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), FigureCanvasCairo_230283, 'filetypes')
# Processing the call keyword arguments (line 34)
kwargs_230285 = {}
# Getting the type of 'FigureCanvasGTKCairo'
FigureCanvasGTKCairo_230279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasGTKCairo', False)
# Obtaining the member 'filetypes' of a type
filetypes_230280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasGTKCairo_230279, 'filetypes')
# Obtaining the member 'update' of a type (line 34)
update_230281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 4), filetypes_230280, 'update')
# Calling update(args, kwargs) (line 34)
update_call_result_230286 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), update_230281, *[filetypes_230284], **kwargs_230285)

# Declaration of the 'FigureManagerGTKCairo' class
# Getting the type of 'FigureManagerGTK' (line 43)
FigureManagerGTK_230287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 28), 'FigureManagerGTK')

class FigureManagerGTKCairo(FigureManagerGTK_230287, ):

    @norecursion
    def _get_toolbar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_toolbar'
        module_type_store = module_type_store.open_function_context('_get_toolbar', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerGTKCairo._get_toolbar.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerGTKCairo._get_toolbar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerGTKCairo._get_toolbar.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerGTKCairo._get_toolbar.__dict__.__setitem__('stypy_function_name', 'FigureManagerGTKCairo._get_toolbar')
        FigureManagerGTKCairo._get_toolbar.__dict__.__setitem__('stypy_param_names_list', ['canvas'])
        FigureManagerGTKCairo._get_toolbar.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerGTKCairo._get_toolbar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerGTKCairo._get_toolbar.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerGTKCairo._get_toolbar.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerGTKCairo._get_toolbar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerGTKCairo._get_toolbar.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTKCairo._get_toolbar', ['canvas'], None, None, defaults, varargs, kwargs)

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
        unicode_230288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 31), 'unicode', u'toolbar')
        # Getting the type of 'matplotlib' (line 47)
        matplotlib_230289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'matplotlib')
        # Obtaining the member 'rcParams' of a type (line 47)
        rcParams_230290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), matplotlib_230289, 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 47)
        getitem___230291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), rcParams_230290, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 47)
        subscript_call_result_230292 = invoke(stypy.reporting.localization.Localization(__file__, 47, 11), getitem___230291, unicode_230288)
        
        unicode_230293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 43), 'unicode', u'toolbar2')
        # Applying the binary operator '==' (line 47)
        result_eq_230294 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '==', subscript_call_result_230292, unicode_230293)
        
        # Testing the type of an if condition (line 47)
        if_condition_230295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 8), result_eq_230294)
        # Assigning a type to the variable 'if_condition_230295' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'if_condition_230295', if_condition_230295)
        # SSA begins for if statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 48):
        
        # Call to NavigationToolbar2GTKCairo(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'canvas' (line 48)
        canvas_230297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 50), 'canvas', False)
        # Getting the type of 'self' (line 48)
        self_230298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 58), 'self', False)
        # Obtaining the member 'window' of a type (line 48)
        window_230299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 58), self_230298, 'window')
        # Processing the call keyword arguments (line 48)
        kwargs_230300 = {}
        # Getting the type of 'NavigationToolbar2GTKCairo' (line 48)
        NavigationToolbar2GTKCairo_230296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'NavigationToolbar2GTKCairo', False)
        # Calling NavigationToolbar2GTKCairo(args, kwargs) (line 48)
        NavigationToolbar2GTKCairo_call_result_230301 = invoke(stypy.reporting.localization.Localization(__file__, 48, 22), NavigationToolbar2GTKCairo_230296, *[canvas_230297, window_230299], **kwargs_230300)
        
        # Assigning a type to the variable 'toolbar' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'toolbar', NavigationToolbar2GTKCairo_call_result_230301)
        # SSA branch for the else part of an if statement (line 47)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 50):
        # Getting the type of 'None' (line 50)
        None_230302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'None')
        # Assigning a type to the variable 'toolbar' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'toolbar', None_230302)
        # SSA join for if statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'toolbar' (line 51)
        toolbar_230303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'toolbar')
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', toolbar_230303)
        
        # ################# End of '_get_toolbar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_toolbar' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_230304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230304)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_toolbar'
        return stypy_return_type_230304


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 42, 0, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTKCairo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureManagerGTKCairo' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'FigureManagerGTKCairo', FigureManagerGTKCairo)
# Declaration of the 'NavigationToolbar2Cairo' class
# Getting the type of 'NavigationToolbar2GTK' (line 56)
NavigationToolbar2GTK_230305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'NavigationToolbar2GTK')

class NavigationToolbar2Cairo(NavigationToolbar2GTK_230305, ):

    @norecursion
    def _get_canvas(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_canvas'
        module_type_store = module_type_store.open_function_context('_get_canvas', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2Cairo._get_canvas.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2Cairo._get_canvas.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2Cairo._get_canvas.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2Cairo._get_canvas.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2Cairo._get_canvas')
        NavigationToolbar2Cairo._get_canvas.__dict__.__setitem__('stypy_param_names_list', ['fig'])
        NavigationToolbar2Cairo._get_canvas.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2Cairo._get_canvas.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2Cairo._get_canvas.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2Cairo._get_canvas.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2Cairo._get_canvas.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2Cairo._get_canvas.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2Cairo._get_canvas', ['fig'], None, None, defaults, varargs, kwargs)

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

        
        # Call to FigureCanvasGTKCairo(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'fig' (line 58)
        fig_230307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 36), 'fig', False)
        # Processing the call keyword arguments (line 58)
        kwargs_230308 = {}
        # Getting the type of 'FigureCanvasGTKCairo' (line 58)
        FigureCanvasGTKCairo_230306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'FigureCanvasGTKCairo', False)
        # Calling FigureCanvasGTKCairo(args, kwargs) (line 58)
        FigureCanvasGTKCairo_call_result_230309 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), FigureCanvasGTKCairo_230306, *[fig_230307], **kwargs_230308)
        
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type', FigureCanvasGTKCairo_call_result_230309)
        
        # ################# End of '_get_canvas(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_canvas' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_230310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230310)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_canvas'
        return stypy_return_type_230310


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 55, 0, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2Cairo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NavigationToolbar2Cairo' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'NavigationToolbar2Cairo', NavigationToolbar2Cairo)
# Declaration of the '_BackendGTKCairo' class
# Getting the type of '_BackendGTK' (line 62)
_BackendGTK_230311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), '_BackendGTK')

class _BackendGTKCairo(_BackendGTK_230311, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 61, 0, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendGTKCairo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendGTKCairo' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), '_BackendGTKCairo', _BackendGTKCairo)

# Assigning a Name to a Name (line 63):
# Getting the type of 'FigureCanvasGTKCairo' (line 63)
FigureCanvasGTKCairo_230312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'FigureCanvasGTKCairo')
# Getting the type of '_BackendGTKCairo'
_BackendGTKCairo_230313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendGTKCairo')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendGTKCairo_230313, 'FigureCanvas', FigureCanvasGTKCairo_230312)

# Assigning a Name to a Name (line 64):
# Getting the type of 'FigureManagerGTK' (line 64)
FigureManagerGTK_230314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'FigureManagerGTK')
# Getting the type of '_BackendGTKCairo'
_BackendGTKCairo_230315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendGTKCairo')
# Setting the type of the member 'FigureManager' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendGTKCairo_230315, 'FigureManager', FigureManagerGTK_230314)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
