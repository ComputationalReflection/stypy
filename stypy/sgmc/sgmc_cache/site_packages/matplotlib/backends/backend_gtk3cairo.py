
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: from . import backend_cairo, backend_gtk3
7: from .backend_cairo import cairo, HAS_CAIRO_CFFI
8: from .backend_gtk3 import _BackendGTK3
9: from matplotlib.backend_bases import cursors
10: from matplotlib.figure import Figure
11: 
12: 
13: class RendererGTK3Cairo(backend_cairo.RendererCairo):
14:     def set_context(self, ctx):
15:         if HAS_CAIRO_CFFI:
16:             ctx = cairo.Context._from_pointer(
17:                 cairo.ffi.cast(
18:                     'cairo_t **',
19:                     id(ctx) + object.__basicsize__)[0],
20:                 incref=True)
21: 
22:         self.gc.ctx = ctx
23: 
24: 
25: class FigureCanvasGTK3Cairo(backend_gtk3.FigureCanvasGTK3,
26:                             backend_cairo.FigureCanvasCairo):
27: 
28:     def _renderer_init(self):
29:         '''use cairo renderer'''
30:         self._renderer = RendererGTK3Cairo(self.figure.dpi)
31: 
32:     def _render_figure(self, width, height):
33:         self._renderer.set_width_height(width, height)
34:         self.figure.draw(self._renderer)
35: 
36:     def on_draw_event(self, widget, ctx):
37:         ''' GtkDrawable draw event, like expose_event in GTK 2.X
38:         '''
39:         toolbar = self.toolbar
40:         if toolbar:
41:             toolbar.set_cursor(cursors.WAIT)
42:         self._renderer.set_context(ctx)
43:         allocation = self.get_allocation()
44:         x, y, w, h = allocation.x, allocation.y, allocation.width, allocation.height
45:         self._render_figure(w, h)
46:         if toolbar:
47:             toolbar.set_cursor(toolbar._lastCursor)
48:         return False  # finish event propagation?
49: 
50: 
51: class FigureManagerGTK3Cairo(backend_gtk3.FigureManagerGTK3):
52:     pass
53: 
54: 
55: @_BackendGTK3.export
56: class _BackendGTK3Cairo(_BackendGTK3):
57:     FigureCanvas = FigureCanvasGTK3Cairo
58:     FigureManager = FigureManagerGTK3Cairo
59: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229780 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_229780) is not StypyTypeError):

    if (import_229780 != 'pyd_module'):
        __import__(import_229780)
        sys_modules_229781 = sys.modules[import_229780]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_229781.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_229780)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from matplotlib.backends import backend_cairo, backend_gtk3' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229782 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.backends')

if (type(import_229782) is not StypyTypeError):

    if (import_229782 != 'pyd_module'):
        __import__(import_229782)
        sys_modules_229783 = sys.modules[import_229782]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.backends', sys_modules_229783.module_type_store, module_type_store, ['backend_cairo', 'backend_gtk3'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_229783, sys_modules_229783.module_type_store, module_type_store)
    else:
        from matplotlib.backends import backend_cairo, backend_gtk3

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.backends', None, module_type_store, ['backend_cairo', 'backend_gtk3'], [backend_cairo, backend_gtk3])

else:
    # Assigning a type to the variable 'matplotlib.backends' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.backends', import_229782)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from matplotlib.backends.backend_cairo import cairo, HAS_CAIRO_CFFI' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229784 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.backends.backend_cairo')

if (type(import_229784) is not StypyTypeError):

    if (import_229784 != 'pyd_module'):
        __import__(import_229784)
        sys_modules_229785 = sys.modules[import_229784]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.backends.backend_cairo', sys_modules_229785.module_type_store, module_type_store, ['cairo', 'HAS_CAIRO_CFFI'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_229785, sys_modules_229785.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_cairo import cairo, HAS_CAIRO_CFFI

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.backends.backend_cairo', None, module_type_store, ['cairo', 'HAS_CAIRO_CFFI'], [cairo, HAS_CAIRO_CFFI])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_cairo' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.backends.backend_cairo', import_229784)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from matplotlib.backends.backend_gtk3 import _BackendGTK3' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229786 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.backends.backend_gtk3')

if (type(import_229786) is not StypyTypeError):

    if (import_229786 != 'pyd_module'):
        __import__(import_229786)
        sys_modules_229787 = sys.modules[import_229786]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.backends.backend_gtk3', sys_modules_229787.module_type_store, module_type_store, ['_BackendGTK3'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_229787, sys_modules_229787.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_gtk3 import _BackendGTK3

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.backends.backend_gtk3', None, module_type_store, ['_BackendGTK3'], [_BackendGTK3])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_gtk3' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.backends.backend_gtk3', import_229786)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from matplotlib.backend_bases import cursors' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229788 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backend_bases')

if (type(import_229788) is not StypyTypeError):

    if (import_229788 != 'pyd_module'):
        __import__(import_229788)
        sys_modules_229789 = sys.modules[import_229788]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backend_bases', sys_modules_229789.module_type_store, module_type_store, ['cursors'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_229789, sys_modules_229789.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import cursors

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backend_bases', None, module_type_store, ['cursors'], [cursors])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backend_bases', import_229788)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from matplotlib.figure import Figure' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_229790 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.figure')

if (type(import_229790) is not StypyTypeError):

    if (import_229790 != 'pyd_module'):
        __import__(import_229790)
        sys_modules_229791 = sys.modules[import_229790]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.figure', sys_modules_229791.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_229791, sys_modules_229791.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.figure', import_229790)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# Declaration of the 'RendererGTK3Cairo' class
# Getting the type of 'backend_cairo' (line 13)
backend_cairo_229792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 24), 'backend_cairo')
# Obtaining the member 'RendererCairo' of a type (line 13)
RendererCairo_229793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 24), backend_cairo_229792, 'RendererCairo')

class RendererGTK3Cairo(RendererCairo_229793, ):

    @norecursion
    def set_context(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_context'
        module_type_store = module_type_store.open_function_context('set_context', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererGTK3Cairo.set_context.__dict__.__setitem__('stypy_localization', localization)
        RendererGTK3Cairo.set_context.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererGTK3Cairo.set_context.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererGTK3Cairo.set_context.__dict__.__setitem__('stypy_function_name', 'RendererGTK3Cairo.set_context')
        RendererGTK3Cairo.set_context.__dict__.__setitem__('stypy_param_names_list', ['ctx'])
        RendererGTK3Cairo.set_context.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererGTK3Cairo.set_context.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererGTK3Cairo.set_context.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererGTK3Cairo.set_context.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererGTK3Cairo.set_context.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererGTK3Cairo.set_context.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGTK3Cairo.set_context', ['ctx'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_context', localization, ['ctx'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_context(...)' code ##################

        
        # Getting the type of 'HAS_CAIRO_CFFI' (line 15)
        HAS_CAIRO_CFFI_229794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'HAS_CAIRO_CFFI')
        # Testing the type of an if condition (line 15)
        if_condition_229795 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 8), HAS_CAIRO_CFFI_229794)
        # Assigning a type to the variable 'if_condition_229795' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'if_condition_229795', if_condition_229795)
        # SSA begins for if statement (line 15)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 16):
        
        # Assigning a Call to a Name (line 16):
        
        # Call to _from_pointer(...): (line 16)
        # Processing the call arguments (line 16)
        
        # Obtaining the type of the subscript
        int_229799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 52), 'int')
        
        # Call to cast(...): (line 17)
        # Processing the call arguments (line 17)
        unicode_229803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'unicode', u'cairo_t **')
        
        # Call to id(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'ctx' (line 19)
        ctx_229805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'ctx', False)
        # Processing the call keyword arguments (line 19)
        kwargs_229806 = {}
        # Getting the type of 'id' (line 19)
        id_229804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'id', False)
        # Calling id(args, kwargs) (line 19)
        id_call_result_229807 = invoke(stypy.reporting.localization.Localization(__file__, 19, 20), id_229804, *[ctx_229805], **kwargs_229806)
        
        # Getting the type of 'object' (line 19)
        object_229808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 30), 'object', False)
        # Obtaining the member '__basicsize__' of a type (line 19)
        basicsize___229809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 30), object_229808, '__basicsize__')
        # Applying the binary operator '+' (line 19)
        result_add_229810 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 20), '+', id_call_result_229807, basicsize___229809)
        
        # Processing the call keyword arguments (line 17)
        kwargs_229811 = {}
        # Getting the type of 'cairo' (line 17)
        cairo_229800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'cairo', False)
        # Obtaining the member 'ffi' of a type (line 17)
        ffi_229801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 16), cairo_229800, 'ffi')
        # Obtaining the member 'cast' of a type (line 17)
        cast_229802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 16), ffi_229801, 'cast')
        # Calling cast(args, kwargs) (line 17)
        cast_call_result_229812 = invoke(stypy.reporting.localization.Localization(__file__, 17, 16), cast_229802, *[unicode_229803, result_add_229810], **kwargs_229811)
        
        # Obtaining the member '__getitem__' of a type (line 17)
        getitem___229813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 16), cast_call_result_229812, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 17)
        subscript_call_result_229814 = invoke(stypy.reporting.localization.Localization(__file__, 17, 16), getitem___229813, int_229799)
        
        # Processing the call keyword arguments (line 16)
        # Getting the type of 'True' (line 20)
        True_229815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 23), 'True', False)
        keyword_229816 = True_229815
        kwargs_229817 = {'incref': keyword_229816}
        # Getting the type of 'cairo' (line 16)
        cairo_229796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'cairo', False)
        # Obtaining the member 'Context' of a type (line 16)
        Context_229797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 18), cairo_229796, 'Context')
        # Obtaining the member '_from_pointer' of a type (line 16)
        _from_pointer_229798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 18), Context_229797, '_from_pointer')
        # Calling _from_pointer(args, kwargs) (line 16)
        _from_pointer_call_result_229818 = invoke(stypy.reporting.localization.Localization(__file__, 16, 18), _from_pointer_229798, *[subscript_call_result_229814], **kwargs_229817)
        
        # Assigning a type to the variable 'ctx' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'ctx', _from_pointer_call_result_229818)
        # SSA join for if statement (line 15)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 22):
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'ctx' (line 22)
        ctx_229819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 22), 'ctx')
        # Getting the type of 'self' (line 22)
        self_229820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Obtaining the member 'gc' of a type (line 22)
        gc_229821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_229820, 'gc')
        # Setting the type of the member 'ctx' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), gc_229821, 'ctx', ctx_229819)
        
        # ################# End of 'set_context(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_context' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_229822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229822)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_context'
        return stypy_return_type_229822


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 0, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererGTK3Cairo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'RendererGTK3Cairo' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'RendererGTK3Cairo', RendererGTK3Cairo)
# Declaration of the 'FigureCanvasGTK3Cairo' class
# Getting the type of 'backend_gtk3' (line 25)
backend_gtk3_229823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 28), 'backend_gtk3')
# Obtaining the member 'FigureCanvasGTK3' of a type (line 25)
FigureCanvasGTK3_229824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 28), backend_gtk3_229823, 'FigureCanvasGTK3')
# Getting the type of 'backend_cairo' (line 26)
backend_cairo_229825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 28), 'backend_cairo')
# Obtaining the member 'FigureCanvasCairo' of a type (line 26)
FigureCanvasCairo_229826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 28), backend_cairo_229825, 'FigureCanvasCairo')

class FigureCanvasGTK3Cairo(FigureCanvasGTK3_229824, FigureCanvasCairo_229826, ):

    @norecursion
    def _renderer_init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_renderer_init'
        module_type_store = module_type_store.open_function_context('_renderer_init', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3Cairo._renderer_init.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3Cairo._renderer_init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3Cairo._renderer_init.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3Cairo._renderer_init.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3Cairo._renderer_init')
        FigureCanvasGTK3Cairo._renderer_init.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasGTK3Cairo._renderer_init.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3Cairo._renderer_init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3Cairo._renderer_init.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3Cairo._renderer_init.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3Cairo._renderer_init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3Cairo._renderer_init.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3Cairo._renderer_init', [], None, None, defaults, varargs, kwargs)

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

        unicode_229827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 8), 'unicode', u'use cairo renderer')
        
        # Assigning a Call to a Attribute (line 30):
        
        # Assigning a Call to a Attribute (line 30):
        
        # Call to RendererGTK3Cairo(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'self' (line 30)
        self_229829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 43), 'self', False)
        # Obtaining the member 'figure' of a type (line 30)
        figure_229830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 43), self_229829, 'figure')
        # Obtaining the member 'dpi' of a type (line 30)
        dpi_229831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 43), figure_229830, 'dpi')
        # Processing the call keyword arguments (line 30)
        kwargs_229832 = {}
        # Getting the type of 'RendererGTK3Cairo' (line 30)
        RendererGTK3Cairo_229828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'RendererGTK3Cairo', False)
        # Calling RendererGTK3Cairo(args, kwargs) (line 30)
        RendererGTK3Cairo_call_result_229833 = invoke(stypy.reporting.localization.Localization(__file__, 30, 25), RendererGTK3Cairo_229828, *[dpi_229831], **kwargs_229832)
        
        # Getting the type of 'self' (line 30)
        self_229834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member '_renderer' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_229834, '_renderer', RendererGTK3Cairo_call_result_229833)
        
        # ################# End of '_renderer_init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_renderer_init' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_229835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229835)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_renderer_init'
        return stypy_return_type_229835


    @norecursion
    def _render_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_render_figure'
        module_type_store = module_type_store.open_function_context('_render_figure', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3Cairo._render_figure.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3Cairo._render_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3Cairo._render_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3Cairo._render_figure.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3Cairo._render_figure')
        FigureCanvasGTK3Cairo._render_figure.__dict__.__setitem__('stypy_param_names_list', ['width', 'height'])
        FigureCanvasGTK3Cairo._render_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3Cairo._render_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3Cairo._render_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3Cairo._render_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3Cairo._render_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3Cairo._render_figure.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3Cairo._render_figure', ['width', 'height'], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_width_height(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'width' (line 33)
        width_229839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 40), 'width', False)
        # Getting the type of 'height' (line 33)
        height_229840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 47), 'height', False)
        # Processing the call keyword arguments (line 33)
        kwargs_229841 = {}
        # Getting the type of 'self' (line 33)
        self_229836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self', False)
        # Obtaining the member '_renderer' of a type (line 33)
        _renderer_229837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_229836, '_renderer')
        # Obtaining the member 'set_width_height' of a type (line 33)
        set_width_height_229838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), _renderer_229837, 'set_width_height')
        # Calling set_width_height(args, kwargs) (line 33)
        set_width_height_call_result_229842 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), set_width_height_229838, *[width_229839, height_229840], **kwargs_229841)
        
        
        # Call to draw(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'self' (line 34)
        self_229846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'self', False)
        # Obtaining the member '_renderer' of a type (line 34)
        _renderer_229847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 25), self_229846, '_renderer')
        # Processing the call keyword arguments (line 34)
        kwargs_229848 = {}
        # Getting the type of 'self' (line 34)
        self_229843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 34)
        figure_229844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_229843, 'figure')
        # Obtaining the member 'draw' of a type (line 34)
        draw_229845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), figure_229844, 'draw')
        # Calling draw(args, kwargs) (line 34)
        draw_call_result_229849 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), draw_229845, *[_renderer_229847], **kwargs_229848)
        
        
        # ################# End of '_render_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_render_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_229850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229850)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_render_figure'
        return stypy_return_type_229850


    @norecursion
    def on_draw_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'on_draw_event'
        module_type_store = module_type_store.open_function_context('on_draw_event', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3Cairo.on_draw_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3Cairo.on_draw_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3Cairo.on_draw_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3Cairo.on_draw_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3Cairo.on_draw_event')
        FigureCanvasGTK3Cairo.on_draw_event.__dict__.__setitem__('stypy_param_names_list', ['widget', 'ctx'])
        FigureCanvasGTK3Cairo.on_draw_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3Cairo.on_draw_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3Cairo.on_draw_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3Cairo.on_draw_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3Cairo.on_draw_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3Cairo.on_draw_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3Cairo.on_draw_event', ['widget', 'ctx'], None, None, defaults, varargs, kwargs)

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

        unicode_229851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'unicode', u' GtkDrawable draw event, like expose_event in GTK 2.X\n        ')
        
        # Assigning a Attribute to a Name (line 39):
        
        # Assigning a Attribute to a Name (line 39):
        # Getting the type of 'self' (line 39)
        self_229852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'self')
        # Obtaining the member 'toolbar' of a type (line 39)
        toolbar_229853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 18), self_229852, 'toolbar')
        # Assigning a type to the variable 'toolbar' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'toolbar', toolbar_229853)
        
        # Getting the type of 'toolbar' (line 40)
        toolbar_229854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'toolbar')
        # Testing the type of an if condition (line 40)
        if_condition_229855 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 8), toolbar_229854)
        # Assigning a type to the variable 'if_condition_229855' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'if_condition_229855', if_condition_229855)
        # SSA begins for if statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_cursor(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'cursors' (line 41)
        cursors_229858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'cursors', False)
        # Obtaining the member 'WAIT' of a type (line 41)
        WAIT_229859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 31), cursors_229858, 'WAIT')
        # Processing the call keyword arguments (line 41)
        kwargs_229860 = {}
        # Getting the type of 'toolbar' (line 41)
        toolbar_229856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'toolbar', False)
        # Obtaining the member 'set_cursor' of a type (line 41)
        set_cursor_229857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), toolbar_229856, 'set_cursor')
        # Calling set_cursor(args, kwargs) (line 41)
        set_cursor_call_result_229861 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), set_cursor_229857, *[WAIT_229859], **kwargs_229860)
        
        # SSA join for if statement (line 40)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_context(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'ctx' (line 42)
        ctx_229865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 35), 'ctx', False)
        # Processing the call keyword arguments (line 42)
        kwargs_229866 = {}
        # Getting the type of 'self' (line 42)
        self_229862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self', False)
        # Obtaining the member '_renderer' of a type (line 42)
        _renderer_229863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_229862, '_renderer')
        # Obtaining the member 'set_context' of a type (line 42)
        set_context_229864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), _renderer_229863, 'set_context')
        # Calling set_context(args, kwargs) (line 42)
        set_context_call_result_229867 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), set_context_229864, *[ctx_229865], **kwargs_229866)
        
        
        # Assigning a Call to a Name (line 43):
        
        # Assigning a Call to a Name (line 43):
        
        # Call to get_allocation(...): (line 43)
        # Processing the call keyword arguments (line 43)
        kwargs_229870 = {}
        # Getting the type of 'self' (line 43)
        self_229868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'self', False)
        # Obtaining the member 'get_allocation' of a type (line 43)
        get_allocation_229869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 21), self_229868, 'get_allocation')
        # Calling get_allocation(args, kwargs) (line 43)
        get_allocation_call_result_229871 = invoke(stypy.reporting.localization.Localization(__file__, 43, 21), get_allocation_229869, *[], **kwargs_229870)
        
        # Assigning a type to the variable 'allocation' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'allocation', get_allocation_call_result_229871)
        
        # Assigning a Tuple to a Tuple (line 44):
        
        # Assigning a Attribute to a Name (line 44):
        # Getting the type of 'allocation' (line 44)
        allocation_229872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), 'allocation')
        # Obtaining the member 'x' of a type (line 44)
        x_229873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 21), allocation_229872, 'x')
        # Assigning a type to the variable 'tuple_assignment_229776' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_assignment_229776', x_229873)
        
        # Assigning a Attribute to a Name (line 44):
        # Getting the type of 'allocation' (line 44)
        allocation_229874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'allocation')
        # Obtaining the member 'y' of a type (line 44)
        y_229875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 35), allocation_229874, 'y')
        # Assigning a type to the variable 'tuple_assignment_229777' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_assignment_229777', y_229875)
        
        # Assigning a Attribute to a Name (line 44):
        # Getting the type of 'allocation' (line 44)
        allocation_229876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 49), 'allocation')
        # Obtaining the member 'width' of a type (line 44)
        width_229877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 49), allocation_229876, 'width')
        # Assigning a type to the variable 'tuple_assignment_229778' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_assignment_229778', width_229877)
        
        # Assigning a Attribute to a Name (line 44):
        # Getting the type of 'allocation' (line 44)
        allocation_229878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 67), 'allocation')
        # Obtaining the member 'height' of a type (line 44)
        height_229879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 67), allocation_229878, 'height')
        # Assigning a type to the variable 'tuple_assignment_229779' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_assignment_229779', height_229879)
        
        # Assigning a Name to a Name (line 44):
        # Getting the type of 'tuple_assignment_229776' (line 44)
        tuple_assignment_229776_229880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_assignment_229776')
        # Assigning a type to the variable 'x' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'x', tuple_assignment_229776_229880)
        
        # Assigning a Name to a Name (line 44):
        # Getting the type of 'tuple_assignment_229777' (line 44)
        tuple_assignment_229777_229881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_assignment_229777')
        # Assigning a type to the variable 'y' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'y', tuple_assignment_229777_229881)
        
        # Assigning a Name to a Name (line 44):
        # Getting the type of 'tuple_assignment_229778' (line 44)
        tuple_assignment_229778_229882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_assignment_229778')
        # Assigning a type to the variable 'w' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 14), 'w', tuple_assignment_229778_229882)
        
        # Assigning a Name to a Name (line 44):
        # Getting the type of 'tuple_assignment_229779' (line 44)
        tuple_assignment_229779_229883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_assignment_229779')
        # Assigning a type to the variable 'h' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'h', tuple_assignment_229779_229883)
        
        # Call to _render_figure(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'w' (line 45)
        w_229886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 28), 'w', False)
        # Getting the type of 'h' (line 45)
        h_229887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'h', False)
        # Processing the call keyword arguments (line 45)
        kwargs_229888 = {}
        # Getting the type of 'self' (line 45)
        self_229884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self', False)
        # Obtaining the member '_render_figure' of a type (line 45)
        _render_figure_229885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_229884, '_render_figure')
        # Calling _render_figure(args, kwargs) (line 45)
        _render_figure_call_result_229889 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), _render_figure_229885, *[w_229886, h_229887], **kwargs_229888)
        
        
        # Getting the type of 'toolbar' (line 46)
        toolbar_229890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'toolbar')
        # Testing the type of an if condition (line 46)
        if_condition_229891 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 8), toolbar_229890)
        # Assigning a type to the variable 'if_condition_229891' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'if_condition_229891', if_condition_229891)
        # SSA begins for if statement (line 46)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_cursor(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'toolbar' (line 47)
        toolbar_229894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'toolbar', False)
        # Obtaining the member '_lastCursor' of a type (line 47)
        _lastCursor_229895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 31), toolbar_229894, '_lastCursor')
        # Processing the call keyword arguments (line 47)
        kwargs_229896 = {}
        # Getting the type of 'toolbar' (line 47)
        toolbar_229892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'toolbar', False)
        # Obtaining the member 'set_cursor' of a type (line 47)
        set_cursor_229893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), toolbar_229892, 'set_cursor')
        # Calling set_cursor(args, kwargs) (line 47)
        set_cursor_call_result_229897 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), set_cursor_229893, *[_lastCursor_229895], **kwargs_229896)
        
        # SSA join for if statement (line 46)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'False' (line 48)
        False_229898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', False_229898)
        
        # ################# End of 'on_draw_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'on_draw_event' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_229899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229899)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'on_draw_event'
        return stypy_return_type_229899


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 25, 0, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3Cairo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureCanvasGTK3Cairo' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'FigureCanvasGTK3Cairo', FigureCanvasGTK3Cairo)
# Declaration of the 'FigureManagerGTK3Cairo' class
# Getting the type of 'backend_gtk3' (line 51)
backend_gtk3_229900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'backend_gtk3')
# Obtaining the member 'FigureManagerGTK3' of a type (line 51)
FigureManagerGTK3_229901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 29), backend_gtk3_229900, 'FigureManagerGTK3')

class FigureManagerGTK3Cairo(FigureManagerGTK3_229901, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 51, 0, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTK3Cairo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureManagerGTK3Cairo' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'FigureManagerGTK3Cairo', FigureManagerGTK3Cairo)
# Declaration of the '_BackendGTK3Cairo' class
# Getting the type of '_BackendGTK3' (line 56)
_BackendGTK3_229902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), '_BackendGTK3')

class _BackendGTK3Cairo(_BackendGTK3_229902, ):
    
    # Assigning a Name to a Name (line 57):
    
    # Assigning a Name to a Name (line 58):
    pass

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


# Assigning a type to the variable '_BackendGTK3Cairo' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), '_BackendGTK3Cairo', _BackendGTK3Cairo)

# Assigning a Name to a Name (line 57):
# Getting the type of 'FigureCanvasGTK3Cairo' (line 57)
FigureCanvasGTK3Cairo_229903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'FigureCanvasGTK3Cairo')
# Getting the type of '_BackendGTK3Cairo'
_BackendGTK3Cairo_229904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendGTK3Cairo')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendGTK3Cairo_229904, 'FigureCanvas', FigureCanvasGTK3Cairo_229903)

# Assigning a Name to a Name (line 58):
# Getting the type of 'FigureManagerGTK3Cairo' (line 58)
FigureManagerGTK3Cairo_229905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'FigureManagerGTK3Cairo')
# Getting the type of '_BackendGTK3Cairo'
_BackendGTK3Cairo_229906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendGTK3Cairo')
# Setting the type of the member 'FigureManager' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendGTK3Cairo_229906, 'FigureManager', FigureManagerGTK3Cairo_229905)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
