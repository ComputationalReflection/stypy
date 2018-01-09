
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import numpy as np
5: 
6: import six
7: 
8: from matplotlib.backends.backend_agg import RendererAgg
9: from matplotlib.tight_bbox import process_figure_for_rasterizing
10: 
11: 
12: class MixedModeRenderer(object):
13:     '''
14:     A helper class to implement a renderer that switches between
15:     vector and raster drawing.  An example may be a PDF writer, where
16:     most things are drawn with PDF vector commands, but some very
17:     complex objects, such as quad meshes, are rasterised and then
18:     output as images.
19:     '''
20:     def __init__(self, figure, width, height, dpi, vector_renderer,
21:                  raster_renderer_class=None,
22:                  bbox_inches_restore=None):
23:         '''
24:         Parameters
25:         ----------
26:         figure : `matplotlib.figure.Figure`
27:             The figure instance.
28: 
29:         width : scalar
30:             The width of the canvas in logical units
31: 
32:         height : scalar
33:             The height of the canvas in logical units
34: 
35:         dpi : scalar
36:             The dpi of the canvas
37: 
38:         vector_renderer : `matplotlib.backend_bases.RendererBase`
39:             An instance of a subclass of
40:             `~matplotlib.backend_bases.RendererBase` that will be used for the
41:             vector drawing.
42: 
43:         raster_renderer_class : `matplotlib.backend_bases.RendererBase`
44:             The renderer class to use for the raster drawing.  If not provided,
45:             this will use the Agg backend (which is currently the only viable
46:             option anyway.)
47: 
48:         '''
49:         if raster_renderer_class is None:
50:             raster_renderer_class = RendererAgg
51: 
52:         self._raster_renderer_class = raster_renderer_class
53:         self._width = width
54:         self._height = height
55:         self.dpi = dpi
56: 
57:         self._vector_renderer = vector_renderer
58: 
59:         self._raster_renderer = None
60:         self._rasterizing = 0
61: 
62:         # A reference to the figure is needed as we need to change
63:         # the figure dpi before and after the rasterization. Although
64:         # this looks ugly, I couldn't find a better solution. -JJL
65:         self.figure = figure
66:         self._figdpi = figure.get_dpi()
67: 
68:         self._bbox_inches_restore = bbox_inches_restore
69: 
70:         self._set_current_renderer(vector_renderer)
71: 
72:     _methods = '''
73:         close_group draw_image draw_markers draw_path
74:         draw_path_collection draw_quad_mesh draw_tex draw_text
75:         finalize flipy get_canvas_width_height get_image_magnification
76:         get_texmanager get_text_width_height_descent new_gc open_group
77:         option_image_nocomposite points_to_pixels strip_math
78:         start_filter stop_filter draw_gouraud_triangle
79:         draw_gouraud_triangles option_scale_image
80:         _text2path _get_text_path_transform height width
81:         '''.split()
82: 
83:     def _set_current_renderer(self, renderer):
84:         self._renderer = renderer
85: 
86:         for method in self._methods:
87:             if hasattr(renderer, method):
88:                 setattr(self, method, getattr(renderer, method))
89:         renderer.start_rasterizing = self.start_rasterizing
90:         renderer.stop_rasterizing = self.stop_rasterizing
91: 
92:     def start_rasterizing(self):
93:         '''
94:         Enter "raster" mode.  All subsequent drawing commands (until
95:         stop_rasterizing is called) will be drawn with the raster
96:         backend.
97: 
98:         If start_rasterizing is called multiple times before
99:         stop_rasterizing is called, this method has no effect.
100:         '''
101: 
102:         # change the dpi of the figure temporarily.
103:         self.figure.set_dpi(self.dpi)
104: 
105:         if self._bbox_inches_restore:  # when tight bbox is used
106:             r = process_figure_for_rasterizing(self.figure,
107:                                                self._bbox_inches_restore)
108:             self._bbox_inches_restore = r
109: 
110:         if self._rasterizing == 0:
111:             self._raster_renderer = self._raster_renderer_class(
112:                 self._width*self.dpi, self._height*self.dpi, self.dpi)
113:             self._set_current_renderer(self._raster_renderer)
114:         self._rasterizing += 1
115: 
116:     def stop_rasterizing(self):
117:         '''
118:         Exit "raster" mode.  All of the drawing that was done since
119:         the last start_rasterizing command will be copied to the
120:         vector backend by calling draw_image.
121: 
122:         If stop_rasterizing is called multiple times before
123:         start_rasterizing is called, this method has no effect.
124:         '''
125:         self._rasterizing -= 1
126:         if self._rasterizing == 0:
127:             self._set_current_renderer(self._vector_renderer)
128: 
129:             height = self._height * self.dpi
130:             buffer, bounds = self._raster_renderer.tostring_rgba_minimized()
131:             l, b, w, h = bounds
132:             if w > 0 and h > 0:
133:                 image = np.frombuffer(buffer, dtype=np.uint8)
134:                 image = image.reshape((h, w, 4))
135:                 image = image[::-1]
136:                 gc = self._renderer.new_gc()
137:                 # TODO: If the mixedmode resolution differs from the figure's
138:                 #       dpi, the image must be scaled (dpi->_figdpi). Not all
139:                 #       backends support this.
140:                 self._renderer.draw_image(
141:                     gc,
142:                     float(l) / self.dpi * self._figdpi,
143:                     (float(height)-b-h) / self.dpi * self._figdpi,
144:                     image)
145:             self._raster_renderer = None
146:             self._rasterizing = False
147: 
148:             # restore the figure dpi.
149:             self.figure.set_dpi(self._figdpi)
150: 
151:         if self._bbox_inches_restore:  # when tight bbox is used
152:             r = process_figure_for_rasterizing(self.figure,
153:                                                self._bbox_inches_restore,
154:                                                self._figdpi)
155:             self._bbox_inches_restore = r
156: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230775 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_230775) is not StypyTypeError):

    if (import_230775 != 'pyd_module'):
        __import__(import_230775)
        sys_modules_230776 = sys.modules[import_230775]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_230776.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_230775)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import six' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230777 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'six')

if (type(import_230777) is not StypyTypeError):

    if (import_230777 != 'pyd_module'):
        __import__(import_230777)
        sys_modules_230778 = sys.modules[import_230777]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'six', sys_modules_230778.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'six', import_230777)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from matplotlib.backends.backend_agg import RendererAgg' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230779 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.backends.backend_agg')

if (type(import_230779) is not StypyTypeError):

    if (import_230779 != 'pyd_module'):
        __import__(import_230779)
        sys_modules_230780 = sys.modules[import_230779]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.backends.backend_agg', sys_modules_230780.module_type_store, module_type_store, ['RendererAgg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_230780, sys_modules_230780.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_agg import RendererAgg

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.backends.backend_agg', None, module_type_store, ['RendererAgg'], [RendererAgg])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_agg' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.backends.backend_agg', import_230779)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from matplotlib.tight_bbox import process_figure_for_rasterizing' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_230781 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.tight_bbox')

if (type(import_230781) is not StypyTypeError):

    if (import_230781 != 'pyd_module'):
        __import__(import_230781)
        sys_modules_230782 = sys.modules[import_230781]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.tight_bbox', sys_modules_230782.module_type_store, module_type_store, ['process_figure_for_rasterizing'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_230782, sys_modules_230782.module_type_store, module_type_store)
    else:
        from matplotlib.tight_bbox import process_figure_for_rasterizing

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.tight_bbox', None, module_type_store, ['process_figure_for_rasterizing'], [process_figure_for_rasterizing])

else:
    # Assigning a type to the variable 'matplotlib.tight_bbox' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.tight_bbox', import_230781)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# Declaration of the 'MixedModeRenderer' class

class MixedModeRenderer(object, ):
    unicode_230783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'unicode', u'\n    A helper class to implement a renderer that switches between\n    vector and raster drawing.  An example may be a PDF writer, where\n    most things are drawn with PDF vector commands, but some very\n    complex objects, such as quad meshes, are rasterised and then\n    output as images.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 21)
        None_230784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 39), 'None')
        # Getting the type of 'None' (line 22)
        None_230785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 37), 'None')
        defaults = [None_230784, None_230785]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MixedModeRenderer.__init__', ['figure', 'width', 'height', 'dpi', 'vector_renderer', 'raster_renderer_class', 'bbox_inches_restore'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['figure', 'width', 'height', 'dpi', 'vector_renderer', 'raster_renderer_class', 'bbox_inches_restore'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_230786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, (-1)), 'unicode', u'\n        Parameters\n        ----------\n        figure : `matplotlib.figure.Figure`\n            The figure instance.\n\n        width : scalar\n            The width of the canvas in logical units\n\n        height : scalar\n            The height of the canvas in logical units\n\n        dpi : scalar\n            The dpi of the canvas\n\n        vector_renderer : `matplotlib.backend_bases.RendererBase`\n            An instance of a subclass of\n            `~matplotlib.backend_bases.RendererBase` that will be used for the\n            vector drawing.\n\n        raster_renderer_class : `matplotlib.backend_bases.RendererBase`\n            The renderer class to use for the raster drawing.  If not provided,\n            this will use the Agg backend (which is currently the only viable\n            option anyway.)\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 49)
        # Getting the type of 'raster_renderer_class' (line 49)
        raster_renderer_class_230787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'raster_renderer_class')
        # Getting the type of 'None' (line 49)
        None_230788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 36), 'None')
        
        (may_be_230789, more_types_in_union_230790) = may_be_none(raster_renderer_class_230787, None_230788)

        if may_be_230789:

            if more_types_in_union_230790:
                # Runtime conditional SSA (line 49)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 50):
            
            # Assigning a Name to a Name (line 50):
            # Getting the type of 'RendererAgg' (line 50)
            RendererAgg_230791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 36), 'RendererAgg')
            # Assigning a type to the variable 'raster_renderer_class' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'raster_renderer_class', RendererAgg_230791)

            if more_types_in_union_230790:
                # SSA join for if statement (line 49)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 52):
        
        # Assigning a Name to a Attribute (line 52):
        # Getting the type of 'raster_renderer_class' (line 52)
        raster_renderer_class_230792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 38), 'raster_renderer_class')
        # Getting the type of 'self' (line 52)
        self_230793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self')
        # Setting the type of the member '_raster_renderer_class' of a type (line 52)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_230793, '_raster_renderer_class', raster_renderer_class_230792)
        
        # Assigning a Name to a Attribute (line 53):
        
        # Assigning a Name to a Attribute (line 53):
        # Getting the type of 'width' (line 53)
        width_230794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'width')
        # Getting the type of 'self' (line 53)
        self_230795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self')
        # Setting the type of the member '_width' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_230795, '_width', width_230794)
        
        # Assigning a Name to a Attribute (line 54):
        
        # Assigning a Name to a Attribute (line 54):
        # Getting the type of 'height' (line 54)
        height_230796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 'height')
        # Getting the type of 'self' (line 54)
        self_230797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self')
        # Setting the type of the member '_height' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_230797, '_height', height_230796)
        
        # Assigning a Name to a Attribute (line 55):
        
        # Assigning a Name to a Attribute (line 55):
        # Getting the type of 'dpi' (line 55)
        dpi_230798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'dpi')
        # Getting the type of 'self' (line 55)
        self_230799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self')
        # Setting the type of the member 'dpi' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_230799, 'dpi', dpi_230798)
        
        # Assigning a Name to a Attribute (line 57):
        
        # Assigning a Name to a Attribute (line 57):
        # Getting the type of 'vector_renderer' (line 57)
        vector_renderer_230800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 32), 'vector_renderer')
        # Getting the type of 'self' (line 57)
        self_230801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member '_vector_renderer' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_230801, '_vector_renderer', vector_renderer_230800)
        
        # Assigning a Name to a Attribute (line 59):
        
        # Assigning a Name to a Attribute (line 59):
        # Getting the type of 'None' (line 59)
        None_230802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'None')
        # Getting the type of 'self' (line 59)
        self_230803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member '_raster_renderer' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_230803, '_raster_renderer', None_230802)
        
        # Assigning a Num to a Attribute (line 60):
        
        # Assigning a Num to a Attribute (line 60):
        int_230804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'int')
        # Getting the type of 'self' (line 60)
        self_230805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self')
        # Setting the type of the member '_rasterizing' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_230805, '_rasterizing', int_230804)
        
        # Assigning a Name to a Attribute (line 65):
        
        # Assigning a Name to a Attribute (line 65):
        # Getting the type of 'figure' (line 65)
        figure_230806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 22), 'figure')
        # Getting the type of 'self' (line 65)
        self_230807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Setting the type of the member 'figure' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_230807, 'figure', figure_230806)
        
        # Assigning a Call to a Attribute (line 66):
        
        # Assigning a Call to a Attribute (line 66):
        
        # Call to get_dpi(...): (line 66)
        # Processing the call keyword arguments (line 66)
        kwargs_230810 = {}
        # Getting the type of 'figure' (line 66)
        figure_230808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'figure', False)
        # Obtaining the member 'get_dpi' of a type (line 66)
        get_dpi_230809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 23), figure_230808, 'get_dpi')
        # Calling get_dpi(args, kwargs) (line 66)
        get_dpi_call_result_230811 = invoke(stypy.reporting.localization.Localization(__file__, 66, 23), get_dpi_230809, *[], **kwargs_230810)
        
        # Getting the type of 'self' (line 66)
        self_230812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member '_figdpi' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_230812, '_figdpi', get_dpi_call_result_230811)
        
        # Assigning a Name to a Attribute (line 68):
        
        # Assigning a Name to a Attribute (line 68):
        # Getting the type of 'bbox_inches_restore' (line 68)
        bbox_inches_restore_230813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 36), 'bbox_inches_restore')
        # Getting the type of 'self' (line 68)
        self_230814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member '_bbox_inches_restore' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_230814, '_bbox_inches_restore', bbox_inches_restore_230813)
        
        # Call to _set_current_renderer(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'vector_renderer' (line 70)
        vector_renderer_230817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'vector_renderer', False)
        # Processing the call keyword arguments (line 70)
        kwargs_230818 = {}
        # Getting the type of 'self' (line 70)
        self_230815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self', False)
        # Obtaining the member '_set_current_renderer' of a type (line 70)
        _set_current_renderer_230816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_230815, '_set_current_renderer')
        # Calling _set_current_renderer(args, kwargs) (line 70)
        _set_current_renderer_call_result_230819 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), _set_current_renderer_230816, *[vector_renderer_230817], **kwargs_230818)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()

    
    # Assigning a Call to a Name (line 72):

    @norecursion
    def _set_current_renderer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_current_renderer'
        module_type_store = module_type_store.open_function_context('_set_current_renderer', 83, 4, False)
        # Assigning a type to the variable 'self' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MixedModeRenderer._set_current_renderer.__dict__.__setitem__('stypy_localization', localization)
        MixedModeRenderer._set_current_renderer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MixedModeRenderer._set_current_renderer.__dict__.__setitem__('stypy_type_store', module_type_store)
        MixedModeRenderer._set_current_renderer.__dict__.__setitem__('stypy_function_name', 'MixedModeRenderer._set_current_renderer')
        MixedModeRenderer._set_current_renderer.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        MixedModeRenderer._set_current_renderer.__dict__.__setitem__('stypy_varargs_param_name', None)
        MixedModeRenderer._set_current_renderer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MixedModeRenderer._set_current_renderer.__dict__.__setitem__('stypy_call_defaults', defaults)
        MixedModeRenderer._set_current_renderer.__dict__.__setitem__('stypy_call_varargs', varargs)
        MixedModeRenderer._set_current_renderer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MixedModeRenderer._set_current_renderer.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MixedModeRenderer._set_current_renderer', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_current_renderer', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_current_renderer(...)' code ##################

        
        # Assigning a Name to a Attribute (line 84):
        
        # Assigning a Name to a Attribute (line 84):
        # Getting the type of 'renderer' (line 84)
        renderer_230820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 'renderer')
        # Getting the type of 'self' (line 84)
        self_230821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self')
        # Setting the type of the member '_renderer' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_230821, '_renderer', renderer_230820)
        
        # Getting the type of 'self' (line 86)
        self_230822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 22), 'self')
        # Obtaining the member '_methods' of a type (line 86)
        _methods_230823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 22), self_230822, '_methods')
        # Testing the type of a for loop iterable (line 86)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 86, 8), _methods_230823)
        # Getting the type of the for loop variable (line 86)
        for_loop_var_230824 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 86, 8), _methods_230823)
        # Assigning a type to the variable 'method' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'method', for_loop_var_230824)
        # SSA begins for a for statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to hasattr(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'renderer' (line 87)
        renderer_230826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'renderer', False)
        # Getting the type of 'method' (line 87)
        method_230827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 33), 'method', False)
        # Processing the call keyword arguments (line 87)
        kwargs_230828 = {}
        # Getting the type of 'hasattr' (line 87)
        hasattr_230825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 87)
        hasattr_call_result_230829 = invoke(stypy.reporting.localization.Localization(__file__, 87, 15), hasattr_230825, *[renderer_230826, method_230827], **kwargs_230828)
        
        # Testing the type of an if condition (line 87)
        if_condition_230830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 12), hasattr_call_result_230829)
        # Assigning a type to the variable 'if_condition_230830' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'if_condition_230830', if_condition_230830)
        # SSA begins for if statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setattr(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'self' (line 88)
        self_230832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'self', False)
        # Getting the type of 'method' (line 88)
        method_230833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 30), 'method', False)
        
        # Call to getattr(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'renderer' (line 88)
        renderer_230835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 46), 'renderer', False)
        # Getting the type of 'method' (line 88)
        method_230836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 56), 'method', False)
        # Processing the call keyword arguments (line 88)
        kwargs_230837 = {}
        # Getting the type of 'getattr' (line 88)
        getattr_230834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 38), 'getattr', False)
        # Calling getattr(args, kwargs) (line 88)
        getattr_call_result_230838 = invoke(stypy.reporting.localization.Localization(__file__, 88, 38), getattr_230834, *[renderer_230835, method_230836], **kwargs_230837)
        
        # Processing the call keyword arguments (line 88)
        kwargs_230839 = {}
        # Getting the type of 'setattr' (line 88)
        setattr_230831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'setattr', False)
        # Calling setattr(args, kwargs) (line 88)
        setattr_call_result_230840 = invoke(stypy.reporting.localization.Localization(__file__, 88, 16), setattr_230831, *[self_230832, method_230833, getattr_call_result_230838], **kwargs_230839)
        
        # SSA join for if statement (line 87)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 89):
        
        # Assigning a Attribute to a Attribute (line 89):
        # Getting the type of 'self' (line 89)
        self_230841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 37), 'self')
        # Obtaining the member 'start_rasterizing' of a type (line 89)
        start_rasterizing_230842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 37), self_230841, 'start_rasterizing')
        # Getting the type of 'renderer' (line 89)
        renderer_230843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'renderer')
        # Setting the type of the member 'start_rasterizing' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), renderer_230843, 'start_rasterizing', start_rasterizing_230842)
        
        # Assigning a Attribute to a Attribute (line 90):
        
        # Assigning a Attribute to a Attribute (line 90):
        # Getting the type of 'self' (line 90)
        self_230844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 36), 'self')
        # Obtaining the member 'stop_rasterizing' of a type (line 90)
        stop_rasterizing_230845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 36), self_230844, 'stop_rasterizing')
        # Getting the type of 'renderer' (line 90)
        renderer_230846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'renderer')
        # Setting the type of the member 'stop_rasterizing' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), renderer_230846, 'stop_rasterizing', stop_rasterizing_230845)
        
        # ################# End of '_set_current_renderer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_current_renderer' in the type store
        # Getting the type of 'stypy_return_type' (line 83)
        stypy_return_type_230847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230847)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_current_renderer'
        return stypy_return_type_230847


    @norecursion
    def start_rasterizing(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'start_rasterizing'
        module_type_store = module_type_store.open_function_context('start_rasterizing', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MixedModeRenderer.start_rasterizing.__dict__.__setitem__('stypy_localization', localization)
        MixedModeRenderer.start_rasterizing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MixedModeRenderer.start_rasterizing.__dict__.__setitem__('stypy_type_store', module_type_store)
        MixedModeRenderer.start_rasterizing.__dict__.__setitem__('stypy_function_name', 'MixedModeRenderer.start_rasterizing')
        MixedModeRenderer.start_rasterizing.__dict__.__setitem__('stypy_param_names_list', [])
        MixedModeRenderer.start_rasterizing.__dict__.__setitem__('stypy_varargs_param_name', None)
        MixedModeRenderer.start_rasterizing.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MixedModeRenderer.start_rasterizing.__dict__.__setitem__('stypy_call_defaults', defaults)
        MixedModeRenderer.start_rasterizing.__dict__.__setitem__('stypy_call_varargs', varargs)
        MixedModeRenderer.start_rasterizing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MixedModeRenderer.start_rasterizing.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MixedModeRenderer.start_rasterizing', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'start_rasterizing', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'start_rasterizing(...)' code ##################

        unicode_230848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, (-1)), 'unicode', u'\n        Enter "raster" mode.  All subsequent drawing commands (until\n        stop_rasterizing is called) will be drawn with the raster\n        backend.\n\n        If start_rasterizing is called multiple times before\n        stop_rasterizing is called, this method has no effect.\n        ')
        
        # Call to set_dpi(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'self' (line 103)
        self_230852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 28), 'self', False)
        # Obtaining the member 'dpi' of a type (line 103)
        dpi_230853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 28), self_230852, 'dpi')
        # Processing the call keyword arguments (line 103)
        kwargs_230854 = {}
        # Getting the type of 'self' (line 103)
        self_230849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 103)
        figure_230850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_230849, 'figure')
        # Obtaining the member 'set_dpi' of a type (line 103)
        set_dpi_230851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), figure_230850, 'set_dpi')
        # Calling set_dpi(args, kwargs) (line 103)
        set_dpi_call_result_230855 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), set_dpi_230851, *[dpi_230853], **kwargs_230854)
        
        
        # Getting the type of 'self' (line 105)
        self_230856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'self')
        # Obtaining the member '_bbox_inches_restore' of a type (line 105)
        _bbox_inches_restore_230857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 11), self_230856, '_bbox_inches_restore')
        # Testing the type of an if condition (line 105)
        if_condition_230858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 8), _bbox_inches_restore_230857)
        # Assigning a type to the variable 'if_condition_230858' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'if_condition_230858', if_condition_230858)
        # SSA begins for if statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to process_figure_for_rasterizing(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'self' (line 106)
        self_230860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 47), 'self', False)
        # Obtaining the member 'figure' of a type (line 106)
        figure_230861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 47), self_230860, 'figure')
        # Getting the type of 'self' (line 107)
        self_230862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 47), 'self', False)
        # Obtaining the member '_bbox_inches_restore' of a type (line 107)
        _bbox_inches_restore_230863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 47), self_230862, '_bbox_inches_restore')
        # Processing the call keyword arguments (line 106)
        kwargs_230864 = {}
        # Getting the type of 'process_figure_for_rasterizing' (line 106)
        process_figure_for_rasterizing_230859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'process_figure_for_rasterizing', False)
        # Calling process_figure_for_rasterizing(args, kwargs) (line 106)
        process_figure_for_rasterizing_call_result_230865 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), process_figure_for_rasterizing_230859, *[figure_230861, _bbox_inches_restore_230863], **kwargs_230864)
        
        # Assigning a type to the variable 'r' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'r', process_figure_for_rasterizing_call_result_230865)
        
        # Assigning a Name to a Attribute (line 108):
        
        # Assigning a Name to a Attribute (line 108):
        # Getting the type of 'r' (line 108)
        r_230866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), 'r')
        # Getting the type of 'self' (line 108)
        self_230867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'self')
        # Setting the type of the member '_bbox_inches_restore' of a type (line 108)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), self_230867, '_bbox_inches_restore', r_230866)
        # SSA join for if statement (line 105)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 110)
        self_230868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'self')
        # Obtaining the member '_rasterizing' of a type (line 110)
        _rasterizing_230869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 11), self_230868, '_rasterizing')
        int_230870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 32), 'int')
        # Applying the binary operator '==' (line 110)
        result_eq_230871 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 11), '==', _rasterizing_230869, int_230870)
        
        # Testing the type of an if condition (line 110)
        if_condition_230872 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 8), result_eq_230871)
        # Assigning a type to the variable 'if_condition_230872' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'if_condition_230872', if_condition_230872)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 111):
        
        # Assigning a Call to a Attribute (line 111):
        
        # Call to _raster_renderer_class(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'self' (line 112)
        self_230875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'self', False)
        # Obtaining the member '_width' of a type (line 112)
        _width_230876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), self_230875, '_width')
        # Getting the type of 'self' (line 112)
        self_230877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'self', False)
        # Obtaining the member 'dpi' of a type (line 112)
        dpi_230878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 28), self_230877, 'dpi')
        # Applying the binary operator '*' (line 112)
        result_mul_230879 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 16), '*', _width_230876, dpi_230878)
        
        # Getting the type of 'self' (line 112)
        self_230880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 38), 'self', False)
        # Obtaining the member '_height' of a type (line 112)
        _height_230881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 38), self_230880, '_height')
        # Getting the type of 'self' (line 112)
        self_230882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 51), 'self', False)
        # Obtaining the member 'dpi' of a type (line 112)
        dpi_230883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 51), self_230882, 'dpi')
        # Applying the binary operator '*' (line 112)
        result_mul_230884 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 38), '*', _height_230881, dpi_230883)
        
        # Getting the type of 'self' (line 112)
        self_230885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 61), 'self', False)
        # Obtaining the member 'dpi' of a type (line 112)
        dpi_230886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 61), self_230885, 'dpi')
        # Processing the call keyword arguments (line 111)
        kwargs_230887 = {}
        # Getting the type of 'self' (line 111)
        self_230873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 36), 'self', False)
        # Obtaining the member '_raster_renderer_class' of a type (line 111)
        _raster_renderer_class_230874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 36), self_230873, '_raster_renderer_class')
        # Calling _raster_renderer_class(args, kwargs) (line 111)
        _raster_renderer_class_call_result_230888 = invoke(stypy.reporting.localization.Localization(__file__, 111, 36), _raster_renderer_class_230874, *[result_mul_230879, result_mul_230884, dpi_230886], **kwargs_230887)
        
        # Getting the type of 'self' (line 111)
        self_230889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'self')
        # Setting the type of the member '_raster_renderer' of a type (line 111)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), self_230889, '_raster_renderer', _raster_renderer_class_call_result_230888)
        
        # Call to _set_current_renderer(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'self' (line 113)
        self_230892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 39), 'self', False)
        # Obtaining the member '_raster_renderer' of a type (line 113)
        _raster_renderer_230893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 39), self_230892, '_raster_renderer')
        # Processing the call keyword arguments (line 113)
        kwargs_230894 = {}
        # Getting the type of 'self' (line 113)
        self_230890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'self', False)
        # Obtaining the member '_set_current_renderer' of a type (line 113)
        _set_current_renderer_230891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), self_230890, '_set_current_renderer')
        # Calling _set_current_renderer(args, kwargs) (line 113)
        _set_current_renderer_call_result_230895 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), _set_current_renderer_230891, *[_raster_renderer_230893], **kwargs_230894)
        
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 114)
        self_230896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self')
        # Obtaining the member '_rasterizing' of a type (line 114)
        _rasterizing_230897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_230896, '_rasterizing')
        int_230898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 29), 'int')
        # Applying the binary operator '+=' (line 114)
        result_iadd_230899 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 8), '+=', _rasterizing_230897, int_230898)
        # Getting the type of 'self' (line 114)
        self_230900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self')
        # Setting the type of the member '_rasterizing' of a type (line 114)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_230900, '_rasterizing', result_iadd_230899)
        
        
        # ################# End of 'start_rasterizing(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'start_rasterizing' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_230901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230901)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'start_rasterizing'
        return stypy_return_type_230901


    @norecursion
    def stop_rasterizing(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'stop_rasterizing'
        module_type_store = module_type_store.open_function_context('stop_rasterizing', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MixedModeRenderer.stop_rasterizing.__dict__.__setitem__('stypy_localization', localization)
        MixedModeRenderer.stop_rasterizing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MixedModeRenderer.stop_rasterizing.__dict__.__setitem__('stypy_type_store', module_type_store)
        MixedModeRenderer.stop_rasterizing.__dict__.__setitem__('stypy_function_name', 'MixedModeRenderer.stop_rasterizing')
        MixedModeRenderer.stop_rasterizing.__dict__.__setitem__('stypy_param_names_list', [])
        MixedModeRenderer.stop_rasterizing.__dict__.__setitem__('stypy_varargs_param_name', None)
        MixedModeRenderer.stop_rasterizing.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MixedModeRenderer.stop_rasterizing.__dict__.__setitem__('stypy_call_defaults', defaults)
        MixedModeRenderer.stop_rasterizing.__dict__.__setitem__('stypy_call_varargs', varargs)
        MixedModeRenderer.stop_rasterizing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MixedModeRenderer.stop_rasterizing.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MixedModeRenderer.stop_rasterizing', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'stop_rasterizing', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'stop_rasterizing(...)' code ##################

        unicode_230902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, (-1)), 'unicode', u'\n        Exit "raster" mode.  All of the drawing that was done since\n        the last start_rasterizing command will be copied to the\n        vector backend by calling draw_image.\n\n        If stop_rasterizing is called multiple times before\n        start_rasterizing is called, this method has no effect.\n        ')
        
        # Getting the type of 'self' (line 125)
        self_230903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self')
        # Obtaining the member '_rasterizing' of a type (line 125)
        _rasterizing_230904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_230903, '_rasterizing')
        int_230905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 29), 'int')
        # Applying the binary operator '-=' (line 125)
        result_isub_230906 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 8), '-=', _rasterizing_230904, int_230905)
        # Getting the type of 'self' (line 125)
        self_230907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self')
        # Setting the type of the member '_rasterizing' of a type (line 125)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_230907, '_rasterizing', result_isub_230906)
        
        
        
        # Getting the type of 'self' (line 126)
        self_230908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'self')
        # Obtaining the member '_rasterizing' of a type (line 126)
        _rasterizing_230909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 11), self_230908, '_rasterizing')
        int_230910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 32), 'int')
        # Applying the binary operator '==' (line 126)
        result_eq_230911 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 11), '==', _rasterizing_230909, int_230910)
        
        # Testing the type of an if condition (line 126)
        if_condition_230912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 8), result_eq_230911)
        # Assigning a type to the variable 'if_condition_230912' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'if_condition_230912', if_condition_230912)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _set_current_renderer(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'self' (line 127)
        self_230915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 39), 'self', False)
        # Obtaining the member '_vector_renderer' of a type (line 127)
        _vector_renderer_230916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 39), self_230915, '_vector_renderer')
        # Processing the call keyword arguments (line 127)
        kwargs_230917 = {}
        # Getting the type of 'self' (line 127)
        self_230913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'self', False)
        # Obtaining the member '_set_current_renderer' of a type (line 127)
        _set_current_renderer_230914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), self_230913, '_set_current_renderer')
        # Calling _set_current_renderer(args, kwargs) (line 127)
        _set_current_renderer_call_result_230918 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), _set_current_renderer_230914, *[_vector_renderer_230916], **kwargs_230917)
        
        
        # Assigning a BinOp to a Name (line 129):
        
        # Assigning a BinOp to a Name (line 129):
        # Getting the type of 'self' (line 129)
        self_230919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'self')
        # Obtaining the member '_height' of a type (line 129)
        _height_230920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 21), self_230919, '_height')
        # Getting the type of 'self' (line 129)
        self_230921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 36), 'self')
        # Obtaining the member 'dpi' of a type (line 129)
        dpi_230922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 36), self_230921, 'dpi')
        # Applying the binary operator '*' (line 129)
        result_mul_230923 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 21), '*', _height_230920, dpi_230922)
        
        # Assigning a type to the variable 'height' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'height', result_mul_230923)
        
        # Assigning a Call to a Tuple (line 130):
        
        # Assigning a Call to a Name:
        
        # Call to tostring_rgba_minimized(...): (line 130)
        # Processing the call keyword arguments (line 130)
        kwargs_230927 = {}
        # Getting the type of 'self' (line 130)
        self_230924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 29), 'self', False)
        # Obtaining the member '_raster_renderer' of a type (line 130)
        _raster_renderer_230925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 29), self_230924, '_raster_renderer')
        # Obtaining the member 'tostring_rgba_minimized' of a type (line 130)
        tostring_rgba_minimized_230926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 29), _raster_renderer_230925, 'tostring_rgba_minimized')
        # Calling tostring_rgba_minimized(args, kwargs) (line 130)
        tostring_rgba_minimized_call_result_230928 = invoke(stypy.reporting.localization.Localization(__file__, 130, 29), tostring_rgba_minimized_230926, *[], **kwargs_230927)
        
        # Assigning a type to the variable 'call_assignment_230768' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'call_assignment_230768', tostring_rgba_minimized_call_result_230928)
        
        # Assigning a Call to a Name (line 130):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_230931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 12), 'int')
        # Processing the call keyword arguments
        kwargs_230932 = {}
        # Getting the type of 'call_assignment_230768' (line 130)
        call_assignment_230768_230929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'call_assignment_230768', False)
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___230930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), call_assignment_230768_230929, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_230933 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___230930, *[int_230931], **kwargs_230932)
        
        # Assigning a type to the variable 'call_assignment_230769' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'call_assignment_230769', getitem___call_result_230933)
        
        # Assigning a Name to a Name (line 130):
        # Getting the type of 'call_assignment_230769' (line 130)
        call_assignment_230769_230934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'call_assignment_230769')
        # Assigning a type to the variable 'buffer' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'buffer', call_assignment_230769_230934)
        
        # Assigning a Call to a Name (line 130):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_230937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 12), 'int')
        # Processing the call keyword arguments
        kwargs_230938 = {}
        # Getting the type of 'call_assignment_230768' (line 130)
        call_assignment_230768_230935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'call_assignment_230768', False)
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___230936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), call_assignment_230768_230935, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_230939 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___230936, *[int_230937], **kwargs_230938)
        
        # Assigning a type to the variable 'call_assignment_230770' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'call_assignment_230770', getitem___call_result_230939)
        
        # Assigning a Name to a Name (line 130):
        # Getting the type of 'call_assignment_230770' (line 130)
        call_assignment_230770_230940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'call_assignment_230770')
        # Assigning a type to the variable 'bounds' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'bounds', call_assignment_230770_230940)
        
        # Assigning a Name to a Tuple (line 131):
        
        # Assigning a Subscript to a Name (line 131):
        
        # Obtaining the type of the subscript
        int_230941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 12), 'int')
        # Getting the type of 'bounds' (line 131)
        bounds_230942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'bounds')
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___230943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), bounds_230942, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_230944 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), getitem___230943, int_230941)
        
        # Assigning a type to the variable 'tuple_var_assignment_230771' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'tuple_var_assignment_230771', subscript_call_result_230944)
        
        # Assigning a Subscript to a Name (line 131):
        
        # Obtaining the type of the subscript
        int_230945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 12), 'int')
        # Getting the type of 'bounds' (line 131)
        bounds_230946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'bounds')
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___230947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), bounds_230946, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_230948 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), getitem___230947, int_230945)
        
        # Assigning a type to the variable 'tuple_var_assignment_230772' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'tuple_var_assignment_230772', subscript_call_result_230948)
        
        # Assigning a Subscript to a Name (line 131):
        
        # Obtaining the type of the subscript
        int_230949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 12), 'int')
        # Getting the type of 'bounds' (line 131)
        bounds_230950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'bounds')
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___230951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), bounds_230950, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_230952 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), getitem___230951, int_230949)
        
        # Assigning a type to the variable 'tuple_var_assignment_230773' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'tuple_var_assignment_230773', subscript_call_result_230952)
        
        # Assigning a Subscript to a Name (line 131):
        
        # Obtaining the type of the subscript
        int_230953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 12), 'int')
        # Getting the type of 'bounds' (line 131)
        bounds_230954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'bounds')
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___230955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), bounds_230954, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_230956 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), getitem___230955, int_230953)
        
        # Assigning a type to the variable 'tuple_var_assignment_230774' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'tuple_var_assignment_230774', subscript_call_result_230956)
        
        # Assigning a Name to a Name (line 131):
        # Getting the type of 'tuple_var_assignment_230771' (line 131)
        tuple_var_assignment_230771_230957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'tuple_var_assignment_230771')
        # Assigning a type to the variable 'l' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'l', tuple_var_assignment_230771_230957)
        
        # Assigning a Name to a Name (line 131):
        # Getting the type of 'tuple_var_assignment_230772' (line 131)
        tuple_var_assignment_230772_230958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'tuple_var_assignment_230772')
        # Assigning a type to the variable 'b' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'b', tuple_var_assignment_230772_230958)
        
        # Assigning a Name to a Name (line 131):
        # Getting the type of 'tuple_var_assignment_230773' (line 131)
        tuple_var_assignment_230773_230959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'tuple_var_assignment_230773')
        # Assigning a type to the variable 'w' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 18), 'w', tuple_var_assignment_230773_230959)
        
        # Assigning a Name to a Name (line 131):
        # Getting the type of 'tuple_var_assignment_230774' (line 131)
        tuple_var_assignment_230774_230960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'tuple_var_assignment_230774')
        # Assigning a type to the variable 'h' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'h', tuple_var_assignment_230774_230960)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'w' (line 132)
        w_230961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'w')
        int_230962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 19), 'int')
        # Applying the binary operator '>' (line 132)
        result_gt_230963 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 15), '>', w_230961, int_230962)
        
        
        # Getting the type of 'h' (line 132)
        h_230964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 25), 'h')
        int_230965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 29), 'int')
        # Applying the binary operator '>' (line 132)
        result_gt_230966 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 25), '>', h_230964, int_230965)
        
        # Applying the binary operator 'and' (line 132)
        result_and_keyword_230967 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 15), 'and', result_gt_230963, result_gt_230966)
        
        # Testing the type of an if condition (line 132)
        if_condition_230968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 12), result_and_keyword_230967)
        # Assigning a type to the variable 'if_condition_230968' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'if_condition_230968', if_condition_230968)
        # SSA begins for if statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 133):
        
        # Assigning a Call to a Name (line 133):
        
        # Call to frombuffer(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'buffer' (line 133)
        buffer_230971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 38), 'buffer', False)
        # Processing the call keyword arguments (line 133)
        # Getting the type of 'np' (line 133)
        np_230972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 52), 'np', False)
        # Obtaining the member 'uint8' of a type (line 133)
        uint8_230973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 52), np_230972, 'uint8')
        keyword_230974 = uint8_230973
        kwargs_230975 = {'dtype': keyword_230974}
        # Getting the type of 'np' (line 133)
        np_230969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'np', False)
        # Obtaining the member 'frombuffer' of a type (line 133)
        frombuffer_230970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 24), np_230969, 'frombuffer')
        # Calling frombuffer(args, kwargs) (line 133)
        frombuffer_call_result_230976 = invoke(stypy.reporting.localization.Localization(__file__, 133, 24), frombuffer_230970, *[buffer_230971], **kwargs_230975)
        
        # Assigning a type to the variable 'image' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'image', frombuffer_call_result_230976)
        
        # Assigning a Call to a Name (line 134):
        
        # Assigning a Call to a Name (line 134):
        
        # Call to reshape(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Obtaining an instance of the builtin type 'tuple' (line 134)
        tuple_230979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 134)
        # Adding element type (line 134)
        # Getting the type of 'h' (line 134)
        h_230980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 39), 'h', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 39), tuple_230979, h_230980)
        # Adding element type (line 134)
        # Getting the type of 'w' (line 134)
        w_230981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 42), 'w', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 39), tuple_230979, w_230981)
        # Adding element type (line 134)
        int_230982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 39), tuple_230979, int_230982)
        
        # Processing the call keyword arguments (line 134)
        kwargs_230983 = {}
        # Getting the type of 'image' (line 134)
        image_230977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 'image', False)
        # Obtaining the member 'reshape' of a type (line 134)
        reshape_230978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 24), image_230977, 'reshape')
        # Calling reshape(args, kwargs) (line 134)
        reshape_call_result_230984 = invoke(stypy.reporting.localization.Localization(__file__, 134, 24), reshape_230978, *[tuple_230979], **kwargs_230983)
        
        # Assigning a type to the variable 'image' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'image', reshape_call_result_230984)
        
        # Assigning a Subscript to a Name (line 135):
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_230985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 32), 'int')
        slice_230986 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 135, 24), None, None, int_230985)
        # Getting the type of 'image' (line 135)
        image_230987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'image')
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___230988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 24), image_230987, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_230989 = invoke(stypy.reporting.localization.Localization(__file__, 135, 24), getitem___230988, slice_230986)
        
        # Assigning a type to the variable 'image' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'image', subscript_call_result_230989)
        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to new_gc(...): (line 136)
        # Processing the call keyword arguments (line 136)
        kwargs_230993 = {}
        # Getting the type of 'self' (line 136)
        self_230990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 21), 'self', False)
        # Obtaining the member '_renderer' of a type (line 136)
        _renderer_230991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 21), self_230990, '_renderer')
        # Obtaining the member 'new_gc' of a type (line 136)
        new_gc_230992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 21), _renderer_230991, 'new_gc')
        # Calling new_gc(args, kwargs) (line 136)
        new_gc_call_result_230994 = invoke(stypy.reporting.localization.Localization(__file__, 136, 21), new_gc_230992, *[], **kwargs_230993)
        
        # Assigning a type to the variable 'gc' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'gc', new_gc_call_result_230994)
        
        # Call to draw_image(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'gc' (line 141)
        gc_230998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'gc', False)
        
        # Call to float(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'l' (line 142)
        l_231000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'l', False)
        # Processing the call keyword arguments (line 142)
        kwargs_231001 = {}
        # Getting the type of 'float' (line 142)
        float_230999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'float', False)
        # Calling float(args, kwargs) (line 142)
        float_call_result_231002 = invoke(stypy.reporting.localization.Localization(__file__, 142, 20), float_230999, *[l_231000], **kwargs_231001)
        
        # Getting the type of 'self' (line 142)
        self_231003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), 'self', False)
        # Obtaining the member 'dpi' of a type (line 142)
        dpi_231004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 31), self_231003, 'dpi')
        # Applying the binary operator 'div' (line 142)
        result_div_231005 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 20), 'div', float_call_result_231002, dpi_231004)
        
        # Getting the type of 'self' (line 142)
        self_231006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'self', False)
        # Obtaining the member '_figdpi' of a type (line 142)
        _figdpi_231007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 42), self_231006, '_figdpi')
        # Applying the binary operator '*' (line 142)
        result_mul_231008 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 40), '*', result_div_231005, _figdpi_231007)
        
        
        # Call to float(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'height' (line 143)
        height_231010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 27), 'height', False)
        # Processing the call keyword arguments (line 143)
        kwargs_231011 = {}
        # Getting the type of 'float' (line 143)
        float_231009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 21), 'float', False)
        # Calling float(args, kwargs) (line 143)
        float_call_result_231012 = invoke(stypy.reporting.localization.Localization(__file__, 143, 21), float_231009, *[height_231010], **kwargs_231011)
        
        # Getting the type of 'b' (line 143)
        b_231013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 35), 'b', False)
        # Applying the binary operator '-' (line 143)
        result_sub_231014 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 21), '-', float_call_result_231012, b_231013)
        
        # Getting the type of 'h' (line 143)
        h_231015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 37), 'h', False)
        # Applying the binary operator '-' (line 143)
        result_sub_231016 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 36), '-', result_sub_231014, h_231015)
        
        # Getting the type of 'self' (line 143)
        self_231017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 42), 'self', False)
        # Obtaining the member 'dpi' of a type (line 143)
        dpi_231018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 42), self_231017, 'dpi')
        # Applying the binary operator 'div' (line 143)
        result_div_231019 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 20), 'div', result_sub_231016, dpi_231018)
        
        # Getting the type of 'self' (line 143)
        self_231020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 53), 'self', False)
        # Obtaining the member '_figdpi' of a type (line 143)
        _figdpi_231021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 53), self_231020, '_figdpi')
        # Applying the binary operator '*' (line 143)
        result_mul_231022 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 51), '*', result_div_231019, _figdpi_231021)
        
        # Getting the type of 'image' (line 144)
        image_231023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'image', False)
        # Processing the call keyword arguments (line 140)
        kwargs_231024 = {}
        # Getting the type of 'self' (line 140)
        self_230995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'self', False)
        # Obtaining the member '_renderer' of a type (line 140)
        _renderer_230996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 16), self_230995, '_renderer')
        # Obtaining the member 'draw_image' of a type (line 140)
        draw_image_230997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 16), _renderer_230996, 'draw_image')
        # Calling draw_image(args, kwargs) (line 140)
        draw_image_call_result_231025 = invoke(stypy.reporting.localization.Localization(__file__, 140, 16), draw_image_230997, *[gc_230998, result_mul_231008, result_mul_231022, image_231023], **kwargs_231024)
        
        # SSA join for if statement (line 132)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 145):
        
        # Assigning a Name to a Attribute (line 145):
        # Getting the type of 'None' (line 145)
        None_231026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 36), 'None')
        # Getting the type of 'self' (line 145)
        self_231027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'self')
        # Setting the type of the member '_raster_renderer' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), self_231027, '_raster_renderer', None_231026)
        
        # Assigning a Name to a Attribute (line 146):
        
        # Assigning a Name to a Attribute (line 146):
        # Getting the type of 'False' (line 146)
        False_231028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 32), 'False')
        # Getting the type of 'self' (line 146)
        self_231029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'self')
        # Setting the type of the member '_rasterizing' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), self_231029, '_rasterizing', False_231028)
        
        # Call to set_dpi(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'self' (line 149)
        self_231033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 32), 'self', False)
        # Obtaining the member '_figdpi' of a type (line 149)
        _figdpi_231034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 32), self_231033, '_figdpi')
        # Processing the call keyword arguments (line 149)
        kwargs_231035 = {}
        # Getting the type of 'self' (line 149)
        self_231030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'self', False)
        # Obtaining the member 'figure' of a type (line 149)
        figure_231031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), self_231030, 'figure')
        # Obtaining the member 'set_dpi' of a type (line 149)
        set_dpi_231032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), figure_231031, 'set_dpi')
        # Calling set_dpi(args, kwargs) (line 149)
        set_dpi_call_result_231036 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), set_dpi_231032, *[_figdpi_231034], **kwargs_231035)
        
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 151)
        self_231037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'self')
        # Obtaining the member '_bbox_inches_restore' of a type (line 151)
        _bbox_inches_restore_231038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), self_231037, '_bbox_inches_restore')
        # Testing the type of an if condition (line 151)
        if_condition_231039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), _bbox_inches_restore_231038)
        # Assigning a type to the variable 'if_condition_231039' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_231039', if_condition_231039)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to process_figure_for_rasterizing(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'self' (line 152)
        self_231041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 47), 'self', False)
        # Obtaining the member 'figure' of a type (line 152)
        figure_231042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 47), self_231041, 'figure')
        # Getting the type of 'self' (line 153)
        self_231043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 47), 'self', False)
        # Obtaining the member '_bbox_inches_restore' of a type (line 153)
        _bbox_inches_restore_231044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 47), self_231043, '_bbox_inches_restore')
        # Getting the type of 'self' (line 154)
        self_231045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 47), 'self', False)
        # Obtaining the member '_figdpi' of a type (line 154)
        _figdpi_231046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 47), self_231045, '_figdpi')
        # Processing the call keyword arguments (line 152)
        kwargs_231047 = {}
        # Getting the type of 'process_figure_for_rasterizing' (line 152)
        process_figure_for_rasterizing_231040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'process_figure_for_rasterizing', False)
        # Calling process_figure_for_rasterizing(args, kwargs) (line 152)
        process_figure_for_rasterizing_call_result_231048 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), process_figure_for_rasterizing_231040, *[figure_231042, _bbox_inches_restore_231044, _figdpi_231046], **kwargs_231047)
        
        # Assigning a type to the variable 'r' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'r', process_figure_for_rasterizing_call_result_231048)
        
        # Assigning a Name to a Attribute (line 155):
        
        # Assigning a Name to a Attribute (line 155):
        # Getting the type of 'r' (line 155)
        r_231049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 40), 'r')
        # Getting the type of 'self' (line 155)
        self_231050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'self')
        # Setting the type of the member '_bbox_inches_restore' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), self_231050, '_bbox_inches_restore', r_231049)
        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'stop_rasterizing(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'stop_rasterizing' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_231051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231051)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'stop_rasterizing'
        return stypy_return_type_231051


# Assigning a type to the variable 'MixedModeRenderer' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'MixedModeRenderer', MixedModeRenderer)

# Assigning a Call to a Name (line 72):

# Call to split(...): (line 81)
# Processing the call keyword arguments (line 81)
kwargs_231054 = {}
unicode_231052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'unicode', u'\n        close_group draw_image draw_markers draw_path\n        draw_path_collection draw_quad_mesh draw_tex draw_text\n        finalize flipy get_canvas_width_height get_image_magnification\n        get_texmanager get_text_width_height_descent new_gc open_group\n        option_image_nocomposite points_to_pixels strip_math\n        start_filter stop_filter draw_gouraud_triangle\n        draw_gouraud_triangles option_scale_image\n        _text2path _get_text_path_transform height width\n        ')
# Obtaining the member 'split' of a type (line 81)
split_231053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, (-1)), unicode_231052, 'split')
# Calling split(args, kwargs) (line 81)
split_call_result_231055 = invoke(stypy.reporting.localization.Localization(__file__, 81, (-1)), split_231053, *[], **kwargs_231054)

# Getting the type of 'MixedModeRenderer'
MixedModeRenderer_231056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MixedModeRenderer')
# Setting the type of the member '_methods' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MixedModeRenderer_231056, '_methods', split_call_result_231055)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
