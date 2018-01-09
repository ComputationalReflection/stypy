
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This is a fully functional do nothing backend to provide a template to
3: backend writers.  It is fully functional in that you can select it as
4: a backend with
5: 
6:   import matplotlib
7:   matplotlib.use('Template')
8: 
9: and your matplotlib scripts will (should!) run without error, though
10: no output is produced.  This provides a nice starting point for
11: backend writers because you can selectively implement methods
12: (draw_rectangle, draw_lines, etc...) and slowly see your figure come
13: to life w/o having to have a full blown implementation before getting
14: any results.
15: 
16: Copy this to backend_xxx.py and replace all instances of 'template'
17: with 'xxx'.  Then implement the class methods and functions below, and
18: add 'xxx' to the switchyard in matplotlib/backends/__init__.py and
19: 'xxx' to the backends list in the validate_backend methon in
20: matplotlib/__init__.py and you're off.  You can use your backend with::
21: 
22:   import matplotlib
23:   matplotlib.use('xxx')
24:   from pylab import *
25:   plot([1,2,3])
26:   show()
27: 
28: matplotlib also supports external backends, so you can place you can
29: use any module in your PYTHONPATH with the syntax::
30: 
31:   import matplotlib
32:   matplotlib.use('module://my_backend')
33: 
34: where my_backend.py is your module name.  This syntax is also
35: recognized in the rc file and in the -d argument in pylab, e.g.,::
36: 
37:   python simple_plot.py -dmodule://my_backend
38: 
39: If your backend implements support for saving figures (i.e. has a print_xyz()
40: method) you can register it as the default handler for a given file type
41: 
42:   from matplotlib.backend_bases import register_backend
43:   register_backend('xyz', 'my_backend', 'XYZ File Format')
44:   ...
45:   plt.savefig("figure.xyz")
46: 
47: The files that are most relevant to backend_writers are
48: 
49:   matplotlib/backends/backend_your_backend.py
50:   matplotlib/backend_bases.py
51:   matplotlib/backends/__init__.py
52:   matplotlib/__init__.py
53:   matplotlib/_pylab_helpers.py
54: 
55: Naming Conventions
56: 
57:   * classes Upper or MixedUpperCase
58: 
59:   * varables lower or lowerUpper
60: 
61:   * functions lower or underscore_separated
62: 
63: '''
64: 
65: from __future__ import (absolute_import, division, print_function,
66:                         unicode_literals)
67: 
68: import six
69: 
70: import matplotlib
71: from matplotlib._pylab_helpers import Gcf
72: from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
73:      FigureManagerBase, FigureCanvasBase
74: from matplotlib.figure import Figure
75: from matplotlib.transforms import Bbox
76: 
77: 
78: class RendererTemplate(RendererBase):
79:     '''
80:     The renderer handles drawing/rendering operations.
81: 
82:     This is a minimal do-nothing class that can be used to get started when
83:     writing a new backend. Refer to backend_bases.RendererBase for
84:     documentation of the classes methods.
85:     '''
86:     def __init__(self, dpi):
87:         self.dpi = dpi
88: 
89:     def draw_path(self, gc, path, transform, rgbFace=None):
90:         pass
91: 
92:     # draw_markers is optional, and we get more correct relative
93:     # timings by leaving it out.  backend implementers concerned with
94:     # performance will probably want to implement it
95: #     def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
96: #         pass
97: 
98:     # draw_path_collection is optional, and we get more correct
99:     # relative timings by leaving it out. backend implementers concerned with
100:     # performance will probably want to implement it
101: #     def draw_path_collection(self, gc, master_transform, paths,
102: #                              all_transforms, offsets, offsetTrans, facecolors,
103: #                              edgecolors, linewidths, linestyles,
104: #                              antialiaseds):
105: #         pass
106: 
107:     # draw_quad_mesh is optional, and we get more correct
108:     # relative timings by leaving it out.  backend implementers concerned with
109:     # performance will probably want to implement it
110: #     def draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight,
111: #                        coordinates, offsets, offsetTrans, facecolors,
112: #                        antialiased, edgecolors):
113: #         pass
114: 
115:     def draw_image(self, gc, x, y, im):
116:         pass
117: 
118:     def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
119:         pass
120: 
121:     def flipy(self):
122:         return True
123: 
124:     def get_canvas_width_height(self):
125:         return 100, 100
126: 
127:     def get_text_width_height_descent(self, s, prop, ismath):
128:         return 1, 1, 1
129: 
130:     def new_gc(self):
131:         return GraphicsContextTemplate()
132: 
133:     def points_to_pixels(self, points):
134:         # if backend doesn't have dpi, e.g., postscript or svg
135:         return points
136:         # elif backend assumes a value for pixels_per_inch
137:         #return points/72.0 * self.dpi.get() * pixels_per_inch/72.0
138:         # else
139:         #return points/72.0 * self.dpi.get()
140: 
141: 
142: class GraphicsContextTemplate(GraphicsContextBase):
143:     '''
144:     The graphics context provides the color, line styles, etc...  See the gtk
145:     and postscript backends for examples of mapping the graphics context
146:     attributes (cap styles, join styles, line widths, colors) to a particular
147:     backend.  In GTK this is done by wrapping a gtk.gdk.GC object and
148:     forwarding the appropriate calls to it using a dictionary mapping styles
149:     to gdk constants.  In Postscript, all the work is done by the renderer,
150:     mapping line styles to postscript calls.
151: 
152:     If it's more appropriate to do the mapping at the renderer level (as in
153:     the postscript backend), you don't need to override any of the GC methods.
154:     If it's more appropriate to wrap an instance (as in the GTK backend) and
155:     do the mapping here, you'll need to override several of the setter
156:     methods.
157: 
158:     The base GraphicsContext stores colors as a RGB tuple on the unit
159:     interval, e.g., (0.5, 0.0, 1.0). You may need to map this to colors
160:     appropriate for your backend.
161:     '''
162:     pass
163: 
164: 
165: 
166: ########################################################################
167: #
168: # The following functions and classes are for pylab and implement
169: # window/figure managers, etc...
170: #
171: ########################################################################
172: 
173: def draw_if_interactive():
174:     '''
175:     For image backends - is not required
176:     For GUI backends - this should be overridden if drawing should be done in
177:     interactive python mode
178:     '''
179:     # May be implemented via the `_draw_if_interactive_template` helper.
180: 
181: 
182: def show():
183:     '''
184:     For image backends - is not required
185:     For GUI backends - show() is usually the last line of a pylab script and
186:     tells the backend that it is time to draw.  In interactive mode, this may
187:     be a do nothing func.  See the GTK backend for an example of how to handle
188:     interactive versus batch mode
189:     '''
190:     for manager in Gcf.get_all_fig_managers():
191:         # do something to display the GUI
192:         pass
193: 
194: 
195: def new_figure_manager(num, *args, **kwargs):
196:     '''
197:     Create a new figure manager instance
198:     '''
199:     # May be implemented via the `_new_figure_manager_template` helper.
200:     # If a main-level app must be created, this (and
201:     # new_figure_manager_given_figure) is the usual place to do it -- see
202:     # backend_wx, backend_wxagg and backend_tkagg for examples.  Not all GUIs
203:     # require explicit instantiation of a main-level app (egg backend_gtk,
204:     # backend_gtkagg) for pylab.
205:     FigureClass = kwargs.pop('FigureClass', Figure)
206:     thisFig = FigureClass(*args, **kwargs)
207:     return new_figure_manager_given_figure(num, thisFig)
208: 
209: 
210: def new_figure_manager_given_figure(num, figure):
211:     '''
212:     Create a new figure manager instance for the given figure.
213:     '''
214:     # May be implemented via the `_new_figure_manager_template` helper.
215:     canvas = FigureCanvasTemplate(figure)
216:     manager = FigureManagerTemplate(canvas, num)
217:     return manager
218: 
219: 
220: class FigureCanvasTemplate(FigureCanvasBase):
221:     '''
222:     The canvas the figure renders into.  Calls the draw and print fig
223:     methods, creates the renderers, etc...
224: 
225:     Note GUI templates will want to connect events for button presses,
226:     mouse movements and key presses to functions that call the base
227:     class methods button_press_event, button_release_event,
228:     motion_notify_event, key_press_event, and key_release_event.  See,
229:     e.g., backend_gtk.py, backend_wx.py and backend_tkagg.py
230: 
231:     Attributes
232:     ----------
233:     figure : `matplotlib.figure.Figure`
234:         A high-level Figure instance
235: 
236:     '''
237: 
238:     def draw(self):
239:         '''
240:         Draw the figure using the renderer
241:         '''
242:         renderer = RendererTemplate(self.figure.dpi)
243:         self.figure.draw(renderer)
244: 
245:     # You should provide a print_xxx function for every file format
246:     # you can write.
247: 
248:     # If the file type is not in the base set of filetypes,
249:     # you should add it to the class-scope filetypes dictionary as follows:
250:     filetypes = FigureCanvasBase.filetypes.copy()
251:     filetypes['foo'] = 'My magic Foo format'
252: 
253:     def print_foo(self, filename, *args, **kwargs):
254:         '''
255:         Write out format foo.  The dpi, facecolor and edgecolor are restored
256:         to their original values after this call, so you don't need to
257:         save and restore them.
258:         '''
259:         pass
260: 
261:     def get_default_filetype(self):
262:         return 'foo'
263: 
264: 
265: class FigureManagerTemplate(FigureManagerBase):
266:     '''
267:     Wrap everything up into a window for the pylab interface
268: 
269:     For non interactive backends, the base class does all the work
270:     '''
271:     pass
272: 
273: ########################################################################
274: #
275: # Now just provide the standard names that backend.__init__ is expecting
276: #
277: ########################################################################
278: 
279: FigureCanvas = FigureCanvasTemplate
280: FigureManager = FigureManagerTemplate
281: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_257329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, (-1)), 'unicode', u'\nThis is a fully functional do nothing backend to provide a template to\nbackend writers.  It is fully functional in that you can select it as\na backend with\n\n  import matplotlib\n  matplotlib.use(\'Template\')\n\nand your matplotlib scripts will (should!) run without error, though\nno output is produced.  This provides a nice starting point for\nbackend writers because you can selectively implement methods\n(draw_rectangle, draw_lines, etc...) and slowly see your figure come\nto life w/o having to have a full blown implementation before getting\nany results.\n\nCopy this to backend_xxx.py and replace all instances of \'template\'\nwith \'xxx\'.  Then implement the class methods and functions below, and\nadd \'xxx\' to the switchyard in matplotlib/backends/__init__.py and\n\'xxx\' to the backends list in the validate_backend methon in\nmatplotlib/__init__.py and you\'re off.  You can use your backend with::\n\n  import matplotlib\n  matplotlib.use(\'xxx\')\n  from pylab import *\n  plot([1,2,3])\n  show()\n\nmatplotlib also supports external backends, so you can place you can\nuse any module in your PYTHONPATH with the syntax::\n\n  import matplotlib\n  matplotlib.use(\'module://my_backend\')\n\nwhere my_backend.py is your module name.  This syntax is also\nrecognized in the rc file and in the -d argument in pylab, e.g.,::\n\n  python simple_plot.py -dmodule://my_backend\n\nIf your backend implements support for saving figures (i.e. has a print_xyz()\nmethod) you can register it as the default handler for a given file type\n\n  from matplotlib.backend_bases import register_backend\n  register_backend(\'xyz\', \'my_backend\', \'XYZ File Format\')\n  ...\n  plt.savefig("figure.xyz")\n\nThe files that are most relevant to backend_writers are\n\n  matplotlib/backends/backend_your_backend.py\n  matplotlib/backend_bases.py\n  matplotlib/backends/__init__.py\n  matplotlib/__init__.py\n  matplotlib/_pylab_helpers.py\n\nNaming Conventions\n\n  * classes Upper or MixedUpperCase\n\n  * varables lower or lowerUpper\n\n  * functions lower or underscore_separated\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 68, 0))

# 'import six' statement (line 68)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_257330 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 68, 0), 'six')

if (type(import_257330) is not StypyTypeError):

    if (import_257330 != 'pyd_module'):
        __import__(import_257330)
        sys_modules_257331 = sys.modules[import_257330]
        import_module(stypy.reporting.localization.Localization(__file__, 68, 0), 'six', sys_modules_257331.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 68, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'six', import_257330)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 70, 0))

# 'import matplotlib' statement (line 70)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_257332 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 70, 0), 'matplotlib')

if (type(import_257332) is not StypyTypeError):

    if (import_257332 != 'pyd_module'):
        __import__(import_257332)
        sys_modules_257333 = sys.modules[import_257332]
        import_module(stypy.reporting.localization.Localization(__file__, 70, 0), 'matplotlib', sys_modules_257333.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 70, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'matplotlib', import_257332)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 71, 0))

# 'from matplotlib._pylab_helpers import Gcf' statement (line 71)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_257334 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 71, 0), 'matplotlib._pylab_helpers')

if (type(import_257334) is not StypyTypeError):

    if (import_257334 != 'pyd_module'):
        __import__(import_257334)
        sys_modules_257335 = sys.modules[import_257334]
        import_from_module(stypy.reporting.localization.Localization(__file__, 71, 0), 'matplotlib._pylab_helpers', sys_modules_257335.module_type_store, module_type_store, ['Gcf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 71, 0), __file__, sys_modules_257335, sys_modules_257335.module_type_store, module_type_store)
    else:
        from matplotlib._pylab_helpers import Gcf

        import_from_module(stypy.reporting.localization.Localization(__file__, 71, 0), 'matplotlib._pylab_helpers', None, module_type_store, ['Gcf'], [Gcf])

else:
    # Assigning a type to the variable 'matplotlib._pylab_helpers' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'matplotlib._pylab_helpers', import_257334)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 72, 0))

# 'from matplotlib.backend_bases import RendererBase, GraphicsContextBase, FigureManagerBase, FigureCanvasBase' statement (line 72)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_257336 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 72, 0), 'matplotlib.backend_bases')

if (type(import_257336) is not StypyTypeError):

    if (import_257336 != 'pyd_module'):
        __import__(import_257336)
        sys_modules_257337 = sys.modules[import_257336]
        import_from_module(stypy.reporting.localization.Localization(__file__, 72, 0), 'matplotlib.backend_bases', sys_modules_257337.module_type_store, module_type_store, ['RendererBase', 'GraphicsContextBase', 'FigureManagerBase', 'FigureCanvasBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 72, 0), __file__, sys_modules_257337, sys_modules_257337.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import RendererBase, GraphicsContextBase, FigureManagerBase, FigureCanvasBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 72, 0), 'matplotlib.backend_bases', None, module_type_store, ['RendererBase', 'GraphicsContextBase', 'FigureManagerBase', 'FigureCanvasBase'], [RendererBase, GraphicsContextBase, FigureManagerBase, FigureCanvasBase])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'matplotlib.backend_bases', import_257336)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 74, 0))

# 'from matplotlib.figure import Figure' statement (line 74)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_257338 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 74, 0), 'matplotlib.figure')

if (type(import_257338) is not StypyTypeError):

    if (import_257338 != 'pyd_module'):
        __import__(import_257338)
        sys_modules_257339 = sys.modules[import_257338]
        import_from_module(stypy.reporting.localization.Localization(__file__, 74, 0), 'matplotlib.figure', sys_modules_257339.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 74, 0), __file__, sys_modules_257339, sys_modules_257339.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 74, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'matplotlib.figure', import_257338)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 75, 0))

# 'from matplotlib.transforms import Bbox' statement (line 75)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_257340 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 75, 0), 'matplotlib.transforms')

if (type(import_257340) is not StypyTypeError):

    if (import_257340 != 'pyd_module'):
        __import__(import_257340)
        sys_modules_257341 = sys.modules[import_257340]
        import_from_module(stypy.reporting.localization.Localization(__file__, 75, 0), 'matplotlib.transforms', sys_modules_257341.module_type_store, module_type_store, ['Bbox'])
        nest_module(stypy.reporting.localization.Localization(__file__, 75, 0), __file__, sys_modules_257341, sys_modules_257341.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Bbox

        import_from_module(stypy.reporting.localization.Localization(__file__, 75, 0), 'matplotlib.transforms', None, module_type_store, ['Bbox'], [Bbox])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'matplotlib.transforms', import_257340)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# Declaration of the 'RendererTemplate' class
# Getting the type of 'RendererBase' (line 78)
RendererBase_257342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'RendererBase')

class RendererTemplate(RendererBase_257342, ):
    unicode_257343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'unicode', u'\n    The renderer handles drawing/rendering operations.\n\n    This is a minimal do-nothing class that can be used to get started when\n    writing a new backend. Refer to backend_bases.RendererBase for\n    documentation of the classes methods.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererTemplate.__init__', ['dpi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['dpi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 87):
        # Getting the type of 'dpi' (line 87)
        dpi_257344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'dpi')
        # Getting the type of 'self' (line 87)
        self_257345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self')
        # Setting the type of the member 'dpi' of a type (line 87)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_257345, 'dpi', dpi_257344)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def draw_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 89)
        None_257346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 53), 'None')
        defaults = [None_257346]
        # Create a new context for function 'draw_path'
        module_type_store = module_type_store.open_function_context('draw_path', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererTemplate.draw_path.__dict__.__setitem__('stypy_localization', localization)
        RendererTemplate.draw_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererTemplate.draw_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererTemplate.draw_path.__dict__.__setitem__('stypy_function_name', 'RendererTemplate.draw_path')
        RendererTemplate.draw_path.__dict__.__setitem__('stypy_param_names_list', ['gc', 'path', 'transform', 'rgbFace'])
        RendererTemplate.draw_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererTemplate.draw_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererTemplate.draw_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererTemplate.draw_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererTemplate.draw_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererTemplate.draw_path.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererTemplate.draw_path', ['gc', 'path', 'transform', 'rgbFace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_path', localization, ['gc', 'path', 'transform', 'rgbFace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_path(...)' code ##################

        pass
        
        # ################# End of 'draw_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_257347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_257347)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path'
        return stypy_return_type_257347


    @norecursion
    def draw_image(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_image'
        module_type_store = module_type_store.open_function_context('draw_image', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererTemplate.draw_image.__dict__.__setitem__('stypy_localization', localization)
        RendererTemplate.draw_image.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererTemplate.draw_image.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererTemplate.draw_image.__dict__.__setitem__('stypy_function_name', 'RendererTemplate.draw_image')
        RendererTemplate.draw_image.__dict__.__setitem__('stypy_param_names_list', ['gc', 'x', 'y', 'im'])
        RendererTemplate.draw_image.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererTemplate.draw_image.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererTemplate.draw_image.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererTemplate.draw_image.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererTemplate.draw_image.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererTemplate.draw_image.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererTemplate.draw_image', ['gc', 'x', 'y', 'im'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_image', localization, ['gc', 'x', 'y', 'im'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_image(...)' code ##################

        pass
        
        # ################# End of 'draw_image(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_image' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_257348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_257348)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_image'
        return stypy_return_type_257348


    @norecursion
    def draw_text(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 118)
        False_257349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 57), 'False')
        # Getting the type of 'None' (line 118)
        None_257350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 70), 'None')
        defaults = [False_257349, None_257350]
        # Create a new context for function 'draw_text'
        module_type_store = module_type_store.open_function_context('draw_text', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererTemplate.draw_text.__dict__.__setitem__('stypy_localization', localization)
        RendererTemplate.draw_text.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererTemplate.draw_text.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererTemplate.draw_text.__dict__.__setitem__('stypy_function_name', 'RendererTemplate.draw_text')
        RendererTemplate.draw_text.__dict__.__setitem__('stypy_param_names_list', ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath', 'mtext'])
        RendererTemplate.draw_text.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererTemplate.draw_text.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererTemplate.draw_text.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererTemplate.draw_text.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererTemplate.draw_text.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererTemplate.draw_text.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererTemplate.draw_text', ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath', 'mtext'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_text', localization, ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath', 'mtext'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_text(...)' code ##################

        pass
        
        # ################# End of 'draw_text(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_text' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_257351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_257351)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_text'
        return stypy_return_type_257351


    @norecursion
    def flipy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'flipy'
        module_type_store = module_type_store.open_function_context('flipy', 121, 4, False)
        # Assigning a type to the variable 'self' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererTemplate.flipy.__dict__.__setitem__('stypy_localization', localization)
        RendererTemplate.flipy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererTemplate.flipy.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererTemplate.flipy.__dict__.__setitem__('stypy_function_name', 'RendererTemplate.flipy')
        RendererTemplate.flipy.__dict__.__setitem__('stypy_param_names_list', [])
        RendererTemplate.flipy.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererTemplate.flipy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererTemplate.flipy.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererTemplate.flipy.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererTemplate.flipy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererTemplate.flipy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererTemplate.flipy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'flipy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'flipy(...)' code ##################

        # Getting the type of 'True' (line 122)
        True_257352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'stypy_return_type', True_257352)
        
        # ################# End of 'flipy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'flipy' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_257353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_257353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'flipy'
        return stypy_return_type_257353


    @norecursion
    def get_canvas_width_height(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_canvas_width_height'
        module_type_store = module_type_store.open_function_context('get_canvas_width_height', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererTemplate.get_canvas_width_height.__dict__.__setitem__('stypy_localization', localization)
        RendererTemplate.get_canvas_width_height.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererTemplate.get_canvas_width_height.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererTemplate.get_canvas_width_height.__dict__.__setitem__('stypy_function_name', 'RendererTemplate.get_canvas_width_height')
        RendererTemplate.get_canvas_width_height.__dict__.__setitem__('stypy_param_names_list', [])
        RendererTemplate.get_canvas_width_height.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererTemplate.get_canvas_width_height.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererTemplate.get_canvas_width_height.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererTemplate.get_canvas_width_height.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererTemplate.get_canvas_width_height.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererTemplate.get_canvas_width_height.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererTemplate.get_canvas_width_height', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_canvas_width_height', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_canvas_width_height(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 125)
        tuple_257354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 125)
        # Adding element type (line 125)
        int_257355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 15), tuple_257354, int_257355)
        # Adding element type (line 125)
        int_257356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 15), tuple_257354, int_257356)
        
        # Assigning a type to the variable 'stypy_return_type' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'stypy_return_type', tuple_257354)
        
        # ################# End of 'get_canvas_width_height(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_canvas_width_height' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_257357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_257357)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_canvas_width_height'
        return stypy_return_type_257357


    @norecursion
    def get_text_width_height_descent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_text_width_height_descent'
        module_type_store = module_type_store.open_function_context('get_text_width_height_descent', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererTemplate.get_text_width_height_descent.__dict__.__setitem__('stypy_localization', localization)
        RendererTemplate.get_text_width_height_descent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererTemplate.get_text_width_height_descent.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererTemplate.get_text_width_height_descent.__dict__.__setitem__('stypy_function_name', 'RendererTemplate.get_text_width_height_descent')
        RendererTemplate.get_text_width_height_descent.__dict__.__setitem__('stypy_param_names_list', ['s', 'prop', 'ismath'])
        RendererTemplate.get_text_width_height_descent.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererTemplate.get_text_width_height_descent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererTemplate.get_text_width_height_descent.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererTemplate.get_text_width_height_descent.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererTemplate.get_text_width_height_descent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererTemplate.get_text_width_height_descent.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererTemplate.get_text_width_height_descent', ['s', 'prop', 'ismath'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_text_width_height_descent', localization, ['s', 'prop', 'ismath'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_text_width_height_descent(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 128)
        tuple_257358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 128)
        # Adding element type (line 128)
        int_257359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 15), tuple_257358, int_257359)
        # Adding element type (line 128)
        int_257360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 15), tuple_257358, int_257360)
        # Adding element type (line 128)
        int_257361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 15), tuple_257358, int_257361)
        
        # Assigning a type to the variable 'stypy_return_type' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'stypy_return_type', tuple_257358)
        
        # ################# End of 'get_text_width_height_descent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_text_width_height_descent' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_257362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_257362)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_text_width_height_descent'
        return stypy_return_type_257362


    @norecursion
    def new_gc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_gc'
        module_type_store = module_type_store.open_function_context('new_gc', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererTemplate.new_gc.__dict__.__setitem__('stypy_localization', localization)
        RendererTemplate.new_gc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererTemplate.new_gc.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererTemplate.new_gc.__dict__.__setitem__('stypy_function_name', 'RendererTemplate.new_gc')
        RendererTemplate.new_gc.__dict__.__setitem__('stypy_param_names_list', [])
        RendererTemplate.new_gc.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererTemplate.new_gc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererTemplate.new_gc.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererTemplate.new_gc.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererTemplate.new_gc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererTemplate.new_gc.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererTemplate.new_gc', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_gc', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_gc(...)' code ##################

        
        # Call to GraphicsContextTemplate(...): (line 131)
        # Processing the call keyword arguments (line 131)
        kwargs_257364 = {}
        # Getting the type of 'GraphicsContextTemplate' (line 131)
        GraphicsContextTemplate_257363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'GraphicsContextTemplate', False)
        # Calling GraphicsContextTemplate(args, kwargs) (line 131)
        GraphicsContextTemplate_call_result_257365 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), GraphicsContextTemplate_257363, *[], **kwargs_257364)
        
        # Assigning a type to the variable 'stypy_return_type' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', GraphicsContextTemplate_call_result_257365)
        
        # ################# End of 'new_gc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_gc' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_257366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_257366)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_gc'
        return stypy_return_type_257366


    @norecursion
    def points_to_pixels(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'points_to_pixels'
        module_type_store = module_type_store.open_function_context('points_to_pixels', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererTemplate.points_to_pixels.__dict__.__setitem__('stypy_localization', localization)
        RendererTemplate.points_to_pixels.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererTemplate.points_to_pixels.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererTemplate.points_to_pixels.__dict__.__setitem__('stypy_function_name', 'RendererTemplate.points_to_pixels')
        RendererTemplate.points_to_pixels.__dict__.__setitem__('stypy_param_names_list', ['points'])
        RendererTemplate.points_to_pixels.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererTemplate.points_to_pixels.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererTemplate.points_to_pixels.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererTemplate.points_to_pixels.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererTemplate.points_to_pixels.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererTemplate.points_to_pixels.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererTemplate.points_to_pixels', ['points'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'points_to_pixels', localization, ['points'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'points_to_pixels(...)' code ##################

        # Getting the type of 'points' (line 135)
        points_257367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'points')
        # Assigning a type to the variable 'stypy_return_type' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'stypy_return_type', points_257367)
        
        # ################# End of 'points_to_pixels(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'points_to_pixels' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_257368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_257368)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'points_to_pixels'
        return stypy_return_type_257368


# Assigning a type to the variable 'RendererTemplate' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'RendererTemplate', RendererTemplate)
# Declaration of the 'GraphicsContextTemplate' class
# Getting the type of 'GraphicsContextBase' (line 142)
GraphicsContextBase_257369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), 'GraphicsContextBase')

class GraphicsContextTemplate(GraphicsContextBase_257369, ):
    unicode_257370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, (-1)), 'unicode', u"\n    The graphics context provides the color, line styles, etc...  See the gtk\n    and postscript backends for examples of mapping the graphics context\n    attributes (cap styles, join styles, line widths, colors) to a particular\n    backend.  In GTK this is done by wrapping a gtk.gdk.GC object and\n    forwarding the appropriate calls to it using a dictionary mapping styles\n    to gdk constants.  In Postscript, all the work is done by the renderer,\n    mapping line styles to postscript calls.\n\n    If it's more appropriate to do the mapping at the renderer level (as in\n    the postscript backend), you don't need to override any of the GC methods.\n    If it's more appropriate to wrap an instance (as in the GTK backend) and\n    do the mapping here, you'll need to override several of the setter\n    methods.\n\n    The base GraphicsContext stores colors as a RGB tuple on the unit\n    interval, e.g., (0.5, 0.0, 1.0). You may need to map this to colors\n    appropriate for your backend.\n    ")
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 142, 0, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextTemplate.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'GraphicsContextTemplate' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'GraphicsContextTemplate', GraphicsContextTemplate)

@norecursion
def draw_if_interactive(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'draw_if_interactive'
    module_type_store = module_type_store.open_function_context('draw_if_interactive', 173, 0, False)
    
    # Passed parameters checking function
    draw_if_interactive.stypy_localization = localization
    draw_if_interactive.stypy_type_of_self = None
    draw_if_interactive.stypy_type_store = module_type_store
    draw_if_interactive.stypy_function_name = 'draw_if_interactive'
    draw_if_interactive.stypy_param_names_list = []
    draw_if_interactive.stypy_varargs_param_name = None
    draw_if_interactive.stypy_kwargs_param_name = None
    draw_if_interactive.stypy_call_defaults = defaults
    draw_if_interactive.stypy_call_varargs = varargs
    draw_if_interactive.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'draw_if_interactive', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'draw_if_interactive', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'draw_if_interactive(...)' code ##################

    unicode_257371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, (-1)), 'unicode', u'\n    For image backends - is not required\n    For GUI backends - this should be overridden if drawing should be done in\n    interactive python mode\n    ')
    
    # ################# End of 'draw_if_interactive(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'draw_if_interactive' in the type store
    # Getting the type of 'stypy_return_type' (line 173)
    stypy_return_type_257372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_257372)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'draw_if_interactive'
    return stypy_return_type_257372

# Assigning a type to the variable 'draw_if_interactive' (line 173)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'draw_if_interactive', draw_if_interactive)

@norecursion
def show(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'show'
    module_type_store = module_type_store.open_function_context('show', 182, 0, False)
    
    # Passed parameters checking function
    show.stypy_localization = localization
    show.stypy_type_of_self = None
    show.stypy_type_store = module_type_store
    show.stypy_function_name = 'show'
    show.stypy_param_names_list = []
    show.stypy_varargs_param_name = None
    show.stypy_kwargs_param_name = None
    show.stypy_call_defaults = defaults
    show.stypy_call_varargs = varargs
    show.stypy_call_kwargs = kwargs
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

    unicode_257373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, (-1)), 'unicode', u'\n    For image backends - is not required\n    For GUI backends - show() is usually the last line of a pylab script and\n    tells the backend that it is time to draw.  In interactive mode, this may\n    be a do nothing func.  See the GTK backend for an example of how to handle\n    interactive versus batch mode\n    ')
    
    
    # Call to get_all_fig_managers(...): (line 190)
    # Processing the call keyword arguments (line 190)
    kwargs_257376 = {}
    # Getting the type of 'Gcf' (line 190)
    Gcf_257374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'Gcf', False)
    # Obtaining the member 'get_all_fig_managers' of a type (line 190)
    get_all_fig_managers_257375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 19), Gcf_257374, 'get_all_fig_managers')
    # Calling get_all_fig_managers(args, kwargs) (line 190)
    get_all_fig_managers_call_result_257377 = invoke(stypy.reporting.localization.Localization(__file__, 190, 19), get_all_fig_managers_257375, *[], **kwargs_257376)
    
    # Testing the type of a for loop iterable (line 190)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 190, 4), get_all_fig_managers_call_result_257377)
    # Getting the type of the for loop variable (line 190)
    for_loop_var_257378 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 190, 4), get_all_fig_managers_call_result_257377)
    # Assigning a type to the variable 'manager' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'manager', for_loop_var_257378)
    # SSA begins for a for statement (line 190)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    pass
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'show(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'show' in the type store
    # Getting the type of 'stypy_return_type' (line 182)
    stypy_return_type_257379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_257379)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'show'
    return stypy_return_type_257379

# Assigning a type to the variable 'show' (line 182)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 0), 'show', show)

@norecursion
def new_figure_manager(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'new_figure_manager'
    module_type_store = module_type_store.open_function_context('new_figure_manager', 195, 0, False)
    
    # Passed parameters checking function
    new_figure_manager.stypy_localization = localization
    new_figure_manager.stypy_type_of_self = None
    new_figure_manager.stypy_type_store = module_type_store
    new_figure_manager.stypy_function_name = 'new_figure_manager'
    new_figure_manager.stypy_param_names_list = ['num']
    new_figure_manager.stypy_varargs_param_name = 'args'
    new_figure_manager.stypy_kwargs_param_name = 'kwargs'
    new_figure_manager.stypy_call_defaults = defaults
    new_figure_manager.stypy_call_varargs = varargs
    new_figure_manager.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'new_figure_manager', ['num'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'new_figure_manager', localization, ['num'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'new_figure_manager(...)' code ##################

    unicode_257380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, (-1)), 'unicode', u'\n    Create a new figure manager instance\n    ')
    
    # Assigning a Call to a Name (line 205):
    
    # Call to pop(...): (line 205)
    # Processing the call arguments (line 205)
    unicode_257383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 29), 'unicode', u'FigureClass')
    # Getting the type of 'Figure' (line 205)
    Figure_257384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 44), 'Figure', False)
    # Processing the call keyword arguments (line 205)
    kwargs_257385 = {}
    # Getting the type of 'kwargs' (line 205)
    kwargs_257381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 18), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 205)
    pop_257382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 18), kwargs_257381, 'pop')
    # Calling pop(args, kwargs) (line 205)
    pop_call_result_257386 = invoke(stypy.reporting.localization.Localization(__file__, 205, 18), pop_257382, *[unicode_257383, Figure_257384], **kwargs_257385)
    
    # Assigning a type to the variable 'FigureClass' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'FigureClass', pop_call_result_257386)
    
    # Assigning a Call to a Name (line 206):
    
    # Call to FigureClass(...): (line 206)
    # Getting the type of 'args' (line 206)
    args_257388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 27), 'args', False)
    # Processing the call keyword arguments (line 206)
    # Getting the type of 'kwargs' (line 206)
    kwargs_257389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 35), 'kwargs', False)
    kwargs_257390 = {'kwargs_257389': kwargs_257389}
    # Getting the type of 'FigureClass' (line 206)
    FigureClass_257387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 14), 'FigureClass', False)
    # Calling FigureClass(args, kwargs) (line 206)
    FigureClass_call_result_257391 = invoke(stypy.reporting.localization.Localization(__file__, 206, 14), FigureClass_257387, *[args_257388], **kwargs_257390)
    
    # Assigning a type to the variable 'thisFig' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'thisFig', FigureClass_call_result_257391)
    
    # Call to new_figure_manager_given_figure(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'num' (line 207)
    num_257393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 43), 'num', False)
    # Getting the type of 'thisFig' (line 207)
    thisFig_257394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 48), 'thisFig', False)
    # Processing the call keyword arguments (line 207)
    kwargs_257395 = {}
    # Getting the type of 'new_figure_manager_given_figure' (line 207)
    new_figure_manager_given_figure_257392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'new_figure_manager_given_figure', False)
    # Calling new_figure_manager_given_figure(args, kwargs) (line 207)
    new_figure_manager_given_figure_call_result_257396 = invoke(stypy.reporting.localization.Localization(__file__, 207, 11), new_figure_manager_given_figure_257392, *[num_257393, thisFig_257394], **kwargs_257395)
    
    # Assigning a type to the variable 'stypy_return_type' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type', new_figure_manager_given_figure_call_result_257396)
    
    # ################# End of 'new_figure_manager(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'new_figure_manager' in the type store
    # Getting the type of 'stypy_return_type' (line 195)
    stypy_return_type_257397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_257397)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'new_figure_manager'
    return stypy_return_type_257397

# Assigning a type to the variable 'new_figure_manager' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'new_figure_manager', new_figure_manager)

@norecursion
def new_figure_manager_given_figure(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'new_figure_manager_given_figure'
    module_type_store = module_type_store.open_function_context('new_figure_manager_given_figure', 210, 0, False)
    
    # Passed parameters checking function
    new_figure_manager_given_figure.stypy_localization = localization
    new_figure_manager_given_figure.stypy_type_of_self = None
    new_figure_manager_given_figure.stypy_type_store = module_type_store
    new_figure_manager_given_figure.stypy_function_name = 'new_figure_manager_given_figure'
    new_figure_manager_given_figure.stypy_param_names_list = ['num', 'figure']
    new_figure_manager_given_figure.stypy_varargs_param_name = None
    new_figure_manager_given_figure.stypy_kwargs_param_name = None
    new_figure_manager_given_figure.stypy_call_defaults = defaults
    new_figure_manager_given_figure.stypy_call_varargs = varargs
    new_figure_manager_given_figure.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'new_figure_manager_given_figure', ['num', 'figure'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'new_figure_manager_given_figure', localization, ['num', 'figure'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'new_figure_manager_given_figure(...)' code ##################

    unicode_257398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, (-1)), 'unicode', u'\n    Create a new figure manager instance for the given figure.\n    ')
    
    # Assigning a Call to a Name (line 215):
    
    # Call to FigureCanvasTemplate(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'figure' (line 215)
    figure_257400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 34), 'figure', False)
    # Processing the call keyword arguments (line 215)
    kwargs_257401 = {}
    # Getting the type of 'FigureCanvasTemplate' (line 215)
    FigureCanvasTemplate_257399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 13), 'FigureCanvasTemplate', False)
    # Calling FigureCanvasTemplate(args, kwargs) (line 215)
    FigureCanvasTemplate_call_result_257402 = invoke(stypy.reporting.localization.Localization(__file__, 215, 13), FigureCanvasTemplate_257399, *[figure_257400], **kwargs_257401)
    
    # Assigning a type to the variable 'canvas' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'canvas', FigureCanvasTemplate_call_result_257402)
    
    # Assigning a Call to a Name (line 216):
    
    # Call to FigureManagerTemplate(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'canvas' (line 216)
    canvas_257404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 36), 'canvas', False)
    # Getting the type of 'num' (line 216)
    num_257405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 44), 'num', False)
    # Processing the call keyword arguments (line 216)
    kwargs_257406 = {}
    # Getting the type of 'FigureManagerTemplate' (line 216)
    FigureManagerTemplate_257403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 14), 'FigureManagerTemplate', False)
    # Calling FigureManagerTemplate(args, kwargs) (line 216)
    FigureManagerTemplate_call_result_257407 = invoke(stypy.reporting.localization.Localization(__file__, 216, 14), FigureManagerTemplate_257403, *[canvas_257404, num_257405], **kwargs_257406)
    
    # Assigning a type to the variable 'manager' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'manager', FigureManagerTemplate_call_result_257407)
    # Getting the type of 'manager' (line 217)
    manager_257408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'manager')
    # Assigning a type to the variable 'stypy_return_type' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type', manager_257408)
    
    # ################# End of 'new_figure_manager_given_figure(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'new_figure_manager_given_figure' in the type store
    # Getting the type of 'stypy_return_type' (line 210)
    stypy_return_type_257409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_257409)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'new_figure_manager_given_figure'
    return stypy_return_type_257409

# Assigning a type to the variable 'new_figure_manager_given_figure' (line 210)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 0), 'new_figure_manager_given_figure', new_figure_manager_given_figure)
# Declaration of the 'FigureCanvasTemplate' class
# Getting the type of 'FigureCanvasBase' (line 220)
FigureCanvasBase_257410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 27), 'FigureCanvasBase')

class FigureCanvasTemplate(FigureCanvasBase_257410, ):
    unicode_257411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, (-1)), 'unicode', u'\n    The canvas the figure renders into.  Calls the draw and print fig\n    methods, creates the renderers, etc...\n\n    Note GUI templates will want to connect events for button presses,\n    mouse movements and key presses to functions that call the base\n    class methods button_press_event, button_release_event,\n    motion_notify_event, key_press_event, and key_release_event.  See,\n    e.g., backend_gtk.py, backend_wx.py and backend_tkagg.py\n\n    Attributes\n    ----------\n    figure : `matplotlib.figure.Figure`\n        A high-level Figure instance\n\n    ')

    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 238, 4, False)
        # Assigning a type to the variable 'self' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasTemplate.draw.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasTemplate.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasTemplate.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasTemplate.draw.__dict__.__setitem__('stypy_function_name', 'FigureCanvasTemplate.draw')
        FigureCanvasTemplate.draw.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasTemplate.draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasTemplate.draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasTemplate.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasTemplate.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasTemplate.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasTemplate.draw.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasTemplate.draw', [], None, None, defaults, varargs, kwargs)

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

        unicode_257412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, (-1)), 'unicode', u'\n        Draw the figure using the renderer\n        ')
        
        # Assigning a Call to a Name (line 242):
        
        # Call to RendererTemplate(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'self' (line 242)
        self_257414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 36), 'self', False)
        # Obtaining the member 'figure' of a type (line 242)
        figure_257415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 36), self_257414, 'figure')
        # Obtaining the member 'dpi' of a type (line 242)
        dpi_257416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 36), figure_257415, 'dpi')
        # Processing the call keyword arguments (line 242)
        kwargs_257417 = {}
        # Getting the type of 'RendererTemplate' (line 242)
        RendererTemplate_257413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'RendererTemplate', False)
        # Calling RendererTemplate(args, kwargs) (line 242)
        RendererTemplate_call_result_257418 = invoke(stypy.reporting.localization.Localization(__file__, 242, 19), RendererTemplate_257413, *[dpi_257416], **kwargs_257417)
        
        # Assigning a type to the variable 'renderer' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'renderer', RendererTemplate_call_result_257418)
        
        # Call to draw(...): (line 243)
        # Processing the call arguments (line 243)
        # Getting the type of 'renderer' (line 243)
        renderer_257422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 25), 'renderer', False)
        # Processing the call keyword arguments (line 243)
        kwargs_257423 = {}
        # Getting the type of 'self' (line 243)
        self_257419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 243)
        figure_257420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), self_257419, 'figure')
        # Obtaining the member 'draw' of a type (line 243)
        draw_257421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), figure_257420, 'draw')
        # Calling draw(args, kwargs) (line 243)
        draw_call_result_257424 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), draw_257421, *[renderer_257422], **kwargs_257423)
        
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 238)
        stypy_return_type_257425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_257425)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_257425


    @norecursion
    def print_foo(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_foo'
        module_type_store = module_type_store.open_function_context('print_foo', 253, 4, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasTemplate.print_foo.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasTemplate.print_foo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasTemplate.print_foo.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasTemplate.print_foo.__dict__.__setitem__('stypy_function_name', 'FigureCanvasTemplate.print_foo')
        FigureCanvasTemplate.print_foo.__dict__.__setitem__('stypy_param_names_list', ['filename'])
        FigureCanvasTemplate.print_foo.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasTemplate.print_foo.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasTemplate.print_foo.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasTemplate.print_foo.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasTemplate.print_foo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasTemplate.print_foo.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasTemplate.print_foo', ['filename'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_foo', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_foo(...)' code ##################

        unicode_257426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, (-1)), 'unicode', u"\n        Write out format foo.  The dpi, facecolor and edgecolor are restored\n        to their original values after this call, so you don't need to\n        save and restore them.\n        ")
        pass
        
        # ################# End of 'print_foo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_foo' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_257427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_257427)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_foo'
        return stypy_return_type_257427


    @norecursion
    def get_default_filetype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_default_filetype'
        module_type_store = module_type_store.open_function_context('get_default_filetype', 261, 4, False)
        # Assigning a type to the variable 'self' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasTemplate.get_default_filetype.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasTemplate.get_default_filetype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasTemplate.get_default_filetype.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasTemplate.get_default_filetype.__dict__.__setitem__('stypy_function_name', 'FigureCanvasTemplate.get_default_filetype')
        FigureCanvasTemplate.get_default_filetype.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasTemplate.get_default_filetype.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasTemplate.get_default_filetype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasTemplate.get_default_filetype.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasTemplate.get_default_filetype.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasTemplate.get_default_filetype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasTemplate.get_default_filetype.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasTemplate.get_default_filetype', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_default_filetype', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_default_filetype(...)' code ##################

        unicode_257428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 15), 'unicode', u'foo')
        # Assigning a type to the variable 'stypy_return_type' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'stypy_return_type', unicode_257428)
        
        # ################# End of 'get_default_filetype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_default_filetype' in the type store
        # Getting the type of 'stypy_return_type' (line 261)
        stypy_return_type_257429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_257429)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_default_filetype'
        return stypy_return_type_257429


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 220, 0, False)
        # Assigning a type to the variable 'self' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasTemplate.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureCanvasTemplate' (line 220)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'FigureCanvasTemplate', FigureCanvasTemplate)

# Assigning a Call to a Name (line 250):

# Call to copy(...): (line 250)
# Processing the call keyword arguments (line 250)
kwargs_257433 = {}
# Getting the type of 'FigureCanvasBase' (line 250)
FigureCanvasBase_257430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'FigureCanvasBase', False)
# Obtaining the member 'filetypes' of a type (line 250)
filetypes_257431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), FigureCanvasBase_257430, 'filetypes')
# Obtaining the member 'copy' of a type (line 250)
copy_257432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), filetypes_257431, 'copy')
# Calling copy(args, kwargs) (line 250)
copy_call_result_257434 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), copy_257432, *[], **kwargs_257433)

# Getting the type of 'FigureCanvasTemplate'
FigureCanvasTemplate_257435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasTemplate')
# Setting the type of the member 'filetypes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasTemplate_257435, 'filetypes', copy_call_result_257434)

# Assigning a Str to a Subscript (line 251):
unicode_257436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 23), 'unicode', u'My magic Foo format')
# Getting the type of 'FigureCanvasTemplate'
FigureCanvasTemplate_257437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasTemplate')
# Setting the type of the member 'filetypes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasTemplate_257437, 'filetypes', unicode_257436)
# Declaration of the 'FigureManagerTemplate' class
# Getting the type of 'FigureManagerBase' (line 265)
FigureManagerBase_257438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 28), 'FigureManagerBase')

class FigureManagerTemplate(FigureManagerBase_257438, ):
    unicode_257439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, (-1)), 'unicode', u'\n    Wrap everything up into a window for the pylab interface\n\n    For non interactive backends, the base class does all the work\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 265, 0, False)
        # Assigning a type to the variable 'self' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerTemplate.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureManagerTemplate' (line 265)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 0), 'FigureManagerTemplate', FigureManagerTemplate)

# Assigning a Name to a Name (line 279):
# Getting the type of 'FigureCanvasTemplate' (line 279)
FigureCanvasTemplate_257440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 15), 'FigureCanvasTemplate')
# Assigning a type to the variable 'FigureCanvas' (line 279)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 0), 'FigureCanvas', FigureCanvasTemplate_257440)

# Assigning a Name to a Name (line 280):
# Getting the type of 'FigureManagerTemplate' (line 280)
FigureManagerTemplate_257441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'FigureManagerTemplate')
# Assigning a type to the variable 'FigureManager' (line 280)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 0), 'FigureManager', FigureManagerTemplate_257441)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
