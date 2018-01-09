
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: from matplotlib.contour import ContourSet
7: from matplotlib.tri.triangulation import Triangulation
8: import matplotlib._tri as _tri
9: import numpy as np
10: 
11: 
12: class TriContourSet(ContourSet):
13:     '''
14:     Create and store a set of contour lines or filled regions for
15:     a triangular grid.
16: 
17:     User-callable method: clabel
18: 
19:     Useful attributes:
20:       ax:
21:         the axes object in which the contours are drawn
22:       collections:
23:         a silent_list of LineCollections or PolyCollections
24:       levels:
25:         contour levels
26:       layers:
27:         same as levels for line contours; half-way between
28:         levels for filled contours.  See _process_colors method.
29:     '''
30:     def __init__(self, ax, *args, **kwargs):
31:         '''
32:         Draw triangular grid contour lines or filled regions,
33:         depending on whether keyword arg 'filled' is False
34:         (default) or True.
35: 
36:         The first argument of the initializer must be an axes
37:         object.  The remaining arguments and keyword arguments
38:         are described in TriContourSet.tricontour_doc.
39:         '''
40:         ContourSet.__init__(self, ax, *args, **kwargs)
41: 
42:     def _process_args(self, *args, **kwargs):
43:         '''
44:         Process args and kwargs.
45:         '''
46:         if isinstance(args[0], TriContourSet):
47:             C = args[0].cppContourGenerator
48:             if self.levels is None:
49:                 self.levels = args[0].levels
50:         else:
51:             tri, z = self._contour_args(args, kwargs)
52:             C = _tri.TriContourGenerator(tri.get_cpp_triangulation(), z)
53:             self._mins = [tri.x.min(), tri.y.min()]
54:             self._maxs = [tri.x.max(), tri.y.max()]
55: 
56:         self.cppContourGenerator = C
57:         return kwargs
58: 
59:     def _get_allsegs_and_allkinds(self):
60:         '''
61:         Create and return allsegs and allkinds by calling underlying C code.
62:         '''
63:         allsegs = []
64:         if self.filled:
65:             lowers, uppers = self._get_lowers_and_uppers()
66:             allkinds = []
67:             for lower, upper in zip(lowers, uppers):
68:                 segs, kinds = self.cppContourGenerator.create_filled_contour(
69:                     lower, upper)
70:                 allsegs.append([segs])
71:                 allkinds.append([kinds])
72:         else:
73:             allkinds = None
74:             for level in self.levels:
75:                 segs = self.cppContourGenerator.create_contour(level)
76:                 allsegs.append(segs)
77:         return allsegs, allkinds
78: 
79:     def _contour_args(self, args, kwargs):
80:         if self.filled:
81:             fn = 'contourf'
82:         else:
83:             fn = 'contour'
84:         tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args,
85:                                                                    **kwargs)
86:         z = np.asarray(args[0])
87:         if z.shape != tri.x.shape:
88:             raise ValueError('z array must have same length as triangulation x'
89:                              ' and y arrays')
90:         self.zmax = z.max()
91:         self.zmin = z.min()
92:         if self.logscale and self.zmin <= 0:
93:             raise ValueError('Cannot %s log of negative values.' % fn)
94:         self._contour_level_args(z, args[1:])
95:         return (tri, z)
96: 
97:     tricontour_doc = '''
98:         Draw contours on an unstructured triangular grid.
99:         :func:`~matplotlib.pyplot.tricontour` and
100:         :func:`~matplotlib.pyplot.tricontourf` draw contour lines and
101:         filled contours, respectively.  Except as noted, function
102:         signatures and return values are the same for both versions.
103: 
104:         The triangulation can be specified in one of two ways; either::
105: 
106:           tricontour(triangulation, ...)
107: 
108:         where triangulation is a :class:`matplotlib.tri.Triangulation`
109:         object, or
110: 
111:         ::
112: 
113:           tricontour(x, y, ...)
114:           tricontour(x, y, triangles, ...)
115:           tricontour(x, y, triangles=triangles, ...)
116:           tricontour(x, y, mask=mask, ...)
117:           tricontour(x, y, triangles, mask=mask, ...)
118: 
119:         in which case a Triangulation object will be created.  See
120:         :class:`~matplotlib.tri.Triangulation` for a explanation of
121:         these possibilities.
122: 
123:         The remaining arguments may be::
124: 
125:           tricontour(..., Z)
126: 
127:         where *Z* is the array of values to contour, one per point
128:         in the triangulation.  The level values are chosen
129:         automatically.
130: 
131:         ::
132: 
133:           tricontour(..., Z, N)
134: 
135:         contour *N* automatically-chosen levels.
136: 
137:         ::
138: 
139:           tricontour(..., Z, V)
140: 
141:         draw contour lines at the values specified in sequence *V*,
142:         which must be in increasing order.
143: 
144:         ::
145: 
146:           tricontourf(..., Z, V)
147: 
148:         fill the (len(*V*)-1) regions between the values in *V*,
149:         which must be in increasing order.
150: 
151:         ::
152: 
153:           tricontour(Z, **kwargs)
154: 
155:         Use keyword args to control colors, linewidth, origin, cmap ... see
156:         below for more details.
157: 
158:         ``C = tricontour(...)`` returns a
159:         :class:`~matplotlib.contour.TriContourSet` object.
160: 
161:         Optional keyword arguments:
162: 
163:           *colors*: [ *None* | string | (mpl_colors) ]
164:             If *None*, the colormap specified by cmap will be used.
165: 
166:             If a string, like 'r' or 'red', all levels will be plotted in this
167:             color.
168: 
169:             If a tuple of matplotlib color args (string, float, rgb, etc),
170:             different levels will be plotted in different colors in the order
171:             specified.
172: 
173:           *alpha*: float
174:             The alpha blending value
175: 
176:           *cmap*: [ *None* | Colormap ]
177:             A cm :class:`~matplotlib.colors.Colormap` instance or
178:             *None*. If *cmap* is *None* and *colors* is *None*, a
179:             default Colormap is used.
180: 
181:           *norm*: [ *None* | Normalize ]
182:             A :class:`matplotlib.colors.Normalize` instance for
183:             scaling data values to colors. If *norm* is *None* and
184:             *colors* is *None*, the default linear scaling is used.
185: 
186:           *levels* [level0, level1, ..., leveln]
187:             A list of floating point numbers indicating the level
188:             curves to draw, in increasing order; e.g., to draw just
189:             the zero contour pass ``levels=[0]``
190: 
191:           *origin*: [ *None* | 'upper' | 'lower' | 'image' ]
192:             If *None*, the first value of *Z* will correspond to the
193:             lower left corner, location (0,0). If 'image', the rc
194:             value for ``image.origin`` will be used.
195: 
196:             This keyword is not active if *X* and *Y* are specified in
197:             the call to contour.
198: 
199:           *extent*: [ *None* | (x0,x1,y0,y1) ]
200: 
201:             If *origin* is not *None*, then *extent* is interpreted as
202:             in :func:`matplotlib.pyplot.imshow`: it gives the outer
203:             pixel boundaries. In this case, the position of Z[0,0]
204:             is the center of the pixel, not a corner. If *origin* is
205:             *None*, then (*x0*, *y0*) is the position of Z[0,0], and
206:             (*x1*, *y1*) is the position of Z[-1,-1].
207: 
208:             This keyword is not active if *X* and *Y* are specified in
209:             the call to contour.
210: 
211:           *locator*: [ *None* | ticker.Locator subclass ]
212:             If *locator* is None, the default
213:             :class:`~matplotlib.ticker.MaxNLocator` is used. The
214:             locator is used to determine the contour levels if they
215:             are not given explicitly via the *V* argument.
216: 
217:           *extend*: [ 'neither' | 'both' | 'min' | 'max' ]
218:             Unless this is 'neither', contour levels are automatically
219:             added to one or both ends of the range so that all data
220:             are included. These added ranges are then mapped to the
221:             special colormap values which default to the ends of the
222:             colormap range, but can be set via
223:             :meth:`matplotlib.colors.Colormap.set_under` and
224:             :meth:`matplotlib.colors.Colormap.set_over` methods.
225: 
226:           *xunits*, *yunits*: [ *None* | registered units ]
227:             Override axis units by specifying an instance of a
228:             :class:`matplotlib.units.ConversionInterface`.
229: 
230: 
231:         tricontour-only keyword arguments:
232: 
233:           *linewidths*: [ *None* | number | tuple of numbers ]
234:             If *linewidths* is *None*, the default width in
235:             ``lines.linewidth`` in ``matplotlibrc`` is used.
236: 
237:             If a number, all levels will be plotted with this linewidth.
238: 
239:             If a tuple, different levels will be plotted with different
240:             linewidths in the order specified
241: 
242:           *linestyles*: [ *None* | 'solid' | 'dashed' | 'dashdot' | 'dotted' ]
243:             If *linestyles* is *None*, the 'solid' is used.
244: 
245:             *linestyles* can also be an iterable of the above strings
246:             specifying a set of linestyles to be used. If this
247:             iterable is shorter than the number of contour levels
248:             it will be repeated as necessary.
249: 
250:             If contour is using a monochrome colormap and the contour
251:             level is less than 0, then the linestyle specified
252:             in ``contour.negative_linestyle`` in ``matplotlibrc``
253:             will be used.
254: 
255:         tricontourf-only keyword arguments:
256: 
257:           *antialiased*: [ *True* | *False* ]
258:             enable antialiasing
259: 
260:         Note: tricontourf fills intervals that are closed at the top; that
261:         is, for boundaries *z1* and *z2*, the filled region is::
262: 
263:             z1 < z <= z2
264: 
265:         There is one exception: if the lowest boundary coincides with
266:         the minimum value of the *z* array, then that minimum value
267:         will be included in the lowest interval.
268:         '''
269: 
270: 
271: def tricontour(ax, *args, **kwargs):
272:     if not ax._hold:
273:         ax.cla()
274:     kwargs['filled'] = False
275:     return TriContourSet(ax, *args, **kwargs)
276: tricontour.__doc__ = TriContourSet.tricontour_doc
277: 
278: 
279: def tricontourf(ax, *args, **kwargs):
280:     if not ax._hold:
281:         ax.cla()
282:     kwargs['filled'] = True
283:     return TriContourSet(ax, *args, **kwargs)
284: tricontourf.__doc__ = TriContourSet.tricontour_doc
285: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_294825 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_294825) is not StypyTypeError):

    if (import_294825 != 'pyd_module'):
        __import__(import_294825)
        sys_modules_294826 = sys.modules[import_294825]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_294826.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_294825)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from matplotlib.contour import ContourSet' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_294827 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.contour')

if (type(import_294827) is not StypyTypeError):

    if (import_294827 != 'pyd_module'):
        __import__(import_294827)
        sys_modules_294828 = sys.modules[import_294827]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.contour', sys_modules_294828.module_type_store, module_type_store, ['ContourSet'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_294828, sys_modules_294828.module_type_store, module_type_store)
    else:
        from matplotlib.contour import ContourSet

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.contour', None, module_type_store, ['ContourSet'], [ContourSet])

else:
    # Assigning a type to the variable 'matplotlib.contour' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.contour', import_294827)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from matplotlib.tri.triangulation import Triangulation' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_294829 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.tri.triangulation')

if (type(import_294829) is not StypyTypeError):

    if (import_294829 != 'pyd_module'):
        __import__(import_294829)
        sys_modules_294830 = sys.modules[import_294829]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.tri.triangulation', sys_modules_294830.module_type_store, module_type_store, ['Triangulation'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_294830, sys_modules_294830.module_type_store, module_type_store)
    else:
        from matplotlib.tri.triangulation import Triangulation

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.tri.triangulation', None, module_type_store, ['Triangulation'], [Triangulation])

else:
    # Assigning a type to the variable 'matplotlib.tri.triangulation' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.tri.triangulation', import_294829)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import matplotlib._tri' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_294831 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib._tri')

if (type(import_294831) is not StypyTypeError):

    if (import_294831 != 'pyd_module'):
        __import__(import_294831)
        sys_modules_294832 = sys.modules[import_294831]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), '_tri', sys_modules_294832.module_type_store, module_type_store)
    else:
        import matplotlib._tri as _tri

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), '_tri', matplotlib._tri, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib._tri' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib._tri', import_294831)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_294833 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_294833) is not StypyTypeError):

    if (import_294833 != 'pyd_module'):
        __import__(import_294833)
        sys_modules_294834 = sys.modules[import_294833]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_294834.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_294833)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

# Declaration of the 'TriContourSet' class
# Getting the type of 'ContourSet' (line 12)
ContourSet_294835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'ContourSet')

class TriContourSet(ContourSet_294835, ):
    unicode_294836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'unicode', u'\n    Create and store a set of contour lines or filled regions for\n    a triangular grid.\n\n    User-callable method: clabel\n\n    Useful attributes:\n      ax:\n        the axes object in which the contours are drawn\n      collections:\n        a silent_list of LineCollections or PolyCollections\n      levels:\n        contour levels\n      layers:\n        same as levels for line contours; half-way between\n        levels for filled contours.  See _process_colors method.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TriContourSet.__init__', ['ax'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['ax'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_294837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'unicode', u"\n        Draw triangular grid contour lines or filled regions,\n        depending on whether keyword arg 'filled' is False\n        (default) or True.\n\n        The first argument of the initializer must be an axes\n        object.  The remaining arguments and keyword arguments\n        are described in TriContourSet.tricontour_doc.\n        ")
        
        # Call to __init__(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_294840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'self', False)
        # Getting the type of 'ax' (line 40)
        ax_294841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 34), 'ax', False)
        # Getting the type of 'args' (line 40)
        args_294842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 39), 'args', False)
        # Processing the call keyword arguments (line 40)
        # Getting the type of 'kwargs' (line 40)
        kwargs_294843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 47), 'kwargs', False)
        kwargs_294844 = {'kwargs_294843': kwargs_294843}
        # Getting the type of 'ContourSet' (line 40)
        ContourSet_294838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'ContourSet', False)
        # Obtaining the member '__init__' of a type (line 40)
        init___294839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), ContourSet_294838, '__init__')
        # Calling __init__(args, kwargs) (line 40)
        init___call_result_294845 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), init___294839, *[self_294840, ax_294841, args_294842], **kwargs_294844)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _process_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_process_args'
        module_type_store = module_type_store.open_function_context('_process_args', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TriContourSet._process_args.__dict__.__setitem__('stypy_localization', localization)
        TriContourSet._process_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TriContourSet._process_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        TriContourSet._process_args.__dict__.__setitem__('stypy_function_name', 'TriContourSet._process_args')
        TriContourSet._process_args.__dict__.__setitem__('stypy_param_names_list', [])
        TriContourSet._process_args.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        TriContourSet._process_args.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        TriContourSet._process_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        TriContourSet._process_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        TriContourSet._process_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TriContourSet._process_args.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TriContourSet._process_args', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_process_args', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_process_args(...)' code ##################

        unicode_294846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, (-1)), 'unicode', u'\n        Process args and kwargs.\n        ')
        
        
        # Call to isinstance(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Obtaining the type of the subscript
        int_294848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 27), 'int')
        # Getting the type of 'args' (line 46)
        args_294849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___294850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 22), args_294849, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_294851 = invoke(stypy.reporting.localization.Localization(__file__, 46, 22), getitem___294850, int_294848)
        
        # Getting the type of 'TriContourSet' (line 46)
        TriContourSet_294852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 31), 'TriContourSet', False)
        # Processing the call keyword arguments (line 46)
        kwargs_294853 = {}
        # Getting the type of 'isinstance' (line 46)
        isinstance_294847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 46)
        isinstance_call_result_294854 = invoke(stypy.reporting.localization.Localization(__file__, 46, 11), isinstance_294847, *[subscript_call_result_294851, TriContourSet_294852], **kwargs_294853)
        
        # Testing the type of an if condition (line 46)
        if_condition_294855 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 8), isinstance_call_result_294854)
        # Assigning a type to the variable 'if_condition_294855' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'if_condition_294855', if_condition_294855)
        # SSA begins for if statement (line 46)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 47):
        
        # Assigning a Attribute to a Name (line 47):
        
        # Obtaining the type of the subscript
        int_294856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'int')
        # Getting the type of 'args' (line 47)
        args_294857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'args')
        # Obtaining the member '__getitem__' of a type (line 47)
        getitem___294858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), args_294857, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 47)
        subscript_call_result_294859 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), getitem___294858, int_294856)
        
        # Obtaining the member 'cppContourGenerator' of a type (line 47)
        cppContourGenerator_294860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), subscript_call_result_294859, 'cppContourGenerator')
        # Assigning a type to the variable 'C' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'C', cppContourGenerator_294860)
        
        # Type idiom detected: calculating its left and rigth part (line 48)
        # Getting the type of 'self' (line 48)
        self_294861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'self')
        # Obtaining the member 'levels' of a type (line 48)
        levels_294862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 15), self_294861, 'levels')
        # Getting the type of 'None' (line 48)
        None_294863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'None')
        
        (may_be_294864, more_types_in_union_294865) = may_be_none(levels_294862, None_294863)

        if may_be_294864:

            if more_types_in_union_294865:
                # Runtime conditional SSA (line 48)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 49):
            
            # Assigning a Attribute to a Attribute (line 49):
            
            # Obtaining the type of the subscript
            int_294866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 35), 'int')
            # Getting the type of 'args' (line 49)
            args_294867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'args')
            # Obtaining the member '__getitem__' of a type (line 49)
            getitem___294868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 30), args_294867, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 49)
            subscript_call_result_294869 = invoke(stypy.reporting.localization.Localization(__file__, 49, 30), getitem___294868, int_294866)
            
            # Obtaining the member 'levels' of a type (line 49)
            levels_294870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 30), subscript_call_result_294869, 'levels')
            # Getting the type of 'self' (line 49)
            self_294871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'self')
            # Setting the type of the member 'levels' of a type (line 49)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), self_294871, 'levels', levels_294870)

            if more_types_in_union_294865:
                # SSA join for if statement (line 48)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the else part of an if statement (line 46)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 51):
        
        # Assigning a Call to a Name:
        
        # Call to _contour_args(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'args' (line 51)
        args_294874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 40), 'args', False)
        # Getting the type of 'kwargs' (line 51)
        kwargs_294875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 46), 'kwargs', False)
        # Processing the call keyword arguments (line 51)
        kwargs_294876 = {}
        # Getting the type of 'self' (line 51)
        self_294872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 21), 'self', False)
        # Obtaining the member '_contour_args' of a type (line 51)
        _contour_args_294873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 21), self_294872, '_contour_args')
        # Calling _contour_args(args, kwargs) (line 51)
        _contour_args_call_result_294877 = invoke(stypy.reporting.localization.Localization(__file__, 51, 21), _contour_args_294873, *[args_294874, kwargs_294875], **kwargs_294876)
        
        # Assigning a type to the variable 'call_assignment_294812' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'call_assignment_294812', _contour_args_call_result_294877)
        
        # Assigning a Call to a Name (line 51):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_294880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 12), 'int')
        # Processing the call keyword arguments
        kwargs_294881 = {}
        # Getting the type of 'call_assignment_294812' (line 51)
        call_assignment_294812_294878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'call_assignment_294812', False)
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___294879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), call_assignment_294812_294878, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_294882 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___294879, *[int_294880], **kwargs_294881)
        
        # Assigning a type to the variable 'call_assignment_294813' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'call_assignment_294813', getitem___call_result_294882)
        
        # Assigning a Name to a Name (line 51):
        # Getting the type of 'call_assignment_294813' (line 51)
        call_assignment_294813_294883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'call_assignment_294813')
        # Assigning a type to the variable 'tri' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'tri', call_assignment_294813_294883)
        
        # Assigning a Call to a Name (line 51):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_294886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 12), 'int')
        # Processing the call keyword arguments
        kwargs_294887 = {}
        # Getting the type of 'call_assignment_294812' (line 51)
        call_assignment_294812_294884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'call_assignment_294812', False)
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___294885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), call_assignment_294812_294884, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_294888 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___294885, *[int_294886], **kwargs_294887)
        
        # Assigning a type to the variable 'call_assignment_294814' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'call_assignment_294814', getitem___call_result_294888)
        
        # Assigning a Name to a Name (line 51):
        # Getting the type of 'call_assignment_294814' (line 51)
        call_assignment_294814_294889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'call_assignment_294814')
        # Assigning a type to the variable 'z' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'z', call_assignment_294814_294889)
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to TriContourGenerator(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Call to get_cpp_triangulation(...): (line 52)
        # Processing the call keyword arguments (line 52)
        kwargs_294894 = {}
        # Getting the type of 'tri' (line 52)
        tri_294892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 41), 'tri', False)
        # Obtaining the member 'get_cpp_triangulation' of a type (line 52)
        get_cpp_triangulation_294893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 41), tri_294892, 'get_cpp_triangulation')
        # Calling get_cpp_triangulation(args, kwargs) (line 52)
        get_cpp_triangulation_call_result_294895 = invoke(stypy.reporting.localization.Localization(__file__, 52, 41), get_cpp_triangulation_294893, *[], **kwargs_294894)
        
        # Getting the type of 'z' (line 52)
        z_294896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 70), 'z', False)
        # Processing the call keyword arguments (line 52)
        kwargs_294897 = {}
        # Getting the type of '_tri' (line 52)
        _tri_294890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), '_tri', False)
        # Obtaining the member 'TriContourGenerator' of a type (line 52)
        TriContourGenerator_294891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), _tri_294890, 'TriContourGenerator')
        # Calling TriContourGenerator(args, kwargs) (line 52)
        TriContourGenerator_call_result_294898 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), TriContourGenerator_294891, *[get_cpp_triangulation_call_result_294895, z_294896], **kwargs_294897)
        
        # Assigning a type to the variable 'C' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'C', TriContourGenerator_call_result_294898)
        
        # Assigning a List to a Attribute (line 53):
        
        # Assigning a List to a Attribute (line 53):
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_294899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        
        # Call to min(...): (line 53)
        # Processing the call keyword arguments (line 53)
        kwargs_294903 = {}
        # Getting the type of 'tri' (line 53)
        tri_294900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'tri', False)
        # Obtaining the member 'x' of a type (line 53)
        x_294901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 26), tri_294900, 'x')
        # Obtaining the member 'min' of a type (line 53)
        min_294902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 26), x_294901, 'min')
        # Calling min(args, kwargs) (line 53)
        min_call_result_294904 = invoke(stypy.reporting.localization.Localization(__file__, 53, 26), min_294902, *[], **kwargs_294903)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 25), list_294899, min_call_result_294904)
        # Adding element type (line 53)
        
        # Call to min(...): (line 53)
        # Processing the call keyword arguments (line 53)
        kwargs_294908 = {}
        # Getting the type of 'tri' (line 53)
        tri_294905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 39), 'tri', False)
        # Obtaining the member 'y' of a type (line 53)
        y_294906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 39), tri_294905, 'y')
        # Obtaining the member 'min' of a type (line 53)
        min_294907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 39), y_294906, 'min')
        # Calling min(args, kwargs) (line 53)
        min_call_result_294909 = invoke(stypy.reporting.localization.Localization(__file__, 53, 39), min_294907, *[], **kwargs_294908)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 25), list_294899, min_call_result_294909)
        
        # Getting the type of 'self' (line 53)
        self_294910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'self')
        # Setting the type of the member '_mins' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 12), self_294910, '_mins', list_294899)
        
        # Assigning a List to a Attribute (line 54):
        
        # Assigning a List to a Attribute (line 54):
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_294911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        
        # Call to max(...): (line 54)
        # Processing the call keyword arguments (line 54)
        kwargs_294915 = {}
        # Getting the type of 'tri' (line 54)
        tri_294912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 26), 'tri', False)
        # Obtaining the member 'x' of a type (line 54)
        x_294913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 26), tri_294912, 'x')
        # Obtaining the member 'max' of a type (line 54)
        max_294914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 26), x_294913, 'max')
        # Calling max(args, kwargs) (line 54)
        max_call_result_294916 = invoke(stypy.reporting.localization.Localization(__file__, 54, 26), max_294914, *[], **kwargs_294915)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 25), list_294911, max_call_result_294916)
        # Adding element type (line 54)
        
        # Call to max(...): (line 54)
        # Processing the call keyword arguments (line 54)
        kwargs_294920 = {}
        # Getting the type of 'tri' (line 54)
        tri_294917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 39), 'tri', False)
        # Obtaining the member 'y' of a type (line 54)
        y_294918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 39), tri_294917, 'y')
        # Obtaining the member 'max' of a type (line 54)
        max_294919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 39), y_294918, 'max')
        # Calling max(args, kwargs) (line 54)
        max_call_result_294921 = invoke(stypy.reporting.localization.Localization(__file__, 54, 39), max_294919, *[], **kwargs_294920)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 25), list_294911, max_call_result_294921)
        
        # Getting the type of 'self' (line 54)
        self_294922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'self')
        # Setting the type of the member '_maxs' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), self_294922, '_maxs', list_294911)
        # SSA join for if statement (line 46)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 56):
        
        # Assigning a Name to a Attribute (line 56):
        # Getting the type of 'C' (line 56)
        C_294923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 35), 'C')
        # Getting the type of 'self' (line 56)
        self_294924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'cppContourGenerator' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_294924, 'cppContourGenerator', C_294923)
        # Getting the type of 'kwargs' (line 57)
        kwargs_294925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'kwargs')
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', kwargs_294925)
        
        # ################# End of '_process_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_process_args' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_294926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294926)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_process_args'
        return stypy_return_type_294926


    @norecursion
    def _get_allsegs_and_allkinds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_allsegs_and_allkinds'
        module_type_store = module_type_store.open_function_context('_get_allsegs_and_allkinds', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TriContourSet._get_allsegs_and_allkinds.__dict__.__setitem__('stypy_localization', localization)
        TriContourSet._get_allsegs_and_allkinds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TriContourSet._get_allsegs_and_allkinds.__dict__.__setitem__('stypy_type_store', module_type_store)
        TriContourSet._get_allsegs_and_allkinds.__dict__.__setitem__('stypy_function_name', 'TriContourSet._get_allsegs_and_allkinds')
        TriContourSet._get_allsegs_and_allkinds.__dict__.__setitem__('stypy_param_names_list', [])
        TriContourSet._get_allsegs_and_allkinds.__dict__.__setitem__('stypy_varargs_param_name', None)
        TriContourSet._get_allsegs_and_allkinds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TriContourSet._get_allsegs_and_allkinds.__dict__.__setitem__('stypy_call_defaults', defaults)
        TriContourSet._get_allsegs_and_allkinds.__dict__.__setitem__('stypy_call_varargs', varargs)
        TriContourSet._get_allsegs_and_allkinds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TriContourSet._get_allsegs_and_allkinds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TriContourSet._get_allsegs_and_allkinds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_allsegs_and_allkinds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_allsegs_and_allkinds(...)' code ##################

        unicode_294927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'unicode', u'\n        Create and return allsegs and allkinds by calling underlying C code.\n        ')
        
        # Assigning a List to a Name (line 63):
        
        # Assigning a List to a Name (line 63):
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_294928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        
        # Assigning a type to the variable 'allsegs' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'allsegs', list_294928)
        
        # Getting the type of 'self' (line 64)
        self_294929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'self')
        # Obtaining the member 'filled' of a type (line 64)
        filled_294930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), self_294929, 'filled')
        # Testing the type of an if condition (line 64)
        if_condition_294931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), filled_294930)
        # Assigning a type to the variable 'if_condition_294931' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_294931', if_condition_294931)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 65):
        
        # Assigning a Call to a Name:
        
        # Call to _get_lowers_and_uppers(...): (line 65)
        # Processing the call keyword arguments (line 65)
        kwargs_294934 = {}
        # Getting the type of 'self' (line 65)
        self_294932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 29), 'self', False)
        # Obtaining the member '_get_lowers_and_uppers' of a type (line 65)
        _get_lowers_and_uppers_294933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 29), self_294932, '_get_lowers_and_uppers')
        # Calling _get_lowers_and_uppers(args, kwargs) (line 65)
        _get_lowers_and_uppers_call_result_294935 = invoke(stypy.reporting.localization.Localization(__file__, 65, 29), _get_lowers_and_uppers_294933, *[], **kwargs_294934)
        
        # Assigning a type to the variable 'call_assignment_294815' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'call_assignment_294815', _get_lowers_and_uppers_call_result_294935)
        
        # Assigning a Call to a Name (line 65):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_294938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 12), 'int')
        # Processing the call keyword arguments
        kwargs_294939 = {}
        # Getting the type of 'call_assignment_294815' (line 65)
        call_assignment_294815_294936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'call_assignment_294815', False)
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___294937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), call_assignment_294815_294936, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_294940 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___294937, *[int_294938], **kwargs_294939)
        
        # Assigning a type to the variable 'call_assignment_294816' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'call_assignment_294816', getitem___call_result_294940)
        
        # Assigning a Name to a Name (line 65):
        # Getting the type of 'call_assignment_294816' (line 65)
        call_assignment_294816_294941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'call_assignment_294816')
        # Assigning a type to the variable 'lowers' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'lowers', call_assignment_294816_294941)
        
        # Assigning a Call to a Name (line 65):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_294944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 12), 'int')
        # Processing the call keyword arguments
        kwargs_294945 = {}
        # Getting the type of 'call_assignment_294815' (line 65)
        call_assignment_294815_294942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'call_assignment_294815', False)
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___294943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), call_assignment_294815_294942, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_294946 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___294943, *[int_294944], **kwargs_294945)
        
        # Assigning a type to the variable 'call_assignment_294817' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'call_assignment_294817', getitem___call_result_294946)
        
        # Assigning a Name to a Name (line 65):
        # Getting the type of 'call_assignment_294817' (line 65)
        call_assignment_294817_294947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'call_assignment_294817')
        # Assigning a type to the variable 'uppers' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'uppers', call_assignment_294817_294947)
        
        # Assigning a List to a Name (line 66):
        
        # Assigning a List to a Name (line 66):
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_294948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        
        # Assigning a type to the variable 'allkinds' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'allkinds', list_294948)
        
        
        # Call to zip(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'lowers' (line 67)
        lowers_294950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 36), 'lowers', False)
        # Getting the type of 'uppers' (line 67)
        uppers_294951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 44), 'uppers', False)
        # Processing the call keyword arguments (line 67)
        kwargs_294952 = {}
        # Getting the type of 'zip' (line 67)
        zip_294949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 32), 'zip', False)
        # Calling zip(args, kwargs) (line 67)
        zip_call_result_294953 = invoke(stypy.reporting.localization.Localization(__file__, 67, 32), zip_294949, *[lowers_294950, uppers_294951], **kwargs_294952)
        
        # Testing the type of a for loop iterable (line 67)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 67, 12), zip_call_result_294953)
        # Getting the type of the for loop variable (line 67)
        for_loop_var_294954 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 67, 12), zip_call_result_294953)
        # Assigning a type to the variable 'lower' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'lower', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 12), for_loop_var_294954))
        # Assigning a type to the variable 'upper' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'upper', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 12), for_loop_var_294954))
        # SSA begins for a for statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 68):
        
        # Assigning a Call to a Name:
        
        # Call to create_filled_contour(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'lower' (line 69)
        lower_294958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'lower', False)
        # Getting the type of 'upper' (line 69)
        upper_294959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'upper', False)
        # Processing the call keyword arguments (line 68)
        kwargs_294960 = {}
        # Getting the type of 'self' (line 68)
        self_294955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'self', False)
        # Obtaining the member 'cppContourGenerator' of a type (line 68)
        cppContourGenerator_294956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 30), self_294955, 'cppContourGenerator')
        # Obtaining the member 'create_filled_contour' of a type (line 68)
        create_filled_contour_294957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 30), cppContourGenerator_294956, 'create_filled_contour')
        # Calling create_filled_contour(args, kwargs) (line 68)
        create_filled_contour_call_result_294961 = invoke(stypy.reporting.localization.Localization(__file__, 68, 30), create_filled_contour_294957, *[lower_294958, upper_294959], **kwargs_294960)
        
        # Assigning a type to the variable 'call_assignment_294818' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'call_assignment_294818', create_filled_contour_call_result_294961)
        
        # Assigning a Call to a Name (line 68):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_294964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 16), 'int')
        # Processing the call keyword arguments
        kwargs_294965 = {}
        # Getting the type of 'call_assignment_294818' (line 68)
        call_assignment_294818_294962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'call_assignment_294818', False)
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___294963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), call_assignment_294818_294962, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_294966 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___294963, *[int_294964], **kwargs_294965)
        
        # Assigning a type to the variable 'call_assignment_294819' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'call_assignment_294819', getitem___call_result_294966)
        
        # Assigning a Name to a Name (line 68):
        # Getting the type of 'call_assignment_294819' (line 68)
        call_assignment_294819_294967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'call_assignment_294819')
        # Assigning a type to the variable 'segs' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'segs', call_assignment_294819_294967)
        
        # Assigning a Call to a Name (line 68):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_294970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 16), 'int')
        # Processing the call keyword arguments
        kwargs_294971 = {}
        # Getting the type of 'call_assignment_294818' (line 68)
        call_assignment_294818_294968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'call_assignment_294818', False)
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___294969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), call_assignment_294818_294968, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_294972 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___294969, *[int_294970], **kwargs_294971)
        
        # Assigning a type to the variable 'call_assignment_294820' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'call_assignment_294820', getitem___call_result_294972)
        
        # Assigning a Name to a Name (line 68):
        # Getting the type of 'call_assignment_294820' (line 68)
        call_assignment_294820_294973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'call_assignment_294820')
        # Assigning a type to the variable 'kinds' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 22), 'kinds', call_assignment_294820_294973)
        
        # Call to append(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_294976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        # Getting the type of 'segs' (line 70)
        segs_294977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'segs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 31), list_294976, segs_294977)
        
        # Processing the call keyword arguments (line 70)
        kwargs_294978 = {}
        # Getting the type of 'allsegs' (line 70)
        allsegs_294974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'allsegs', False)
        # Obtaining the member 'append' of a type (line 70)
        append_294975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), allsegs_294974, 'append')
        # Calling append(args, kwargs) (line 70)
        append_call_result_294979 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), append_294975, *[list_294976], **kwargs_294978)
        
        
        # Call to append(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_294982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        # Getting the type of 'kinds' (line 71)
        kinds_294983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 33), 'kinds', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 32), list_294982, kinds_294983)
        
        # Processing the call keyword arguments (line 71)
        kwargs_294984 = {}
        # Getting the type of 'allkinds' (line 71)
        allkinds_294980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'allkinds', False)
        # Obtaining the member 'append' of a type (line 71)
        append_294981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 16), allkinds_294980, 'append')
        # Calling append(args, kwargs) (line 71)
        append_call_result_294985 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), append_294981, *[list_294982], **kwargs_294984)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 64)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 73):
        
        # Assigning a Name to a Name (line 73):
        # Getting the type of 'None' (line 73)
        None_294986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'None')
        # Assigning a type to the variable 'allkinds' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'allkinds', None_294986)
        
        # Getting the type of 'self' (line 74)
        self_294987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'self')
        # Obtaining the member 'levels' of a type (line 74)
        levels_294988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 25), self_294987, 'levels')
        # Testing the type of a for loop iterable (line 74)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 12), levels_294988)
        # Getting the type of the for loop variable (line 74)
        for_loop_var_294989 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 12), levels_294988)
        # Assigning a type to the variable 'level' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'level', for_loop_var_294989)
        # SSA begins for a for statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to create_contour(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'level' (line 75)
        level_294993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 63), 'level', False)
        # Processing the call keyword arguments (line 75)
        kwargs_294994 = {}
        # Getting the type of 'self' (line 75)
        self_294990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'self', False)
        # Obtaining the member 'cppContourGenerator' of a type (line 75)
        cppContourGenerator_294991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 23), self_294990, 'cppContourGenerator')
        # Obtaining the member 'create_contour' of a type (line 75)
        create_contour_294992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 23), cppContourGenerator_294991, 'create_contour')
        # Calling create_contour(args, kwargs) (line 75)
        create_contour_call_result_294995 = invoke(stypy.reporting.localization.Localization(__file__, 75, 23), create_contour_294992, *[level_294993], **kwargs_294994)
        
        # Assigning a type to the variable 'segs' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'segs', create_contour_call_result_294995)
        
        # Call to append(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'segs' (line 76)
        segs_294998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'segs', False)
        # Processing the call keyword arguments (line 76)
        kwargs_294999 = {}
        # Getting the type of 'allsegs' (line 76)
        allsegs_294996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'allsegs', False)
        # Obtaining the member 'append' of a type (line 76)
        append_294997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 16), allsegs_294996, 'append')
        # Calling append(args, kwargs) (line 76)
        append_call_result_295000 = invoke(stypy.reporting.localization.Localization(__file__, 76, 16), append_294997, *[segs_294998], **kwargs_294999)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 77)
        tuple_295001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 77)
        # Adding element type (line 77)
        # Getting the type of 'allsegs' (line 77)
        allsegs_295002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'allsegs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 15), tuple_295001, allsegs_295002)
        # Adding element type (line 77)
        # Getting the type of 'allkinds' (line 77)
        allkinds_295003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'allkinds')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 15), tuple_295001, allkinds_295003)
        
        # Assigning a type to the variable 'stypy_return_type' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'stypy_return_type', tuple_295001)
        
        # ################# End of '_get_allsegs_and_allkinds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_allsegs_and_allkinds' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_295004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295004)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_allsegs_and_allkinds'
        return stypy_return_type_295004


    @norecursion
    def _contour_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_contour_args'
        module_type_store = module_type_store.open_function_context('_contour_args', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TriContourSet._contour_args.__dict__.__setitem__('stypy_localization', localization)
        TriContourSet._contour_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TriContourSet._contour_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        TriContourSet._contour_args.__dict__.__setitem__('stypy_function_name', 'TriContourSet._contour_args')
        TriContourSet._contour_args.__dict__.__setitem__('stypy_param_names_list', ['args', 'kwargs'])
        TriContourSet._contour_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        TriContourSet._contour_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TriContourSet._contour_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        TriContourSet._contour_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        TriContourSet._contour_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TriContourSet._contour_args.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TriContourSet._contour_args', ['args', 'kwargs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_contour_args', localization, ['args', 'kwargs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_contour_args(...)' code ##################

        
        # Getting the type of 'self' (line 80)
        self_295005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'self')
        # Obtaining the member 'filled' of a type (line 80)
        filled_295006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 11), self_295005, 'filled')
        # Testing the type of an if condition (line 80)
        if_condition_295007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), filled_295006)
        # Assigning a type to the variable 'if_condition_295007' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_295007', if_condition_295007)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 81):
        
        # Assigning a Str to a Name (line 81):
        unicode_295008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 17), 'unicode', u'contourf')
        # Assigning a type to the variable 'fn' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'fn', unicode_295008)
        # SSA branch for the else part of an if statement (line 80)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 83):
        
        # Assigning a Str to a Name (line 83):
        unicode_295009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 17), 'unicode', u'contour')
        # Assigning a type to the variable 'fn' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'fn', unicode_295009)
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 84):
        
        # Assigning a Call to a Name:
        
        # Call to get_from_args_and_kwargs(...): (line 84)
        # Getting the type of 'args' (line 84)
        args_295012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 68), 'args', False)
        # Processing the call keyword arguments (line 84)
        # Getting the type of 'kwargs' (line 85)
        kwargs_295013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 69), 'kwargs', False)
        kwargs_295014 = {'kwargs_295013': kwargs_295013}
        # Getting the type of 'Triangulation' (line 84)
        Triangulation_295010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 28), 'Triangulation', False)
        # Obtaining the member 'get_from_args_and_kwargs' of a type (line 84)
        get_from_args_and_kwargs_295011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 28), Triangulation_295010, 'get_from_args_and_kwargs')
        # Calling get_from_args_and_kwargs(args, kwargs) (line 84)
        get_from_args_and_kwargs_call_result_295015 = invoke(stypy.reporting.localization.Localization(__file__, 84, 28), get_from_args_and_kwargs_295011, *[args_295012], **kwargs_295014)
        
        # Assigning a type to the variable 'call_assignment_294821' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'call_assignment_294821', get_from_args_and_kwargs_call_result_295015)
        
        # Assigning a Call to a Name (line 84):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_295018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'int')
        # Processing the call keyword arguments
        kwargs_295019 = {}
        # Getting the type of 'call_assignment_294821' (line 84)
        call_assignment_294821_295016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'call_assignment_294821', False)
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___295017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), call_assignment_294821_295016, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_295020 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___295017, *[int_295018], **kwargs_295019)
        
        # Assigning a type to the variable 'call_assignment_294822' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'call_assignment_294822', getitem___call_result_295020)
        
        # Assigning a Name to a Name (line 84):
        # Getting the type of 'call_assignment_294822' (line 84)
        call_assignment_294822_295021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'call_assignment_294822')
        # Assigning a type to the variable 'tri' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tri', call_assignment_294822_295021)
        
        # Assigning a Call to a Name (line 84):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_295024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'int')
        # Processing the call keyword arguments
        kwargs_295025 = {}
        # Getting the type of 'call_assignment_294821' (line 84)
        call_assignment_294821_295022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'call_assignment_294821', False)
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___295023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), call_assignment_294821_295022, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_295026 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___295023, *[int_295024], **kwargs_295025)
        
        # Assigning a type to the variable 'call_assignment_294823' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'call_assignment_294823', getitem___call_result_295026)
        
        # Assigning a Name to a Name (line 84):
        # Getting the type of 'call_assignment_294823' (line 84)
        call_assignment_294823_295027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'call_assignment_294823')
        # Assigning a type to the variable 'args' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'args', call_assignment_294823_295027)
        
        # Assigning a Call to a Name (line 84):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_295030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'int')
        # Processing the call keyword arguments
        kwargs_295031 = {}
        # Getting the type of 'call_assignment_294821' (line 84)
        call_assignment_294821_295028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'call_assignment_294821', False)
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___295029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), call_assignment_294821_295028, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_295032 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___295029, *[int_295030], **kwargs_295031)
        
        # Assigning a type to the variable 'call_assignment_294824' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'call_assignment_294824', getitem___call_result_295032)
        
        # Assigning a Name to a Name (line 84):
        # Getting the type of 'call_assignment_294824' (line 84)
        call_assignment_294824_295033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'call_assignment_294824')
        # Assigning a type to the variable 'kwargs' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'kwargs', call_assignment_294824_295033)
        
        # Assigning a Call to a Name (line 86):
        
        # Assigning a Call to a Name (line 86):
        
        # Call to asarray(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Obtaining the type of the subscript
        int_295036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 28), 'int')
        # Getting the type of 'args' (line 86)
        args_295037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___295038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 23), args_295037, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_295039 = invoke(stypy.reporting.localization.Localization(__file__, 86, 23), getitem___295038, int_295036)
        
        # Processing the call keyword arguments (line 86)
        kwargs_295040 = {}
        # Getting the type of 'np' (line 86)
        np_295034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 86)
        asarray_295035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), np_295034, 'asarray')
        # Calling asarray(args, kwargs) (line 86)
        asarray_call_result_295041 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), asarray_295035, *[subscript_call_result_295039], **kwargs_295040)
        
        # Assigning a type to the variable 'z' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'z', asarray_call_result_295041)
        
        
        # Getting the type of 'z' (line 87)
        z_295042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'z')
        # Obtaining the member 'shape' of a type (line 87)
        shape_295043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 11), z_295042, 'shape')
        # Getting the type of 'tri' (line 87)
        tri_295044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'tri')
        # Obtaining the member 'x' of a type (line 87)
        x_295045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 22), tri_295044, 'x')
        # Obtaining the member 'shape' of a type (line 87)
        shape_295046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 22), x_295045, 'shape')
        # Applying the binary operator '!=' (line 87)
        result_ne_295047 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 11), '!=', shape_295043, shape_295046)
        
        # Testing the type of an if condition (line 87)
        if_condition_295048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 8), result_ne_295047)
        # Assigning a type to the variable 'if_condition_295048' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'if_condition_295048', if_condition_295048)
        # SSA begins for if statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 88)
        # Processing the call arguments (line 88)
        unicode_295050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 29), 'unicode', u'z array must have same length as triangulation x and y arrays')
        # Processing the call keyword arguments (line 88)
        kwargs_295051 = {}
        # Getting the type of 'ValueError' (line 88)
        ValueError_295049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 88)
        ValueError_call_result_295052 = invoke(stypy.reporting.localization.Localization(__file__, 88, 18), ValueError_295049, *[unicode_295050], **kwargs_295051)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 88, 12), ValueError_call_result_295052, 'raise parameter', BaseException)
        # SSA join for if statement (line 87)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 90):
        
        # Assigning a Call to a Attribute (line 90):
        
        # Call to max(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_295055 = {}
        # Getting the type of 'z' (line 90)
        z_295053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'z', False)
        # Obtaining the member 'max' of a type (line 90)
        max_295054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 20), z_295053, 'max')
        # Calling max(args, kwargs) (line 90)
        max_call_result_295056 = invoke(stypy.reporting.localization.Localization(__file__, 90, 20), max_295054, *[], **kwargs_295055)
        
        # Getting the type of 'self' (line 90)
        self_295057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Setting the type of the member 'zmax' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_295057, 'zmax', max_call_result_295056)
        
        # Assigning a Call to a Attribute (line 91):
        
        # Assigning a Call to a Attribute (line 91):
        
        # Call to min(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_295060 = {}
        # Getting the type of 'z' (line 91)
        z_295058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'z', False)
        # Obtaining the member 'min' of a type (line 91)
        min_295059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), z_295058, 'min')
        # Calling min(args, kwargs) (line 91)
        min_call_result_295061 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), min_295059, *[], **kwargs_295060)
        
        # Getting the type of 'self' (line 91)
        self_295062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member 'zmin' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_295062, 'zmin', min_call_result_295061)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 92)
        self_295063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'self')
        # Obtaining the member 'logscale' of a type (line 92)
        logscale_295064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 11), self_295063, 'logscale')
        
        # Getting the type of 'self' (line 92)
        self_295065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 29), 'self')
        # Obtaining the member 'zmin' of a type (line 92)
        zmin_295066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 29), self_295065, 'zmin')
        int_295067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 42), 'int')
        # Applying the binary operator '<=' (line 92)
        result_le_295068 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 29), '<=', zmin_295066, int_295067)
        
        # Applying the binary operator 'and' (line 92)
        result_and_keyword_295069 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 11), 'and', logscale_295064, result_le_295068)
        
        # Testing the type of an if condition (line 92)
        if_condition_295070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 8), result_and_keyword_295069)
        # Assigning a type to the variable 'if_condition_295070' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'if_condition_295070', if_condition_295070)
        # SSA begins for if statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 93)
        # Processing the call arguments (line 93)
        unicode_295072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 29), 'unicode', u'Cannot %s log of negative values.')
        # Getting the type of 'fn' (line 93)
        fn_295073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 67), 'fn', False)
        # Applying the binary operator '%' (line 93)
        result_mod_295074 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 29), '%', unicode_295072, fn_295073)
        
        # Processing the call keyword arguments (line 93)
        kwargs_295075 = {}
        # Getting the type of 'ValueError' (line 93)
        ValueError_295071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 93)
        ValueError_call_result_295076 = invoke(stypy.reporting.localization.Localization(__file__, 93, 18), ValueError_295071, *[result_mod_295074], **kwargs_295075)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 93, 12), ValueError_call_result_295076, 'raise parameter', BaseException)
        # SSA join for if statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _contour_level_args(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'z' (line 94)
        z_295079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 33), 'z', False)
        
        # Obtaining the type of the subscript
        int_295080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 41), 'int')
        slice_295081 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 94, 36), int_295080, None, None)
        # Getting the type of 'args' (line 94)
        args_295082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 36), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___295083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 36), args_295082, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_295084 = invoke(stypy.reporting.localization.Localization(__file__, 94, 36), getitem___295083, slice_295081)
        
        # Processing the call keyword arguments (line 94)
        kwargs_295085 = {}
        # Getting the type of 'self' (line 94)
        self_295077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self', False)
        # Obtaining the member '_contour_level_args' of a type (line 94)
        _contour_level_args_295078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_295077, '_contour_level_args')
        # Calling _contour_level_args(args, kwargs) (line 94)
        _contour_level_args_call_result_295086 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), _contour_level_args_295078, *[z_295079, subscript_call_result_295084], **kwargs_295085)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 95)
        tuple_295087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 95)
        # Adding element type (line 95)
        # Getting the type of 'tri' (line 95)
        tri_295088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'tri')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 16), tuple_295087, tri_295088)
        # Adding element type (line 95)
        # Getting the type of 'z' (line 95)
        z_295089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 16), tuple_295087, z_295089)
        
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'stypy_return_type', tuple_295087)
        
        # ################# End of '_contour_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_contour_args' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_295090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295090)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_contour_args'
        return stypy_return_type_295090

    
    # Assigning a Str to a Name (line 97):

# Assigning a type to the variable 'TriContourSet' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'TriContourSet', TriContourSet)

# Assigning a Str to a Name (line 97):
unicode_295091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, (-1)), 'unicode', u"\n        Draw contours on an unstructured triangular grid.\n        :func:`~matplotlib.pyplot.tricontour` and\n        :func:`~matplotlib.pyplot.tricontourf` draw contour lines and\n        filled contours, respectively.  Except as noted, function\n        signatures and return values are the same for both versions.\n\n        The triangulation can be specified in one of two ways; either::\n\n          tricontour(triangulation, ...)\n\n        where triangulation is a :class:`matplotlib.tri.Triangulation`\n        object, or\n\n        ::\n\n          tricontour(x, y, ...)\n          tricontour(x, y, triangles, ...)\n          tricontour(x, y, triangles=triangles, ...)\n          tricontour(x, y, mask=mask, ...)\n          tricontour(x, y, triangles, mask=mask, ...)\n\n        in which case a Triangulation object will be created.  See\n        :class:`~matplotlib.tri.Triangulation` for a explanation of\n        these possibilities.\n\n        The remaining arguments may be::\n\n          tricontour(..., Z)\n\n        where *Z* is the array of values to contour, one per point\n        in the triangulation.  The level values are chosen\n        automatically.\n\n        ::\n\n          tricontour(..., Z, N)\n\n        contour *N* automatically-chosen levels.\n\n        ::\n\n          tricontour(..., Z, V)\n\n        draw contour lines at the values specified in sequence *V*,\n        which must be in increasing order.\n\n        ::\n\n          tricontourf(..., Z, V)\n\n        fill the (len(*V*)-1) regions between the values in *V*,\n        which must be in increasing order.\n\n        ::\n\n          tricontour(Z, **kwargs)\n\n        Use keyword args to control colors, linewidth, origin, cmap ... see\n        below for more details.\n\n        ``C = tricontour(...)`` returns a\n        :class:`~matplotlib.contour.TriContourSet` object.\n\n        Optional keyword arguments:\n\n          *colors*: [ *None* | string | (mpl_colors) ]\n            If *None*, the colormap specified by cmap will be used.\n\n            If a string, like 'r' or 'red', all levels will be plotted in this\n            color.\n\n            If a tuple of matplotlib color args (string, float, rgb, etc),\n            different levels will be plotted in different colors in the order\n            specified.\n\n          *alpha*: float\n            The alpha blending value\n\n          *cmap*: [ *None* | Colormap ]\n            A cm :class:`~matplotlib.colors.Colormap` instance or\n            *None*. If *cmap* is *None* and *colors* is *None*, a\n            default Colormap is used.\n\n          *norm*: [ *None* | Normalize ]\n            A :class:`matplotlib.colors.Normalize` instance for\n            scaling data values to colors. If *norm* is *None* and\n            *colors* is *None*, the default linear scaling is used.\n\n          *levels* [level0, level1, ..., leveln]\n            A list of floating point numbers indicating the level\n            curves to draw, in increasing order; e.g., to draw just\n            the zero contour pass ``levels=[0]``\n\n          *origin*: [ *None* | 'upper' | 'lower' | 'image' ]\n            If *None*, the first value of *Z* will correspond to the\n            lower left corner, location (0,0). If 'image', the rc\n            value for ``image.origin`` will be used.\n\n            This keyword is not active if *X* and *Y* are specified in\n            the call to contour.\n\n          *extent*: [ *None* | (x0,x1,y0,y1) ]\n\n            If *origin* is not *None*, then *extent* is interpreted as\n            in :func:`matplotlib.pyplot.imshow`: it gives the outer\n            pixel boundaries. In this case, the position of Z[0,0]\n            is the center of the pixel, not a corner. If *origin* is\n            *None*, then (*x0*, *y0*) is the position of Z[0,0], and\n            (*x1*, *y1*) is the position of Z[-1,-1].\n\n            This keyword is not active if *X* and *Y* are specified in\n            the call to contour.\n\n          *locator*: [ *None* | ticker.Locator subclass ]\n            If *locator* is None, the default\n            :class:`~matplotlib.ticker.MaxNLocator` is used. The\n            locator is used to determine the contour levels if they\n            are not given explicitly via the *V* argument.\n\n          *extend*: [ 'neither' | 'both' | 'min' | 'max' ]\n            Unless this is 'neither', contour levels are automatically\n            added to one or both ends of the range so that all data\n            are included. These added ranges are then mapped to the\n            special colormap values which default to the ends of the\n            colormap range, but can be set via\n            :meth:`matplotlib.colors.Colormap.set_under` and\n            :meth:`matplotlib.colors.Colormap.set_over` methods.\n\n          *xunits*, *yunits*: [ *None* | registered units ]\n            Override axis units by specifying an instance of a\n            :class:`matplotlib.units.ConversionInterface`.\n\n\n        tricontour-only keyword arguments:\n\n          *linewidths*: [ *None* | number | tuple of numbers ]\n            If *linewidths* is *None*, the default width in\n            ``lines.linewidth`` in ``matplotlibrc`` is used.\n\n            If a number, all levels will be plotted with this linewidth.\n\n            If a tuple, different levels will be plotted with different\n            linewidths in the order specified\n\n          *linestyles*: [ *None* | 'solid' | 'dashed' | 'dashdot' | 'dotted' ]\n            If *linestyles* is *None*, the 'solid' is used.\n\n            *linestyles* can also be an iterable of the above strings\n            specifying a set of linestyles to be used. If this\n            iterable is shorter than the number of contour levels\n            it will be repeated as necessary.\n\n            If contour is using a monochrome colormap and the contour\n            level is less than 0, then the linestyle specified\n            in ``contour.negative_linestyle`` in ``matplotlibrc``\n            will be used.\n\n        tricontourf-only keyword arguments:\n\n          *antialiased*: [ *True* | *False* ]\n            enable antialiasing\n\n        Note: tricontourf fills intervals that are closed at the top; that\n        is, for boundaries *z1* and *z2*, the filled region is::\n\n            z1 < z <= z2\n\n        There is one exception: if the lowest boundary coincides with\n        the minimum value of the *z* array, then that minimum value\n        will be included in the lowest interval.\n        ")
# Getting the type of 'TriContourSet'
TriContourSet_295092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TriContourSet')
# Setting the type of the member 'tricontour_doc' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TriContourSet_295092, 'tricontour_doc', unicode_295091)

@norecursion
def tricontour(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tricontour'
    module_type_store = module_type_store.open_function_context('tricontour', 271, 0, False)
    
    # Passed parameters checking function
    tricontour.stypy_localization = localization
    tricontour.stypy_type_of_self = None
    tricontour.stypy_type_store = module_type_store
    tricontour.stypy_function_name = 'tricontour'
    tricontour.stypy_param_names_list = ['ax']
    tricontour.stypy_varargs_param_name = 'args'
    tricontour.stypy_kwargs_param_name = 'kwargs'
    tricontour.stypy_call_defaults = defaults
    tricontour.stypy_call_varargs = varargs
    tricontour.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tricontour', ['ax'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tricontour', localization, ['ax'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tricontour(...)' code ##################

    
    
    # Getting the type of 'ax' (line 272)
    ax_295093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'ax')
    # Obtaining the member '_hold' of a type (line 272)
    _hold_295094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 11), ax_295093, '_hold')
    # Applying the 'not' unary operator (line 272)
    result_not__295095 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 7), 'not', _hold_295094)
    
    # Testing the type of an if condition (line 272)
    if_condition_295096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 4), result_not__295095)
    # Assigning a type to the variable 'if_condition_295096' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'if_condition_295096', if_condition_295096)
    # SSA begins for if statement (line 272)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to cla(...): (line 273)
    # Processing the call keyword arguments (line 273)
    kwargs_295099 = {}
    # Getting the type of 'ax' (line 273)
    ax_295097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'ax', False)
    # Obtaining the member 'cla' of a type (line 273)
    cla_295098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), ax_295097, 'cla')
    # Calling cla(args, kwargs) (line 273)
    cla_call_result_295100 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), cla_295098, *[], **kwargs_295099)
    
    # SSA join for if statement (line 272)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 274):
    
    # Assigning a Name to a Subscript (line 274):
    # Getting the type of 'False' (line 274)
    False_295101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 23), 'False')
    # Getting the type of 'kwargs' (line 274)
    kwargs_295102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'kwargs')
    unicode_295103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 11), 'unicode', u'filled')
    # Storing an element on a container (line 274)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 4), kwargs_295102, (unicode_295103, False_295101))
    
    # Call to TriContourSet(...): (line 275)
    # Processing the call arguments (line 275)
    # Getting the type of 'ax' (line 275)
    ax_295105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 25), 'ax', False)
    # Getting the type of 'args' (line 275)
    args_295106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 30), 'args', False)
    # Processing the call keyword arguments (line 275)
    # Getting the type of 'kwargs' (line 275)
    kwargs_295107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 38), 'kwargs', False)
    kwargs_295108 = {'kwargs_295107': kwargs_295107}
    # Getting the type of 'TriContourSet' (line 275)
    TriContourSet_295104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 11), 'TriContourSet', False)
    # Calling TriContourSet(args, kwargs) (line 275)
    TriContourSet_call_result_295109 = invoke(stypy.reporting.localization.Localization(__file__, 275, 11), TriContourSet_295104, *[ax_295105, args_295106], **kwargs_295108)
    
    # Assigning a type to the variable 'stypy_return_type' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'stypy_return_type', TriContourSet_call_result_295109)
    
    # ################# End of 'tricontour(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tricontour' in the type store
    # Getting the type of 'stypy_return_type' (line 271)
    stypy_return_type_295110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_295110)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tricontour'
    return stypy_return_type_295110

# Assigning a type to the variable 'tricontour' (line 271)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 0), 'tricontour', tricontour)

# Assigning a Attribute to a Attribute (line 276):

# Assigning a Attribute to a Attribute (line 276):
# Getting the type of 'TriContourSet' (line 276)
TriContourSet_295111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 21), 'TriContourSet')
# Obtaining the member 'tricontour_doc' of a type (line 276)
tricontour_doc_295112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 21), TriContourSet_295111, 'tricontour_doc')
# Getting the type of 'tricontour' (line 276)
tricontour_295113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 0), 'tricontour')
# Setting the type of the member '__doc__' of a type (line 276)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 0), tricontour_295113, '__doc__', tricontour_doc_295112)

@norecursion
def tricontourf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tricontourf'
    module_type_store = module_type_store.open_function_context('tricontourf', 279, 0, False)
    
    # Passed parameters checking function
    tricontourf.stypy_localization = localization
    tricontourf.stypy_type_of_self = None
    tricontourf.stypy_type_store = module_type_store
    tricontourf.stypy_function_name = 'tricontourf'
    tricontourf.stypy_param_names_list = ['ax']
    tricontourf.stypy_varargs_param_name = 'args'
    tricontourf.stypy_kwargs_param_name = 'kwargs'
    tricontourf.stypy_call_defaults = defaults
    tricontourf.stypy_call_varargs = varargs
    tricontourf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tricontourf', ['ax'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tricontourf', localization, ['ax'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tricontourf(...)' code ##################

    
    
    # Getting the type of 'ax' (line 280)
    ax_295114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'ax')
    # Obtaining the member '_hold' of a type (line 280)
    _hold_295115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 11), ax_295114, '_hold')
    # Applying the 'not' unary operator (line 280)
    result_not__295116 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 7), 'not', _hold_295115)
    
    # Testing the type of an if condition (line 280)
    if_condition_295117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 4), result_not__295116)
    # Assigning a type to the variable 'if_condition_295117' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'if_condition_295117', if_condition_295117)
    # SSA begins for if statement (line 280)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to cla(...): (line 281)
    # Processing the call keyword arguments (line 281)
    kwargs_295120 = {}
    # Getting the type of 'ax' (line 281)
    ax_295118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'ax', False)
    # Obtaining the member 'cla' of a type (line 281)
    cla_295119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), ax_295118, 'cla')
    # Calling cla(args, kwargs) (line 281)
    cla_call_result_295121 = invoke(stypy.reporting.localization.Localization(__file__, 281, 8), cla_295119, *[], **kwargs_295120)
    
    # SSA join for if statement (line 280)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 282):
    
    # Assigning a Name to a Subscript (line 282):
    # Getting the type of 'True' (line 282)
    True_295122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 23), 'True')
    # Getting the type of 'kwargs' (line 282)
    kwargs_295123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'kwargs')
    unicode_295124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 11), 'unicode', u'filled')
    # Storing an element on a container (line 282)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 4), kwargs_295123, (unicode_295124, True_295122))
    
    # Call to TriContourSet(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'ax' (line 283)
    ax_295126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 25), 'ax', False)
    # Getting the type of 'args' (line 283)
    args_295127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 30), 'args', False)
    # Processing the call keyword arguments (line 283)
    # Getting the type of 'kwargs' (line 283)
    kwargs_295128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 38), 'kwargs', False)
    kwargs_295129 = {'kwargs_295128': kwargs_295128}
    # Getting the type of 'TriContourSet' (line 283)
    TriContourSet_295125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 11), 'TriContourSet', False)
    # Calling TriContourSet(args, kwargs) (line 283)
    TriContourSet_call_result_295130 = invoke(stypy.reporting.localization.Localization(__file__, 283, 11), TriContourSet_295125, *[ax_295126, args_295127], **kwargs_295129)
    
    # Assigning a type to the variable 'stypy_return_type' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'stypy_return_type', TriContourSet_call_result_295130)
    
    # ################# End of 'tricontourf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tricontourf' in the type store
    # Getting the type of 'stypy_return_type' (line 279)
    stypy_return_type_295131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_295131)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tricontourf'
    return stypy_return_type_295131

# Assigning a type to the variable 'tricontourf' (line 279)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 0), 'tricontourf', tricontourf)

# Assigning a Attribute to a Attribute (line 284):

# Assigning a Attribute to a Attribute (line 284):
# Getting the type of 'TriContourSet' (line 284)
TriContourSet_295132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 22), 'TriContourSet')
# Obtaining the member 'tricontour_doc' of a type (line 284)
tricontour_doc_295133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 22), TriContourSet_295132, 'tricontour_doc')
# Getting the type of 'tricontourf' (line 284)
tricontourf_295134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), 'tricontourf')
# Setting the type of the member '__doc__' of a type (line 284)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 0), tricontourf_295134, '__doc__', tricontour_doc_295133)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
