
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from scipy._lib.decorator import decorator as _decorator
5: 
6: __all__ = ['delaunay_plot_2d', 'convex_hull_plot_2d', 'voronoi_plot_2d']
7: 
8: 
9: @_decorator
10: def _held_figure(func, obj, ax=None, **kw):
11:     import matplotlib.pyplot as plt
12: 
13:     if ax is None:
14:         fig = plt.figure()
15:         ax = fig.gca()
16:         return func(obj, ax=ax, **kw)
17: 
18:     # As of matplotlib 2.0, the "hold" mechanism is deprecated.
19:     # When matplotlib 1.x is no longer supported, this check can be removed.
20:     was_held = ax.ishold()
21:     if was_held:
22:         return func(obj, ax=ax, **kw)
23:     try:
24:         ax.hold(True)
25:         return func(obj, ax=ax, **kw)
26:     finally:
27:         ax.hold(was_held)
28: 
29: 
30: def _adjust_bounds(ax, points):
31:     margin = 0.1 * points.ptp(axis=0)
32:     xy_min = points.min(axis=0) - margin 
33:     xy_max = points.max(axis=0) + margin 
34:     ax.set_xlim(xy_min[0], xy_max[0])
35:     ax.set_ylim(xy_min[1], xy_max[1])
36: 
37: 
38: @_held_figure
39: def delaunay_plot_2d(tri, ax=None):
40:     '''
41:     Plot the given Delaunay triangulation in 2-D
42: 
43:     Parameters
44:     ----------
45:     tri : scipy.spatial.Delaunay instance
46:         Triangulation to plot
47:     ax : matplotlib.axes.Axes instance, optional
48:         Axes to plot on
49: 
50:     Returns
51:     -------
52:     fig : matplotlib.figure.Figure instance
53:         Figure for the plot
54: 
55:     See Also
56:     --------
57:     Delaunay
58:     matplotlib.pyplot.triplot
59: 
60:     Notes
61:     -----
62:     Requires Matplotlib.
63: 
64:     '''
65:     if tri.points.shape[1] != 2:
66:         raise ValueError("Delaunay triangulation is not 2-D")
67: 
68:     x, y = tri.points.T
69:     ax.plot(x, y, 'o')
70:     ax.triplot(x, y, tri.simplices.copy())
71: 
72:     _adjust_bounds(ax, tri.points)
73: 
74:     return ax.figure
75: 
76: 
77: @_held_figure
78: def convex_hull_plot_2d(hull, ax=None):
79:     '''
80:     Plot the given convex hull diagram in 2-D
81: 
82:     Parameters
83:     ----------
84:     hull : scipy.spatial.ConvexHull instance
85:         Convex hull to plot
86:     ax : matplotlib.axes.Axes instance, optional
87:         Axes to plot on
88: 
89:     Returns
90:     -------
91:     fig : matplotlib.figure.Figure instance
92:         Figure for the plot
93: 
94:     See Also
95:     --------
96:     ConvexHull
97: 
98:     Notes
99:     -----
100:     Requires Matplotlib.
101: 
102:     '''
103:     from matplotlib.collections import LineCollection
104: 
105:     if hull.points.shape[1] != 2:
106:         raise ValueError("Convex hull is not 2-D")
107: 
108:     ax.plot(hull.points[:,0], hull.points[:,1], 'o')
109:     line_segments = [hull.points[simplex] for simplex in hull.simplices]
110:     ax.add_collection(LineCollection(line_segments,
111:                                      colors='k',
112:                                      linestyle='solid'))
113:     _adjust_bounds(ax, hull.points)
114: 
115:     return ax.figure
116: 
117: 
118: @_held_figure
119: def voronoi_plot_2d(vor, ax=None, **kw):
120:     '''
121:     Plot the given Voronoi diagram in 2-D
122: 
123:     Parameters
124:     ----------
125:     vor : scipy.spatial.Voronoi instance
126:         Diagram to plot
127:     ax : matplotlib.axes.Axes instance, optional
128:         Axes to plot on
129:     show_points: bool, optional
130:         Add the Voronoi points to the plot.
131:     show_vertices : bool, optional
132:         Add the Voronoi vertices to the plot.
133:     line_colors : string, optional
134:         Specifies the line color for polygon boundaries
135:     line_width : float, optional
136:         Specifies the line width for polygon boundaries
137:     line_alpha: float, optional
138:         Specifies the line alpha for polygon boundaries
139: 
140:     Returns
141:     -------
142:     fig : matplotlib.figure.Figure instance
143:         Figure for the plot
144: 
145:     See Also
146:     --------
147:     Voronoi
148: 
149:     Notes
150:     -----
151:     Requires Matplotlib.
152: 
153:     '''
154:     from matplotlib.collections import LineCollection
155: 
156:     if vor.points.shape[1] != 2:
157:         raise ValueError("Voronoi diagram is not 2-D")
158: 
159:     if kw.get('show_points', True):
160:         ax.plot(vor.points[:,0], vor.points[:,1], '.')
161:     if kw.get('show_vertices', True):
162:         ax.plot(vor.vertices[:,0], vor.vertices[:,1], 'o')
163: 
164:     line_colors = kw.get('line_colors', 'k')
165:     line_width = kw.get('line_width', 1.0)
166:     line_alpha = kw.get('line_alpha', 1.0)
167: 
168:     center = vor.points.mean(axis=0)
169:     ptp_bound = vor.points.ptp(axis=0)
170: 
171:     finite_segments = []
172:     infinite_segments = []
173:     for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
174:         simplex = np.asarray(simplex)
175:         if np.all(simplex >= 0):
176:             finite_segments.append(vor.vertices[simplex])
177:         else:
178:             i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
179: 
180:             t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
181:             t /= np.linalg.norm(t)
182:             n = np.array([-t[1], t[0]])  # normal
183: 
184:             midpoint = vor.points[pointidx].mean(axis=0)
185:             direction = np.sign(np.dot(midpoint - center, n)) * n
186:             far_point = vor.vertices[i] + direction * ptp_bound.max()
187: 
188:             infinite_segments.append([vor.vertices[i], far_point])
189: 
190:     ax.add_collection(LineCollection(finite_segments,
191:                                      colors=line_colors,
192:                                      lw=line_width,
193:                                      alpha=line_alpha,
194:                                      linestyle='solid'))
195:     ax.add_collection(LineCollection(infinite_segments,
196:                                      colors=line_colors,
197:                                      lw=line_width,
198:                                      alpha=line_alpha,
199:                                      linestyle='dashed'))
200: 
201:     _adjust_bounds(ax, vor.points)
202: 
203:     return ax.figure
204: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_470297 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_470297) is not StypyTypeError):

    if (import_470297 != 'pyd_module'):
        __import__(import_470297)
        sys_modules_470298 = sys.modules[import_470297]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_470298.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_470297)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy._lib.decorator import _decorator' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_470299 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib.decorator')

if (type(import_470299) is not StypyTypeError):

    if (import_470299 != 'pyd_module'):
        __import__(import_470299)
        sys_modules_470300 = sys.modules[import_470299]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib.decorator', sys_modules_470300.module_type_store, module_type_store, ['decorator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_470300, sys_modules_470300.module_type_store, module_type_store)
    else:
        from scipy._lib.decorator import decorator as _decorator

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib.decorator', None, module_type_store, ['decorator'], [_decorator])

else:
    # Assigning a type to the variable 'scipy._lib.decorator' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib.decorator', import_470299)

# Adding an alias
module_type_store.add_alias('_decorator', 'decorator')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')


# Assigning a List to a Name (line 6):

# Assigning a List to a Name (line 6):
__all__ = ['delaunay_plot_2d', 'convex_hull_plot_2d', 'voronoi_plot_2d']
module_type_store.set_exportable_members(['delaunay_plot_2d', 'convex_hull_plot_2d', 'voronoi_plot_2d'])

# Obtaining an instance of the builtin type 'list' (line 6)
list_470301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_470302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'str', 'delaunay_plot_2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_470301, str_470302)
# Adding element type (line 6)
str_470303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 31), 'str', 'convex_hull_plot_2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_470301, str_470303)
# Adding element type (line 6)
str_470304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 54), 'str', 'voronoi_plot_2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_470301, str_470304)

# Assigning a type to the variable '__all__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__all__', list_470301)

@norecursion
def _held_figure(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 10)
    None_470305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 31), 'None')
    defaults = [None_470305]
    # Create a new context for function '_held_figure'
    module_type_store = module_type_store.open_function_context('_held_figure', 9, 0, False)
    
    # Passed parameters checking function
    _held_figure.stypy_localization = localization
    _held_figure.stypy_type_of_self = None
    _held_figure.stypy_type_store = module_type_store
    _held_figure.stypy_function_name = '_held_figure'
    _held_figure.stypy_param_names_list = ['func', 'obj', 'ax']
    _held_figure.stypy_varargs_param_name = None
    _held_figure.stypy_kwargs_param_name = 'kw'
    _held_figure.stypy_call_defaults = defaults
    _held_figure.stypy_call_varargs = varargs
    _held_figure.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_held_figure', ['func', 'obj', 'ax'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_held_figure', localization, ['func', 'obj', 'ax'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_held_figure(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))
    
    # 'import matplotlib.pyplot' statement (line 11)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
    import_470306 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'matplotlib.pyplot')

    if (type(import_470306) is not StypyTypeError):

        if (import_470306 != 'pyd_module'):
            __import__(import_470306)
            sys_modules_470307 = sys.modules[import_470306]
            import_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'plt', sys_modules_470307.module_type_store, module_type_store)
        else:
            import matplotlib.pyplot as plt

            import_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'plt', matplotlib.pyplot, module_type_store)

    else:
        # Assigning a type to the variable 'matplotlib.pyplot' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'matplotlib.pyplot', import_470306)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')
    
    
    # Type idiom detected: calculating its left and rigth part (line 13)
    # Getting the type of 'ax' (line 13)
    ax_470308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 7), 'ax')
    # Getting the type of 'None' (line 13)
    None_470309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'None')
    
    (may_be_470310, more_types_in_union_470311) = may_be_none(ax_470308, None_470309)

    if may_be_470310:

        if more_types_in_union_470311:
            # Runtime conditional SSA (line 13)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 14):
        
        # Assigning a Call to a Name (line 14):
        
        # Call to figure(...): (line 14)
        # Processing the call keyword arguments (line 14)
        kwargs_470314 = {}
        # Getting the type of 'plt' (line 14)
        plt_470312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'plt', False)
        # Obtaining the member 'figure' of a type (line 14)
        figure_470313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 14), plt_470312, 'figure')
        # Calling figure(args, kwargs) (line 14)
        figure_call_result_470315 = invoke(stypy.reporting.localization.Localization(__file__, 14, 14), figure_470313, *[], **kwargs_470314)
        
        # Assigning a type to the variable 'fig' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'fig', figure_call_result_470315)
        
        # Assigning a Call to a Name (line 15):
        
        # Assigning a Call to a Name (line 15):
        
        # Call to gca(...): (line 15)
        # Processing the call keyword arguments (line 15)
        kwargs_470318 = {}
        # Getting the type of 'fig' (line 15)
        fig_470316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'fig', False)
        # Obtaining the member 'gca' of a type (line 15)
        gca_470317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 13), fig_470316, 'gca')
        # Calling gca(args, kwargs) (line 15)
        gca_call_result_470319 = invoke(stypy.reporting.localization.Localization(__file__, 15, 13), gca_470317, *[], **kwargs_470318)
        
        # Assigning a type to the variable 'ax' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'ax', gca_call_result_470319)
        
        # Call to func(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'obj' (line 16)
        obj_470321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'obj', False)
        # Processing the call keyword arguments (line 16)
        # Getting the type of 'ax' (line 16)
        ax_470322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 28), 'ax', False)
        keyword_470323 = ax_470322
        # Getting the type of 'kw' (line 16)
        kw_470324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 34), 'kw', False)
        kwargs_470325 = {'ax': keyword_470323, 'kw_470324': kw_470324}
        # Getting the type of 'func' (line 16)
        func_470320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'func', False)
        # Calling func(args, kwargs) (line 16)
        func_call_result_470326 = invoke(stypy.reporting.localization.Localization(__file__, 16, 15), func_470320, *[obj_470321], **kwargs_470325)
        
        # Assigning a type to the variable 'stypy_return_type' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type', func_call_result_470326)

        if more_types_in_union_470311:
            # SSA join for if statement (line 13)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 20):
    
    # Call to ishold(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_470329 = {}
    # Getting the type of 'ax' (line 20)
    ax_470327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'ax', False)
    # Obtaining the member 'ishold' of a type (line 20)
    ishold_470328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 15), ax_470327, 'ishold')
    # Calling ishold(args, kwargs) (line 20)
    ishold_call_result_470330 = invoke(stypy.reporting.localization.Localization(__file__, 20, 15), ishold_470328, *[], **kwargs_470329)
    
    # Assigning a type to the variable 'was_held' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'was_held', ishold_call_result_470330)
    
    # Getting the type of 'was_held' (line 21)
    was_held_470331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 7), 'was_held')
    # Testing the type of an if condition (line 21)
    if_condition_470332 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 4), was_held_470331)
    # Assigning a type to the variable 'if_condition_470332' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'if_condition_470332', if_condition_470332)
    # SSA begins for if statement (line 21)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to func(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'obj' (line 22)
    obj_470334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'obj', False)
    # Processing the call keyword arguments (line 22)
    # Getting the type of 'ax' (line 22)
    ax_470335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 28), 'ax', False)
    keyword_470336 = ax_470335
    # Getting the type of 'kw' (line 22)
    kw_470337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 34), 'kw', False)
    kwargs_470338 = {'ax': keyword_470336, 'kw_470337': kw_470337}
    # Getting the type of 'func' (line 22)
    func_470333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'func', False)
    # Calling func(args, kwargs) (line 22)
    func_call_result_470339 = invoke(stypy.reporting.localization.Localization(__file__, 22, 15), func_470333, *[obj_470334], **kwargs_470338)
    
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', func_call_result_470339)
    # SSA join for if statement (line 21)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Try-finally block (line 23)
    
    # Call to hold(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'True' (line 24)
    True_470342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'True', False)
    # Processing the call keyword arguments (line 24)
    kwargs_470343 = {}
    # Getting the type of 'ax' (line 24)
    ax_470340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'ax', False)
    # Obtaining the member 'hold' of a type (line 24)
    hold_470341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), ax_470340, 'hold')
    # Calling hold(args, kwargs) (line 24)
    hold_call_result_470344 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), hold_470341, *[True_470342], **kwargs_470343)
    
    
    # Call to func(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'obj' (line 25)
    obj_470346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'obj', False)
    # Processing the call keyword arguments (line 25)
    # Getting the type of 'ax' (line 25)
    ax_470347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 28), 'ax', False)
    keyword_470348 = ax_470347
    # Getting the type of 'kw' (line 25)
    kw_470349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 34), 'kw', False)
    kwargs_470350 = {'ax': keyword_470348, 'kw_470349': kw_470349}
    # Getting the type of 'func' (line 25)
    func_470345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'func', False)
    # Calling func(args, kwargs) (line 25)
    func_call_result_470351 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), func_470345, *[obj_470346], **kwargs_470350)
    
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', func_call_result_470351)
    
    # finally branch of the try-finally block (line 23)
    
    # Call to hold(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'was_held' (line 27)
    was_held_470354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'was_held', False)
    # Processing the call keyword arguments (line 27)
    kwargs_470355 = {}
    # Getting the type of 'ax' (line 27)
    ax_470352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'ax', False)
    # Obtaining the member 'hold' of a type (line 27)
    hold_470353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), ax_470352, 'hold')
    # Calling hold(args, kwargs) (line 27)
    hold_call_result_470356 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), hold_470353, *[was_held_470354], **kwargs_470355)
    
    
    
    # ################# End of '_held_figure(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_held_figure' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_470357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_470357)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_held_figure'
    return stypy_return_type_470357

# Assigning a type to the variable '_held_figure' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '_held_figure', _held_figure)

@norecursion
def _adjust_bounds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_adjust_bounds'
    module_type_store = module_type_store.open_function_context('_adjust_bounds', 30, 0, False)
    
    # Passed parameters checking function
    _adjust_bounds.stypy_localization = localization
    _adjust_bounds.stypy_type_of_self = None
    _adjust_bounds.stypy_type_store = module_type_store
    _adjust_bounds.stypy_function_name = '_adjust_bounds'
    _adjust_bounds.stypy_param_names_list = ['ax', 'points']
    _adjust_bounds.stypy_varargs_param_name = None
    _adjust_bounds.stypy_kwargs_param_name = None
    _adjust_bounds.stypy_call_defaults = defaults
    _adjust_bounds.stypy_call_varargs = varargs
    _adjust_bounds.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_adjust_bounds', ['ax', 'points'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_adjust_bounds', localization, ['ax', 'points'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_adjust_bounds(...)' code ##################

    
    # Assigning a BinOp to a Name (line 31):
    
    # Assigning a BinOp to a Name (line 31):
    float_470358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 13), 'float')
    
    # Call to ptp(...): (line 31)
    # Processing the call keyword arguments (line 31)
    int_470361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 35), 'int')
    keyword_470362 = int_470361
    kwargs_470363 = {'axis': keyword_470362}
    # Getting the type of 'points' (line 31)
    points_470359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'points', False)
    # Obtaining the member 'ptp' of a type (line 31)
    ptp_470360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 19), points_470359, 'ptp')
    # Calling ptp(args, kwargs) (line 31)
    ptp_call_result_470364 = invoke(stypy.reporting.localization.Localization(__file__, 31, 19), ptp_470360, *[], **kwargs_470363)
    
    # Applying the binary operator '*' (line 31)
    result_mul_470365 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 13), '*', float_470358, ptp_call_result_470364)
    
    # Assigning a type to the variable 'margin' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'margin', result_mul_470365)
    
    # Assigning a BinOp to a Name (line 32):
    
    # Assigning a BinOp to a Name (line 32):
    
    # Call to min(...): (line 32)
    # Processing the call keyword arguments (line 32)
    int_470368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 29), 'int')
    keyword_470369 = int_470368
    kwargs_470370 = {'axis': keyword_470369}
    # Getting the type of 'points' (line 32)
    points_470366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 13), 'points', False)
    # Obtaining the member 'min' of a type (line 32)
    min_470367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 13), points_470366, 'min')
    # Calling min(args, kwargs) (line 32)
    min_call_result_470371 = invoke(stypy.reporting.localization.Localization(__file__, 32, 13), min_470367, *[], **kwargs_470370)
    
    # Getting the type of 'margin' (line 32)
    margin_470372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 34), 'margin')
    # Applying the binary operator '-' (line 32)
    result_sub_470373 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 13), '-', min_call_result_470371, margin_470372)
    
    # Assigning a type to the variable 'xy_min' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'xy_min', result_sub_470373)
    
    # Assigning a BinOp to a Name (line 33):
    
    # Assigning a BinOp to a Name (line 33):
    
    # Call to max(...): (line 33)
    # Processing the call keyword arguments (line 33)
    int_470376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 29), 'int')
    keyword_470377 = int_470376
    kwargs_470378 = {'axis': keyword_470377}
    # Getting the type of 'points' (line 33)
    points_470374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'points', False)
    # Obtaining the member 'max' of a type (line 33)
    max_470375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 13), points_470374, 'max')
    # Calling max(args, kwargs) (line 33)
    max_call_result_470379 = invoke(stypy.reporting.localization.Localization(__file__, 33, 13), max_470375, *[], **kwargs_470378)
    
    # Getting the type of 'margin' (line 33)
    margin_470380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 34), 'margin')
    # Applying the binary operator '+' (line 33)
    result_add_470381 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 13), '+', max_call_result_470379, margin_470380)
    
    # Assigning a type to the variable 'xy_max' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'xy_max', result_add_470381)
    
    # Call to set_xlim(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Obtaining the type of the subscript
    int_470384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'int')
    # Getting the type of 'xy_min' (line 34)
    xy_min_470385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'xy_min', False)
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___470386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 16), xy_min_470385, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_470387 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), getitem___470386, int_470384)
    
    
    # Obtaining the type of the subscript
    int_470388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 34), 'int')
    # Getting the type of 'xy_max' (line 34)
    xy_max_470389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'xy_max', False)
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___470390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 27), xy_max_470389, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_470391 = invoke(stypy.reporting.localization.Localization(__file__, 34, 27), getitem___470390, int_470388)
    
    # Processing the call keyword arguments (line 34)
    kwargs_470392 = {}
    # Getting the type of 'ax' (line 34)
    ax_470382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'ax', False)
    # Obtaining the member 'set_xlim' of a type (line 34)
    set_xlim_470383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 4), ax_470382, 'set_xlim')
    # Calling set_xlim(args, kwargs) (line 34)
    set_xlim_call_result_470393 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), set_xlim_470383, *[subscript_call_result_470387, subscript_call_result_470391], **kwargs_470392)
    
    
    # Call to set_ylim(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Obtaining the type of the subscript
    int_470396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'int')
    # Getting the type of 'xy_min' (line 35)
    xy_min_470397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'xy_min', False)
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___470398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 16), xy_min_470397, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_470399 = invoke(stypy.reporting.localization.Localization(__file__, 35, 16), getitem___470398, int_470396)
    
    
    # Obtaining the type of the subscript
    int_470400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 34), 'int')
    # Getting the type of 'xy_max' (line 35)
    xy_max_470401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 27), 'xy_max', False)
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___470402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 27), xy_max_470401, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_470403 = invoke(stypy.reporting.localization.Localization(__file__, 35, 27), getitem___470402, int_470400)
    
    # Processing the call keyword arguments (line 35)
    kwargs_470404 = {}
    # Getting the type of 'ax' (line 35)
    ax_470394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'ax', False)
    # Obtaining the member 'set_ylim' of a type (line 35)
    set_ylim_470395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), ax_470394, 'set_ylim')
    # Calling set_ylim(args, kwargs) (line 35)
    set_ylim_call_result_470405 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), set_ylim_470395, *[subscript_call_result_470399, subscript_call_result_470403], **kwargs_470404)
    
    
    # ################# End of '_adjust_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_adjust_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_470406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_470406)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_adjust_bounds'
    return stypy_return_type_470406

# Assigning a type to the variable '_adjust_bounds' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), '_adjust_bounds', _adjust_bounds)

@norecursion
def delaunay_plot_2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 39)
    None_470407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'None')
    defaults = [None_470407]
    # Create a new context for function 'delaunay_plot_2d'
    module_type_store = module_type_store.open_function_context('delaunay_plot_2d', 38, 0, False)
    
    # Passed parameters checking function
    delaunay_plot_2d.stypy_localization = localization
    delaunay_plot_2d.stypy_type_of_self = None
    delaunay_plot_2d.stypy_type_store = module_type_store
    delaunay_plot_2d.stypy_function_name = 'delaunay_plot_2d'
    delaunay_plot_2d.stypy_param_names_list = ['tri', 'ax']
    delaunay_plot_2d.stypy_varargs_param_name = None
    delaunay_plot_2d.stypy_kwargs_param_name = None
    delaunay_plot_2d.stypy_call_defaults = defaults
    delaunay_plot_2d.stypy_call_varargs = varargs
    delaunay_plot_2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'delaunay_plot_2d', ['tri', 'ax'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'delaunay_plot_2d', localization, ['tri', 'ax'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'delaunay_plot_2d(...)' code ##################

    str_470408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', '\n    Plot the given Delaunay triangulation in 2-D\n\n    Parameters\n    ----------\n    tri : scipy.spatial.Delaunay instance\n        Triangulation to plot\n    ax : matplotlib.axes.Axes instance, optional\n        Axes to plot on\n\n    Returns\n    -------\n    fig : matplotlib.figure.Figure instance\n        Figure for the plot\n\n    See Also\n    --------\n    Delaunay\n    matplotlib.pyplot.triplot\n\n    Notes\n    -----\n    Requires Matplotlib.\n\n    ')
    
    
    
    # Obtaining the type of the subscript
    int_470409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'int')
    # Getting the type of 'tri' (line 65)
    tri_470410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 7), 'tri')
    # Obtaining the member 'points' of a type (line 65)
    points_470411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 7), tri_470410, 'points')
    # Obtaining the member 'shape' of a type (line 65)
    shape_470412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 7), points_470411, 'shape')
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___470413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 7), shape_470412, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_470414 = invoke(stypy.reporting.localization.Localization(__file__, 65, 7), getitem___470413, int_470409)
    
    int_470415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'int')
    # Applying the binary operator '!=' (line 65)
    result_ne_470416 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 7), '!=', subscript_call_result_470414, int_470415)
    
    # Testing the type of an if condition (line 65)
    if_condition_470417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 4), result_ne_470416)
    # Assigning a type to the variable 'if_condition_470417' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'if_condition_470417', if_condition_470417)
    # SSA begins for if statement (line 65)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 66)
    # Processing the call arguments (line 66)
    str_470419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'str', 'Delaunay triangulation is not 2-D')
    # Processing the call keyword arguments (line 66)
    kwargs_470420 = {}
    # Getting the type of 'ValueError' (line 66)
    ValueError_470418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 66)
    ValueError_call_result_470421 = invoke(stypy.reporting.localization.Localization(__file__, 66, 14), ValueError_470418, *[str_470419], **kwargs_470420)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 66, 8), ValueError_call_result_470421, 'raise parameter', BaseException)
    # SSA join for if statement (line 65)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 68):
    
    # Assigning a Subscript to a Name (line 68):
    
    # Obtaining the type of the subscript
    int_470422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'int')
    # Getting the type of 'tri' (line 68)
    tri_470423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'tri')
    # Obtaining the member 'points' of a type (line 68)
    points_470424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), tri_470423, 'points')
    # Obtaining the member 'T' of a type (line 68)
    T_470425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), points_470424, 'T')
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___470426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), T_470425, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_470427 = invoke(stypy.reporting.localization.Localization(__file__, 68, 4), getitem___470426, int_470422)
    
    # Assigning a type to the variable 'tuple_var_assignment_470295' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_470295', subscript_call_result_470427)
    
    # Assigning a Subscript to a Name (line 68):
    
    # Obtaining the type of the subscript
    int_470428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'int')
    # Getting the type of 'tri' (line 68)
    tri_470429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'tri')
    # Obtaining the member 'points' of a type (line 68)
    points_470430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), tri_470429, 'points')
    # Obtaining the member 'T' of a type (line 68)
    T_470431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), points_470430, 'T')
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___470432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), T_470431, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_470433 = invoke(stypy.reporting.localization.Localization(__file__, 68, 4), getitem___470432, int_470428)
    
    # Assigning a type to the variable 'tuple_var_assignment_470296' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_470296', subscript_call_result_470433)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'tuple_var_assignment_470295' (line 68)
    tuple_var_assignment_470295_470434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_470295')
    # Assigning a type to the variable 'x' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'x', tuple_var_assignment_470295_470434)
    
    # Assigning a Name to a Name (line 68):
    # Getting the type of 'tuple_var_assignment_470296' (line 68)
    tuple_var_assignment_470296_470435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tuple_var_assignment_470296')
    # Assigning a type to the variable 'y' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 7), 'y', tuple_var_assignment_470296_470435)
    
    # Call to plot(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'x' (line 69)
    x_470438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'x', False)
    # Getting the type of 'y' (line 69)
    y_470439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'y', False)
    str_470440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 18), 'str', 'o')
    # Processing the call keyword arguments (line 69)
    kwargs_470441 = {}
    # Getting the type of 'ax' (line 69)
    ax_470436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'ax', False)
    # Obtaining the member 'plot' of a type (line 69)
    plot_470437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), ax_470436, 'plot')
    # Calling plot(args, kwargs) (line 69)
    plot_call_result_470442 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), plot_470437, *[x_470438, y_470439, str_470440], **kwargs_470441)
    
    
    # Call to triplot(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'x' (line 70)
    x_470445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'x', False)
    # Getting the type of 'y' (line 70)
    y_470446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'y', False)
    
    # Call to copy(...): (line 70)
    # Processing the call keyword arguments (line 70)
    kwargs_470450 = {}
    # Getting the type of 'tri' (line 70)
    tri_470447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'tri', False)
    # Obtaining the member 'simplices' of a type (line 70)
    simplices_470448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 21), tri_470447, 'simplices')
    # Obtaining the member 'copy' of a type (line 70)
    copy_470449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 21), simplices_470448, 'copy')
    # Calling copy(args, kwargs) (line 70)
    copy_call_result_470451 = invoke(stypy.reporting.localization.Localization(__file__, 70, 21), copy_470449, *[], **kwargs_470450)
    
    # Processing the call keyword arguments (line 70)
    kwargs_470452 = {}
    # Getting the type of 'ax' (line 70)
    ax_470443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'ax', False)
    # Obtaining the member 'triplot' of a type (line 70)
    triplot_470444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 4), ax_470443, 'triplot')
    # Calling triplot(args, kwargs) (line 70)
    triplot_call_result_470453 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), triplot_470444, *[x_470445, y_470446, copy_call_result_470451], **kwargs_470452)
    
    
    # Call to _adjust_bounds(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'ax' (line 72)
    ax_470455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'ax', False)
    # Getting the type of 'tri' (line 72)
    tri_470456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'tri', False)
    # Obtaining the member 'points' of a type (line 72)
    points_470457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 23), tri_470456, 'points')
    # Processing the call keyword arguments (line 72)
    kwargs_470458 = {}
    # Getting the type of '_adjust_bounds' (line 72)
    _adjust_bounds_470454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), '_adjust_bounds', False)
    # Calling _adjust_bounds(args, kwargs) (line 72)
    _adjust_bounds_call_result_470459 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), _adjust_bounds_470454, *[ax_470455, points_470457], **kwargs_470458)
    
    # Getting the type of 'ax' (line 74)
    ax_470460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'ax')
    # Obtaining the member 'figure' of a type (line 74)
    figure_470461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 11), ax_470460, 'figure')
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type', figure_470461)
    
    # ################# End of 'delaunay_plot_2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'delaunay_plot_2d' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_470462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_470462)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'delaunay_plot_2d'
    return stypy_return_type_470462

# Assigning a type to the variable 'delaunay_plot_2d' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'delaunay_plot_2d', delaunay_plot_2d)

@norecursion
def convex_hull_plot_2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 78)
    None_470463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 33), 'None')
    defaults = [None_470463]
    # Create a new context for function 'convex_hull_plot_2d'
    module_type_store = module_type_store.open_function_context('convex_hull_plot_2d', 77, 0, False)
    
    # Passed parameters checking function
    convex_hull_plot_2d.stypy_localization = localization
    convex_hull_plot_2d.stypy_type_of_self = None
    convex_hull_plot_2d.stypy_type_store = module_type_store
    convex_hull_plot_2d.stypy_function_name = 'convex_hull_plot_2d'
    convex_hull_plot_2d.stypy_param_names_list = ['hull', 'ax']
    convex_hull_plot_2d.stypy_varargs_param_name = None
    convex_hull_plot_2d.stypy_kwargs_param_name = None
    convex_hull_plot_2d.stypy_call_defaults = defaults
    convex_hull_plot_2d.stypy_call_varargs = varargs
    convex_hull_plot_2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'convex_hull_plot_2d', ['hull', 'ax'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'convex_hull_plot_2d', localization, ['hull', 'ax'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'convex_hull_plot_2d(...)' code ##################

    str_470464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, (-1)), 'str', '\n    Plot the given convex hull diagram in 2-D\n\n    Parameters\n    ----------\n    hull : scipy.spatial.ConvexHull instance\n        Convex hull to plot\n    ax : matplotlib.axes.Axes instance, optional\n        Axes to plot on\n\n    Returns\n    -------\n    fig : matplotlib.figure.Figure instance\n        Figure for the plot\n\n    See Also\n    --------\n    ConvexHull\n\n    Notes\n    -----\n    Requires Matplotlib.\n\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 103, 4))
    
    # 'from matplotlib.collections import LineCollection' statement (line 103)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
    import_470465 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 103, 4), 'matplotlib.collections')

    if (type(import_470465) is not StypyTypeError):

        if (import_470465 != 'pyd_module'):
            __import__(import_470465)
            sys_modules_470466 = sys.modules[import_470465]
            import_from_module(stypy.reporting.localization.Localization(__file__, 103, 4), 'matplotlib.collections', sys_modules_470466.module_type_store, module_type_store, ['LineCollection'])
            nest_module(stypy.reporting.localization.Localization(__file__, 103, 4), __file__, sys_modules_470466, sys_modules_470466.module_type_store, module_type_store)
        else:
            from matplotlib.collections import LineCollection

            import_from_module(stypy.reporting.localization.Localization(__file__, 103, 4), 'matplotlib.collections', None, module_type_store, ['LineCollection'], [LineCollection])

    else:
        # Assigning a type to the variable 'matplotlib.collections' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'matplotlib.collections', import_470465)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')
    
    
    
    
    # Obtaining the type of the subscript
    int_470467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 25), 'int')
    # Getting the type of 'hull' (line 105)
    hull_470468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 7), 'hull')
    # Obtaining the member 'points' of a type (line 105)
    points_470469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 7), hull_470468, 'points')
    # Obtaining the member 'shape' of a type (line 105)
    shape_470470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 7), points_470469, 'shape')
    # Obtaining the member '__getitem__' of a type (line 105)
    getitem___470471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 7), shape_470470, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 105)
    subscript_call_result_470472 = invoke(stypy.reporting.localization.Localization(__file__, 105, 7), getitem___470471, int_470467)
    
    int_470473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 31), 'int')
    # Applying the binary operator '!=' (line 105)
    result_ne_470474 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 7), '!=', subscript_call_result_470472, int_470473)
    
    # Testing the type of an if condition (line 105)
    if_condition_470475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 4), result_ne_470474)
    # Assigning a type to the variable 'if_condition_470475' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'if_condition_470475', if_condition_470475)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 106)
    # Processing the call arguments (line 106)
    str_470477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 25), 'str', 'Convex hull is not 2-D')
    # Processing the call keyword arguments (line 106)
    kwargs_470478 = {}
    # Getting the type of 'ValueError' (line 106)
    ValueError_470476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 106)
    ValueError_call_result_470479 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), ValueError_470476, *[str_470477], **kwargs_470478)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 106, 8), ValueError_call_result_470479, 'raise parameter', BaseException)
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to plot(...): (line 108)
    # Processing the call arguments (line 108)
    
    # Obtaining the type of the subscript
    slice_470482 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 108, 12), None, None, None)
    int_470483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 26), 'int')
    # Getting the type of 'hull' (line 108)
    hull_470484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'hull', False)
    # Obtaining the member 'points' of a type (line 108)
    points_470485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), hull_470484, 'points')
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___470486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), points_470485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_470487 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), getitem___470486, (slice_470482, int_470483))
    
    
    # Obtaining the type of the subscript
    slice_470488 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 108, 30), None, None, None)
    int_470489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 44), 'int')
    # Getting the type of 'hull' (line 108)
    hull_470490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 30), 'hull', False)
    # Obtaining the member 'points' of a type (line 108)
    points_470491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 30), hull_470490, 'points')
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___470492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 30), points_470491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_470493 = invoke(stypy.reporting.localization.Localization(__file__, 108, 30), getitem___470492, (slice_470488, int_470489))
    
    str_470494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 48), 'str', 'o')
    # Processing the call keyword arguments (line 108)
    kwargs_470495 = {}
    # Getting the type of 'ax' (line 108)
    ax_470480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'ax', False)
    # Obtaining the member 'plot' of a type (line 108)
    plot_470481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 4), ax_470480, 'plot')
    # Calling plot(args, kwargs) (line 108)
    plot_call_result_470496 = invoke(stypy.reporting.localization.Localization(__file__, 108, 4), plot_470481, *[subscript_call_result_470487, subscript_call_result_470493, str_470494], **kwargs_470495)
    
    
    # Assigning a ListComp to a Name (line 109):
    
    # Assigning a ListComp to a Name (line 109):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'hull' (line 109)
    hull_470502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 57), 'hull')
    # Obtaining the member 'simplices' of a type (line 109)
    simplices_470503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 57), hull_470502, 'simplices')
    comprehension_470504 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 21), simplices_470503)
    # Assigning a type to the variable 'simplex' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'simplex', comprehension_470504)
    
    # Obtaining the type of the subscript
    # Getting the type of 'simplex' (line 109)
    simplex_470497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 33), 'simplex')
    # Getting the type of 'hull' (line 109)
    hull_470498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'hull')
    # Obtaining the member 'points' of a type (line 109)
    points_470499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 21), hull_470498, 'points')
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___470500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 21), points_470499, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_470501 = invoke(stypy.reporting.localization.Localization(__file__, 109, 21), getitem___470500, simplex_470497)
    
    list_470505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 21), list_470505, subscript_call_result_470501)
    # Assigning a type to the variable 'line_segments' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'line_segments', list_470505)
    
    # Call to add_collection(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Call to LineCollection(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'line_segments' (line 110)
    line_segments_470509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 37), 'line_segments', False)
    # Processing the call keyword arguments (line 110)
    str_470510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 44), 'str', 'k')
    keyword_470511 = str_470510
    str_470512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 47), 'str', 'solid')
    keyword_470513 = str_470512
    kwargs_470514 = {'colors': keyword_470511, 'linestyle': keyword_470513}
    # Getting the type of 'LineCollection' (line 110)
    LineCollection_470508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'LineCollection', False)
    # Calling LineCollection(args, kwargs) (line 110)
    LineCollection_call_result_470515 = invoke(stypy.reporting.localization.Localization(__file__, 110, 22), LineCollection_470508, *[line_segments_470509], **kwargs_470514)
    
    # Processing the call keyword arguments (line 110)
    kwargs_470516 = {}
    # Getting the type of 'ax' (line 110)
    ax_470506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'ax', False)
    # Obtaining the member 'add_collection' of a type (line 110)
    add_collection_470507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 4), ax_470506, 'add_collection')
    # Calling add_collection(args, kwargs) (line 110)
    add_collection_call_result_470517 = invoke(stypy.reporting.localization.Localization(__file__, 110, 4), add_collection_470507, *[LineCollection_call_result_470515], **kwargs_470516)
    
    
    # Call to _adjust_bounds(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'ax' (line 113)
    ax_470519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'ax', False)
    # Getting the type of 'hull' (line 113)
    hull_470520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'hull', False)
    # Obtaining the member 'points' of a type (line 113)
    points_470521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 23), hull_470520, 'points')
    # Processing the call keyword arguments (line 113)
    kwargs_470522 = {}
    # Getting the type of '_adjust_bounds' (line 113)
    _adjust_bounds_470518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), '_adjust_bounds', False)
    # Calling _adjust_bounds(args, kwargs) (line 113)
    _adjust_bounds_call_result_470523 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), _adjust_bounds_470518, *[ax_470519, points_470521], **kwargs_470522)
    
    # Getting the type of 'ax' (line 115)
    ax_470524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'ax')
    # Obtaining the member 'figure' of a type (line 115)
    figure_470525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 11), ax_470524, 'figure')
    # Assigning a type to the variable 'stypy_return_type' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type', figure_470525)
    
    # ################# End of 'convex_hull_plot_2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'convex_hull_plot_2d' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_470526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_470526)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'convex_hull_plot_2d'
    return stypy_return_type_470526

# Assigning a type to the variable 'convex_hull_plot_2d' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'convex_hull_plot_2d', convex_hull_plot_2d)

@norecursion
def voronoi_plot_2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 119)
    None_470527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 28), 'None')
    defaults = [None_470527]
    # Create a new context for function 'voronoi_plot_2d'
    module_type_store = module_type_store.open_function_context('voronoi_plot_2d', 118, 0, False)
    
    # Passed parameters checking function
    voronoi_plot_2d.stypy_localization = localization
    voronoi_plot_2d.stypy_type_of_self = None
    voronoi_plot_2d.stypy_type_store = module_type_store
    voronoi_plot_2d.stypy_function_name = 'voronoi_plot_2d'
    voronoi_plot_2d.stypy_param_names_list = ['vor', 'ax']
    voronoi_plot_2d.stypy_varargs_param_name = None
    voronoi_plot_2d.stypy_kwargs_param_name = 'kw'
    voronoi_plot_2d.stypy_call_defaults = defaults
    voronoi_plot_2d.stypy_call_varargs = varargs
    voronoi_plot_2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'voronoi_plot_2d', ['vor', 'ax'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'voronoi_plot_2d', localization, ['vor', 'ax'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'voronoi_plot_2d(...)' code ##################

    str_470528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, (-1)), 'str', '\n    Plot the given Voronoi diagram in 2-D\n\n    Parameters\n    ----------\n    vor : scipy.spatial.Voronoi instance\n        Diagram to plot\n    ax : matplotlib.axes.Axes instance, optional\n        Axes to plot on\n    show_points: bool, optional\n        Add the Voronoi points to the plot.\n    show_vertices : bool, optional\n        Add the Voronoi vertices to the plot.\n    line_colors : string, optional\n        Specifies the line color for polygon boundaries\n    line_width : float, optional\n        Specifies the line width for polygon boundaries\n    line_alpha: float, optional\n        Specifies the line alpha for polygon boundaries\n\n    Returns\n    -------\n    fig : matplotlib.figure.Figure instance\n        Figure for the plot\n\n    See Also\n    --------\n    Voronoi\n\n    Notes\n    -----\n    Requires Matplotlib.\n\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 154, 4))
    
    # 'from matplotlib.collections import LineCollection' statement (line 154)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
    import_470529 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 154, 4), 'matplotlib.collections')

    if (type(import_470529) is not StypyTypeError):

        if (import_470529 != 'pyd_module'):
            __import__(import_470529)
            sys_modules_470530 = sys.modules[import_470529]
            import_from_module(stypy.reporting.localization.Localization(__file__, 154, 4), 'matplotlib.collections', sys_modules_470530.module_type_store, module_type_store, ['LineCollection'])
            nest_module(stypy.reporting.localization.Localization(__file__, 154, 4), __file__, sys_modules_470530, sys_modules_470530.module_type_store, module_type_store)
        else:
            from matplotlib.collections import LineCollection

            import_from_module(stypy.reporting.localization.Localization(__file__, 154, 4), 'matplotlib.collections', None, module_type_store, ['LineCollection'], [LineCollection])

    else:
        # Assigning a type to the variable 'matplotlib.collections' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'matplotlib.collections', import_470529)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')
    
    
    
    
    # Obtaining the type of the subscript
    int_470531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 24), 'int')
    # Getting the type of 'vor' (line 156)
    vor_470532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 7), 'vor')
    # Obtaining the member 'points' of a type (line 156)
    points_470533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 7), vor_470532, 'points')
    # Obtaining the member 'shape' of a type (line 156)
    shape_470534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 7), points_470533, 'shape')
    # Obtaining the member '__getitem__' of a type (line 156)
    getitem___470535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 7), shape_470534, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 156)
    subscript_call_result_470536 = invoke(stypy.reporting.localization.Localization(__file__, 156, 7), getitem___470535, int_470531)
    
    int_470537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 30), 'int')
    # Applying the binary operator '!=' (line 156)
    result_ne_470538 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 7), '!=', subscript_call_result_470536, int_470537)
    
    # Testing the type of an if condition (line 156)
    if_condition_470539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 4), result_ne_470538)
    # Assigning a type to the variable 'if_condition_470539' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'if_condition_470539', if_condition_470539)
    # SSA begins for if statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 157)
    # Processing the call arguments (line 157)
    str_470541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 25), 'str', 'Voronoi diagram is not 2-D')
    # Processing the call keyword arguments (line 157)
    kwargs_470542 = {}
    # Getting the type of 'ValueError' (line 157)
    ValueError_470540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 157)
    ValueError_call_result_470543 = invoke(stypy.reporting.localization.Localization(__file__, 157, 14), ValueError_470540, *[str_470541], **kwargs_470542)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 157, 8), ValueError_call_result_470543, 'raise parameter', BaseException)
    # SSA join for if statement (line 156)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to get(...): (line 159)
    # Processing the call arguments (line 159)
    str_470546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 14), 'str', 'show_points')
    # Getting the type of 'True' (line 159)
    True_470547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 29), 'True', False)
    # Processing the call keyword arguments (line 159)
    kwargs_470548 = {}
    # Getting the type of 'kw' (line 159)
    kw_470544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 7), 'kw', False)
    # Obtaining the member 'get' of a type (line 159)
    get_470545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 7), kw_470544, 'get')
    # Calling get(args, kwargs) (line 159)
    get_call_result_470549 = invoke(stypy.reporting.localization.Localization(__file__, 159, 7), get_470545, *[str_470546, True_470547], **kwargs_470548)
    
    # Testing the type of an if condition (line 159)
    if_condition_470550 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 4), get_call_result_470549)
    # Assigning a type to the variable 'if_condition_470550' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'if_condition_470550', if_condition_470550)
    # SSA begins for if statement (line 159)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to plot(...): (line 160)
    # Processing the call arguments (line 160)
    
    # Obtaining the type of the subscript
    slice_470553 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 16), None, None, None)
    int_470554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 29), 'int')
    # Getting the type of 'vor' (line 160)
    vor_470555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'vor', False)
    # Obtaining the member 'points' of a type (line 160)
    points_470556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 16), vor_470555, 'points')
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___470557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 16), points_470556, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_470558 = invoke(stypy.reporting.localization.Localization(__file__, 160, 16), getitem___470557, (slice_470553, int_470554))
    
    
    # Obtaining the type of the subscript
    slice_470559 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 33), None, None, None)
    int_470560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 46), 'int')
    # Getting the type of 'vor' (line 160)
    vor_470561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 33), 'vor', False)
    # Obtaining the member 'points' of a type (line 160)
    points_470562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 33), vor_470561, 'points')
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___470563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 33), points_470562, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_470564 = invoke(stypy.reporting.localization.Localization(__file__, 160, 33), getitem___470563, (slice_470559, int_470560))
    
    str_470565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 50), 'str', '.')
    # Processing the call keyword arguments (line 160)
    kwargs_470566 = {}
    # Getting the type of 'ax' (line 160)
    ax_470551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'ax', False)
    # Obtaining the member 'plot' of a type (line 160)
    plot_470552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), ax_470551, 'plot')
    # Calling plot(args, kwargs) (line 160)
    plot_call_result_470567 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), plot_470552, *[subscript_call_result_470558, subscript_call_result_470564, str_470565], **kwargs_470566)
    
    # SSA join for if statement (line 159)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to get(...): (line 161)
    # Processing the call arguments (line 161)
    str_470570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 14), 'str', 'show_vertices')
    # Getting the type of 'True' (line 161)
    True_470571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 31), 'True', False)
    # Processing the call keyword arguments (line 161)
    kwargs_470572 = {}
    # Getting the type of 'kw' (line 161)
    kw_470568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 7), 'kw', False)
    # Obtaining the member 'get' of a type (line 161)
    get_470569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 7), kw_470568, 'get')
    # Calling get(args, kwargs) (line 161)
    get_call_result_470573 = invoke(stypy.reporting.localization.Localization(__file__, 161, 7), get_470569, *[str_470570, True_470571], **kwargs_470572)
    
    # Testing the type of an if condition (line 161)
    if_condition_470574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 4), get_call_result_470573)
    # Assigning a type to the variable 'if_condition_470574' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'if_condition_470574', if_condition_470574)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to plot(...): (line 162)
    # Processing the call arguments (line 162)
    
    # Obtaining the type of the subscript
    slice_470577 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 162, 16), None, None, None)
    int_470578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 31), 'int')
    # Getting the type of 'vor' (line 162)
    vor_470579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'vor', False)
    # Obtaining the member 'vertices' of a type (line 162)
    vertices_470580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 16), vor_470579, 'vertices')
    # Obtaining the member '__getitem__' of a type (line 162)
    getitem___470581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 16), vertices_470580, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 162)
    subscript_call_result_470582 = invoke(stypy.reporting.localization.Localization(__file__, 162, 16), getitem___470581, (slice_470577, int_470578))
    
    
    # Obtaining the type of the subscript
    slice_470583 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 162, 35), None, None, None)
    int_470584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 50), 'int')
    # Getting the type of 'vor' (line 162)
    vor_470585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 35), 'vor', False)
    # Obtaining the member 'vertices' of a type (line 162)
    vertices_470586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 35), vor_470585, 'vertices')
    # Obtaining the member '__getitem__' of a type (line 162)
    getitem___470587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 35), vertices_470586, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 162)
    subscript_call_result_470588 = invoke(stypy.reporting.localization.Localization(__file__, 162, 35), getitem___470587, (slice_470583, int_470584))
    
    str_470589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 54), 'str', 'o')
    # Processing the call keyword arguments (line 162)
    kwargs_470590 = {}
    # Getting the type of 'ax' (line 162)
    ax_470575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'ax', False)
    # Obtaining the member 'plot' of a type (line 162)
    plot_470576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), ax_470575, 'plot')
    # Calling plot(args, kwargs) (line 162)
    plot_call_result_470591 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), plot_470576, *[subscript_call_result_470582, subscript_call_result_470588, str_470589], **kwargs_470590)
    
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 164):
    
    # Assigning a Call to a Name (line 164):
    
    # Call to get(...): (line 164)
    # Processing the call arguments (line 164)
    str_470594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 25), 'str', 'line_colors')
    str_470595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 40), 'str', 'k')
    # Processing the call keyword arguments (line 164)
    kwargs_470596 = {}
    # Getting the type of 'kw' (line 164)
    kw_470592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 18), 'kw', False)
    # Obtaining the member 'get' of a type (line 164)
    get_470593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 18), kw_470592, 'get')
    # Calling get(args, kwargs) (line 164)
    get_call_result_470597 = invoke(stypy.reporting.localization.Localization(__file__, 164, 18), get_470593, *[str_470594, str_470595], **kwargs_470596)
    
    # Assigning a type to the variable 'line_colors' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'line_colors', get_call_result_470597)
    
    # Assigning a Call to a Name (line 165):
    
    # Assigning a Call to a Name (line 165):
    
    # Call to get(...): (line 165)
    # Processing the call arguments (line 165)
    str_470600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 24), 'str', 'line_width')
    float_470601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 38), 'float')
    # Processing the call keyword arguments (line 165)
    kwargs_470602 = {}
    # Getting the type of 'kw' (line 165)
    kw_470598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 17), 'kw', False)
    # Obtaining the member 'get' of a type (line 165)
    get_470599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 17), kw_470598, 'get')
    # Calling get(args, kwargs) (line 165)
    get_call_result_470603 = invoke(stypy.reporting.localization.Localization(__file__, 165, 17), get_470599, *[str_470600, float_470601], **kwargs_470602)
    
    # Assigning a type to the variable 'line_width' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'line_width', get_call_result_470603)
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to get(...): (line 166)
    # Processing the call arguments (line 166)
    str_470606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 24), 'str', 'line_alpha')
    float_470607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 38), 'float')
    # Processing the call keyword arguments (line 166)
    kwargs_470608 = {}
    # Getting the type of 'kw' (line 166)
    kw_470604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 17), 'kw', False)
    # Obtaining the member 'get' of a type (line 166)
    get_470605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 17), kw_470604, 'get')
    # Calling get(args, kwargs) (line 166)
    get_call_result_470609 = invoke(stypy.reporting.localization.Localization(__file__, 166, 17), get_470605, *[str_470606, float_470607], **kwargs_470608)
    
    # Assigning a type to the variable 'line_alpha' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'line_alpha', get_call_result_470609)
    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to mean(...): (line 168)
    # Processing the call keyword arguments (line 168)
    int_470613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 34), 'int')
    keyword_470614 = int_470613
    kwargs_470615 = {'axis': keyword_470614}
    # Getting the type of 'vor' (line 168)
    vor_470610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 13), 'vor', False)
    # Obtaining the member 'points' of a type (line 168)
    points_470611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 13), vor_470610, 'points')
    # Obtaining the member 'mean' of a type (line 168)
    mean_470612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 13), points_470611, 'mean')
    # Calling mean(args, kwargs) (line 168)
    mean_call_result_470616 = invoke(stypy.reporting.localization.Localization(__file__, 168, 13), mean_470612, *[], **kwargs_470615)
    
    # Assigning a type to the variable 'center' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'center', mean_call_result_470616)
    
    # Assigning a Call to a Name (line 169):
    
    # Assigning a Call to a Name (line 169):
    
    # Call to ptp(...): (line 169)
    # Processing the call keyword arguments (line 169)
    int_470620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 36), 'int')
    keyword_470621 = int_470620
    kwargs_470622 = {'axis': keyword_470621}
    # Getting the type of 'vor' (line 169)
    vor_470617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'vor', False)
    # Obtaining the member 'points' of a type (line 169)
    points_470618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 16), vor_470617, 'points')
    # Obtaining the member 'ptp' of a type (line 169)
    ptp_470619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 16), points_470618, 'ptp')
    # Calling ptp(args, kwargs) (line 169)
    ptp_call_result_470623 = invoke(stypy.reporting.localization.Localization(__file__, 169, 16), ptp_470619, *[], **kwargs_470622)
    
    # Assigning a type to the variable 'ptp_bound' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'ptp_bound', ptp_call_result_470623)
    
    # Assigning a List to a Name (line 171):
    
    # Assigning a List to a Name (line 171):
    
    # Obtaining an instance of the builtin type 'list' (line 171)
    list_470624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 171)
    
    # Assigning a type to the variable 'finite_segments' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'finite_segments', list_470624)
    
    # Assigning a List to a Name (line 172):
    
    # Assigning a List to a Name (line 172):
    
    # Obtaining an instance of the builtin type 'list' (line 172)
    list_470625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 172)
    
    # Assigning a type to the variable 'infinite_segments' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'infinite_segments', list_470625)
    
    
    # Call to zip(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'vor' (line 173)
    vor_470627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 33), 'vor', False)
    # Obtaining the member 'ridge_points' of a type (line 173)
    ridge_points_470628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 33), vor_470627, 'ridge_points')
    # Getting the type of 'vor' (line 173)
    vor_470629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 51), 'vor', False)
    # Obtaining the member 'ridge_vertices' of a type (line 173)
    ridge_vertices_470630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 51), vor_470629, 'ridge_vertices')
    # Processing the call keyword arguments (line 173)
    kwargs_470631 = {}
    # Getting the type of 'zip' (line 173)
    zip_470626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 29), 'zip', False)
    # Calling zip(args, kwargs) (line 173)
    zip_call_result_470632 = invoke(stypy.reporting.localization.Localization(__file__, 173, 29), zip_470626, *[ridge_points_470628, ridge_vertices_470630], **kwargs_470631)
    
    # Testing the type of a for loop iterable (line 173)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 173, 4), zip_call_result_470632)
    # Getting the type of the for loop variable (line 173)
    for_loop_var_470633 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 173, 4), zip_call_result_470632)
    # Assigning a type to the variable 'pointidx' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'pointidx', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 4), for_loop_var_470633))
    # Assigning a type to the variable 'simplex' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'simplex', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 4), for_loop_var_470633))
    # SSA begins for a for statement (line 173)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 174):
    
    # Assigning a Call to a Name (line 174):
    
    # Call to asarray(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'simplex' (line 174)
    simplex_470636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 29), 'simplex', False)
    # Processing the call keyword arguments (line 174)
    kwargs_470637 = {}
    # Getting the type of 'np' (line 174)
    np_470634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'np', False)
    # Obtaining the member 'asarray' of a type (line 174)
    asarray_470635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 18), np_470634, 'asarray')
    # Calling asarray(args, kwargs) (line 174)
    asarray_call_result_470638 = invoke(stypy.reporting.localization.Localization(__file__, 174, 18), asarray_470635, *[simplex_470636], **kwargs_470637)
    
    # Assigning a type to the variable 'simplex' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'simplex', asarray_call_result_470638)
    
    
    # Call to all(...): (line 175)
    # Processing the call arguments (line 175)
    
    # Getting the type of 'simplex' (line 175)
    simplex_470641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 18), 'simplex', False)
    int_470642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 29), 'int')
    # Applying the binary operator '>=' (line 175)
    result_ge_470643 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 18), '>=', simplex_470641, int_470642)
    
    # Processing the call keyword arguments (line 175)
    kwargs_470644 = {}
    # Getting the type of 'np' (line 175)
    np_470639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 11), 'np', False)
    # Obtaining the member 'all' of a type (line 175)
    all_470640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 11), np_470639, 'all')
    # Calling all(args, kwargs) (line 175)
    all_call_result_470645 = invoke(stypy.reporting.localization.Localization(__file__, 175, 11), all_470640, *[result_ge_470643], **kwargs_470644)
    
    # Testing the type of an if condition (line 175)
    if_condition_470646 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 8), all_call_result_470645)
    # Assigning a type to the variable 'if_condition_470646' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'if_condition_470646', if_condition_470646)
    # SSA begins for if statement (line 175)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 176)
    # Processing the call arguments (line 176)
    
    # Obtaining the type of the subscript
    # Getting the type of 'simplex' (line 176)
    simplex_470649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 48), 'simplex', False)
    # Getting the type of 'vor' (line 176)
    vor_470650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 35), 'vor', False)
    # Obtaining the member 'vertices' of a type (line 176)
    vertices_470651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 35), vor_470650, 'vertices')
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___470652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 35), vertices_470651, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_470653 = invoke(stypy.reporting.localization.Localization(__file__, 176, 35), getitem___470652, simplex_470649)
    
    # Processing the call keyword arguments (line 176)
    kwargs_470654 = {}
    # Getting the type of 'finite_segments' (line 176)
    finite_segments_470647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'finite_segments', False)
    # Obtaining the member 'append' of a type (line 176)
    append_470648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), finite_segments_470647, 'append')
    # Calling append(args, kwargs) (line 176)
    append_call_result_470655 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), append_470648, *[subscript_call_result_470653], **kwargs_470654)
    
    # SSA branch for the else part of an if statement (line 175)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 178):
    
    # Assigning a Subscript to a Name (line 178):
    
    # Obtaining the type of the subscript
    int_470656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 38), 'int')
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'simplex' (line 178)
    simplex_470657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'simplex')
    int_470658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 35), 'int')
    # Applying the binary operator '>=' (line 178)
    result_ge_470659 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 24), '>=', simplex_470657, int_470658)
    
    # Getting the type of 'simplex' (line 178)
    simplex_470660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'simplex')
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___470661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 16), simplex_470660, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_470662 = invoke(stypy.reporting.localization.Localization(__file__, 178, 16), getitem___470661, result_ge_470659)
    
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___470663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 16), subscript_call_result_470662, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_470664 = invoke(stypy.reporting.localization.Localization(__file__, 178, 16), getitem___470663, int_470656)
    
    # Assigning a type to the variable 'i' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'i', subscript_call_result_470664)
    
    # Assigning a BinOp to a Name (line 180):
    
    # Assigning a BinOp to a Name (line 180):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_470665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 36), 'int')
    # Getting the type of 'pointidx' (line 180)
    pointidx_470666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 27), 'pointidx')
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___470667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 27), pointidx_470666, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_470668 = invoke(stypy.reporting.localization.Localization(__file__, 180, 27), getitem___470667, int_470665)
    
    # Getting the type of 'vor' (line 180)
    vor_470669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'vor')
    # Obtaining the member 'points' of a type (line 180)
    points_470670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 16), vor_470669, 'points')
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___470671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 16), points_470670, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_470672 = invoke(stypy.reporting.localization.Localization(__file__, 180, 16), getitem___470671, subscript_call_result_470668)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_470673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 62), 'int')
    # Getting the type of 'pointidx' (line 180)
    pointidx_470674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 53), 'pointidx')
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___470675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 53), pointidx_470674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_470676 = invoke(stypy.reporting.localization.Localization(__file__, 180, 53), getitem___470675, int_470673)
    
    # Getting the type of 'vor' (line 180)
    vor_470677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 42), 'vor')
    # Obtaining the member 'points' of a type (line 180)
    points_470678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 42), vor_470677, 'points')
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___470679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 42), points_470678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_470680 = invoke(stypy.reporting.localization.Localization(__file__, 180, 42), getitem___470679, subscript_call_result_470676)
    
    # Applying the binary operator '-' (line 180)
    result_sub_470681 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 16), '-', subscript_call_result_470672, subscript_call_result_470680)
    
    # Assigning a type to the variable 't' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 't', result_sub_470681)
    
    # Getting the type of 't' (line 181)
    t_470682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 't')
    
    # Call to norm(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 't' (line 181)
    t_470686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 32), 't', False)
    # Processing the call keyword arguments (line 181)
    kwargs_470687 = {}
    # Getting the type of 'np' (line 181)
    np_470683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 17), 'np', False)
    # Obtaining the member 'linalg' of a type (line 181)
    linalg_470684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 17), np_470683, 'linalg')
    # Obtaining the member 'norm' of a type (line 181)
    norm_470685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 17), linalg_470684, 'norm')
    # Calling norm(args, kwargs) (line 181)
    norm_call_result_470688 = invoke(stypy.reporting.localization.Localization(__file__, 181, 17), norm_470685, *[t_470686], **kwargs_470687)
    
    # Applying the binary operator 'div=' (line 181)
    result_div_470689 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 12), 'div=', t_470682, norm_call_result_470688)
    # Assigning a type to the variable 't' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 't', result_div_470689)
    
    
    # Assigning a Call to a Name (line 182):
    
    # Assigning a Call to a Name (line 182):
    
    # Call to array(...): (line 182)
    # Processing the call arguments (line 182)
    
    # Obtaining an instance of the builtin type 'list' (line 182)
    list_470692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 182)
    # Adding element type (line 182)
    
    
    # Obtaining the type of the subscript
    int_470693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 29), 'int')
    # Getting the type of 't' (line 182)
    t_470694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 27), 't', False)
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___470695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 27), t_470694, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_470696 = invoke(stypy.reporting.localization.Localization(__file__, 182, 27), getitem___470695, int_470693)
    
    # Applying the 'usub' unary operator (line 182)
    result___neg___470697 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 26), 'usub', subscript_call_result_470696)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 25), list_470692, result___neg___470697)
    # Adding element type (line 182)
    
    # Obtaining the type of the subscript
    int_470698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 35), 'int')
    # Getting the type of 't' (line 182)
    t_470699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 33), 't', False)
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___470700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 33), t_470699, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_470701 = invoke(stypy.reporting.localization.Localization(__file__, 182, 33), getitem___470700, int_470698)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 25), list_470692, subscript_call_result_470701)
    
    # Processing the call keyword arguments (line 182)
    kwargs_470702 = {}
    # Getting the type of 'np' (line 182)
    np_470690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 182)
    array_470691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 16), np_470690, 'array')
    # Calling array(args, kwargs) (line 182)
    array_call_result_470703 = invoke(stypy.reporting.localization.Localization(__file__, 182, 16), array_470691, *[list_470692], **kwargs_470702)
    
    # Assigning a type to the variable 'n' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'n', array_call_result_470703)
    
    # Assigning a Call to a Name (line 184):
    
    # Assigning a Call to a Name (line 184):
    
    # Call to mean(...): (line 184)
    # Processing the call keyword arguments (line 184)
    int_470710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 54), 'int')
    keyword_470711 = int_470710
    kwargs_470712 = {'axis': keyword_470711}
    
    # Obtaining the type of the subscript
    # Getting the type of 'pointidx' (line 184)
    pointidx_470704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'pointidx', False)
    # Getting the type of 'vor' (line 184)
    vor_470705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 23), 'vor', False)
    # Obtaining the member 'points' of a type (line 184)
    points_470706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 23), vor_470705, 'points')
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___470707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 23), points_470706, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_470708 = invoke(stypy.reporting.localization.Localization(__file__, 184, 23), getitem___470707, pointidx_470704)
    
    # Obtaining the member 'mean' of a type (line 184)
    mean_470709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 23), subscript_call_result_470708, 'mean')
    # Calling mean(args, kwargs) (line 184)
    mean_call_result_470713 = invoke(stypy.reporting.localization.Localization(__file__, 184, 23), mean_470709, *[], **kwargs_470712)
    
    # Assigning a type to the variable 'midpoint' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'midpoint', mean_call_result_470713)
    
    # Assigning a BinOp to a Name (line 185):
    
    # Assigning a BinOp to a Name (line 185):
    
    # Call to sign(...): (line 185)
    # Processing the call arguments (line 185)
    
    # Call to dot(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'midpoint' (line 185)
    midpoint_470718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 39), 'midpoint', False)
    # Getting the type of 'center' (line 185)
    center_470719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 50), 'center', False)
    # Applying the binary operator '-' (line 185)
    result_sub_470720 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 39), '-', midpoint_470718, center_470719)
    
    # Getting the type of 'n' (line 185)
    n_470721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 58), 'n', False)
    # Processing the call keyword arguments (line 185)
    kwargs_470722 = {}
    # Getting the type of 'np' (line 185)
    np_470716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 32), 'np', False)
    # Obtaining the member 'dot' of a type (line 185)
    dot_470717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 32), np_470716, 'dot')
    # Calling dot(args, kwargs) (line 185)
    dot_call_result_470723 = invoke(stypy.reporting.localization.Localization(__file__, 185, 32), dot_470717, *[result_sub_470720, n_470721], **kwargs_470722)
    
    # Processing the call keyword arguments (line 185)
    kwargs_470724 = {}
    # Getting the type of 'np' (line 185)
    np_470714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'np', False)
    # Obtaining the member 'sign' of a type (line 185)
    sign_470715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 24), np_470714, 'sign')
    # Calling sign(args, kwargs) (line 185)
    sign_call_result_470725 = invoke(stypy.reporting.localization.Localization(__file__, 185, 24), sign_470715, *[dot_call_result_470723], **kwargs_470724)
    
    # Getting the type of 'n' (line 185)
    n_470726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 64), 'n')
    # Applying the binary operator '*' (line 185)
    result_mul_470727 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 24), '*', sign_call_result_470725, n_470726)
    
    # Assigning a type to the variable 'direction' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'direction', result_mul_470727)
    
    # Assigning a BinOp to a Name (line 186):
    
    # Assigning a BinOp to a Name (line 186):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 186)
    i_470728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 37), 'i')
    # Getting the type of 'vor' (line 186)
    vor_470729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'vor')
    # Obtaining the member 'vertices' of a type (line 186)
    vertices_470730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 24), vor_470729, 'vertices')
    # Obtaining the member '__getitem__' of a type (line 186)
    getitem___470731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 24), vertices_470730, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
    subscript_call_result_470732 = invoke(stypy.reporting.localization.Localization(__file__, 186, 24), getitem___470731, i_470728)
    
    # Getting the type of 'direction' (line 186)
    direction_470733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 42), 'direction')
    
    # Call to max(...): (line 186)
    # Processing the call keyword arguments (line 186)
    kwargs_470736 = {}
    # Getting the type of 'ptp_bound' (line 186)
    ptp_bound_470734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 54), 'ptp_bound', False)
    # Obtaining the member 'max' of a type (line 186)
    max_470735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 54), ptp_bound_470734, 'max')
    # Calling max(args, kwargs) (line 186)
    max_call_result_470737 = invoke(stypy.reporting.localization.Localization(__file__, 186, 54), max_470735, *[], **kwargs_470736)
    
    # Applying the binary operator '*' (line 186)
    result_mul_470738 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 42), '*', direction_470733, max_call_result_470737)
    
    # Applying the binary operator '+' (line 186)
    result_add_470739 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 24), '+', subscript_call_result_470732, result_mul_470738)
    
    # Assigning a type to the variable 'far_point' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'far_point', result_add_470739)
    
    # Call to append(...): (line 188)
    # Processing the call arguments (line 188)
    
    # Obtaining an instance of the builtin type 'list' (line 188)
    list_470742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 188)
    # Adding element type (line 188)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 188)
    i_470743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 51), 'i', False)
    # Getting the type of 'vor' (line 188)
    vor_470744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 38), 'vor', False)
    # Obtaining the member 'vertices' of a type (line 188)
    vertices_470745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 38), vor_470744, 'vertices')
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___470746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 38), vertices_470745, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_470747 = invoke(stypy.reporting.localization.Localization(__file__, 188, 38), getitem___470746, i_470743)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 37), list_470742, subscript_call_result_470747)
    # Adding element type (line 188)
    # Getting the type of 'far_point' (line 188)
    far_point_470748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 55), 'far_point', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 37), list_470742, far_point_470748)
    
    # Processing the call keyword arguments (line 188)
    kwargs_470749 = {}
    # Getting the type of 'infinite_segments' (line 188)
    infinite_segments_470740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'infinite_segments', False)
    # Obtaining the member 'append' of a type (line 188)
    append_470741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), infinite_segments_470740, 'append')
    # Calling append(args, kwargs) (line 188)
    append_call_result_470750 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), append_470741, *[list_470742], **kwargs_470749)
    
    # SSA join for if statement (line 175)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to add_collection(...): (line 190)
    # Processing the call arguments (line 190)
    
    # Call to LineCollection(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'finite_segments' (line 190)
    finite_segments_470754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 37), 'finite_segments', False)
    # Processing the call keyword arguments (line 190)
    # Getting the type of 'line_colors' (line 191)
    line_colors_470755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 44), 'line_colors', False)
    keyword_470756 = line_colors_470755
    # Getting the type of 'line_width' (line 192)
    line_width_470757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 40), 'line_width', False)
    keyword_470758 = line_width_470757
    # Getting the type of 'line_alpha' (line 193)
    line_alpha_470759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 43), 'line_alpha', False)
    keyword_470760 = line_alpha_470759
    str_470761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 47), 'str', 'solid')
    keyword_470762 = str_470761
    kwargs_470763 = {'lw': keyword_470758, 'colors': keyword_470756, 'linestyle': keyword_470762, 'alpha': keyword_470760}
    # Getting the type of 'LineCollection' (line 190)
    LineCollection_470753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 22), 'LineCollection', False)
    # Calling LineCollection(args, kwargs) (line 190)
    LineCollection_call_result_470764 = invoke(stypy.reporting.localization.Localization(__file__, 190, 22), LineCollection_470753, *[finite_segments_470754], **kwargs_470763)
    
    # Processing the call keyword arguments (line 190)
    kwargs_470765 = {}
    # Getting the type of 'ax' (line 190)
    ax_470751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'ax', False)
    # Obtaining the member 'add_collection' of a type (line 190)
    add_collection_470752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 4), ax_470751, 'add_collection')
    # Calling add_collection(args, kwargs) (line 190)
    add_collection_call_result_470766 = invoke(stypy.reporting.localization.Localization(__file__, 190, 4), add_collection_470752, *[LineCollection_call_result_470764], **kwargs_470765)
    
    
    # Call to add_collection(...): (line 195)
    # Processing the call arguments (line 195)
    
    # Call to LineCollection(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'infinite_segments' (line 195)
    infinite_segments_470770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 37), 'infinite_segments', False)
    # Processing the call keyword arguments (line 195)
    # Getting the type of 'line_colors' (line 196)
    line_colors_470771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 44), 'line_colors', False)
    keyword_470772 = line_colors_470771
    # Getting the type of 'line_width' (line 197)
    line_width_470773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 40), 'line_width', False)
    keyword_470774 = line_width_470773
    # Getting the type of 'line_alpha' (line 198)
    line_alpha_470775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 43), 'line_alpha', False)
    keyword_470776 = line_alpha_470775
    str_470777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 47), 'str', 'dashed')
    keyword_470778 = str_470777
    kwargs_470779 = {'lw': keyword_470774, 'colors': keyword_470772, 'linestyle': keyword_470778, 'alpha': keyword_470776}
    # Getting the type of 'LineCollection' (line 195)
    LineCollection_470769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'LineCollection', False)
    # Calling LineCollection(args, kwargs) (line 195)
    LineCollection_call_result_470780 = invoke(stypy.reporting.localization.Localization(__file__, 195, 22), LineCollection_470769, *[infinite_segments_470770], **kwargs_470779)
    
    # Processing the call keyword arguments (line 195)
    kwargs_470781 = {}
    # Getting the type of 'ax' (line 195)
    ax_470767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'ax', False)
    # Obtaining the member 'add_collection' of a type (line 195)
    add_collection_470768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 4), ax_470767, 'add_collection')
    # Calling add_collection(args, kwargs) (line 195)
    add_collection_call_result_470782 = invoke(stypy.reporting.localization.Localization(__file__, 195, 4), add_collection_470768, *[LineCollection_call_result_470780], **kwargs_470781)
    
    
    # Call to _adjust_bounds(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'ax' (line 201)
    ax_470784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'ax', False)
    # Getting the type of 'vor' (line 201)
    vor_470785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), 'vor', False)
    # Obtaining the member 'points' of a type (line 201)
    points_470786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 23), vor_470785, 'points')
    # Processing the call keyword arguments (line 201)
    kwargs_470787 = {}
    # Getting the type of '_adjust_bounds' (line 201)
    _adjust_bounds_470783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), '_adjust_bounds', False)
    # Calling _adjust_bounds(args, kwargs) (line 201)
    _adjust_bounds_call_result_470788 = invoke(stypy.reporting.localization.Localization(__file__, 201, 4), _adjust_bounds_470783, *[ax_470784, points_470786], **kwargs_470787)
    
    # Getting the type of 'ax' (line 203)
    ax_470789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'ax')
    # Obtaining the member 'figure' of a type (line 203)
    figure_470790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 11), ax_470789, 'figure')
    # Assigning a type to the variable 'stypy_return_type' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stypy_return_type', figure_470790)
    
    # ################# End of 'voronoi_plot_2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'voronoi_plot_2d' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_470791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_470791)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'voronoi_plot_2d'
    return stypy_return_type_470791

# Assigning a type to the variable 'voronoi_plot_2d' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'voronoi_plot_2d', voronoi_plot_2d)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
