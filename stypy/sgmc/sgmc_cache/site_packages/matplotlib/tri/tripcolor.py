
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: from matplotlib.collections import PolyCollection, TriMesh
7: from matplotlib.colors import Normalize
8: from matplotlib.tri.triangulation import Triangulation
9: import numpy as np
10: 
11: 
12: def tripcolor(ax, *args, **kwargs):
13:     '''
14:     Create a pseudocolor plot of an unstructured triangular grid.
15: 
16:     The triangulation can be specified in one of two ways; either::
17: 
18:       tripcolor(triangulation, ...)
19: 
20:     where triangulation is a :class:`matplotlib.tri.Triangulation`
21:     object, or
22: 
23:     ::
24: 
25:       tripcolor(x, y, ...)
26:       tripcolor(x, y, triangles, ...)
27:       tripcolor(x, y, triangles=triangles, ...)
28:       tripcolor(x, y, mask=mask, ...)
29:       tripcolor(x, y, triangles, mask=mask, ...)
30: 
31:     in which case a Triangulation object will be created.  See
32:     :class:`~matplotlib.tri.Triangulation` for a explanation of these
33:     possibilities.
34: 
35:     The next argument must be *C*, the array of color values, either
36:     one per point in the triangulation if color values are defined at
37:     points, or one per triangle in the triangulation if color values
38:     are defined at triangles. If there are the same number of points
39:     and triangles in the triangulation it is assumed that color
40:     values are defined at points; to force the use of color values at
41:     triangles use the kwarg ``facecolors=C`` instead of just ``C``.
42: 
43:     *shading* may be 'flat' (the default) or 'gouraud'. If *shading*
44:     is 'flat' and C values are defined at points, the color values
45:     used for each triangle are from the mean C of the triangle's
46:     three points. If *shading* is 'gouraud' then color values must be
47:     defined at points.
48: 
49:     The remaining kwargs are the same as for
50:     :meth:`~matplotlib.axes.Axes.pcolor`.
51:     '''
52:     if not ax._hold:
53:         ax.cla()
54: 
55:     alpha = kwargs.pop('alpha', 1.0)
56:     norm = kwargs.pop('norm', None)
57:     cmap = kwargs.pop('cmap', None)
58:     vmin = kwargs.pop('vmin', None)
59:     vmax = kwargs.pop('vmax', None)
60:     shading = kwargs.pop('shading', 'flat')
61:     facecolors = kwargs.pop('facecolors', None)
62: 
63:     if shading not in ['flat', 'gouraud']:
64:         raise ValueError("shading must be one of ['flat', 'gouraud'] "
65:                          "not {0}".format(shading))
66: 
67:     tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
68: 
69:     # C is the colors array defined at either points or faces (i.e. triangles).
70:     # If facecolors is None, C are defined at points.
71:     # If facecolors is not None, C are defined at faces.
72:     if facecolors is not None:
73:         C = facecolors
74:     else:
75:         C = np.asarray(args[0])
76: 
77:     # If there are a different number of points and triangles in the
78:     # triangulation, can omit facecolors kwarg as it is obvious from
79:     # length of C whether it refers to points or faces.
80:     # Do not do this for gouraud shading.
81:     if (facecolors is None and len(C) == len(tri.triangles) and
82:             len(C) != len(tri.x) and shading != 'gouraud'):
83:         facecolors = C
84: 
85:     # Check length of C is OK.
86:     if ((facecolors is None and len(C) != len(tri.x)) or
87:             (facecolors is not None and len(C) != len(tri.triangles))):
88:         raise ValueError('Length of color values array must be the same '
89:                          'as either the number of triangulation points '
90:                          'or triangles')
91: 
92:     # Handling of linewidths, shading, edgecolors and antialiased as
93:     # in Axes.pcolor
94:     linewidths = (0.25,)
95:     if 'linewidth' in kwargs:
96:         kwargs['linewidths'] = kwargs.pop('linewidth')
97:     kwargs.setdefault('linewidths', linewidths)
98: 
99:     edgecolors = 'none'
100:     if 'edgecolor' in kwargs:
101:         kwargs['edgecolors'] = kwargs.pop('edgecolor')
102:     ec = kwargs.setdefault('edgecolors', edgecolors)
103: 
104:     if 'antialiased' in kwargs:
105:         kwargs['antialiaseds'] = kwargs.pop('antialiased')
106:     if 'antialiaseds' not in kwargs and ec.lower() == "none":
107:         kwargs['antialiaseds'] = False
108: 
109:     if shading == 'gouraud':
110:         if facecolors is not None:
111:             raise ValueError('Gouraud shading does not support the use '
112:                              'of facecolors kwarg')
113:         if len(C) != len(tri.x):
114:             raise ValueError('For gouraud shading, the length of color '
115:                              'values array must be the same as the '
116:                              'number of triangulation points')
117:         collection = TriMesh(tri, **kwargs)
118:     else:
119:         # Vertices of triangles.
120:         maskedTris = tri.get_masked_triangles()
121:         verts = np.concatenate((tri.x[maskedTris][..., np.newaxis],
122:                                 tri.y[maskedTris][..., np.newaxis]), axis=2)
123: 
124:         # Color values.
125:         if facecolors is None:
126:             # One color per triangle, the mean of the 3 vertex color values.
127:             C = C[maskedTris].mean(axis=1)
128:         elif tri.mask is not None:
129:             # Remove color values of masked triangles.
130:             C = C.compress(1-tri.mask)
131: 
132:         collection = PolyCollection(verts, **kwargs)
133: 
134:     collection.set_alpha(alpha)
135:     collection.set_array(C)
136:     if norm is not None:
137:         if not isinstance(norm, Normalize):
138:             msg = "'norm' must be an instance of 'Normalize'"
139:             raise ValueError(msg)
140:     collection.set_cmap(cmap)
141:     collection.set_norm(norm)
142:     if vmin is not None or vmax is not None:
143:         collection.set_clim(vmin, vmax)
144:     else:
145:         collection.autoscale_None()
146:     ax.grid(False)
147: 
148:     minx = tri.x.min()
149:     maxx = tri.x.max()
150:     miny = tri.y.min()
151:     maxy = tri.y.max()
152:     corners = (minx, miny), (maxx, maxy)
153:     ax.update_datalim(corners)
154:     ax.autoscale_view()
155:     ax.add_collection(collection)
156:     return collection
157: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_300353 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_300353) is not StypyTypeError):

    if (import_300353 != 'pyd_module'):
        __import__(import_300353)
        sys_modules_300354 = sys.modules[import_300353]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_300354.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_300353)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from matplotlib.collections import PolyCollection, TriMesh' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_300355 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.collections')

if (type(import_300355) is not StypyTypeError):

    if (import_300355 != 'pyd_module'):
        __import__(import_300355)
        sys_modules_300356 = sys.modules[import_300355]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.collections', sys_modules_300356.module_type_store, module_type_store, ['PolyCollection', 'TriMesh'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_300356, sys_modules_300356.module_type_store, module_type_store)
    else:
        from matplotlib.collections import PolyCollection, TriMesh

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.collections', None, module_type_store, ['PolyCollection', 'TriMesh'], [PolyCollection, TriMesh])

else:
    # Assigning a type to the variable 'matplotlib.collections' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.collections', import_300355)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from matplotlib.colors import Normalize' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_300357 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.colors')

if (type(import_300357) is not StypyTypeError):

    if (import_300357 != 'pyd_module'):
        __import__(import_300357)
        sys_modules_300358 = sys.modules[import_300357]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.colors', sys_modules_300358.module_type_store, module_type_store, ['Normalize'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_300358, sys_modules_300358.module_type_store, module_type_store)
    else:
        from matplotlib.colors import Normalize

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.colors', None, module_type_store, ['Normalize'], [Normalize])

else:
    # Assigning a type to the variable 'matplotlib.colors' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.colors', import_300357)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from matplotlib.tri.triangulation import Triangulation' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_300359 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.tri.triangulation')

if (type(import_300359) is not StypyTypeError):

    if (import_300359 != 'pyd_module'):
        __import__(import_300359)
        sys_modules_300360 = sys.modules[import_300359]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.tri.triangulation', sys_modules_300360.module_type_store, module_type_store, ['Triangulation'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_300360, sys_modules_300360.module_type_store, module_type_store)
    else:
        from matplotlib.tri.triangulation import Triangulation

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.tri.triangulation', None, module_type_store, ['Triangulation'], [Triangulation])

else:
    # Assigning a type to the variable 'matplotlib.tri.triangulation' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.tri.triangulation', import_300359)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_300361 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_300361) is not StypyTypeError):

    if (import_300361 != 'pyd_module'):
        __import__(import_300361)
        sys_modules_300362 = sys.modules[import_300361]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_300362.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_300361)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')


@norecursion
def tripcolor(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tripcolor'
    module_type_store = module_type_store.open_function_context('tripcolor', 12, 0, False)
    
    # Passed parameters checking function
    tripcolor.stypy_localization = localization
    tripcolor.stypy_type_of_self = None
    tripcolor.stypy_type_store = module_type_store
    tripcolor.stypy_function_name = 'tripcolor'
    tripcolor.stypy_param_names_list = ['ax']
    tripcolor.stypy_varargs_param_name = 'args'
    tripcolor.stypy_kwargs_param_name = 'kwargs'
    tripcolor.stypy_call_defaults = defaults
    tripcolor.stypy_call_varargs = varargs
    tripcolor.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tripcolor', ['ax'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tripcolor', localization, ['ax'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tripcolor(...)' code ##################

    unicode_300363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, (-1)), 'unicode', u"\n    Create a pseudocolor plot of an unstructured triangular grid.\n\n    The triangulation can be specified in one of two ways; either::\n\n      tripcolor(triangulation, ...)\n\n    where triangulation is a :class:`matplotlib.tri.Triangulation`\n    object, or\n\n    ::\n\n      tripcolor(x, y, ...)\n      tripcolor(x, y, triangles, ...)\n      tripcolor(x, y, triangles=triangles, ...)\n      tripcolor(x, y, mask=mask, ...)\n      tripcolor(x, y, triangles, mask=mask, ...)\n\n    in which case a Triangulation object will be created.  See\n    :class:`~matplotlib.tri.Triangulation` for a explanation of these\n    possibilities.\n\n    The next argument must be *C*, the array of color values, either\n    one per point in the triangulation if color values are defined at\n    points, or one per triangle in the triangulation if color values\n    are defined at triangles. If there are the same number of points\n    and triangles in the triangulation it is assumed that color\n    values are defined at points; to force the use of color values at\n    triangles use the kwarg ``facecolors=C`` instead of just ``C``.\n\n    *shading* may be 'flat' (the default) or 'gouraud'. If *shading*\n    is 'flat' and C values are defined at points, the color values\n    used for each triangle are from the mean C of the triangle's\n    three points. If *shading* is 'gouraud' then color values must be\n    defined at points.\n\n    The remaining kwargs are the same as for\n    :meth:`~matplotlib.axes.Axes.pcolor`.\n    ")
    
    
    # Getting the type of 'ax' (line 52)
    ax_300364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'ax')
    # Obtaining the member '_hold' of a type (line 52)
    _hold_300365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 11), ax_300364, '_hold')
    # Applying the 'not' unary operator (line 52)
    result_not__300366 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 7), 'not', _hold_300365)
    
    # Testing the type of an if condition (line 52)
    if_condition_300367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 4), result_not__300366)
    # Assigning a type to the variable 'if_condition_300367' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'if_condition_300367', if_condition_300367)
    # SSA begins for if statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to cla(...): (line 53)
    # Processing the call keyword arguments (line 53)
    kwargs_300370 = {}
    # Getting the type of 'ax' (line 53)
    ax_300368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'ax', False)
    # Obtaining the member 'cla' of a type (line 53)
    cla_300369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), ax_300368, 'cla')
    # Calling cla(args, kwargs) (line 53)
    cla_call_result_300371 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), cla_300369, *[], **kwargs_300370)
    
    # SSA join for if statement (line 52)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to pop(...): (line 55)
    # Processing the call arguments (line 55)
    unicode_300374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 23), 'unicode', u'alpha')
    float_300375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 32), 'float')
    # Processing the call keyword arguments (line 55)
    kwargs_300376 = {}
    # Getting the type of 'kwargs' (line 55)
    kwargs_300372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 55)
    pop_300373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 12), kwargs_300372, 'pop')
    # Calling pop(args, kwargs) (line 55)
    pop_call_result_300377 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), pop_300373, *[unicode_300374, float_300375], **kwargs_300376)
    
    # Assigning a type to the variable 'alpha' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'alpha', pop_call_result_300377)
    
    # Assigning a Call to a Name (line 56):
    
    # Assigning a Call to a Name (line 56):
    
    # Call to pop(...): (line 56)
    # Processing the call arguments (line 56)
    unicode_300380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 22), 'unicode', u'norm')
    # Getting the type of 'None' (line 56)
    None_300381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'None', False)
    # Processing the call keyword arguments (line 56)
    kwargs_300382 = {}
    # Getting the type of 'kwargs' (line 56)
    kwargs_300378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 56)
    pop_300379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), kwargs_300378, 'pop')
    # Calling pop(args, kwargs) (line 56)
    pop_call_result_300383 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), pop_300379, *[unicode_300380, None_300381], **kwargs_300382)
    
    # Assigning a type to the variable 'norm' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'norm', pop_call_result_300383)
    
    # Assigning a Call to a Name (line 57):
    
    # Assigning a Call to a Name (line 57):
    
    # Call to pop(...): (line 57)
    # Processing the call arguments (line 57)
    unicode_300386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'unicode', u'cmap')
    # Getting the type of 'None' (line 57)
    None_300387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'None', False)
    # Processing the call keyword arguments (line 57)
    kwargs_300388 = {}
    # Getting the type of 'kwargs' (line 57)
    kwargs_300384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 57)
    pop_300385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), kwargs_300384, 'pop')
    # Calling pop(args, kwargs) (line 57)
    pop_call_result_300389 = invoke(stypy.reporting.localization.Localization(__file__, 57, 11), pop_300385, *[unicode_300386, None_300387], **kwargs_300388)
    
    # Assigning a type to the variable 'cmap' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'cmap', pop_call_result_300389)
    
    # Assigning a Call to a Name (line 58):
    
    # Assigning a Call to a Name (line 58):
    
    # Call to pop(...): (line 58)
    # Processing the call arguments (line 58)
    unicode_300392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 22), 'unicode', u'vmin')
    # Getting the type of 'None' (line 58)
    None_300393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 30), 'None', False)
    # Processing the call keyword arguments (line 58)
    kwargs_300394 = {}
    # Getting the type of 'kwargs' (line 58)
    kwargs_300390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 58)
    pop_300391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), kwargs_300390, 'pop')
    # Calling pop(args, kwargs) (line 58)
    pop_call_result_300395 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), pop_300391, *[unicode_300392, None_300393], **kwargs_300394)
    
    # Assigning a type to the variable 'vmin' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'vmin', pop_call_result_300395)
    
    # Assigning a Call to a Name (line 59):
    
    # Assigning a Call to a Name (line 59):
    
    # Call to pop(...): (line 59)
    # Processing the call arguments (line 59)
    unicode_300398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'unicode', u'vmax')
    # Getting the type of 'None' (line 59)
    None_300399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 30), 'None', False)
    # Processing the call keyword arguments (line 59)
    kwargs_300400 = {}
    # Getting the type of 'kwargs' (line 59)
    kwargs_300396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 59)
    pop_300397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 11), kwargs_300396, 'pop')
    # Calling pop(args, kwargs) (line 59)
    pop_call_result_300401 = invoke(stypy.reporting.localization.Localization(__file__, 59, 11), pop_300397, *[unicode_300398, None_300399], **kwargs_300400)
    
    # Assigning a type to the variable 'vmax' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'vmax', pop_call_result_300401)
    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to pop(...): (line 60)
    # Processing the call arguments (line 60)
    unicode_300404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 25), 'unicode', u'shading')
    unicode_300405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 36), 'unicode', u'flat')
    # Processing the call keyword arguments (line 60)
    kwargs_300406 = {}
    # Getting the type of 'kwargs' (line 60)
    kwargs_300402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 60)
    pop_300403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 14), kwargs_300402, 'pop')
    # Calling pop(args, kwargs) (line 60)
    pop_call_result_300407 = invoke(stypy.reporting.localization.Localization(__file__, 60, 14), pop_300403, *[unicode_300404, unicode_300405], **kwargs_300406)
    
    # Assigning a type to the variable 'shading' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'shading', pop_call_result_300407)
    
    # Assigning a Call to a Name (line 61):
    
    # Assigning a Call to a Name (line 61):
    
    # Call to pop(...): (line 61)
    # Processing the call arguments (line 61)
    unicode_300410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 28), 'unicode', u'facecolors')
    # Getting the type of 'None' (line 61)
    None_300411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 42), 'None', False)
    # Processing the call keyword arguments (line 61)
    kwargs_300412 = {}
    # Getting the type of 'kwargs' (line 61)
    kwargs_300408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 61)
    pop_300409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 17), kwargs_300408, 'pop')
    # Calling pop(args, kwargs) (line 61)
    pop_call_result_300413 = invoke(stypy.reporting.localization.Localization(__file__, 61, 17), pop_300409, *[unicode_300410, None_300411], **kwargs_300412)
    
    # Assigning a type to the variable 'facecolors' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'facecolors', pop_call_result_300413)
    
    
    # Getting the type of 'shading' (line 63)
    shading_300414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 7), 'shading')
    
    # Obtaining an instance of the builtin type 'list' (line 63)
    list_300415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 63)
    # Adding element type (line 63)
    unicode_300416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 23), 'unicode', u'flat')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 22), list_300415, unicode_300416)
    # Adding element type (line 63)
    unicode_300417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 31), 'unicode', u'gouraud')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 22), list_300415, unicode_300417)
    
    # Applying the binary operator 'notin' (line 63)
    result_contains_300418 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 7), 'notin', shading_300414, list_300415)
    
    # Testing the type of an if condition (line 63)
    if_condition_300419 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 4), result_contains_300418)
    # Assigning a type to the variable 'if_condition_300419' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'if_condition_300419', if_condition_300419)
    # SSA begins for if statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 64)
    # Processing the call arguments (line 64)
    
    # Call to format(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'shading' (line 65)
    shading_300423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 42), 'shading', False)
    # Processing the call keyword arguments (line 64)
    kwargs_300424 = {}
    unicode_300421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'unicode', u"shading must be one of ['flat', 'gouraud'] not {0}")
    # Obtaining the member 'format' of a type (line 64)
    format_300422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 25), unicode_300421, 'format')
    # Calling format(args, kwargs) (line 64)
    format_call_result_300425 = invoke(stypy.reporting.localization.Localization(__file__, 64, 25), format_300422, *[shading_300423], **kwargs_300424)
    
    # Processing the call keyword arguments (line 64)
    kwargs_300426 = {}
    # Getting the type of 'ValueError' (line 64)
    ValueError_300420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 64)
    ValueError_call_result_300427 = invoke(stypy.reporting.localization.Localization(__file__, 64, 14), ValueError_300420, *[format_call_result_300425], **kwargs_300426)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 64, 8), ValueError_call_result_300427, 'raise parameter', BaseException)
    # SSA join for if statement (line 63)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 67):
    
    # Assigning a Call to a Name:
    
    # Call to get_from_args_and_kwargs(...): (line 67)
    # Getting the type of 'args' (line 67)
    args_300430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 64), 'args', False)
    # Processing the call keyword arguments (line 67)
    # Getting the type of 'kwargs' (line 67)
    kwargs_300431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 72), 'kwargs', False)
    kwargs_300432 = {'kwargs_300431': kwargs_300431}
    # Getting the type of 'Triangulation' (line 67)
    Triangulation_300428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'Triangulation', False)
    # Obtaining the member 'get_from_args_and_kwargs' of a type (line 67)
    get_from_args_and_kwargs_300429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 24), Triangulation_300428, 'get_from_args_and_kwargs')
    # Calling get_from_args_and_kwargs(args, kwargs) (line 67)
    get_from_args_and_kwargs_call_result_300433 = invoke(stypy.reporting.localization.Localization(__file__, 67, 24), get_from_args_and_kwargs_300429, *[args_300430], **kwargs_300432)
    
    # Assigning a type to the variable 'call_assignment_300349' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'call_assignment_300349', get_from_args_and_kwargs_call_result_300433)
    
    # Assigning a Call to a Name (line 67):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_300436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'int')
    # Processing the call keyword arguments
    kwargs_300437 = {}
    # Getting the type of 'call_assignment_300349' (line 67)
    call_assignment_300349_300434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'call_assignment_300349', False)
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___300435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 4), call_assignment_300349_300434, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_300438 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___300435, *[int_300436], **kwargs_300437)
    
    # Assigning a type to the variable 'call_assignment_300350' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'call_assignment_300350', getitem___call_result_300438)
    
    # Assigning a Name to a Name (line 67):
    # Getting the type of 'call_assignment_300350' (line 67)
    call_assignment_300350_300439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'call_assignment_300350')
    # Assigning a type to the variable 'tri' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'tri', call_assignment_300350_300439)
    
    # Assigning a Call to a Name (line 67):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_300442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'int')
    # Processing the call keyword arguments
    kwargs_300443 = {}
    # Getting the type of 'call_assignment_300349' (line 67)
    call_assignment_300349_300440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'call_assignment_300349', False)
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___300441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 4), call_assignment_300349_300440, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_300444 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___300441, *[int_300442], **kwargs_300443)
    
    # Assigning a type to the variable 'call_assignment_300351' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'call_assignment_300351', getitem___call_result_300444)
    
    # Assigning a Name to a Name (line 67):
    # Getting the type of 'call_assignment_300351' (line 67)
    call_assignment_300351_300445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'call_assignment_300351')
    # Assigning a type to the variable 'args' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 9), 'args', call_assignment_300351_300445)
    
    # Assigning a Call to a Name (line 67):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_300448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'int')
    # Processing the call keyword arguments
    kwargs_300449 = {}
    # Getting the type of 'call_assignment_300349' (line 67)
    call_assignment_300349_300446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'call_assignment_300349', False)
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___300447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 4), call_assignment_300349_300446, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_300450 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___300447, *[int_300448], **kwargs_300449)
    
    # Assigning a type to the variable 'call_assignment_300352' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'call_assignment_300352', getitem___call_result_300450)
    
    # Assigning a Name to a Name (line 67):
    # Getting the type of 'call_assignment_300352' (line 67)
    call_assignment_300352_300451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'call_assignment_300352')
    # Assigning a type to the variable 'kwargs' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'kwargs', call_assignment_300352_300451)
    
    # Type idiom detected: calculating its left and rigth part (line 72)
    # Getting the type of 'facecolors' (line 72)
    facecolors_300452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'facecolors')
    # Getting the type of 'None' (line 72)
    None_300453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'None')
    
    (may_be_300454, more_types_in_union_300455) = may_not_be_none(facecolors_300452, None_300453)

    if may_be_300454:

        if more_types_in_union_300455:
            # Runtime conditional SSA (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 73):
        
        # Assigning a Name to a Name (line 73):
        # Getting the type of 'facecolors' (line 73)
        facecolors_300456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'facecolors')
        # Assigning a type to the variable 'C' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'C', facecolors_300456)

        if more_types_in_union_300455:
            # Runtime conditional SSA for else branch (line 72)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_300454) or more_types_in_union_300455):
        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to asarray(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Obtaining the type of the subscript
        int_300459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 28), 'int')
        # Getting the type of 'args' (line 75)
        args_300460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 75)
        getitem___300461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 23), args_300460, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
        subscript_call_result_300462 = invoke(stypy.reporting.localization.Localization(__file__, 75, 23), getitem___300461, int_300459)
        
        # Processing the call keyword arguments (line 75)
        kwargs_300463 = {}
        # Getting the type of 'np' (line 75)
        np_300457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 75)
        asarray_300458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), np_300457, 'asarray')
        # Calling asarray(args, kwargs) (line 75)
        asarray_call_result_300464 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), asarray_300458, *[subscript_call_result_300462], **kwargs_300463)
        
        # Assigning a type to the variable 'C' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'C', asarray_call_result_300464)

        if (may_be_300454 and more_types_in_union_300455):
            # SSA join for if statement (line 72)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'facecolors' (line 81)
    facecolors_300465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'facecolors')
    # Getting the type of 'None' (line 81)
    None_300466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'None')
    # Applying the binary operator 'is' (line 81)
    result_is__300467 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 8), 'is', facecolors_300465, None_300466)
    
    
    
    # Call to len(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'C' (line 81)
    C_300469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 35), 'C', False)
    # Processing the call keyword arguments (line 81)
    kwargs_300470 = {}
    # Getting the type of 'len' (line 81)
    len_300468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 31), 'len', False)
    # Calling len(args, kwargs) (line 81)
    len_call_result_300471 = invoke(stypy.reporting.localization.Localization(__file__, 81, 31), len_300468, *[C_300469], **kwargs_300470)
    
    
    # Call to len(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'tri' (line 81)
    tri_300473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 45), 'tri', False)
    # Obtaining the member 'triangles' of a type (line 81)
    triangles_300474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 45), tri_300473, 'triangles')
    # Processing the call keyword arguments (line 81)
    kwargs_300475 = {}
    # Getting the type of 'len' (line 81)
    len_300472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 41), 'len', False)
    # Calling len(args, kwargs) (line 81)
    len_call_result_300476 = invoke(stypy.reporting.localization.Localization(__file__, 81, 41), len_300472, *[triangles_300474], **kwargs_300475)
    
    # Applying the binary operator '==' (line 81)
    result_eq_300477 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 31), '==', len_call_result_300471, len_call_result_300476)
    
    # Applying the binary operator 'and' (line 81)
    result_and_keyword_300478 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 8), 'and', result_is__300467, result_eq_300477)
    
    
    # Call to len(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'C' (line 82)
    C_300480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'C', False)
    # Processing the call keyword arguments (line 82)
    kwargs_300481 = {}
    # Getting the type of 'len' (line 82)
    len_300479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'len', False)
    # Calling len(args, kwargs) (line 82)
    len_call_result_300482 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), len_300479, *[C_300480], **kwargs_300481)
    
    
    # Call to len(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'tri' (line 82)
    tri_300484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'tri', False)
    # Obtaining the member 'x' of a type (line 82)
    x_300485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 26), tri_300484, 'x')
    # Processing the call keyword arguments (line 82)
    kwargs_300486 = {}
    # Getting the type of 'len' (line 82)
    len_300483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'len', False)
    # Calling len(args, kwargs) (line 82)
    len_call_result_300487 = invoke(stypy.reporting.localization.Localization(__file__, 82, 22), len_300483, *[x_300485], **kwargs_300486)
    
    # Applying the binary operator '!=' (line 82)
    result_ne_300488 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 12), '!=', len_call_result_300482, len_call_result_300487)
    
    # Applying the binary operator 'and' (line 81)
    result_and_keyword_300489 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 8), 'and', result_and_keyword_300478, result_ne_300488)
    
    # Getting the type of 'shading' (line 82)
    shading_300490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 37), 'shading')
    unicode_300491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 48), 'unicode', u'gouraud')
    # Applying the binary operator '!=' (line 82)
    result_ne_300492 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 37), '!=', shading_300490, unicode_300491)
    
    # Applying the binary operator 'and' (line 81)
    result_and_keyword_300493 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 8), 'and', result_and_keyword_300489, result_ne_300492)
    
    # Testing the type of an if condition (line 81)
    if_condition_300494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 4), result_and_keyword_300493)
    # Assigning a type to the variable 'if_condition_300494' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'if_condition_300494', if_condition_300494)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 83):
    
    # Assigning a Name to a Name (line 83):
    # Getting the type of 'C' (line 83)
    C_300495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 21), 'C')
    # Assigning a type to the variable 'facecolors' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'facecolors', C_300495)
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Getting the type of 'facecolors' (line 86)
    facecolors_300496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 9), 'facecolors')
    # Getting the type of 'None' (line 86)
    None_300497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), 'None')
    # Applying the binary operator 'is' (line 86)
    result_is__300498 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 9), 'is', facecolors_300496, None_300497)
    
    
    
    # Call to len(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'C' (line 86)
    C_300500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 36), 'C', False)
    # Processing the call keyword arguments (line 86)
    kwargs_300501 = {}
    # Getting the type of 'len' (line 86)
    len_300499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 32), 'len', False)
    # Calling len(args, kwargs) (line 86)
    len_call_result_300502 = invoke(stypy.reporting.localization.Localization(__file__, 86, 32), len_300499, *[C_300500], **kwargs_300501)
    
    
    # Call to len(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'tri' (line 86)
    tri_300504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 46), 'tri', False)
    # Obtaining the member 'x' of a type (line 86)
    x_300505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 46), tri_300504, 'x')
    # Processing the call keyword arguments (line 86)
    kwargs_300506 = {}
    # Getting the type of 'len' (line 86)
    len_300503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 42), 'len', False)
    # Calling len(args, kwargs) (line 86)
    len_call_result_300507 = invoke(stypy.reporting.localization.Localization(__file__, 86, 42), len_300503, *[x_300505], **kwargs_300506)
    
    # Applying the binary operator '!=' (line 86)
    result_ne_300508 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 32), '!=', len_call_result_300502, len_call_result_300507)
    
    # Applying the binary operator 'and' (line 86)
    result_and_keyword_300509 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 9), 'and', result_is__300498, result_ne_300508)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'facecolors' (line 87)
    facecolors_300510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'facecolors')
    # Getting the type of 'None' (line 87)
    None_300511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 31), 'None')
    # Applying the binary operator 'isnot' (line 87)
    result_is_not_300512 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 13), 'isnot', facecolors_300510, None_300511)
    
    
    
    # Call to len(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'C' (line 87)
    C_300514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 44), 'C', False)
    # Processing the call keyword arguments (line 87)
    kwargs_300515 = {}
    # Getting the type of 'len' (line 87)
    len_300513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 40), 'len', False)
    # Calling len(args, kwargs) (line 87)
    len_call_result_300516 = invoke(stypy.reporting.localization.Localization(__file__, 87, 40), len_300513, *[C_300514], **kwargs_300515)
    
    
    # Call to len(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'tri' (line 87)
    tri_300518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 54), 'tri', False)
    # Obtaining the member 'triangles' of a type (line 87)
    triangles_300519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 54), tri_300518, 'triangles')
    # Processing the call keyword arguments (line 87)
    kwargs_300520 = {}
    # Getting the type of 'len' (line 87)
    len_300517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 50), 'len', False)
    # Calling len(args, kwargs) (line 87)
    len_call_result_300521 = invoke(stypy.reporting.localization.Localization(__file__, 87, 50), len_300517, *[triangles_300519], **kwargs_300520)
    
    # Applying the binary operator '!=' (line 87)
    result_ne_300522 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 40), '!=', len_call_result_300516, len_call_result_300521)
    
    # Applying the binary operator 'and' (line 87)
    result_and_keyword_300523 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 13), 'and', result_is_not_300512, result_ne_300522)
    
    # Applying the binary operator 'or' (line 86)
    result_or_keyword_300524 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 8), 'or', result_and_keyword_300509, result_and_keyword_300523)
    
    # Testing the type of an if condition (line 86)
    if_condition_300525 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 4), result_or_keyword_300524)
    # Assigning a type to the variable 'if_condition_300525' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'if_condition_300525', if_condition_300525)
    # SSA begins for if statement (line 86)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 88)
    # Processing the call arguments (line 88)
    unicode_300527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 25), 'unicode', u'Length of color values array must be the same as either the number of triangulation points or triangles')
    # Processing the call keyword arguments (line 88)
    kwargs_300528 = {}
    # Getting the type of 'ValueError' (line 88)
    ValueError_300526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 88)
    ValueError_call_result_300529 = invoke(stypy.reporting.localization.Localization(__file__, 88, 14), ValueError_300526, *[unicode_300527], **kwargs_300528)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 88, 8), ValueError_call_result_300529, 'raise parameter', BaseException)
    # SSA join for if statement (line 86)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Name (line 94):
    
    # Assigning a Tuple to a Name (line 94):
    
    # Obtaining an instance of the builtin type 'tuple' (line 94)
    tuple_300530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 94)
    # Adding element type (line 94)
    float_300531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 18), tuple_300530, float_300531)
    
    # Assigning a type to the variable 'linewidths' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'linewidths', tuple_300530)
    
    
    unicode_300532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 7), 'unicode', u'linewidth')
    # Getting the type of 'kwargs' (line 95)
    kwargs_300533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 22), 'kwargs')
    # Applying the binary operator 'in' (line 95)
    result_contains_300534 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 7), 'in', unicode_300532, kwargs_300533)
    
    # Testing the type of an if condition (line 95)
    if_condition_300535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 4), result_contains_300534)
    # Assigning a type to the variable 'if_condition_300535' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'if_condition_300535', if_condition_300535)
    # SSA begins for if statement (line 95)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 96):
    
    # Assigning a Call to a Subscript (line 96):
    
    # Call to pop(...): (line 96)
    # Processing the call arguments (line 96)
    unicode_300538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 42), 'unicode', u'linewidth')
    # Processing the call keyword arguments (line 96)
    kwargs_300539 = {}
    # Getting the type of 'kwargs' (line 96)
    kwargs_300536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 31), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 96)
    pop_300537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 31), kwargs_300536, 'pop')
    # Calling pop(args, kwargs) (line 96)
    pop_call_result_300540 = invoke(stypy.reporting.localization.Localization(__file__, 96, 31), pop_300537, *[unicode_300538], **kwargs_300539)
    
    # Getting the type of 'kwargs' (line 96)
    kwargs_300541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'kwargs')
    unicode_300542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 15), 'unicode', u'linewidths')
    # Storing an element on a container (line 96)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 8), kwargs_300541, (unicode_300542, pop_call_result_300540))
    # SSA join for if statement (line 95)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to setdefault(...): (line 97)
    # Processing the call arguments (line 97)
    unicode_300545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 22), 'unicode', u'linewidths')
    # Getting the type of 'linewidths' (line 97)
    linewidths_300546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 36), 'linewidths', False)
    # Processing the call keyword arguments (line 97)
    kwargs_300547 = {}
    # Getting the type of 'kwargs' (line 97)
    kwargs_300543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'kwargs', False)
    # Obtaining the member 'setdefault' of a type (line 97)
    setdefault_300544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 4), kwargs_300543, 'setdefault')
    # Calling setdefault(args, kwargs) (line 97)
    setdefault_call_result_300548 = invoke(stypy.reporting.localization.Localization(__file__, 97, 4), setdefault_300544, *[unicode_300545, linewidths_300546], **kwargs_300547)
    
    
    # Assigning a Str to a Name (line 99):
    
    # Assigning a Str to a Name (line 99):
    unicode_300549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 17), 'unicode', u'none')
    # Assigning a type to the variable 'edgecolors' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'edgecolors', unicode_300549)
    
    
    unicode_300550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 7), 'unicode', u'edgecolor')
    # Getting the type of 'kwargs' (line 100)
    kwargs_300551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'kwargs')
    # Applying the binary operator 'in' (line 100)
    result_contains_300552 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 7), 'in', unicode_300550, kwargs_300551)
    
    # Testing the type of an if condition (line 100)
    if_condition_300553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 4), result_contains_300552)
    # Assigning a type to the variable 'if_condition_300553' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'if_condition_300553', if_condition_300553)
    # SSA begins for if statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 101):
    
    # Assigning a Call to a Subscript (line 101):
    
    # Call to pop(...): (line 101)
    # Processing the call arguments (line 101)
    unicode_300556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 42), 'unicode', u'edgecolor')
    # Processing the call keyword arguments (line 101)
    kwargs_300557 = {}
    # Getting the type of 'kwargs' (line 101)
    kwargs_300554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 101)
    pop_300555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 31), kwargs_300554, 'pop')
    # Calling pop(args, kwargs) (line 101)
    pop_call_result_300558 = invoke(stypy.reporting.localization.Localization(__file__, 101, 31), pop_300555, *[unicode_300556], **kwargs_300557)
    
    # Getting the type of 'kwargs' (line 101)
    kwargs_300559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'kwargs')
    unicode_300560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'unicode', u'edgecolors')
    # Storing an element on a container (line 101)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 8), kwargs_300559, (unicode_300560, pop_call_result_300558))
    # SSA join for if statement (line 100)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 102):
    
    # Assigning a Call to a Name (line 102):
    
    # Call to setdefault(...): (line 102)
    # Processing the call arguments (line 102)
    unicode_300563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 27), 'unicode', u'edgecolors')
    # Getting the type of 'edgecolors' (line 102)
    edgecolors_300564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 41), 'edgecolors', False)
    # Processing the call keyword arguments (line 102)
    kwargs_300565 = {}
    # Getting the type of 'kwargs' (line 102)
    kwargs_300561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 9), 'kwargs', False)
    # Obtaining the member 'setdefault' of a type (line 102)
    setdefault_300562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 9), kwargs_300561, 'setdefault')
    # Calling setdefault(args, kwargs) (line 102)
    setdefault_call_result_300566 = invoke(stypy.reporting.localization.Localization(__file__, 102, 9), setdefault_300562, *[unicode_300563, edgecolors_300564], **kwargs_300565)
    
    # Assigning a type to the variable 'ec' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'ec', setdefault_call_result_300566)
    
    
    unicode_300567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 7), 'unicode', u'antialiased')
    # Getting the type of 'kwargs' (line 104)
    kwargs_300568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'kwargs')
    # Applying the binary operator 'in' (line 104)
    result_contains_300569 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 7), 'in', unicode_300567, kwargs_300568)
    
    # Testing the type of an if condition (line 104)
    if_condition_300570 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 4), result_contains_300569)
    # Assigning a type to the variable 'if_condition_300570' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'if_condition_300570', if_condition_300570)
    # SSA begins for if statement (line 104)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 105):
    
    # Assigning a Call to a Subscript (line 105):
    
    # Call to pop(...): (line 105)
    # Processing the call arguments (line 105)
    unicode_300573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 44), 'unicode', u'antialiased')
    # Processing the call keyword arguments (line 105)
    kwargs_300574 = {}
    # Getting the type of 'kwargs' (line 105)
    kwargs_300571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 105)
    pop_300572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 33), kwargs_300571, 'pop')
    # Calling pop(args, kwargs) (line 105)
    pop_call_result_300575 = invoke(stypy.reporting.localization.Localization(__file__, 105, 33), pop_300572, *[unicode_300573], **kwargs_300574)
    
    # Getting the type of 'kwargs' (line 105)
    kwargs_300576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'kwargs')
    unicode_300577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 15), 'unicode', u'antialiaseds')
    # Storing an element on a container (line 105)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 8), kwargs_300576, (unicode_300577, pop_call_result_300575))
    # SSA join for if statement (line 104)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    unicode_300578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 7), 'unicode', u'antialiaseds')
    # Getting the type of 'kwargs' (line 106)
    kwargs_300579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'kwargs')
    # Applying the binary operator 'notin' (line 106)
    result_contains_300580 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 7), 'notin', unicode_300578, kwargs_300579)
    
    
    
    # Call to lower(...): (line 106)
    # Processing the call keyword arguments (line 106)
    kwargs_300583 = {}
    # Getting the type of 'ec' (line 106)
    ec_300581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'ec', False)
    # Obtaining the member 'lower' of a type (line 106)
    lower_300582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 40), ec_300581, 'lower')
    # Calling lower(args, kwargs) (line 106)
    lower_call_result_300584 = invoke(stypy.reporting.localization.Localization(__file__, 106, 40), lower_300582, *[], **kwargs_300583)
    
    unicode_300585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 54), 'unicode', u'none')
    # Applying the binary operator '==' (line 106)
    result_eq_300586 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 40), '==', lower_call_result_300584, unicode_300585)
    
    # Applying the binary operator 'and' (line 106)
    result_and_keyword_300587 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 7), 'and', result_contains_300580, result_eq_300586)
    
    # Testing the type of an if condition (line 106)
    if_condition_300588 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 4), result_and_keyword_300587)
    # Assigning a type to the variable 'if_condition_300588' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'if_condition_300588', if_condition_300588)
    # SSA begins for if statement (line 106)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 107):
    
    # Assigning a Name to a Subscript (line 107):
    # Getting the type of 'False' (line 107)
    False_300589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 33), 'False')
    # Getting the type of 'kwargs' (line 107)
    kwargs_300590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'kwargs')
    unicode_300591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 15), 'unicode', u'antialiaseds')
    # Storing an element on a container (line 107)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 8), kwargs_300590, (unicode_300591, False_300589))
    # SSA join for if statement (line 106)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'shading' (line 109)
    shading_300592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 7), 'shading')
    unicode_300593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 18), 'unicode', u'gouraud')
    # Applying the binary operator '==' (line 109)
    result_eq_300594 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 7), '==', shading_300592, unicode_300593)
    
    # Testing the type of an if condition (line 109)
    if_condition_300595 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 4), result_eq_300594)
    # Assigning a type to the variable 'if_condition_300595' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'if_condition_300595', if_condition_300595)
    # SSA begins for if statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 110)
    # Getting the type of 'facecolors' (line 110)
    facecolors_300596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'facecolors')
    # Getting the type of 'None' (line 110)
    None_300597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'None')
    
    (may_be_300598, more_types_in_union_300599) = may_not_be_none(facecolors_300596, None_300597)

    if may_be_300598:

        if more_types_in_union_300599:
            # Runtime conditional SSA (line 110)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 111)
        # Processing the call arguments (line 111)
        unicode_300601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 29), 'unicode', u'Gouraud shading does not support the use of facecolors kwarg')
        # Processing the call keyword arguments (line 111)
        kwargs_300602 = {}
        # Getting the type of 'ValueError' (line 111)
        ValueError_300600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 111)
        ValueError_call_result_300603 = invoke(stypy.reporting.localization.Localization(__file__, 111, 18), ValueError_300600, *[unicode_300601], **kwargs_300602)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 111, 12), ValueError_call_result_300603, 'raise parameter', BaseException)

        if more_types_in_union_300599:
            # SSA join for if statement (line 110)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to len(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'C' (line 113)
    C_300605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'C', False)
    # Processing the call keyword arguments (line 113)
    kwargs_300606 = {}
    # Getting the type of 'len' (line 113)
    len_300604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'len', False)
    # Calling len(args, kwargs) (line 113)
    len_call_result_300607 = invoke(stypy.reporting.localization.Localization(__file__, 113, 11), len_300604, *[C_300605], **kwargs_300606)
    
    
    # Call to len(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'tri' (line 113)
    tri_300609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'tri', False)
    # Obtaining the member 'x' of a type (line 113)
    x_300610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 25), tri_300609, 'x')
    # Processing the call keyword arguments (line 113)
    kwargs_300611 = {}
    # Getting the type of 'len' (line 113)
    len_300608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 21), 'len', False)
    # Calling len(args, kwargs) (line 113)
    len_call_result_300612 = invoke(stypy.reporting.localization.Localization(__file__, 113, 21), len_300608, *[x_300610], **kwargs_300611)
    
    # Applying the binary operator '!=' (line 113)
    result_ne_300613 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 11), '!=', len_call_result_300607, len_call_result_300612)
    
    # Testing the type of an if condition (line 113)
    if_condition_300614 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 8), result_ne_300613)
    # Assigning a type to the variable 'if_condition_300614' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'if_condition_300614', if_condition_300614)
    # SSA begins for if statement (line 113)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 114)
    # Processing the call arguments (line 114)
    unicode_300616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 29), 'unicode', u'For gouraud shading, the length of color values array must be the same as the number of triangulation points')
    # Processing the call keyword arguments (line 114)
    kwargs_300617 = {}
    # Getting the type of 'ValueError' (line 114)
    ValueError_300615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 114)
    ValueError_call_result_300618 = invoke(stypy.reporting.localization.Localization(__file__, 114, 18), ValueError_300615, *[unicode_300616], **kwargs_300617)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 114, 12), ValueError_call_result_300618, 'raise parameter', BaseException)
    # SSA join for if statement (line 113)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 117):
    
    # Assigning a Call to a Name (line 117):
    
    # Call to TriMesh(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'tri' (line 117)
    tri_300620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'tri', False)
    # Processing the call keyword arguments (line 117)
    # Getting the type of 'kwargs' (line 117)
    kwargs_300621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 36), 'kwargs', False)
    kwargs_300622 = {'kwargs_300621': kwargs_300621}
    # Getting the type of 'TriMesh' (line 117)
    TriMesh_300619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'TriMesh', False)
    # Calling TriMesh(args, kwargs) (line 117)
    TriMesh_call_result_300623 = invoke(stypy.reporting.localization.Localization(__file__, 117, 21), TriMesh_300619, *[tri_300620], **kwargs_300622)
    
    # Assigning a type to the variable 'collection' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'collection', TriMesh_call_result_300623)
    # SSA branch for the else part of an if statement (line 109)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 120):
    
    # Assigning a Call to a Name (line 120):
    
    # Call to get_masked_triangles(...): (line 120)
    # Processing the call keyword arguments (line 120)
    kwargs_300626 = {}
    # Getting the type of 'tri' (line 120)
    tri_300624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'tri', False)
    # Obtaining the member 'get_masked_triangles' of a type (line 120)
    get_masked_triangles_300625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 21), tri_300624, 'get_masked_triangles')
    # Calling get_masked_triangles(args, kwargs) (line 120)
    get_masked_triangles_call_result_300627 = invoke(stypy.reporting.localization.Localization(__file__, 120, 21), get_masked_triangles_300625, *[], **kwargs_300626)
    
    # Assigning a type to the variable 'maskedTris' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'maskedTris', get_masked_triangles_call_result_300627)
    
    # Assigning a Call to a Name (line 121):
    
    # Assigning a Call to a Name (line 121):
    
    # Call to concatenate(...): (line 121)
    # Processing the call arguments (line 121)
    
    # Obtaining an instance of the builtin type 'tuple' (line 121)
    tuple_300630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 121)
    # Adding element type (line 121)
    
    # Obtaining the type of the subscript
    Ellipsis_300631 = Ellipsis
    # Getting the type of 'np' (line 121)
    np_300632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 55), 'np', False)
    # Obtaining the member 'newaxis' of a type (line 121)
    newaxis_300633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 55), np_300632, 'newaxis')
    
    # Obtaining the type of the subscript
    # Getting the type of 'maskedTris' (line 121)
    maskedTris_300634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 38), 'maskedTris', False)
    # Getting the type of 'tri' (line 121)
    tri_300635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'tri', False)
    # Obtaining the member 'x' of a type (line 121)
    x_300636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 32), tri_300635, 'x')
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___300637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 32), x_300636, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_300638 = invoke(stypy.reporting.localization.Localization(__file__, 121, 32), getitem___300637, maskedTris_300634)
    
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___300639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 32), subscript_call_result_300638, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_300640 = invoke(stypy.reporting.localization.Localization(__file__, 121, 32), getitem___300639, (Ellipsis_300631, newaxis_300633))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 32), tuple_300630, subscript_call_result_300640)
    # Adding element type (line 121)
    
    # Obtaining the type of the subscript
    Ellipsis_300641 = Ellipsis
    # Getting the type of 'np' (line 122)
    np_300642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 55), 'np', False)
    # Obtaining the member 'newaxis' of a type (line 122)
    newaxis_300643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 55), np_300642, 'newaxis')
    
    # Obtaining the type of the subscript
    # Getting the type of 'maskedTris' (line 122)
    maskedTris_300644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 38), 'maskedTris', False)
    # Getting the type of 'tri' (line 122)
    tri_300645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 32), 'tri', False)
    # Obtaining the member 'y' of a type (line 122)
    y_300646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 32), tri_300645, 'y')
    # Obtaining the member '__getitem__' of a type (line 122)
    getitem___300647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 32), y_300646, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 122)
    subscript_call_result_300648 = invoke(stypy.reporting.localization.Localization(__file__, 122, 32), getitem___300647, maskedTris_300644)
    
    # Obtaining the member '__getitem__' of a type (line 122)
    getitem___300649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 32), subscript_call_result_300648, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 122)
    subscript_call_result_300650 = invoke(stypy.reporting.localization.Localization(__file__, 122, 32), getitem___300649, (Ellipsis_300641, newaxis_300643))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 32), tuple_300630, subscript_call_result_300650)
    
    # Processing the call keyword arguments (line 121)
    int_300651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 74), 'int')
    keyword_300652 = int_300651
    kwargs_300653 = {'axis': keyword_300652}
    # Getting the type of 'np' (line 121)
    np_300628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 121)
    concatenate_300629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), np_300628, 'concatenate')
    # Calling concatenate(args, kwargs) (line 121)
    concatenate_call_result_300654 = invoke(stypy.reporting.localization.Localization(__file__, 121, 16), concatenate_300629, *[tuple_300630], **kwargs_300653)
    
    # Assigning a type to the variable 'verts' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'verts', concatenate_call_result_300654)
    
    # Type idiom detected: calculating its left and rigth part (line 125)
    # Getting the type of 'facecolors' (line 125)
    facecolors_300655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'facecolors')
    # Getting the type of 'None' (line 125)
    None_300656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'None')
    
    (may_be_300657, more_types_in_union_300658) = may_be_none(facecolors_300655, None_300656)

    if may_be_300657:

        if more_types_in_union_300658:
            # Runtime conditional SSA (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 127):
        
        # Assigning a Call to a Name (line 127):
        
        # Call to mean(...): (line 127)
        # Processing the call keyword arguments (line 127)
        int_300664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 40), 'int')
        keyword_300665 = int_300664
        kwargs_300666 = {'axis': keyword_300665}
        
        # Obtaining the type of the subscript
        # Getting the type of 'maskedTris' (line 127)
        maskedTris_300659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 18), 'maskedTris', False)
        # Getting the type of 'C' (line 127)
        C_300660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'C', False)
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___300661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 16), C_300660, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_300662 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), getitem___300661, maskedTris_300659)
        
        # Obtaining the member 'mean' of a type (line 127)
        mean_300663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 16), subscript_call_result_300662, 'mean')
        # Calling mean(args, kwargs) (line 127)
        mean_call_result_300667 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), mean_300663, *[], **kwargs_300666)
        
        # Assigning a type to the variable 'C' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'C', mean_call_result_300667)

        if more_types_in_union_300658:
            # Runtime conditional SSA for else branch (line 125)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_300657) or more_types_in_union_300658):
        
        
        # Getting the type of 'tri' (line 128)
        tri_300668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 13), 'tri')
        # Obtaining the member 'mask' of a type (line 128)
        mask_300669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 13), tri_300668, 'mask')
        # Getting the type of 'None' (line 128)
        None_300670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'None')
        # Applying the binary operator 'isnot' (line 128)
        result_is_not_300671 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 13), 'isnot', mask_300669, None_300670)
        
        # Testing the type of an if condition (line 128)
        if_condition_300672 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 13), result_is_not_300671)
        # Assigning a type to the variable 'if_condition_300672' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 13), 'if_condition_300672', if_condition_300672)
        # SSA begins for if statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to compress(...): (line 130)
        # Processing the call arguments (line 130)
        int_300675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 27), 'int')
        # Getting the type of 'tri' (line 130)
        tri_300676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 29), 'tri', False)
        # Obtaining the member 'mask' of a type (line 130)
        mask_300677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 29), tri_300676, 'mask')
        # Applying the binary operator '-' (line 130)
        result_sub_300678 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 27), '-', int_300675, mask_300677)
        
        # Processing the call keyword arguments (line 130)
        kwargs_300679 = {}
        # Getting the type of 'C' (line 130)
        C_300673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'C', False)
        # Obtaining the member 'compress' of a type (line 130)
        compress_300674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 16), C_300673, 'compress')
        # Calling compress(args, kwargs) (line 130)
        compress_call_result_300680 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), compress_300674, *[result_sub_300678], **kwargs_300679)
        
        # Assigning a type to the variable 'C' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'C', compress_call_result_300680)
        # SSA join for if statement (line 128)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_300657 and more_types_in_union_300658):
            # SSA join for if statement (line 125)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 132):
    
    # Assigning a Call to a Name (line 132):
    
    # Call to PolyCollection(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'verts' (line 132)
    verts_300682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 36), 'verts', False)
    # Processing the call keyword arguments (line 132)
    # Getting the type of 'kwargs' (line 132)
    kwargs_300683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 45), 'kwargs', False)
    kwargs_300684 = {'kwargs_300683': kwargs_300683}
    # Getting the type of 'PolyCollection' (line 132)
    PolyCollection_300681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 21), 'PolyCollection', False)
    # Calling PolyCollection(args, kwargs) (line 132)
    PolyCollection_call_result_300685 = invoke(stypy.reporting.localization.Localization(__file__, 132, 21), PolyCollection_300681, *[verts_300682], **kwargs_300684)
    
    # Assigning a type to the variable 'collection' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'collection', PolyCollection_call_result_300685)
    # SSA join for if statement (line 109)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to set_alpha(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'alpha' (line 134)
    alpha_300688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'alpha', False)
    # Processing the call keyword arguments (line 134)
    kwargs_300689 = {}
    # Getting the type of 'collection' (line 134)
    collection_300686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'collection', False)
    # Obtaining the member 'set_alpha' of a type (line 134)
    set_alpha_300687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 4), collection_300686, 'set_alpha')
    # Calling set_alpha(args, kwargs) (line 134)
    set_alpha_call_result_300690 = invoke(stypy.reporting.localization.Localization(__file__, 134, 4), set_alpha_300687, *[alpha_300688], **kwargs_300689)
    
    
    # Call to set_array(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'C' (line 135)
    C_300693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 25), 'C', False)
    # Processing the call keyword arguments (line 135)
    kwargs_300694 = {}
    # Getting the type of 'collection' (line 135)
    collection_300691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'collection', False)
    # Obtaining the member 'set_array' of a type (line 135)
    set_array_300692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 4), collection_300691, 'set_array')
    # Calling set_array(args, kwargs) (line 135)
    set_array_call_result_300695 = invoke(stypy.reporting.localization.Localization(__file__, 135, 4), set_array_300692, *[C_300693], **kwargs_300694)
    
    
    # Type idiom detected: calculating its left and rigth part (line 136)
    # Getting the type of 'norm' (line 136)
    norm_300696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'norm')
    # Getting the type of 'None' (line 136)
    None_300697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'None')
    
    (may_be_300698, more_types_in_union_300699) = may_not_be_none(norm_300696, None_300697)

    if may_be_300698:

        if more_types_in_union_300699:
            # Runtime conditional SSA (line 136)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to isinstance(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'norm' (line 137)
        norm_300701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 26), 'norm', False)
        # Getting the type of 'Normalize' (line 137)
        Normalize_300702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 32), 'Normalize', False)
        # Processing the call keyword arguments (line 137)
        kwargs_300703 = {}
        # Getting the type of 'isinstance' (line 137)
        isinstance_300700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 137)
        isinstance_call_result_300704 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), isinstance_300700, *[norm_300701, Normalize_300702], **kwargs_300703)
        
        # Applying the 'not' unary operator (line 137)
        result_not__300705 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 11), 'not', isinstance_call_result_300704)
        
        # Testing the type of an if condition (line 137)
        if_condition_300706 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 8), result_not__300705)
        # Assigning a type to the variable 'if_condition_300706' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'if_condition_300706', if_condition_300706)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 138):
        
        # Assigning a Str to a Name (line 138):
        unicode_300707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 18), 'unicode', u"'norm' must be an instance of 'Normalize'")
        # Assigning a type to the variable 'msg' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'msg', unicode_300707)
        
        # Call to ValueError(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'msg' (line 139)
        msg_300709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 29), 'msg', False)
        # Processing the call keyword arguments (line 139)
        kwargs_300710 = {}
        # Getting the type of 'ValueError' (line 139)
        ValueError_300708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 139)
        ValueError_call_result_300711 = invoke(stypy.reporting.localization.Localization(__file__, 139, 18), ValueError_300708, *[msg_300709], **kwargs_300710)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 139, 12), ValueError_call_result_300711, 'raise parameter', BaseException)
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_300699:
            # SSA join for if statement (line 136)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to set_cmap(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'cmap' (line 140)
    cmap_300714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'cmap', False)
    # Processing the call keyword arguments (line 140)
    kwargs_300715 = {}
    # Getting the type of 'collection' (line 140)
    collection_300712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'collection', False)
    # Obtaining the member 'set_cmap' of a type (line 140)
    set_cmap_300713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 4), collection_300712, 'set_cmap')
    # Calling set_cmap(args, kwargs) (line 140)
    set_cmap_call_result_300716 = invoke(stypy.reporting.localization.Localization(__file__, 140, 4), set_cmap_300713, *[cmap_300714], **kwargs_300715)
    
    
    # Call to set_norm(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'norm' (line 141)
    norm_300719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 24), 'norm', False)
    # Processing the call keyword arguments (line 141)
    kwargs_300720 = {}
    # Getting the type of 'collection' (line 141)
    collection_300717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'collection', False)
    # Obtaining the member 'set_norm' of a type (line 141)
    set_norm_300718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 4), collection_300717, 'set_norm')
    # Calling set_norm(args, kwargs) (line 141)
    set_norm_call_result_300721 = invoke(stypy.reporting.localization.Localization(__file__, 141, 4), set_norm_300718, *[norm_300719], **kwargs_300720)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'vmin' (line 142)
    vmin_300722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 7), 'vmin')
    # Getting the type of 'None' (line 142)
    None_300723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'None')
    # Applying the binary operator 'isnot' (line 142)
    result_is_not_300724 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 7), 'isnot', vmin_300722, None_300723)
    
    
    # Getting the type of 'vmax' (line 142)
    vmax_300725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 27), 'vmax')
    # Getting the type of 'None' (line 142)
    None_300726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 39), 'None')
    # Applying the binary operator 'isnot' (line 142)
    result_is_not_300727 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 27), 'isnot', vmax_300725, None_300726)
    
    # Applying the binary operator 'or' (line 142)
    result_or_keyword_300728 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 7), 'or', result_is_not_300724, result_is_not_300727)
    
    # Testing the type of an if condition (line 142)
    if_condition_300729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 4), result_or_keyword_300728)
    # Assigning a type to the variable 'if_condition_300729' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'if_condition_300729', if_condition_300729)
    # SSA begins for if statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_clim(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'vmin' (line 143)
    vmin_300732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 28), 'vmin', False)
    # Getting the type of 'vmax' (line 143)
    vmax_300733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 34), 'vmax', False)
    # Processing the call keyword arguments (line 143)
    kwargs_300734 = {}
    # Getting the type of 'collection' (line 143)
    collection_300730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'collection', False)
    # Obtaining the member 'set_clim' of a type (line 143)
    set_clim_300731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), collection_300730, 'set_clim')
    # Calling set_clim(args, kwargs) (line 143)
    set_clim_call_result_300735 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), set_clim_300731, *[vmin_300732, vmax_300733], **kwargs_300734)
    
    # SSA branch for the else part of an if statement (line 142)
    module_type_store.open_ssa_branch('else')
    
    # Call to autoscale_None(...): (line 145)
    # Processing the call keyword arguments (line 145)
    kwargs_300738 = {}
    # Getting the type of 'collection' (line 145)
    collection_300736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'collection', False)
    # Obtaining the member 'autoscale_None' of a type (line 145)
    autoscale_None_300737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), collection_300736, 'autoscale_None')
    # Calling autoscale_None(args, kwargs) (line 145)
    autoscale_None_call_result_300739 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), autoscale_None_300737, *[], **kwargs_300738)
    
    # SSA join for if statement (line 142)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to grid(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'False' (line 146)
    False_300742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'False', False)
    # Processing the call keyword arguments (line 146)
    kwargs_300743 = {}
    # Getting the type of 'ax' (line 146)
    ax_300740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'ax', False)
    # Obtaining the member 'grid' of a type (line 146)
    grid_300741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 4), ax_300740, 'grid')
    # Calling grid(args, kwargs) (line 146)
    grid_call_result_300744 = invoke(stypy.reporting.localization.Localization(__file__, 146, 4), grid_300741, *[False_300742], **kwargs_300743)
    
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to min(...): (line 148)
    # Processing the call keyword arguments (line 148)
    kwargs_300748 = {}
    # Getting the type of 'tri' (line 148)
    tri_300745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'tri', False)
    # Obtaining the member 'x' of a type (line 148)
    x_300746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 11), tri_300745, 'x')
    # Obtaining the member 'min' of a type (line 148)
    min_300747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 11), x_300746, 'min')
    # Calling min(args, kwargs) (line 148)
    min_call_result_300749 = invoke(stypy.reporting.localization.Localization(__file__, 148, 11), min_300747, *[], **kwargs_300748)
    
    # Assigning a type to the variable 'minx' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'minx', min_call_result_300749)
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to max(...): (line 149)
    # Processing the call keyword arguments (line 149)
    kwargs_300753 = {}
    # Getting the type of 'tri' (line 149)
    tri_300750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'tri', False)
    # Obtaining the member 'x' of a type (line 149)
    x_300751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 11), tri_300750, 'x')
    # Obtaining the member 'max' of a type (line 149)
    max_300752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 11), x_300751, 'max')
    # Calling max(args, kwargs) (line 149)
    max_call_result_300754 = invoke(stypy.reporting.localization.Localization(__file__, 149, 11), max_300752, *[], **kwargs_300753)
    
    # Assigning a type to the variable 'maxx' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'maxx', max_call_result_300754)
    
    # Assigning a Call to a Name (line 150):
    
    # Assigning a Call to a Name (line 150):
    
    # Call to min(...): (line 150)
    # Processing the call keyword arguments (line 150)
    kwargs_300758 = {}
    # Getting the type of 'tri' (line 150)
    tri_300755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'tri', False)
    # Obtaining the member 'y' of a type (line 150)
    y_300756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 11), tri_300755, 'y')
    # Obtaining the member 'min' of a type (line 150)
    min_300757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 11), y_300756, 'min')
    # Calling min(args, kwargs) (line 150)
    min_call_result_300759 = invoke(stypy.reporting.localization.Localization(__file__, 150, 11), min_300757, *[], **kwargs_300758)
    
    # Assigning a type to the variable 'miny' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'miny', min_call_result_300759)
    
    # Assigning a Call to a Name (line 151):
    
    # Assigning a Call to a Name (line 151):
    
    # Call to max(...): (line 151)
    # Processing the call keyword arguments (line 151)
    kwargs_300763 = {}
    # Getting the type of 'tri' (line 151)
    tri_300760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'tri', False)
    # Obtaining the member 'y' of a type (line 151)
    y_300761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), tri_300760, 'y')
    # Obtaining the member 'max' of a type (line 151)
    max_300762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), y_300761, 'max')
    # Calling max(args, kwargs) (line 151)
    max_call_result_300764 = invoke(stypy.reporting.localization.Localization(__file__, 151, 11), max_300762, *[], **kwargs_300763)
    
    # Assigning a type to the variable 'maxy' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'maxy', max_call_result_300764)
    
    # Assigning a Tuple to a Name (line 152):
    
    # Assigning a Tuple to a Name (line 152):
    
    # Obtaining an instance of the builtin type 'tuple' (line 152)
    tuple_300765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 152)
    # Adding element type (line 152)
    
    # Obtaining an instance of the builtin type 'tuple' (line 152)
    tuple_300766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 152)
    # Adding element type (line 152)
    # Getting the type of 'minx' (line 152)
    minx_300767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'minx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 15), tuple_300766, minx_300767)
    # Adding element type (line 152)
    # Getting the type of 'miny' (line 152)
    miny_300768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 21), 'miny')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 15), tuple_300766, miny_300768)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 14), tuple_300765, tuple_300766)
    # Adding element type (line 152)
    
    # Obtaining an instance of the builtin type 'tuple' (line 152)
    tuple_300769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 152)
    # Adding element type (line 152)
    # Getting the type of 'maxx' (line 152)
    maxx_300770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 29), 'maxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 29), tuple_300769, maxx_300770)
    # Adding element type (line 152)
    # Getting the type of 'maxy' (line 152)
    maxy_300771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 35), 'maxy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 29), tuple_300769, maxy_300771)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 14), tuple_300765, tuple_300769)
    
    # Assigning a type to the variable 'corners' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'corners', tuple_300765)
    
    # Call to update_datalim(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'corners' (line 153)
    corners_300774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 22), 'corners', False)
    # Processing the call keyword arguments (line 153)
    kwargs_300775 = {}
    # Getting the type of 'ax' (line 153)
    ax_300772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'ax', False)
    # Obtaining the member 'update_datalim' of a type (line 153)
    update_datalim_300773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), ax_300772, 'update_datalim')
    # Calling update_datalim(args, kwargs) (line 153)
    update_datalim_call_result_300776 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), update_datalim_300773, *[corners_300774], **kwargs_300775)
    
    
    # Call to autoscale_view(...): (line 154)
    # Processing the call keyword arguments (line 154)
    kwargs_300779 = {}
    # Getting the type of 'ax' (line 154)
    ax_300777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'ax', False)
    # Obtaining the member 'autoscale_view' of a type (line 154)
    autoscale_view_300778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 4), ax_300777, 'autoscale_view')
    # Calling autoscale_view(args, kwargs) (line 154)
    autoscale_view_call_result_300780 = invoke(stypy.reporting.localization.Localization(__file__, 154, 4), autoscale_view_300778, *[], **kwargs_300779)
    
    
    # Call to add_collection(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'collection' (line 155)
    collection_300783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'collection', False)
    # Processing the call keyword arguments (line 155)
    kwargs_300784 = {}
    # Getting the type of 'ax' (line 155)
    ax_300781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'ax', False)
    # Obtaining the member 'add_collection' of a type (line 155)
    add_collection_300782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), ax_300781, 'add_collection')
    # Calling add_collection(args, kwargs) (line 155)
    add_collection_call_result_300785 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), add_collection_300782, *[collection_300783], **kwargs_300784)
    
    # Getting the type of 'collection' (line 156)
    collection_300786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'collection')
    # Assigning a type to the variable 'stypy_return_type' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type', collection_300786)
    
    # ################# End of 'tripcolor(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tripcolor' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_300787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_300787)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tripcolor'
    return stypy_return_type_300787

# Assigning a type to the variable 'tripcolor' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tripcolor', tripcolor)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
