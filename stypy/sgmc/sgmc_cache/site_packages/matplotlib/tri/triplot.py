
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import numpy as np
7: from matplotlib.tri.triangulation import Triangulation
8: 
9: 
10: def triplot(ax, *args, **kwargs):
11:     '''
12:     Draw a unstructured triangular grid as lines and/or markers.
13: 
14:     The triangulation to plot can be specified in one of two ways;
15:     either::
16: 
17:       triplot(triangulation, ...)
18: 
19:     where triangulation is a :class:`matplotlib.tri.Triangulation`
20:     object, or
21: 
22:     ::
23: 
24:       triplot(x, y, ...)
25:       triplot(x, y, triangles, ...)
26:       triplot(x, y, triangles=triangles, ...)
27:       triplot(x, y, mask=mask, ...)
28:       triplot(x, y, triangles, mask=mask, ...)
29: 
30:     in which case a Triangulation object will be created.  See
31:     :class:`~matplotlib.tri.Triangulation` for a explanation of these
32:     possibilities.
33: 
34:     The remaining args and kwargs are the same as for
35:     :meth:`~matplotlib.axes.Axes.plot`.
36: 
37:     Return a list of 2 :class:`~matplotlib.lines.Line2D` containing
38:     respectively:
39: 
40:         - the lines plotted for triangles edges
41:         - the markers plotted for triangles nodes
42:     '''
43:     import matplotlib.axes
44: 
45:     tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
46:     x, y, edges = (tri.x, tri.y, tri.edges)
47: 
48:     # Decode plot format string, e.g., 'ro-'
49:     fmt = ""
50:     if len(args) > 0:
51:         fmt = args[0]
52:     linestyle, marker, color = matplotlib.axes._base._process_plot_format(fmt)
53: 
54:     # Insert plot format string into a copy of kwargs (kwargs values prevail).
55:     kw = kwargs.copy()
56:     for key, val in zip(('linestyle', 'marker', 'color'),
57:                         (linestyle, marker, color)):
58:         if val is not None:
59:             kw[key] = kwargs.get(key, val)
60: 
61:     # Draw lines without markers.
62:     # Note 1: If we drew markers here, most markers would be drawn more than
63:     #         once as they belong to several edges.
64:     # Note 2: We insert nan values in the flattened edges arrays rather than
65:     #         plotting directly (triang.x[edges].T, triang.y[edges].T)
66:     #         as it considerably speeds-up code execution.
67:     linestyle = kw['linestyle']
68:     kw_lines = kw.copy()
69:     kw_lines['marker'] = 'None'  # No marker to draw.
70:     kw_lines['zorder'] = kw.get('zorder', 1)  # Path default zorder is used.
71:     if (linestyle is not None) and (linestyle not in ['None', '', ' ']):
72:         tri_lines_x = np.insert(x[edges], 2, np.nan, axis=1)
73:         tri_lines_y = np.insert(y[edges], 2, np.nan, axis=1)
74:         tri_lines = ax.plot(tri_lines_x.ravel(), tri_lines_y.ravel(),
75:                             **kw_lines)
76:     else:
77:         tri_lines = ax.plot([], [], **kw_lines)
78: 
79:     # Draw markers separately.
80:     marker = kw['marker']
81:     kw_markers = kw.copy()
82:     kw_markers['linestyle'] = 'None'  # No line to draw.
83:     if (marker is not None) and (marker not in ['None', '', ' ']):
84:         tri_markers = ax.plot(x, y, **kw_markers)
85:     else:
86:         tri_markers = ax.plot([], [], **kw_markers)
87: 
88:     return tri_lines + tri_markers
89: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_300799 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_300799) is not StypyTypeError):

    if (import_300799 != 'pyd_module'):
        __import__(import_300799)
        sys_modules_300800 = sys.modules[import_300799]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_300800.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_300799)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_300801 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_300801) is not StypyTypeError):

    if (import_300801 != 'pyd_module'):
        __import__(import_300801)
        sys_modules_300802 = sys.modules[import_300801]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_300802.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_300801)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from matplotlib.tri.triangulation import Triangulation' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_300803 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.tri.triangulation')

if (type(import_300803) is not StypyTypeError):

    if (import_300803 != 'pyd_module'):
        __import__(import_300803)
        sys_modules_300804 = sys.modules[import_300803]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.tri.triangulation', sys_modules_300804.module_type_store, module_type_store, ['Triangulation'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_300804, sys_modules_300804.module_type_store, module_type_store)
    else:
        from matplotlib.tri.triangulation import Triangulation

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.tri.triangulation', None, module_type_store, ['Triangulation'], [Triangulation])

else:
    # Assigning a type to the variable 'matplotlib.tri.triangulation' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.tri.triangulation', import_300803)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')


@norecursion
def triplot(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'triplot'
    module_type_store = module_type_store.open_function_context('triplot', 10, 0, False)
    
    # Passed parameters checking function
    triplot.stypy_localization = localization
    triplot.stypy_type_of_self = None
    triplot.stypy_type_store = module_type_store
    triplot.stypy_function_name = 'triplot'
    triplot.stypy_param_names_list = ['ax']
    triplot.stypy_varargs_param_name = 'args'
    triplot.stypy_kwargs_param_name = 'kwargs'
    triplot.stypy_call_defaults = defaults
    triplot.stypy_call_varargs = varargs
    triplot.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'triplot', ['ax'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'triplot', localization, ['ax'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'triplot(...)' code ##################

    unicode_300805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, (-1)), 'unicode', u'\n    Draw a unstructured triangular grid as lines and/or markers.\n\n    The triangulation to plot can be specified in one of two ways;\n    either::\n\n      triplot(triangulation, ...)\n\n    where triangulation is a :class:`matplotlib.tri.Triangulation`\n    object, or\n\n    ::\n\n      triplot(x, y, ...)\n      triplot(x, y, triangles, ...)\n      triplot(x, y, triangles=triangles, ...)\n      triplot(x, y, mask=mask, ...)\n      triplot(x, y, triangles, mask=mask, ...)\n\n    in which case a Triangulation object will be created.  See\n    :class:`~matplotlib.tri.Triangulation` for a explanation of these\n    possibilities.\n\n    The remaining args and kwargs are the same as for\n    :meth:`~matplotlib.axes.Axes.plot`.\n\n    Return a list of 2 :class:`~matplotlib.lines.Line2D` containing\n    respectively:\n\n        - the lines plotted for triangles edges\n        - the markers plotted for triangles nodes\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 43, 4))
    
    # 'import matplotlib.axes' statement (line 43)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
    import_300806 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 43, 4), 'matplotlib.axes')

    if (type(import_300806) is not StypyTypeError):

        if (import_300806 != 'pyd_module'):
            __import__(import_300806)
            sys_modules_300807 = sys.modules[import_300806]
            import_module(stypy.reporting.localization.Localization(__file__, 43, 4), 'matplotlib.axes', sys_modules_300807.module_type_store, module_type_store)
        else:
            import matplotlib.axes

            import_module(stypy.reporting.localization.Localization(__file__, 43, 4), 'matplotlib.axes', matplotlib.axes, module_type_store)

    else:
        # Assigning a type to the variable 'matplotlib.axes' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'matplotlib.axes', import_300806)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')
    
    
    # Assigning a Call to a Tuple (line 45):
    
    # Assigning a Call to a Name:
    
    # Call to get_from_args_and_kwargs(...): (line 45)
    # Getting the type of 'args' (line 45)
    args_300810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 64), 'args', False)
    # Processing the call keyword arguments (line 45)
    # Getting the type of 'kwargs' (line 45)
    kwargs_300811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 72), 'kwargs', False)
    kwargs_300812 = {'kwargs_300811': kwargs_300811}
    # Getting the type of 'Triangulation' (line 45)
    Triangulation_300808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'Triangulation', False)
    # Obtaining the member 'get_from_args_and_kwargs' of a type (line 45)
    get_from_args_and_kwargs_300809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 24), Triangulation_300808, 'get_from_args_and_kwargs')
    # Calling get_from_args_and_kwargs(args, kwargs) (line 45)
    get_from_args_and_kwargs_call_result_300813 = invoke(stypy.reporting.localization.Localization(__file__, 45, 24), get_from_args_and_kwargs_300809, *[args_300810], **kwargs_300812)
    
    # Assigning a type to the variable 'call_assignment_300788' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'call_assignment_300788', get_from_args_and_kwargs_call_result_300813)
    
    # Assigning a Call to a Name (line 45):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_300816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'int')
    # Processing the call keyword arguments
    kwargs_300817 = {}
    # Getting the type of 'call_assignment_300788' (line 45)
    call_assignment_300788_300814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'call_assignment_300788', False)
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___300815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 4), call_assignment_300788_300814, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_300818 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___300815, *[int_300816], **kwargs_300817)
    
    # Assigning a type to the variable 'call_assignment_300789' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'call_assignment_300789', getitem___call_result_300818)
    
    # Assigning a Name to a Name (line 45):
    # Getting the type of 'call_assignment_300789' (line 45)
    call_assignment_300789_300819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'call_assignment_300789')
    # Assigning a type to the variable 'tri' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'tri', call_assignment_300789_300819)
    
    # Assigning a Call to a Name (line 45):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_300822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'int')
    # Processing the call keyword arguments
    kwargs_300823 = {}
    # Getting the type of 'call_assignment_300788' (line 45)
    call_assignment_300788_300820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'call_assignment_300788', False)
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___300821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 4), call_assignment_300788_300820, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_300824 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___300821, *[int_300822], **kwargs_300823)
    
    # Assigning a type to the variable 'call_assignment_300790' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'call_assignment_300790', getitem___call_result_300824)
    
    # Assigning a Name to a Name (line 45):
    # Getting the type of 'call_assignment_300790' (line 45)
    call_assignment_300790_300825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'call_assignment_300790')
    # Assigning a type to the variable 'args' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 9), 'args', call_assignment_300790_300825)
    
    # Assigning a Call to a Name (line 45):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_300828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'int')
    # Processing the call keyword arguments
    kwargs_300829 = {}
    # Getting the type of 'call_assignment_300788' (line 45)
    call_assignment_300788_300826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'call_assignment_300788', False)
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___300827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 4), call_assignment_300788_300826, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_300830 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___300827, *[int_300828], **kwargs_300829)
    
    # Assigning a type to the variable 'call_assignment_300791' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'call_assignment_300791', getitem___call_result_300830)
    
    # Assigning a Name to a Name (line 45):
    # Getting the type of 'call_assignment_300791' (line 45)
    call_assignment_300791_300831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'call_assignment_300791')
    # Assigning a type to the variable 'kwargs' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'kwargs', call_assignment_300791_300831)
    
    # Assigning a Tuple to a Tuple (line 46):
    
    # Assigning a Attribute to a Name (line 46):
    # Getting the type of 'tri' (line 46)
    tri_300832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'tri')
    # Obtaining the member 'x' of a type (line 46)
    x_300833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 19), tri_300832, 'x')
    # Assigning a type to the variable 'tuple_assignment_300792' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_assignment_300792', x_300833)
    
    # Assigning a Attribute to a Name (line 46):
    # Getting the type of 'tri' (line 46)
    tri_300834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'tri')
    # Obtaining the member 'y' of a type (line 46)
    y_300835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 26), tri_300834, 'y')
    # Assigning a type to the variable 'tuple_assignment_300793' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_assignment_300793', y_300835)
    
    # Assigning a Attribute to a Name (line 46):
    # Getting the type of 'tri' (line 46)
    tri_300836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 33), 'tri')
    # Obtaining the member 'edges' of a type (line 46)
    edges_300837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 33), tri_300836, 'edges')
    # Assigning a type to the variable 'tuple_assignment_300794' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_assignment_300794', edges_300837)
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'tuple_assignment_300792' (line 46)
    tuple_assignment_300792_300838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_assignment_300792')
    # Assigning a type to the variable 'x' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'x', tuple_assignment_300792_300838)
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'tuple_assignment_300793' (line 46)
    tuple_assignment_300793_300839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_assignment_300793')
    # Assigning a type to the variable 'y' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 7), 'y', tuple_assignment_300793_300839)
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'tuple_assignment_300794' (line 46)
    tuple_assignment_300794_300840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tuple_assignment_300794')
    # Assigning a type to the variable 'edges' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 10), 'edges', tuple_assignment_300794_300840)
    
    # Assigning a Str to a Name (line 49):
    
    # Assigning a Str to a Name (line 49):
    unicode_300841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 10), 'unicode', u'')
    # Assigning a type to the variable 'fmt' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'fmt', unicode_300841)
    
    
    
    # Call to len(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'args' (line 50)
    args_300843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'args', False)
    # Processing the call keyword arguments (line 50)
    kwargs_300844 = {}
    # Getting the type of 'len' (line 50)
    len_300842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'len', False)
    # Calling len(args, kwargs) (line 50)
    len_call_result_300845 = invoke(stypy.reporting.localization.Localization(__file__, 50, 7), len_300842, *[args_300843], **kwargs_300844)
    
    int_300846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 19), 'int')
    # Applying the binary operator '>' (line 50)
    result_gt_300847 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 7), '>', len_call_result_300845, int_300846)
    
    # Testing the type of an if condition (line 50)
    if_condition_300848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 4), result_gt_300847)
    # Assigning a type to the variable 'if_condition_300848' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'if_condition_300848', if_condition_300848)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 51):
    
    # Assigning a Subscript to a Name (line 51):
    
    # Obtaining the type of the subscript
    int_300849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 19), 'int')
    # Getting the type of 'args' (line 51)
    args_300850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'args')
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___300851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 14), args_300850, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_300852 = invoke(stypy.reporting.localization.Localization(__file__, 51, 14), getitem___300851, int_300849)
    
    # Assigning a type to the variable 'fmt' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'fmt', subscript_call_result_300852)
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 52):
    
    # Assigning a Call to a Name:
    
    # Call to _process_plot_format(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'fmt' (line 52)
    fmt_300857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 74), 'fmt', False)
    # Processing the call keyword arguments (line 52)
    kwargs_300858 = {}
    # Getting the type of 'matplotlib' (line 52)
    matplotlib_300853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 31), 'matplotlib', False)
    # Obtaining the member 'axes' of a type (line 52)
    axes_300854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 31), matplotlib_300853, 'axes')
    # Obtaining the member '_base' of a type (line 52)
    _base_300855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 31), axes_300854, '_base')
    # Obtaining the member '_process_plot_format' of a type (line 52)
    _process_plot_format_300856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 31), _base_300855, '_process_plot_format')
    # Calling _process_plot_format(args, kwargs) (line 52)
    _process_plot_format_call_result_300859 = invoke(stypy.reporting.localization.Localization(__file__, 52, 31), _process_plot_format_300856, *[fmt_300857], **kwargs_300858)
    
    # Assigning a type to the variable 'call_assignment_300795' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_300795', _process_plot_format_call_result_300859)
    
    # Assigning a Call to a Name (line 52):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_300862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'int')
    # Processing the call keyword arguments
    kwargs_300863 = {}
    # Getting the type of 'call_assignment_300795' (line 52)
    call_assignment_300795_300860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_300795', False)
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___300861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), call_assignment_300795_300860, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_300864 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___300861, *[int_300862], **kwargs_300863)
    
    # Assigning a type to the variable 'call_assignment_300796' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_300796', getitem___call_result_300864)
    
    # Assigning a Name to a Name (line 52):
    # Getting the type of 'call_assignment_300796' (line 52)
    call_assignment_300796_300865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_300796')
    # Assigning a type to the variable 'linestyle' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'linestyle', call_assignment_300796_300865)
    
    # Assigning a Call to a Name (line 52):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_300868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'int')
    # Processing the call keyword arguments
    kwargs_300869 = {}
    # Getting the type of 'call_assignment_300795' (line 52)
    call_assignment_300795_300866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_300795', False)
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___300867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), call_assignment_300795_300866, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_300870 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___300867, *[int_300868], **kwargs_300869)
    
    # Assigning a type to the variable 'call_assignment_300797' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_300797', getitem___call_result_300870)
    
    # Assigning a Name to a Name (line 52):
    # Getting the type of 'call_assignment_300797' (line 52)
    call_assignment_300797_300871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_300797')
    # Assigning a type to the variable 'marker' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'marker', call_assignment_300797_300871)
    
    # Assigning a Call to a Name (line 52):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_300874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'int')
    # Processing the call keyword arguments
    kwargs_300875 = {}
    # Getting the type of 'call_assignment_300795' (line 52)
    call_assignment_300795_300872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_300795', False)
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___300873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), call_assignment_300795_300872, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_300876 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___300873, *[int_300874], **kwargs_300875)
    
    # Assigning a type to the variable 'call_assignment_300798' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_300798', getitem___call_result_300876)
    
    # Assigning a Name to a Name (line 52):
    # Getting the type of 'call_assignment_300798' (line 52)
    call_assignment_300798_300877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'call_assignment_300798')
    # Assigning a type to the variable 'color' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 23), 'color', call_assignment_300798_300877)
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to copy(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_300880 = {}
    # Getting the type of 'kwargs' (line 55)
    kwargs_300878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 9), 'kwargs', False)
    # Obtaining the member 'copy' of a type (line 55)
    copy_300879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 9), kwargs_300878, 'copy')
    # Calling copy(args, kwargs) (line 55)
    copy_call_result_300881 = invoke(stypy.reporting.localization.Localization(__file__, 55, 9), copy_300879, *[], **kwargs_300880)
    
    # Assigning a type to the variable 'kw' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'kw', copy_call_result_300881)
    
    
    # Call to zip(...): (line 56)
    # Processing the call arguments (line 56)
    
    # Obtaining an instance of the builtin type 'tuple' (line 56)
    tuple_300883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 56)
    # Adding element type (line 56)
    unicode_300884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'unicode', u'linestyle')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), tuple_300883, unicode_300884)
    # Adding element type (line 56)
    unicode_300885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 38), 'unicode', u'marker')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), tuple_300883, unicode_300885)
    # Adding element type (line 56)
    unicode_300886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 48), 'unicode', u'color')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), tuple_300883, unicode_300886)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 57)
    tuple_300887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 57)
    # Adding element type (line 57)
    # Getting the type of 'linestyle' (line 57)
    linestyle_300888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'linestyle', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 25), tuple_300887, linestyle_300888)
    # Adding element type (line 57)
    # Getting the type of 'marker' (line 57)
    marker_300889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 36), 'marker', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 25), tuple_300887, marker_300889)
    # Adding element type (line 57)
    # Getting the type of 'color' (line 57)
    color_300890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 44), 'color', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 25), tuple_300887, color_300890)
    
    # Processing the call keyword arguments (line 56)
    kwargs_300891 = {}
    # Getting the type of 'zip' (line 56)
    zip_300882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'zip', False)
    # Calling zip(args, kwargs) (line 56)
    zip_call_result_300892 = invoke(stypy.reporting.localization.Localization(__file__, 56, 20), zip_300882, *[tuple_300883, tuple_300887], **kwargs_300891)
    
    # Testing the type of a for loop iterable (line 56)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 56, 4), zip_call_result_300892)
    # Getting the type of the for loop variable (line 56)
    for_loop_var_300893 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 56, 4), zip_call_result_300892)
    # Assigning a type to the variable 'key' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 4), for_loop_var_300893))
    # Assigning a type to the variable 'val' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 4), for_loop_var_300893))
    # SSA begins for a for statement (line 56)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Type idiom detected: calculating its left and rigth part (line 58)
    # Getting the type of 'val' (line 58)
    val_300894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'val')
    # Getting the type of 'None' (line 58)
    None_300895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'None')
    
    (may_be_300896, more_types_in_union_300897) = may_not_be_none(val_300894, None_300895)

    if may_be_300896:

        if more_types_in_union_300897:
            # Runtime conditional SSA (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Subscript (line 59):
        
        # Assigning a Call to a Subscript (line 59):
        
        # Call to get(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'key' (line 59)
        key_300900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'key', False)
        # Getting the type of 'val' (line 59)
        val_300901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 38), 'val', False)
        # Processing the call keyword arguments (line 59)
        kwargs_300902 = {}
        # Getting the type of 'kwargs' (line 59)
        kwargs_300898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'kwargs', False)
        # Obtaining the member 'get' of a type (line 59)
        get_300899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 22), kwargs_300898, 'get')
        # Calling get(args, kwargs) (line 59)
        get_call_result_300903 = invoke(stypy.reporting.localization.Localization(__file__, 59, 22), get_300899, *[key_300900, val_300901], **kwargs_300902)
        
        # Getting the type of 'kw' (line 59)
        kw_300904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'kw')
        # Getting the type of 'key' (line 59)
        key_300905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'key')
        # Storing an element on a container (line 59)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 12), kw_300904, (key_300905, get_call_result_300903))

        if more_types_in_union_300897:
            # SSA join for if statement (line 58)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 67):
    
    # Assigning a Subscript to a Name (line 67):
    
    # Obtaining the type of the subscript
    unicode_300906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 19), 'unicode', u'linestyle')
    # Getting the type of 'kw' (line 67)
    kw_300907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'kw')
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___300908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 16), kw_300907, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_300909 = invoke(stypy.reporting.localization.Localization(__file__, 67, 16), getitem___300908, unicode_300906)
    
    # Assigning a type to the variable 'linestyle' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'linestyle', subscript_call_result_300909)
    
    # Assigning a Call to a Name (line 68):
    
    # Assigning a Call to a Name (line 68):
    
    # Call to copy(...): (line 68)
    # Processing the call keyword arguments (line 68)
    kwargs_300912 = {}
    # Getting the type of 'kw' (line 68)
    kw_300910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'kw', False)
    # Obtaining the member 'copy' of a type (line 68)
    copy_300911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 15), kw_300910, 'copy')
    # Calling copy(args, kwargs) (line 68)
    copy_call_result_300913 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), copy_300911, *[], **kwargs_300912)
    
    # Assigning a type to the variable 'kw_lines' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'kw_lines', copy_call_result_300913)
    
    # Assigning a Str to a Subscript (line 69):
    
    # Assigning a Str to a Subscript (line 69):
    unicode_300914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 25), 'unicode', u'None')
    # Getting the type of 'kw_lines' (line 69)
    kw_lines_300915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'kw_lines')
    unicode_300916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 13), 'unicode', u'marker')
    # Storing an element on a container (line 69)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 4), kw_lines_300915, (unicode_300916, unicode_300914))
    
    # Assigning a Call to a Subscript (line 70):
    
    # Assigning a Call to a Subscript (line 70):
    
    # Call to get(...): (line 70)
    # Processing the call arguments (line 70)
    unicode_300919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 32), 'unicode', u'zorder')
    int_300920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 42), 'int')
    # Processing the call keyword arguments (line 70)
    kwargs_300921 = {}
    # Getting the type of 'kw' (line 70)
    kw_300917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'kw', False)
    # Obtaining the member 'get' of a type (line 70)
    get_300918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 25), kw_300917, 'get')
    # Calling get(args, kwargs) (line 70)
    get_call_result_300922 = invoke(stypy.reporting.localization.Localization(__file__, 70, 25), get_300918, *[unicode_300919, int_300920], **kwargs_300921)
    
    # Getting the type of 'kw_lines' (line 70)
    kw_lines_300923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'kw_lines')
    unicode_300924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 13), 'unicode', u'zorder')
    # Storing an element on a container (line 70)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 4), kw_lines_300923, (unicode_300924, get_call_result_300922))
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'linestyle' (line 71)
    linestyle_300925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'linestyle')
    # Getting the type of 'None' (line 71)
    None_300926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'None')
    # Applying the binary operator 'isnot' (line 71)
    result_is_not_300927 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 8), 'isnot', linestyle_300925, None_300926)
    
    
    # Getting the type of 'linestyle' (line 71)
    linestyle_300928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 36), 'linestyle')
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_300929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    # Adding element type (line 71)
    unicode_300930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 54), 'unicode', u'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 53), list_300929, unicode_300930)
    # Adding element type (line 71)
    unicode_300931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 62), 'unicode', u'')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 53), list_300929, unicode_300931)
    # Adding element type (line 71)
    unicode_300932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 66), 'unicode', u' ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 53), list_300929, unicode_300932)
    
    # Applying the binary operator 'notin' (line 71)
    result_contains_300933 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 36), 'notin', linestyle_300928, list_300929)
    
    # Applying the binary operator 'and' (line 71)
    result_and_keyword_300934 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 7), 'and', result_is_not_300927, result_contains_300933)
    
    # Testing the type of an if condition (line 71)
    if_condition_300935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 4), result_and_keyword_300934)
    # Assigning a type to the variable 'if_condition_300935' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'if_condition_300935', if_condition_300935)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 72):
    
    # Assigning a Call to a Name (line 72):
    
    # Call to insert(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Obtaining the type of the subscript
    # Getting the type of 'edges' (line 72)
    edges_300938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 34), 'edges', False)
    # Getting the type of 'x' (line 72)
    x_300939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___300940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 32), x_300939, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_300941 = invoke(stypy.reporting.localization.Localization(__file__, 72, 32), getitem___300940, edges_300938)
    
    int_300942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 42), 'int')
    # Getting the type of 'np' (line 72)
    np_300943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 45), 'np', False)
    # Obtaining the member 'nan' of a type (line 72)
    nan_300944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 45), np_300943, 'nan')
    # Processing the call keyword arguments (line 72)
    int_300945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 58), 'int')
    keyword_300946 = int_300945
    kwargs_300947 = {'axis': keyword_300946}
    # Getting the type of 'np' (line 72)
    np_300936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'np', False)
    # Obtaining the member 'insert' of a type (line 72)
    insert_300937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 22), np_300936, 'insert')
    # Calling insert(args, kwargs) (line 72)
    insert_call_result_300948 = invoke(stypy.reporting.localization.Localization(__file__, 72, 22), insert_300937, *[subscript_call_result_300941, int_300942, nan_300944], **kwargs_300947)
    
    # Assigning a type to the variable 'tri_lines_x' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'tri_lines_x', insert_call_result_300948)
    
    # Assigning a Call to a Name (line 73):
    
    # Assigning a Call to a Name (line 73):
    
    # Call to insert(...): (line 73)
    # Processing the call arguments (line 73)
    
    # Obtaining the type of the subscript
    # Getting the type of 'edges' (line 73)
    edges_300951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 34), 'edges', False)
    # Getting the type of 'y' (line 73)
    y_300952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 32), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___300953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 32), y_300952, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
    subscript_call_result_300954 = invoke(stypy.reporting.localization.Localization(__file__, 73, 32), getitem___300953, edges_300951)
    
    int_300955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 42), 'int')
    # Getting the type of 'np' (line 73)
    np_300956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 45), 'np', False)
    # Obtaining the member 'nan' of a type (line 73)
    nan_300957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 45), np_300956, 'nan')
    # Processing the call keyword arguments (line 73)
    int_300958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 58), 'int')
    keyword_300959 = int_300958
    kwargs_300960 = {'axis': keyword_300959}
    # Getting the type of 'np' (line 73)
    np_300949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 22), 'np', False)
    # Obtaining the member 'insert' of a type (line 73)
    insert_300950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 22), np_300949, 'insert')
    # Calling insert(args, kwargs) (line 73)
    insert_call_result_300961 = invoke(stypy.reporting.localization.Localization(__file__, 73, 22), insert_300950, *[subscript_call_result_300954, int_300955, nan_300957], **kwargs_300960)
    
    # Assigning a type to the variable 'tri_lines_y' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tri_lines_y', insert_call_result_300961)
    
    # Assigning a Call to a Name (line 74):
    
    # Assigning a Call to a Name (line 74):
    
    # Call to plot(...): (line 74)
    # Processing the call arguments (line 74)
    
    # Call to ravel(...): (line 74)
    # Processing the call keyword arguments (line 74)
    kwargs_300966 = {}
    # Getting the type of 'tri_lines_x' (line 74)
    tri_lines_x_300964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'tri_lines_x', False)
    # Obtaining the member 'ravel' of a type (line 74)
    ravel_300965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 28), tri_lines_x_300964, 'ravel')
    # Calling ravel(args, kwargs) (line 74)
    ravel_call_result_300967 = invoke(stypy.reporting.localization.Localization(__file__, 74, 28), ravel_300965, *[], **kwargs_300966)
    
    
    # Call to ravel(...): (line 74)
    # Processing the call keyword arguments (line 74)
    kwargs_300970 = {}
    # Getting the type of 'tri_lines_y' (line 74)
    tri_lines_y_300968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 49), 'tri_lines_y', False)
    # Obtaining the member 'ravel' of a type (line 74)
    ravel_300969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 49), tri_lines_y_300968, 'ravel')
    # Calling ravel(args, kwargs) (line 74)
    ravel_call_result_300971 = invoke(stypy.reporting.localization.Localization(__file__, 74, 49), ravel_300969, *[], **kwargs_300970)
    
    # Processing the call keyword arguments (line 74)
    # Getting the type of 'kw_lines' (line 75)
    kw_lines_300972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), 'kw_lines', False)
    kwargs_300973 = {'kw_lines_300972': kw_lines_300972}
    # Getting the type of 'ax' (line 74)
    ax_300962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'ax', False)
    # Obtaining the member 'plot' of a type (line 74)
    plot_300963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 20), ax_300962, 'plot')
    # Calling plot(args, kwargs) (line 74)
    plot_call_result_300974 = invoke(stypy.reporting.localization.Localization(__file__, 74, 20), plot_300963, *[ravel_call_result_300967, ravel_call_result_300971], **kwargs_300973)
    
    # Assigning a type to the variable 'tri_lines' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'tri_lines', plot_call_result_300974)
    # SSA branch for the else part of an if statement (line 71)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to plot(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Obtaining an instance of the builtin type 'list' (line 77)
    list_300977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 77)
    
    
    # Obtaining an instance of the builtin type 'list' (line 77)
    list_300978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 77)
    
    # Processing the call keyword arguments (line 77)
    # Getting the type of 'kw_lines' (line 77)
    kw_lines_300979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'kw_lines', False)
    kwargs_300980 = {'kw_lines_300979': kw_lines_300979}
    # Getting the type of 'ax' (line 77)
    ax_300975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'ax', False)
    # Obtaining the member 'plot' of a type (line 77)
    plot_300976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 20), ax_300975, 'plot')
    # Calling plot(args, kwargs) (line 77)
    plot_call_result_300981 = invoke(stypy.reporting.localization.Localization(__file__, 77, 20), plot_300976, *[list_300977, list_300978], **kwargs_300980)
    
    # Assigning a type to the variable 'tri_lines' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'tri_lines', plot_call_result_300981)
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 80):
    
    # Assigning a Subscript to a Name (line 80):
    
    # Obtaining the type of the subscript
    unicode_300982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 16), 'unicode', u'marker')
    # Getting the type of 'kw' (line 80)
    kw_300983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'kw')
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___300984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 13), kw_300983, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_300985 = invoke(stypy.reporting.localization.Localization(__file__, 80, 13), getitem___300984, unicode_300982)
    
    # Assigning a type to the variable 'marker' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'marker', subscript_call_result_300985)
    
    # Assigning a Call to a Name (line 81):
    
    # Assigning a Call to a Name (line 81):
    
    # Call to copy(...): (line 81)
    # Processing the call keyword arguments (line 81)
    kwargs_300988 = {}
    # Getting the type of 'kw' (line 81)
    kw_300986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'kw', False)
    # Obtaining the member 'copy' of a type (line 81)
    copy_300987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 17), kw_300986, 'copy')
    # Calling copy(args, kwargs) (line 81)
    copy_call_result_300989 = invoke(stypy.reporting.localization.Localization(__file__, 81, 17), copy_300987, *[], **kwargs_300988)
    
    # Assigning a type to the variable 'kw_markers' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'kw_markers', copy_call_result_300989)
    
    # Assigning a Str to a Subscript (line 82):
    
    # Assigning a Str to a Subscript (line 82):
    unicode_300990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 30), 'unicode', u'None')
    # Getting the type of 'kw_markers' (line 82)
    kw_markers_300991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'kw_markers')
    unicode_300992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 15), 'unicode', u'linestyle')
    # Storing an element on a container (line 82)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 4), kw_markers_300991, (unicode_300992, unicode_300990))
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'marker' (line 83)
    marker_300993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'marker')
    # Getting the type of 'None' (line 83)
    None_300994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'None')
    # Applying the binary operator 'isnot' (line 83)
    result_is_not_300995 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 8), 'isnot', marker_300993, None_300994)
    
    
    # Getting the type of 'marker' (line 83)
    marker_300996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 33), 'marker')
    
    # Obtaining an instance of the builtin type 'list' (line 83)
    list_300997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 47), 'list')
    # Adding type elements to the builtin type 'list' instance (line 83)
    # Adding element type (line 83)
    unicode_300998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 48), 'unicode', u'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 47), list_300997, unicode_300998)
    # Adding element type (line 83)
    unicode_300999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 56), 'unicode', u'')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 47), list_300997, unicode_300999)
    # Adding element type (line 83)
    unicode_301000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 60), 'unicode', u' ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 47), list_300997, unicode_301000)
    
    # Applying the binary operator 'notin' (line 83)
    result_contains_301001 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 33), 'notin', marker_300996, list_300997)
    
    # Applying the binary operator 'and' (line 83)
    result_and_keyword_301002 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 7), 'and', result_is_not_300995, result_contains_301001)
    
    # Testing the type of an if condition (line 83)
    if_condition_301003 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 4), result_and_keyword_301002)
    # Assigning a type to the variable 'if_condition_301003' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'if_condition_301003', if_condition_301003)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 84):
    
    # Assigning a Call to a Name (line 84):
    
    # Call to plot(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'x' (line 84)
    x_301006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'x', False)
    # Getting the type of 'y' (line 84)
    y_301007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 33), 'y', False)
    # Processing the call keyword arguments (line 84)
    # Getting the type of 'kw_markers' (line 84)
    kw_markers_301008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 38), 'kw_markers', False)
    kwargs_301009 = {'kw_markers_301008': kw_markers_301008}
    # Getting the type of 'ax' (line 84)
    ax_301004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 22), 'ax', False)
    # Obtaining the member 'plot' of a type (line 84)
    plot_301005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 22), ax_301004, 'plot')
    # Calling plot(args, kwargs) (line 84)
    plot_call_result_301010 = invoke(stypy.reporting.localization.Localization(__file__, 84, 22), plot_301005, *[x_301006, y_301007], **kwargs_301009)
    
    # Assigning a type to the variable 'tri_markers' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tri_markers', plot_call_result_301010)
    # SSA branch for the else part of an if statement (line 83)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to plot(...): (line 86)
    # Processing the call arguments (line 86)
    
    # Obtaining an instance of the builtin type 'list' (line 86)
    list_301013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    
    
    # Obtaining an instance of the builtin type 'list' (line 86)
    list_301014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    
    # Processing the call keyword arguments (line 86)
    # Getting the type of 'kw_markers' (line 86)
    kw_markers_301015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 40), 'kw_markers', False)
    kwargs_301016 = {'kw_markers_301015': kw_markers_301015}
    # Getting the type of 'ax' (line 86)
    ax_301011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 22), 'ax', False)
    # Obtaining the member 'plot' of a type (line 86)
    plot_301012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 22), ax_301011, 'plot')
    # Calling plot(args, kwargs) (line 86)
    plot_call_result_301017 = invoke(stypy.reporting.localization.Localization(__file__, 86, 22), plot_301012, *[list_301013, list_301014], **kwargs_301016)
    
    # Assigning a type to the variable 'tri_markers' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tri_markers', plot_call_result_301017)
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'tri_lines' (line 88)
    tri_lines_301018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'tri_lines')
    # Getting the type of 'tri_markers' (line 88)
    tri_markers_301019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'tri_markers')
    # Applying the binary operator '+' (line 88)
    result_add_301020 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 11), '+', tri_lines_301018, tri_markers_301019)
    
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type', result_add_301020)
    
    # ################# End of 'triplot(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'triplot' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_301021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_301021)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'triplot'
    return stypy_return_type_301021

# Assigning a type to the variable 'triplot' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'triplot', triplot)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
