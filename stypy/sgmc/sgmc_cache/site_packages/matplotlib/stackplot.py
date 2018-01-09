
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Stacked area plot for 1D arrays inspired by Douglas Y'barbo's stackoverflow
3: answer:
4: http://stackoverflow.com/questions/2225995/how-can-i-create-stacked-line-graph-with-matplotlib
5: 
6: (http://stackoverflow.com/users/66549/doug)
7: 
8: '''
9: from __future__ import (absolute_import, division, print_function,
10:                         unicode_literals)
11: 
12: import six
13: from six.moves import xrange
14: 
15: from cycler import cycler
16: import numpy as np
17: 
18: __all__ = ['stackplot']
19: 
20: 
21: def stackplot(axes, x, *args, **kwargs):
22:     '''Draws a stacked area plot.
23: 
24:     *x* : 1d array of dimension N
25: 
26:     *y* : 2d array of dimension MxN, OR any number 1d arrays each of dimension
27:           1xN. The data is assumed to be unstacked. Each of the following
28:           calls is legal::
29: 
30:             stackplot(x, y)               # where y is MxN
31:             stackplot(x, y1, y2, y3, y4)  # where y1, y2, y3, y4, are all 1xNm
32: 
33:     Keyword arguments:
34: 
35:     *baseline* : ['zero', 'sym', 'wiggle', 'weighted_wiggle']
36:                 Method used to calculate the baseline. 'zero' is just a
37:                 simple stacked plot. 'sym' is symmetric around zero and
38:                 is sometimes called `ThemeRiver`.  'wiggle' minimizes the
39:                 sum of the squared slopes. 'weighted_wiggle' does the
40:                 same but weights to account for size of each layer.
41:                 It is also called `Streamgraph`-layout. More details
42:                 can be found at http://leebyron.com/streamgraph/.
43: 
44: 
45:     *labels* : A list or tuple of labels to assign to each data series.
46: 
47: 
48:     *colors* : A list or tuple of colors. These will be cycled through and
49:                used to colour the stacked areas.
50:                All other keyword arguments are passed to
51:                :func:`~matplotlib.Axes.fill_between`
52: 
53:     Returns *r* : A list of
54:     :class:`~matplotlib.collections.PolyCollection`, one for each
55:     element in the stacked area plot.
56:     '''
57: 
58:     y = np.row_stack(args)
59: 
60:     labels = iter(kwargs.pop('labels', []))
61: 
62:     colors = kwargs.pop('colors', None)
63:     if colors is not None:
64:         axes.set_prop_cycle(cycler('color', colors))
65: 
66:     baseline = kwargs.pop('baseline', 'zero')
67:     # Assume data passed has not been 'stacked', so stack it here.
68:     # We'll need a float buffer for the upcoming calculations.
69:     stack = np.cumsum(y, axis=0, dtype=np.promote_types(y.dtype, np.float32))
70: 
71:     if baseline == 'zero':
72:         first_line = 0.
73: 
74:     elif baseline == 'sym':
75:         first_line = -np.sum(y, 0) * 0.5
76:         stack += first_line[None, :]
77: 
78:     elif baseline == 'wiggle':
79:         m = y.shape[0]
80:         first_line = (y * (m - 0.5 - np.arange(m)[:, None])).sum(0)
81:         first_line /= -m
82:         stack += first_line
83: 
84:     elif baseline == 'weighted_wiggle':
85:         m, n = y.shape
86:         total = np.sum(y, 0)
87:         # multiply by 1/total (or zero) to avoid infinities in the division:
88:         inv_total = np.zeros_like(total)
89:         mask = total > 0
90:         inv_total[mask] = 1.0 / total[mask]
91:         increase = np.hstack((y[:, 0:1], np.diff(y)))
92:         below_size = total - stack
93:         below_size += 0.5 * y
94:         move_up = below_size * inv_total
95:         move_up[:, 0] = 0.5
96:         center = (move_up - 0.5) * increase
97:         center = np.cumsum(center.sum(0))
98:         first_line = center - 0.5 * total
99:         stack += first_line
100: 
101:     else:
102:         errstr = "Baseline method %s not recognised. " % baseline
103:         errstr += "Expected 'zero', 'sym', 'wiggle' or 'weighted_wiggle'"
104:         raise ValueError(errstr)
105: 
106:     # Color between x = 0 and the first array.
107:     color = axes._get_lines.get_next_color()
108:     coll = axes.fill_between(x, first_line, stack[0, :],
109:                              facecolor=color, label=six.next(labels, None),
110:                              **kwargs)
111:     coll.sticky_edges.y[:] = [0]
112:     r = [coll]
113: 
114:     # Color between array i-1 and array i
115:     for i in xrange(len(y) - 1):
116:         color = axes._get_lines.get_next_color()
117:         r.append(axes.fill_between(x, stack[i, :], stack[i + 1, :],
118:                                    facecolor=color,
119:                                    label= six.next(labels, None),
120:                                    **kwargs))
121:     return r
122: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_132856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'unicode', u"\nStacked area plot for 1D arrays inspired by Douglas Y'barbo's stackoverflow\nanswer:\nhttp://stackoverflow.com/questions/2225995/how-can-i-create-stacked-line-graph-with-matplotlib\n\n(http://stackoverflow.com/users/66549/doug)\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import six' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_132857 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'six')

if (type(import_132857) is not StypyTypeError):

    if (import_132857 != 'pyd_module'):
        __import__(import_132857)
        sys_modules_132858 = sys.modules[import_132857]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'six', sys_modules_132858.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'six', import_132857)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from six.moves import xrange' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_132859 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'six.moves')

if (type(import_132859) is not StypyTypeError):

    if (import_132859 != 'pyd_module'):
        __import__(import_132859)
        sys_modules_132860 = sys.modules[import_132859]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'six.moves', sys_modules_132860.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_132860, sys_modules_132860.module_type_store, module_type_store)
    else:
        from six.moves import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'six.moves', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'six.moves' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'six.moves', import_132859)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from cycler import cycler' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_132861 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'cycler')

if (type(import_132861) is not StypyTypeError):

    if (import_132861 != 'pyd_module'):
        __import__(import_132861)
        sys_modules_132862 = sys.modules[import_132861]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'cycler', sys_modules_132862.module_type_store, module_type_store, ['cycler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_132862, sys_modules_132862.module_type_store, module_type_store)
    else:
        from cycler import cycler

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'cycler', None, module_type_store, ['cycler'], [cycler])

else:
    # Assigning a type to the variable 'cycler' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'cycler', import_132861)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import numpy' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_132863 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy')

if (type(import_132863) is not StypyTypeError):

    if (import_132863 != 'pyd_module'):
        __import__(import_132863)
        sys_modules_132864 = sys.modules[import_132863]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'np', sys_modules_132864.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy', import_132863)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')


# Assigning a List to a Name (line 18):

# Assigning a List to a Name (line 18):
__all__ = [u'stackplot']
module_type_store.set_exportable_members([u'stackplot'])

# Obtaining an instance of the builtin type 'list' (line 18)
list_132865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
unicode_132866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 11), 'unicode', u'stackplot')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), list_132865, unicode_132866)

# Assigning a type to the variable '__all__' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '__all__', list_132865)

@norecursion
def stackplot(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'stackplot'
    module_type_store = module_type_store.open_function_context('stackplot', 21, 0, False)
    
    # Passed parameters checking function
    stackplot.stypy_localization = localization
    stackplot.stypy_type_of_self = None
    stackplot.stypy_type_store = module_type_store
    stackplot.stypy_function_name = 'stackplot'
    stackplot.stypy_param_names_list = ['axes', 'x']
    stackplot.stypy_varargs_param_name = 'args'
    stackplot.stypy_kwargs_param_name = 'kwargs'
    stackplot.stypy_call_defaults = defaults
    stackplot.stypy_call_varargs = varargs
    stackplot.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'stackplot', ['axes', 'x'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'stackplot', localization, ['axes', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'stackplot(...)' code ##################

    unicode_132867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'unicode', u"Draws a stacked area plot.\n\n    *x* : 1d array of dimension N\n\n    *y* : 2d array of dimension MxN, OR any number 1d arrays each of dimension\n          1xN. The data is assumed to be unstacked. Each of the following\n          calls is legal::\n\n            stackplot(x, y)               # where y is MxN\n            stackplot(x, y1, y2, y3, y4)  # where y1, y2, y3, y4, are all 1xNm\n\n    Keyword arguments:\n\n    *baseline* : ['zero', 'sym', 'wiggle', 'weighted_wiggle']\n                Method used to calculate the baseline. 'zero' is just a\n                simple stacked plot. 'sym' is symmetric around zero and\n                is sometimes called `ThemeRiver`.  'wiggle' minimizes the\n                sum of the squared slopes. 'weighted_wiggle' does the\n                same but weights to account for size of each layer.\n                It is also called `Streamgraph`-layout. More details\n                can be found at http://leebyron.com/streamgraph/.\n\n\n    *labels* : A list or tuple of labels to assign to each data series.\n\n\n    *colors* : A list or tuple of colors. These will be cycled through and\n               used to colour the stacked areas.\n               All other keyword arguments are passed to\n               :func:`~matplotlib.Axes.fill_between`\n\n    Returns *r* : A list of\n    :class:`~matplotlib.collections.PolyCollection`, one for each\n    element in the stacked area plot.\n    ")
    
    # Assigning a Call to a Name (line 58):
    
    # Assigning a Call to a Name (line 58):
    
    # Call to row_stack(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'args' (line 58)
    args_132870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'args', False)
    # Processing the call keyword arguments (line 58)
    kwargs_132871 = {}
    # Getting the type of 'np' (line 58)
    np_132868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'np', False)
    # Obtaining the member 'row_stack' of a type (line 58)
    row_stack_132869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), np_132868, 'row_stack')
    # Calling row_stack(args, kwargs) (line 58)
    row_stack_call_result_132872 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), row_stack_132869, *[args_132870], **kwargs_132871)
    
    # Assigning a type to the variable 'y' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'y', row_stack_call_result_132872)
    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to iter(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Call to pop(...): (line 60)
    # Processing the call arguments (line 60)
    unicode_132876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 29), 'unicode', u'labels')
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_132877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    
    # Processing the call keyword arguments (line 60)
    kwargs_132878 = {}
    # Getting the type of 'kwargs' (line 60)
    kwargs_132874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 60)
    pop_132875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 18), kwargs_132874, 'pop')
    # Calling pop(args, kwargs) (line 60)
    pop_call_result_132879 = invoke(stypy.reporting.localization.Localization(__file__, 60, 18), pop_132875, *[unicode_132876, list_132877], **kwargs_132878)
    
    # Processing the call keyword arguments (line 60)
    kwargs_132880 = {}
    # Getting the type of 'iter' (line 60)
    iter_132873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 13), 'iter', False)
    # Calling iter(args, kwargs) (line 60)
    iter_call_result_132881 = invoke(stypy.reporting.localization.Localization(__file__, 60, 13), iter_132873, *[pop_call_result_132879], **kwargs_132880)
    
    # Assigning a type to the variable 'labels' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'labels', iter_call_result_132881)
    
    # Assigning a Call to a Name (line 62):
    
    # Assigning a Call to a Name (line 62):
    
    # Call to pop(...): (line 62)
    # Processing the call arguments (line 62)
    unicode_132884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'unicode', u'colors')
    # Getting the type of 'None' (line 62)
    None_132885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 34), 'None', False)
    # Processing the call keyword arguments (line 62)
    kwargs_132886 = {}
    # Getting the type of 'kwargs' (line 62)
    kwargs_132882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 13), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 62)
    pop_132883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 13), kwargs_132882, 'pop')
    # Calling pop(args, kwargs) (line 62)
    pop_call_result_132887 = invoke(stypy.reporting.localization.Localization(__file__, 62, 13), pop_132883, *[unicode_132884, None_132885], **kwargs_132886)
    
    # Assigning a type to the variable 'colors' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'colors', pop_call_result_132887)
    
    # Type idiom detected: calculating its left and rigth part (line 63)
    # Getting the type of 'colors' (line 63)
    colors_132888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'colors')
    # Getting the type of 'None' (line 63)
    None_132889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 21), 'None')
    
    (may_be_132890, more_types_in_union_132891) = may_not_be_none(colors_132888, None_132889)

    if may_be_132890:

        if more_types_in_union_132891:
            # Runtime conditional SSA (line 63)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to set_prop_cycle(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to cycler(...): (line 64)
        # Processing the call arguments (line 64)
        unicode_132895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 35), 'unicode', u'color')
        # Getting the type of 'colors' (line 64)
        colors_132896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 44), 'colors', False)
        # Processing the call keyword arguments (line 64)
        kwargs_132897 = {}
        # Getting the type of 'cycler' (line 64)
        cycler_132894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'cycler', False)
        # Calling cycler(args, kwargs) (line 64)
        cycler_call_result_132898 = invoke(stypy.reporting.localization.Localization(__file__, 64, 28), cycler_132894, *[unicode_132895, colors_132896], **kwargs_132897)
        
        # Processing the call keyword arguments (line 64)
        kwargs_132899 = {}
        # Getting the type of 'axes' (line 64)
        axes_132892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'axes', False)
        # Obtaining the member 'set_prop_cycle' of a type (line 64)
        set_prop_cycle_132893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), axes_132892, 'set_prop_cycle')
        # Calling set_prop_cycle(args, kwargs) (line 64)
        set_prop_cycle_call_result_132900 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), set_prop_cycle_132893, *[cycler_call_result_132898], **kwargs_132899)
        

        if more_types_in_union_132891:
            # SSA join for if statement (line 63)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 66):
    
    # Assigning a Call to a Name (line 66):
    
    # Call to pop(...): (line 66)
    # Processing the call arguments (line 66)
    unicode_132903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 26), 'unicode', u'baseline')
    unicode_132904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 38), 'unicode', u'zero')
    # Processing the call keyword arguments (line 66)
    kwargs_132905 = {}
    # Getting the type of 'kwargs' (line 66)
    kwargs_132901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 15), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 66)
    pop_132902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 15), kwargs_132901, 'pop')
    # Calling pop(args, kwargs) (line 66)
    pop_call_result_132906 = invoke(stypy.reporting.localization.Localization(__file__, 66, 15), pop_132902, *[unicode_132903, unicode_132904], **kwargs_132905)
    
    # Assigning a type to the variable 'baseline' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'baseline', pop_call_result_132906)
    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to cumsum(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'y' (line 69)
    y_132909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'y', False)
    # Processing the call keyword arguments (line 69)
    int_132910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 30), 'int')
    keyword_132911 = int_132910
    
    # Call to promote_types(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'y' (line 69)
    y_132914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 56), 'y', False)
    # Obtaining the member 'dtype' of a type (line 69)
    dtype_132915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 56), y_132914, 'dtype')
    # Getting the type of 'np' (line 69)
    np_132916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 65), 'np', False)
    # Obtaining the member 'float32' of a type (line 69)
    float32_132917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 65), np_132916, 'float32')
    # Processing the call keyword arguments (line 69)
    kwargs_132918 = {}
    # Getting the type of 'np' (line 69)
    np_132912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 39), 'np', False)
    # Obtaining the member 'promote_types' of a type (line 69)
    promote_types_132913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 39), np_132912, 'promote_types')
    # Calling promote_types(args, kwargs) (line 69)
    promote_types_call_result_132919 = invoke(stypy.reporting.localization.Localization(__file__, 69, 39), promote_types_132913, *[dtype_132915, float32_132917], **kwargs_132918)
    
    keyword_132920 = promote_types_call_result_132919
    kwargs_132921 = {'dtype': keyword_132920, 'axis': keyword_132911}
    # Getting the type of 'np' (line 69)
    np_132907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'np', False)
    # Obtaining the member 'cumsum' of a type (line 69)
    cumsum_132908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), np_132907, 'cumsum')
    # Calling cumsum(args, kwargs) (line 69)
    cumsum_call_result_132922 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), cumsum_132908, *[y_132909], **kwargs_132921)
    
    # Assigning a type to the variable 'stack' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stack', cumsum_call_result_132922)
    
    
    # Getting the type of 'baseline' (line 71)
    baseline_132923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 7), 'baseline')
    unicode_132924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 19), 'unicode', u'zero')
    # Applying the binary operator '==' (line 71)
    result_eq_132925 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 7), '==', baseline_132923, unicode_132924)
    
    # Testing the type of an if condition (line 71)
    if_condition_132926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 4), result_eq_132925)
    # Assigning a type to the variable 'if_condition_132926' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'if_condition_132926', if_condition_132926)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 72):
    
    # Assigning a Num to a Name (line 72):
    float_132927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 21), 'float')
    # Assigning a type to the variable 'first_line' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'first_line', float_132927)
    # SSA branch for the else part of an if statement (line 71)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'baseline' (line 74)
    baseline_132928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 9), 'baseline')
    unicode_132929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 21), 'unicode', u'sym')
    # Applying the binary operator '==' (line 74)
    result_eq_132930 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 9), '==', baseline_132928, unicode_132929)
    
    # Testing the type of an if condition (line 74)
    if_condition_132931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 9), result_eq_132930)
    # Assigning a type to the variable 'if_condition_132931' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 9), 'if_condition_132931', if_condition_132931)
    # SSA begins for if statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 75):
    
    # Assigning a BinOp to a Name (line 75):
    
    
    # Call to sum(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'y' (line 75)
    y_132934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 29), 'y', False)
    int_132935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 32), 'int')
    # Processing the call keyword arguments (line 75)
    kwargs_132936 = {}
    # Getting the type of 'np' (line 75)
    np_132932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'np', False)
    # Obtaining the member 'sum' of a type (line 75)
    sum_132933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 22), np_132932, 'sum')
    # Calling sum(args, kwargs) (line 75)
    sum_call_result_132937 = invoke(stypy.reporting.localization.Localization(__file__, 75, 22), sum_132933, *[y_132934, int_132935], **kwargs_132936)
    
    # Applying the 'usub' unary operator (line 75)
    result___neg___132938 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 21), 'usub', sum_call_result_132937)
    
    float_132939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 37), 'float')
    # Applying the binary operator '*' (line 75)
    result_mul_132940 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 21), '*', result___neg___132938, float_132939)
    
    # Assigning a type to the variable 'first_line' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'first_line', result_mul_132940)
    
    # Getting the type of 'stack' (line 76)
    stack_132941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stack')
    
    # Obtaining the type of the subscript
    # Getting the type of 'None' (line 76)
    None_132942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 28), 'None')
    slice_132943 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 76, 17), None, None, None)
    # Getting the type of 'first_line' (line 76)
    first_line_132944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'first_line')
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___132945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 17), first_line_132944, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_132946 = invoke(stypy.reporting.localization.Localization(__file__, 76, 17), getitem___132945, (None_132942, slice_132943))
    
    # Applying the binary operator '+=' (line 76)
    result_iadd_132947 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 8), '+=', stack_132941, subscript_call_result_132946)
    # Assigning a type to the variable 'stack' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stack', result_iadd_132947)
    
    # SSA branch for the else part of an if statement (line 74)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'baseline' (line 78)
    baseline_132948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 9), 'baseline')
    unicode_132949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 21), 'unicode', u'wiggle')
    # Applying the binary operator '==' (line 78)
    result_eq_132950 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 9), '==', baseline_132948, unicode_132949)
    
    # Testing the type of an if condition (line 78)
    if_condition_132951 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 9), result_eq_132950)
    # Assigning a type to the variable 'if_condition_132951' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 9), 'if_condition_132951', if_condition_132951)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 79):
    
    # Assigning a Subscript to a Name (line 79):
    
    # Obtaining the type of the subscript
    int_132952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 20), 'int')
    # Getting the type of 'y' (line 79)
    y_132953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'y')
    # Obtaining the member 'shape' of a type (line 79)
    shape_132954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), y_132953, 'shape')
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___132955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), shape_132954, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_132956 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), getitem___132955, int_132952)
    
    # Assigning a type to the variable 'm' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'm', subscript_call_result_132956)
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to sum(...): (line 80)
    # Processing the call arguments (line 80)
    int_132973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 65), 'int')
    # Processing the call keyword arguments (line 80)
    kwargs_132974 = {}
    # Getting the type of 'y' (line 80)
    y_132957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'y', False)
    # Getting the type of 'm' (line 80)
    m_132958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'm', False)
    float_132959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 31), 'float')
    # Applying the binary operator '-' (line 80)
    result_sub_132960 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 27), '-', m_132958, float_132959)
    
    
    # Obtaining the type of the subscript
    slice_132961 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 80, 37), None, None, None)
    # Getting the type of 'None' (line 80)
    None_132962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 53), 'None', False)
    
    # Call to arange(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'm' (line 80)
    m_132965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 47), 'm', False)
    # Processing the call keyword arguments (line 80)
    kwargs_132966 = {}
    # Getting the type of 'np' (line 80)
    np_132963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 37), 'np', False)
    # Obtaining the member 'arange' of a type (line 80)
    arange_132964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 37), np_132963, 'arange')
    # Calling arange(args, kwargs) (line 80)
    arange_call_result_132967 = invoke(stypy.reporting.localization.Localization(__file__, 80, 37), arange_132964, *[m_132965], **kwargs_132966)
    
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___132968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 37), arange_call_result_132967, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_132969 = invoke(stypy.reporting.localization.Localization(__file__, 80, 37), getitem___132968, (slice_132961, None_132962))
    
    # Applying the binary operator '-' (line 80)
    result_sub_132970 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 35), '-', result_sub_132960, subscript_call_result_132969)
    
    # Applying the binary operator '*' (line 80)
    result_mul_132971 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 22), '*', y_132957, result_sub_132970)
    
    # Obtaining the member 'sum' of a type (line 80)
    sum_132972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 22), result_mul_132971, 'sum')
    # Calling sum(args, kwargs) (line 80)
    sum_call_result_132975 = invoke(stypy.reporting.localization.Localization(__file__, 80, 22), sum_132972, *[int_132973], **kwargs_132974)
    
    # Assigning a type to the variable 'first_line' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'first_line', sum_call_result_132975)
    
    # Getting the type of 'first_line' (line 81)
    first_line_132976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'first_line')
    
    # Getting the type of 'm' (line 81)
    m_132977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 23), 'm')
    # Applying the 'usub' unary operator (line 81)
    result___neg___132978 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 22), 'usub', m_132977)
    
    # Applying the binary operator 'div=' (line 81)
    result_div_132979 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 8), 'div=', first_line_132976, result___neg___132978)
    # Assigning a type to the variable 'first_line' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'first_line', result_div_132979)
    
    
    # Getting the type of 'stack' (line 82)
    stack_132980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stack')
    # Getting the type of 'first_line' (line 82)
    first_line_132981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'first_line')
    # Applying the binary operator '+=' (line 82)
    result_iadd_132982 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 8), '+=', stack_132980, first_line_132981)
    # Assigning a type to the variable 'stack' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stack', result_iadd_132982)
    
    # SSA branch for the else part of an if statement (line 78)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'baseline' (line 84)
    baseline_132983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 9), 'baseline')
    unicode_132984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 21), 'unicode', u'weighted_wiggle')
    # Applying the binary operator '==' (line 84)
    result_eq_132985 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 9), '==', baseline_132983, unicode_132984)
    
    # Testing the type of an if condition (line 84)
    if_condition_132986 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 9), result_eq_132985)
    # Assigning a type to the variable 'if_condition_132986' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 9), 'if_condition_132986', if_condition_132986)
    # SSA begins for if statement (line 84)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Tuple (line 85):
    
    # Assigning a Subscript to a Name (line 85):
    
    # Obtaining the type of the subscript
    int_132987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
    # Getting the type of 'y' (line 85)
    y_132988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'y')
    # Obtaining the member 'shape' of a type (line 85)
    shape_132989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 15), y_132988, 'shape')
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___132990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), shape_132989, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_132991 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), getitem___132990, int_132987)
    
    # Assigning a type to the variable 'tuple_var_assignment_132854' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_132854', subscript_call_result_132991)
    
    # Assigning a Subscript to a Name (line 85):
    
    # Obtaining the type of the subscript
    int_132992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
    # Getting the type of 'y' (line 85)
    y_132993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'y')
    # Obtaining the member 'shape' of a type (line 85)
    shape_132994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 15), y_132993, 'shape')
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___132995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), shape_132994, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_132996 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), getitem___132995, int_132992)
    
    # Assigning a type to the variable 'tuple_var_assignment_132855' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_132855', subscript_call_result_132996)
    
    # Assigning a Name to a Name (line 85):
    # Getting the type of 'tuple_var_assignment_132854' (line 85)
    tuple_var_assignment_132854_132997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_132854')
    # Assigning a type to the variable 'm' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'm', tuple_var_assignment_132854_132997)
    
    # Assigning a Name to a Name (line 85):
    # Getting the type of 'tuple_var_assignment_132855' (line 85)
    tuple_var_assignment_132855_132998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_132855')
    # Assigning a type to the variable 'n' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'n', tuple_var_assignment_132855_132998)
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to sum(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'y' (line 86)
    y_133001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), 'y', False)
    int_133002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 26), 'int')
    # Processing the call keyword arguments (line 86)
    kwargs_133003 = {}
    # Getting the type of 'np' (line 86)
    np_132999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'np', False)
    # Obtaining the member 'sum' of a type (line 86)
    sum_133000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 16), np_132999, 'sum')
    # Calling sum(args, kwargs) (line 86)
    sum_call_result_133004 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), sum_133000, *[y_133001, int_133002], **kwargs_133003)
    
    # Assigning a type to the variable 'total' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'total', sum_call_result_133004)
    
    # Assigning a Call to a Name (line 88):
    
    # Assigning a Call to a Name (line 88):
    
    # Call to zeros_like(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'total' (line 88)
    total_133007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 34), 'total', False)
    # Processing the call keyword arguments (line 88)
    kwargs_133008 = {}
    # Getting the type of 'np' (line 88)
    np_133005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 88)
    zeros_like_133006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 20), np_133005, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 88)
    zeros_like_call_result_133009 = invoke(stypy.reporting.localization.Localization(__file__, 88, 20), zeros_like_133006, *[total_133007], **kwargs_133008)
    
    # Assigning a type to the variable 'inv_total' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'inv_total', zeros_like_call_result_133009)
    
    # Assigning a Compare to a Name (line 89):
    
    # Assigning a Compare to a Name (line 89):
    
    # Getting the type of 'total' (line 89)
    total_133010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'total')
    int_133011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 23), 'int')
    # Applying the binary operator '>' (line 89)
    result_gt_133012 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 15), '>', total_133010, int_133011)
    
    # Assigning a type to the variable 'mask' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'mask', result_gt_133012)
    
    # Assigning a BinOp to a Subscript (line 90):
    
    # Assigning a BinOp to a Subscript (line 90):
    float_133013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 26), 'float')
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 90)
    mask_133014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 38), 'mask')
    # Getting the type of 'total' (line 90)
    total_133015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 32), 'total')
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___133016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 32), total_133015, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_133017 = invoke(stypy.reporting.localization.Localization(__file__, 90, 32), getitem___133016, mask_133014)
    
    # Applying the binary operator 'div' (line 90)
    result_div_133018 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 26), 'div', float_133013, subscript_call_result_133017)
    
    # Getting the type of 'inv_total' (line 90)
    inv_total_133019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'inv_total')
    # Getting the type of 'mask' (line 90)
    mask_133020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'mask')
    # Storing an element on a container (line 90)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), inv_total_133019, (mask_133020, result_div_133018))
    
    # Assigning a Call to a Name (line 91):
    
    # Assigning a Call to a Name (line 91):
    
    # Call to hstack(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Obtaining an instance of the builtin type 'tuple' (line 91)
    tuple_133023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 91)
    # Adding element type (line 91)
    
    # Obtaining the type of the subscript
    slice_133024 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 91, 30), None, None, None)
    int_133025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 35), 'int')
    int_133026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 37), 'int')
    slice_133027 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 91, 30), int_133025, int_133026, None)
    # Getting the type of 'y' (line 91)
    y_133028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 30), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 91)
    getitem___133029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 30), y_133028, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 91)
    subscript_call_result_133030 = invoke(stypy.reporting.localization.Localization(__file__, 91, 30), getitem___133029, (slice_133024, slice_133027))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 30), tuple_133023, subscript_call_result_133030)
    # Adding element type (line 91)
    
    # Call to diff(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'y' (line 91)
    y_133033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 49), 'y', False)
    # Processing the call keyword arguments (line 91)
    kwargs_133034 = {}
    # Getting the type of 'np' (line 91)
    np_133031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 41), 'np', False)
    # Obtaining the member 'diff' of a type (line 91)
    diff_133032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 41), np_133031, 'diff')
    # Calling diff(args, kwargs) (line 91)
    diff_call_result_133035 = invoke(stypy.reporting.localization.Localization(__file__, 91, 41), diff_133032, *[y_133033], **kwargs_133034)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 30), tuple_133023, diff_call_result_133035)
    
    # Processing the call keyword arguments (line 91)
    kwargs_133036 = {}
    # Getting the type of 'np' (line 91)
    np_133021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'np', False)
    # Obtaining the member 'hstack' of a type (line 91)
    hstack_133022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 19), np_133021, 'hstack')
    # Calling hstack(args, kwargs) (line 91)
    hstack_call_result_133037 = invoke(stypy.reporting.localization.Localization(__file__, 91, 19), hstack_133022, *[tuple_133023], **kwargs_133036)
    
    # Assigning a type to the variable 'increase' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'increase', hstack_call_result_133037)
    
    # Assigning a BinOp to a Name (line 92):
    
    # Assigning a BinOp to a Name (line 92):
    # Getting the type of 'total' (line 92)
    total_133038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 21), 'total')
    # Getting the type of 'stack' (line 92)
    stack_133039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 29), 'stack')
    # Applying the binary operator '-' (line 92)
    result_sub_133040 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 21), '-', total_133038, stack_133039)
    
    # Assigning a type to the variable 'below_size' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'below_size', result_sub_133040)
    
    # Getting the type of 'below_size' (line 93)
    below_size_133041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'below_size')
    float_133042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 22), 'float')
    # Getting the type of 'y' (line 93)
    y_133043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'y')
    # Applying the binary operator '*' (line 93)
    result_mul_133044 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 22), '*', float_133042, y_133043)
    
    # Applying the binary operator '+=' (line 93)
    result_iadd_133045 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 8), '+=', below_size_133041, result_mul_133044)
    # Assigning a type to the variable 'below_size' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'below_size', result_iadd_133045)
    
    
    # Assigning a BinOp to a Name (line 94):
    
    # Assigning a BinOp to a Name (line 94):
    # Getting the type of 'below_size' (line 94)
    below_size_133046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'below_size')
    # Getting the type of 'inv_total' (line 94)
    inv_total_133047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 31), 'inv_total')
    # Applying the binary operator '*' (line 94)
    result_mul_133048 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 18), '*', below_size_133046, inv_total_133047)
    
    # Assigning a type to the variable 'move_up' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'move_up', result_mul_133048)
    
    # Assigning a Num to a Subscript (line 95):
    
    # Assigning a Num to a Subscript (line 95):
    float_133049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 24), 'float')
    # Getting the type of 'move_up' (line 95)
    move_up_133050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'move_up')
    slice_133051 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 95, 8), None, None, None)
    int_133052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 19), 'int')
    # Storing an element on a container (line 95)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 8), move_up_133050, ((slice_133051, int_133052), float_133049))
    
    # Assigning a BinOp to a Name (line 96):
    
    # Assigning a BinOp to a Name (line 96):
    # Getting the type of 'move_up' (line 96)
    move_up_133053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'move_up')
    float_133054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 28), 'float')
    # Applying the binary operator '-' (line 96)
    result_sub_133055 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 18), '-', move_up_133053, float_133054)
    
    # Getting the type of 'increase' (line 96)
    increase_133056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 35), 'increase')
    # Applying the binary operator '*' (line 96)
    result_mul_133057 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 17), '*', result_sub_133055, increase_133056)
    
    # Assigning a type to the variable 'center' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'center', result_mul_133057)
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to cumsum(...): (line 97)
    # Processing the call arguments (line 97)
    
    # Call to sum(...): (line 97)
    # Processing the call arguments (line 97)
    int_133062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 38), 'int')
    # Processing the call keyword arguments (line 97)
    kwargs_133063 = {}
    # Getting the type of 'center' (line 97)
    center_133060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 27), 'center', False)
    # Obtaining the member 'sum' of a type (line 97)
    sum_133061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 27), center_133060, 'sum')
    # Calling sum(args, kwargs) (line 97)
    sum_call_result_133064 = invoke(stypy.reporting.localization.Localization(__file__, 97, 27), sum_133061, *[int_133062], **kwargs_133063)
    
    # Processing the call keyword arguments (line 97)
    kwargs_133065 = {}
    # Getting the type of 'np' (line 97)
    np_133058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 17), 'np', False)
    # Obtaining the member 'cumsum' of a type (line 97)
    cumsum_133059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 17), np_133058, 'cumsum')
    # Calling cumsum(args, kwargs) (line 97)
    cumsum_call_result_133066 = invoke(stypy.reporting.localization.Localization(__file__, 97, 17), cumsum_133059, *[sum_call_result_133064], **kwargs_133065)
    
    # Assigning a type to the variable 'center' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'center', cumsum_call_result_133066)
    
    # Assigning a BinOp to a Name (line 98):
    
    # Assigning a BinOp to a Name (line 98):
    # Getting the type of 'center' (line 98)
    center_133067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'center')
    float_133068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 30), 'float')
    # Getting the type of 'total' (line 98)
    total_133069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 36), 'total')
    # Applying the binary operator '*' (line 98)
    result_mul_133070 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 30), '*', float_133068, total_133069)
    
    # Applying the binary operator '-' (line 98)
    result_sub_133071 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 21), '-', center_133067, result_mul_133070)
    
    # Assigning a type to the variable 'first_line' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'first_line', result_sub_133071)
    
    # Getting the type of 'stack' (line 99)
    stack_133072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'stack')
    # Getting the type of 'first_line' (line 99)
    first_line_133073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'first_line')
    # Applying the binary operator '+=' (line 99)
    result_iadd_133074 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 8), '+=', stack_133072, first_line_133073)
    # Assigning a type to the variable 'stack' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'stack', result_iadd_133074)
    
    # SSA branch for the else part of an if statement (line 84)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 102):
    
    # Assigning a BinOp to a Name (line 102):
    unicode_133075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 17), 'unicode', u'Baseline method %s not recognised. ')
    # Getting the type of 'baseline' (line 102)
    baseline_133076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 57), 'baseline')
    # Applying the binary operator '%' (line 102)
    result_mod_133077 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 17), '%', unicode_133075, baseline_133076)
    
    # Assigning a type to the variable 'errstr' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'errstr', result_mod_133077)
    
    # Getting the type of 'errstr' (line 103)
    errstr_133078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'errstr')
    unicode_133079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 18), 'unicode', u"Expected 'zero', 'sym', 'wiggle' or 'weighted_wiggle'")
    # Applying the binary operator '+=' (line 103)
    result_iadd_133080 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 8), '+=', errstr_133078, unicode_133079)
    # Assigning a type to the variable 'errstr' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'errstr', result_iadd_133080)
    
    
    # Call to ValueError(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'errstr' (line 104)
    errstr_133082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'errstr', False)
    # Processing the call keyword arguments (line 104)
    kwargs_133083 = {}
    # Getting the type of 'ValueError' (line 104)
    ValueError_133081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 104)
    ValueError_call_result_133084 = invoke(stypy.reporting.localization.Localization(__file__, 104, 14), ValueError_133081, *[errstr_133082], **kwargs_133083)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 104, 8), ValueError_call_result_133084, 'raise parameter', BaseException)
    # SSA join for if statement (line 84)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 74)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 107):
    
    # Assigning a Call to a Name (line 107):
    
    # Call to get_next_color(...): (line 107)
    # Processing the call keyword arguments (line 107)
    kwargs_133088 = {}
    # Getting the type of 'axes' (line 107)
    axes_133085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'axes', False)
    # Obtaining the member '_get_lines' of a type (line 107)
    _get_lines_133086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), axes_133085, '_get_lines')
    # Obtaining the member 'get_next_color' of a type (line 107)
    get_next_color_133087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), _get_lines_133086, 'get_next_color')
    # Calling get_next_color(args, kwargs) (line 107)
    get_next_color_call_result_133089 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), get_next_color_133087, *[], **kwargs_133088)
    
    # Assigning a type to the variable 'color' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'color', get_next_color_call_result_133089)
    
    # Assigning a Call to a Name (line 108):
    
    # Assigning a Call to a Name (line 108):
    
    # Call to fill_between(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'x' (line 108)
    x_133092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 'x', False)
    # Getting the type of 'first_line' (line 108)
    first_line_133093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 32), 'first_line', False)
    
    # Obtaining the type of the subscript
    int_133094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 50), 'int')
    slice_133095 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 108, 44), None, None, None)
    # Getting the type of 'stack' (line 108)
    stack_133096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 44), 'stack', False)
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___133097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 44), stack_133096, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_133098 = invoke(stypy.reporting.localization.Localization(__file__, 108, 44), getitem___133097, (int_133094, slice_133095))
    
    # Processing the call keyword arguments (line 108)
    # Getting the type of 'color' (line 109)
    color_133099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 39), 'color', False)
    keyword_133100 = color_133099
    
    # Call to next(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'labels' (line 109)
    labels_133103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 61), 'labels', False)
    # Getting the type of 'None' (line 109)
    None_133104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 69), 'None', False)
    # Processing the call keyword arguments (line 109)
    kwargs_133105 = {}
    # Getting the type of 'six' (line 109)
    six_133101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 52), 'six', False)
    # Obtaining the member 'next' of a type (line 109)
    next_133102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 52), six_133101, 'next')
    # Calling next(args, kwargs) (line 109)
    next_call_result_133106 = invoke(stypy.reporting.localization.Localization(__file__, 109, 52), next_133102, *[labels_133103, None_133104], **kwargs_133105)
    
    keyword_133107 = next_call_result_133106
    # Getting the type of 'kwargs' (line 110)
    kwargs_133108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 31), 'kwargs', False)
    kwargs_133109 = {'kwargs_133108': kwargs_133108, 'facecolor': keyword_133100, 'label': keyword_133107}
    # Getting the type of 'axes' (line 108)
    axes_133090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'axes', False)
    # Obtaining the member 'fill_between' of a type (line 108)
    fill_between_133091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 11), axes_133090, 'fill_between')
    # Calling fill_between(args, kwargs) (line 108)
    fill_between_call_result_133110 = invoke(stypy.reporting.localization.Localization(__file__, 108, 11), fill_between_133091, *[x_133092, first_line_133093, subscript_call_result_133098], **kwargs_133109)
    
    # Assigning a type to the variable 'coll' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'coll', fill_between_call_result_133110)
    
    # Assigning a List to a Subscript (line 111):
    
    # Assigning a List to a Subscript (line 111):
    
    # Obtaining an instance of the builtin type 'list' (line 111)
    list_133111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 111)
    # Adding element type (line 111)
    int_133112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 29), list_133111, int_133112)
    
    # Getting the type of 'coll' (line 111)
    coll_133113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'coll')
    # Obtaining the member 'sticky_edges' of a type (line 111)
    sticky_edges_133114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 4), coll_133113, 'sticky_edges')
    # Obtaining the member 'y' of a type (line 111)
    y_133115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 4), sticky_edges_133114, 'y')
    slice_133116 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 111, 4), None, None, None)
    # Storing an element on a container (line 111)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 4), y_133115, (slice_133116, list_133111))
    
    # Assigning a List to a Name (line 112):
    
    # Assigning a List to a Name (line 112):
    
    # Obtaining an instance of the builtin type 'list' (line 112)
    list_133117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 112)
    # Adding element type (line 112)
    # Getting the type of 'coll' (line 112)
    coll_133118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 9), 'coll')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 8), list_133117, coll_133118)
    
    # Assigning a type to the variable 'r' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'r', list_133117)
    
    
    # Call to xrange(...): (line 115)
    # Processing the call arguments (line 115)
    
    # Call to len(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'y' (line 115)
    y_133121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 24), 'y', False)
    # Processing the call keyword arguments (line 115)
    kwargs_133122 = {}
    # Getting the type of 'len' (line 115)
    len_133120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'len', False)
    # Calling len(args, kwargs) (line 115)
    len_call_result_133123 = invoke(stypy.reporting.localization.Localization(__file__, 115, 20), len_133120, *[y_133121], **kwargs_133122)
    
    int_133124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 29), 'int')
    # Applying the binary operator '-' (line 115)
    result_sub_133125 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 20), '-', len_call_result_133123, int_133124)
    
    # Processing the call keyword arguments (line 115)
    kwargs_133126 = {}
    # Getting the type of 'xrange' (line 115)
    xrange_133119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 115)
    xrange_call_result_133127 = invoke(stypy.reporting.localization.Localization(__file__, 115, 13), xrange_133119, *[result_sub_133125], **kwargs_133126)
    
    # Testing the type of a for loop iterable (line 115)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 4), xrange_call_result_133127)
    # Getting the type of the for loop variable (line 115)
    for_loop_var_133128 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 4), xrange_call_result_133127)
    # Assigning a type to the variable 'i' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'i', for_loop_var_133128)
    # SSA begins for a for statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 116):
    
    # Assigning a Call to a Name (line 116):
    
    # Call to get_next_color(...): (line 116)
    # Processing the call keyword arguments (line 116)
    kwargs_133132 = {}
    # Getting the type of 'axes' (line 116)
    axes_133129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'axes', False)
    # Obtaining the member '_get_lines' of a type (line 116)
    _get_lines_133130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 16), axes_133129, '_get_lines')
    # Obtaining the member 'get_next_color' of a type (line 116)
    get_next_color_133131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 16), _get_lines_133130, 'get_next_color')
    # Calling get_next_color(args, kwargs) (line 116)
    get_next_color_call_result_133133 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), get_next_color_133131, *[], **kwargs_133132)
    
    # Assigning a type to the variable 'color' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'color', get_next_color_call_result_133133)
    
    # Call to append(...): (line 117)
    # Processing the call arguments (line 117)
    
    # Call to fill_between(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'x' (line 117)
    x_133138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 35), 'x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 117)
    i_133139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 44), 'i', False)
    slice_133140 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 117, 38), None, None, None)
    # Getting the type of 'stack' (line 117)
    stack_133141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 38), 'stack', False)
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___133142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 38), stack_133141, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_133143 = invoke(stypy.reporting.localization.Localization(__file__, 117, 38), getitem___133142, (i_133139, slice_133140))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 117)
    i_133144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 57), 'i', False)
    int_133145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 61), 'int')
    # Applying the binary operator '+' (line 117)
    result_add_133146 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 57), '+', i_133144, int_133145)
    
    slice_133147 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 117, 51), None, None, None)
    # Getting the type of 'stack' (line 117)
    stack_133148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 51), 'stack', False)
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___133149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 51), stack_133148, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_133150 = invoke(stypy.reporting.localization.Localization(__file__, 117, 51), getitem___133149, (result_add_133146, slice_133147))
    
    # Processing the call keyword arguments (line 117)
    # Getting the type of 'color' (line 118)
    color_133151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 45), 'color', False)
    keyword_133152 = color_133151
    
    # Call to next(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'labels' (line 119)
    labels_133155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 51), 'labels', False)
    # Getting the type of 'None' (line 119)
    None_133156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 59), 'None', False)
    # Processing the call keyword arguments (line 119)
    kwargs_133157 = {}
    # Getting the type of 'six' (line 119)
    six_133153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 42), 'six', False)
    # Obtaining the member 'next' of a type (line 119)
    next_133154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 42), six_133153, 'next')
    # Calling next(args, kwargs) (line 119)
    next_call_result_133158 = invoke(stypy.reporting.localization.Localization(__file__, 119, 42), next_133154, *[labels_133155, None_133156], **kwargs_133157)
    
    keyword_133159 = next_call_result_133158
    # Getting the type of 'kwargs' (line 120)
    kwargs_133160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 37), 'kwargs', False)
    kwargs_133161 = {'kwargs_133160': kwargs_133160, 'facecolor': keyword_133152, 'label': keyword_133159}
    # Getting the type of 'axes' (line 117)
    axes_133136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'axes', False)
    # Obtaining the member 'fill_between' of a type (line 117)
    fill_between_133137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 17), axes_133136, 'fill_between')
    # Calling fill_between(args, kwargs) (line 117)
    fill_between_call_result_133162 = invoke(stypy.reporting.localization.Localization(__file__, 117, 17), fill_between_133137, *[x_133138, subscript_call_result_133143, subscript_call_result_133150], **kwargs_133161)
    
    # Processing the call keyword arguments (line 117)
    kwargs_133163 = {}
    # Getting the type of 'r' (line 117)
    r_133134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'r', False)
    # Obtaining the member 'append' of a type (line 117)
    append_133135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), r_133134, 'append')
    # Calling append(args, kwargs) (line 117)
    append_call_result_133164 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), append_133135, *[fill_between_call_result_133162], **kwargs_133163)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'r' (line 121)
    r_133165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type', r_133165)
    
    # ################# End of 'stackplot(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'stackplot' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_133166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_133166)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'stackplot'
    return stypy_return_type_133166

# Assigning a type to the variable 'stackplot' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stackplot', stackplot)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
