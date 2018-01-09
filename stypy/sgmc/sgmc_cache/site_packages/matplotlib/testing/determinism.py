
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Provides utilities to test output reproducibility.
3: '''
4: 
5: from __future__ import (absolute_import, division, print_function,
6:                         unicode_literals)
7: 
8: import six
9: 
10: import io
11: import os
12: import re
13: import sys
14: from subprocess import check_output
15: 
16: import pytest
17: 
18: import matplotlib
19: from matplotlib import pyplot as plt
20: 
21: 
22: def _determinism_save(objects='mhi', format="pdf", usetex=False):
23:     # save current value of SOURCE_DATE_EPOCH and set it
24:     # to a constant value, so that time difference is not
25:     # taken into account
26:     sde = os.environ.pop('SOURCE_DATE_EPOCH', None)
27:     os.environ['SOURCE_DATE_EPOCH'] = "946684800"
28: 
29:     matplotlib.rcParams['text.usetex'] = usetex
30: 
31:     fig = plt.figure()
32: 
33:     if 'm' in objects:
34:         # use different markers...
35:         ax1 = fig.add_subplot(1, 6, 1)
36:         x = range(10)
37:         ax1.plot(x, [1] * 10, marker=u'D')
38:         ax1.plot(x, [2] * 10, marker=u'x')
39:         ax1.plot(x, [3] * 10, marker=u'^')
40:         ax1.plot(x, [4] * 10, marker=u'H')
41:         ax1.plot(x, [5] * 10, marker=u'v')
42: 
43:     if 'h' in objects:
44:         # also use different hatch patterns
45:         ax2 = fig.add_subplot(1, 6, 2)
46:         bars = (ax2.bar(range(1, 5), range(1, 5)) +
47:                 ax2.bar(range(1, 5), [6] * 4, bottom=range(1, 5)))
48:         ax2.set_xticks([1.5, 2.5, 3.5, 4.5])
49: 
50:         patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
51:         for bar, pattern in zip(bars, patterns):
52:             bar.set_hatch(pattern)
53: 
54:     if 'i' in objects:
55:         # also use different images
56:         A = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
57:         fig.add_subplot(1, 6, 3).imshow(A, interpolation='nearest')
58:         A = [[1, 3, 2], [1, 2, 3], [3, 1, 2]]
59:         fig.add_subplot(1, 6, 4).imshow(A, interpolation='bilinear')
60:         A = [[2, 3, 1], [1, 2, 3], [2, 1, 3]]
61:         fig.add_subplot(1, 6, 5).imshow(A, interpolation='bicubic')
62: 
63:     x = range(5)
64:     fig.add_subplot(1, 6, 6).plot(x, x)
65: 
66:     if six.PY2 and format == 'ps':
67:         stdout = io.StringIO()
68:     else:
69:         stdout = getattr(sys.stdout, 'buffer', sys.stdout)
70:     fig.savefig(stdout, format=format)
71:     if six.PY2 and format == 'ps':
72:         sys.stdout.write(stdout.getvalue())
73: 
74:     # Restores SOURCE_DATE_EPOCH
75:     if sde is None:
76:         os.environ.pop('SOURCE_DATE_EPOCH', None)
77:     else:
78:         os.environ['SOURCE_DATE_EPOCH'] = sde
79: 
80: 
81: def _determinism_check(objects='mhi', format="pdf", usetex=False):
82:     '''
83:     Output three times the same graphs and checks that the outputs are exactly
84:     the same.
85: 
86:     Parameters
87:     ----------
88:     objects : str
89:         contains characters corresponding to objects to be included in the test
90:         document: 'm' for markers, 'h' for hatch patterns, 'i' for images. The
91:         default value is "mhi", so that the test includes all these objects.
92:     format : str
93:         format string. The default value is "pdf".
94:     '''
95:     plots = []
96:     for i in range(3):
97:         result = check_output([sys.executable, '-R', '-c',
98:                                'import matplotlib; '
99:                                'matplotlib._called_from_pytest = True; '
100:                                'matplotlib.use(%r); '
101:                                'from matplotlib.testing.determinism '
102:                                'import _determinism_save;'
103:                                '_determinism_save(%r,%r,%r)'
104:                                % (format, objects, format, usetex)])
105:         plots.append(result)
106:     for p in plots[1:]:
107:         if usetex:
108:             if p != plots[0]:
109:                 pytest.skip("failed, maybe due to ghostscript timestamps")
110:         else:
111:             assert p == plots[0]
112: 
113: 
114: def _determinism_source_date_epoch(format, string, keyword=b"CreationDate"):
115:     '''
116:     Test SOURCE_DATE_EPOCH support. Output a document with the envionment
117:     variable SOURCE_DATE_EPOCH set to 2000-01-01 00:00 UTC and check that the
118:     document contains the timestamp that corresponds to this date (given as an
119:     argument).
120: 
121:     Parameters
122:     ----------
123:     format : str
124:         format string, such as "pdf".
125:     string : str
126:         timestamp string for 2000-01-01 00:00 UTC.
127:     keyword : bytes
128:         a string to look at when searching for the timestamp in the document
129:         (used in case the test fails).
130:     '''
131:     buff = check_output([sys.executable, '-R', '-c',
132:                          'import matplotlib; '
133:                          'matplotlib._called_from_pytest = True; '
134:                          'matplotlib.use(%r); '
135:                          'from matplotlib.testing.determinism '
136:                          'import _determinism_save;'
137:                          '_determinism_save(%r,%r)'
138:                          % (format, "", format)])
139:     find_keyword = re.compile(b".*" + keyword + b".*")
140:     key = find_keyword.search(buff)
141:     if key:
142:         print(key.group())
143:     else:
144:         print("Timestamp keyword (%s) not found!" % keyword)
145:     assert string in buff
146: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_291262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nProvides utilities to test output reproducibility.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import six' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_291263 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six')

if (type(import_291263) is not StypyTypeError):

    if (import_291263 != 'pyd_module'):
        __import__(import_291263)
        sys_modules_291264 = sys.modules[import_291263]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', sys_modules_291264.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', import_291263)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import io' statement (line 10)
import io

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'io', io, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import os' statement (line 11)
import os

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import re' statement (line 12)
import re

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import sys' statement (line 13)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from subprocess import check_output' statement (line 14)
try:
    from subprocess import check_output

except:
    check_output = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'subprocess', None, module_type_store, ['check_output'], [check_output])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import pytest' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_291265 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'pytest')

if (type(import_291265) is not StypyTypeError):

    if (import_291265 != 'pyd_module'):
        __import__(import_291265)
        sys_modules_291266 = sys.modules[import_291265]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'pytest', sys_modules_291266.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'pytest', import_291265)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import matplotlib' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_291267 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib')

if (type(import_291267) is not StypyTypeError):

    if (import_291267 != 'pyd_module'):
        __import__(import_291267)
        sys_modules_291268 = sys.modules[import_291267]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib', sys_modules_291268.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib', import_291267)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from matplotlib import plt' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_291269 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib')

if (type(import_291269) is not StypyTypeError):

    if (import_291269 != 'pyd_module'):
        __import__(import_291269)
        sys_modules_291270 = sys.modules[import_291269]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib', sys_modules_291270.module_type_store, module_type_store, ['pyplot'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_291270, sys_modules_291270.module_type_store, module_type_store)
    else:
        from matplotlib import pyplot as plt

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib', None, module_type_store, ['pyplot'], [plt])

else:
    # Assigning a type to the variable 'matplotlib' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib', import_291269)

# Adding an alias
module_type_store.add_alias('plt', 'pyplot')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')


@norecursion
def _determinism_save(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    unicode_291271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'unicode', u'mhi')
    unicode_291272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 44), 'unicode', u'pdf')
    # Getting the type of 'False' (line 22)
    False_291273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 58), 'False')
    defaults = [unicode_291271, unicode_291272, False_291273]
    # Create a new context for function '_determinism_save'
    module_type_store = module_type_store.open_function_context('_determinism_save', 22, 0, False)
    
    # Passed parameters checking function
    _determinism_save.stypy_localization = localization
    _determinism_save.stypy_type_of_self = None
    _determinism_save.stypy_type_store = module_type_store
    _determinism_save.stypy_function_name = '_determinism_save'
    _determinism_save.stypy_param_names_list = ['objects', 'format', 'usetex']
    _determinism_save.stypy_varargs_param_name = None
    _determinism_save.stypy_kwargs_param_name = None
    _determinism_save.stypy_call_defaults = defaults
    _determinism_save.stypy_call_varargs = varargs
    _determinism_save.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_determinism_save', ['objects', 'format', 'usetex'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_determinism_save', localization, ['objects', 'format', 'usetex'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_determinism_save(...)' code ##################

    
    # Assigning a Call to a Name (line 26):
    
    # Call to pop(...): (line 26)
    # Processing the call arguments (line 26)
    unicode_291277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'unicode', u'SOURCE_DATE_EPOCH')
    # Getting the type of 'None' (line 26)
    None_291278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 46), 'None', False)
    # Processing the call keyword arguments (line 26)
    kwargs_291279 = {}
    # Getting the type of 'os' (line 26)
    os_291274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'os', False)
    # Obtaining the member 'environ' of a type (line 26)
    environ_291275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 10), os_291274, 'environ')
    # Obtaining the member 'pop' of a type (line 26)
    pop_291276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 10), environ_291275, 'pop')
    # Calling pop(args, kwargs) (line 26)
    pop_call_result_291280 = invoke(stypy.reporting.localization.Localization(__file__, 26, 10), pop_291276, *[unicode_291277, None_291278], **kwargs_291279)
    
    # Assigning a type to the variable 'sde' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'sde', pop_call_result_291280)
    
    # Assigning a Str to a Subscript (line 27):
    unicode_291281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 38), 'unicode', u'946684800')
    # Getting the type of 'os' (line 27)
    os_291282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'os')
    # Obtaining the member 'environ' of a type (line 27)
    environ_291283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 4), os_291282, 'environ')
    unicode_291284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'unicode', u'SOURCE_DATE_EPOCH')
    # Storing an element on a container (line 27)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), environ_291283, (unicode_291284, unicode_291281))
    
    # Assigning a Name to a Subscript (line 29):
    # Getting the type of 'usetex' (line 29)
    usetex_291285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 41), 'usetex')
    # Getting the type of 'matplotlib' (line 29)
    matplotlib_291286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'matplotlib')
    # Obtaining the member 'rcParams' of a type (line 29)
    rcParams_291287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), matplotlib_291286, 'rcParams')
    unicode_291288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 24), 'unicode', u'text.usetex')
    # Storing an element on a container (line 29)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 4), rcParams_291287, (unicode_291288, usetex_291285))
    
    # Assigning a Call to a Name (line 31):
    
    # Call to figure(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_291291 = {}
    # Getting the type of 'plt' (line 31)
    plt_291289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'plt', False)
    # Obtaining the member 'figure' of a type (line 31)
    figure_291290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 10), plt_291289, 'figure')
    # Calling figure(args, kwargs) (line 31)
    figure_call_result_291292 = invoke(stypy.reporting.localization.Localization(__file__, 31, 10), figure_291290, *[], **kwargs_291291)
    
    # Assigning a type to the variable 'fig' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'fig', figure_call_result_291292)
    
    
    unicode_291293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 7), 'unicode', u'm')
    # Getting the type of 'objects' (line 33)
    objects_291294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'objects')
    # Applying the binary operator 'in' (line 33)
    result_contains_291295 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 7), 'in', unicode_291293, objects_291294)
    
    # Testing the type of an if condition (line 33)
    if_condition_291296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 4), result_contains_291295)
    # Assigning a type to the variable 'if_condition_291296' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'if_condition_291296', if_condition_291296)
    # SSA begins for if statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 35):
    
    # Call to add_subplot(...): (line 35)
    # Processing the call arguments (line 35)
    int_291299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 30), 'int')
    int_291300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 33), 'int')
    int_291301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'int')
    # Processing the call keyword arguments (line 35)
    kwargs_291302 = {}
    # Getting the type of 'fig' (line 35)
    fig_291297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'fig', False)
    # Obtaining the member 'add_subplot' of a type (line 35)
    add_subplot_291298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 14), fig_291297, 'add_subplot')
    # Calling add_subplot(args, kwargs) (line 35)
    add_subplot_call_result_291303 = invoke(stypy.reporting.localization.Localization(__file__, 35, 14), add_subplot_291298, *[int_291299, int_291300, int_291301], **kwargs_291302)
    
    # Assigning a type to the variable 'ax1' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'ax1', add_subplot_call_result_291303)
    
    # Assigning a Call to a Name (line 36):
    
    # Call to range(...): (line 36)
    # Processing the call arguments (line 36)
    int_291305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 18), 'int')
    # Processing the call keyword arguments (line 36)
    kwargs_291306 = {}
    # Getting the type of 'range' (line 36)
    range_291304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'range', False)
    # Calling range(args, kwargs) (line 36)
    range_call_result_291307 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), range_291304, *[int_291305], **kwargs_291306)
    
    # Assigning a type to the variable 'x' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'x', range_call_result_291307)
    
    # Call to plot(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'x' (line 37)
    x_291310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_291311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    int_291312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 20), list_291311, int_291312)
    
    int_291313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 26), 'int')
    # Applying the binary operator '*' (line 37)
    result_mul_291314 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 20), '*', list_291311, int_291313)
    
    # Processing the call keyword arguments (line 37)
    unicode_291315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 37), 'unicode', u'D')
    keyword_291316 = unicode_291315
    kwargs_291317 = {'marker': keyword_291316}
    # Getting the type of 'ax1' (line 37)
    ax1_291308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'ax1', False)
    # Obtaining the member 'plot' of a type (line 37)
    plot_291309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), ax1_291308, 'plot')
    # Calling plot(args, kwargs) (line 37)
    plot_call_result_291318 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), plot_291309, *[x_291310, result_mul_291314], **kwargs_291317)
    
    
    # Call to plot(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'x' (line 38)
    x_291321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_291322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    int_291323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 20), list_291322, int_291323)
    
    int_291324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'int')
    # Applying the binary operator '*' (line 38)
    result_mul_291325 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 20), '*', list_291322, int_291324)
    
    # Processing the call keyword arguments (line 38)
    unicode_291326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 37), 'unicode', u'x')
    keyword_291327 = unicode_291326
    kwargs_291328 = {'marker': keyword_291327}
    # Getting the type of 'ax1' (line 38)
    ax1_291319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'ax1', False)
    # Obtaining the member 'plot' of a type (line 38)
    plot_291320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), ax1_291319, 'plot')
    # Calling plot(args, kwargs) (line 38)
    plot_call_result_291329 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), plot_291320, *[x_291321, result_mul_291325], **kwargs_291328)
    
    
    # Call to plot(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'x' (line 39)
    x_291332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_291333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    int_291334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_291333, int_291334)
    
    int_291335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 26), 'int')
    # Applying the binary operator '*' (line 39)
    result_mul_291336 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 20), '*', list_291333, int_291335)
    
    # Processing the call keyword arguments (line 39)
    unicode_291337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 37), 'unicode', u'^')
    keyword_291338 = unicode_291337
    kwargs_291339 = {'marker': keyword_291338}
    # Getting the type of 'ax1' (line 39)
    ax1_291330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'ax1', False)
    # Obtaining the member 'plot' of a type (line 39)
    plot_291331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), ax1_291330, 'plot')
    # Calling plot(args, kwargs) (line 39)
    plot_call_result_291340 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), plot_291331, *[x_291332, result_mul_291336], **kwargs_291339)
    
    
    # Call to plot(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'x' (line 40)
    x_291343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_291344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    int_291345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 20), list_291344, int_291345)
    
    int_291346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'int')
    # Applying the binary operator '*' (line 40)
    result_mul_291347 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 20), '*', list_291344, int_291346)
    
    # Processing the call keyword arguments (line 40)
    unicode_291348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 37), 'unicode', u'H')
    keyword_291349 = unicode_291348
    kwargs_291350 = {'marker': keyword_291349}
    # Getting the type of 'ax1' (line 40)
    ax1_291341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'ax1', False)
    # Obtaining the member 'plot' of a type (line 40)
    plot_291342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), ax1_291341, 'plot')
    # Calling plot(args, kwargs) (line 40)
    plot_call_result_291351 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), plot_291342, *[x_291343, result_mul_291347], **kwargs_291350)
    
    
    # Call to plot(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'x' (line 41)
    x_291354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_291355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    int_291356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), list_291355, int_291356)
    
    int_291357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'int')
    # Applying the binary operator '*' (line 41)
    result_mul_291358 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 20), '*', list_291355, int_291357)
    
    # Processing the call keyword arguments (line 41)
    unicode_291359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 37), 'unicode', u'v')
    keyword_291360 = unicode_291359
    kwargs_291361 = {'marker': keyword_291360}
    # Getting the type of 'ax1' (line 41)
    ax1_291352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'ax1', False)
    # Obtaining the member 'plot' of a type (line 41)
    plot_291353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), ax1_291352, 'plot')
    # Calling plot(args, kwargs) (line 41)
    plot_call_result_291362 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), plot_291353, *[x_291354, result_mul_291358], **kwargs_291361)
    
    # SSA join for if statement (line 33)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    unicode_291363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 7), 'unicode', u'h')
    # Getting the type of 'objects' (line 43)
    objects_291364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'objects')
    # Applying the binary operator 'in' (line 43)
    result_contains_291365 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 7), 'in', unicode_291363, objects_291364)
    
    # Testing the type of an if condition (line 43)
    if_condition_291366 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 4), result_contains_291365)
    # Assigning a type to the variable 'if_condition_291366' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'if_condition_291366', if_condition_291366)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 45):
    
    # Call to add_subplot(...): (line 45)
    # Processing the call arguments (line 45)
    int_291369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 30), 'int')
    int_291370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 33), 'int')
    int_291371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 36), 'int')
    # Processing the call keyword arguments (line 45)
    kwargs_291372 = {}
    # Getting the type of 'fig' (line 45)
    fig_291367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 14), 'fig', False)
    # Obtaining the member 'add_subplot' of a type (line 45)
    add_subplot_291368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 14), fig_291367, 'add_subplot')
    # Calling add_subplot(args, kwargs) (line 45)
    add_subplot_call_result_291373 = invoke(stypy.reporting.localization.Localization(__file__, 45, 14), add_subplot_291368, *[int_291369, int_291370, int_291371], **kwargs_291372)
    
    # Assigning a type to the variable 'ax2' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'ax2', add_subplot_call_result_291373)
    
    # Assigning a BinOp to a Name (line 46):
    
    # Call to bar(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Call to range(...): (line 46)
    # Processing the call arguments (line 46)
    int_291377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 30), 'int')
    int_291378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 33), 'int')
    # Processing the call keyword arguments (line 46)
    kwargs_291379 = {}
    # Getting the type of 'range' (line 46)
    range_291376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'range', False)
    # Calling range(args, kwargs) (line 46)
    range_call_result_291380 = invoke(stypy.reporting.localization.Localization(__file__, 46, 24), range_291376, *[int_291377, int_291378], **kwargs_291379)
    
    
    # Call to range(...): (line 46)
    # Processing the call arguments (line 46)
    int_291382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 43), 'int')
    int_291383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 46), 'int')
    # Processing the call keyword arguments (line 46)
    kwargs_291384 = {}
    # Getting the type of 'range' (line 46)
    range_291381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 37), 'range', False)
    # Calling range(args, kwargs) (line 46)
    range_call_result_291385 = invoke(stypy.reporting.localization.Localization(__file__, 46, 37), range_291381, *[int_291382, int_291383], **kwargs_291384)
    
    # Processing the call keyword arguments (line 46)
    kwargs_291386 = {}
    # Getting the type of 'ax2' (line 46)
    ax2_291374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'ax2', False)
    # Obtaining the member 'bar' of a type (line 46)
    bar_291375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 16), ax2_291374, 'bar')
    # Calling bar(args, kwargs) (line 46)
    bar_call_result_291387 = invoke(stypy.reporting.localization.Localization(__file__, 46, 16), bar_291375, *[range_call_result_291380, range_call_result_291385], **kwargs_291386)
    
    
    # Call to bar(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Call to range(...): (line 47)
    # Processing the call arguments (line 47)
    int_291391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 30), 'int')
    int_291392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 33), 'int')
    # Processing the call keyword arguments (line 47)
    kwargs_291393 = {}
    # Getting the type of 'range' (line 47)
    range_291390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'range', False)
    # Calling range(args, kwargs) (line 47)
    range_call_result_291394 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), range_291390, *[int_291391, int_291392], **kwargs_291393)
    
    
    # Obtaining an instance of the builtin type 'list' (line 47)
    list_291395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 47)
    # Adding element type (line 47)
    int_291396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 37), list_291395, int_291396)
    
    int_291397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 43), 'int')
    # Applying the binary operator '*' (line 47)
    result_mul_291398 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 37), '*', list_291395, int_291397)
    
    # Processing the call keyword arguments (line 47)
    
    # Call to range(...): (line 47)
    # Processing the call arguments (line 47)
    int_291400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 59), 'int')
    int_291401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 62), 'int')
    # Processing the call keyword arguments (line 47)
    kwargs_291402 = {}
    # Getting the type of 'range' (line 47)
    range_291399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 53), 'range', False)
    # Calling range(args, kwargs) (line 47)
    range_call_result_291403 = invoke(stypy.reporting.localization.Localization(__file__, 47, 53), range_291399, *[int_291400, int_291401], **kwargs_291402)
    
    keyword_291404 = range_call_result_291403
    kwargs_291405 = {'bottom': keyword_291404}
    # Getting the type of 'ax2' (line 47)
    ax2_291388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'ax2', False)
    # Obtaining the member 'bar' of a type (line 47)
    bar_291389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), ax2_291388, 'bar')
    # Calling bar(args, kwargs) (line 47)
    bar_call_result_291406 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), bar_291389, *[range_call_result_291394, result_mul_291398], **kwargs_291405)
    
    # Applying the binary operator '+' (line 46)
    result_add_291407 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 16), '+', bar_call_result_291387, bar_call_result_291406)
    
    # Assigning a type to the variable 'bars' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'bars', result_add_291407)
    
    # Call to set_xticks(...): (line 48)
    # Processing the call arguments (line 48)
    
    # Obtaining an instance of the builtin type 'list' (line 48)
    list_291410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 48)
    # Adding element type (line 48)
    float_291411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 23), list_291410, float_291411)
    # Adding element type (line 48)
    float_291412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 23), list_291410, float_291412)
    # Adding element type (line 48)
    float_291413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 23), list_291410, float_291413)
    # Adding element type (line 48)
    float_291414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 23), list_291410, float_291414)
    
    # Processing the call keyword arguments (line 48)
    kwargs_291415 = {}
    # Getting the type of 'ax2' (line 48)
    ax2_291408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'ax2', False)
    # Obtaining the member 'set_xticks' of a type (line 48)
    set_xticks_291409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), ax2_291408, 'set_xticks')
    # Calling set_xticks(args, kwargs) (line 48)
    set_xticks_call_result_291416 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), set_xticks_291409, *[list_291410], **kwargs_291415)
    
    
    # Assigning a Tuple to a Name (line 50):
    
    # Obtaining an instance of the builtin type 'tuple' (line 50)
    tuple_291417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 50)
    # Adding element type (line 50)
    unicode_291418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 20), 'unicode', u'-')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), tuple_291417, unicode_291418)
    # Adding element type (line 50)
    unicode_291419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 25), 'unicode', u'+')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), tuple_291417, unicode_291419)
    # Adding element type (line 50)
    unicode_291420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'unicode', u'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), tuple_291417, unicode_291420)
    # Adding element type (line 50)
    unicode_291421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 35), 'unicode', u'\\')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), tuple_291417, unicode_291421)
    # Adding element type (line 50)
    unicode_291422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 41), 'unicode', u'*')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), tuple_291417, unicode_291422)
    # Adding element type (line 50)
    unicode_291423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 46), 'unicode', u'o')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), tuple_291417, unicode_291423)
    # Adding element type (line 50)
    unicode_291424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 51), 'unicode', u'O')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), tuple_291417, unicode_291424)
    # Adding element type (line 50)
    unicode_291425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 56), 'unicode', u'.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), tuple_291417, unicode_291425)
    
    # Assigning a type to the variable 'patterns' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'patterns', tuple_291417)
    
    
    # Call to zip(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'bars' (line 51)
    bars_291427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 32), 'bars', False)
    # Getting the type of 'patterns' (line 51)
    patterns_291428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 38), 'patterns', False)
    # Processing the call keyword arguments (line 51)
    kwargs_291429 = {}
    # Getting the type of 'zip' (line 51)
    zip_291426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'zip', False)
    # Calling zip(args, kwargs) (line 51)
    zip_call_result_291430 = invoke(stypy.reporting.localization.Localization(__file__, 51, 28), zip_291426, *[bars_291427, patterns_291428], **kwargs_291429)
    
    # Testing the type of a for loop iterable (line 51)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 51, 8), zip_call_result_291430)
    # Getting the type of the for loop variable (line 51)
    for_loop_var_291431 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 51, 8), zip_call_result_291430)
    # Assigning a type to the variable 'bar' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'bar', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), for_loop_var_291431))
    # Assigning a type to the variable 'pattern' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'pattern', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), for_loop_var_291431))
    # SSA begins for a for statement (line 51)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to set_hatch(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'pattern' (line 52)
    pattern_291434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 26), 'pattern', False)
    # Processing the call keyword arguments (line 52)
    kwargs_291435 = {}
    # Getting the type of 'bar' (line 52)
    bar_291432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'bar', False)
    # Obtaining the member 'set_hatch' of a type (line 52)
    set_hatch_291433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), bar_291432, 'set_hatch')
    # Calling set_hatch(args, kwargs) (line 52)
    set_hatch_call_result_291436 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), set_hatch_291433, *[pattern_291434], **kwargs_291435)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    unicode_291437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 7), 'unicode', u'i')
    # Getting the type of 'objects' (line 54)
    objects_291438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 14), 'objects')
    # Applying the binary operator 'in' (line 54)
    result_contains_291439 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), 'in', unicode_291437, objects_291438)
    
    # Testing the type of an if condition (line 54)
    if_condition_291440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 4), result_contains_291439)
    # Assigning a type to the variable 'if_condition_291440' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'if_condition_291440', if_condition_291440)
    # SSA begins for if statement (line 54)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 56):
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_291441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_291442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    int_291443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 13), list_291442, int_291443)
    # Adding element type (line 56)
    int_291444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 13), list_291442, int_291444)
    # Adding element type (line 56)
    int_291445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 13), list_291442, int_291445)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 12), list_291441, list_291442)
    # Adding element type (line 56)
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_291446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    int_291447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 24), list_291446, int_291447)
    # Adding element type (line 56)
    int_291448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 24), list_291446, int_291448)
    # Adding element type (line 56)
    int_291449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 24), list_291446, int_291449)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 12), list_291441, list_291446)
    # Adding element type (line 56)
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_291450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    int_291451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 35), list_291450, int_291451)
    # Adding element type (line 56)
    int_291452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 35), list_291450, int_291452)
    # Adding element type (line 56)
    int_291453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 35), list_291450, int_291453)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 12), list_291441, list_291450)
    
    # Assigning a type to the variable 'A' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'A', list_291441)
    
    # Call to imshow(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'A' (line 57)
    A_291462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 40), 'A', False)
    # Processing the call keyword arguments (line 57)
    unicode_291463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 57), 'unicode', u'nearest')
    keyword_291464 = unicode_291463
    kwargs_291465 = {'interpolation': keyword_291464}
    
    # Call to add_subplot(...): (line 57)
    # Processing the call arguments (line 57)
    int_291456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 24), 'int')
    int_291457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 27), 'int')
    int_291458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 30), 'int')
    # Processing the call keyword arguments (line 57)
    kwargs_291459 = {}
    # Getting the type of 'fig' (line 57)
    fig_291454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'fig', False)
    # Obtaining the member 'add_subplot' of a type (line 57)
    add_subplot_291455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), fig_291454, 'add_subplot')
    # Calling add_subplot(args, kwargs) (line 57)
    add_subplot_call_result_291460 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), add_subplot_291455, *[int_291456, int_291457, int_291458], **kwargs_291459)
    
    # Obtaining the member 'imshow' of a type (line 57)
    imshow_291461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), add_subplot_call_result_291460, 'imshow')
    # Calling imshow(args, kwargs) (line 57)
    imshow_call_result_291466 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), imshow_291461, *[A_291462], **kwargs_291465)
    
    
    # Assigning a List to a Name (line 58):
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_291467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_291468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    int_291469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 13), list_291468, int_291469)
    # Adding element type (line 58)
    int_291470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 13), list_291468, int_291470)
    # Adding element type (line 58)
    int_291471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 13), list_291468, int_291471)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 12), list_291467, list_291468)
    # Adding element type (line 58)
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_291472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    int_291473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 24), list_291472, int_291473)
    # Adding element type (line 58)
    int_291474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 24), list_291472, int_291474)
    # Adding element type (line 58)
    int_291475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 24), list_291472, int_291475)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 12), list_291467, list_291472)
    # Adding element type (line 58)
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_291476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    int_291477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 35), list_291476, int_291477)
    # Adding element type (line 58)
    int_291478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 35), list_291476, int_291478)
    # Adding element type (line 58)
    int_291479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 35), list_291476, int_291479)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 12), list_291467, list_291476)
    
    # Assigning a type to the variable 'A' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'A', list_291467)
    
    # Call to imshow(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'A' (line 59)
    A_291488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 40), 'A', False)
    # Processing the call keyword arguments (line 59)
    unicode_291489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 57), 'unicode', u'bilinear')
    keyword_291490 = unicode_291489
    kwargs_291491 = {'interpolation': keyword_291490}
    
    # Call to add_subplot(...): (line 59)
    # Processing the call arguments (line 59)
    int_291482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 24), 'int')
    int_291483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 27), 'int')
    int_291484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 30), 'int')
    # Processing the call keyword arguments (line 59)
    kwargs_291485 = {}
    # Getting the type of 'fig' (line 59)
    fig_291480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'fig', False)
    # Obtaining the member 'add_subplot' of a type (line 59)
    add_subplot_291481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), fig_291480, 'add_subplot')
    # Calling add_subplot(args, kwargs) (line 59)
    add_subplot_call_result_291486 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), add_subplot_291481, *[int_291482, int_291483, int_291484], **kwargs_291485)
    
    # Obtaining the member 'imshow' of a type (line 59)
    imshow_291487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), add_subplot_call_result_291486, 'imshow')
    # Calling imshow(args, kwargs) (line 59)
    imshow_call_result_291492 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), imshow_291487, *[A_291488], **kwargs_291491)
    
    
    # Assigning a List to a Name (line 60):
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_291493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    # Adding element type (line 60)
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_291494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    # Adding element type (line 60)
    int_291495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_291494, int_291495)
    # Adding element type (line 60)
    int_291496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_291494, int_291496)
    # Adding element type (line 60)
    int_291497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_291494, int_291497)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 12), list_291493, list_291494)
    # Adding element type (line 60)
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_291498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    # Adding element type (line 60)
    int_291499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 24), list_291498, int_291499)
    # Adding element type (line 60)
    int_291500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 24), list_291498, int_291500)
    # Adding element type (line 60)
    int_291501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 24), list_291498, int_291501)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 12), list_291493, list_291498)
    # Adding element type (line 60)
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_291502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    # Adding element type (line 60)
    int_291503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 35), list_291502, int_291503)
    # Adding element type (line 60)
    int_291504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 35), list_291502, int_291504)
    # Adding element type (line 60)
    int_291505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 35), list_291502, int_291505)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 12), list_291493, list_291502)
    
    # Assigning a type to the variable 'A' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'A', list_291493)
    
    # Call to imshow(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'A' (line 61)
    A_291514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 40), 'A', False)
    # Processing the call keyword arguments (line 61)
    unicode_291515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 57), 'unicode', u'bicubic')
    keyword_291516 = unicode_291515
    kwargs_291517 = {'interpolation': keyword_291516}
    
    # Call to add_subplot(...): (line 61)
    # Processing the call arguments (line 61)
    int_291508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 24), 'int')
    int_291509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 27), 'int')
    int_291510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 30), 'int')
    # Processing the call keyword arguments (line 61)
    kwargs_291511 = {}
    # Getting the type of 'fig' (line 61)
    fig_291506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'fig', False)
    # Obtaining the member 'add_subplot' of a type (line 61)
    add_subplot_291507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), fig_291506, 'add_subplot')
    # Calling add_subplot(args, kwargs) (line 61)
    add_subplot_call_result_291512 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), add_subplot_291507, *[int_291508, int_291509, int_291510], **kwargs_291511)
    
    # Obtaining the member 'imshow' of a type (line 61)
    imshow_291513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), add_subplot_call_result_291512, 'imshow')
    # Calling imshow(args, kwargs) (line 61)
    imshow_call_result_291518 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), imshow_291513, *[A_291514], **kwargs_291517)
    
    # SSA join for if statement (line 54)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 63):
    
    # Call to range(...): (line 63)
    # Processing the call arguments (line 63)
    int_291520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 14), 'int')
    # Processing the call keyword arguments (line 63)
    kwargs_291521 = {}
    # Getting the type of 'range' (line 63)
    range_291519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'range', False)
    # Calling range(args, kwargs) (line 63)
    range_call_result_291522 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), range_291519, *[int_291520], **kwargs_291521)
    
    # Assigning a type to the variable 'x' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'x', range_call_result_291522)
    
    # Call to plot(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'x' (line 64)
    x_291531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'x', False)
    # Getting the type of 'x' (line 64)
    x_291532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 37), 'x', False)
    # Processing the call keyword arguments (line 64)
    kwargs_291533 = {}
    
    # Call to add_subplot(...): (line 64)
    # Processing the call arguments (line 64)
    int_291525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 20), 'int')
    int_291526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 23), 'int')
    int_291527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 26), 'int')
    # Processing the call keyword arguments (line 64)
    kwargs_291528 = {}
    # Getting the type of 'fig' (line 64)
    fig_291523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'fig', False)
    # Obtaining the member 'add_subplot' of a type (line 64)
    add_subplot_291524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 4), fig_291523, 'add_subplot')
    # Calling add_subplot(args, kwargs) (line 64)
    add_subplot_call_result_291529 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), add_subplot_291524, *[int_291525, int_291526, int_291527], **kwargs_291528)
    
    # Obtaining the member 'plot' of a type (line 64)
    plot_291530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 4), add_subplot_call_result_291529, 'plot')
    # Calling plot(args, kwargs) (line 64)
    plot_call_result_291534 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), plot_291530, *[x_291531, x_291532], **kwargs_291533)
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'six' (line 66)
    six_291535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'six')
    # Obtaining the member 'PY2' of a type (line 66)
    PY2_291536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 7), six_291535, 'PY2')
    
    # Getting the type of 'format' (line 66)
    format_291537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'format')
    unicode_291538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 29), 'unicode', u'ps')
    # Applying the binary operator '==' (line 66)
    result_eq_291539 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 19), '==', format_291537, unicode_291538)
    
    # Applying the binary operator 'and' (line 66)
    result_and_keyword_291540 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 7), 'and', PY2_291536, result_eq_291539)
    
    # Testing the type of an if condition (line 66)
    if_condition_291541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 4), result_and_keyword_291540)
    # Assigning a type to the variable 'if_condition_291541' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'if_condition_291541', if_condition_291541)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 67):
    
    # Call to StringIO(...): (line 67)
    # Processing the call keyword arguments (line 67)
    kwargs_291544 = {}
    # Getting the type of 'io' (line 67)
    io_291542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'io', False)
    # Obtaining the member 'StringIO' of a type (line 67)
    StringIO_291543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 17), io_291542, 'StringIO')
    # Calling StringIO(args, kwargs) (line 67)
    StringIO_call_result_291545 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), StringIO_291543, *[], **kwargs_291544)
    
    # Assigning a type to the variable 'stdout' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stdout', StringIO_call_result_291545)
    # SSA branch for the else part of an if statement (line 66)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 69):
    
    # Call to getattr(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'sys' (line 69)
    sys_291547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 69)
    stdout_291548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 25), sys_291547, 'stdout')
    unicode_291549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 37), 'unicode', u'buffer')
    # Getting the type of 'sys' (line 69)
    sys_291550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 47), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 69)
    stdout_291551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 47), sys_291550, 'stdout')
    # Processing the call keyword arguments (line 69)
    kwargs_291552 = {}
    # Getting the type of 'getattr' (line 69)
    getattr_291546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'getattr', False)
    # Calling getattr(args, kwargs) (line 69)
    getattr_call_result_291553 = invoke(stypy.reporting.localization.Localization(__file__, 69, 17), getattr_291546, *[stdout_291548, unicode_291549, stdout_291551], **kwargs_291552)
    
    # Assigning a type to the variable 'stdout' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stdout', getattr_call_result_291553)
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to savefig(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'stdout' (line 70)
    stdout_291556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'stdout', False)
    # Processing the call keyword arguments (line 70)
    # Getting the type of 'format' (line 70)
    format_291557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 31), 'format', False)
    keyword_291558 = format_291557
    kwargs_291559 = {'format': keyword_291558}
    # Getting the type of 'fig' (line 70)
    fig_291554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'fig', False)
    # Obtaining the member 'savefig' of a type (line 70)
    savefig_291555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 4), fig_291554, 'savefig')
    # Calling savefig(args, kwargs) (line 70)
    savefig_call_result_291560 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), savefig_291555, *[stdout_291556], **kwargs_291559)
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'six' (line 71)
    six_291561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 7), 'six')
    # Obtaining the member 'PY2' of a type (line 71)
    PY2_291562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 7), six_291561, 'PY2')
    
    # Getting the type of 'format' (line 71)
    format_291563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'format')
    unicode_291564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'unicode', u'ps')
    # Applying the binary operator '==' (line 71)
    result_eq_291565 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 19), '==', format_291563, unicode_291564)
    
    # Applying the binary operator 'and' (line 71)
    result_and_keyword_291566 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 7), 'and', PY2_291562, result_eq_291565)
    
    # Testing the type of an if condition (line 71)
    if_condition_291567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 4), result_and_keyword_291566)
    # Assigning a type to the variable 'if_condition_291567' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'if_condition_291567', if_condition_291567)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to write(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Call to getvalue(...): (line 72)
    # Processing the call keyword arguments (line 72)
    kwargs_291573 = {}
    # Getting the type of 'stdout' (line 72)
    stdout_291571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'stdout', False)
    # Obtaining the member 'getvalue' of a type (line 72)
    getvalue_291572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 25), stdout_291571, 'getvalue')
    # Calling getvalue(args, kwargs) (line 72)
    getvalue_call_result_291574 = invoke(stypy.reporting.localization.Localization(__file__, 72, 25), getvalue_291572, *[], **kwargs_291573)
    
    # Processing the call keyword arguments (line 72)
    kwargs_291575 = {}
    # Getting the type of 'sys' (line 72)
    sys_291568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 72)
    stdout_291569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), sys_291568, 'stdout')
    # Obtaining the member 'write' of a type (line 72)
    write_291570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), stdout_291569, 'write')
    # Calling write(args, kwargs) (line 72)
    write_call_result_291576 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), write_291570, *[getvalue_call_result_291574], **kwargs_291575)
    
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 75)
    # Getting the type of 'sde' (line 75)
    sde_291577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 7), 'sde')
    # Getting the type of 'None' (line 75)
    None_291578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'None')
    
    (may_be_291579, more_types_in_union_291580) = may_be_none(sde_291577, None_291578)

    if may_be_291579:

        if more_types_in_union_291580:
            # Runtime conditional SSA (line 75)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to pop(...): (line 76)
        # Processing the call arguments (line 76)
        unicode_291584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 23), 'unicode', u'SOURCE_DATE_EPOCH')
        # Getting the type of 'None' (line 76)
        None_291585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'None', False)
        # Processing the call keyword arguments (line 76)
        kwargs_291586 = {}
        # Getting the type of 'os' (line 76)
        os_291581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'os', False)
        # Obtaining the member 'environ' of a type (line 76)
        environ_291582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), os_291581, 'environ')
        # Obtaining the member 'pop' of a type (line 76)
        pop_291583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), environ_291582, 'pop')
        # Calling pop(args, kwargs) (line 76)
        pop_call_result_291587 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), pop_291583, *[unicode_291584, None_291585], **kwargs_291586)
        

        if more_types_in_union_291580:
            # Runtime conditional SSA for else branch (line 75)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_291579) or more_types_in_union_291580):
        
        # Assigning a Name to a Subscript (line 78):
        # Getting the type of 'sde' (line 78)
        sde_291588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 42), 'sde')
        # Getting the type of 'os' (line 78)
        os_291589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'os')
        # Obtaining the member 'environ' of a type (line 78)
        environ_291590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), os_291589, 'environ')
        unicode_291591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 19), 'unicode', u'SOURCE_DATE_EPOCH')
        # Storing an element on a container (line 78)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 8), environ_291590, (unicode_291591, sde_291588))

        if (may_be_291579 and more_types_in_union_291580):
            # SSA join for if statement (line 75)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_determinism_save(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_determinism_save' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_291592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_291592)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_determinism_save'
    return stypy_return_type_291592

# Assigning a type to the variable '_determinism_save' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '_determinism_save', _determinism_save)

@norecursion
def _determinism_check(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    unicode_291593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 31), 'unicode', u'mhi')
    unicode_291594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 45), 'unicode', u'pdf')
    # Getting the type of 'False' (line 81)
    False_291595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 59), 'False')
    defaults = [unicode_291593, unicode_291594, False_291595]
    # Create a new context for function '_determinism_check'
    module_type_store = module_type_store.open_function_context('_determinism_check', 81, 0, False)
    
    # Passed parameters checking function
    _determinism_check.stypy_localization = localization
    _determinism_check.stypy_type_of_self = None
    _determinism_check.stypy_type_store = module_type_store
    _determinism_check.stypy_function_name = '_determinism_check'
    _determinism_check.stypy_param_names_list = ['objects', 'format', 'usetex']
    _determinism_check.stypy_varargs_param_name = None
    _determinism_check.stypy_kwargs_param_name = None
    _determinism_check.stypy_call_defaults = defaults
    _determinism_check.stypy_call_varargs = varargs
    _determinism_check.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_determinism_check', ['objects', 'format', 'usetex'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_determinism_check', localization, ['objects', 'format', 'usetex'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_determinism_check(...)' code ##################

    unicode_291596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, (-1)), 'unicode', u'\n    Output three times the same graphs and checks that the outputs are exactly\n    the same.\n\n    Parameters\n    ----------\n    objects : str\n        contains characters corresponding to objects to be included in the test\n        document: \'m\' for markers, \'h\' for hatch patterns, \'i\' for images. The\n        default value is "mhi", so that the test includes all these objects.\n    format : str\n        format string. The default value is "pdf".\n    ')
    
    # Assigning a List to a Name (line 95):
    
    # Obtaining an instance of the builtin type 'list' (line 95)
    list_291597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 95)
    
    # Assigning a type to the variable 'plots' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'plots', list_291597)
    
    
    # Call to range(...): (line 96)
    # Processing the call arguments (line 96)
    int_291599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 19), 'int')
    # Processing the call keyword arguments (line 96)
    kwargs_291600 = {}
    # Getting the type of 'range' (line 96)
    range_291598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'range', False)
    # Calling range(args, kwargs) (line 96)
    range_call_result_291601 = invoke(stypy.reporting.localization.Localization(__file__, 96, 13), range_291598, *[int_291599], **kwargs_291600)
    
    # Testing the type of a for loop iterable (line 96)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 96, 4), range_call_result_291601)
    # Getting the type of the for loop variable (line 96)
    for_loop_var_291602 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 96, 4), range_call_result_291601)
    # Assigning a type to the variable 'i' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'i', for_loop_var_291602)
    # SSA begins for a for statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 97):
    
    # Call to check_output(...): (line 97)
    # Processing the call arguments (line 97)
    
    # Obtaining an instance of the builtin type 'list' (line 97)
    list_291604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 97)
    # Adding element type (line 97)
    # Getting the type of 'sys' (line 97)
    sys_291605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 31), 'sys', False)
    # Obtaining the member 'executable' of a type (line 97)
    executable_291606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 31), sys_291605, 'executable')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 30), list_291604, executable_291606)
    # Adding element type (line 97)
    unicode_291607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 47), 'unicode', u'-R')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 30), list_291604, unicode_291607)
    # Adding element type (line 97)
    unicode_291608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 53), 'unicode', u'-c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 30), list_291604, unicode_291608)
    # Adding element type (line 97)
    unicode_291609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 31), 'unicode', u'import matplotlib; matplotlib._called_from_pytest = True; matplotlib.use(%r); from matplotlib.testing.determinism import _determinism_save;_determinism_save(%r,%r,%r)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_291610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    # Getting the type of 'format' (line 104)
    format_291611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'format', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 34), tuple_291610, format_291611)
    # Adding element type (line 104)
    # Getting the type of 'objects' (line 104)
    objects_291612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 42), 'objects', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 34), tuple_291610, objects_291612)
    # Adding element type (line 104)
    # Getting the type of 'format' (line 104)
    format_291613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 51), 'format', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 34), tuple_291610, format_291613)
    # Adding element type (line 104)
    # Getting the type of 'usetex' (line 104)
    usetex_291614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 59), 'usetex', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 34), tuple_291610, usetex_291614)
    
    # Applying the binary operator '%' (line 98)
    result_mod_291615 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 31), '%', unicode_291609, tuple_291610)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 30), list_291604, result_mod_291615)
    
    # Processing the call keyword arguments (line 97)
    kwargs_291616 = {}
    # Getting the type of 'check_output' (line 97)
    check_output_291603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 17), 'check_output', False)
    # Calling check_output(args, kwargs) (line 97)
    check_output_call_result_291617 = invoke(stypy.reporting.localization.Localization(__file__, 97, 17), check_output_291603, *[list_291604], **kwargs_291616)
    
    # Assigning a type to the variable 'result' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'result', check_output_call_result_291617)
    
    # Call to append(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'result' (line 105)
    result_291620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'result', False)
    # Processing the call keyword arguments (line 105)
    kwargs_291621 = {}
    # Getting the type of 'plots' (line 105)
    plots_291618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'plots', False)
    # Obtaining the member 'append' of a type (line 105)
    append_291619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), plots_291618, 'append')
    # Calling append(args, kwargs) (line 105)
    append_call_result_291622 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), append_291619, *[result_291620], **kwargs_291621)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    int_291623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 19), 'int')
    slice_291624 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 106, 13), int_291623, None, None)
    # Getting the type of 'plots' (line 106)
    plots_291625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'plots')
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___291626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 13), plots_291625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_291627 = invoke(stypy.reporting.localization.Localization(__file__, 106, 13), getitem___291626, slice_291624)
    
    # Testing the type of a for loop iterable (line 106)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 106, 4), subscript_call_result_291627)
    # Getting the type of the for loop variable (line 106)
    for_loop_var_291628 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 106, 4), subscript_call_result_291627)
    # Assigning a type to the variable 'p' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'p', for_loop_var_291628)
    # SSA begins for a for statement (line 106)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'usetex' (line 107)
    usetex_291629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'usetex')
    # Testing the type of an if condition (line 107)
    if_condition_291630 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 8), usetex_291629)
    # Assigning a type to the variable 'if_condition_291630' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'if_condition_291630', if_condition_291630)
    # SSA begins for if statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'p' (line 108)
    p_291631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'p')
    
    # Obtaining the type of the subscript
    int_291632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 26), 'int')
    # Getting the type of 'plots' (line 108)
    plots_291633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'plots')
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___291634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 20), plots_291633, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_291635 = invoke(stypy.reporting.localization.Localization(__file__, 108, 20), getitem___291634, int_291632)
    
    # Applying the binary operator '!=' (line 108)
    result_ne_291636 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 15), '!=', p_291631, subscript_call_result_291635)
    
    # Testing the type of an if condition (line 108)
    if_condition_291637 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 12), result_ne_291636)
    # Assigning a type to the variable 'if_condition_291637' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'if_condition_291637', if_condition_291637)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to skip(...): (line 109)
    # Processing the call arguments (line 109)
    unicode_291640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 28), 'unicode', u'failed, maybe due to ghostscript timestamps')
    # Processing the call keyword arguments (line 109)
    kwargs_291641 = {}
    # Getting the type of 'pytest' (line 109)
    pytest_291638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'pytest', False)
    # Obtaining the member 'skip' of a type (line 109)
    skip_291639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), pytest_291638, 'skip')
    # Calling skip(args, kwargs) (line 109)
    skip_call_result_291642 = invoke(stypy.reporting.localization.Localization(__file__, 109, 16), skip_291639, *[unicode_291640], **kwargs_291641)
    
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 107)
    module_type_store.open_ssa_branch('else')
    # Evaluating assert statement condition
    
    # Getting the type of 'p' (line 111)
    p_291643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'p')
    
    # Obtaining the type of the subscript
    int_291644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 30), 'int')
    # Getting the type of 'plots' (line 111)
    plots_291645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'plots')
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___291646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), plots_291645, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_291647 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), getitem___291646, int_291644)
    
    # Applying the binary operator '==' (line 111)
    result_eq_291648 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 19), '==', p_291643, subscript_call_result_291647)
    
    # SSA join for if statement (line 107)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_determinism_check(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_determinism_check' in the type store
    # Getting the type of 'stypy_return_type' (line 81)
    stypy_return_type_291649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_291649)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_determinism_check'
    return stypy_return_type_291649

# Assigning a type to the variable '_determinism_check' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), '_determinism_check', _determinism_check)

@norecursion
def _determinism_source_date_epoch(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_291650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 59), 'str', 'CreationDate')
    defaults = [str_291650]
    # Create a new context for function '_determinism_source_date_epoch'
    module_type_store = module_type_store.open_function_context('_determinism_source_date_epoch', 114, 0, False)
    
    # Passed parameters checking function
    _determinism_source_date_epoch.stypy_localization = localization
    _determinism_source_date_epoch.stypy_type_of_self = None
    _determinism_source_date_epoch.stypy_type_store = module_type_store
    _determinism_source_date_epoch.stypy_function_name = '_determinism_source_date_epoch'
    _determinism_source_date_epoch.stypy_param_names_list = ['format', 'string', 'keyword']
    _determinism_source_date_epoch.stypy_varargs_param_name = None
    _determinism_source_date_epoch.stypy_kwargs_param_name = None
    _determinism_source_date_epoch.stypy_call_defaults = defaults
    _determinism_source_date_epoch.stypy_call_varargs = varargs
    _determinism_source_date_epoch.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_determinism_source_date_epoch', ['format', 'string', 'keyword'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_determinism_source_date_epoch', localization, ['format', 'string', 'keyword'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_determinism_source_date_epoch(...)' code ##################

    unicode_291651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, (-1)), 'unicode', u'\n    Test SOURCE_DATE_EPOCH support. Output a document with the envionment\n    variable SOURCE_DATE_EPOCH set to 2000-01-01 00:00 UTC and check that the\n    document contains the timestamp that corresponds to this date (given as an\n    argument).\n\n    Parameters\n    ----------\n    format : str\n        format string, such as "pdf".\n    string : str\n        timestamp string for 2000-01-01 00:00 UTC.\n    keyword : bytes\n        a string to look at when searching for the timestamp in the document\n        (used in case the test fails).\n    ')
    
    # Assigning a Call to a Name (line 131):
    
    # Call to check_output(...): (line 131)
    # Processing the call arguments (line 131)
    
    # Obtaining an instance of the builtin type 'list' (line 131)
    list_291653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 131)
    # Adding element type (line 131)
    # Getting the type of 'sys' (line 131)
    sys_291654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'sys', False)
    # Obtaining the member 'executable' of a type (line 131)
    executable_291655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 25), sys_291654, 'executable')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 24), list_291653, executable_291655)
    # Adding element type (line 131)
    unicode_291656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 41), 'unicode', u'-R')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 24), list_291653, unicode_291656)
    # Adding element type (line 131)
    unicode_291657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 47), 'unicode', u'-c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 24), list_291653, unicode_291657)
    # Adding element type (line 131)
    unicode_291658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 25), 'unicode', u'import matplotlib; matplotlib._called_from_pytest = True; matplotlib.use(%r); from matplotlib.testing.determinism import _determinism_save;_determinism_save(%r,%r)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 138)
    tuple_291659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 138)
    # Adding element type (line 138)
    # Getting the type of 'format' (line 138)
    format_291660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'format', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 28), tuple_291659, format_291660)
    # Adding element type (line 138)
    unicode_291661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 36), 'unicode', u'')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 28), tuple_291659, unicode_291661)
    # Adding element type (line 138)
    # Getting the type of 'format' (line 138)
    format_291662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 40), 'format', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 28), tuple_291659, format_291662)
    
    # Applying the binary operator '%' (line 132)
    result_mod_291663 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 25), '%', unicode_291658, tuple_291659)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 24), list_291653, result_mod_291663)
    
    # Processing the call keyword arguments (line 131)
    kwargs_291664 = {}
    # Getting the type of 'check_output' (line 131)
    check_output_291652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'check_output', False)
    # Calling check_output(args, kwargs) (line 131)
    check_output_call_result_291665 = invoke(stypy.reporting.localization.Localization(__file__, 131, 11), check_output_291652, *[list_291653], **kwargs_291664)
    
    # Assigning a type to the variable 'buff' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'buff', check_output_call_result_291665)
    
    # Assigning a Call to a Name (line 139):
    
    # Call to compile(...): (line 139)
    # Processing the call arguments (line 139)
    str_291668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 30), 'str', '.*')
    # Getting the type of 'keyword' (line 139)
    keyword_291669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 38), 'keyword', False)
    # Applying the binary operator '+' (line 139)
    result_add_291670 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 30), '+', str_291668, keyword_291669)
    
    str_291671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 48), 'str', '.*')
    # Applying the binary operator '+' (line 139)
    result_add_291672 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 46), '+', result_add_291670, str_291671)
    
    # Processing the call keyword arguments (line 139)
    kwargs_291673 = {}
    # Getting the type of 're' (line 139)
    re_291666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 're', False)
    # Obtaining the member 'compile' of a type (line 139)
    compile_291667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 19), re_291666, 'compile')
    # Calling compile(args, kwargs) (line 139)
    compile_call_result_291674 = invoke(stypy.reporting.localization.Localization(__file__, 139, 19), compile_291667, *[result_add_291672], **kwargs_291673)
    
    # Assigning a type to the variable 'find_keyword' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'find_keyword', compile_call_result_291674)
    
    # Assigning a Call to a Name (line 140):
    
    # Call to search(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'buff' (line 140)
    buff_291677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'buff', False)
    # Processing the call keyword arguments (line 140)
    kwargs_291678 = {}
    # Getting the type of 'find_keyword' (line 140)
    find_keyword_291675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 10), 'find_keyword', False)
    # Obtaining the member 'search' of a type (line 140)
    search_291676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 10), find_keyword_291675, 'search')
    # Calling search(args, kwargs) (line 140)
    search_call_result_291679 = invoke(stypy.reporting.localization.Localization(__file__, 140, 10), search_291676, *[buff_291677], **kwargs_291678)
    
    # Assigning a type to the variable 'key' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'key', search_call_result_291679)
    
    # Getting the type of 'key' (line 141)
    key_291680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 7), 'key')
    # Testing the type of an if condition (line 141)
    if_condition_291681 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 4), key_291680)
    # Assigning a type to the variable 'if_condition_291681' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'if_condition_291681', if_condition_291681)
    # SSA begins for if statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 142)
    # Processing the call arguments (line 142)
    
    # Call to group(...): (line 142)
    # Processing the call keyword arguments (line 142)
    kwargs_291685 = {}
    # Getting the type of 'key' (line 142)
    key_291683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 14), 'key', False)
    # Obtaining the member 'group' of a type (line 142)
    group_291684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 14), key_291683, 'group')
    # Calling group(args, kwargs) (line 142)
    group_call_result_291686 = invoke(stypy.reporting.localization.Localization(__file__, 142, 14), group_291684, *[], **kwargs_291685)
    
    # Processing the call keyword arguments (line 142)
    kwargs_291687 = {}
    # Getting the type of 'print' (line 142)
    print_291682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'print', False)
    # Calling print(args, kwargs) (line 142)
    print_call_result_291688 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), print_291682, *[group_call_result_291686], **kwargs_291687)
    
    # SSA branch for the else part of an if statement (line 141)
    module_type_store.open_ssa_branch('else')
    
    # Call to print(...): (line 144)
    # Processing the call arguments (line 144)
    unicode_291690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 14), 'unicode', u'Timestamp keyword (%s) not found!')
    # Getting the type of 'keyword' (line 144)
    keyword_291691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 52), 'keyword', False)
    # Applying the binary operator '%' (line 144)
    result_mod_291692 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 14), '%', unicode_291690, keyword_291691)
    
    # Processing the call keyword arguments (line 144)
    kwargs_291693 = {}
    # Getting the type of 'print' (line 144)
    print_291689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'print', False)
    # Calling print(args, kwargs) (line 144)
    print_call_result_291694 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), print_291689, *[result_mod_291692], **kwargs_291693)
    
    # SSA join for if statement (line 141)
    module_type_store = module_type_store.join_ssa_context()
    
    # Evaluating assert statement condition
    
    # Getting the type of 'string' (line 145)
    string_291695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'string')
    # Getting the type of 'buff' (line 145)
    buff_291696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 21), 'buff')
    # Applying the binary operator 'in' (line 145)
    result_contains_291697 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), 'in', string_291695, buff_291696)
    
    
    # ################# End of '_determinism_source_date_epoch(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_determinism_source_date_epoch' in the type store
    # Getting the type of 'stypy_return_type' (line 114)
    stypy_return_type_291698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_291698)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_determinism_source_date_epoch'
    return stypy_return_type_291698

# Assigning a type to the variable '_determinism_source_date_epoch' (line 114)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), '_determinism_source_date_epoch', _determinism_source_date_epoch)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
