
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_array_almost_equal, assert_array_equal
5: from pytest import raises as assert_raises
6: from scipy.sparse.csgraph import (shortest_path, dijkstra, johnson,
7:     bellman_ford, construct_dist_matrix, NegativeCycleError)
8: 
9: 
10: directed_G = np.array([[0, 3, 3, 0, 0],
11:                        [0, 0, 0, 2, 4],
12:                        [0, 0, 0, 0, 0],
13:                        [1, 0, 0, 0, 0],
14:                        [2, 0, 0, 2, 0]], dtype=float)
15: 
16: undirected_G = np.array([[0, 3, 3, 1, 2],
17:                          [3, 0, 0, 2, 4],
18:                          [3, 0, 0, 0, 0],
19:                          [1, 2, 0, 0, 2],
20:                          [2, 4, 0, 2, 0]], dtype=float)
21: 
22: unweighted_G = (directed_G > 0).astype(float)
23: 
24: directed_SP = [[0, 3, 3, 5, 7],
25:                [3, 0, 6, 2, 4],
26:                [np.inf, np.inf, 0, np.inf, np.inf],
27:                [1, 4, 4, 0, 8],
28:                [2, 5, 5, 2, 0]]
29: 
30: directed_pred = np.array([[-9999, 0, 0, 1, 1],
31:                           [3, -9999, 0, 1, 1],
32:                           [-9999, -9999, -9999, -9999, -9999],
33:                           [3, 0, 0, -9999, 1],
34:                           [4, 0, 0, 4, -9999]], dtype=float)
35: 
36: undirected_SP = np.array([[0, 3, 3, 1, 2],
37:                           [3, 0, 6, 2, 4],
38:                           [3, 6, 0, 4, 5],
39:                           [1, 2, 4, 0, 2],
40:                           [2, 4, 5, 2, 0]], dtype=float)
41: 
42: undirected_SP_limit_2 = np.array([[0, np.inf, np.inf, 1, 2],
43:                                   [np.inf, 0, np.inf, 2, np.inf],
44:                                   [np.inf, np.inf, 0, np.inf, np.inf],
45:                                   [1, 2, np.inf, 0, 2],
46:                                   [2, np.inf, np.inf, 2, 0]], dtype=float)
47: 
48: undirected_SP_limit_0 = np.ones((5, 5), dtype=float) - np.eye(5)
49: undirected_SP_limit_0[undirected_SP_limit_0 > 0] = np.inf
50: 
51: undirected_pred = np.array([[-9999, 0, 0, 0, 0],
52:                             [1, -9999, 0, 1, 1],
53:                             [2, 0, -9999, 0, 0],
54:                             [3, 3, 0, -9999, 3],
55:                             [4, 4, 0, 4, -9999]], dtype=float)
56: 
57: methods = ['auto', 'FW', 'D', 'BF', 'J']
58: 
59: 
60: def test_dijkstra_limit():
61:     limits = [0, 2, np.inf]
62:     results = [undirected_SP_limit_0,
63:                undirected_SP_limit_2,
64:                undirected_SP]
65: 
66:     def check(limit, result):
67:         SP = dijkstra(undirected_G, directed=False, limit=limit)
68:         assert_array_almost_equal(SP, result)
69: 
70:     for limit, result in zip(limits, results):
71:         check(limit, result)
72: 
73: 
74: def test_directed():
75:     def check(method):
76:         SP = shortest_path(directed_G, method=method, directed=True,
77:                            overwrite=False)
78:         assert_array_almost_equal(SP, directed_SP)
79: 
80:     for method in methods:
81:         check(method)
82: 
83: 
84: def test_undirected():
85:     def check(method, directed_in):
86:         if directed_in:
87:             SP1 = shortest_path(directed_G, method=method, directed=False,
88:                                 overwrite=False)
89:             assert_array_almost_equal(SP1, undirected_SP)
90:         else:
91:             SP2 = shortest_path(undirected_G, method=method, directed=True,
92:                                 overwrite=False)
93:             assert_array_almost_equal(SP2, undirected_SP)
94: 
95:     for method in methods:
96:         for directed_in in (True, False):
97:             check(method, directed_in)
98: 
99: 
100: def test_shortest_path_indices():
101:     indices = np.arange(4)
102: 
103:     def check(func, indshape):
104:         outshape = indshape + (5,)
105:         SP = func(directed_G, directed=False,
106:                   indices=indices.reshape(indshape))
107:         assert_array_almost_equal(SP, undirected_SP[indices].reshape(outshape))
108: 
109:     for indshape in [(4,), (4, 1), (2, 2)]:
110:         for func in (dijkstra, bellman_ford, johnson, shortest_path):
111:             check(func, indshape)
112: 
113:     assert_raises(ValueError, shortest_path, directed_G, method='FW',
114:                   indices=indices)
115: 
116: 
117: def test_predecessors():
118:     SP_res = {True: directed_SP,
119:               False: undirected_SP}
120:     pred_res = {True: directed_pred,
121:                 False: undirected_pred}
122: 
123:     def check(method, directed):
124:         SP, pred = shortest_path(directed_G, method, directed=directed,
125:                                  overwrite=False,
126:                                  return_predecessors=True)
127:         assert_array_almost_equal(SP, SP_res[directed])
128:         assert_array_almost_equal(pred, pred_res[directed])
129: 
130:     for method in methods:
131:         for directed in (True, False):
132:             check(method, directed)
133: 
134: 
135: def test_construct_shortest_path():
136:     def check(method, directed):
137:         SP1, pred = shortest_path(directed_G,
138:                                   directed=directed,
139:                                   overwrite=False,
140:                                   return_predecessors=True)
141:         SP2 = construct_dist_matrix(directed_G, pred, directed=directed)
142:         assert_array_almost_equal(SP1, SP2)
143: 
144:     for method in methods:
145:         for directed in (True, False):
146:             check(method, directed)
147: 
148: 
149: def test_unweighted_path():
150:     def check(method, directed):
151:         SP1 = shortest_path(directed_G,
152:                             directed=directed,
153:                             overwrite=False,
154:                             unweighted=True)
155:         SP2 = shortest_path(unweighted_G,
156:                             directed=directed,
157:                             overwrite=False,
158:                             unweighted=False)
159:         assert_array_almost_equal(SP1, SP2)
160: 
161:     for method in methods:
162:         for directed in (True, False):
163:             check(method, directed)
164: 
165: 
166: def test_negative_cycles():
167:     # create a small graph with a negative cycle
168:     graph = np.ones([5, 5])
169:     graph.flat[::6] = 0
170:     graph[1, 2] = -2
171: 
172:     def check(method, directed):
173:         assert_raises(NegativeCycleError, shortest_path, graph, method,
174:                       directed)
175: 
176:     for method in ['FW', 'J', 'BF']:
177:         for directed in (True, False):
178:             check(method, directed)
179: 
180: 
181: def test_masked_input():
182:     G = np.ma.masked_equal(directed_G, 0)
183: 
184:     def check(method):
185:         SP = shortest_path(directed_G, method=method, directed=True,
186:                            overwrite=False)
187:         assert_array_almost_equal(SP, directed_SP)
188: 
189:     for method in methods:
190:         check(method)
191: 
192: 
193: def test_overwrite():
194:     G = np.array([[0, 3, 3, 1, 2],
195:                   [3, 0, 0, 2, 4],
196:                   [3, 0, 0, 0, 0],
197:                   [1, 2, 0, 0, 2],
198:                   [2, 4, 0, 2, 0]], dtype=float)
199:     foo = G.copy()
200:     shortest_path(foo, overwrite=False)
201:     assert_array_equal(foo, G)
202: 
203: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_383656 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_383656) is not StypyTypeError):

    if (import_383656 != 'pyd_module'):
        __import__(import_383656)
        sys_modules_383657 = sys.modules[import_383656]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_383657.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_383656)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_array_almost_equal, assert_array_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_383658 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_383658) is not StypyTypeError):

    if (import_383658 != 'pyd_module'):
        __import__(import_383658)
        sys_modules_383659 = sys.modules[import_383658]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_383659.module_type_store, module_type_store, ['assert_array_almost_equal', 'assert_array_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_383659, sys_modules_383659.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal, assert_array_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal', 'assert_array_equal'], [assert_array_almost_equal, assert_array_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_383658)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from pytest import assert_raises' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_383660 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest')

if (type(import_383660) is not StypyTypeError):

    if (import_383660 != 'pyd_module'):
        __import__(import_383660)
        sys_modules_383661 = sys.modules[import_383660]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', sys_modules_383661.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_383661, sys_modules_383661.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', import_383660)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.sparse.csgraph import shortest_path, dijkstra, johnson, bellman_ford, construct_dist_matrix, NegativeCycleError' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_383662 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.csgraph')

if (type(import_383662) is not StypyTypeError):

    if (import_383662 != 'pyd_module'):
        __import__(import_383662)
        sys_modules_383663 = sys.modules[import_383662]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.csgraph', sys_modules_383663.module_type_store, module_type_store, ['shortest_path', 'dijkstra', 'johnson', 'bellman_ford', 'construct_dist_matrix', 'NegativeCycleError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_383663, sys_modules_383663.module_type_store, module_type_store)
    else:
        from scipy.sparse.csgraph import shortest_path, dijkstra, johnson, bellman_ford, construct_dist_matrix, NegativeCycleError

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.csgraph', None, module_type_store, ['shortest_path', 'dijkstra', 'johnson', 'bellman_ford', 'construct_dist_matrix', 'NegativeCycleError'], [shortest_path, dijkstra, johnson, bellman_ford, construct_dist_matrix, NegativeCycleError])

else:
    # Assigning a type to the variable 'scipy.sparse.csgraph' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.csgraph', import_383662)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')


# Assigning a Call to a Name (line 10):

# Assigning a Call to a Name (line 10):

# Call to array(...): (line 10)
# Processing the call arguments (line 10)

# Obtaining an instance of the builtin type 'list' (line 10)
list_383666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'list' (line 10)
list_383667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
int_383668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 23), list_383667, int_383668)
# Adding element type (line 10)
int_383669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 23), list_383667, int_383669)
# Adding element type (line 10)
int_383670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 23), list_383667, int_383670)
# Adding element type (line 10)
int_383671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 23), list_383667, int_383671)
# Adding element type (line 10)
int_383672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 23), list_383667, int_383672)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 22), list_383666, list_383667)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'list' (line 11)
list_383673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
int_383674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 23), list_383673, int_383674)
# Adding element type (line 11)
int_383675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 23), list_383673, int_383675)
# Adding element type (line 11)
int_383676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 23), list_383673, int_383676)
# Adding element type (line 11)
int_383677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 23), list_383673, int_383677)
# Adding element type (line 11)
int_383678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 23), list_383673, int_383678)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 22), list_383666, list_383673)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'list' (line 12)
list_383679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
int_383680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 23), list_383679, int_383680)
# Adding element type (line 12)
int_383681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 23), list_383679, int_383681)
# Adding element type (line 12)
int_383682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 23), list_383679, int_383682)
# Adding element type (line 12)
int_383683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 23), list_383679, int_383683)
# Adding element type (line 12)
int_383684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 23), list_383679, int_383684)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 22), list_383666, list_383679)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'list' (line 13)
list_383685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
int_383686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 23), list_383685, int_383686)
# Adding element type (line 13)
int_383687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 23), list_383685, int_383687)
# Adding element type (line 13)
int_383688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 23), list_383685, int_383688)
# Adding element type (line 13)
int_383689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 23), list_383685, int_383689)
# Adding element type (line 13)
int_383690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 23), list_383685, int_383690)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 22), list_383666, list_383685)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'list' (line 14)
list_383691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_383692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 23), list_383691, int_383692)
# Adding element type (line 14)
int_383693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 23), list_383691, int_383693)
# Adding element type (line 14)
int_383694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 23), list_383691, int_383694)
# Adding element type (line 14)
int_383695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 23), list_383691, int_383695)
# Adding element type (line 14)
int_383696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 23), list_383691, int_383696)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 22), list_383666, list_383691)

# Processing the call keyword arguments (line 10)
# Getting the type of 'float' (line 14)
float_383697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 47), 'float', False)
keyword_383698 = float_383697
kwargs_383699 = {'dtype': keyword_383698}
# Getting the type of 'np' (line 10)
np_383664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 13), 'np', False)
# Obtaining the member 'array' of a type (line 10)
array_383665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 13), np_383664, 'array')
# Calling array(args, kwargs) (line 10)
array_call_result_383700 = invoke(stypy.reporting.localization.Localization(__file__, 10, 13), array_383665, *[list_383666], **kwargs_383699)

# Assigning a type to the variable 'directed_G' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'directed_G', array_call_result_383700)

# Assigning a Call to a Name (line 16):

# Assigning a Call to a Name (line 16):

# Call to array(...): (line 16)
# Processing the call arguments (line 16)

# Obtaining an instance of the builtin type 'list' (line 16)
list_383703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'list' (line 16)
list_383704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
int_383705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_383704, int_383705)
# Adding element type (line 16)
int_383706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_383704, int_383706)
# Adding element type (line 16)
int_383707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_383704, int_383707)
# Adding element type (line 16)
int_383708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_383704, int_383708)
# Adding element type (line 16)
int_383709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_383704, int_383709)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 24), list_383703, list_383704)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'list' (line 17)
list_383710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
int_383711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_383710, int_383711)
# Adding element type (line 17)
int_383712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_383710, int_383712)
# Adding element type (line 17)
int_383713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_383710, int_383713)
# Adding element type (line 17)
int_383714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_383710, int_383714)
# Adding element type (line 17)
int_383715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_383710, int_383715)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 24), list_383703, list_383710)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'list' (line 18)
list_383716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
int_383717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_383716, int_383717)
# Adding element type (line 18)
int_383718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_383716, int_383718)
# Adding element type (line 18)
int_383719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_383716, int_383719)
# Adding element type (line 18)
int_383720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_383716, int_383720)
# Adding element type (line 18)
int_383721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_383716, int_383721)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 24), list_383703, list_383716)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'list' (line 19)
list_383722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
int_383723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_383722, int_383723)
# Adding element type (line 19)
int_383724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_383722, int_383724)
# Adding element type (line 19)
int_383725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_383722, int_383725)
# Adding element type (line 19)
int_383726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_383722, int_383726)
# Adding element type (line 19)
int_383727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_383722, int_383727)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 24), list_383703, list_383722)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'list' (line 20)
list_383728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
int_383729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_383728, int_383729)
# Adding element type (line 20)
int_383730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_383728, int_383730)
# Adding element type (line 20)
int_383731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_383728, int_383731)
# Adding element type (line 20)
int_383732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_383728, int_383732)
# Adding element type (line 20)
int_383733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_383728, int_383733)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 24), list_383703, list_383728)

# Processing the call keyword arguments (line 16)
# Getting the type of 'float' (line 20)
float_383734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 49), 'float', False)
keyword_383735 = float_383734
kwargs_383736 = {'dtype': keyword_383735}
# Getting the type of 'np' (line 16)
np_383701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'np', False)
# Obtaining the member 'array' of a type (line 16)
array_383702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 15), np_383701, 'array')
# Calling array(args, kwargs) (line 16)
array_call_result_383737 = invoke(stypy.reporting.localization.Localization(__file__, 16, 15), array_383702, *[list_383703], **kwargs_383736)

# Assigning a type to the variable 'undirected_G' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'undirected_G', array_call_result_383737)

# Assigning a Call to a Name (line 22):

# Assigning a Call to a Name (line 22):

# Call to astype(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'float' (line 22)
float_383742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 39), 'float', False)
# Processing the call keyword arguments (line 22)
kwargs_383743 = {}

# Getting the type of 'directed_G' (line 22)
directed_G_383738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'directed_G', False)
int_383739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 29), 'int')
# Applying the binary operator '>' (line 22)
result_gt_383740 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 16), '>', directed_G_383738, int_383739)

# Obtaining the member 'astype' of a type (line 22)
astype_383741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 16), result_gt_383740, 'astype')
# Calling astype(args, kwargs) (line 22)
astype_call_result_383744 = invoke(stypy.reporting.localization.Localization(__file__, 22, 16), astype_383741, *[float_383742], **kwargs_383743)

# Assigning a type to the variable 'unweighted_G' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'unweighted_G', astype_call_result_383744)

# Assigning a List to a Name (line 24):

# Assigning a List to a Name (line 24):

# Obtaining an instance of the builtin type 'list' (line 24)
list_383745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'list' (line 24)
list_383746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
int_383747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_383746, int_383747)
# Adding element type (line 24)
int_383748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_383746, int_383748)
# Adding element type (line 24)
int_383749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_383746, int_383749)
# Adding element type (line 24)
int_383750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_383746, int_383750)
# Adding element type (line 24)
int_383751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_383746, int_383751)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_383745, list_383746)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'list' (line 25)
list_383752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
int_383753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_383752, int_383753)
# Adding element type (line 25)
int_383754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_383752, int_383754)
# Adding element type (line 25)
int_383755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_383752, int_383755)
# Adding element type (line 25)
int_383756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_383752, int_383756)
# Adding element type (line 25)
int_383757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_383752, int_383757)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_383745, list_383752)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'list' (line 26)
list_383758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
# Getting the type of 'np' (line 26)
np_383759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'np')
# Obtaining the member 'inf' of a type (line 26)
inf_383760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), np_383759, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), list_383758, inf_383760)
# Adding element type (line 26)
# Getting the type of 'np' (line 26)
np_383761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 24), 'np')
# Obtaining the member 'inf' of a type (line 26)
inf_383762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 24), np_383761, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), list_383758, inf_383762)
# Adding element type (line 26)
int_383763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), list_383758, int_383763)
# Adding element type (line 26)
# Getting the type of 'np' (line 26)
np_383764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 35), 'np')
# Obtaining the member 'inf' of a type (line 26)
inf_383765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 35), np_383764, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), list_383758, inf_383765)
# Adding element type (line 26)
# Getting the type of 'np' (line 26)
np_383766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 43), 'np')
# Obtaining the member 'inf' of a type (line 26)
inf_383767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 43), np_383766, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), list_383758, inf_383767)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_383745, list_383758)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'list' (line 27)
list_383768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
int_383769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), list_383768, int_383769)
# Adding element type (line 27)
int_383770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), list_383768, int_383770)
# Adding element type (line 27)
int_383771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), list_383768, int_383771)
# Adding element type (line 27)
int_383772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), list_383768, int_383772)
# Adding element type (line 27)
int_383773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), list_383768, int_383773)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_383745, list_383768)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'list' (line 28)
list_383774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)
int_383775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), list_383774, int_383775)
# Adding element type (line 28)
int_383776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), list_383774, int_383776)
# Adding element type (line 28)
int_383777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), list_383774, int_383777)
# Adding element type (line 28)
int_383778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), list_383774, int_383778)
# Adding element type (line 28)
int_383779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), list_383774, int_383779)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_383745, list_383774)

# Assigning a type to the variable 'directed_SP' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'directed_SP', list_383745)

# Assigning a Call to a Name (line 30):

# Assigning a Call to a Name (line 30):

# Call to array(...): (line 30)
# Processing the call arguments (line 30)

# Obtaining an instance of the builtin type 'list' (line 30)
list_383782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)
# Adding element type (line 30)

# Obtaining an instance of the builtin type 'list' (line 30)
list_383783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)
# Adding element type (line 30)
int_383784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 26), list_383783, int_383784)
# Adding element type (line 30)
int_383785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 26), list_383783, int_383785)
# Adding element type (line 30)
int_383786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 26), list_383783, int_383786)
# Adding element type (line 30)
int_383787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 26), list_383783, int_383787)
# Adding element type (line 30)
int_383788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 26), list_383783, int_383788)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), list_383782, list_383783)
# Adding element type (line 30)

# Obtaining an instance of the builtin type 'list' (line 31)
list_383789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)
# Adding element type (line 31)
int_383790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 26), list_383789, int_383790)
# Adding element type (line 31)
int_383791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 26), list_383789, int_383791)
# Adding element type (line 31)
int_383792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 26), list_383789, int_383792)
# Adding element type (line 31)
int_383793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 26), list_383789, int_383793)
# Adding element type (line 31)
int_383794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 26), list_383789, int_383794)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), list_383782, list_383789)
# Adding element type (line 30)

# Obtaining an instance of the builtin type 'list' (line 32)
list_383795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)
int_383796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 26), list_383795, int_383796)
# Adding element type (line 32)
int_383797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 26), list_383795, int_383797)
# Adding element type (line 32)
int_383798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 26), list_383795, int_383798)
# Adding element type (line 32)
int_383799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 48), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 26), list_383795, int_383799)
# Adding element type (line 32)
int_383800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 26), list_383795, int_383800)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), list_383782, list_383795)
# Adding element type (line 30)

# Obtaining an instance of the builtin type 'list' (line 33)
list_383801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)
int_383802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 26), list_383801, int_383802)
# Adding element type (line 33)
int_383803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 26), list_383801, int_383803)
# Adding element type (line 33)
int_383804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 26), list_383801, int_383804)
# Adding element type (line 33)
int_383805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 26), list_383801, int_383805)
# Adding element type (line 33)
int_383806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 26), list_383801, int_383806)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), list_383782, list_383801)
# Adding element type (line 30)

# Obtaining an instance of the builtin type 'list' (line 34)
list_383807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 34)
# Adding element type (line 34)
int_383808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 26), list_383807, int_383808)
# Adding element type (line 34)
int_383809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 26), list_383807, int_383809)
# Adding element type (line 34)
int_383810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 26), list_383807, int_383810)
# Adding element type (line 34)
int_383811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 26), list_383807, int_383811)
# Adding element type (line 34)
int_383812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 26), list_383807, int_383812)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), list_383782, list_383807)

# Processing the call keyword arguments (line 30)
# Getting the type of 'float' (line 34)
float_383813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 54), 'float', False)
keyword_383814 = float_383813
kwargs_383815 = {'dtype': keyword_383814}
# Getting the type of 'np' (line 30)
np_383780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'np', False)
# Obtaining the member 'array' of a type (line 30)
array_383781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 16), np_383780, 'array')
# Calling array(args, kwargs) (line 30)
array_call_result_383816 = invoke(stypy.reporting.localization.Localization(__file__, 30, 16), array_383781, *[list_383782], **kwargs_383815)

# Assigning a type to the variable 'directed_pred' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'directed_pred', array_call_result_383816)

# Assigning a Call to a Name (line 36):

# Assigning a Call to a Name (line 36):

# Call to array(...): (line 36)
# Processing the call arguments (line 36)

# Obtaining an instance of the builtin type 'list' (line 36)
list_383819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 36)
# Adding element type (line 36)

# Obtaining an instance of the builtin type 'list' (line 36)
list_383820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 36)
# Adding element type (line 36)
int_383821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 26), list_383820, int_383821)
# Adding element type (line 36)
int_383822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 26), list_383820, int_383822)
# Adding element type (line 36)
int_383823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 26), list_383820, int_383823)
# Adding element type (line 36)
int_383824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 26), list_383820, int_383824)
# Adding element type (line 36)
int_383825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 26), list_383820, int_383825)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), list_383819, list_383820)
# Adding element type (line 36)

# Obtaining an instance of the builtin type 'list' (line 37)
list_383826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 37)
# Adding element type (line 37)
int_383827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 26), list_383826, int_383827)
# Adding element type (line 37)
int_383828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 26), list_383826, int_383828)
# Adding element type (line 37)
int_383829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 26), list_383826, int_383829)
# Adding element type (line 37)
int_383830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 26), list_383826, int_383830)
# Adding element type (line 37)
int_383831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 26), list_383826, int_383831)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), list_383819, list_383826)
# Adding element type (line 36)

# Obtaining an instance of the builtin type 'list' (line 38)
list_383832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 38)
# Adding element type (line 38)
int_383833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 26), list_383832, int_383833)
# Adding element type (line 38)
int_383834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 26), list_383832, int_383834)
# Adding element type (line 38)
int_383835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 26), list_383832, int_383835)
# Adding element type (line 38)
int_383836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 26), list_383832, int_383836)
# Adding element type (line 38)
int_383837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 26), list_383832, int_383837)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), list_383819, list_383832)
# Adding element type (line 36)

# Obtaining an instance of the builtin type 'list' (line 39)
list_383838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 39)
# Adding element type (line 39)
int_383839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 26), list_383838, int_383839)
# Adding element type (line 39)
int_383840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 26), list_383838, int_383840)
# Adding element type (line 39)
int_383841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 26), list_383838, int_383841)
# Adding element type (line 39)
int_383842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 26), list_383838, int_383842)
# Adding element type (line 39)
int_383843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 26), list_383838, int_383843)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), list_383819, list_383838)
# Adding element type (line 36)

# Obtaining an instance of the builtin type 'list' (line 40)
list_383844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 40)
# Adding element type (line 40)
int_383845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 26), list_383844, int_383845)
# Adding element type (line 40)
int_383846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 26), list_383844, int_383846)
# Adding element type (line 40)
int_383847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 26), list_383844, int_383847)
# Adding element type (line 40)
int_383848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 26), list_383844, int_383848)
# Adding element type (line 40)
int_383849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 26), list_383844, int_383849)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), list_383819, list_383844)

# Processing the call keyword arguments (line 36)
# Getting the type of 'float' (line 40)
float_383850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 50), 'float', False)
keyword_383851 = float_383850
kwargs_383852 = {'dtype': keyword_383851}
# Getting the type of 'np' (line 36)
np_383817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'np', False)
# Obtaining the member 'array' of a type (line 36)
array_383818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), np_383817, 'array')
# Calling array(args, kwargs) (line 36)
array_call_result_383853 = invoke(stypy.reporting.localization.Localization(__file__, 36, 16), array_383818, *[list_383819], **kwargs_383852)

# Assigning a type to the variable 'undirected_SP' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'undirected_SP', array_call_result_383853)

# Assigning a Call to a Name (line 42):

# Assigning a Call to a Name (line 42):

# Call to array(...): (line 42)
# Processing the call arguments (line 42)

# Obtaining an instance of the builtin type 'list' (line 42)
list_383856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 42)
# Adding element type (line 42)

# Obtaining an instance of the builtin type 'list' (line 42)
list_383857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 42)
# Adding element type (line 42)
int_383858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 34), list_383857, int_383858)
# Adding element type (line 42)
# Getting the type of 'np' (line 42)
np_383859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 38), 'np', False)
# Obtaining the member 'inf' of a type (line 42)
inf_383860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 38), np_383859, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 34), list_383857, inf_383860)
# Adding element type (line 42)
# Getting the type of 'np' (line 42)
np_383861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 46), 'np', False)
# Obtaining the member 'inf' of a type (line 42)
inf_383862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 46), np_383861, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 34), list_383857, inf_383862)
# Adding element type (line 42)
int_383863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 34), list_383857, int_383863)
# Adding element type (line 42)
int_383864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 34), list_383857, int_383864)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 33), list_383856, list_383857)
# Adding element type (line 42)

# Obtaining an instance of the builtin type 'list' (line 43)
list_383865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 43)
# Adding element type (line 43)
# Getting the type of 'np' (line 43)
np_383866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 35), 'np', False)
# Obtaining the member 'inf' of a type (line 43)
inf_383867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 35), np_383866, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 34), list_383865, inf_383867)
# Adding element type (line 43)
int_383868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 34), list_383865, int_383868)
# Adding element type (line 43)
# Getting the type of 'np' (line 43)
np_383869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 46), 'np', False)
# Obtaining the member 'inf' of a type (line 43)
inf_383870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 46), np_383869, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 34), list_383865, inf_383870)
# Adding element type (line 43)
int_383871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 34), list_383865, int_383871)
# Adding element type (line 43)
# Getting the type of 'np' (line 43)
np_383872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 57), 'np', False)
# Obtaining the member 'inf' of a type (line 43)
inf_383873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 57), np_383872, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 34), list_383865, inf_383873)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 33), list_383856, list_383865)
# Adding element type (line 42)

# Obtaining an instance of the builtin type 'list' (line 44)
list_383874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 44)
# Adding element type (line 44)
# Getting the type of 'np' (line 44)
np_383875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'np', False)
# Obtaining the member 'inf' of a type (line 44)
inf_383876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 35), np_383875, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 34), list_383874, inf_383876)
# Adding element type (line 44)
# Getting the type of 'np' (line 44)
np_383877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 43), 'np', False)
# Obtaining the member 'inf' of a type (line 44)
inf_383878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 43), np_383877, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 34), list_383874, inf_383878)
# Adding element type (line 44)
int_383879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 51), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 34), list_383874, int_383879)
# Adding element type (line 44)
# Getting the type of 'np' (line 44)
np_383880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 54), 'np', False)
# Obtaining the member 'inf' of a type (line 44)
inf_383881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 54), np_383880, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 34), list_383874, inf_383881)
# Adding element type (line 44)
# Getting the type of 'np' (line 44)
np_383882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 62), 'np', False)
# Obtaining the member 'inf' of a type (line 44)
inf_383883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 62), np_383882, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 34), list_383874, inf_383883)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 33), list_383856, list_383874)
# Adding element type (line 42)

# Obtaining an instance of the builtin type 'list' (line 45)
list_383884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 45)
# Adding element type (line 45)
int_383885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 34), list_383884, int_383885)
# Adding element type (line 45)
int_383886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 34), list_383884, int_383886)
# Adding element type (line 45)
# Getting the type of 'np' (line 45)
np_383887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 41), 'np', False)
# Obtaining the member 'inf' of a type (line 45)
inf_383888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 41), np_383887, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 34), list_383884, inf_383888)
# Adding element type (line 45)
int_383889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 34), list_383884, int_383889)
# Adding element type (line 45)
int_383890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 52), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 34), list_383884, int_383890)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 33), list_383856, list_383884)
# Adding element type (line 42)

# Obtaining an instance of the builtin type 'list' (line 46)
list_383891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 46)
# Adding element type (line 46)
int_383892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), list_383891, int_383892)
# Adding element type (line 46)
# Getting the type of 'np' (line 46)
np_383893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 38), 'np', False)
# Obtaining the member 'inf' of a type (line 46)
inf_383894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 38), np_383893, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), list_383891, inf_383894)
# Adding element type (line 46)
# Getting the type of 'np' (line 46)
np_383895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'np', False)
# Obtaining the member 'inf' of a type (line 46)
inf_383896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 46), np_383895, 'inf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), list_383891, inf_383896)
# Adding element type (line 46)
int_383897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), list_383891, int_383897)
# Adding element type (line 46)
int_383898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), list_383891, int_383898)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 33), list_383856, list_383891)

# Processing the call keyword arguments (line 42)
# Getting the type of 'float' (line 46)
float_383899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 68), 'float', False)
keyword_383900 = float_383899
kwargs_383901 = {'dtype': keyword_383900}
# Getting the type of 'np' (line 42)
np_383854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'np', False)
# Obtaining the member 'array' of a type (line 42)
array_383855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), np_383854, 'array')
# Calling array(args, kwargs) (line 42)
array_call_result_383902 = invoke(stypy.reporting.localization.Localization(__file__, 42, 24), array_383855, *[list_383856], **kwargs_383901)

# Assigning a type to the variable 'undirected_SP_limit_2' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'undirected_SP_limit_2', array_call_result_383902)

# Assigning a BinOp to a Name (line 48):

# Assigning a BinOp to a Name (line 48):

# Call to ones(...): (line 48)
# Processing the call arguments (line 48)

# Obtaining an instance of the builtin type 'tuple' (line 48)
tuple_383905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 48)
# Adding element type (line 48)
int_383906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 33), tuple_383905, int_383906)
# Adding element type (line 48)
int_383907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 33), tuple_383905, int_383907)

# Processing the call keyword arguments (line 48)
# Getting the type of 'float' (line 48)
float_383908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 46), 'float', False)
keyword_383909 = float_383908
kwargs_383910 = {'dtype': keyword_383909}
# Getting the type of 'np' (line 48)
np_383903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'np', False)
# Obtaining the member 'ones' of a type (line 48)
ones_383904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 24), np_383903, 'ones')
# Calling ones(args, kwargs) (line 48)
ones_call_result_383911 = invoke(stypy.reporting.localization.Localization(__file__, 48, 24), ones_383904, *[tuple_383905], **kwargs_383910)


# Call to eye(...): (line 48)
# Processing the call arguments (line 48)
int_383914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 62), 'int')
# Processing the call keyword arguments (line 48)
kwargs_383915 = {}
# Getting the type of 'np' (line 48)
np_383912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 55), 'np', False)
# Obtaining the member 'eye' of a type (line 48)
eye_383913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 55), np_383912, 'eye')
# Calling eye(args, kwargs) (line 48)
eye_call_result_383916 = invoke(stypy.reporting.localization.Localization(__file__, 48, 55), eye_383913, *[int_383914], **kwargs_383915)

# Applying the binary operator '-' (line 48)
result_sub_383917 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 24), '-', ones_call_result_383911, eye_call_result_383916)

# Assigning a type to the variable 'undirected_SP_limit_0' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'undirected_SP_limit_0', result_sub_383917)

# Assigning a Attribute to a Subscript (line 49):

# Assigning a Attribute to a Subscript (line 49):
# Getting the type of 'np' (line 49)
np_383918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 51), 'np')
# Obtaining the member 'inf' of a type (line 49)
inf_383919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 51), np_383918, 'inf')
# Getting the type of 'undirected_SP_limit_0' (line 49)
undirected_SP_limit_0_383920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'undirected_SP_limit_0')

# Getting the type of 'undirected_SP_limit_0' (line 49)
undirected_SP_limit_0_383921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'undirected_SP_limit_0')
int_383922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 46), 'int')
# Applying the binary operator '>' (line 49)
result_gt_383923 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 22), '>', undirected_SP_limit_0_383921, int_383922)

# Storing an element on a container (line 49)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 0), undirected_SP_limit_0_383920, (result_gt_383923, inf_383919))

# Assigning a Call to a Name (line 51):

# Assigning a Call to a Name (line 51):

# Call to array(...): (line 51)
# Processing the call arguments (line 51)

# Obtaining an instance of the builtin type 'list' (line 51)
list_383926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 51)
# Adding element type (line 51)

# Obtaining an instance of the builtin type 'list' (line 51)
list_383927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 51)
# Adding element type (line 51)
int_383928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 28), list_383927, int_383928)
# Adding element type (line 51)
int_383929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 28), list_383927, int_383929)
# Adding element type (line 51)
int_383930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 28), list_383927, int_383930)
# Adding element type (line 51)
int_383931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 42), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 28), list_383927, int_383931)
# Adding element type (line 51)
int_383932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 45), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 28), list_383927, int_383932)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 27), list_383926, list_383927)
# Adding element type (line 51)

# Obtaining an instance of the builtin type 'list' (line 52)
list_383933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 52)
# Adding element type (line 52)
int_383934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 28), list_383933, int_383934)
# Adding element type (line 52)
int_383935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 28), list_383933, int_383935)
# Adding element type (line 52)
int_383936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 28), list_383933, int_383936)
# Adding element type (line 52)
int_383937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 42), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 28), list_383933, int_383937)
# Adding element type (line 52)
int_383938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 45), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 28), list_383933, int_383938)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 27), list_383926, list_383933)
# Adding element type (line 51)

# Obtaining an instance of the builtin type 'list' (line 53)
list_383939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 53)
# Adding element type (line 53)
int_383940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 28), list_383939, int_383940)
# Adding element type (line 53)
int_383941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 28), list_383939, int_383941)
# Adding element type (line 53)
int_383942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 28), list_383939, int_383942)
# Adding element type (line 53)
int_383943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 42), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 28), list_383939, int_383943)
# Adding element type (line 53)
int_383944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 45), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 28), list_383939, int_383944)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 27), list_383926, list_383939)
# Adding element type (line 51)

# Obtaining an instance of the builtin type 'list' (line 54)
list_383945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 54)
# Adding element type (line 54)
int_383946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 28), list_383945, int_383946)
# Adding element type (line 54)
int_383947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 28), list_383945, int_383947)
# Adding element type (line 54)
int_383948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 28), list_383945, int_383948)
# Adding element type (line 54)
int_383949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 28), list_383945, int_383949)
# Adding element type (line 54)
int_383950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 45), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 28), list_383945, int_383950)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 27), list_383926, list_383945)
# Adding element type (line 51)

# Obtaining an instance of the builtin type 'list' (line 55)
list_383951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 55)
# Adding element type (line 55)
int_383952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 28), list_383951, int_383952)
# Adding element type (line 55)
int_383953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 28), list_383951, int_383953)
# Adding element type (line 55)
int_383954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 28), list_383951, int_383954)
# Adding element type (line 55)
int_383955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 28), list_383951, int_383955)
# Adding element type (line 55)
int_383956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 28), list_383951, int_383956)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 27), list_383926, list_383951)

# Processing the call keyword arguments (line 51)
# Getting the type of 'float' (line 55)
float_383957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 56), 'float', False)
keyword_383958 = float_383957
kwargs_383959 = {'dtype': keyword_383958}
# Getting the type of 'np' (line 51)
np_383924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'np', False)
# Obtaining the member 'array' of a type (line 51)
array_383925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 18), np_383924, 'array')
# Calling array(args, kwargs) (line 51)
array_call_result_383960 = invoke(stypy.reporting.localization.Localization(__file__, 51, 18), array_383925, *[list_383926], **kwargs_383959)

# Assigning a type to the variable 'undirected_pred' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'undirected_pred', array_call_result_383960)

# Assigning a List to a Name (line 57):

# Assigning a List to a Name (line 57):

# Obtaining an instance of the builtin type 'list' (line 57)
list_383961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 57)
# Adding element type (line 57)
str_383962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 11), 'str', 'auto')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_383961, str_383962)
# Adding element type (line 57)
str_383963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 19), 'str', 'FW')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_383961, str_383963)
# Adding element type (line 57)
str_383964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'str', 'D')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_383961, str_383964)
# Adding element type (line 57)
str_383965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 30), 'str', 'BF')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_383961, str_383965)
# Adding element type (line 57)
str_383966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 36), 'str', 'J')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_383961, str_383966)

# Assigning a type to the variable 'methods' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'methods', list_383961)

@norecursion
def test_dijkstra_limit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_dijkstra_limit'
    module_type_store = module_type_store.open_function_context('test_dijkstra_limit', 60, 0, False)
    
    # Passed parameters checking function
    test_dijkstra_limit.stypy_localization = localization
    test_dijkstra_limit.stypy_type_of_self = None
    test_dijkstra_limit.stypy_type_store = module_type_store
    test_dijkstra_limit.stypy_function_name = 'test_dijkstra_limit'
    test_dijkstra_limit.stypy_param_names_list = []
    test_dijkstra_limit.stypy_varargs_param_name = None
    test_dijkstra_limit.stypy_kwargs_param_name = None
    test_dijkstra_limit.stypy_call_defaults = defaults
    test_dijkstra_limit.stypy_call_varargs = varargs
    test_dijkstra_limit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_dijkstra_limit', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_dijkstra_limit', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_dijkstra_limit(...)' code ##################

    
    # Assigning a List to a Name (line 61):
    
    # Assigning a List to a Name (line 61):
    
    # Obtaining an instance of the builtin type 'list' (line 61)
    list_383967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 61)
    # Adding element type (line 61)
    int_383968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 13), list_383967, int_383968)
    # Adding element type (line 61)
    int_383969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 13), list_383967, int_383969)
    # Adding element type (line 61)
    # Getting the type of 'np' (line 61)
    np_383970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'np')
    # Obtaining the member 'inf' of a type (line 61)
    inf_383971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 20), np_383970, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 13), list_383967, inf_383971)
    
    # Assigning a type to the variable 'limits' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'limits', list_383967)
    
    # Assigning a List to a Name (line 62):
    
    # Assigning a List to a Name (line 62):
    
    # Obtaining an instance of the builtin type 'list' (line 62)
    list_383972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 62)
    # Adding element type (line 62)
    # Getting the type of 'undirected_SP_limit_0' (line 62)
    undirected_SP_limit_0_383973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'undirected_SP_limit_0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 14), list_383972, undirected_SP_limit_0_383973)
    # Adding element type (line 62)
    # Getting the type of 'undirected_SP_limit_2' (line 63)
    undirected_SP_limit_2_383974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'undirected_SP_limit_2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 14), list_383972, undirected_SP_limit_2_383974)
    # Adding element type (line 62)
    # Getting the type of 'undirected_SP' (line 64)
    undirected_SP_383975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'undirected_SP')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 14), list_383972, undirected_SP_383975)
    
    # Assigning a type to the variable 'results' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'results', list_383972)

    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 66, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['limit', 'result']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['limit', 'result'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['limit', 'result'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to dijkstra(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'undirected_G' (line 67)
        undirected_G_383977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'undirected_G', False)
        # Processing the call keyword arguments (line 67)
        # Getting the type of 'False' (line 67)
        False_383978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 45), 'False', False)
        keyword_383979 = False_383978
        # Getting the type of 'limit' (line 67)
        limit_383980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 58), 'limit', False)
        keyword_383981 = limit_383980
        kwargs_383982 = {'directed': keyword_383979, 'limit': keyword_383981}
        # Getting the type of 'dijkstra' (line 67)
        dijkstra_383976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'dijkstra', False)
        # Calling dijkstra(args, kwargs) (line 67)
        dijkstra_call_result_383983 = invoke(stypy.reporting.localization.Localization(__file__, 67, 13), dijkstra_383976, *[undirected_G_383977], **kwargs_383982)
        
        # Assigning a type to the variable 'SP' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'SP', dijkstra_call_result_383983)
        
        # Call to assert_array_almost_equal(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'SP' (line 68)
        SP_383985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'SP', False)
        # Getting the type of 'result' (line 68)
        result_383986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 38), 'result', False)
        # Processing the call keyword arguments (line 68)
        kwargs_383987 = {}
        # Getting the type of 'assert_array_almost_equal' (line 68)
        assert_array_almost_equal_383984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 68)
        assert_array_almost_equal_call_result_383988 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), assert_array_almost_equal_383984, *[SP_383985, result_383986], **kwargs_383987)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_383989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_383989)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_383989

    # Assigning a type to the variable 'check' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'check', check)
    
    
    # Call to zip(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'limits' (line 70)
    limits_383991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 29), 'limits', False)
    # Getting the type of 'results' (line 70)
    results_383992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 37), 'results', False)
    # Processing the call keyword arguments (line 70)
    kwargs_383993 = {}
    # Getting the type of 'zip' (line 70)
    zip_383990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'zip', False)
    # Calling zip(args, kwargs) (line 70)
    zip_call_result_383994 = invoke(stypy.reporting.localization.Localization(__file__, 70, 25), zip_383990, *[limits_383991, results_383992], **kwargs_383993)
    
    # Testing the type of a for loop iterable (line 70)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 70, 4), zip_call_result_383994)
    # Getting the type of the for loop variable (line 70)
    for_loop_var_383995 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 70, 4), zip_call_result_383994)
    # Assigning a type to the variable 'limit' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'limit', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 4), for_loop_var_383995))
    # Assigning a type to the variable 'result' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'result', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 4), for_loop_var_383995))
    # SSA begins for a for statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'limit' (line 71)
    limit_383997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'limit', False)
    # Getting the type of 'result' (line 71)
    result_383998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'result', False)
    # Processing the call keyword arguments (line 71)
    kwargs_383999 = {}
    # Getting the type of 'check' (line 71)
    check_383996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'check', False)
    # Calling check(args, kwargs) (line 71)
    check_call_result_384000 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), check_383996, *[limit_383997, result_383998], **kwargs_383999)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_dijkstra_limit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_dijkstra_limit' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_384001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384001)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_dijkstra_limit'
    return stypy_return_type_384001

# Assigning a type to the variable 'test_dijkstra_limit' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'test_dijkstra_limit', test_dijkstra_limit)

@norecursion
def test_directed(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_directed'
    module_type_store = module_type_store.open_function_context('test_directed', 74, 0, False)
    
    # Passed parameters checking function
    test_directed.stypy_localization = localization
    test_directed.stypy_type_of_self = None
    test_directed.stypy_type_store = module_type_store
    test_directed.stypy_function_name = 'test_directed'
    test_directed.stypy_param_names_list = []
    test_directed.stypy_varargs_param_name = None
    test_directed.stypy_kwargs_param_name = None
    test_directed.stypy_call_defaults = defaults
    test_directed.stypy_call_varargs = varargs
    test_directed.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_directed', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_directed', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_directed(...)' code ##################


    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 75, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['method']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['method'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['method'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Call to a Name (line 76):
        
        # Assigning a Call to a Name (line 76):
        
        # Call to shortest_path(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'directed_G' (line 76)
        directed_G_384003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'directed_G', False)
        # Processing the call keyword arguments (line 76)
        # Getting the type of 'method' (line 76)
        method_384004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 46), 'method', False)
        keyword_384005 = method_384004
        # Getting the type of 'True' (line 76)
        True_384006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 63), 'True', False)
        keyword_384007 = True_384006
        # Getting the type of 'False' (line 77)
        False_384008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 37), 'False', False)
        keyword_384009 = False_384008
        kwargs_384010 = {'directed': keyword_384007, 'method': keyword_384005, 'overwrite': keyword_384009}
        # Getting the type of 'shortest_path' (line 76)
        shortest_path_384002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 13), 'shortest_path', False)
        # Calling shortest_path(args, kwargs) (line 76)
        shortest_path_call_result_384011 = invoke(stypy.reporting.localization.Localization(__file__, 76, 13), shortest_path_384002, *[directed_G_384003], **kwargs_384010)
        
        # Assigning a type to the variable 'SP' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'SP', shortest_path_call_result_384011)
        
        # Call to assert_array_almost_equal(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'SP' (line 78)
        SP_384013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 34), 'SP', False)
        # Getting the type of 'directed_SP' (line 78)
        directed_SP_384014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 38), 'directed_SP', False)
        # Processing the call keyword arguments (line 78)
        kwargs_384015 = {}
        # Getting the type of 'assert_array_almost_equal' (line 78)
        assert_array_almost_equal_384012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 78)
        assert_array_almost_equal_call_result_384016 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), assert_array_almost_equal_384012, *[SP_384013, directed_SP_384014], **kwargs_384015)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_384017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_384017)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_384017

    # Assigning a type to the variable 'check' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'check', check)
    
    # Getting the type of 'methods' (line 80)
    methods_384018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'methods')
    # Testing the type of a for loop iterable (line 80)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 80, 4), methods_384018)
    # Getting the type of the for loop variable (line 80)
    for_loop_var_384019 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 80, 4), methods_384018)
    # Assigning a type to the variable 'method' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'method', for_loop_var_384019)
    # SSA begins for a for statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'method' (line 81)
    method_384021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 14), 'method', False)
    # Processing the call keyword arguments (line 81)
    kwargs_384022 = {}
    # Getting the type of 'check' (line 81)
    check_384020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'check', False)
    # Calling check(args, kwargs) (line 81)
    check_call_result_384023 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), check_384020, *[method_384021], **kwargs_384022)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_directed(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_directed' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_384024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384024)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_directed'
    return stypy_return_type_384024

# Assigning a type to the variable 'test_directed' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'test_directed', test_directed)

@norecursion
def test_undirected(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_undirected'
    module_type_store = module_type_store.open_function_context('test_undirected', 84, 0, False)
    
    # Passed parameters checking function
    test_undirected.stypy_localization = localization
    test_undirected.stypy_type_of_self = None
    test_undirected.stypy_type_store = module_type_store
    test_undirected.stypy_function_name = 'test_undirected'
    test_undirected.stypy_param_names_list = []
    test_undirected.stypy_varargs_param_name = None
    test_undirected.stypy_kwargs_param_name = None
    test_undirected.stypy_call_defaults = defaults
    test_undirected.stypy_call_varargs = varargs
    test_undirected.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_undirected', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_undirected', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_undirected(...)' code ##################


    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 85, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['method', 'directed_in']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['method', 'directed_in'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['method', 'directed_in'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Getting the type of 'directed_in' (line 86)
        directed_in_384025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'directed_in')
        # Testing the type of an if condition (line 86)
        if_condition_384026 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 8), directed_in_384025)
        # Assigning a type to the variable 'if_condition_384026' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'if_condition_384026', if_condition_384026)
        # SSA begins for if statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 87):
        
        # Assigning a Call to a Name (line 87):
        
        # Call to shortest_path(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'directed_G' (line 87)
        directed_G_384028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 32), 'directed_G', False)
        # Processing the call keyword arguments (line 87)
        # Getting the type of 'method' (line 87)
        method_384029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 51), 'method', False)
        keyword_384030 = method_384029
        # Getting the type of 'False' (line 87)
        False_384031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 68), 'False', False)
        keyword_384032 = False_384031
        # Getting the type of 'False' (line 88)
        False_384033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 42), 'False', False)
        keyword_384034 = False_384033
        kwargs_384035 = {'directed': keyword_384032, 'method': keyword_384030, 'overwrite': keyword_384034}
        # Getting the type of 'shortest_path' (line 87)
        shortest_path_384027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), 'shortest_path', False)
        # Calling shortest_path(args, kwargs) (line 87)
        shortest_path_call_result_384036 = invoke(stypy.reporting.localization.Localization(__file__, 87, 18), shortest_path_384027, *[directed_G_384028], **kwargs_384035)
        
        # Assigning a type to the variable 'SP1' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'SP1', shortest_path_call_result_384036)
        
        # Call to assert_array_almost_equal(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'SP1' (line 89)
        SP1_384038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 38), 'SP1', False)
        # Getting the type of 'undirected_SP' (line 89)
        undirected_SP_384039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 43), 'undirected_SP', False)
        # Processing the call keyword arguments (line 89)
        kwargs_384040 = {}
        # Getting the type of 'assert_array_almost_equal' (line 89)
        assert_array_almost_equal_384037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 89)
        assert_array_almost_equal_call_result_384041 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), assert_array_almost_equal_384037, *[SP1_384038, undirected_SP_384039], **kwargs_384040)
        
        # SSA branch for the else part of an if statement (line 86)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to shortest_path(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'undirected_G' (line 91)
        undirected_G_384043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'undirected_G', False)
        # Processing the call keyword arguments (line 91)
        # Getting the type of 'method' (line 91)
        method_384044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 53), 'method', False)
        keyword_384045 = method_384044
        # Getting the type of 'True' (line 91)
        True_384046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 70), 'True', False)
        keyword_384047 = True_384046
        # Getting the type of 'False' (line 92)
        False_384048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'False', False)
        keyword_384049 = False_384048
        kwargs_384050 = {'directed': keyword_384047, 'method': keyword_384045, 'overwrite': keyword_384049}
        # Getting the type of 'shortest_path' (line 91)
        shortest_path_384042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'shortest_path', False)
        # Calling shortest_path(args, kwargs) (line 91)
        shortest_path_call_result_384051 = invoke(stypy.reporting.localization.Localization(__file__, 91, 18), shortest_path_384042, *[undirected_G_384043], **kwargs_384050)
        
        # Assigning a type to the variable 'SP2' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'SP2', shortest_path_call_result_384051)
        
        # Call to assert_array_almost_equal(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'SP2' (line 93)
        SP2_384053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'SP2', False)
        # Getting the type of 'undirected_SP' (line 93)
        undirected_SP_384054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 43), 'undirected_SP', False)
        # Processing the call keyword arguments (line 93)
        kwargs_384055 = {}
        # Getting the type of 'assert_array_almost_equal' (line 93)
        assert_array_almost_equal_384052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 93)
        assert_array_almost_equal_call_result_384056 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), assert_array_almost_equal_384052, *[SP2_384053, undirected_SP_384054], **kwargs_384055)
        
        # SSA join for if statement (line 86)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_384057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_384057)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_384057

    # Assigning a type to the variable 'check' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'check', check)
    
    # Getting the type of 'methods' (line 95)
    methods_384058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'methods')
    # Testing the type of a for loop iterable (line 95)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 95, 4), methods_384058)
    # Getting the type of the for loop variable (line 95)
    for_loop_var_384059 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 95, 4), methods_384058)
    # Assigning a type to the variable 'method' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'method', for_loop_var_384059)
    # SSA begins for a for statement (line 95)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 96)
    tuple_384060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 96)
    # Adding element type (line 96)
    # Getting the type of 'True' (line 96)
    True_384061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 28), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 28), tuple_384060, True_384061)
    # Adding element type (line 96)
    # Getting the type of 'False' (line 96)
    False_384062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 34), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 28), tuple_384060, False_384062)
    
    # Testing the type of a for loop iterable (line 96)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 96, 8), tuple_384060)
    # Getting the type of the for loop variable (line 96)
    for_loop_var_384063 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 96, 8), tuple_384060)
    # Assigning a type to the variable 'directed_in' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'directed_in', for_loop_var_384063)
    # SSA begins for a for statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'method' (line 97)
    method_384065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'method', False)
    # Getting the type of 'directed_in' (line 97)
    directed_in_384066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'directed_in', False)
    # Processing the call keyword arguments (line 97)
    kwargs_384067 = {}
    # Getting the type of 'check' (line 97)
    check_384064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'check', False)
    # Calling check(args, kwargs) (line 97)
    check_call_result_384068 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), check_384064, *[method_384065, directed_in_384066], **kwargs_384067)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_undirected(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_undirected' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_384069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384069)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_undirected'
    return stypy_return_type_384069

# Assigning a type to the variable 'test_undirected' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'test_undirected', test_undirected)

@norecursion
def test_shortest_path_indices(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_shortest_path_indices'
    module_type_store = module_type_store.open_function_context('test_shortest_path_indices', 100, 0, False)
    
    # Passed parameters checking function
    test_shortest_path_indices.stypy_localization = localization
    test_shortest_path_indices.stypy_type_of_self = None
    test_shortest_path_indices.stypy_type_store = module_type_store
    test_shortest_path_indices.stypy_function_name = 'test_shortest_path_indices'
    test_shortest_path_indices.stypy_param_names_list = []
    test_shortest_path_indices.stypy_varargs_param_name = None
    test_shortest_path_indices.stypy_kwargs_param_name = None
    test_shortest_path_indices.stypy_call_defaults = defaults
    test_shortest_path_indices.stypy_call_varargs = varargs
    test_shortest_path_indices.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_shortest_path_indices', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_shortest_path_indices', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_shortest_path_indices(...)' code ##################

    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to arange(...): (line 101)
    # Processing the call arguments (line 101)
    int_384072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 24), 'int')
    # Processing the call keyword arguments (line 101)
    kwargs_384073 = {}
    # Getting the type of 'np' (line 101)
    np_384070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 14), 'np', False)
    # Obtaining the member 'arange' of a type (line 101)
    arange_384071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 14), np_384070, 'arange')
    # Calling arange(args, kwargs) (line 101)
    arange_call_result_384074 = invoke(stypy.reporting.localization.Localization(__file__, 101, 14), arange_384071, *[int_384072], **kwargs_384073)
    
    # Assigning a type to the variable 'indices' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'indices', arange_call_result_384074)

    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 103, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['func', 'indshape']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['func', 'indshape'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['func', 'indshape'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a BinOp to a Name (line 104):
        
        # Assigning a BinOp to a Name (line 104):
        # Getting the type of 'indshape' (line 104)
        indshape_384075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'indshape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 104)
        tuple_384076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 104)
        # Adding element type (line 104)
        int_384077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 31), tuple_384076, int_384077)
        
        # Applying the binary operator '+' (line 104)
        result_add_384078 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 19), '+', indshape_384075, tuple_384076)
        
        # Assigning a type to the variable 'outshape' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'outshape', result_add_384078)
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to func(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'directed_G' (line 105)
        directed_G_384080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'directed_G', False)
        # Processing the call keyword arguments (line 105)
        # Getting the type of 'False' (line 105)
        False_384081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 39), 'False', False)
        keyword_384082 = False_384081
        
        # Call to reshape(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'indshape' (line 106)
        indshape_384085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 42), 'indshape', False)
        # Processing the call keyword arguments (line 106)
        kwargs_384086 = {}
        # Getting the type of 'indices' (line 106)
        indices_384083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'indices', False)
        # Obtaining the member 'reshape' of a type (line 106)
        reshape_384084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 26), indices_384083, 'reshape')
        # Calling reshape(args, kwargs) (line 106)
        reshape_call_result_384087 = invoke(stypy.reporting.localization.Localization(__file__, 106, 26), reshape_384084, *[indshape_384085], **kwargs_384086)
        
        keyword_384088 = reshape_call_result_384087
        kwargs_384089 = {'directed': keyword_384082, 'indices': keyword_384088}
        # Getting the type of 'func' (line 105)
        func_384079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'func', False)
        # Calling func(args, kwargs) (line 105)
        func_call_result_384090 = invoke(stypy.reporting.localization.Localization(__file__, 105, 13), func_384079, *[directed_G_384080], **kwargs_384089)
        
        # Assigning a type to the variable 'SP' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'SP', func_call_result_384090)
        
        # Call to assert_array_almost_equal(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'SP' (line 107)
        SP_384092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 34), 'SP', False)
        
        # Call to reshape(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'outshape' (line 107)
        outshape_384098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 69), 'outshape', False)
        # Processing the call keyword arguments (line 107)
        kwargs_384099 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'indices' (line 107)
        indices_384093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 52), 'indices', False)
        # Getting the type of 'undirected_SP' (line 107)
        undirected_SP_384094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 38), 'undirected_SP', False)
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___384095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 38), undirected_SP_384094, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_384096 = invoke(stypy.reporting.localization.Localization(__file__, 107, 38), getitem___384095, indices_384093)
        
        # Obtaining the member 'reshape' of a type (line 107)
        reshape_384097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 38), subscript_call_result_384096, 'reshape')
        # Calling reshape(args, kwargs) (line 107)
        reshape_call_result_384100 = invoke(stypy.reporting.localization.Localization(__file__, 107, 38), reshape_384097, *[outshape_384098], **kwargs_384099)
        
        # Processing the call keyword arguments (line 107)
        kwargs_384101 = {}
        # Getting the type of 'assert_array_almost_equal' (line 107)
        assert_array_almost_equal_384091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 107)
        assert_array_almost_equal_call_result_384102 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), assert_array_almost_equal_384091, *[SP_384092, reshape_call_result_384100], **kwargs_384101)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_384103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_384103)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_384103

    # Assigning a type to the variable 'check' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'check', check)
    
    
    # Obtaining an instance of the builtin type 'list' (line 109)
    list_384104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 109)
    # Adding element type (line 109)
    
    # Obtaining an instance of the builtin type 'tuple' (line 109)
    tuple_384105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 109)
    # Adding element type (line 109)
    int_384106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), tuple_384105, int_384106)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_384104, tuple_384105)
    # Adding element type (line 109)
    
    # Obtaining an instance of the builtin type 'tuple' (line 109)
    tuple_384107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 109)
    # Adding element type (line 109)
    int_384108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 28), tuple_384107, int_384108)
    # Adding element type (line 109)
    int_384109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 28), tuple_384107, int_384109)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_384104, tuple_384107)
    # Adding element type (line 109)
    
    # Obtaining an instance of the builtin type 'tuple' (line 109)
    tuple_384110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 109)
    # Adding element type (line 109)
    int_384111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 36), tuple_384110, int_384111)
    # Adding element type (line 109)
    int_384112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 36), tuple_384110, int_384112)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_384104, tuple_384110)
    
    # Testing the type of a for loop iterable (line 109)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 4), list_384104)
    # Getting the type of the for loop variable (line 109)
    for_loop_var_384113 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 4), list_384104)
    # Assigning a type to the variable 'indshape' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'indshape', for_loop_var_384113)
    # SSA begins for a for statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 110)
    tuple_384114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 110)
    # Adding element type (line 110)
    # Getting the type of 'dijkstra' (line 110)
    dijkstra_384115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'dijkstra')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), tuple_384114, dijkstra_384115)
    # Adding element type (line 110)
    # Getting the type of 'bellman_ford' (line 110)
    bellman_ford_384116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 31), 'bellman_ford')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), tuple_384114, bellman_ford_384116)
    # Adding element type (line 110)
    # Getting the type of 'johnson' (line 110)
    johnson_384117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 45), 'johnson')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), tuple_384114, johnson_384117)
    # Adding element type (line 110)
    # Getting the type of 'shortest_path' (line 110)
    shortest_path_384118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 54), 'shortest_path')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), tuple_384114, shortest_path_384118)
    
    # Testing the type of a for loop iterable (line 110)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 8), tuple_384114)
    # Getting the type of the for loop variable (line 110)
    for_loop_var_384119 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 8), tuple_384114)
    # Assigning a type to the variable 'func' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'func', for_loop_var_384119)
    # SSA begins for a for statement (line 110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'func' (line 111)
    func_384121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 18), 'func', False)
    # Getting the type of 'indshape' (line 111)
    indshape_384122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'indshape', False)
    # Processing the call keyword arguments (line 111)
    kwargs_384123 = {}
    # Getting the type of 'check' (line 111)
    check_384120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'check', False)
    # Calling check(args, kwargs) (line 111)
    check_call_result_384124 = invoke(stypy.reporting.localization.Localization(__file__, 111, 12), check_384120, *[func_384121, indshape_384122], **kwargs_384123)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_raises(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'ValueError' (line 113)
    ValueError_384126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'ValueError', False)
    # Getting the type of 'shortest_path' (line 113)
    shortest_path_384127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'shortest_path', False)
    # Getting the type of 'directed_G' (line 113)
    directed_G_384128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 45), 'directed_G', False)
    # Processing the call keyword arguments (line 113)
    str_384129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 64), 'str', 'FW')
    keyword_384130 = str_384129
    # Getting the type of 'indices' (line 114)
    indices_384131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 26), 'indices', False)
    keyword_384132 = indices_384131
    kwargs_384133 = {'indices': keyword_384132, 'method': keyword_384130}
    # Getting the type of 'assert_raises' (line 113)
    assert_raises_384125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 113)
    assert_raises_call_result_384134 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), assert_raises_384125, *[ValueError_384126, shortest_path_384127, directed_G_384128], **kwargs_384133)
    
    
    # ################# End of 'test_shortest_path_indices(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_shortest_path_indices' in the type store
    # Getting the type of 'stypy_return_type' (line 100)
    stypy_return_type_384135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384135)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_shortest_path_indices'
    return stypy_return_type_384135

# Assigning a type to the variable 'test_shortest_path_indices' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'test_shortest_path_indices', test_shortest_path_indices)

@norecursion
def test_predecessors(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_predecessors'
    module_type_store = module_type_store.open_function_context('test_predecessors', 117, 0, False)
    
    # Passed parameters checking function
    test_predecessors.stypy_localization = localization
    test_predecessors.stypy_type_of_self = None
    test_predecessors.stypy_type_store = module_type_store
    test_predecessors.stypy_function_name = 'test_predecessors'
    test_predecessors.stypy_param_names_list = []
    test_predecessors.stypy_varargs_param_name = None
    test_predecessors.stypy_kwargs_param_name = None
    test_predecessors.stypy_call_defaults = defaults
    test_predecessors.stypy_call_varargs = varargs
    test_predecessors.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_predecessors', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_predecessors', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_predecessors(...)' code ##################

    
    # Assigning a Dict to a Name (line 118):
    
    # Assigning a Dict to a Name (line 118):
    
    # Obtaining an instance of the builtin type 'dict' (line 118)
    dict_384136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 118)
    # Adding element type (key, value) (line 118)
    # Getting the type of 'True' (line 118)
    True_384137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'True')
    # Getting the type of 'directed_SP' (line 118)
    directed_SP_384138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'directed_SP')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 13), dict_384136, (True_384137, directed_SP_384138))
    # Adding element type (key, value) (line 118)
    # Getting the type of 'False' (line 119)
    False_384139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 14), 'False')
    # Getting the type of 'undirected_SP' (line 119)
    undirected_SP_384140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'undirected_SP')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 13), dict_384136, (False_384139, undirected_SP_384140))
    
    # Assigning a type to the variable 'SP_res' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'SP_res', dict_384136)
    
    # Assigning a Dict to a Name (line 120):
    
    # Assigning a Dict to a Name (line 120):
    
    # Obtaining an instance of the builtin type 'dict' (line 120)
    dict_384141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 120)
    # Adding element type (key, value) (line 120)
    # Getting the type of 'True' (line 120)
    True_384142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'True')
    # Getting the type of 'directed_pred' (line 120)
    directed_pred_384143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'directed_pred')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 15), dict_384141, (True_384142, directed_pred_384143))
    # Adding element type (key, value) (line 120)
    # Getting the type of 'False' (line 121)
    False_384144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'False')
    # Getting the type of 'undirected_pred' (line 121)
    undirected_pred_384145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'undirected_pred')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 15), dict_384141, (False_384144, undirected_pred_384145))
    
    # Assigning a type to the variable 'pred_res' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'pred_res', dict_384141)

    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 123, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['method', 'directed']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['method', 'directed'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['method', 'directed'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Call to a Tuple (line 124):
        
        # Assigning a Subscript to a Name (line 124):
        
        # Obtaining the type of the subscript
        int_384146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 8), 'int')
        
        # Call to shortest_path(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'directed_G' (line 124)
        directed_G_384148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 33), 'directed_G', False)
        # Getting the type of 'method' (line 124)
        method_384149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 45), 'method', False)
        # Processing the call keyword arguments (line 124)
        # Getting the type of 'directed' (line 124)
        directed_384150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 62), 'directed', False)
        keyword_384151 = directed_384150
        # Getting the type of 'False' (line 125)
        False_384152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 43), 'False', False)
        keyword_384153 = False_384152
        # Getting the type of 'True' (line 126)
        True_384154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 53), 'True', False)
        keyword_384155 = True_384154
        kwargs_384156 = {'directed': keyword_384151, 'return_predecessors': keyword_384155, 'overwrite': keyword_384153}
        # Getting the type of 'shortest_path' (line 124)
        shortest_path_384147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'shortest_path', False)
        # Calling shortest_path(args, kwargs) (line 124)
        shortest_path_call_result_384157 = invoke(stypy.reporting.localization.Localization(__file__, 124, 19), shortest_path_384147, *[directed_G_384148, method_384149], **kwargs_384156)
        
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___384158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), shortest_path_call_result_384157, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_384159 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), getitem___384158, int_384146)
        
        # Assigning a type to the variable 'tuple_var_assignment_383652' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'tuple_var_assignment_383652', subscript_call_result_384159)
        
        # Assigning a Subscript to a Name (line 124):
        
        # Obtaining the type of the subscript
        int_384160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 8), 'int')
        
        # Call to shortest_path(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'directed_G' (line 124)
        directed_G_384162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 33), 'directed_G', False)
        # Getting the type of 'method' (line 124)
        method_384163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 45), 'method', False)
        # Processing the call keyword arguments (line 124)
        # Getting the type of 'directed' (line 124)
        directed_384164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 62), 'directed', False)
        keyword_384165 = directed_384164
        # Getting the type of 'False' (line 125)
        False_384166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 43), 'False', False)
        keyword_384167 = False_384166
        # Getting the type of 'True' (line 126)
        True_384168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 53), 'True', False)
        keyword_384169 = True_384168
        kwargs_384170 = {'directed': keyword_384165, 'return_predecessors': keyword_384169, 'overwrite': keyword_384167}
        # Getting the type of 'shortest_path' (line 124)
        shortest_path_384161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'shortest_path', False)
        # Calling shortest_path(args, kwargs) (line 124)
        shortest_path_call_result_384171 = invoke(stypy.reporting.localization.Localization(__file__, 124, 19), shortest_path_384161, *[directed_G_384162, method_384163], **kwargs_384170)
        
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___384172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), shortest_path_call_result_384171, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_384173 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), getitem___384172, int_384160)
        
        # Assigning a type to the variable 'tuple_var_assignment_383653' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'tuple_var_assignment_383653', subscript_call_result_384173)
        
        # Assigning a Name to a Name (line 124):
        # Getting the type of 'tuple_var_assignment_383652' (line 124)
        tuple_var_assignment_383652_384174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'tuple_var_assignment_383652')
        # Assigning a type to the variable 'SP' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'SP', tuple_var_assignment_383652_384174)
        
        # Assigning a Name to a Name (line 124):
        # Getting the type of 'tuple_var_assignment_383653' (line 124)
        tuple_var_assignment_383653_384175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'tuple_var_assignment_383653')
        # Assigning a type to the variable 'pred' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'pred', tuple_var_assignment_383653_384175)
        
        # Call to assert_array_almost_equal(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'SP' (line 127)
        SP_384177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 34), 'SP', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'directed' (line 127)
        directed_384178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 45), 'directed', False)
        # Getting the type of 'SP_res' (line 127)
        SP_res_384179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 38), 'SP_res', False)
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___384180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 38), SP_res_384179, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_384181 = invoke(stypy.reporting.localization.Localization(__file__, 127, 38), getitem___384180, directed_384178)
        
        # Processing the call keyword arguments (line 127)
        kwargs_384182 = {}
        # Getting the type of 'assert_array_almost_equal' (line 127)
        assert_array_almost_equal_384176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 127)
        assert_array_almost_equal_call_result_384183 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), assert_array_almost_equal_384176, *[SP_384177, subscript_call_result_384181], **kwargs_384182)
        
        
        # Call to assert_array_almost_equal(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'pred' (line 128)
        pred_384185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 34), 'pred', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'directed' (line 128)
        directed_384186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 49), 'directed', False)
        # Getting the type of 'pred_res' (line 128)
        pred_res_384187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 40), 'pred_res', False)
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___384188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 40), pred_res_384187, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_384189 = invoke(stypy.reporting.localization.Localization(__file__, 128, 40), getitem___384188, directed_384186)
        
        # Processing the call keyword arguments (line 128)
        kwargs_384190 = {}
        # Getting the type of 'assert_array_almost_equal' (line 128)
        assert_array_almost_equal_384184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 128)
        assert_array_almost_equal_call_result_384191 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), assert_array_almost_equal_384184, *[pred_384185, subscript_call_result_384189], **kwargs_384190)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_384192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_384192)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_384192

    # Assigning a type to the variable 'check' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'check', check)
    
    # Getting the type of 'methods' (line 130)
    methods_384193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'methods')
    # Testing the type of a for loop iterable (line 130)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 130, 4), methods_384193)
    # Getting the type of the for loop variable (line 130)
    for_loop_var_384194 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 130, 4), methods_384193)
    # Assigning a type to the variable 'method' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'method', for_loop_var_384194)
    # SSA begins for a for statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 131)
    tuple_384195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 131)
    # Adding element type (line 131)
    # Getting the type of 'True' (line 131)
    True_384196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 25), tuple_384195, True_384196)
    # Adding element type (line 131)
    # Getting the type of 'False' (line 131)
    False_384197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 31), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 25), tuple_384195, False_384197)
    
    # Testing the type of a for loop iterable (line 131)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 131, 8), tuple_384195)
    # Getting the type of the for loop variable (line 131)
    for_loop_var_384198 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 131, 8), tuple_384195)
    # Assigning a type to the variable 'directed' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'directed', for_loop_var_384198)
    # SSA begins for a for statement (line 131)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'method' (line 132)
    method_384200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 18), 'method', False)
    # Getting the type of 'directed' (line 132)
    directed_384201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 26), 'directed', False)
    # Processing the call keyword arguments (line 132)
    kwargs_384202 = {}
    # Getting the type of 'check' (line 132)
    check_384199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'check', False)
    # Calling check(args, kwargs) (line 132)
    check_call_result_384203 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), check_384199, *[method_384200, directed_384201], **kwargs_384202)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_predecessors(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_predecessors' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_384204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384204)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_predecessors'
    return stypy_return_type_384204

# Assigning a type to the variable 'test_predecessors' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'test_predecessors', test_predecessors)

@norecursion
def test_construct_shortest_path(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_construct_shortest_path'
    module_type_store = module_type_store.open_function_context('test_construct_shortest_path', 135, 0, False)
    
    # Passed parameters checking function
    test_construct_shortest_path.stypy_localization = localization
    test_construct_shortest_path.stypy_type_of_self = None
    test_construct_shortest_path.stypy_type_store = module_type_store
    test_construct_shortest_path.stypy_function_name = 'test_construct_shortest_path'
    test_construct_shortest_path.stypy_param_names_list = []
    test_construct_shortest_path.stypy_varargs_param_name = None
    test_construct_shortest_path.stypy_kwargs_param_name = None
    test_construct_shortest_path.stypy_call_defaults = defaults
    test_construct_shortest_path.stypy_call_varargs = varargs
    test_construct_shortest_path.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_construct_shortest_path', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_construct_shortest_path', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_construct_shortest_path(...)' code ##################


    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 136, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['method', 'directed']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['method', 'directed'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['method', 'directed'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Call to a Tuple (line 137):
        
        # Assigning a Subscript to a Name (line 137):
        
        # Obtaining the type of the subscript
        int_384205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 8), 'int')
        
        # Call to shortest_path(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'directed_G' (line 137)
        directed_G_384207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 34), 'directed_G', False)
        # Processing the call keyword arguments (line 137)
        # Getting the type of 'directed' (line 138)
        directed_384208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 43), 'directed', False)
        keyword_384209 = directed_384208
        # Getting the type of 'False' (line 139)
        False_384210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 44), 'False', False)
        keyword_384211 = False_384210
        # Getting the type of 'True' (line 140)
        True_384212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 54), 'True', False)
        keyword_384213 = True_384212
        kwargs_384214 = {'directed': keyword_384209, 'return_predecessors': keyword_384213, 'overwrite': keyword_384211}
        # Getting the type of 'shortest_path' (line 137)
        shortest_path_384206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 20), 'shortest_path', False)
        # Calling shortest_path(args, kwargs) (line 137)
        shortest_path_call_result_384215 = invoke(stypy.reporting.localization.Localization(__file__, 137, 20), shortest_path_384206, *[directed_G_384207], **kwargs_384214)
        
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___384216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), shortest_path_call_result_384215, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_384217 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), getitem___384216, int_384205)
        
        # Assigning a type to the variable 'tuple_var_assignment_383654' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_383654', subscript_call_result_384217)
        
        # Assigning a Subscript to a Name (line 137):
        
        # Obtaining the type of the subscript
        int_384218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 8), 'int')
        
        # Call to shortest_path(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'directed_G' (line 137)
        directed_G_384220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 34), 'directed_G', False)
        # Processing the call keyword arguments (line 137)
        # Getting the type of 'directed' (line 138)
        directed_384221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 43), 'directed', False)
        keyword_384222 = directed_384221
        # Getting the type of 'False' (line 139)
        False_384223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 44), 'False', False)
        keyword_384224 = False_384223
        # Getting the type of 'True' (line 140)
        True_384225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 54), 'True', False)
        keyword_384226 = True_384225
        kwargs_384227 = {'directed': keyword_384222, 'return_predecessors': keyword_384226, 'overwrite': keyword_384224}
        # Getting the type of 'shortest_path' (line 137)
        shortest_path_384219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 20), 'shortest_path', False)
        # Calling shortest_path(args, kwargs) (line 137)
        shortest_path_call_result_384228 = invoke(stypy.reporting.localization.Localization(__file__, 137, 20), shortest_path_384219, *[directed_G_384220], **kwargs_384227)
        
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___384229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), shortest_path_call_result_384228, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_384230 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), getitem___384229, int_384218)
        
        # Assigning a type to the variable 'tuple_var_assignment_383655' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_383655', subscript_call_result_384230)
        
        # Assigning a Name to a Name (line 137):
        # Getting the type of 'tuple_var_assignment_383654' (line 137)
        tuple_var_assignment_383654_384231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_383654')
        # Assigning a type to the variable 'SP1' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'SP1', tuple_var_assignment_383654_384231)
        
        # Assigning a Name to a Name (line 137):
        # Getting the type of 'tuple_var_assignment_383655' (line 137)
        tuple_var_assignment_383655_384232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_383655')
        # Assigning a type to the variable 'pred' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 13), 'pred', tuple_var_assignment_383655_384232)
        
        # Assigning a Call to a Name (line 141):
        
        # Assigning a Call to a Name (line 141):
        
        # Call to construct_dist_matrix(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'directed_G' (line 141)
        directed_G_384234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 36), 'directed_G', False)
        # Getting the type of 'pred' (line 141)
        pred_384235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 48), 'pred', False)
        # Processing the call keyword arguments (line 141)
        # Getting the type of 'directed' (line 141)
        directed_384236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 63), 'directed', False)
        keyword_384237 = directed_384236
        kwargs_384238 = {'directed': keyword_384237}
        # Getting the type of 'construct_dist_matrix' (line 141)
        construct_dist_matrix_384233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 14), 'construct_dist_matrix', False)
        # Calling construct_dist_matrix(args, kwargs) (line 141)
        construct_dist_matrix_call_result_384239 = invoke(stypy.reporting.localization.Localization(__file__, 141, 14), construct_dist_matrix_384233, *[directed_G_384234, pred_384235], **kwargs_384238)
        
        # Assigning a type to the variable 'SP2' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'SP2', construct_dist_matrix_call_result_384239)
        
        # Call to assert_array_almost_equal(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'SP1' (line 142)
        SP1_384241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 34), 'SP1', False)
        # Getting the type of 'SP2' (line 142)
        SP2_384242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 39), 'SP2', False)
        # Processing the call keyword arguments (line 142)
        kwargs_384243 = {}
        # Getting the type of 'assert_array_almost_equal' (line 142)
        assert_array_almost_equal_384240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 142)
        assert_array_almost_equal_call_result_384244 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), assert_array_almost_equal_384240, *[SP1_384241, SP2_384242], **kwargs_384243)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 136)
        stypy_return_type_384245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_384245)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_384245

    # Assigning a type to the variable 'check' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'check', check)
    
    # Getting the type of 'methods' (line 144)
    methods_384246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 18), 'methods')
    # Testing the type of a for loop iterable (line 144)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 144, 4), methods_384246)
    # Getting the type of the for loop variable (line 144)
    for_loop_var_384247 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 144, 4), methods_384246)
    # Assigning a type to the variable 'method' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'method', for_loop_var_384247)
    # SSA begins for a for statement (line 144)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 145)
    tuple_384248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 145)
    # Adding element type (line 145)
    # Getting the type of 'True' (line 145)
    True_384249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), tuple_384248, True_384249)
    # Adding element type (line 145)
    # Getting the type of 'False' (line 145)
    False_384250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 31), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), tuple_384248, False_384250)
    
    # Testing the type of a for loop iterable (line 145)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 145, 8), tuple_384248)
    # Getting the type of the for loop variable (line 145)
    for_loop_var_384251 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 145, 8), tuple_384248)
    # Assigning a type to the variable 'directed' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'directed', for_loop_var_384251)
    # SSA begins for a for statement (line 145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'method' (line 146)
    method_384253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'method', False)
    # Getting the type of 'directed' (line 146)
    directed_384254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'directed', False)
    # Processing the call keyword arguments (line 146)
    kwargs_384255 = {}
    # Getting the type of 'check' (line 146)
    check_384252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'check', False)
    # Calling check(args, kwargs) (line 146)
    check_call_result_384256 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), check_384252, *[method_384253, directed_384254], **kwargs_384255)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_construct_shortest_path(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_construct_shortest_path' in the type store
    # Getting the type of 'stypy_return_type' (line 135)
    stypy_return_type_384257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384257)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_construct_shortest_path'
    return stypy_return_type_384257

# Assigning a type to the variable 'test_construct_shortest_path' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'test_construct_shortest_path', test_construct_shortest_path)

@norecursion
def test_unweighted_path(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_unweighted_path'
    module_type_store = module_type_store.open_function_context('test_unweighted_path', 149, 0, False)
    
    # Passed parameters checking function
    test_unweighted_path.stypy_localization = localization
    test_unweighted_path.stypy_type_of_self = None
    test_unweighted_path.stypy_type_store = module_type_store
    test_unweighted_path.stypy_function_name = 'test_unweighted_path'
    test_unweighted_path.stypy_param_names_list = []
    test_unweighted_path.stypy_varargs_param_name = None
    test_unweighted_path.stypy_kwargs_param_name = None
    test_unweighted_path.stypy_call_defaults = defaults
    test_unweighted_path.stypy_call_varargs = varargs
    test_unweighted_path.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_unweighted_path', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_unweighted_path', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_unweighted_path(...)' code ##################


    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 150, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['method', 'directed']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['method', 'directed'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['method', 'directed'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to shortest_path(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'directed_G' (line 151)
        directed_G_384259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 28), 'directed_G', False)
        # Processing the call keyword arguments (line 151)
        # Getting the type of 'directed' (line 152)
        directed_384260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 37), 'directed', False)
        keyword_384261 = directed_384260
        # Getting the type of 'False' (line 153)
        False_384262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 38), 'False', False)
        keyword_384263 = False_384262
        # Getting the type of 'True' (line 154)
        True_384264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 39), 'True', False)
        keyword_384265 = True_384264
        kwargs_384266 = {'directed': keyword_384261, 'unweighted': keyword_384265, 'overwrite': keyword_384263}
        # Getting the type of 'shortest_path' (line 151)
        shortest_path_384258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 14), 'shortest_path', False)
        # Calling shortest_path(args, kwargs) (line 151)
        shortest_path_call_result_384267 = invoke(stypy.reporting.localization.Localization(__file__, 151, 14), shortest_path_384258, *[directed_G_384259], **kwargs_384266)
        
        # Assigning a type to the variable 'SP1' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'SP1', shortest_path_call_result_384267)
        
        # Assigning a Call to a Name (line 155):
        
        # Assigning a Call to a Name (line 155):
        
        # Call to shortest_path(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'unweighted_G' (line 155)
        unweighted_G_384269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'unweighted_G', False)
        # Processing the call keyword arguments (line 155)
        # Getting the type of 'directed' (line 156)
        directed_384270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'directed', False)
        keyword_384271 = directed_384270
        # Getting the type of 'False' (line 157)
        False_384272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 38), 'False', False)
        keyword_384273 = False_384272
        # Getting the type of 'False' (line 158)
        False_384274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 39), 'False', False)
        keyword_384275 = False_384274
        kwargs_384276 = {'directed': keyword_384271, 'unweighted': keyword_384275, 'overwrite': keyword_384273}
        # Getting the type of 'shortest_path' (line 155)
        shortest_path_384268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 14), 'shortest_path', False)
        # Calling shortest_path(args, kwargs) (line 155)
        shortest_path_call_result_384277 = invoke(stypy.reporting.localization.Localization(__file__, 155, 14), shortest_path_384268, *[unweighted_G_384269], **kwargs_384276)
        
        # Assigning a type to the variable 'SP2' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'SP2', shortest_path_call_result_384277)
        
        # Call to assert_array_almost_equal(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'SP1' (line 159)
        SP1_384279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'SP1', False)
        # Getting the type of 'SP2' (line 159)
        SP2_384280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 39), 'SP2', False)
        # Processing the call keyword arguments (line 159)
        kwargs_384281 = {}
        # Getting the type of 'assert_array_almost_equal' (line 159)
        assert_array_almost_equal_384278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 159)
        assert_array_almost_equal_call_result_384282 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), assert_array_almost_equal_384278, *[SP1_384279, SP2_384280], **kwargs_384281)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_384283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_384283)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_384283

    # Assigning a type to the variable 'check' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'check', check)
    
    # Getting the type of 'methods' (line 161)
    methods_384284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 18), 'methods')
    # Testing the type of a for loop iterable (line 161)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 161, 4), methods_384284)
    # Getting the type of the for loop variable (line 161)
    for_loop_var_384285 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 161, 4), methods_384284)
    # Assigning a type to the variable 'method' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'method', for_loop_var_384285)
    # SSA begins for a for statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 162)
    tuple_384286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 162)
    # Adding element type (line 162)
    # Getting the type of 'True' (line 162)
    True_384287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 25), tuple_384286, True_384287)
    # Adding element type (line 162)
    # Getting the type of 'False' (line 162)
    False_384288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 31), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 25), tuple_384286, False_384288)
    
    # Testing the type of a for loop iterable (line 162)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 162, 8), tuple_384286)
    # Getting the type of the for loop variable (line 162)
    for_loop_var_384289 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 162, 8), tuple_384286)
    # Assigning a type to the variable 'directed' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'directed', for_loop_var_384289)
    # SSA begins for a for statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'method' (line 163)
    method_384291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 18), 'method', False)
    # Getting the type of 'directed' (line 163)
    directed_384292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 26), 'directed', False)
    # Processing the call keyword arguments (line 163)
    kwargs_384293 = {}
    # Getting the type of 'check' (line 163)
    check_384290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'check', False)
    # Calling check(args, kwargs) (line 163)
    check_call_result_384294 = invoke(stypy.reporting.localization.Localization(__file__, 163, 12), check_384290, *[method_384291, directed_384292], **kwargs_384293)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_unweighted_path(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_unweighted_path' in the type store
    # Getting the type of 'stypy_return_type' (line 149)
    stypy_return_type_384295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384295)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_unweighted_path'
    return stypy_return_type_384295

# Assigning a type to the variable 'test_unweighted_path' (line 149)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'test_unweighted_path', test_unweighted_path)

@norecursion
def test_negative_cycles(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_negative_cycles'
    module_type_store = module_type_store.open_function_context('test_negative_cycles', 166, 0, False)
    
    # Passed parameters checking function
    test_negative_cycles.stypy_localization = localization
    test_negative_cycles.stypy_type_of_self = None
    test_negative_cycles.stypy_type_store = module_type_store
    test_negative_cycles.stypy_function_name = 'test_negative_cycles'
    test_negative_cycles.stypy_param_names_list = []
    test_negative_cycles.stypy_varargs_param_name = None
    test_negative_cycles.stypy_kwargs_param_name = None
    test_negative_cycles.stypy_call_defaults = defaults
    test_negative_cycles.stypy_call_varargs = varargs
    test_negative_cycles.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_negative_cycles', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_negative_cycles', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_negative_cycles(...)' code ##################

    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to ones(...): (line 168)
    # Processing the call arguments (line 168)
    
    # Obtaining an instance of the builtin type 'list' (line 168)
    list_384298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 168)
    # Adding element type (line 168)
    int_384299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_384298, int_384299)
    # Adding element type (line 168)
    int_384300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_384298, int_384300)
    
    # Processing the call keyword arguments (line 168)
    kwargs_384301 = {}
    # Getting the type of 'np' (line 168)
    np_384296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'np', False)
    # Obtaining the member 'ones' of a type (line 168)
    ones_384297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 12), np_384296, 'ones')
    # Calling ones(args, kwargs) (line 168)
    ones_call_result_384302 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), ones_384297, *[list_384298], **kwargs_384301)
    
    # Assigning a type to the variable 'graph' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'graph', ones_call_result_384302)
    
    # Assigning a Num to a Subscript (line 169):
    
    # Assigning a Num to a Subscript (line 169):
    int_384303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 22), 'int')
    # Getting the type of 'graph' (line 169)
    graph_384304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'graph')
    # Obtaining the member 'flat' of a type (line 169)
    flat_384305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 4), graph_384304, 'flat')
    int_384306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 17), 'int')
    slice_384307 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 169, 4), None, None, int_384306)
    # Storing an element on a container (line 169)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 4), flat_384305, (slice_384307, int_384303))
    
    # Assigning a Num to a Subscript (line 170):
    
    # Assigning a Num to a Subscript (line 170):
    int_384308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 18), 'int')
    # Getting the type of 'graph' (line 170)
    graph_384309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'graph')
    
    # Obtaining an instance of the builtin type 'tuple' (line 170)
    tuple_384310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 170)
    # Adding element type (line 170)
    int_384311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 10), tuple_384310, int_384311)
    # Adding element type (line 170)
    int_384312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 10), tuple_384310, int_384312)
    
    # Storing an element on a container (line 170)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 4), graph_384309, (tuple_384310, int_384308))

    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 172, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['method', 'directed']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['method', 'directed'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['method', 'directed'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Call to assert_raises(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'NegativeCycleError' (line 173)
        NegativeCycleError_384314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'NegativeCycleError', False)
        # Getting the type of 'shortest_path' (line 173)
        shortest_path_384315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 42), 'shortest_path', False)
        # Getting the type of 'graph' (line 173)
        graph_384316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 57), 'graph', False)
        # Getting the type of 'method' (line 173)
        method_384317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 64), 'method', False)
        # Getting the type of 'directed' (line 174)
        directed_384318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 22), 'directed', False)
        # Processing the call keyword arguments (line 173)
        kwargs_384319 = {}
        # Getting the type of 'assert_raises' (line 173)
        assert_raises_384313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 173)
        assert_raises_call_result_384320 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), assert_raises_384313, *[NegativeCycleError_384314, shortest_path_384315, graph_384316, method_384317, directed_384318], **kwargs_384319)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_384321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_384321)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_384321

    # Assigning a type to the variable 'check' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'check', check)
    
    
    # Obtaining an instance of the builtin type 'list' (line 176)
    list_384322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 176)
    # Adding element type (line 176)
    str_384323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 19), 'str', 'FW')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 18), list_384322, str_384323)
    # Adding element type (line 176)
    str_384324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 25), 'str', 'J')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 18), list_384322, str_384324)
    # Adding element type (line 176)
    str_384325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 30), 'str', 'BF')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 18), list_384322, str_384325)
    
    # Testing the type of a for loop iterable (line 176)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 176, 4), list_384322)
    # Getting the type of the for loop variable (line 176)
    for_loop_var_384326 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 176, 4), list_384322)
    # Assigning a type to the variable 'method' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'method', for_loop_var_384326)
    # SSA begins for a for statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 177)
    tuple_384327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 177)
    # Adding element type (line 177)
    # Getting the type of 'True' (line 177)
    True_384328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 25), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 25), tuple_384327, True_384328)
    # Adding element type (line 177)
    # Getting the type of 'False' (line 177)
    False_384329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 31), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 25), tuple_384327, False_384329)
    
    # Testing the type of a for loop iterable (line 177)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 177, 8), tuple_384327)
    # Getting the type of the for loop variable (line 177)
    for_loop_var_384330 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 177, 8), tuple_384327)
    # Assigning a type to the variable 'directed' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'directed', for_loop_var_384330)
    # SSA begins for a for statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'method' (line 178)
    method_384332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 18), 'method', False)
    # Getting the type of 'directed' (line 178)
    directed_384333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 26), 'directed', False)
    # Processing the call keyword arguments (line 178)
    kwargs_384334 = {}
    # Getting the type of 'check' (line 178)
    check_384331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'check', False)
    # Calling check(args, kwargs) (line 178)
    check_call_result_384335 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), check_384331, *[method_384332, directed_384333], **kwargs_384334)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_negative_cycles(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_negative_cycles' in the type store
    # Getting the type of 'stypy_return_type' (line 166)
    stypy_return_type_384336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384336)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_negative_cycles'
    return stypy_return_type_384336

# Assigning a type to the variable 'test_negative_cycles' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'test_negative_cycles', test_negative_cycles)

@norecursion
def test_masked_input(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_masked_input'
    module_type_store = module_type_store.open_function_context('test_masked_input', 181, 0, False)
    
    # Passed parameters checking function
    test_masked_input.stypy_localization = localization
    test_masked_input.stypy_type_of_self = None
    test_masked_input.stypy_type_store = module_type_store
    test_masked_input.stypy_function_name = 'test_masked_input'
    test_masked_input.stypy_param_names_list = []
    test_masked_input.stypy_varargs_param_name = None
    test_masked_input.stypy_kwargs_param_name = None
    test_masked_input.stypy_call_defaults = defaults
    test_masked_input.stypy_call_varargs = varargs
    test_masked_input.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_masked_input', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_masked_input', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_masked_input(...)' code ##################

    
    # Assigning a Call to a Name (line 182):
    
    # Assigning a Call to a Name (line 182):
    
    # Call to masked_equal(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'directed_G' (line 182)
    directed_G_384340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 27), 'directed_G', False)
    int_384341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 39), 'int')
    # Processing the call keyword arguments (line 182)
    kwargs_384342 = {}
    # Getting the type of 'np' (line 182)
    np_384337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'np', False)
    # Obtaining the member 'ma' of a type (line 182)
    ma_384338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), np_384337, 'ma')
    # Obtaining the member 'masked_equal' of a type (line 182)
    masked_equal_384339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), ma_384338, 'masked_equal')
    # Calling masked_equal(args, kwargs) (line 182)
    masked_equal_call_result_384343 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), masked_equal_384339, *[directed_G_384340, int_384341], **kwargs_384342)
    
    # Assigning a type to the variable 'G' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'G', masked_equal_call_result_384343)

    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 184, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['method']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['method'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['method'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Call to a Name (line 185):
        
        # Assigning a Call to a Name (line 185):
        
        # Call to shortest_path(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'directed_G' (line 185)
        directed_G_384345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 27), 'directed_G', False)
        # Processing the call keyword arguments (line 185)
        # Getting the type of 'method' (line 185)
        method_384346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 46), 'method', False)
        keyword_384347 = method_384346
        # Getting the type of 'True' (line 185)
        True_384348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 63), 'True', False)
        keyword_384349 = True_384348
        # Getting the type of 'False' (line 186)
        False_384350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 37), 'False', False)
        keyword_384351 = False_384350
        kwargs_384352 = {'directed': keyword_384349, 'method': keyword_384347, 'overwrite': keyword_384351}
        # Getting the type of 'shortest_path' (line 185)
        shortest_path_384344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), 'shortest_path', False)
        # Calling shortest_path(args, kwargs) (line 185)
        shortest_path_call_result_384353 = invoke(stypy.reporting.localization.Localization(__file__, 185, 13), shortest_path_384344, *[directed_G_384345], **kwargs_384352)
        
        # Assigning a type to the variable 'SP' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'SP', shortest_path_call_result_384353)
        
        # Call to assert_array_almost_equal(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'SP' (line 187)
        SP_384355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 34), 'SP', False)
        # Getting the type of 'directed_SP' (line 187)
        directed_SP_384356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 38), 'directed_SP', False)
        # Processing the call keyword arguments (line 187)
        kwargs_384357 = {}
        # Getting the type of 'assert_array_almost_equal' (line 187)
        assert_array_almost_equal_384354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 187)
        assert_array_almost_equal_call_result_384358 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), assert_array_almost_equal_384354, *[SP_384355, directed_SP_384356], **kwargs_384357)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_384359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_384359)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_384359

    # Assigning a type to the variable 'check' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'check', check)
    
    # Getting the type of 'methods' (line 189)
    methods_384360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 18), 'methods')
    # Testing the type of a for loop iterable (line 189)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 189, 4), methods_384360)
    # Getting the type of the for loop variable (line 189)
    for_loop_var_384361 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 189, 4), methods_384360)
    # Assigning a type to the variable 'method' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'method', for_loop_var_384361)
    # SSA begins for a for statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'method' (line 190)
    method_384363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 14), 'method', False)
    # Processing the call keyword arguments (line 190)
    kwargs_384364 = {}
    # Getting the type of 'check' (line 190)
    check_384362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'check', False)
    # Calling check(args, kwargs) (line 190)
    check_call_result_384365 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), check_384362, *[method_384363], **kwargs_384364)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_masked_input(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_masked_input' in the type store
    # Getting the type of 'stypy_return_type' (line 181)
    stypy_return_type_384366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384366)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_masked_input'
    return stypy_return_type_384366

# Assigning a type to the variable 'test_masked_input' (line 181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'test_masked_input', test_masked_input)

@norecursion
def test_overwrite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_overwrite'
    module_type_store = module_type_store.open_function_context('test_overwrite', 193, 0, False)
    
    # Passed parameters checking function
    test_overwrite.stypy_localization = localization
    test_overwrite.stypy_type_of_self = None
    test_overwrite.stypy_type_store = module_type_store
    test_overwrite.stypy_function_name = 'test_overwrite'
    test_overwrite.stypy_param_names_list = []
    test_overwrite.stypy_varargs_param_name = None
    test_overwrite.stypy_kwargs_param_name = None
    test_overwrite.stypy_call_defaults = defaults
    test_overwrite.stypy_call_varargs = varargs
    test_overwrite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_overwrite', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_overwrite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_overwrite(...)' code ##################

    
    # Assigning a Call to a Name (line 194):
    
    # Assigning a Call to a Name (line 194):
    
    # Call to array(...): (line 194)
    # Processing the call arguments (line 194)
    
    # Obtaining an instance of the builtin type 'list' (line 194)
    list_384369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 194)
    # Adding element type (line 194)
    
    # Obtaining an instance of the builtin type 'list' (line 194)
    list_384370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 194)
    # Adding element type (line 194)
    int_384371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 18), list_384370, int_384371)
    # Adding element type (line 194)
    int_384372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 18), list_384370, int_384372)
    # Adding element type (line 194)
    int_384373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 18), list_384370, int_384373)
    # Adding element type (line 194)
    int_384374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 18), list_384370, int_384374)
    # Adding element type (line 194)
    int_384375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 18), list_384370, int_384375)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 17), list_384369, list_384370)
    # Adding element type (line 194)
    
    # Obtaining an instance of the builtin type 'list' (line 195)
    list_384376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 195)
    # Adding element type (line 195)
    int_384377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), list_384376, int_384377)
    # Adding element type (line 195)
    int_384378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), list_384376, int_384378)
    # Adding element type (line 195)
    int_384379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), list_384376, int_384379)
    # Adding element type (line 195)
    int_384380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), list_384376, int_384380)
    # Adding element type (line 195)
    int_384381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), list_384376, int_384381)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 17), list_384369, list_384376)
    # Adding element type (line 194)
    
    # Obtaining an instance of the builtin type 'list' (line 196)
    list_384382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 196)
    # Adding element type (line 196)
    int_384383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 18), list_384382, int_384383)
    # Adding element type (line 196)
    int_384384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 18), list_384382, int_384384)
    # Adding element type (line 196)
    int_384385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 18), list_384382, int_384385)
    # Adding element type (line 196)
    int_384386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 18), list_384382, int_384386)
    # Adding element type (line 196)
    int_384387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 18), list_384382, int_384387)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 17), list_384369, list_384382)
    # Adding element type (line 194)
    
    # Obtaining an instance of the builtin type 'list' (line 197)
    list_384388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 197)
    # Adding element type (line 197)
    int_384389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 18), list_384388, int_384389)
    # Adding element type (line 197)
    int_384390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 18), list_384388, int_384390)
    # Adding element type (line 197)
    int_384391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 18), list_384388, int_384391)
    # Adding element type (line 197)
    int_384392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 18), list_384388, int_384392)
    # Adding element type (line 197)
    int_384393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 18), list_384388, int_384393)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 17), list_384369, list_384388)
    # Adding element type (line 194)
    
    # Obtaining an instance of the builtin type 'list' (line 198)
    list_384394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 198)
    # Adding element type (line 198)
    int_384395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), list_384394, int_384395)
    # Adding element type (line 198)
    int_384396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), list_384394, int_384396)
    # Adding element type (line 198)
    int_384397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), list_384394, int_384397)
    # Adding element type (line 198)
    int_384398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), list_384394, int_384398)
    # Adding element type (line 198)
    int_384399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), list_384394, int_384399)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 17), list_384369, list_384394)
    
    # Processing the call keyword arguments (line 194)
    # Getting the type of 'float' (line 198)
    float_384400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 42), 'float', False)
    keyword_384401 = float_384400
    kwargs_384402 = {'dtype': keyword_384401}
    # Getting the type of 'np' (line 194)
    np_384367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 194)
    array_384368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), np_384367, 'array')
    # Calling array(args, kwargs) (line 194)
    array_call_result_384403 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), array_384368, *[list_384369], **kwargs_384402)
    
    # Assigning a type to the variable 'G' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'G', array_call_result_384403)
    
    # Assigning a Call to a Name (line 199):
    
    # Assigning a Call to a Name (line 199):
    
    # Call to copy(...): (line 199)
    # Processing the call keyword arguments (line 199)
    kwargs_384406 = {}
    # Getting the type of 'G' (line 199)
    G_384404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 10), 'G', False)
    # Obtaining the member 'copy' of a type (line 199)
    copy_384405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 10), G_384404, 'copy')
    # Calling copy(args, kwargs) (line 199)
    copy_call_result_384407 = invoke(stypy.reporting.localization.Localization(__file__, 199, 10), copy_384405, *[], **kwargs_384406)
    
    # Assigning a type to the variable 'foo' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'foo', copy_call_result_384407)
    
    # Call to shortest_path(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'foo' (line 200)
    foo_384409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), 'foo', False)
    # Processing the call keyword arguments (line 200)
    # Getting the type of 'False' (line 200)
    False_384410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 33), 'False', False)
    keyword_384411 = False_384410
    kwargs_384412 = {'overwrite': keyword_384411}
    # Getting the type of 'shortest_path' (line 200)
    shortest_path_384408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'shortest_path', False)
    # Calling shortest_path(args, kwargs) (line 200)
    shortest_path_call_result_384413 = invoke(stypy.reporting.localization.Localization(__file__, 200, 4), shortest_path_384408, *[foo_384409], **kwargs_384412)
    
    
    # Call to assert_array_equal(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'foo' (line 201)
    foo_384415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), 'foo', False)
    # Getting the type of 'G' (line 201)
    G_384416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 28), 'G', False)
    # Processing the call keyword arguments (line 201)
    kwargs_384417 = {}
    # Getting the type of 'assert_array_equal' (line 201)
    assert_array_equal_384414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 201)
    assert_array_equal_call_result_384418 = invoke(stypy.reporting.localization.Localization(__file__, 201, 4), assert_array_equal_384414, *[foo_384415, G_384416], **kwargs_384417)
    
    
    # ################# End of 'test_overwrite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_overwrite' in the type store
    # Getting the type of 'stypy_return_type' (line 193)
    stypy_return_type_384419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384419)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_overwrite'
    return stypy_return_type_384419

# Assigning a type to the variable 'test_overwrite' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'test_overwrite', test_overwrite)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
