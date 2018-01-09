
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Hungarian algorithm (Kuhn-Munkres) for solving the linear sum assignment
2: # problem. Taken from scikit-learn. Based on original code by Brian Clapper,
3: # adapted to NumPy by Gael Varoquaux.
4: # Further improvements by Ben Root, Vlad Niculae and Lars Buitinck.
5: #
6: # Copyright (c) 2008 Brian M. Clapper <bmc@clapper.org>, Gael Varoquaux
7: # Author: Brian M. Clapper, Gael Varoquaux
8: # License: 3-clause BSD
9: 
10: import numpy as np
11: 
12: 
13: def linear_sum_assignment(cost_matrix):
14:     '''Solve the linear sum assignment problem.
15: 
16:     The linear sum assignment problem is also known as minimum weight matching
17:     in bipartite graphs. A problem instance is described by a matrix C, where
18:     each C[i,j] is the cost of matching vertex i of the first partite set
19:     (a "worker") and vertex j of the second set (a "job"). The goal is to find
20:     a complete assignment of workers to jobs of minimal cost.
21: 
22:     Formally, let X be a boolean matrix where :math:`X[i,j] = 1` iff row i is
23:     assigned to column j. Then the optimal assignment has cost
24: 
25:     .. math::
26:         \\min \\sum_i \\sum_j C_{i,j} X_{i,j}
27: 
28:     s.t. each row is assignment to at most one column, and each column to at
29:     most one row.
30: 
31:     This function can also solve a generalization of the classic assignment
32:     problem where the cost matrix is rectangular. If it has more rows than
33:     columns, then not every row needs to be assigned to a column, and vice
34:     versa.
35: 
36:     The method used is the Hungarian algorithm, also known as the Munkres or
37:     Kuhn-Munkres algorithm.
38: 
39:     Parameters
40:     ----------
41:     cost_matrix : array
42:         The cost matrix of the bipartite graph.
43: 
44:     Returns
45:     -------
46:     row_ind, col_ind : array
47:         An array of row indices and one of corresponding column indices giving
48:         the optimal assignment. The cost of the assignment can be computed
49:         as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
50:         sorted; in the case of a square cost matrix they will be equal to
51:         ``numpy.arange(cost_matrix.shape[0])``.
52: 
53:     Notes
54:     -----
55:     .. versionadded:: 0.17.0
56: 
57:     Examples
58:     --------
59:     >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
60:     >>> from scipy.optimize import linear_sum_assignment
61:     >>> row_ind, col_ind = linear_sum_assignment(cost)
62:     >>> col_ind
63:     array([1, 0, 2])
64:     >>> cost[row_ind, col_ind].sum()
65:     5
66: 
67:     References
68:     ----------
69:     1. http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html
70: 
71:     2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
72:        *Naval Research Logistics Quarterly*, 2:83-97, 1955.
73: 
74:     3. Harold W. Kuhn. Variants of the Hungarian method for assignment
75:        problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.
76: 
77:     4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
78:        *J. SIAM*, 5(1):32-38, March, 1957.
79: 
80:     5. https://en.wikipedia.org/wiki/Hungarian_algorithm
81:     '''
82:     cost_matrix = np.asarray(cost_matrix)
83:     if len(cost_matrix.shape) != 2:
84:         raise ValueError("expected a matrix (2-d array), got a %r array"
85:                          % (cost_matrix.shape,))
86: 
87:     if not (np.issubdtype(cost_matrix.dtype, np.number) or
88:             cost_matrix.dtype == np.dtype(np.bool)):
89:         raise ValueError("expected a matrix containing numerical entries, got %s"
90:                          % (cost_matrix.dtype,))
91: 
92:     if np.any(np.isinf(cost_matrix) | np.isnan(cost_matrix)):
93:         raise ValueError("matrix contains invalid numeric entries")
94: 
95:     if cost_matrix.dtype == np.dtype(np.bool):
96:         cost_matrix = cost_matrix.astype(np.int)
97: 
98:     # The algorithm expects more columns than rows in the cost matrix.
99:     if cost_matrix.shape[1] < cost_matrix.shape[0]:
100:         cost_matrix = cost_matrix.T
101:         transposed = True
102:     else:
103:         transposed = False
104: 
105:     state = _Hungary(cost_matrix)
106: 
107:     # No need to bother with assignments if one of the dimensions
108:     # of the cost matrix is zero-length.
109:     step = None if 0 in cost_matrix.shape else _step1
110: 
111:     while step is not None:
112:         step = step(state)
113: 
114:     if transposed:
115:         marked = state.marked.T
116:     else:
117:         marked = state.marked
118:     return np.where(marked == 1)
119: 
120: 
121: class _Hungary(object):
122:     '''State of the Hungarian algorithm.
123: 
124:     Parameters
125:     ----------
126:     cost_matrix : 2D matrix
127:         The cost matrix. Must have shape[1] >= shape[0].
128:     '''
129: 
130:     def __init__(self, cost_matrix):
131:         self.C = cost_matrix.copy()
132: 
133:         n, m = self.C.shape
134:         self.row_uncovered = np.ones(n, dtype=bool)
135:         self.col_uncovered = np.ones(m, dtype=bool)
136:         self.Z0_r = 0
137:         self.Z0_c = 0
138:         self.path = np.zeros((n + m, 2), dtype=int)
139:         self.marked = np.zeros((n, m), dtype=int)
140: 
141:     def _clear_covers(self):
142:         '''Clear all covered matrix cells'''
143:         self.row_uncovered[:] = True
144:         self.col_uncovered[:] = True
145: 
146: 
147: # Individual steps of the algorithm follow, as a state machine: they return
148: # the next step to be taken (function to be called), if any.
149: 
150: def _step1(state):
151:     '''Steps 1 and 2 in the Wikipedia page.'''
152: 
153:     # Step 1: For each row of the matrix, find the smallest element and
154:     # subtract it from every element in its row.
155:     state.C -= state.C.min(axis=1)[:, np.newaxis]
156:     # Step 2: Find a zero (Z) in the resulting matrix. If there is no
157:     # starred zero in its row or column, star Z. Repeat for each element
158:     # in the matrix.
159:     for i, j in zip(*np.where(state.C == 0)):
160:         if state.col_uncovered[j] and state.row_uncovered[i]:
161:             state.marked[i, j] = 1
162:             state.col_uncovered[j] = False
163:             state.row_uncovered[i] = False
164: 
165:     state._clear_covers()
166:     return _step3
167: 
168: 
169: def _step3(state):
170:     '''
171:     Cover each column containing a starred zero. If n columns are covered,
172:     the starred zeros describe a complete set of unique assignments.
173:     In this case, Go to DONE, otherwise, Go to Step 4.
174:     '''
175:     marked = (state.marked == 1)
176:     state.col_uncovered[np.any(marked, axis=0)] = False
177: 
178:     if marked.sum() < state.C.shape[0]:
179:         return _step4
180: 
181: 
182: def _step4(state):
183:     '''
184:     Find a noncovered zero and prime it. If there is no starred zero
185:     in the row containing this primed zero, Go to Step 5. Otherwise,
186:     cover this row and uncover the column containing the starred
187:     zero. Continue in this manner until there are no uncovered zeros
188:     left. Save the smallest uncovered value and Go to Step 6.
189:     '''
190:     # We convert to int as numpy operations are faster on int
191:     C = (state.C == 0).astype(int)
192:     covered_C = C * state.row_uncovered[:, np.newaxis]
193:     covered_C *= np.asarray(state.col_uncovered, dtype=int)
194:     n = state.C.shape[0]
195:     m = state.C.shape[1]
196: 
197:     while True:
198:         # Find an uncovered zero
199:         row, col = np.unravel_index(np.argmax(covered_C), (n, m))
200:         if covered_C[row, col] == 0:
201:             return _step6
202:         else:
203:             state.marked[row, col] = 2
204:             # Find the first starred element in the row
205:             star_col = np.argmax(state.marked[row] == 1)
206:             if state.marked[row, star_col] != 1:
207:                 # Could not find one
208:                 state.Z0_r = row
209:                 state.Z0_c = col
210:                 return _step5
211:             else:
212:                 col = star_col
213:                 state.row_uncovered[row] = False
214:                 state.col_uncovered[col] = True
215:                 covered_C[:, col] = C[:, col] * (
216:                     np.asarray(state.row_uncovered, dtype=int))
217:                 covered_C[row] = 0
218: 
219: 
220: def _step5(state):
221:     '''
222:     Construct a series of alternating primed and starred zeros as follows.
223:     Let Z0 represent the uncovered primed zero found in Step 4.
224:     Let Z1 denote the starred zero in the column of Z0 (if any).
225:     Let Z2 denote the primed zero in the row of Z1 (there will always be one).
226:     Continue until the series terminates at a primed zero that has no starred
227:     zero in its column. Unstar each starred zero of the series, star each
228:     primed zero of the series, erase all primes and uncover every line in the
229:     matrix. Return to Step 3
230:     '''
231:     count = 0
232:     path = state.path
233:     path[count, 0] = state.Z0_r
234:     path[count, 1] = state.Z0_c
235: 
236:     while True:
237:         # Find the first starred element in the col defined by
238:         # the path.
239:         row = np.argmax(state.marked[:, path[count, 1]] == 1)
240:         if state.marked[row, path[count, 1]] != 1:
241:             # Could not find one
242:             break
243:         else:
244:             count += 1
245:             path[count, 0] = row
246:             path[count, 1] = path[count - 1, 1]
247: 
248:         # Find the first prime element in the row defined by the
249:         # first path step
250:         col = np.argmax(state.marked[path[count, 0]] == 2)
251:         if state.marked[row, col] != 2:
252:             col = -1
253:         count += 1
254:         path[count, 0] = path[count - 1, 0]
255:         path[count, 1] = col
256: 
257:     # Convert paths
258:     for i in range(count + 1):
259:         if state.marked[path[i, 0], path[i, 1]] == 1:
260:             state.marked[path[i, 0], path[i, 1]] = 0
261:         else:
262:             state.marked[path[i, 0], path[i, 1]] = 1
263: 
264:     state._clear_covers()
265:     # Erase all prime markings
266:     state.marked[state.marked == 2] = 0
267:     return _step3
268: 
269: 
270: def _step6(state):
271:     '''
272:     Add the value found in Step 4 to every element of each covered row,
273:     and subtract it from every element of each uncovered column.
274:     Return to Step 4 without altering any stars, primes, or covered lines.
275:     '''
276:     # the smallest uncovered value in the matrix
277:     if np.any(state.row_uncovered) and np.any(state.col_uncovered):
278:         minval = np.min(state.C[state.row_uncovered], axis=0)
279:         minval = np.min(minval[state.col_uncovered])
280:         state.C[~state.row_uncovered] += minval
281:         state.C[:, state.col_uncovered] -= minval
282:     return _step4
283: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_189771 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_189771) is not StypyTypeError):

    if (import_189771 != 'pyd_module'):
        __import__(import_189771)
        sys_modules_189772 = sys.modules[import_189771]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_189772.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_189771)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


@norecursion
def linear_sum_assignment(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'linear_sum_assignment'
    module_type_store = module_type_store.open_function_context('linear_sum_assignment', 13, 0, False)
    
    # Passed parameters checking function
    linear_sum_assignment.stypy_localization = localization
    linear_sum_assignment.stypy_type_of_self = None
    linear_sum_assignment.stypy_type_store = module_type_store
    linear_sum_assignment.stypy_function_name = 'linear_sum_assignment'
    linear_sum_assignment.stypy_param_names_list = ['cost_matrix']
    linear_sum_assignment.stypy_varargs_param_name = None
    linear_sum_assignment.stypy_kwargs_param_name = None
    linear_sum_assignment.stypy_call_defaults = defaults
    linear_sum_assignment.stypy_call_varargs = varargs
    linear_sum_assignment.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'linear_sum_assignment', ['cost_matrix'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'linear_sum_assignment', localization, ['cost_matrix'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'linear_sum_assignment(...)' code ##################

    str_189773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'str', 'Solve the linear sum assignment problem.\n\n    The linear sum assignment problem is also known as minimum weight matching\n    in bipartite graphs. A problem instance is described by a matrix C, where\n    each C[i,j] is the cost of matching vertex i of the first partite set\n    (a "worker") and vertex j of the second set (a "job"). The goal is to find\n    a complete assignment of workers to jobs of minimal cost.\n\n    Formally, let X be a boolean matrix where :math:`X[i,j] = 1` iff row i is\n    assigned to column j. Then the optimal assignment has cost\n\n    .. math::\n        \\min \\sum_i \\sum_j C_{i,j} X_{i,j}\n\n    s.t. each row is assignment to at most one column, and each column to at\n    most one row.\n\n    This function can also solve a generalization of the classic assignment\n    problem where the cost matrix is rectangular. If it has more rows than\n    columns, then not every row needs to be assigned to a column, and vice\n    versa.\n\n    The method used is the Hungarian algorithm, also known as the Munkres or\n    Kuhn-Munkres algorithm.\n\n    Parameters\n    ----------\n    cost_matrix : array\n        The cost matrix of the bipartite graph.\n\n    Returns\n    -------\n    row_ind, col_ind : array\n        An array of row indices and one of corresponding column indices giving\n        the optimal assignment. The cost of the assignment can be computed\n        as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be\n        sorted; in the case of a square cost matrix they will be equal to\n        ``numpy.arange(cost_matrix.shape[0])``.\n\n    Notes\n    -----\n    .. versionadded:: 0.17.0\n\n    Examples\n    --------\n    >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])\n    >>> from scipy.optimize import linear_sum_assignment\n    >>> row_ind, col_ind = linear_sum_assignment(cost)\n    >>> col_ind\n    array([1, 0, 2])\n    >>> cost[row_ind, col_ind].sum()\n    5\n\n    References\n    ----------\n    1. http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html\n\n    2. Harold W. Kuhn. The Hungarian Method for the assignment problem.\n       *Naval Research Logistics Quarterly*, 2:83-97, 1955.\n\n    3. Harold W. Kuhn. Variants of the Hungarian method for assignment\n       problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.\n\n    4. Munkres, J. Algorithms for the Assignment and Transportation Problems.\n       *J. SIAM*, 5(1):32-38, March, 1957.\n\n    5. https://en.wikipedia.org/wiki/Hungarian_algorithm\n    ')
    
    # Assigning a Call to a Name (line 82):
    
    # Assigning a Call to a Name (line 82):
    
    # Call to asarray(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'cost_matrix' (line 82)
    cost_matrix_189776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'cost_matrix', False)
    # Processing the call keyword arguments (line 82)
    kwargs_189777 = {}
    # Getting the type of 'np' (line 82)
    np_189774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'np', False)
    # Obtaining the member 'asarray' of a type (line 82)
    asarray_189775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 18), np_189774, 'asarray')
    # Calling asarray(args, kwargs) (line 82)
    asarray_call_result_189778 = invoke(stypy.reporting.localization.Localization(__file__, 82, 18), asarray_189775, *[cost_matrix_189776], **kwargs_189777)
    
    # Assigning a type to the variable 'cost_matrix' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'cost_matrix', asarray_call_result_189778)
    
    
    
    # Call to len(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'cost_matrix' (line 83)
    cost_matrix_189780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'cost_matrix', False)
    # Obtaining the member 'shape' of a type (line 83)
    shape_189781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 11), cost_matrix_189780, 'shape')
    # Processing the call keyword arguments (line 83)
    kwargs_189782 = {}
    # Getting the type of 'len' (line 83)
    len_189779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 7), 'len', False)
    # Calling len(args, kwargs) (line 83)
    len_call_result_189783 = invoke(stypy.reporting.localization.Localization(__file__, 83, 7), len_189779, *[shape_189781], **kwargs_189782)
    
    int_189784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 33), 'int')
    # Applying the binary operator '!=' (line 83)
    result_ne_189785 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 7), '!=', len_call_result_189783, int_189784)
    
    # Testing the type of an if condition (line 83)
    if_condition_189786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 4), result_ne_189785)
    # Assigning a type to the variable 'if_condition_189786' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'if_condition_189786', if_condition_189786)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 84)
    # Processing the call arguments (line 84)
    str_189788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 25), 'str', 'expected a matrix (2-d array), got a %r array')
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_189789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    # Getting the type of 'cost_matrix' (line 85)
    cost_matrix_189790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'cost_matrix', False)
    # Obtaining the member 'shape' of a type (line 85)
    shape_189791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 28), cost_matrix_189790, 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 28), tuple_189789, shape_189791)
    
    # Applying the binary operator '%' (line 84)
    result_mod_189792 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 25), '%', str_189788, tuple_189789)
    
    # Processing the call keyword arguments (line 84)
    kwargs_189793 = {}
    # Getting the type of 'ValueError' (line 84)
    ValueError_189787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 84)
    ValueError_call_result_189794 = invoke(stypy.reporting.localization.Localization(__file__, 84, 14), ValueError_189787, *[result_mod_189792], **kwargs_189793)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 84, 8), ValueError_call_result_189794, 'raise parameter', BaseException)
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Evaluating a boolean operation
    
    # Call to issubdtype(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'cost_matrix' (line 87)
    cost_matrix_189797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 26), 'cost_matrix', False)
    # Obtaining the member 'dtype' of a type (line 87)
    dtype_189798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 26), cost_matrix_189797, 'dtype')
    # Getting the type of 'np' (line 87)
    np_189799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 45), 'np', False)
    # Obtaining the member 'number' of a type (line 87)
    number_189800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 45), np_189799, 'number')
    # Processing the call keyword arguments (line 87)
    kwargs_189801 = {}
    # Getting the type of 'np' (line 87)
    np_189795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 87)
    issubdtype_189796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), np_189795, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 87)
    issubdtype_call_result_189802 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), issubdtype_189796, *[dtype_189798, number_189800], **kwargs_189801)
    
    
    # Getting the type of 'cost_matrix' (line 88)
    cost_matrix_189803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'cost_matrix')
    # Obtaining the member 'dtype' of a type (line 88)
    dtype_189804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), cost_matrix_189803, 'dtype')
    
    # Call to dtype(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'np' (line 88)
    np_189807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 42), 'np', False)
    # Obtaining the member 'bool' of a type (line 88)
    bool_189808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 42), np_189807, 'bool')
    # Processing the call keyword arguments (line 88)
    kwargs_189809 = {}
    # Getting the type of 'np' (line 88)
    np_189805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 33), 'np', False)
    # Obtaining the member 'dtype' of a type (line 88)
    dtype_189806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 33), np_189805, 'dtype')
    # Calling dtype(args, kwargs) (line 88)
    dtype_call_result_189810 = invoke(stypy.reporting.localization.Localization(__file__, 88, 33), dtype_189806, *[bool_189808], **kwargs_189809)
    
    # Applying the binary operator '==' (line 88)
    result_eq_189811 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 12), '==', dtype_189804, dtype_call_result_189810)
    
    # Applying the binary operator 'or' (line 87)
    result_or_keyword_189812 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 12), 'or', issubdtype_call_result_189802, result_eq_189811)
    
    # Applying the 'not' unary operator (line 87)
    result_not__189813 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 7), 'not', result_or_keyword_189812)
    
    # Testing the type of an if condition (line 87)
    if_condition_189814 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 4), result_not__189813)
    # Assigning a type to the variable 'if_condition_189814' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'if_condition_189814', if_condition_189814)
    # SSA begins for if statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 89)
    # Processing the call arguments (line 89)
    str_189816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 25), 'str', 'expected a matrix containing numerical entries, got %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 90)
    tuple_189817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 90)
    # Adding element type (line 90)
    # Getting the type of 'cost_matrix' (line 90)
    cost_matrix_189818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'cost_matrix', False)
    # Obtaining the member 'dtype' of a type (line 90)
    dtype_189819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 28), cost_matrix_189818, 'dtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 28), tuple_189817, dtype_189819)
    
    # Applying the binary operator '%' (line 89)
    result_mod_189820 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 25), '%', str_189816, tuple_189817)
    
    # Processing the call keyword arguments (line 89)
    kwargs_189821 = {}
    # Getting the type of 'ValueError' (line 89)
    ValueError_189815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 89)
    ValueError_call_result_189822 = invoke(stypy.reporting.localization.Localization(__file__, 89, 14), ValueError_189815, *[result_mod_189820], **kwargs_189821)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 89, 8), ValueError_call_result_189822, 'raise parameter', BaseException)
    # SSA join for if statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 92)
    # Processing the call arguments (line 92)
    
    # Call to isinf(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'cost_matrix' (line 92)
    cost_matrix_189827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'cost_matrix', False)
    # Processing the call keyword arguments (line 92)
    kwargs_189828 = {}
    # Getting the type of 'np' (line 92)
    np_189825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'np', False)
    # Obtaining the member 'isinf' of a type (line 92)
    isinf_189826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 14), np_189825, 'isinf')
    # Calling isinf(args, kwargs) (line 92)
    isinf_call_result_189829 = invoke(stypy.reporting.localization.Localization(__file__, 92, 14), isinf_189826, *[cost_matrix_189827], **kwargs_189828)
    
    
    # Call to isnan(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'cost_matrix' (line 92)
    cost_matrix_189832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 47), 'cost_matrix', False)
    # Processing the call keyword arguments (line 92)
    kwargs_189833 = {}
    # Getting the type of 'np' (line 92)
    np_189830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 38), 'np', False)
    # Obtaining the member 'isnan' of a type (line 92)
    isnan_189831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 38), np_189830, 'isnan')
    # Calling isnan(args, kwargs) (line 92)
    isnan_call_result_189834 = invoke(stypy.reporting.localization.Localization(__file__, 92, 38), isnan_189831, *[cost_matrix_189832], **kwargs_189833)
    
    # Applying the binary operator '|' (line 92)
    result_or__189835 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 14), '|', isinf_call_result_189829, isnan_call_result_189834)
    
    # Processing the call keyword arguments (line 92)
    kwargs_189836 = {}
    # Getting the type of 'np' (line 92)
    np_189823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 92)
    any_189824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 7), np_189823, 'any')
    # Calling any(args, kwargs) (line 92)
    any_call_result_189837 = invoke(stypy.reporting.localization.Localization(__file__, 92, 7), any_189824, *[result_or__189835], **kwargs_189836)
    
    # Testing the type of an if condition (line 92)
    if_condition_189838 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 4), any_call_result_189837)
    # Assigning a type to the variable 'if_condition_189838' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'if_condition_189838', if_condition_189838)
    # SSA begins for if statement (line 92)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 93)
    # Processing the call arguments (line 93)
    str_189840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'str', 'matrix contains invalid numeric entries')
    # Processing the call keyword arguments (line 93)
    kwargs_189841 = {}
    # Getting the type of 'ValueError' (line 93)
    ValueError_189839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 93)
    ValueError_call_result_189842 = invoke(stypy.reporting.localization.Localization(__file__, 93, 14), ValueError_189839, *[str_189840], **kwargs_189841)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 93, 8), ValueError_call_result_189842, 'raise parameter', BaseException)
    # SSA join for if statement (line 92)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cost_matrix' (line 95)
    cost_matrix_189843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 7), 'cost_matrix')
    # Obtaining the member 'dtype' of a type (line 95)
    dtype_189844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 7), cost_matrix_189843, 'dtype')
    
    # Call to dtype(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'np' (line 95)
    np_189847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 37), 'np', False)
    # Obtaining the member 'bool' of a type (line 95)
    bool_189848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 37), np_189847, 'bool')
    # Processing the call keyword arguments (line 95)
    kwargs_189849 = {}
    # Getting the type of 'np' (line 95)
    np_189845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'np', False)
    # Obtaining the member 'dtype' of a type (line 95)
    dtype_189846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 28), np_189845, 'dtype')
    # Calling dtype(args, kwargs) (line 95)
    dtype_call_result_189850 = invoke(stypy.reporting.localization.Localization(__file__, 95, 28), dtype_189846, *[bool_189848], **kwargs_189849)
    
    # Applying the binary operator '==' (line 95)
    result_eq_189851 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 7), '==', dtype_189844, dtype_call_result_189850)
    
    # Testing the type of an if condition (line 95)
    if_condition_189852 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 4), result_eq_189851)
    # Assigning a type to the variable 'if_condition_189852' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'if_condition_189852', if_condition_189852)
    # SSA begins for if statement (line 95)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to astype(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'np' (line 96)
    np_189855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 41), 'np', False)
    # Obtaining the member 'int' of a type (line 96)
    int_189856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 41), np_189855, 'int')
    # Processing the call keyword arguments (line 96)
    kwargs_189857 = {}
    # Getting the type of 'cost_matrix' (line 96)
    cost_matrix_189853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'cost_matrix', False)
    # Obtaining the member 'astype' of a type (line 96)
    astype_189854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 22), cost_matrix_189853, 'astype')
    # Calling astype(args, kwargs) (line 96)
    astype_call_result_189858 = invoke(stypy.reporting.localization.Localization(__file__, 96, 22), astype_189854, *[int_189856], **kwargs_189857)
    
    # Assigning a type to the variable 'cost_matrix' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'cost_matrix', astype_call_result_189858)
    # SSA join for if statement (line 95)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_189859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 25), 'int')
    # Getting the type of 'cost_matrix' (line 99)
    cost_matrix_189860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 7), 'cost_matrix')
    # Obtaining the member 'shape' of a type (line 99)
    shape_189861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 7), cost_matrix_189860, 'shape')
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___189862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 7), shape_189861, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_189863 = invoke(stypy.reporting.localization.Localization(__file__, 99, 7), getitem___189862, int_189859)
    
    
    # Obtaining the type of the subscript
    int_189864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 48), 'int')
    # Getting the type of 'cost_matrix' (line 99)
    cost_matrix_189865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 30), 'cost_matrix')
    # Obtaining the member 'shape' of a type (line 99)
    shape_189866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 30), cost_matrix_189865, 'shape')
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___189867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 30), shape_189866, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_189868 = invoke(stypy.reporting.localization.Localization(__file__, 99, 30), getitem___189867, int_189864)
    
    # Applying the binary operator '<' (line 99)
    result_lt_189869 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), '<', subscript_call_result_189863, subscript_call_result_189868)
    
    # Testing the type of an if condition (line 99)
    if_condition_189870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 4), result_lt_189869)
    # Assigning a type to the variable 'if_condition_189870' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'if_condition_189870', if_condition_189870)
    # SSA begins for if statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 100):
    
    # Assigning a Attribute to a Name (line 100):
    # Getting the type of 'cost_matrix' (line 100)
    cost_matrix_189871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'cost_matrix')
    # Obtaining the member 'T' of a type (line 100)
    T_189872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 22), cost_matrix_189871, 'T')
    # Assigning a type to the variable 'cost_matrix' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'cost_matrix', T_189872)
    
    # Assigning a Name to a Name (line 101):
    
    # Assigning a Name to a Name (line 101):
    # Getting the type of 'True' (line 101)
    True_189873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'True')
    # Assigning a type to the variable 'transposed' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'transposed', True_189873)
    # SSA branch for the else part of an if statement (line 99)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 103):
    
    # Assigning a Name to a Name (line 103):
    # Getting the type of 'False' (line 103)
    False_189874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 21), 'False')
    # Assigning a type to the variable 'transposed' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'transposed', False_189874)
    # SSA join for if statement (line 99)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 105):
    
    # Assigning a Call to a Name (line 105):
    
    # Call to _Hungary(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'cost_matrix' (line 105)
    cost_matrix_189876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'cost_matrix', False)
    # Processing the call keyword arguments (line 105)
    kwargs_189877 = {}
    # Getting the type of '_Hungary' (line 105)
    _Hungary_189875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), '_Hungary', False)
    # Calling _Hungary(args, kwargs) (line 105)
    _Hungary_call_result_189878 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), _Hungary_189875, *[cost_matrix_189876], **kwargs_189877)
    
    # Assigning a type to the variable 'state' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'state', _Hungary_call_result_189878)
    
    # Assigning a IfExp to a Name (line 109):
    
    # Assigning a IfExp to a Name (line 109):
    
    
    int_189879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 19), 'int')
    # Getting the type of 'cost_matrix' (line 109)
    cost_matrix_189880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 24), 'cost_matrix')
    # Obtaining the member 'shape' of a type (line 109)
    shape_189881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 24), cost_matrix_189880, 'shape')
    # Applying the binary operator 'in' (line 109)
    result_contains_189882 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 19), 'in', int_189879, shape_189881)
    
    # Testing the type of an if expression (line 109)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 11), result_contains_189882)
    # SSA begins for if expression (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'None' (line 109)
    None_189883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'None')
    # SSA branch for the else part of an if expression (line 109)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of '_step1' (line 109)
    _step1_189884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 47), '_step1')
    # SSA join for if expression (line 109)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_189885 = union_type.UnionType.add(None_189883, _step1_189884)
    
    # Assigning a type to the variable 'step' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'step', if_exp_189885)
    
    
    # Getting the type of 'step' (line 111)
    step_189886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 10), 'step')
    # Getting the type of 'None' (line 111)
    None_189887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'None')
    # Applying the binary operator 'isnot' (line 111)
    result_is_not_189888 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 10), 'isnot', step_189886, None_189887)
    
    # Testing the type of an if condition (line 111)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 4), result_is_not_189888)
    # SSA begins for while statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 112):
    
    # Assigning a Call to a Name (line 112):
    
    # Call to step(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'state' (line 112)
    state_189890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'state', False)
    # Processing the call keyword arguments (line 112)
    kwargs_189891 = {}
    # Getting the type of 'step' (line 112)
    step_189889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'step', False)
    # Calling step(args, kwargs) (line 112)
    step_call_result_189892 = invoke(stypy.reporting.localization.Localization(__file__, 112, 15), step_189889, *[state_189890], **kwargs_189891)
    
    # Assigning a type to the variable 'step' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'step', step_call_result_189892)
    # SSA join for while statement (line 111)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'transposed' (line 114)
    transposed_189893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 7), 'transposed')
    # Testing the type of an if condition (line 114)
    if_condition_189894 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 4), transposed_189893)
    # Assigning a type to the variable 'if_condition_189894' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'if_condition_189894', if_condition_189894)
    # SSA begins for if statement (line 114)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 115):
    
    # Assigning a Attribute to a Name (line 115):
    # Getting the type of 'state' (line 115)
    state_189895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'state')
    # Obtaining the member 'marked' of a type (line 115)
    marked_189896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 17), state_189895, 'marked')
    # Obtaining the member 'T' of a type (line 115)
    T_189897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 17), marked_189896, 'T')
    # Assigning a type to the variable 'marked' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'marked', T_189897)
    # SSA branch for the else part of an if statement (line 114)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 117):
    
    # Assigning a Attribute to a Name (line 117):
    # Getting the type of 'state' (line 117)
    state_189898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'state')
    # Obtaining the member 'marked' of a type (line 117)
    marked_189899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 17), state_189898, 'marked')
    # Assigning a type to the variable 'marked' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'marked', marked_189899)
    # SSA join for if statement (line 114)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to where(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Getting the type of 'marked' (line 118)
    marked_189902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'marked', False)
    int_189903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 30), 'int')
    # Applying the binary operator '==' (line 118)
    result_eq_189904 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 20), '==', marked_189902, int_189903)
    
    # Processing the call keyword arguments (line 118)
    kwargs_189905 = {}
    # Getting the type of 'np' (line 118)
    np_189900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'np', False)
    # Obtaining the member 'where' of a type (line 118)
    where_189901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 11), np_189900, 'where')
    # Calling where(args, kwargs) (line 118)
    where_call_result_189906 = invoke(stypy.reporting.localization.Localization(__file__, 118, 11), where_189901, *[result_eq_189904], **kwargs_189905)
    
    # Assigning a type to the variable 'stypy_return_type' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type', where_call_result_189906)
    
    # ################# End of 'linear_sum_assignment(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'linear_sum_assignment' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_189907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_189907)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'linear_sum_assignment'
    return stypy_return_type_189907

# Assigning a type to the variable 'linear_sum_assignment' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'linear_sum_assignment', linear_sum_assignment)
# Declaration of the '_Hungary' class

class _Hungary(object, ):
    str_189908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, (-1)), 'str', 'State of the Hungarian algorithm.\n\n    Parameters\n    ----------\n    cost_matrix : 2D matrix\n        The cost matrix. Must have shape[1] >= shape[0].\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Hungary.__init__', ['cost_matrix'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['cost_matrix'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 131):
        
        # Assigning a Call to a Attribute (line 131):
        
        # Call to copy(...): (line 131)
        # Processing the call keyword arguments (line 131)
        kwargs_189911 = {}
        # Getting the type of 'cost_matrix' (line 131)
        cost_matrix_189909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 17), 'cost_matrix', False)
        # Obtaining the member 'copy' of a type (line 131)
        copy_189910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 17), cost_matrix_189909, 'copy')
        # Calling copy(args, kwargs) (line 131)
        copy_call_result_189912 = invoke(stypy.reporting.localization.Localization(__file__, 131, 17), copy_189910, *[], **kwargs_189911)
        
        # Getting the type of 'self' (line 131)
        self_189913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self')
        # Setting the type of the member 'C' of a type (line 131)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_189913, 'C', copy_call_result_189912)
        
        # Assigning a Attribute to a Tuple (line 133):
        
        # Assigning a Subscript to a Name (line 133):
        
        # Obtaining the type of the subscript
        int_189914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'int')
        # Getting the type of 'self' (line 133)
        self_189915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'self')
        # Obtaining the member 'C' of a type (line 133)
        C_189916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 15), self_189915, 'C')
        # Obtaining the member 'shape' of a type (line 133)
        shape_189917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 15), C_189916, 'shape')
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___189918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), shape_189917, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_189919 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), getitem___189918, int_189914)
        
        # Assigning a type to the variable 'tuple_var_assignment_189767' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_189767', subscript_call_result_189919)
        
        # Assigning a Subscript to a Name (line 133):
        
        # Obtaining the type of the subscript
        int_189920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'int')
        # Getting the type of 'self' (line 133)
        self_189921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'self')
        # Obtaining the member 'C' of a type (line 133)
        C_189922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 15), self_189921, 'C')
        # Obtaining the member 'shape' of a type (line 133)
        shape_189923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 15), C_189922, 'shape')
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___189924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), shape_189923, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_189925 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), getitem___189924, int_189920)
        
        # Assigning a type to the variable 'tuple_var_assignment_189768' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_189768', subscript_call_result_189925)
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'tuple_var_assignment_189767' (line 133)
        tuple_var_assignment_189767_189926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_189767')
        # Assigning a type to the variable 'n' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'n', tuple_var_assignment_189767_189926)
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'tuple_var_assignment_189768' (line 133)
        tuple_var_assignment_189768_189927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_189768')
        # Assigning a type to the variable 'm' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'm', tuple_var_assignment_189768_189927)
        
        # Assigning a Call to a Attribute (line 134):
        
        # Assigning a Call to a Attribute (line 134):
        
        # Call to ones(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'n' (line 134)
        n_189930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 37), 'n', False)
        # Processing the call keyword arguments (line 134)
        # Getting the type of 'bool' (line 134)
        bool_189931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 46), 'bool', False)
        keyword_189932 = bool_189931
        kwargs_189933 = {'dtype': keyword_189932}
        # Getting the type of 'np' (line 134)
        np_189928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 29), 'np', False)
        # Obtaining the member 'ones' of a type (line 134)
        ones_189929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 29), np_189928, 'ones')
        # Calling ones(args, kwargs) (line 134)
        ones_call_result_189934 = invoke(stypy.reporting.localization.Localization(__file__, 134, 29), ones_189929, *[n_189930], **kwargs_189933)
        
        # Getting the type of 'self' (line 134)
        self_189935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'self')
        # Setting the type of the member 'row_uncovered' of a type (line 134)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), self_189935, 'row_uncovered', ones_call_result_189934)
        
        # Assigning a Call to a Attribute (line 135):
        
        # Assigning a Call to a Attribute (line 135):
        
        # Call to ones(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'm' (line 135)
        m_189938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 37), 'm', False)
        # Processing the call keyword arguments (line 135)
        # Getting the type of 'bool' (line 135)
        bool_189939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 46), 'bool', False)
        keyword_189940 = bool_189939
        kwargs_189941 = {'dtype': keyword_189940}
        # Getting the type of 'np' (line 135)
        np_189936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'np', False)
        # Obtaining the member 'ones' of a type (line 135)
        ones_189937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 29), np_189936, 'ones')
        # Calling ones(args, kwargs) (line 135)
        ones_call_result_189942 = invoke(stypy.reporting.localization.Localization(__file__, 135, 29), ones_189937, *[m_189938], **kwargs_189941)
        
        # Getting the type of 'self' (line 135)
        self_189943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'self')
        # Setting the type of the member 'col_uncovered' of a type (line 135)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), self_189943, 'col_uncovered', ones_call_result_189942)
        
        # Assigning a Num to a Attribute (line 136):
        
        # Assigning a Num to a Attribute (line 136):
        int_189944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 20), 'int')
        # Getting the type of 'self' (line 136)
        self_189945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'self')
        # Setting the type of the member 'Z0_r' of a type (line 136)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), self_189945, 'Z0_r', int_189944)
        
        # Assigning a Num to a Attribute (line 137):
        
        # Assigning a Num to a Attribute (line 137):
        int_189946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 20), 'int')
        # Getting the type of 'self' (line 137)
        self_189947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self')
        # Setting the type of the member 'Z0_c' of a type (line 137)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_189947, 'Z0_c', int_189946)
        
        # Assigning a Call to a Attribute (line 138):
        
        # Assigning a Call to a Attribute (line 138):
        
        # Call to zeros(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Obtaining an instance of the builtin type 'tuple' (line 138)
        tuple_189950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 138)
        # Adding element type (line 138)
        # Getting the type of 'n' (line 138)
        n_189951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 30), 'n', False)
        # Getting the type of 'm' (line 138)
        m_189952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 34), 'm', False)
        # Applying the binary operator '+' (line 138)
        result_add_189953 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 30), '+', n_189951, m_189952)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 30), tuple_189950, result_add_189953)
        # Adding element type (line 138)
        int_189954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 30), tuple_189950, int_189954)
        
        # Processing the call keyword arguments (line 138)
        # Getting the type of 'int' (line 138)
        int_189955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 47), 'int', False)
        keyword_189956 = int_189955
        kwargs_189957 = {'dtype': keyword_189956}
        # Getting the type of 'np' (line 138)
        np_189948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'np', False)
        # Obtaining the member 'zeros' of a type (line 138)
        zeros_189949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 20), np_189948, 'zeros')
        # Calling zeros(args, kwargs) (line 138)
        zeros_call_result_189958 = invoke(stypy.reporting.localization.Localization(__file__, 138, 20), zeros_189949, *[tuple_189950], **kwargs_189957)
        
        # Getting the type of 'self' (line 138)
        self_189959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self')
        # Setting the type of the member 'path' of a type (line 138)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_189959, 'path', zeros_call_result_189958)
        
        # Assigning a Call to a Attribute (line 139):
        
        # Assigning a Call to a Attribute (line 139):
        
        # Call to zeros(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Obtaining an instance of the builtin type 'tuple' (line 139)
        tuple_189962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 139)
        # Adding element type (line 139)
        # Getting the type of 'n' (line 139)
        n_189963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 32), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 32), tuple_189962, n_189963)
        # Adding element type (line 139)
        # Getting the type of 'm' (line 139)
        m_189964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 35), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 32), tuple_189962, m_189964)
        
        # Processing the call keyword arguments (line 139)
        # Getting the type of 'int' (line 139)
        int_189965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 45), 'int', False)
        keyword_189966 = int_189965
        kwargs_189967 = {'dtype': keyword_189966}
        # Getting the type of 'np' (line 139)
        np_189960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'np', False)
        # Obtaining the member 'zeros' of a type (line 139)
        zeros_189961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 22), np_189960, 'zeros')
        # Calling zeros(args, kwargs) (line 139)
        zeros_call_result_189968 = invoke(stypy.reporting.localization.Localization(__file__, 139, 22), zeros_189961, *[tuple_189962], **kwargs_189967)
        
        # Getting the type of 'self' (line 139)
        self_189969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self')
        # Setting the type of the member 'marked' of a type (line 139)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_189969, 'marked', zeros_call_result_189968)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _clear_covers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_clear_covers'
        module_type_store = module_type_store.open_function_context('_clear_covers', 141, 4, False)
        # Assigning a type to the variable 'self' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _Hungary._clear_covers.__dict__.__setitem__('stypy_localization', localization)
        _Hungary._clear_covers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _Hungary._clear_covers.__dict__.__setitem__('stypy_type_store', module_type_store)
        _Hungary._clear_covers.__dict__.__setitem__('stypy_function_name', '_Hungary._clear_covers')
        _Hungary._clear_covers.__dict__.__setitem__('stypy_param_names_list', [])
        _Hungary._clear_covers.__dict__.__setitem__('stypy_varargs_param_name', None)
        _Hungary._clear_covers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _Hungary._clear_covers.__dict__.__setitem__('stypy_call_defaults', defaults)
        _Hungary._clear_covers.__dict__.__setitem__('stypy_call_varargs', varargs)
        _Hungary._clear_covers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _Hungary._clear_covers.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Hungary._clear_covers', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_clear_covers', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_clear_covers(...)' code ##################

        str_189970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 8), 'str', 'Clear all covered matrix cells')
        
        # Assigning a Name to a Subscript (line 143):
        
        # Assigning a Name to a Subscript (line 143):
        # Getting the type of 'True' (line 143)
        True_189971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 32), 'True')
        # Getting the type of 'self' (line 143)
        self_189972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self')
        # Obtaining the member 'row_uncovered' of a type (line 143)
        row_uncovered_189973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_189972, 'row_uncovered')
        slice_189974 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 143, 8), None, None, None)
        # Storing an element on a container (line 143)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 8), row_uncovered_189973, (slice_189974, True_189971))
        
        # Assigning a Name to a Subscript (line 144):
        
        # Assigning a Name to a Subscript (line 144):
        # Getting the type of 'True' (line 144)
        True_189975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), 'True')
        # Getting the type of 'self' (line 144)
        self_189976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self')
        # Obtaining the member 'col_uncovered' of a type (line 144)
        col_uncovered_189977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_189976, 'col_uncovered')
        slice_189978 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 144, 8), None, None, None)
        # Storing an element on a container (line 144)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 8), col_uncovered_189977, (slice_189978, True_189975))
        
        # ################# End of '_clear_covers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_clear_covers' in the type store
        # Getting the type of 'stypy_return_type' (line 141)
        stypy_return_type_189979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189979)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_clear_covers'
        return stypy_return_type_189979


# Assigning a type to the variable '_Hungary' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), '_Hungary', _Hungary)

@norecursion
def _step1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_step1'
    module_type_store = module_type_store.open_function_context('_step1', 150, 0, False)
    
    # Passed parameters checking function
    _step1.stypy_localization = localization
    _step1.stypy_type_of_self = None
    _step1.stypy_type_store = module_type_store
    _step1.stypy_function_name = '_step1'
    _step1.stypy_param_names_list = ['state']
    _step1.stypy_varargs_param_name = None
    _step1.stypy_kwargs_param_name = None
    _step1.stypy_call_defaults = defaults
    _step1.stypy_call_varargs = varargs
    _step1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_step1', ['state'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_step1', localization, ['state'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_step1(...)' code ##################

    str_189980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 4), 'str', 'Steps 1 and 2 in the Wikipedia page.')
    
    # Getting the type of 'state' (line 155)
    state_189981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'state')
    # Obtaining the member 'C' of a type (line 155)
    C_189982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), state_189981, 'C')
    
    # Obtaining the type of the subscript
    slice_189983 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 155, 15), None, None, None)
    # Getting the type of 'np' (line 155)
    np_189984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 38), 'np')
    # Obtaining the member 'newaxis' of a type (line 155)
    newaxis_189985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 38), np_189984, 'newaxis')
    
    # Call to min(...): (line 155)
    # Processing the call keyword arguments (line 155)
    int_189989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 32), 'int')
    keyword_189990 = int_189989
    kwargs_189991 = {'axis': keyword_189990}
    # Getting the type of 'state' (line 155)
    state_189986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'state', False)
    # Obtaining the member 'C' of a type (line 155)
    C_189987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 15), state_189986, 'C')
    # Obtaining the member 'min' of a type (line 155)
    min_189988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 15), C_189987, 'min')
    # Calling min(args, kwargs) (line 155)
    min_call_result_189992 = invoke(stypy.reporting.localization.Localization(__file__, 155, 15), min_189988, *[], **kwargs_189991)
    
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___189993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 15), min_call_result_189992, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_189994 = invoke(stypy.reporting.localization.Localization(__file__, 155, 15), getitem___189993, (slice_189983, newaxis_189985))
    
    # Applying the binary operator '-=' (line 155)
    result_isub_189995 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 4), '-=', C_189982, subscript_call_result_189994)
    # Getting the type of 'state' (line 155)
    state_189996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'state')
    # Setting the type of the member 'C' of a type (line 155)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), state_189996, 'C', result_isub_189995)
    
    
    
    # Call to zip(...): (line 159)
    
    # Call to where(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Getting the type of 'state' (line 159)
    state_190000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 30), 'state', False)
    # Obtaining the member 'C' of a type (line 159)
    C_190001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 30), state_190000, 'C')
    int_190002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 41), 'int')
    # Applying the binary operator '==' (line 159)
    result_eq_190003 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 30), '==', C_190001, int_190002)
    
    # Processing the call keyword arguments (line 159)
    kwargs_190004 = {}
    # Getting the type of 'np' (line 159)
    np_189998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 21), 'np', False)
    # Obtaining the member 'where' of a type (line 159)
    where_189999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 21), np_189998, 'where')
    # Calling where(args, kwargs) (line 159)
    where_call_result_190005 = invoke(stypy.reporting.localization.Localization(__file__, 159, 21), where_189999, *[result_eq_190003], **kwargs_190004)
    
    # Processing the call keyword arguments (line 159)
    kwargs_190006 = {}
    # Getting the type of 'zip' (line 159)
    zip_189997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'zip', False)
    # Calling zip(args, kwargs) (line 159)
    zip_call_result_190007 = invoke(stypy.reporting.localization.Localization(__file__, 159, 16), zip_189997, *[where_call_result_190005], **kwargs_190006)
    
    # Testing the type of a for loop iterable (line 159)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 159, 4), zip_call_result_190007)
    # Getting the type of the for loop variable (line 159)
    for_loop_var_190008 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 159, 4), zip_call_result_190007)
    # Assigning a type to the variable 'i' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 4), for_loop_var_190008))
    # Assigning a type to the variable 'j' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 4), for_loop_var_190008))
    # SSA begins for a for statement (line 159)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 160)
    j_190009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 31), 'j')
    # Getting the type of 'state' (line 160)
    state_190010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'state')
    # Obtaining the member 'col_uncovered' of a type (line 160)
    col_uncovered_190011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 11), state_190010, 'col_uncovered')
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___190012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 11), col_uncovered_190011, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_190013 = invoke(stypy.reporting.localization.Localization(__file__, 160, 11), getitem___190012, j_190009)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 160)
    i_190014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 58), 'i')
    # Getting the type of 'state' (line 160)
    state_190015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 38), 'state')
    # Obtaining the member 'row_uncovered' of a type (line 160)
    row_uncovered_190016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 38), state_190015, 'row_uncovered')
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___190017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 38), row_uncovered_190016, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_190018 = invoke(stypy.reporting.localization.Localization(__file__, 160, 38), getitem___190017, i_190014)
    
    # Applying the binary operator 'and' (line 160)
    result_and_keyword_190019 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 11), 'and', subscript_call_result_190013, subscript_call_result_190018)
    
    # Testing the type of an if condition (line 160)
    if_condition_190020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 8), result_and_keyword_190019)
    # Assigning a type to the variable 'if_condition_190020' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'if_condition_190020', if_condition_190020)
    # SSA begins for if statement (line 160)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 161):
    
    # Assigning a Num to a Subscript (line 161):
    int_190021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 33), 'int')
    # Getting the type of 'state' (line 161)
    state_190022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'state')
    # Obtaining the member 'marked' of a type (line 161)
    marked_190023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), state_190022, 'marked')
    
    # Obtaining an instance of the builtin type 'tuple' (line 161)
    tuple_190024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 161)
    # Adding element type (line 161)
    # Getting the type of 'i' (line 161)
    i_190025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 25), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 25), tuple_190024, i_190025)
    # Adding element type (line 161)
    # Getting the type of 'j' (line 161)
    j_190026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 25), tuple_190024, j_190026)
    
    # Storing an element on a container (line 161)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 12), marked_190023, (tuple_190024, int_190021))
    
    # Assigning a Name to a Subscript (line 162):
    
    # Assigning a Name to a Subscript (line 162):
    # Getting the type of 'False' (line 162)
    False_190027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 37), 'False')
    # Getting the type of 'state' (line 162)
    state_190028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'state')
    # Obtaining the member 'col_uncovered' of a type (line 162)
    col_uncovered_190029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), state_190028, 'col_uncovered')
    # Getting the type of 'j' (line 162)
    j_190030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'j')
    # Storing an element on a container (line 162)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 12), col_uncovered_190029, (j_190030, False_190027))
    
    # Assigning a Name to a Subscript (line 163):
    
    # Assigning a Name to a Subscript (line 163):
    # Getting the type of 'False' (line 163)
    False_190031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'False')
    # Getting the type of 'state' (line 163)
    state_190032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'state')
    # Obtaining the member 'row_uncovered' of a type (line 163)
    row_uncovered_190033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), state_190032, 'row_uncovered')
    # Getting the type of 'i' (line 163)
    i_190034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 32), 'i')
    # Storing an element on a container (line 163)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 12), row_uncovered_190033, (i_190034, False_190031))
    # SSA join for if statement (line 160)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _clear_covers(...): (line 165)
    # Processing the call keyword arguments (line 165)
    kwargs_190037 = {}
    # Getting the type of 'state' (line 165)
    state_190035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'state', False)
    # Obtaining the member '_clear_covers' of a type (line 165)
    _clear_covers_190036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 4), state_190035, '_clear_covers')
    # Calling _clear_covers(args, kwargs) (line 165)
    _clear_covers_call_result_190038 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), _clear_covers_190036, *[], **kwargs_190037)
    
    # Getting the type of '_step3' (line 166)
    _step3_190039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), '_step3')
    # Assigning a type to the variable 'stypy_return_type' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'stypy_return_type', _step3_190039)
    
    # ################# End of '_step1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_step1' in the type store
    # Getting the type of 'stypy_return_type' (line 150)
    stypy_return_type_190040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190040)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_step1'
    return stypy_return_type_190040

# Assigning a type to the variable '_step1' (line 150)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 0), '_step1', _step1)

@norecursion
def _step3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_step3'
    module_type_store = module_type_store.open_function_context('_step3', 169, 0, False)
    
    # Passed parameters checking function
    _step3.stypy_localization = localization
    _step3.stypy_type_of_self = None
    _step3.stypy_type_store = module_type_store
    _step3.stypy_function_name = '_step3'
    _step3.stypy_param_names_list = ['state']
    _step3.stypy_varargs_param_name = None
    _step3.stypy_kwargs_param_name = None
    _step3.stypy_call_defaults = defaults
    _step3.stypy_call_varargs = varargs
    _step3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_step3', ['state'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_step3', localization, ['state'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_step3(...)' code ##################

    str_190041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, (-1)), 'str', '\n    Cover each column containing a starred zero. If n columns are covered,\n    the starred zeros describe a complete set of unique assignments.\n    In this case, Go to DONE, otherwise, Go to Step 4.\n    ')
    
    # Assigning a Compare to a Name (line 175):
    
    # Assigning a Compare to a Name (line 175):
    
    # Getting the type of 'state' (line 175)
    state_190042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 14), 'state')
    # Obtaining the member 'marked' of a type (line 175)
    marked_190043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 14), state_190042, 'marked')
    int_190044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 30), 'int')
    # Applying the binary operator '==' (line 175)
    result_eq_190045 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 14), '==', marked_190043, int_190044)
    
    # Assigning a type to the variable 'marked' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'marked', result_eq_190045)
    
    # Assigning a Name to a Subscript (line 176):
    
    # Assigning a Name to a Subscript (line 176):
    # Getting the type of 'False' (line 176)
    False_190046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 50), 'False')
    # Getting the type of 'state' (line 176)
    state_190047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'state')
    # Obtaining the member 'col_uncovered' of a type (line 176)
    col_uncovered_190048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), state_190047, 'col_uncovered')
    
    # Call to any(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'marked' (line 176)
    marked_190051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 31), 'marked', False)
    # Processing the call keyword arguments (line 176)
    int_190052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 44), 'int')
    keyword_190053 = int_190052
    kwargs_190054 = {'axis': keyword_190053}
    # Getting the type of 'np' (line 176)
    np_190049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'np', False)
    # Obtaining the member 'any' of a type (line 176)
    any_190050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 24), np_190049, 'any')
    # Calling any(args, kwargs) (line 176)
    any_call_result_190055 = invoke(stypy.reporting.localization.Localization(__file__, 176, 24), any_190050, *[marked_190051], **kwargs_190054)
    
    # Storing an element on a container (line 176)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 4), col_uncovered_190048, (any_call_result_190055, False_190046))
    
    
    
    # Call to sum(...): (line 178)
    # Processing the call keyword arguments (line 178)
    kwargs_190058 = {}
    # Getting the type of 'marked' (line 178)
    marked_190056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 7), 'marked', False)
    # Obtaining the member 'sum' of a type (line 178)
    sum_190057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 7), marked_190056, 'sum')
    # Calling sum(args, kwargs) (line 178)
    sum_call_result_190059 = invoke(stypy.reporting.localization.Localization(__file__, 178, 7), sum_190057, *[], **kwargs_190058)
    
    
    # Obtaining the type of the subscript
    int_190060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 36), 'int')
    # Getting the type of 'state' (line 178)
    state_190061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 22), 'state')
    # Obtaining the member 'C' of a type (line 178)
    C_190062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 22), state_190061, 'C')
    # Obtaining the member 'shape' of a type (line 178)
    shape_190063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 22), C_190062, 'shape')
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___190064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 22), shape_190063, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_190065 = invoke(stypy.reporting.localization.Localization(__file__, 178, 22), getitem___190064, int_190060)
    
    # Applying the binary operator '<' (line 178)
    result_lt_190066 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 7), '<', sum_call_result_190059, subscript_call_result_190065)
    
    # Testing the type of an if condition (line 178)
    if_condition_190067 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 4), result_lt_190066)
    # Assigning a type to the variable 'if_condition_190067' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'if_condition_190067', if_condition_190067)
    # SSA begins for if statement (line 178)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_step4' (line 179)
    _step4_190068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 15), '_step4')
    # Assigning a type to the variable 'stypy_return_type' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'stypy_return_type', _step4_190068)
    # SSA join for if statement (line 178)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_step3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_step3' in the type store
    # Getting the type of 'stypy_return_type' (line 169)
    stypy_return_type_190069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190069)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_step3'
    return stypy_return_type_190069

# Assigning a type to the variable '_step3' (line 169)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), '_step3', _step3)

@norecursion
def _step4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_step4'
    module_type_store = module_type_store.open_function_context('_step4', 182, 0, False)
    
    # Passed parameters checking function
    _step4.stypy_localization = localization
    _step4.stypy_type_of_self = None
    _step4.stypy_type_store = module_type_store
    _step4.stypy_function_name = '_step4'
    _step4.stypy_param_names_list = ['state']
    _step4.stypy_varargs_param_name = None
    _step4.stypy_kwargs_param_name = None
    _step4.stypy_call_defaults = defaults
    _step4.stypy_call_varargs = varargs
    _step4.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_step4', ['state'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_step4', localization, ['state'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_step4(...)' code ##################

    str_190070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, (-1)), 'str', '\n    Find a noncovered zero and prime it. If there is no starred zero\n    in the row containing this primed zero, Go to Step 5. Otherwise,\n    cover this row and uncover the column containing the starred\n    zero. Continue in this manner until there are no uncovered zeros\n    left. Save the smallest uncovered value and Go to Step 6.\n    ')
    
    # Assigning a Call to a Name (line 191):
    
    # Assigning a Call to a Name (line 191):
    
    # Call to astype(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'int' (line 191)
    int_190076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), 'int', False)
    # Processing the call keyword arguments (line 191)
    kwargs_190077 = {}
    
    # Getting the type of 'state' (line 191)
    state_190071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 9), 'state', False)
    # Obtaining the member 'C' of a type (line 191)
    C_190072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 9), state_190071, 'C')
    int_190073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 20), 'int')
    # Applying the binary operator '==' (line 191)
    result_eq_190074 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 9), '==', C_190072, int_190073)
    
    # Obtaining the member 'astype' of a type (line 191)
    astype_190075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 9), result_eq_190074, 'astype')
    # Calling astype(args, kwargs) (line 191)
    astype_call_result_190078 = invoke(stypy.reporting.localization.Localization(__file__, 191, 9), astype_190075, *[int_190076], **kwargs_190077)
    
    # Assigning a type to the variable 'C' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'C', astype_call_result_190078)
    
    # Assigning a BinOp to a Name (line 192):
    
    # Assigning a BinOp to a Name (line 192):
    # Getting the type of 'C' (line 192)
    C_190079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'C')
    
    # Obtaining the type of the subscript
    slice_190080 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 192, 20), None, None, None)
    # Getting the type of 'np' (line 192)
    np_190081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 43), 'np')
    # Obtaining the member 'newaxis' of a type (line 192)
    newaxis_190082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 43), np_190081, 'newaxis')
    # Getting the type of 'state' (line 192)
    state_190083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'state')
    # Obtaining the member 'row_uncovered' of a type (line 192)
    row_uncovered_190084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 20), state_190083, 'row_uncovered')
    # Obtaining the member '__getitem__' of a type (line 192)
    getitem___190085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 20), row_uncovered_190084, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 192)
    subscript_call_result_190086 = invoke(stypy.reporting.localization.Localization(__file__, 192, 20), getitem___190085, (slice_190080, newaxis_190082))
    
    # Applying the binary operator '*' (line 192)
    result_mul_190087 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 16), '*', C_190079, subscript_call_result_190086)
    
    # Assigning a type to the variable 'covered_C' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'covered_C', result_mul_190087)
    
    # Getting the type of 'covered_C' (line 193)
    covered_C_190088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'covered_C')
    
    # Call to asarray(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'state' (line 193)
    state_190091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 28), 'state', False)
    # Obtaining the member 'col_uncovered' of a type (line 193)
    col_uncovered_190092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 28), state_190091, 'col_uncovered')
    # Processing the call keyword arguments (line 193)
    # Getting the type of 'int' (line 193)
    int_190093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 55), 'int', False)
    keyword_190094 = int_190093
    kwargs_190095 = {'dtype': keyword_190094}
    # Getting the type of 'np' (line 193)
    np_190089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 17), 'np', False)
    # Obtaining the member 'asarray' of a type (line 193)
    asarray_190090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 17), np_190089, 'asarray')
    # Calling asarray(args, kwargs) (line 193)
    asarray_call_result_190096 = invoke(stypy.reporting.localization.Localization(__file__, 193, 17), asarray_190090, *[col_uncovered_190092], **kwargs_190095)
    
    # Applying the binary operator '*=' (line 193)
    result_imul_190097 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 4), '*=', covered_C_190088, asarray_call_result_190096)
    # Assigning a type to the variable 'covered_C' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'covered_C', result_imul_190097)
    
    
    # Assigning a Subscript to a Name (line 194):
    
    # Assigning a Subscript to a Name (line 194):
    
    # Obtaining the type of the subscript
    int_190098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 22), 'int')
    # Getting the type of 'state' (line 194)
    state_190099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'state')
    # Obtaining the member 'C' of a type (line 194)
    C_190100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), state_190099, 'C')
    # Obtaining the member 'shape' of a type (line 194)
    shape_190101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), C_190100, 'shape')
    # Obtaining the member '__getitem__' of a type (line 194)
    getitem___190102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), shape_190101, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 194)
    subscript_call_result_190103 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), getitem___190102, int_190098)
    
    # Assigning a type to the variable 'n' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'n', subscript_call_result_190103)
    
    # Assigning a Subscript to a Name (line 195):
    
    # Assigning a Subscript to a Name (line 195):
    
    # Obtaining the type of the subscript
    int_190104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 22), 'int')
    # Getting the type of 'state' (line 195)
    state_190105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'state')
    # Obtaining the member 'C' of a type (line 195)
    C_190106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), state_190105, 'C')
    # Obtaining the member 'shape' of a type (line 195)
    shape_190107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), C_190106, 'shape')
    # Obtaining the member '__getitem__' of a type (line 195)
    getitem___190108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), shape_190107, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 195)
    subscript_call_result_190109 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), getitem___190108, int_190104)
    
    # Assigning a type to the variable 'm' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'm', subscript_call_result_190109)
    
    # Getting the type of 'True' (line 197)
    True_190110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 10), 'True')
    # Testing the type of an if condition (line 197)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 4), True_190110)
    # SSA begins for while statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 199):
    
    # Assigning a Subscript to a Name (line 199):
    
    # Obtaining the type of the subscript
    int_190111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'int')
    
    # Call to unravel_index(...): (line 199)
    # Processing the call arguments (line 199)
    
    # Call to argmax(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'covered_C' (line 199)
    covered_C_190116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 46), 'covered_C', False)
    # Processing the call keyword arguments (line 199)
    kwargs_190117 = {}
    # Getting the type of 'np' (line 199)
    np_190114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 36), 'np', False)
    # Obtaining the member 'argmax' of a type (line 199)
    argmax_190115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 36), np_190114, 'argmax')
    # Calling argmax(args, kwargs) (line 199)
    argmax_call_result_190118 = invoke(stypy.reporting.localization.Localization(__file__, 199, 36), argmax_190115, *[covered_C_190116], **kwargs_190117)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 199)
    tuple_190119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 199)
    # Adding element type (line 199)
    # Getting the type of 'n' (line 199)
    n_190120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 59), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 59), tuple_190119, n_190120)
    # Adding element type (line 199)
    # Getting the type of 'm' (line 199)
    m_190121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 62), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 59), tuple_190119, m_190121)
    
    # Processing the call keyword arguments (line 199)
    kwargs_190122 = {}
    # Getting the type of 'np' (line 199)
    np_190112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'np', False)
    # Obtaining the member 'unravel_index' of a type (line 199)
    unravel_index_190113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 19), np_190112, 'unravel_index')
    # Calling unravel_index(args, kwargs) (line 199)
    unravel_index_call_result_190123 = invoke(stypy.reporting.localization.Localization(__file__, 199, 19), unravel_index_190113, *[argmax_call_result_190118, tuple_190119], **kwargs_190122)
    
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___190124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), unravel_index_call_result_190123, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_190125 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), getitem___190124, int_190111)
    
    # Assigning a type to the variable 'tuple_var_assignment_189769' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'tuple_var_assignment_189769', subscript_call_result_190125)
    
    # Assigning a Subscript to a Name (line 199):
    
    # Obtaining the type of the subscript
    int_190126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'int')
    
    # Call to unravel_index(...): (line 199)
    # Processing the call arguments (line 199)
    
    # Call to argmax(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'covered_C' (line 199)
    covered_C_190131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 46), 'covered_C', False)
    # Processing the call keyword arguments (line 199)
    kwargs_190132 = {}
    # Getting the type of 'np' (line 199)
    np_190129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 36), 'np', False)
    # Obtaining the member 'argmax' of a type (line 199)
    argmax_190130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 36), np_190129, 'argmax')
    # Calling argmax(args, kwargs) (line 199)
    argmax_call_result_190133 = invoke(stypy.reporting.localization.Localization(__file__, 199, 36), argmax_190130, *[covered_C_190131], **kwargs_190132)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 199)
    tuple_190134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 199)
    # Adding element type (line 199)
    # Getting the type of 'n' (line 199)
    n_190135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 59), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 59), tuple_190134, n_190135)
    # Adding element type (line 199)
    # Getting the type of 'm' (line 199)
    m_190136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 62), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 59), tuple_190134, m_190136)
    
    # Processing the call keyword arguments (line 199)
    kwargs_190137 = {}
    # Getting the type of 'np' (line 199)
    np_190127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'np', False)
    # Obtaining the member 'unravel_index' of a type (line 199)
    unravel_index_190128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 19), np_190127, 'unravel_index')
    # Calling unravel_index(args, kwargs) (line 199)
    unravel_index_call_result_190138 = invoke(stypy.reporting.localization.Localization(__file__, 199, 19), unravel_index_190128, *[argmax_call_result_190133, tuple_190134], **kwargs_190137)
    
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___190139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), unravel_index_call_result_190138, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_190140 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), getitem___190139, int_190126)
    
    # Assigning a type to the variable 'tuple_var_assignment_189770' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'tuple_var_assignment_189770', subscript_call_result_190140)
    
    # Assigning a Name to a Name (line 199):
    # Getting the type of 'tuple_var_assignment_189769' (line 199)
    tuple_var_assignment_189769_190141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'tuple_var_assignment_189769')
    # Assigning a type to the variable 'row' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'row', tuple_var_assignment_189769_190141)
    
    # Assigning a Name to a Name (line 199):
    # Getting the type of 'tuple_var_assignment_189770' (line 199)
    tuple_var_assignment_189770_190142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'tuple_var_assignment_189770')
    # Assigning a type to the variable 'col' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 13), 'col', tuple_var_assignment_189770_190142)
    
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 200)
    tuple_190143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 200)
    # Adding element type (line 200)
    # Getting the type of 'row' (line 200)
    row_190144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'row')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 21), tuple_190143, row_190144)
    # Adding element type (line 200)
    # Getting the type of 'col' (line 200)
    col_190145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'col')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 21), tuple_190143, col_190145)
    
    # Getting the type of 'covered_C' (line 200)
    covered_C_190146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'covered_C')
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___190147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 11), covered_C_190146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_190148 = invoke(stypy.reporting.localization.Localization(__file__, 200, 11), getitem___190147, tuple_190143)
    
    int_190149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 34), 'int')
    # Applying the binary operator '==' (line 200)
    result_eq_190150 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 11), '==', subscript_call_result_190148, int_190149)
    
    # Testing the type of an if condition (line 200)
    if_condition_190151 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 8), result_eq_190150)
    # Assigning a type to the variable 'if_condition_190151' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'if_condition_190151', if_condition_190151)
    # SSA begins for if statement (line 200)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_step6' (line 201)
    _step6_190152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), '_step6')
    # Assigning a type to the variable 'stypy_return_type' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'stypy_return_type', _step6_190152)
    # SSA branch for the else part of an if statement (line 200)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Subscript (line 203):
    
    # Assigning a Num to a Subscript (line 203):
    int_190153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 37), 'int')
    # Getting the type of 'state' (line 203)
    state_190154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'state')
    # Obtaining the member 'marked' of a type (line 203)
    marked_190155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 12), state_190154, 'marked')
    
    # Obtaining an instance of the builtin type 'tuple' (line 203)
    tuple_190156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 203)
    # Adding element type (line 203)
    # Getting the type of 'row' (line 203)
    row_190157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'row')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 25), tuple_190156, row_190157)
    # Adding element type (line 203)
    # Getting the type of 'col' (line 203)
    col_190158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 30), 'col')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 25), tuple_190156, col_190158)
    
    # Storing an element on a container (line 203)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 12), marked_190155, (tuple_190156, int_190153))
    
    # Assigning a Call to a Name (line 205):
    
    # Assigning a Call to a Name (line 205):
    
    # Call to argmax(...): (line 205)
    # Processing the call arguments (line 205)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'row' (line 205)
    row_190161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 46), 'row', False)
    # Getting the type of 'state' (line 205)
    state_190162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 33), 'state', False)
    # Obtaining the member 'marked' of a type (line 205)
    marked_190163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 33), state_190162, 'marked')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___190164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 33), marked_190163, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_190165 = invoke(stypy.reporting.localization.Localization(__file__, 205, 33), getitem___190164, row_190161)
    
    int_190166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 54), 'int')
    # Applying the binary operator '==' (line 205)
    result_eq_190167 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 33), '==', subscript_call_result_190165, int_190166)
    
    # Processing the call keyword arguments (line 205)
    kwargs_190168 = {}
    # Getting the type of 'np' (line 205)
    np_190159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 23), 'np', False)
    # Obtaining the member 'argmax' of a type (line 205)
    argmax_190160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 23), np_190159, 'argmax')
    # Calling argmax(args, kwargs) (line 205)
    argmax_call_result_190169 = invoke(stypy.reporting.localization.Localization(__file__, 205, 23), argmax_190160, *[result_eq_190167], **kwargs_190168)
    
    # Assigning a type to the variable 'star_col' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'star_col', argmax_call_result_190169)
    
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 206)
    tuple_190170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 206)
    # Adding element type (line 206)
    # Getting the type of 'row' (line 206)
    row_190171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 28), 'row')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 28), tuple_190170, row_190171)
    # Adding element type (line 206)
    # Getting the type of 'star_col' (line 206)
    star_col_190172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 33), 'star_col')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 28), tuple_190170, star_col_190172)
    
    # Getting the type of 'state' (line 206)
    state_190173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), 'state')
    # Obtaining the member 'marked' of a type (line 206)
    marked_190174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 15), state_190173, 'marked')
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___190175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 15), marked_190174, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_190176 = invoke(stypy.reporting.localization.Localization(__file__, 206, 15), getitem___190175, tuple_190170)
    
    int_190177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 46), 'int')
    # Applying the binary operator '!=' (line 206)
    result_ne_190178 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 15), '!=', subscript_call_result_190176, int_190177)
    
    # Testing the type of an if condition (line 206)
    if_condition_190179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 12), result_ne_190178)
    # Assigning a type to the variable 'if_condition_190179' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'if_condition_190179', if_condition_190179)
    # SSA begins for if statement (line 206)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Attribute (line 208):
    
    # Assigning a Name to a Attribute (line 208):
    # Getting the type of 'row' (line 208)
    row_190180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 29), 'row')
    # Getting the type of 'state' (line 208)
    state_190181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'state')
    # Setting the type of the member 'Z0_r' of a type (line 208)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 16), state_190181, 'Z0_r', row_190180)
    
    # Assigning a Name to a Attribute (line 209):
    
    # Assigning a Name to a Attribute (line 209):
    # Getting the type of 'col' (line 209)
    col_190182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 29), 'col')
    # Getting the type of 'state' (line 209)
    state_190183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'state')
    # Setting the type of the member 'Z0_c' of a type (line 209)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 16), state_190183, 'Z0_c', col_190182)
    # Getting the type of '_step5' (line 210)
    _step5_190184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 23), '_step5')
    # Assigning a type to the variable 'stypy_return_type' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'stypy_return_type', _step5_190184)
    # SSA branch for the else part of an if statement (line 206)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 212):
    
    # Assigning a Name to a Name (line 212):
    # Getting the type of 'star_col' (line 212)
    star_col_190185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 22), 'star_col')
    # Assigning a type to the variable 'col' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'col', star_col_190185)
    
    # Assigning a Name to a Subscript (line 213):
    
    # Assigning a Name to a Subscript (line 213):
    # Getting the type of 'False' (line 213)
    False_190186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 43), 'False')
    # Getting the type of 'state' (line 213)
    state_190187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'state')
    # Obtaining the member 'row_uncovered' of a type (line 213)
    row_uncovered_190188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), state_190187, 'row_uncovered')
    # Getting the type of 'row' (line 213)
    row_190189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 36), 'row')
    # Storing an element on a container (line 213)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 16), row_uncovered_190188, (row_190189, False_190186))
    
    # Assigning a Name to a Subscript (line 214):
    
    # Assigning a Name to a Subscript (line 214):
    # Getting the type of 'True' (line 214)
    True_190190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 43), 'True')
    # Getting the type of 'state' (line 214)
    state_190191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'state')
    # Obtaining the member 'col_uncovered' of a type (line 214)
    col_uncovered_190192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 16), state_190191, 'col_uncovered')
    # Getting the type of 'col' (line 214)
    col_190193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 36), 'col')
    # Storing an element on a container (line 214)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 16), col_uncovered_190192, (col_190193, True_190190))
    
    # Assigning a BinOp to a Subscript (line 215):
    
    # Assigning a BinOp to a Subscript (line 215):
    
    # Obtaining the type of the subscript
    slice_190194 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 215, 36), None, None, None)
    # Getting the type of 'col' (line 215)
    col_190195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 41), 'col')
    # Getting the type of 'C' (line 215)
    C_190196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 36), 'C')
    # Obtaining the member '__getitem__' of a type (line 215)
    getitem___190197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 36), C_190196, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 215)
    subscript_call_result_190198 = invoke(stypy.reporting.localization.Localization(__file__, 215, 36), getitem___190197, (slice_190194, col_190195))
    
    
    # Call to asarray(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'state' (line 216)
    state_190201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 31), 'state', False)
    # Obtaining the member 'row_uncovered' of a type (line 216)
    row_uncovered_190202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 31), state_190201, 'row_uncovered')
    # Processing the call keyword arguments (line 216)
    # Getting the type of 'int' (line 216)
    int_190203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 58), 'int', False)
    keyword_190204 = int_190203
    kwargs_190205 = {'dtype': keyword_190204}
    # Getting the type of 'np' (line 216)
    np_190199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 20), 'np', False)
    # Obtaining the member 'asarray' of a type (line 216)
    asarray_190200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 20), np_190199, 'asarray')
    # Calling asarray(args, kwargs) (line 216)
    asarray_call_result_190206 = invoke(stypy.reporting.localization.Localization(__file__, 216, 20), asarray_190200, *[row_uncovered_190202], **kwargs_190205)
    
    # Applying the binary operator '*' (line 215)
    result_mul_190207 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 36), '*', subscript_call_result_190198, asarray_call_result_190206)
    
    # Getting the type of 'covered_C' (line 215)
    covered_C_190208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'covered_C')
    slice_190209 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 215, 16), None, None, None)
    # Getting the type of 'col' (line 215)
    col_190210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 29), 'col')
    # Storing an element on a container (line 215)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 16), covered_C_190208, ((slice_190209, col_190210), result_mul_190207))
    
    # Assigning a Num to a Subscript (line 217):
    
    # Assigning a Num to a Subscript (line 217):
    int_190211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 33), 'int')
    # Getting the type of 'covered_C' (line 217)
    covered_C_190212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'covered_C')
    # Getting the type of 'row' (line 217)
    row_190213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 26), 'row')
    # Storing an element on a container (line 217)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 16), covered_C_190212, (row_190213, int_190211))
    # SSA join for if statement (line 206)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 200)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 197)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_step4(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_step4' in the type store
    # Getting the type of 'stypy_return_type' (line 182)
    stypy_return_type_190214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190214)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_step4'
    return stypy_return_type_190214

# Assigning a type to the variable '_step4' (line 182)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 0), '_step4', _step4)

@norecursion
def _step5(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_step5'
    module_type_store = module_type_store.open_function_context('_step5', 220, 0, False)
    
    # Passed parameters checking function
    _step5.stypy_localization = localization
    _step5.stypy_type_of_self = None
    _step5.stypy_type_store = module_type_store
    _step5.stypy_function_name = '_step5'
    _step5.stypy_param_names_list = ['state']
    _step5.stypy_varargs_param_name = None
    _step5.stypy_kwargs_param_name = None
    _step5.stypy_call_defaults = defaults
    _step5.stypy_call_varargs = varargs
    _step5.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_step5', ['state'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_step5', localization, ['state'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_step5(...)' code ##################

    str_190215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, (-1)), 'str', '\n    Construct a series of alternating primed and starred zeros as follows.\n    Let Z0 represent the uncovered primed zero found in Step 4.\n    Let Z1 denote the starred zero in the column of Z0 (if any).\n    Let Z2 denote the primed zero in the row of Z1 (there will always be one).\n    Continue until the series terminates at a primed zero that has no starred\n    zero in its column. Unstar each starred zero of the series, star each\n    primed zero of the series, erase all primes and uncover every line in the\n    matrix. Return to Step 3\n    ')
    
    # Assigning a Num to a Name (line 231):
    
    # Assigning a Num to a Name (line 231):
    int_190216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'int')
    # Assigning a type to the variable 'count' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'count', int_190216)
    
    # Assigning a Attribute to a Name (line 232):
    
    # Assigning a Attribute to a Name (line 232):
    # Getting the type of 'state' (line 232)
    state_190217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'state')
    # Obtaining the member 'path' of a type (line 232)
    path_190218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 11), state_190217, 'path')
    # Assigning a type to the variable 'path' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'path', path_190218)
    
    # Assigning a Attribute to a Subscript (line 233):
    
    # Assigning a Attribute to a Subscript (line 233):
    # Getting the type of 'state' (line 233)
    state_190219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 21), 'state')
    # Obtaining the member 'Z0_r' of a type (line 233)
    Z0_r_190220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 21), state_190219, 'Z0_r')
    # Getting the type of 'path' (line 233)
    path_190221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'path')
    
    # Obtaining an instance of the builtin type 'tuple' (line 233)
    tuple_190222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 233)
    # Adding element type (line 233)
    # Getting the type of 'count' (line 233)
    count_190223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 9), 'count')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 9), tuple_190222, count_190223)
    # Adding element type (line 233)
    int_190224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 9), tuple_190222, int_190224)
    
    # Storing an element on a container (line 233)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 4), path_190221, (tuple_190222, Z0_r_190220))
    
    # Assigning a Attribute to a Subscript (line 234):
    
    # Assigning a Attribute to a Subscript (line 234):
    # Getting the type of 'state' (line 234)
    state_190225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 21), 'state')
    # Obtaining the member 'Z0_c' of a type (line 234)
    Z0_c_190226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 21), state_190225, 'Z0_c')
    # Getting the type of 'path' (line 234)
    path_190227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'path')
    
    # Obtaining an instance of the builtin type 'tuple' (line 234)
    tuple_190228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 234)
    # Adding element type (line 234)
    # Getting the type of 'count' (line 234)
    count_190229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 9), 'count')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 9), tuple_190228, count_190229)
    # Adding element type (line 234)
    int_190230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 9), tuple_190228, int_190230)
    
    # Storing an element on a container (line 234)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 4), path_190227, (tuple_190228, Z0_c_190226))
    
    # Getting the type of 'True' (line 236)
    True_190231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 10), 'True')
    # Testing the type of an if condition (line 236)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 4), True_190231)
    # SSA begins for while statement (line 236)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 239):
    
    # Assigning a Call to a Name (line 239):
    
    # Call to argmax(...): (line 239)
    # Processing the call arguments (line 239)
    
    
    # Obtaining the type of the subscript
    slice_190234 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 239, 24), None, None, None)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 239)
    tuple_190235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 239)
    # Adding element type (line 239)
    # Getting the type of 'count' (line 239)
    count_190236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 45), 'count', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 45), tuple_190235, count_190236)
    # Adding element type (line 239)
    int_190237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 45), tuple_190235, int_190237)
    
    # Getting the type of 'path' (line 239)
    path_190238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 40), 'path', False)
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___190239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 40), path_190238, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_190240 = invoke(stypy.reporting.localization.Localization(__file__, 239, 40), getitem___190239, tuple_190235)
    
    # Getting the type of 'state' (line 239)
    state_190241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 24), 'state', False)
    # Obtaining the member 'marked' of a type (line 239)
    marked_190242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 24), state_190241, 'marked')
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___190243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 24), marked_190242, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_190244 = invoke(stypy.reporting.localization.Localization(__file__, 239, 24), getitem___190243, (slice_190234, subscript_call_result_190240))
    
    int_190245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 59), 'int')
    # Applying the binary operator '==' (line 239)
    result_eq_190246 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 24), '==', subscript_call_result_190244, int_190245)
    
    # Processing the call keyword arguments (line 239)
    kwargs_190247 = {}
    # Getting the type of 'np' (line 239)
    np_190232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 14), 'np', False)
    # Obtaining the member 'argmax' of a type (line 239)
    argmax_190233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 14), np_190232, 'argmax')
    # Calling argmax(args, kwargs) (line 239)
    argmax_call_result_190248 = invoke(stypy.reporting.localization.Localization(__file__, 239, 14), argmax_190233, *[result_eq_190246], **kwargs_190247)
    
    # Assigning a type to the variable 'row' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'row', argmax_call_result_190248)
    
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 240)
    tuple_190249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 240)
    # Adding element type (line 240)
    # Getting the type of 'row' (line 240)
    row_190250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 24), 'row')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 24), tuple_190249, row_190250)
    # Adding element type (line 240)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 240)
    tuple_190251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 240)
    # Adding element type (line 240)
    # Getting the type of 'count' (line 240)
    count_190252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 34), 'count')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 34), tuple_190251, count_190252)
    # Adding element type (line 240)
    int_190253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 34), tuple_190251, int_190253)
    
    # Getting the type of 'path' (line 240)
    path_190254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 29), 'path')
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___190255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 29), path_190254, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_190256 = invoke(stypy.reporting.localization.Localization(__file__, 240, 29), getitem___190255, tuple_190251)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 24), tuple_190249, subscript_call_result_190256)
    
    # Getting the type of 'state' (line 240)
    state_190257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'state')
    # Obtaining the member 'marked' of a type (line 240)
    marked_190258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 11), state_190257, 'marked')
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___190259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 11), marked_190258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_190260 = invoke(stypy.reporting.localization.Localization(__file__, 240, 11), getitem___190259, tuple_190249)
    
    int_190261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 48), 'int')
    # Applying the binary operator '!=' (line 240)
    result_ne_190262 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 11), '!=', subscript_call_result_190260, int_190261)
    
    # Testing the type of an if condition (line 240)
    if_condition_190263 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), result_ne_190262)
    # Assigning a type to the variable 'if_condition_190263' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_190263', if_condition_190263)
    # SSA begins for if statement (line 240)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA branch for the else part of an if statement (line 240)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'count' (line 244)
    count_190264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'count')
    int_190265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 21), 'int')
    # Applying the binary operator '+=' (line 244)
    result_iadd_190266 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 12), '+=', count_190264, int_190265)
    # Assigning a type to the variable 'count' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'count', result_iadd_190266)
    
    
    # Assigning a Name to a Subscript (line 245):
    
    # Assigning a Name to a Subscript (line 245):
    # Getting the type of 'row' (line 245)
    row_190267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 29), 'row')
    # Getting the type of 'path' (line 245)
    path_190268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'path')
    
    # Obtaining an instance of the builtin type 'tuple' (line 245)
    tuple_190269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 245)
    # Adding element type (line 245)
    # Getting the type of 'count' (line 245)
    count_190270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 17), 'count')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 17), tuple_190269, count_190270)
    # Adding element type (line 245)
    int_190271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 17), tuple_190269, int_190271)
    
    # Storing an element on a container (line 245)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 12), path_190268, (tuple_190269, row_190267))
    
    # Assigning a Subscript to a Subscript (line 246):
    
    # Assigning a Subscript to a Subscript (line 246):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 246)
    tuple_190272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 246)
    # Adding element type (line 246)
    # Getting the type of 'count' (line 246)
    count_190273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 34), 'count')
    int_190274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 42), 'int')
    # Applying the binary operator '-' (line 246)
    result_sub_190275 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 34), '-', count_190273, int_190274)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 34), tuple_190272, result_sub_190275)
    # Adding element type (line 246)
    int_190276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 34), tuple_190272, int_190276)
    
    # Getting the type of 'path' (line 246)
    path_190277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 29), 'path')
    # Obtaining the member '__getitem__' of a type (line 246)
    getitem___190278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 29), path_190277, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 246)
    subscript_call_result_190279 = invoke(stypy.reporting.localization.Localization(__file__, 246, 29), getitem___190278, tuple_190272)
    
    # Getting the type of 'path' (line 246)
    path_190280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'path')
    
    # Obtaining an instance of the builtin type 'tuple' (line 246)
    tuple_190281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 246)
    # Adding element type (line 246)
    # Getting the type of 'count' (line 246)
    count_190282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 17), 'count')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 17), tuple_190281, count_190282)
    # Adding element type (line 246)
    int_190283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 17), tuple_190281, int_190283)
    
    # Storing an element on a container (line 246)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 12), path_190280, (tuple_190281, subscript_call_result_190279))
    # SSA join for if statement (line 240)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 250):
    
    # Assigning a Call to a Name (line 250):
    
    # Call to argmax(...): (line 250)
    # Processing the call arguments (line 250)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 250)
    tuple_190286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 250)
    # Adding element type (line 250)
    # Getting the type of 'count' (line 250)
    count_190287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 42), 'count', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 42), tuple_190286, count_190287)
    # Adding element type (line 250)
    int_190288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 42), tuple_190286, int_190288)
    
    # Getting the type of 'path' (line 250)
    path_190289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 37), 'path', False)
    # Obtaining the member '__getitem__' of a type (line 250)
    getitem___190290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 37), path_190289, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 250)
    subscript_call_result_190291 = invoke(stypy.reporting.localization.Localization(__file__, 250, 37), getitem___190290, tuple_190286)
    
    # Getting the type of 'state' (line 250)
    state_190292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 24), 'state', False)
    # Obtaining the member 'marked' of a type (line 250)
    marked_190293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 24), state_190292, 'marked')
    # Obtaining the member '__getitem__' of a type (line 250)
    getitem___190294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 24), marked_190293, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 250)
    subscript_call_result_190295 = invoke(stypy.reporting.localization.Localization(__file__, 250, 24), getitem___190294, subscript_call_result_190291)
    
    int_190296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 56), 'int')
    # Applying the binary operator '==' (line 250)
    result_eq_190297 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 24), '==', subscript_call_result_190295, int_190296)
    
    # Processing the call keyword arguments (line 250)
    kwargs_190298 = {}
    # Getting the type of 'np' (line 250)
    np_190284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 14), 'np', False)
    # Obtaining the member 'argmax' of a type (line 250)
    argmax_190285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 14), np_190284, 'argmax')
    # Calling argmax(args, kwargs) (line 250)
    argmax_call_result_190299 = invoke(stypy.reporting.localization.Localization(__file__, 250, 14), argmax_190285, *[result_eq_190297], **kwargs_190298)
    
    # Assigning a type to the variable 'col' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'col', argmax_call_result_190299)
    
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 251)
    tuple_190300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 251)
    # Adding element type (line 251)
    # Getting the type of 'row' (line 251)
    row_190301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 24), 'row')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 24), tuple_190300, row_190301)
    # Adding element type (line 251)
    # Getting the type of 'col' (line 251)
    col_190302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 29), 'col')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 24), tuple_190300, col_190302)
    
    # Getting the type of 'state' (line 251)
    state_190303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'state')
    # Obtaining the member 'marked' of a type (line 251)
    marked_190304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 11), state_190303, 'marked')
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___190305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 11), marked_190304, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_190306 = invoke(stypy.reporting.localization.Localization(__file__, 251, 11), getitem___190305, tuple_190300)
    
    int_190307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 37), 'int')
    # Applying the binary operator '!=' (line 251)
    result_ne_190308 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 11), '!=', subscript_call_result_190306, int_190307)
    
    # Testing the type of an if condition (line 251)
    if_condition_190309 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 8), result_ne_190308)
    # Assigning a type to the variable 'if_condition_190309' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'if_condition_190309', if_condition_190309)
    # SSA begins for if statement (line 251)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 252):
    
    # Assigning a Num to a Name (line 252):
    int_190310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 18), 'int')
    # Assigning a type to the variable 'col' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'col', int_190310)
    # SSA join for if statement (line 251)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'count' (line 253)
    count_190311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'count')
    int_190312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 17), 'int')
    # Applying the binary operator '+=' (line 253)
    result_iadd_190313 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 8), '+=', count_190311, int_190312)
    # Assigning a type to the variable 'count' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'count', result_iadd_190313)
    
    
    # Assigning a Subscript to a Subscript (line 254):
    
    # Assigning a Subscript to a Subscript (line 254):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 254)
    tuple_190314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 254)
    # Adding element type (line 254)
    # Getting the type of 'count' (line 254)
    count_190315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 30), 'count')
    int_190316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 38), 'int')
    # Applying the binary operator '-' (line 254)
    result_sub_190317 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 30), '-', count_190315, int_190316)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 30), tuple_190314, result_sub_190317)
    # Adding element type (line 254)
    int_190318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 30), tuple_190314, int_190318)
    
    # Getting the type of 'path' (line 254)
    path_190319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 25), 'path')
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___190320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 25), path_190319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_190321 = invoke(stypy.reporting.localization.Localization(__file__, 254, 25), getitem___190320, tuple_190314)
    
    # Getting the type of 'path' (line 254)
    path_190322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'path')
    
    # Obtaining an instance of the builtin type 'tuple' (line 254)
    tuple_190323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 254)
    # Adding element type (line 254)
    # Getting the type of 'count' (line 254)
    count_190324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 13), 'count')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 13), tuple_190323, count_190324)
    # Adding element type (line 254)
    int_190325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 13), tuple_190323, int_190325)
    
    # Storing an element on a container (line 254)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 8), path_190322, (tuple_190323, subscript_call_result_190321))
    
    # Assigning a Name to a Subscript (line 255):
    
    # Assigning a Name to a Subscript (line 255):
    # Getting the type of 'col' (line 255)
    col_190326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 25), 'col')
    # Getting the type of 'path' (line 255)
    path_190327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'path')
    
    # Obtaining an instance of the builtin type 'tuple' (line 255)
    tuple_190328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 255)
    # Adding element type (line 255)
    # Getting the type of 'count' (line 255)
    count_190329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 13), 'count')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 13), tuple_190328, count_190329)
    # Adding element type (line 255)
    int_190330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 13), tuple_190328, int_190330)
    
    # Storing an element on a container (line 255)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 8), path_190327, (tuple_190328, col_190326))
    # SSA join for while statement (line 236)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 258)
    # Processing the call arguments (line 258)
    # Getting the type of 'count' (line 258)
    count_190332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 19), 'count', False)
    int_190333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 27), 'int')
    # Applying the binary operator '+' (line 258)
    result_add_190334 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 19), '+', count_190332, int_190333)
    
    # Processing the call keyword arguments (line 258)
    kwargs_190335 = {}
    # Getting the type of 'range' (line 258)
    range_190331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 13), 'range', False)
    # Calling range(args, kwargs) (line 258)
    range_call_result_190336 = invoke(stypy.reporting.localization.Localization(__file__, 258, 13), range_190331, *[result_add_190334], **kwargs_190335)
    
    # Testing the type of a for loop iterable (line 258)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 258, 4), range_call_result_190336)
    # Getting the type of the for loop variable (line 258)
    for_loop_var_190337 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 258, 4), range_call_result_190336)
    # Assigning a type to the variable 'i' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'i', for_loop_var_190337)
    # SSA begins for a for statement (line 258)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 259)
    tuple_190338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 259)
    # Adding element type (line 259)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 259)
    tuple_190339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 259)
    # Adding element type (line 259)
    # Getting the type of 'i' (line 259)
    i_190340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 29), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 29), tuple_190339, i_190340)
    # Adding element type (line 259)
    int_190341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 29), tuple_190339, int_190341)
    
    # Getting the type of 'path' (line 259)
    path_190342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 'path')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___190343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 24), path_190342, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_190344 = invoke(stypy.reporting.localization.Localization(__file__, 259, 24), getitem___190343, tuple_190339)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 24), tuple_190338, subscript_call_result_190344)
    # Adding element type (line 259)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 259)
    tuple_190345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 259)
    # Adding element type (line 259)
    # Getting the type of 'i' (line 259)
    i_190346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 41), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 41), tuple_190345, i_190346)
    # Adding element type (line 259)
    int_190347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 41), tuple_190345, int_190347)
    
    # Getting the type of 'path' (line 259)
    path_190348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 36), 'path')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___190349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 36), path_190348, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_190350 = invoke(stypy.reporting.localization.Localization(__file__, 259, 36), getitem___190349, tuple_190345)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 24), tuple_190338, subscript_call_result_190350)
    
    # Getting the type of 'state' (line 259)
    state_190351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'state')
    # Obtaining the member 'marked' of a type (line 259)
    marked_190352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 11), state_190351, 'marked')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___190353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 11), marked_190352, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_190354 = invoke(stypy.reporting.localization.Localization(__file__, 259, 11), getitem___190353, tuple_190338)
    
    int_190355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 51), 'int')
    # Applying the binary operator '==' (line 259)
    result_eq_190356 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 11), '==', subscript_call_result_190354, int_190355)
    
    # Testing the type of an if condition (line 259)
    if_condition_190357 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 8), result_eq_190356)
    # Assigning a type to the variable 'if_condition_190357' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'if_condition_190357', if_condition_190357)
    # SSA begins for if statement (line 259)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 260):
    
    # Assigning a Num to a Subscript (line 260):
    int_190358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 51), 'int')
    # Getting the type of 'state' (line 260)
    state_190359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'state')
    # Obtaining the member 'marked' of a type (line 260)
    marked_190360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), state_190359, 'marked')
    
    # Obtaining an instance of the builtin type 'tuple' (line 260)
    tuple_190361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 260)
    # Adding element type (line 260)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 260)
    tuple_190362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 260)
    # Adding element type (line 260)
    # Getting the type of 'i' (line 260)
    i_190363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 30), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 30), tuple_190362, i_190363)
    # Adding element type (line 260)
    int_190364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 30), tuple_190362, int_190364)
    
    # Getting the type of 'path' (line 260)
    path_190365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 25), 'path')
    # Obtaining the member '__getitem__' of a type (line 260)
    getitem___190366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 25), path_190365, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 260)
    subscript_call_result_190367 = invoke(stypy.reporting.localization.Localization(__file__, 260, 25), getitem___190366, tuple_190362)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 25), tuple_190361, subscript_call_result_190367)
    # Adding element type (line 260)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 260)
    tuple_190368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 260)
    # Adding element type (line 260)
    # Getting the type of 'i' (line 260)
    i_190369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 42), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 42), tuple_190368, i_190369)
    # Adding element type (line 260)
    int_190370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 42), tuple_190368, int_190370)
    
    # Getting the type of 'path' (line 260)
    path_190371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 37), 'path')
    # Obtaining the member '__getitem__' of a type (line 260)
    getitem___190372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 37), path_190371, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 260)
    subscript_call_result_190373 = invoke(stypy.reporting.localization.Localization(__file__, 260, 37), getitem___190372, tuple_190368)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 25), tuple_190361, subscript_call_result_190373)
    
    # Storing an element on a container (line 260)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 12), marked_190360, (tuple_190361, int_190358))
    # SSA branch for the else part of an if statement (line 259)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Subscript (line 262):
    
    # Assigning a Num to a Subscript (line 262):
    int_190374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 51), 'int')
    # Getting the type of 'state' (line 262)
    state_190375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'state')
    # Obtaining the member 'marked' of a type (line 262)
    marked_190376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 12), state_190375, 'marked')
    
    # Obtaining an instance of the builtin type 'tuple' (line 262)
    tuple_190377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 262)
    # Adding element type (line 262)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 262)
    tuple_190378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 262)
    # Adding element type (line 262)
    # Getting the type of 'i' (line 262)
    i_190379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 30), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 30), tuple_190378, i_190379)
    # Adding element type (line 262)
    int_190380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 30), tuple_190378, int_190380)
    
    # Getting the type of 'path' (line 262)
    path_190381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 25), 'path')
    # Obtaining the member '__getitem__' of a type (line 262)
    getitem___190382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 25), path_190381, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 262)
    subscript_call_result_190383 = invoke(stypy.reporting.localization.Localization(__file__, 262, 25), getitem___190382, tuple_190378)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 25), tuple_190377, subscript_call_result_190383)
    # Adding element type (line 262)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 262)
    tuple_190384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 262)
    # Adding element type (line 262)
    # Getting the type of 'i' (line 262)
    i_190385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 42), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 42), tuple_190384, i_190385)
    # Adding element type (line 262)
    int_190386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 42), tuple_190384, int_190386)
    
    # Getting the type of 'path' (line 262)
    path_190387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 37), 'path')
    # Obtaining the member '__getitem__' of a type (line 262)
    getitem___190388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 37), path_190387, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 262)
    subscript_call_result_190389 = invoke(stypy.reporting.localization.Localization(__file__, 262, 37), getitem___190388, tuple_190384)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 25), tuple_190377, subscript_call_result_190389)
    
    # Storing an element on a container (line 262)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 12), marked_190376, (tuple_190377, int_190374))
    # SSA join for if statement (line 259)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _clear_covers(...): (line 264)
    # Processing the call keyword arguments (line 264)
    kwargs_190392 = {}
    # Getting the type of 'state' (line 264)
    state_190390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'state', False)
    # Obtaining the member '_clear_covers' of a type (line 264)
    _clear_covers_190391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 4), state_190390, '_clear_covers')
    # Calling _clear_covers(args, kwargs) (line 264)
    _clear_covers_call_result_190393 = invoke(stypy.reporting.localization.Localization(__file__, 264, 4), _clear_covers_190391, *[], **kwargs_190392)
    
    
    # Assigning a Num to a Subscript (line 266):
    
    # Assigning a Num to a Subscript (line 266):
    int_190394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 38), 'int')
    # Getting the type of 'state' (line 266)
    state_190395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'state')
    # Obtaining the member 'marked' of a type (line 266)
    marked_190396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 4), state_190395, 'marked')
    
    # Getting the type of 'state' (line 266)
    state_190397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 17), 'state')
    # Obtaining the member 'marked' of a type (line 266)
    marked_190398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 17), state_190397, 'marked')
    int_190399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 33), 'int')
    # Applying the binary operator '==' (line 266)
    result_eq_190400 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 17), '==', marked_190398, int_190399)
    
    # Storing an element on a container (line 266)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 4), marked_190396, (result_eq_190400, int_190394))
    # Getting the type of '_step3' (line 267)
    _step3_190401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 11), '_step3')
    # Assigning a type to the variable 'stypy_return_type' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'stypy_return_type', _step3_190401)
    
    # ################# End of '_step5(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_step5' in the type store
    # Getting the type of 'stypy_return_type' (line 220)
    stypy_return_type_190402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190402)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_step5'
    return stypy_return_type_190402

# Assigning a type to the variable '_step5' (line 220)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), '_step5', _step5)

@norecursion
def _step6(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_step6'
    module_type_store = module_type_store.open_function_context('_step6', 270, 0, False)
    
    # Passed parameters checking function
    _step6.stypy_localization = localization
    _step6.stypy_type_of_self = None
    _step6.stypy_type_store = module_type_store
    _step6.stypy_function_name = '_step6'
    _step6.stypy_param_names_list = ['state']
    _step6.stypy_varargs_param_name = None
    _step6.stypy_kwargs_param_name = None
    _step6.stypy_call_defaults = defaults
    _step6.stypy_call_varargs = varargs
    _step6.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_step6', ['state'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_step6', localization, ['state'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_step6(...)' code ##################

    str_190403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, (-1)), 'str', '\n    Add the value found in Step 4 to every element of each covered row,\n    and subtract it from every element of each uncovered column.\n    Return to Step 4 without altering any stars, primes, or covered lines.\n    ')
    
    
    # Evaluating a boolean operation
    
    # Call to any(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'state' (line 277)
    state_190406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 14), 'state', False)
    # Obtaining the member 'row_uncovered' of a type (line 277)
    row_uncovered_190407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 14), state_190406, 'row_uncovered')
    # Processing the call keyword arguments (line 277)
    kwargs_190408 = {}
    # Getting the type of 'np' (line 277)
    np_190404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 277)
    any_190405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 7), np_190404, 'any')
    # Calling any(args, kwargs) (line 277)
    any_call_result_190409 = invoke(stypy.reporting.localization.Localization(__file__, 277, 7), any_190405, *[row_uncovered_190407], **kwargs_190408)
    
    
    # Call to any(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'state' (line 277)
    state_190412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 46), 'state', False)
    # Obtaining the member 'col_uncovered' of a type (line 277)
    col_uncovered_190413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 46), state_190412, 'col_uncovered')
    # Processing the call keyword arguments (line 277)
    kwargs_190414 = {}
    # Getting the type of 'np' (line 277)
    np_190410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 39), 'np', False)
    # Obtaining the member 'any' of a type (line 277)
    any_190411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 39), np_190410, 'any')
    # Calling any(args, kwargs) (line 277)
    any_call_result_190415 = invoke(stypy.reporting.localization.Localization(__file__, 277, 39), any_190411, *[col_uncovered_190413], **kwargs_190414)
    
    # Applying the binary operator 'and' (line 277)
    result_and_keyword_190416 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 7), 'and', any_call_result_190409, any_call_result_190415)
    
    # Testing the type of an if condition (line 277)
    if_condition_190417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 4), result_and_keyword_190416)
    # Assigning a type to the variable 'if_condition_190417' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'if_condition_190417', if_condition_190417)
    # SSA begins for if statement (line 277)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 278):
    
    # Assigning a Call to a Name (line 278):
    
    # Call to min(...): (line 278)
    # Processing the call arguments (line 278)
    
    # Obtaining the type of the subscript
    # Getting the type of 'state' (line 278)
    state_190420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 32), 'state', False)
    # Obtaining the member 'row_uncovered' of a type (line 278)
    row_uncovered_190421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 32), state_190420, 'row_uncovered')
    # Getting the type of 'state' (line 278)
    state_190422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 24), 'state', False)
    # Obtaining the member 'C' of a type (line 278)
    C_190423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 24), state_190422, 'C')
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___190424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 24), C_190423, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_190425 = invoke(stypy.reporting.localization.Localization(__file__, 278, 24), getitem___190424, row_uncovered_190421)
    
    # Processing the call keyword arguments (line 278)
    int_190426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 59), 'int')
    keyword_190427 = int_190426
    kwargs_190428 = {'axis': keyword_190427}
    # Getting the type of 'np' (line 278)
    np_190418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 17), 'np', False)
    # Obtaining the member 'min' of a type (line 278)
    min_190419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 17), np_190418, 'min')
    # Calling min(args, kwargs) (line 278)
    min_call_result_190429 = invoke(stypy.reporting.localization.Localization(__file__, 278, 17), min_190419, *[subscript_call_result_190425], **kwargs_190428)
    
    # Assigning a type to the variable 'minval' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'minval', min_call_result_190429)
    
    # Assigning a Call to a Name (line 279):
    
    # Assigning a Call to a Name (line 279):
    
    # Call to min(...): (line 279)
    # Processing the call arguments (line 279)
    
    # Obtaining the type of the subscript
    # Getting the type of 'state' (line 279)
    state_190432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 31), 'state', False)
    # Obtaining the member 'col_uncovered' of a type (line 279)
    col_uncovered_190433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 31), state_190432, 'col_uncovered')
    # Getting the type of 'minval' (line 279)
    minval_190434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 'minval', False)
    # Obtaining the member '__getitem__' of a type (line 279)
    getitem___190435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 24), minval_190434, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 279)
    subscript_call_result_190436 = invoke(stypy.reporting.localization.Localization(__file__, 279, 24), getitem___190435, col_uncovered_190433)
    
    # Processing the call keyword arguments (line 279)
    kwargs_190437 = {}
    # Getting the type of 'np' (line 279)
    np_190430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 17), 'np', False)
    # Obtaining the member 'min' of a type (line 279)
    min_190431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 17), np_190430, 'min')
    # Calling min(args, kwargs) (line 279)
    min_call_result_190438 = invoke(stypy.reporting.localization.Localization(__file__, 279, 17), min_190431, *[subscript_call_result_190436], **kwargs_190437)
    
    # Assigning a type to the variable 'minval' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'minval', min_call_result_190438)
    
    # Getting the type of 'state' (line 280)
    state_190439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'state')
    # Obtaining the member 'C' of a type (line 280)
    C_190440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), state_190439, 'C')
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'state' (line 280)
    state_190441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 17), 'state')
    # Obtaining the member 'row_uncovered' of a type (line 280)
    row_uncovered_190442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 17), state_190441, 'row_uncovered')
    # Applying the '~' unary operator (line 280)
    result_inv_190443 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 16), '~', row_uncovered_190442)
    
    # Getting the type of 'state' (line 280)
    state_190444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'state')
    # Obtaining the member 'C' of a type (line 280)
    C_190445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), state_190444, 'C')
    # Obtaining the member '__getitem__' of a type (line 280)
    getitem___190446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), C_190445, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 280)
    subscript_call_result_190447 = invoke(stypy.reporting.localization.Localization(__file__, 280, 8), getitem___190446, result_inv_190443)
    
    # Getting the type of 'minval' (line 280)
    minval_190448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 41), 'minval')
    # Applying the binary operator '+=' (line 280)
    result_iadd_190449 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 8), '+=', subscript_call_result_190447, minval_190448)
    # Getting the type of 'state' (line 280)
    state_190450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'state')
    # Obtaining the member 'C' of a type (line 280)
    C_190451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), state_190450, 'C')
    
    # Getting the type of 'state' (line 280)
    state_190452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 17), 'state')
    # Obtaining the member 'row_uncovered' of a type (line 280)
    row_uncovered_190453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 17), state_190452, 'row_uncovered')
    # Applying the '~' unary operator (line 280)
    result_inv_190454 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 16), '~', row_uncovered_190453)
    
    # Storing an element on a container (line 280)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 8), C_190451, (result_inv_190454, result_iadd_190449))
    
    
    # Getting the type of 'state' (line 281)
    state_190455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'state')
    # Obtaining the member 'C' of a type (line 281)
    C_190456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), state_190455, 'C')
    
    # Obtaining the type of the subscript
    slice_190457 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 281, 8), None, None, None)
    # Getting the type of 'state' (line 281)
    state_190458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 19), 'state')
    # Obtaining the member 'col_uncovered' of a type (line 281)
    col_uncovered_190459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 19), state_190458, 'col_uncovered')
    # Getting the type of 'state' (line 281)
    state_190460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'state')
    # Obtaining the member 'C' of a type (line 281)
    C_190461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), state_190460, 'C')
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___190462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), C_190461, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_190463 = invoke(stypy.reporting.localization.Localization(__file__, 281, 8), getitem___190462, (slice_190457, col_uncovered_190459))
    
    # Getting the type of 'minval' (line 281)
    minval_190464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 43), 'minval')
    # Applying the binary operator '-=' (line 281)
    result_isub_190465 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 8), '-=', subscript_call_result_190463, minval_190464)
    # Getting the type of 'state' (line 281)
    state_190466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'state')
    # Obtaining the member 'C' of a type (line 281)
    C_190467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), state_190466, 'C')
    slice_190468 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 281, 8), None, None, None)
    # Getting the type of 'state' (line 281)
    state_190469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 19), 'state')
    # Obtaining the member 'col_uncovered' of a type (line 281)
    col_uncovered_190470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 19), state_190469, 'col_uncovered')
    # Storing an element on a container (line 281)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 8), C_190467, ((slice_190468, col_uncovered_190470), result_isub_190465))
    
    # SSA join for if statement (line 277)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of '_step4' (line 282)
    _step4_190471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 11), '_step4')
    # Assigning a type to the variable 'stypy_return_type' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'stypy_return_type', _step4_190471)
    
    # ################# End of '_step6(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_step6' in the type store
    # Getting the type of 'stypy_return_type' (line 270)
    stypy_return_type_190472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190472)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_step6'
    return stypy_return_type_190472

# Assigning a type to the variable '_step6' (line 270)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), '_step6', _step6)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
