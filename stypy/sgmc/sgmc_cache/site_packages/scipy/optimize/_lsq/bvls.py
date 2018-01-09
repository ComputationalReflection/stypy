
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Bounded-Variable Least-Squares algorithm.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: import numpy as np
5: from numpy.linalg import norm, lstsq
6: from scipy.optimize import OptimizeResult
7: 
8: from .common import print_header_linear, print_iteration_linear
9: 
10: 
11: def compute_kkt_optimality(g, on_bound):
12:     '''Compute the maximum violation of KKT conditions.'''
13:     g_kkt = g * on_bound
14:     free_set = on_bound == 0
15:     g_kkt[free_set] = np.abs(g[free_set])
16:     return np.max(g_kkt)
17: 
18: 
19: def bvls(A, b, x_lsq, lb, ub, tol, max_iter, verbose):
20:     m, n = A.shape
21: 
22:     x = x_lsq.copy()
23:     on_bound = np.zeros(n)
24: 
25:     mask = x < lb
26:     x[mask] = lb[mask]
27:     on_bound[mask] = -1
28: 
29:     mask = x > ub
30:     x[mask] = ub[mask]
31:     on_bound[mask] = 1
32: 
33:     free_set = on_bound == 0
34:     active_set = ~free_set
35:     free_set, = np.where(free_set)
36: 
37:     r = A.dot(x) - b
38:     cost = 0.5 * np.dot(r, r)
39:     initial_cost = cost
40:     g = A.T.dot(r)
41: 
42:     cost_change = None
43:     step_norm = None
44:     iteration = 0
45: 
46:     if verbose == 2:
47:         print_header_linear()
48: 
49:     # This is the initialization loop. The requirement is that the
50:     # least-squares solution on free variables is feasible before BVLS starts.
51:     # One possible initialization is to set all variables to lower or upper
52:     # bounds, but many iterations may be required from this state later on.
53:     # The implemented ad-hoc procedure which intuitively should give a better
54:     # initial state: find the least-squares solution on current free variables,
55:     # if its feasible then stop, otherwise set violating variables to
56:     # corresponding bounds and continue on the reduced set of free variables.
57: 
58:     while free_set.size > 0:
59:         if verbose == 2:
60:             optimality = compute_kkt_optimality(g, on_bound)
61:             print_iteration_linear(iteration, cost, cost_change, step_norm,
62:                                    optimality)
63: 
64:         iteration += 1
65:         x_free_old = x[free_set].copy()
66: 
67:         A_free = A[:, free_set]
68:         b_free = b - A.dot(x * active_set)
69:         z = lstsq(A_free, b_free, rcond=-1)[0]
70: 
71:         lbv = z < lb[free_set]
72:         ubv = z > ub[free_set]
73:         v = lbv | ubv
74: 
75:         if np.any(lbv):
76:             ind = free_set[lbv]
77:             x[ind] = lb[ind]
78:             active_set[ind] = True
79:             on_bound[ind] = -1
80: 
81:         if np.any(ubv):
82:             ind = free_set[ubv]
83:             x[ind] = ub[ind]
84:             active_set[ind] = True
85:             on_bound[ind] = 1
86: 
87:         ind = free_set[~v]
88:         x[ind] = z[~v]
89: 
90:         r = A.dot(x) - b
91:         cost_new = 0.5 * np.dot(r, r)
92:         cost_change = cost - cost_new
93:         cost = cost_new
94:         g = A.T.dot(r)
95:         step_norm = norm(x[free_set] - x_free_old)
96: 
97:         if np.any(v):
98:             free_set = free_set[~v]
99:         else:
100:             break
101: 
102:     if max_iter is None:
103:         max_iter = n
104:     max_iter += iteration
105: 
106:     termination_status = None
107: 
108:     # Main BVLS loop.
109: 
110:     optimality = compute_kkt_optimality(g, on_bound)
111:     for iteration in range(iteration, max_iter):
112:         if verbose == 2:
113:             print_iteration_linear(iteration, cost, cost_change,
114:                                    step_norm, optimality)
115: 
116:         if optimality < tol:
117:             termination_status = 1
118: 
119:         if termination_status is not None:
120:             break
121: 
122:         move_to_free = np.argmax(g * on_bound)
123:         on_bound[move_to_free] = 0
124:         free_set = on_bound == 0
125:         active_set = ~free_set
126:         free_set, = np.nonzero(free_set)
127: 
128:         x_free = x[free_set]
129:         x_free_old = x_free.copy()
130:         lb_free = lb[free_set]
131:         ub_free = ub[free_set]
132: 
133:         A_free = A[:, free_set]
134:         b_free = b - A.dot(x * active_set)
135:         z = lstsq(A_free, b_free, rcond=-1)[0]
136: 
137:         lbv, = np.nonzero(z < lb_free)
138:         ubv, = np.nonzero(z > ub_free)
139:         v = np.hstack((lbv, ubv))
140: 
141:         if v.size > 0:
142:             alphas = np.hstack((
143:                 lb_free[lbv] - x_free[lbv],
144:                 ub_free[ubv] - x_free[ubv])) / (z[v] - x_free[v])
145: 
146:             i = np.argmin(alphas)
147:             i_free = v[i]
148:             alpha = alphas[i]
149: 
150:             x_free *= 1 - alpha
151:             x_free += alpha * z
152: 
153:             if i < lbv.size:
154:                 on_bound[free_set[i_free]] = -1
155:             else:
156:                 on_bound[free_set[i_free]] = 1
157:         else:
158:             x_free = z
159: 
160:         x[free_set] = x_free
161:         step_norm = norm(x_free - x_free_old)
162: 
163:         r = A.dot(x) - b
164:         cost_new = 0.5 * np.dot(r, r)
165:         cost_change = cost - cost_new
166: 
167:         if cost_change < tol * cost:
168:             termination_status = 2
169:         cost = cost_new
170: 
171:         g = A.T.dot(r)
172:         optimality = compute_kkt_optimality(g, on_bound)
173: 
174:     if termination_status is None:
175:         termination_status = 0
176: 
177:     return OptimizeResult(
178:         x=x, fun=r, cost=cost, optimality=optimality, active_mask=on_bound,
179:         nit=iteration + 1, status=termination_status,
180:         initial_cost=initial_cost)
181: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_247089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Bounded-Variable Least-Squares algorithm.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_247090 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_247090) is not StypyTypeError):

    if (import_247090 != 'pyd_module'):
        __import__(import_247090)
        sys_modules_247091 = sys.modules[import_247090]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_247091.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_247090)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.linalg import norm, lstsq' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_247092 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.linalg')

if (type(import_247092) is not StypyTypeError):

    if (import_247092 != 'pyd_module'):
        __import__(import_247092)
        sys_modules_247093 = sys.modules[import_247092]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.linalg', sys_modules_247093.module_type_store, module_type_store, ['norm', 'lstsq'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_247093, sys_modules_247093.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm, lstsq

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.linalg', None, module_type_store, ['norm', 'lstsq'], [norm, lstsq])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.linalg', import_247092)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.optimize import OptimizeResult' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_247094 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize')

if (type(import_247094) is not StypyTypeError):

    if (import_247094 != 'pyd_module'):
        __import__(import_247094)
        sys_modules_247095 = sys.modules[import_247094]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize', sys_modules_247095.module_type_store, module_type_store, ['OptimizeResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_247095, sys_modules_247095.module_type_store, module_type_store)
    else:
        from scipy.optimize import OptimizeResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize', None, module_type_store, ['OptimizeResult'], [OptimizeResult])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize', import_247094)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.optimize._lsq.common import print_header_linear, print_iteration_linear' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_247096 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize._lsq.common')

if (type(import_247096) is not StypyTypeError):

    if (import_247096 != 'pyd_module'):
        __import__(import_247096)
        sys_modules_247097 = sys.modules[import_247096]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize._lsq.common', sys_modules_247097.module_type_store, module_type_store, ['print_header_linear', 'print_iteration_linear'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_247097, sys_modules_247097.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.common import print_header_linear, print_iteration_linear

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize._lsq.common', None, module_type_store, ['print_header_linear', 'print_iteration_linear'], [print_header_linear, print_iteration_linear])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.common' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize._lsq.common', import_247096)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')


@norecursion
def compute_kkt_optimality(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'compute_kkt_optimality'
    module_type_store = module_type_store.open_function_context('compute_kkt_optimality', 11, 0, False)
    
    # Passed parameters checking function
    compute_kkt_optimality.stypy_localization = localization
    compute_kkt_optimality.stypy_type_of_self = None
    compute_kkt_optimality.stypy_type_store = module_type_store
    compute_kkt_optimality.stypy_function_name = 'compute_kkt_optimality'
    compute_kkt_optimality.stypy_param_names_list = ['g', 'on_bound']
    compute_kkt_optimality.stypy_varargs_param_name = None
    compute_kkt_optimality.stypy_kwargs_param_name = None
    compute_kkt_optimality.stypy_call_defaults = defaults
    compute_kkt_optimality.stypy_call_varargs = varargs
    compute_kkt_optimality.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compute_kkt_optimality', ['g', 'on_bound'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compute_kkt_optimality', localization, ['g', 'on_bound'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compute_kkt_optimality(...)' code ##################

    str_247098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'str', 'Compute the maximum violation of KKT conditions.')
    
    # Assigning a BinOp to a Name (line 13):
    
    # Assigning a BinOp to a Name (line 13):
    # Getting the type of 'g' (line 13)
    g_247099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'g')
    # Getting the type of 'on_bound' (line 13)
    on_bound_247100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'on_bound')
    # Applying the binary operator '*' (line 13)
    result_mul_247101 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 12), '*', g_247099, on_bound_247100)
    
    # Assigning a type to the variable 'g_kkt' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'g_kkt', result_mul_247101)
    
    # Assigning a Compare to a Name (line 14):
    
    # Assigning a Compare to a Name (line 14):
    
    # Getting the type of 'on_bound' (line 14)
    on_bound_247102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'on_bound')
    int_247103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 27), 'int')
    # Applying the binary operator '==' (line 14)
    result_eq_247104 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 15), '==', on_bound_247102, int_247103)
    
    # Assigning a type to the variable 'free_set' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'free_set', result_eq_247104)
    
    # Assigning a Call to a Subscript (line 15):
    
    # Assigning a Call to a Subscript (line 15):
    
    # Call to abs(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 15)
    free_set_247107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 31), 'free_set', False)
    # Getting the type of 'g' (line 15)
    g_247108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 29), 'g', False)
    # Obtaining the member '__getitem__' of a type (line 15)
    getitem___247109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 29), g_247108, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 15)
    subscript_call_result_247110 = invoke(stypy.reporting.localization.Localization(__file__, 15, 29), getitem___247109, free_set_247107)
    
    # Processing the call keyword arguments (line 15)
    kwargs_247111 = {}
    # Getting the type of 'np' (line 15)
    np_247105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'np', False)
    # Obtaining the member 'abs' of a type (line 15)
    abs_247106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 22), np_247105, 'abs')
    # Calling abs(args, kwargs) (line 15)
    abs_call_result_247112 = invoke(stypy.reporting.localization.Localization(__file__, 15, 22), abs_247106, *[subscript_call_result_247110], **kwargs_247111)
    
    # Getting the type of 'g_kkt' (line 15)
    g_kkt_247113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'g_kkt')
    # Getting the type of 'free_set' (line 15)
    free_set_247114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'free_set')
    # Storing an element on a container (line 15)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 4), g_kkt_247113, (free_set_247114, abs_call_result_247112))
    
    # Call to max(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'g_kkt' (line 16)
    g_kkt_247117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'g_kkt', False)
    # Processing the call keyword arguments (line 16)
    kwargs_247118 = {}
    # Getting the type of 'np' (line 16)
    np_247115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'np', False)
    # Obtaining the member 'max' of a type (line 16)
    max_247116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 11), np_247115, 'max')
    # Calling max(args, kwargs) (line 16)
    max_call_result_247119 = invoke(stypy.reporting.localization.Localization(__file__, 16, 11), max_247116, *[g_kkt_247117], **kwargs_247118)
    
    # Assigning a type to the variable 'stypy_return_type' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type', max_call_result_247119)
    
    # ################# End of 'compute_kkt_optimality(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compute_kkt_optimality' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_247120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_247120)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compute_kkt_optimality'
    return stypy_return_type_247120

# Assigning a type to the variable 'compute_kkt_optimality' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'compute_kkt_optimality', compute_kkt_optimality)

@norecursion
def bvls(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'bvls'
    module_type_store = module_type_store.open_function_context('bvls', 19, 0, False)
    
    # Passed parameters checking function
    bvls.stypy_localization = localization
    bvls.stypy_type_of_self = None
    bvls.stypy_type_store = module_type_store
    bvls.stypy_function_name = 'bvls'
    bvls.stypy_param_names_list = ['A', 'b', 'x_lsq', 'lb', 'ub', 'tol', 'max_iter', 'verbose']
    bvls.stypy_varargs_param_name = None
    bvls.stypy_kwargs_param_name = None
    bvls.stypy_call_defaults = defaults
    bvls.stypy_call_varargs = varargs
    bvls.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bvls', ['A', 'b', 'x_lsq', 'lb', 'ub', 'tol', 'max_iter', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bvls', localization, ['A', 'b', 'x_lsq', 'lb', 'ub', 'tol', 'max_iter', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bvls(...)' code ##################

    
    # Assigning a Attribute to a Tuple (line 20):
    
    # Assigning a Subscript to a Name (line 20):
    
    # Obtaining the type of the subscript
    int_247121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'int')
    # Getting the type of 'A' (line 20)
    A_247122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'A')
    # Obtaining the member 'shape' of a type (line 20)
    shape_247123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 11), A_247122, 'shape')
    # Obtaining the member '__getitem__' of a type (line 20)
    getitem___247124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), shape_247123, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 20)
    subscript_call_result_247125 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), getitem___247124, int_247121)
    
    # Assigning a type to the variable 'tuple_var_assignment_247083' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'tuple_var_assignment_247083', subscript_call_result_247125)
    
    # Assigning a Subscript to a Name (line 20):
    
    # Obtaining the type of the subscript
    int_247126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'int')
    # Getting the type of 'A' (line 20)
    A_247127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'A')
    # Obtaining the member 'shape' of a type (line 20)
    shape_247128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 11), A_247127, 'shape')
    # Obtaining the member '__getitem__' of a type (line 20)
    getitem___247129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), shape_247128, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 20)
    subscript_call_result_247130 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), getitem___247129, int_247126)
    
    # Assigning a type to the variable 'tuple_var_assignment_247084' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'tuple_var_assignment_247084', subscript_call_result_247130)
    
    # Assigning a Name to a Name (line 20):
    # Getting the type of 'tuple_var_assignment_247083' (line 20)
    tuple_var_assignment_247083_247131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'tuple_var_assignment_247083')
    # Assigning a type to the variable 'm' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'm', tuple_var_assignment_247083_247131)
    
    # Assigning a Name to a Name (line 20):
    # Getting the type of 'tuple_var_assignment_247084' (line 20)
    tuple_var_assignment_247084_247132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'tuple_var_assignment_247084')
    # Assigning a type to the variable 'n' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 7), 'n', tuple_var_assignment_247084_247132)
    
    # Assigning a Call to a Name (line 22):
    
    # Assigning a Call to a Name (line 22):
    
    # Call to copy(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_247135 = {}
    # Getting the type of 'x_lsq' (line 22)
    x_lsq_247133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'x_lsq', False)
    # Obtaining the member 'copy' of a type (line 22)
    copy_247134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), x_lsq_247133, 'copy')
    # Calling copy(args, kwargs) (line 22)
    copy_call_result_247136 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), copy_247134, *[], **kwargs_247135)
    
    # Assigning a type to the variable 'x' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'x', copy_call_result_247136)
    
    # Assigning a Call to a Name (line 23):
    
    # Assigning a Call to a Name (line 23):
    
    # Call to zeros(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'n' (line 23)
    n_247139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'n', False)
    # Processing the call keyword arguments (line 23)
    kwargs_247140 = {}
    # Getting the type of 'np' (line 23)
    np_247137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'np', False)
    # Obtaining the member 'zeros' of a type (line 23)
    zeros_247138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 15), np_247137, 'zeros')
    # Calling zeros(args, kwargs) (line 23)
    zeros_call_result_247141 = invoke(stypy.reporting.localization.Localization(__file__, 23, 15), zeros_247138, *[n_247139], **kwargs_247140)
    
    # Assigning a type to the variable 'on_bound' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'on_bound', zeros_call_result_247141)
    
    # Assigning a Compare to a Name (line 25):
    
    # Assigning a Compare to a Name (line 25):
    
    # Getting the type of 'x' (line 25)
    x_247142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'x')
    # Getting the type of 'lb' (line 25)
    lb_247143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'lb')
    # Applying the binary operator '<' (line 25)
    result_lt_247144 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 11), '<', x_247142, lb_247143)
    
    # Assigning a type to the variable 'mask' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'mask', result_lt_247144)
    
    # Assigning a Subscript to a Subscript (line 26):
    
    # Assigning a Subscript to a Subscript (line 26):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 26)
    mask_247145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'mask')
    # Getting the type of 'lb' (line 26)
    lb_247146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'lb')
    # Obtaining the member '__getitem__' of a type (line 26)
    getitem___247147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 14), lb_247146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 26)
    subscript_call_result_247148 = invoke(stypy.reporting.localization.Localization(__file__, 26, 14), getitem___247147, mask_247145)
    
    # Getting the type of 'x' (line 26)
    x_247149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'x')
    # Getting the type of 'mask' (line 26)
    mask_247150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 6), 'mask')
    # Storing an element on a container (line 26)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 4), x_247149, (mask_247150, subscript_call_result_247148))
    
    # Assigning a Num to a Subscript (line 27):
    
    # Assigning a Num to a Subscript (line 27):
    int_247151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'int')
    # Getting the type of 'on_bound' (line 27)
    on_bound_247152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'on_bound')
    # Getting the type of 'mask' (line 27)
    mask_247153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 13), 'mask')
    # Storing an element on a container (line 27)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), on_bound_247152, (mask_247153, int_247151))
    
    # Assigning a Compare to a Name (line 29):
    
    # Assigning a Compare to a Name (line 29):
    
    # Getting the type of 'x' (line 29)
    x_247154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'x')
    # Getting the type of 'ub' (line 29)
    ub_247155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'ub')
    # Applying the binary operator '>' (line 29)
    result_gt_247156 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 11), '>', x_247154, ub_247155)
    
    # Assigning a type to the variable 'mask' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'mask', result_gt_247156)
    
    # Assigning a Subscript to a Subscript (line 30):
    
    # Assigning a Subscript to a Subscript (line 30):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 30)
    mask_247157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'mask')
    # Getting the type of 'ub' (line 30)
    ub_247158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 14), 'ub')
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___247159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 14), ub_247158, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_247160 = invoke(stypy.reporting.localization.Localization(__file__, 30, 14), getitem___247159, mask_247157)
    
    # Getting the type of 'x' (line 30)
    x_247161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'x')
    # Getting the type of 'mask' (line 30)
    mask_247162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 6), 'mask')
    # Storing an element on a container (line 30)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), x_247161, (mask_247162, subscript_call_result_247160))
    
    # Assigning a Num to a Subscript (line 31):
    
    # Assigning a Num to a Subscript (line 31):
    int_247163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'int')
    # Getting the type of 'on_bound' (line 31)
    on_bound_247164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'on_bound')
    # Getting the type of 'mask' (line 31)
    mask_247165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 'mask')
    # Storing an element on a container (line 31)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 4), on_bound_247164, (mask_247165, int_247163))
    
    # Assigning a Compare to a Name (line 33):
    
    # Assigning a Compare to a Name (line 33):
    
    # Getting the type of 'on_bound' (line 33)
    on_bound_247166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'on_bound')
    int_247167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 27), 'int')
    # Applying the binary operator '==' (line 33)
    result_eq_247168 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 15), '==', on_bound_247166, int_247167)
    
    # Assigning a type to the variable 'free_set' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'free_set', result_eq_247168)
    
    # Assigning a UnaryOp to a Name (line 34):
    
    # Assigning a UnaryOp to a Name (line 34):
    
    # Getting the type of 'free_set' (line 34)
    free_set_247169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'free_set')
    # Applying the '~' unary operator (line 34)
    result_inv_247170 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 17), '~', free_set_247169)
    
    # Assigning a type to the variable 'active_set' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'active_set', result_inv_247170)
    
    # Assigning a Call to a Tuple (line 35):
    
    # Assigning a Subscript to a Name (line 35):
    
    # Obtaining the type of the subscript
    int_247171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'int')
    
    # Call to where(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'free_set' (line 35)
    free_set_247174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 'free_set', False)
    # Processing the call keyword arguments (line 35)
    kwargs_247175 = {}
    # Getting the type of 'np' (line 35)
    np_247172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'np', False)
    # Obtaining the member 'where' of a type (line 35)
    where_247173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 16), np_247172, 'where')
    # Calling where(args, kwargs) (line 35)
    where_call_result_247176 = invoke(stypy.reporting.localization.Localization(__file__, 35, 16), where_247173, *[free_set_247174], **kwargs_247175)
    
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___247177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), where_call_result_247176, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_247178 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), getitem___247177, int_247171)
    
    # Assigning a type to the variable 'tuple_var_assignment_247085' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'tuple_var_assignment_247085', subscript_call_result_247178)
    
    # Assigning a Name to a Name (line 35):
    # Getting the type of 'tuple_var_assignment_247085' (line 35)
    tuple_var_assignment_247085_247179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'tuple_var_assignment_247085')
    # Assigning a type to the variable 'free_set' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'free_set', tuple_var_assignment_247085_247179)
    
    # Assigning a BinOp to a Name (line 37):
    
    # Assigning a BinOp to a Name (line 37):
    
    # Call to dot(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'x' (line 37)
    x_247182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'x', False)
    # Processing the call keyword arguments (line 37)
    kwargs_247183 = {}
    # Getting the type of 'A' (line 37)
    A_247180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'A', False)
    # Obtaining the member 'dot' of a type (line 37)
    dot_247181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), A_247180, 'dot')
    # Calling dot(args, kwargs) (line 37)
    dot_call_result_247184 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), dot_247181, *[x_247182], **kwargs_247183)
    
    # Getting the type of 'b' (line 37)
    b_247185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), 'b')
    # Applying the binary operator '-' (line 37)
    result_sub_247186 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 8), '-', dot_call_result_247184, b_247185)
    
    # Assigning a type to the variable 'r' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'r', result_sub_247186)
    
    # Assigning a BinOp to a Name (line 38):
    
    # Assigning a BinOp to a Name (line 38):
    float_247187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 11), 'float')
    
    # Call to dot(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'r' (line 38)
    r_247190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'r', False)
    # Getting the type of 'r' (line 38)
    r_247191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'r', False)
    # Processing the call keyword arguments (line 38)
    kwargs_247192 = {}
    # Getting the type of 'np' (line 38)
    np_247188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'np', False)
    # Obtaining the member 'dot' of a type (line 38)
    dot_247189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 17), np_247188, 'dot')
    # Calling dot(args, kwargs) (line 38)
    dot_call_result_247193 = invoke(stypy.reporting.localization.Localization(__file__, 38, 17), dot_247189, *[r_247190, r_247191], **kwargs_247192)
    
    # Applying the binary operator '*' (line 38)
    result_mul_247194 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 11), '*', float_247187, dot_call_result_247193)
    
    # Assigning a type to the variable 'cost' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'cost', result_mul_247194)
    
    # Assigning a Name to a Name (line 39):
    
    # Assigning a Name to a Name (line 39):
    # Getting the type of 'cost' (line 39)
    cost_247195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'cost')
    # Assigning a type to the variable 'initial_cost' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'initial_cost', cost_247195)
    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Call to a Name (line 40):
    
    # Call to dot(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'r' (line 40)
    r_247199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'r', False)
    # Processing the call keyword arguments (line 40)
    kwargs_247200 = {}
    # Getting the type of 'A' (line 40)
    A_247196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'A', False)
    # Obtaining the member 'T' of a type (line 40)
    T_247197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), A_247196, 'T')
    # Obtaining the member 'dot' of a type (line 40)
    dot_247198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), T_247197, 'dot')
    # Calling dot(args, kwargs) (line 40)
    dot_call_result_247201 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), dot_247198, *[r_247199], **kwargs_247200)
    
    # Assigning a type to the variable 'g' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'g', dot_call_result_247201)
    
    # Assigning a Name to a Name (line 42):
    
    # Assigning a Name to a Name (line 42):
    # Getting the type of 'None' (line 42)
    None_247202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 18), 'None')
    # Assigning a type to the variable 'cost_change' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'cost_change', None_247202)
    
    # Assigning a Name to a Name (line 43):
    
    # Assigning a Name to a Name (line 43):
    # Getting the type of 'None' (line 43)
    None_247203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'None')
    # Assigning a type to the variable 'step_norm' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'step_norm', None_247203)
    
    # Assigning a Num to a Name (line 44):
    
    # Assigning a Num to a Name (line 44):
    int_247204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'int')
    # Assigning a type to the variable 'iteration' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'iteration', int_247204)
    
    
    # Getting the type of 'verbose' (line 46)
    verbose_247205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 7), 'verbose')
    int_247206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 18), 'int')
    # Applying the binary operator '==' (line 46)
    result_eq_247207 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 7), '==', verbose_247205, int_247206)
    
    # Testing the type of an if condition (line 46)
    if_condition_247208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 4), result_eq_247207)
    # Assigning a type to the variable 'if_condition_247208' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'if_condition_247208', if_condition_247208)
    # SSA begins for if statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print_header_linear(...): (line 47)
    # Processing the call keyword arguments (line 47)
    kwargs_247210 = {}
    # Getting the type of 'print_header_linear' (line 47)
    print_header_linear_247209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'print_header_linear', False)
    # Calling print_header_linear(args, kwargs) (line 47)
    print_header_linear_call_result_247211 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), print_header_linear_247209, *[], **kwargs_247210)
    
    # SSA join for if statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'free_set' (line 58)
    free_set_247212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 10), 'free_set')
    # Obtaining the member 'size' of a type (line 58)
    size_247213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 10), free_set_247212, 'size')
    int_247214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 26), 'int')
    # Applying the binary operator '>' (line 58)
    result_gt_247215 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 10), '>', size_247213, int_247214)
    
    # Testing the type of an if condition (line 58)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 4), result_gt_247215)
    # SSA begins for while statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # Getting the type of 'verbose' (line 59)
    verbose_247216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'verbose')
    int_247217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'int')
    # Applying the binary operator '==' (line 59)
    result_eq_247218 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 11), '==', verbose_247216, int_247217)
    
    # Testing the type of an if condition (line 59)
    if_condition_247219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_eq_247218)
    # Assigning a type to the variable 'if_condition_247219' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_247219', if_condition_247219)
    # SSA begins for if statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to compute_kkt_optimality(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'g' (line 60)
    g_247221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 48), 'g', False)
    # Getting the type of 'on_bound' (line 60)
    on_bound_247222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 51), 'on_bound', False)
    # Processing the call keyword arguments (line 60)
    kwargs_247223 = {}
    # Getting the type of 'compute_kkt_optimality' (line 60)
    compute_kkt_optimality_247220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'compute_kkt_optimality', False)
    # Calling compute_kkt_optimality(args, kwargs) (line 60)
    compute_kkt_optimality_call_result_247224 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), compute_kkt_optimality_247220, *[g_247221, on_bound_247222], **kwargs_247223)
    
    # Assigning a type to the variable 'optimality' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'optimality', compute_kkt_optimality_call_result_247224)
    
    # Call to print_iteration_linear(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'iteration' (line 61)
    iteration_247226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 35), 'iteration', False)
    # Getting the type of 'cost' (line 61)
    cost_247227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 46), 'cost', False)
    # Getting the type of 'cost_change' (line 61)
    cost_change_247228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 52), 'cost_change', False)
    # Getting the type of 'step_norm' (line 61)
    step_norm_247229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 65), 'step_norm', False)
    # Getting the type of 'optimality' (line 62)
    optimality_247230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 35), 'optimality', False)
    # Processing the call keyword arguments (line 61)
    kwargs_247231 = {}
    # Getting the type of 'print_iteration_linear' (line 61)
    print_iteration_linear_247225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'print_iteration_linear', False)
    # Calling print_iteration_linear(args, kwargs) (line 61)
    print_iteration_linear_call_result_247232 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), print_iteration_linear_247225, *[iteration_247226, cost_247227, cost_change_247228, step_norm_247229, optimality_247230], **kwargs_247231)
    
    # SSA join for if statement (line 59)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'iteration' (line 64)
    iteration_247233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'iteration')
    int_247234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 21), 'int')
    # Applying the binary operator '+=' (line 64)
    result_iadd_247235 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 8), '+=', iteration_247233, int_247234)
    # Assigning a type to the variable 'iteration' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'iteration', result_iadd_247235)
    
    
    # Assigning a Call to a Name (line 65):
    
    # Assigning a Call to a Name (line 65):
    
    # Call to copy(...): (line 65)
    # Processing the call keyword arguments (line 65)
    kwargs_247241 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 65)
    free_set_247236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 'free_set', False)
    # Getting the type of 'x' (line 65)
    x_247237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___247238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 21), x_247237, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_247239 = invoke(stypy.reporting.localization.Localization(__file__, 65, 21), getitem___247238, free_set_247236)
    
    # Obtaining the member 'copy' of a type (line 65)
    copy_247240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 21), subscript_call_result_247239, 'copy')
    # Calling copy(args, kwargs) (line 65)
    copy_call_result_247242 = invoke(stypy.reporting.localization.Localization(__file__, 65, 21), copy_247240, *[], **kwargs_247241)
    
    # Assigning a type to the variable 'x_free_old' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'x_free_old', copy_call_result_247242)
    
    # Assigning a Subscript to a Name (line 67):
    
    # Assigning a Subscript to a Name (line 67):
    
    # Obtaining the type of the subscript
    slice_247243 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 67, 17), None, None, None)
    # Getting the type of 'free_set' (line 67)
    free_set_247244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'free_set')
    # Getting the type of 'A' (line 67)
    A_247245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'A')
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___247246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 17), A_247245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_247247 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), getitem___247246, (slice_247243, free_set_247244))
    
    # Assigning a type to the variable 'A_free' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'A_free', subscript_call_result_247247)
    
    # Assigning a BinOp to a Name (line 68):
    
    # Assigning a BinOp to a Name (line 68):
    # Getting the type of 'b' (line 68)
    b_247248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'b')
    
    # Call to dot(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'x' (line 68)
    x_247251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'x', False)
    # Getting the type of 'active_set' (line 68)
    active_set_247252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 31), 'active_set', False)
    # Applying the binary operator '*' (line 68)
    result_mul_247253 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 27), '*', x_247251, active_set_247252)
    
    # Processing the call keyword arguments (line 68)
    kwargs_247254 = {}
    # Getting the type of 'A' (line 68)
    A_247249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'A', False)
    # Obtaining the member 'dot' of a type (line 68)
    dot_247250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 21), A_247249, 'dot')
    # Calling dot(args, kwargs) (line 68)
    dot_call_result_247255 = invoke(stypy.reporting.localization.Localization(__file__, 68, 21), dot_247250, *[result_mul_247253], **kwargs_247254)
    
    # Applying the binary operator '-' (line 68)
    result_sub_247256 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 17), '-', b_247248, dot_call_result_247255)
    
    # Assigning a type to the variable 'b_free' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'b_free', result_sub_247256)
    
    # Assigning a Subscript to a Name (line 69):
    
    # Assigning a Subscript to a Name (line 69):
    
    # Obtaining the type of the subscript
    int_247257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 44), 'int')
    
    # Call to lstsq(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'A_free' (line 69)
    A_free_247259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 18), 'A_free', False)
    # Getting the type of 'b_free' (line 69)
    b_free_247260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'b_free', False)
    # Processing the call keyword arguments (line 69)
    int_247261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 40), 'int')
    keyword_247262 = int_247261
    kwargs_247263 = {'rcond': keyword_247262}
    # Getting the type of 'lstsq' (line 69)
    lstsq_247258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 69)
    lstsq_call_result_247264 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), lstsq_247258, *[A_free_247259, b_free_247260], **kwargs_247263)
    
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___247265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), lstsq_call_result_247264, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_247266 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), getitem___247265, int_247257)
    
    # Assigning a type to the variable 'z' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'z', subscript_call_result_247266)
    
    # Assigning a Compare to a Name (line 71):
    
    # Assigning a Compare to a Name (line 71):
    
    # Getting the type of 'z' (line 71)
    z_247267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'z')
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 71)
    free_set_247268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'free_set')
    # Getting the type of 'lb' (line 71)
    lb_247269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'lb')
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___247270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 18), lb_247269, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_247271 = invoke(stypy.reporting.localization.Localization(__file__, 71, 18), getitem___247270, free_set_247268)
    
    # Applying the binary operator '<' (line 71)
    result_lt_247272 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 14), '<', z_247267, subscript_call_result_247271)
    
    # Assigning a type to the variable 'lbv' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'lbv', result_lt_247272)
    
    # Assigning a Compare to a Name (line 72):
    
    # Assigning a Compare to a Name (line 72):
    
    # Getting the type of 'z' (line 72)
    z_247273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 14), 'z')
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 72)
    free_set_247274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'free_set')
    # Getting the type of 'ub' (line 72)
    ub_247275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'ub')
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___247276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 18), ub_247275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_247277 = invoke(stypy.reporting.localization.Localization(__file__, 72, 18), getitem___247276, free_set_247274)
    
    # Applying the binary operator '>' (line 72)
    result_gt_247278 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 14), '>', z_247273, subscript_call_result_247277)
    
    # Assigning a type to the variable 'ubv' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'ubv', result_gt_247278)
    
    # Assigning a BinOp to a Name (line 73):
    
    # Assigning a BinOp to a Name (line 73):
    # Getting the type of 'lbv' (line 73)
    lbv_247279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'lbv')
    # Getting the type of 'ubv' (line 73)
    ubv_247280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'ubv')
    # Applying the binary operator '|' (line 73)
    result_or__247281 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 12), '|', lbv_247279, ubv_247280)
    
    # Assigning a type to the variable 'v' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'v', result_or__247281)
    
    
    # Call to any(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'lbv' (line 75)
    lbv_247284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'lbv', False)
    # Processing the call keyword arguments (line 75)
    kwargs_247285 = {}
    # Getting the type of 'np' (line 75)
    np_247282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'np', False)
    # Obtaining the member 'any' of a type (line 75)
    any_247283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 11), np_247282, 'any')
    # Calling any(args, kwargs) (line 75)
    any_call_result_247286 = invoke(stypy.reporting.localization.Localization(__file__, 75, 11), any_247283, *[lbv_247284], **kwargs_247285)
    
    # Testing the type of an if condition (line 75)
    if_condition_247287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 8), any_call_result_247286)
    # Assigning a type to the variable 'if_condition_247287' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'if_condition_247287', if_condition_247287)
    # SSA begins for if statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 76):
    
    # Assigning a Subscript to a Name (line 76):
    
    # Obtaining the type of the subscript
    # Getting the type of 'lbv' (line 76)
    lbv_247288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'lbv')
    # Getting the type of 'free_set' (line 76)
    free_set_247289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'free_set')
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___247290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 18), free_set_247289, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_247291 = invoke(stypy.reporting.localization.Localization(__file__, 76, 18), getitem___247290, lbv_247288)
    
    # Assigning a type to the variable 'ind' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'ind', subscript_call_result_247291)
    
    # Assigning a Subscript to a Subscript (line 77):
    
    # Assigning a Subscript to a Subscript (line 77):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 77)
    ind_247292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'ind')
    # Getting the type of 'lb' (line 77)
    lb_247293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'lb')
    # Obtaining the member '__getitem__' of a type (line 77)
    getitem___247294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 21), lb_247293, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 77)
    subscript_call_result_247295 = invoke(stypy.reporting.localization.Localization(__file__, 77, 21), getitem___247294, ind_247292)
    
    # Getting the type of 'x' (line 77)
    x_247296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'x')
    # Getting the type of 'ind' (line 77)
    ind_247297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), 'ind')
    # Storing an element on a container (line 77)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 12), x_247296, (ind_247297, subscript_call_result_247295))
    
    # Assigning a Name to a Subscript (line 78):
    
    # Assigning a Name to a Subscript (line 78):
    # Getting the type of 'True' (line 78)
    True_247298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'True')
    # Getting the type of 'active_set' (line 78)
    active_set_247299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'active_set')
    # Getting the type of 'ind' (line 78)
    ind_247300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'ind')
    # Storing an element on a container (line 78)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 12), active_set_247299, (ind_247300, True_247298))
    
    # Assigning a Num to a Subscript (line 79):
    
    # Assigning a Num to a Subscript (line 79):
    int_247301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'int')
    # Getting the type of 'on_bound' (line 79)
    on_bound_247302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'on_bound')
    # Getting the type of 'ind' (line 79)
    ind_247303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'ind')
    # Storing an element on a container (line 79)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 12), on_bound_247302, (ind_247303, int_247301))
    # SSA join for if statement (line 75)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'ubv' (line 81)
    ubv_247306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 18), 'ubv', False)
    # Processing the call keyword arguments (line 81)
    kwargs_247307 = {}
    # Getting the type of 'np' (line 81)
    np_247304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'np', False)
    # Obtaining the member 'any' of a type (line 81)
    any_247305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 11), np_247304, 'any')
    # Calling any(args, kwargs) (line 81)
    any_call_result_247308 = invoke(stypy.reporting.localization.Localization(__file__, 81, 11), any_247305, *[ubv_247306], **kwargs_247307)
    
    # Testing the type of an if condition (line 81)
    if_condition_247309 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), any_call_result_247308)
    # Assigning a type to the variable 'if_condition_247309' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_247309', if_condition_247309)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 82):
    
    # Assigning a Subscript to a Name (line 82):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ubv' (line 82)
    ubv_247310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'ubv')
    # Getting the type of 'free_set' (line 82)
    free_set_247311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'free_set')
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___247312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 18), free_set_247311, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_247313 = invoke(stypy.reporting.localization.Localization(__file__, 82, 18), getitem___247312, ubv_247310)
    
    # Assigning a type to the variable 'ind' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'ind', subscript_call_result_247313)
    
    # Assigning a Subscript to a Subscript (line 83):
    
    # Assigning a Subscript to a Subscript (line 83):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 83)
    ind_247314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'ind')
    # Getting the type of 'ub' (line 83)
    ub_247315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 21), 'ub')
    # Obtaining the member '__getitem__' of a type (line 83)
    getitem___247316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 21), ub_247315, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 83)
    subscript_call_result_247317 = invoke(stypy.reporting.localization.Localization(__file__, 83, 21), getitem___247316, ind_247314)
    
    # Getting the type of 'x' (line 83)
    x_247318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'x')
    # Getting the type of 'ind' (line 83)
    ind_247319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 14), 'ind')
    # Storing an element on a container (line 83)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 12), x_247318, (ind_247319, subscript_call_result_247317))
    
    # Assigning a Name to a Subscript (line 84):
    
    # Assigning a Name to a Subscript (line 84):
    # Getting the type of 'True' (line 84)
    True_247320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'True')
    # Getting the type of 'active_set' (line 84)
    active_set_247321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'active_set')
    # Getting the type of 'ind' (line 84)
    ind_247322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'ind')
    # Storing an element on a container (line 84)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 12), active_set_247321, (ind_247322, True_247320))
    
    # Assigning a Num to a Subscript (line 85):
    
    # Assigning a Num to a Subscript (line 85):
    int_247323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 28), 'int')
    # Getting the type of 'on_bound' (line 85)
    on_bound_247324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'on_bound')
    # Getting the type of 'ind' (line 85)
    ind_247325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'ind')
    # Storing an element on a container (line 85)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 12), on_bound_247324, (ind_247325, int_247323))
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 87):
    
    # Assigning a Subscript to a Name (line 87):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'v' (line 87)
    v_247326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'v')
    # Applying the '~' unary operator (line 87)
    result_inv_247327 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 23), '~', v_247326)
    
    # Getting the type of 'free_set' (line 87)
    free_set_247328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 14), 'free_set')
    # Obtaining the member '__getitem__' of a type (line 87)
    getitem___247329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 14), free_set_247328, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 87)
    subscript_call_result_247330 = invoke(stypy.reporting.localization.Localization(__file__, 87, 14), getitem___247329, result_inv_247327)
    
    # Assigning a type to the variable 'ind' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'ind', subscript_call_result_247330)
    
    # Assigning a Subscript to a Subscript (line 88):
    
    # Assigning a Subscript to a Subscript (line 88):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'v' (line 88)
    v_247331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'v')
    # Applying the '~' unary operator (line 88)
    result_inv_247332 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 19), '~', v_247331)
    
    # Getting the type of 'z' (line 88)
    z_247333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'z')
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___247334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 17), z_247333, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_247335 = invoke(stypy.reporting.localization.Localization(__file__, 88, 17), getitem___247334, result_inv_247332)
    
    # Getting the type of 'x' (line 88)
    x_247336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'x')
    # Getting the type of 'ind' (line 88)
    ind_247337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 10), 'ind')
    # Storing an element on a container (line 88)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 8), x_247336, (ind_247337, subscript_call_result_247335))
    
    # Assigning a BinOp to a Name (line 90):
    
    # Assigning a BinOp to a Name (line 90):
    
    # Call to dot(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'x' (line 90)
    x_247340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'x', False)
    # Processing the call keyword arguments (line 90)
    kwargs_247341 = {}
    # Getting the type of 'A' (line 90)
    A_247338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'A', False)
    # Obtaining the member 'dot' of a type (line 90)
    dot_247339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), A_247338, 'dot')
    # Calling dot(args, kwargs) (line 90)
    dot_call_result_247342 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), dot_247339, *[x_247340], **kwargs_247341)
    
    # Getting the type of 'b' (line 90)
    b_247343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'b')
    # Applying the binary operator '-' (line 90)
    result_sub_247344 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 12), '-', dot_call_result_247342, b_247343)
    
    # Assigning a type to the variable 'r' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'r', result_sub_247344)
    
    # Assigning a BinOp to a Name (line 91):
    
    # Assigning a BinOp to a Name (line 91):
    float_247345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'float')
    
    # Call to dot(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'r' (line 91)
    r_247348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'r', False)
    # Getting the type of 'r' (line 91)
    r_247349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 35), 'r', False)
    # Processing the call keyword arguments (line 91)
    kwargs_247350 = {}
    # Getting the type of 'np' (line 91)
    np_247346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'np', False)
    # Obtaining the member 'dot' of a type (line 91)
    dot_247347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 25), np_247346, 'dot')
    # Calling dot(args, kwargs) (line 91)
    dot_call_result_247351 = invoke(stypy.reporting.localization.Localization(__file__, 91, 25), dot_247347, *[r_247348, r_247349], **kwargs_247350)
    
    # Applying the binary operator '*' (line 91)
    result_mul_247352 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 19), '*', float_247345, dot_call_result_247351)
    
    # Assigning a type to the variable 'cost_new' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'cost_new', result_mul_247352)
    
    # Assigning a BinOp to a Name (line 92):
    
    # Assigning a BinOp to a Name (line 92):
    # Getting the type of 'cost' (line 92)
    cost_247353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'cost')
    # Getting the type of 'cost_new' (line 92)
    cost_new_247354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 29), 'cost_new')
    # Applying the binary operator '-' (line 92)
    result_sub_247355 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 22), '-', cost_247353, cost_new_247354)
    
    # Assigning a type to the variable 'cost_change' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'cost_change', result_sub_247355)
    
    # Assigning a Name to a Name (line 93):
    
    # Assigning a Name to a Name (line 93):
    # Getting the type of 'cost_new' (line 93)
    cost_new_247356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'cost_new')
    # Assigning a type to the variable 'cost' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'cost', cost_new_247356)
    
    # Assigning a Call to a Name (line 94):
    
    # Assigning a Call to a Name (line 94):
    
    # Call to dot(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'r' (line 94)
    r_247360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'r', False)
    # Processing the call keyword arguments (line 94)
    kwargs_247361 = {}
    # Getting the type of 'A' (line 94)
    A_247357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'A', False)
    # Obtaining the member 'T' of a type (line 94)
    T_247358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), A_247357, 'T')
    # Obtaining the member 'dot' of a type (line 94)
    dot_247359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), T_247358, 'dot')
    # Calling dot(args, kwargs) (line 94)
    dot_call_result_247362 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), dot_247359, *[r_247360], **kwargs_247361)
    
    # Assigning a type to the variable 'g' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'g', dot_call_result_247362)
    
    # Assigning a Call to a Name (line 95):
    
    # Assigning a Call to a Name (line 95):
    
    # Call to norm(...): (line 95)
    # Processing the call arguments (line 95)
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 95)
    free_set_247364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 27), 'free_set', False)
    # Getting the type of 'x' (line 95)
    x_247365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 95)
    getitem___247366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 25), x_247365, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 95)
    subscript_call_result_247367 = invoke(stypy.reporting.localization.Localization(__file__, 95, 25), getitem___247366, free_set_247364)
    
    # Getting the type of 'x_free_old' (line 95)
    x_free_old_247368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 39), 'x_free_old', False)
    # Applying the binary operator '-' (line 95)
    result_sub_247369 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 25), '-', subscript_call_result_247367, x_free_old_247368)
    
    # Processing the call keyword arguments (line 95)
    kwargs_247370 = {}
    # Getting the type of 'norm' (line 95)
    norm_247363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 20), 'norm', False)
    # Calling norm(args, kwargs) (line 95)
    norm_call_result_247371 = invoke(stypy.reporting.localization.Localization(__file__, 95, 20), norm_247363, *[result_sub_247369], **kwargs_247370)
    
    # Assigning a type to the variable 'step_norm' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'step_norm', norm_call_result_247371)
    
    
    # Call to any(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'v' (line 97)
    v_247374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'v', False)
    # Processing the call keyword arguments (line 97)
    kwargs_247375 = {}
    # Getting the type of 'np' (line 97)
    np_247372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'np', False)
    # Obtaining the member 'any' of a type (line 97)
    any_247373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), np_247372, 'any')
    # Calling any(args, kwargs) (line 97)
    any_call_result_247376 = invoke(stypy.reporting.localization.Localization(__file__, 97, 11), any_247373, *[v_247374], **kwargs_247375)
    
    # Testing the type of an if condition (line 97)
    if_condition_247377 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), any_call_result_247376)
    # Assigning a type to the variable 'if_condition_247377' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_247377', if_condition_247377)
    # SSA begins for if statement (line 97)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 98):
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'v' (line 98)
    v_247378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'v')
    # Applying the '~' unary operator (line 98)
    result_inv_247379 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 32), '~', v_247378)
    
    # Getting the type of 'free_set' (line 98)
    free_set_247380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'free_set')
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___247381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 23), free_set_247380, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_247382 = invoke(stypy.reporting.localization.Localization(__file__, 98, 23), getitem___247381, result_inv_247379)
    
    # Assigning a type to the variable 'free_set' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'free_set', subscript_call_result_247382)
    # SSA branch for the else part of an if statement (line 97)
    module_type_store.open_ssa_branch('else')
    # SSA join for if statement (line 97)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 102)
    # Getting the type of 'max_iter' (line 102)
    max_iter_247383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 7), 'max_iter')
    # Getting the type of 'None' (line 102)
    None_247384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'None')
    
    (may_be_247385, more_types_in_union_247386) = may_be_none(max_iter_247383, None_247384)

    if may_be_247385:

        if more_types_in_union_247386:
            # Runtime conditional SSA (line 102)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 103):
        
        # Assigning a Name to a Name (line 103):
        # Getting the type of 'n' (line 103)
        n_247387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'n')
        # Assigning a type to the variable 'max_iter' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'max_iter', n_247387)

        if more_types_in_union_247386:
            # SSA join for if statement (line 102)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'max_iter' (line 104)
    max_iter_247388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'max_iter')
    # Getting the type of 'iteration' (line 104)
    iteration_247389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'iteration')
    # Applying the binary operator '+=' (line 104)
    result_iadd_247390 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 4), '+=', max_iter_247388, iteration_247389)
    # Assigning a type to the variable 'max_iter' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'max_iter', result_iadd_247390)
    
    
    # Assigning a Name to a Name (line 106):
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'None' (line 106)
    None_247391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'None')
    # Assigning a type to the variable 'termination_status' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'termination_status', None_247391)
    
    # Assigning a Call to a Name (line 110):
    
    # Assigning a Call to a Name (line 110):
    
    # Call to compute_kkt_optimality(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'g' (line 110)
    g_247393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 40), 'g', False)
    # Getting the type of 'on_bound' (line 110)
    on_bound_247394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 43), 'on_bound', False)
    # Processing the call keyword arguments (line 110)
    kwargs_247395 = {}
    # Getting the type of 'compute_kkt_optimality' (line 110)
    compute_kkt_optimality_247392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'compute_kkt_optimality', False)
    # Calling compute_kkt_optimality(args, kwargs) (line 110)
    compute_kkt_optimality_call_result_247396 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), compute_kkt_optimality_247392, *[g_247393, on_bound_247394], **kwargs_247395)
    
    # Assigning a type to the variable 'optimality' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'optimality', compute_kkt_optimality_call_result_247396)
    
    
    # Call to range(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'iteration' (line 111)
    iteration_247398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'iteration', False)
    # Getting the type of 'max_iter' (line 111)
    max_iter_247399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 38), 'max_iter', False)
    # Processing the call keyword arguments (line 111)
    kwargs_247400 = {}
    # Getting the type of 'range' (line 111)
    range_247397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 21), 'range', False)
    # Calling range(args, kwargs) (line 111)
    range_call_result_247401 = invoke(stypy.reporting.localization.Localization(__file__, 111, 21), range_247397, *[iteration_247398, max_iter_247399], **kwargs_247400)
    
    # Testing the type of a for loop iterable (line 111)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 4), range_call_result_247401)
    # Getting the type of the for loop variable (line 111)
    for_loop_var_247402 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 4), range_call_result_247401)
    # Assigning a type to the variable 'iteration' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'iteration', for_loop_var_247402)
    # SSA begins for a for statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'verbose' (line 112)
    verbose_247403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'verbose')
    int_247404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 22), 'int')
    # Applying the binary operator '==' (line 112)
    result_eq_247405 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), '==', verbose_247403, int_247404)
    
    # Testing the type of an if condition (line 112)
    if_condition_247406 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 8), result_eq_247405)
    # Assigning a type to the variable 'if_condition_247406' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'if_condition_247406', if_condition_247406)
    # SSA begins for if statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print_iteration_linear(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'iteration' (line 113)
    iteration_247408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 35), 'iteration', False)
    # Getting the type of 'cost' (line 113)
    cost_247409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 46), 'cost', False)
    # Getting the type of 'cost_change' (line 113)
    cost_change_247410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 52), 'cost_change', False)
    # Getting the type of 'step_norm' (line 114)
    step_norm_247411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 35), 'step_norm', False)
    # Getting the type of 'optimality' (line 114)
    optimality_247412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 46), 'optimality', False)
    # Processing the call keyword arguments (line 113)
    kwargs_247413 = {}
    # Getting the type of 'print_iteration_linear' (line 113)
    print_iteration_linear_247407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'print_iteration_linear', False)
    # Calling print_iteration_linear(args, kwargs) (line 113)
    print_iteration_linear_call_result_247414 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), print_iteration_linear_247407, *[iteration_247408, cost_247409, cost_change_247410, step_norm_247411, optimality_247412], **kwargs_247413)
    
    # SSA join for if statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'optimality' (line 116)
    optimality_247415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'optimality')
    # Getting the type of 'tol' (line 116)
    tol_247416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'tol')
    # Applying the binary operator '<' (line 116)
    result_lt_247417 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 11), '<', optimality_247415, tol_247416)
    
    # Testing the type of an if condition (line 116)
    if_condition_247418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 8), result_lt_247417)
    # Assigning a type to the variable 'if_condition_247418' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'if_condition_247418', if_condition_247418)
    # SSA begins for if statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 117):
    
    # Assigning a Num to a Name (line 117):
    int_247419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 33), 'int')
    # Assigning a type to the variable 'termination_status' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'termination_status', int_247419)
    # SSA join for if statement (line 116)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 119)
    # Getting the type of 'termination_status' (line 119)
    termination_status_247420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'termination_status')
    # Getting the type of 'None' (line 119)
    None_247421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 37), 'None')
    
    (may_be_247422, more_types_in_union_247423) = may_not_be_none(termination_status_247420, None_247421)

    if may_be_247422:

        if more_types_in_union_247423:
            # Runtime conditional SSA (line 119)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_247423:
            # SSA join for if statement (line 119)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to argmax(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'g' (line 122)
    g_247426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 33), 'g', False)
    # Getting the type of 'on_bound' (line 122)
    on_bound_247427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'on_bound', False)
    # Applying the binary operator '*' (line 122)
    result_mul_247428 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 33), '*', g_247426, on_bound_247427)
    
    # Processing the call keyword arguments (line 122)
    kwargs_247429 = {}
    # Getting the type of 'np' (line 122)
    np_247424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 23), 'np', False)
    # Obtaining the member 'argmax' of a type (line 122)
    argmax_247425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 23), np_247424, 'argmax')
    # Calling argmax(args, kwargs) (line 122)
    argmax_call_result_247430 = invoke(stypy.reporting.localization.Localization(__file__, 122, 23), argmax_247425, *[result_mul_247428], **kwargs_247429)
    
    # Assigning a type to the variable 'move_to_free' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'move_to_free', argmax_call_result_247430)
    
    # Assigning a Num to a Subscript (line 123):
    
    # Assigning a Num to a Subscript (line 123):
    int_247431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 33), 'int')
    # Getting the type of 'on_bound' (line 123)
    on_bound_247432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'on_bound')
    # Getting the type of 'move_to_free' (line 123)
    move_to_free_247433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'move_to_free')
    # Storing an element on a container (line 123)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 8), on_bound_247432, (move_to_free_247433, int_247431))
    
    # Assigning a Compare to a Name (line 124):
    
    # Assigning a Compare to a Name (line 124):
    
    # Getting the type of 'on_bound' (line 124)
    on_bound_247434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'on_bound')
    int_247435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 31), 'int')
    # Applying the binary operator '==' (line 124)
    result_eq_247436 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 19), '==', on_bound_247434, int_247435)
    
    # Assigning a type to the variable 'free_set' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'free_set', result_eq_247436)
    
    # Assigning a UnaryOp to a Name (line 125):
    
    # Assigning a UnaryOp to a Name (line 125):
    
    # Getting the type of 'free_set' (line 125)
    free_set_247437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 22), 'free_set')
    # Applying the '~' unary operator (line 125)
    result_inv_247438 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 21), '~', free_set_247437)
    
    # Assigning a type to the variable 'active_set' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'active_set', result_inv_247438)
    
    # Assigning a Call to a Tuple (line 126):
    
    # Assigning a Subscript to a Name (line 126):
    
    # Obtaining the type of the subscript
    int_247439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 8), 'int')
    
    # Call to nonzero(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'free_set' (line 126)
    free_set_247442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 31), 'free_set', False)
    # Processing the call keyword arguments (line 126)
    kwargs_247443 = {}
    # Getting the type of 'np' (line 126)
    np_247440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 126)
    nonzero_247441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 20), np_247440, 'nonzero')
    # Calling nonzero(args, kwargs) (line 126)
    nonzero_call_result_247444 = invoke(stypy.reporting.localization.Localization(__file__, 126, 20), nonzero_247441, *[free_set_247442], **kwargs_247443)
    
    # Obtaining the member '__getitem__' of a type (line 126)
    getitem___247445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), nonzero_call_result_247444, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 126)
    subscript_call_result_247446 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), getitem___247445, int_247439)
    
    # Assigning a type to the variable 'tuple_var_assignment_247086' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'tuple_var_assignment_247086', subscript_call_result_247446)
    
    # Assigning a Name to a Name (line 126):
    # Getting the type of 'tuple_var_assignment_247086' (line 126)
    tuple_var_assignment_247086_247447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'tuple_var_assignment_247086')
    # Assigning a type to the variable 'free_set' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'free_set', tuple_var_assignment_247086_247447)
    
    # Assigning a Subscript to a Name (line 128):
    
    # Assigning a Subscript to a Name (line 128):
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 128)
    free_set_247448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'free_set')
    # Getting the type of 'x' (line 128)
    x_247449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'x')
    # Obtaining the member '__getitem__' of a type (line 128)
    getitem___247450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 17), x_247449, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 128)
    subscript_call_result_247451 = invoke(stypy.reporting.localization.Localization(__file__, 128, 17), getitem___247450, free_set_247448)
    
    # Assigning a type to the variable 'x_free' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'x_free', subscript_call_result_247451)
    
    # Assigning a Call to a Name (line 129):
    
    # Assigning a Call to a Name (line 129):
    
    # Call to copy(...): (line 129)
    # Processing the call keyword arguments (line 129)
    kwargs_247454 = {}
    # Getting the type of 'x_free' (line 129)
    x_free_247452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'x_free', False)
    # Obtaining the member 'copy' of a type (line 129)
    copy_247453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 21), x_free_247452, 'copy')
    # Calling copy(args, kwargs) (line 129)
    copy_call_result_247455 = invoke(stypy.reporting.localization.Localization(__file__, 129, 21), copy_247453, *[], **kwargs_247454)
    
    # Assigning a type to the variable 'x_free_old' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'x_free_old', copy_call_result_247455)
    
    # Assigning a Subscript to a Name (line 130):
    
    # Assigning a Subscript to a Name (line 130):
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 130)
    free_set_247456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'free_set')
    # Getting the type of 'lb' (line 130)
    lb_247457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'lb')
    # Obtaining the member '__getitem__' of a type (line 130)
    getitem___247458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 18), lb_247457, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
    subscript_call_result_247459 = invoke(stypy.reporting.localization.Localization(__file__, 130, 18), getitem___247458, free_set_247456)
    
    # Assigning a type to the variable 'lb_free' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'lb_free', subscript_call_result_247459)
    
    # Assigning a Subscript to a Name (line 131):
    
    # Assigning a Subscript to a Name (line 131):
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 131)
    free_set_247460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'free_set')
    # Getting the type of 'ub' (line 131)
    ub_247461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 18), 'ub')
    # Obtaining the member '__getitem__' of a type (line 131)
    getitem___247462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 18), ub_247461, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 131)
    subscript_call_result_247463 = invoke(stypy.reporting.localization.Localization(__file__, 131, 18), getitem___247462, free_set_247460)
    
    # Assigning a type to the variable 'ub_free' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'ub_free', subscript_call_result_247463)
    
    # Assigning a Subscript to a Name (line 133):
    
    # Assigning a Subscript to a Name (line 133):
    
    # Obtaining the type of the subscript
    slice_247464 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 133, 17), None, None, None)
    # Getting the type of 'free_set' (line 133)
    free_set_247465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'free_set')
    # Getting the type of 'A' (line 133)
    A_247466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 17), 'A')
    # Obtaining the member '__getitem__' of a type (line 133)
    getitem___247467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 17), A_247466, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 133)
    subscript_call_result_247468 = invoke(stypy.reporting.localization.Localization(__file__, 133, 17), getitem___247467, (slice_247464, free_set_247465))
    
    # Assigning a type to the variable 'A_free' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'A_free', subscript_call_result_247468)
    
    # Assigning a BinOp to a Name (line 134):
    
    # Assigning a BinOp to a Name (line 134):
    # Getting the type of 'b' (line 134)
    b_247469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'b')
    
    # Call to dot(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'x' (line 134)
    x_247472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 27), 'x', False)
    # Getting the type of 'active_set' (line 134)
    active_set_247473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 31), 'active_set', False)
    # Applying the binary operator '*' (line 134)
    result_mul_247474 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 27), '*', x_247472, active_set_247473)
    
    # Processing the call keyword arguments (line 134)
    kwargs_247475 = {}
    # Getting the type of 'A' (line 134)
    A_247470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 21), 'A', False)
    # Obtaining the member 'dot' of a type (line 134)
    dot_247471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 21), A_247470, 'dot')
    # Calling dot(args, kwargs) (line 134)
    dot_call_result_247476 = invoke(stypy.reporting.localization.Localization(__file__, 134, 21), dot_247471, *[result_mul_247474], **kwargs_247475)
    
    # Applying the binary operator '-' (line 134)
    result_sub_247477 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 17), '-', b_247469, dot_call_result_247476)
    
    # Assigning a type to the variable 'b_free' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'b_free', result_sub_247477)
    
    # Assigning a Subscript to a Name (line 135):
    
    # Assigning a Subscript to a Name (line 135):
    
    # Obtaining the type of the subscript
    int_247478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 44), 'int')
    
    # Call to lstsq(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'A_free' (line 135)
    A_free_247480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'A_free', False)
    # Getting the type of 'b_free' (line 135)
    b_free_247481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 26), 'b_free', False)
    # Processing the call keyword arguments (line 135)
    int_247482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 40), 'int')
    keyword_247483 = int_247482
    kwargs_247484 = {'rcond': keyword_247483}
    # Getting the type of 'lstsq' (line 135)
    lstsq_247479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 135)
    lstsq_call_result_247485 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), lstsq_247479, *[A_free_247480, b_free_247481], **kwargs_247484)
    
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___247486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), lstsq_call_result_247485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_247487 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), getitem___247486, int_247478)
    
    # Assigning a type to the variable 'z' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'z', subscript_call_result_247487)
    
    # Assigning a Call to a Tuple (line 137):
    
    # Assigning a Subscript to a Name (line 137):
    
    # Obtaining the type of the subscript
    int_247488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 8), 'int')
    
    # Call to nonzero(...): (line 137)
    # Processing the call arguments (line 137)
    
    # Getting the type of 'z' (line 137)
    z_247491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 26), 'z', False)
    # Getting the type of 'lb_free' (line 137)
    lb_free_247492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 30), 'lb_free', False)
    # Applying the binary operator '<' (line 137)
    result_lt_247493 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 26), '<', z_247491, lb_free_247492)
    
    # Processing the call keyword arguments (line 137)
    kwargs_247494 = {}
    # Getting the type of 'np' (line 137)
    np_247489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 137)
    nonzero_247490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 15), np_247489, 'nonzero')
    # Calling nonzero(args, kwargs) (line 137)
    nonzero_call_result_247495 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), nonzero_247490, *[result_lt_247493], **kwargs_247494)
    
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___247496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), nonzero_call_result_247495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_247497 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), getitem___247496, int_247488)
    
    # Assigning a type to the variable 'tuple_var_assignment_247087' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_247087', subscript_call_result_247497)
    
    # Assigning a Name to a Name (line 137):
    # Getting the type of 'tuple_var_assignment_247087' (line 137)
    tuple_var_assignment_247087_247498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_247087')
    # Assigning a type to the variable 'lbv' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'lbv', tuple_var_assignment_247087_247498)
    
    # Assigning a Call to a Tuple (line 138):
    
    # Assigning a Subscript to a Name (line 138):
    
    # Obtaining the type of the subscript
    int_247499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 8), 'int')
    
    # Call to nonzero(...): (line 138)
    # Processing the call arguments (line 138)
    
    # Getting the type of 'z' (line 138)
    z_247502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 26), 'z', False)
    # Getting the type of 'ub_free' (line 138)
    ub_free_247503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 30), 'ub_free', False)
    # Applying the binary operator '>' (line 138)
    result_gt_247504 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 26), '>', z_247502, ub_free_247503)
    
    # Processing the call keyword arguments (line 138)
    kwargs_247505 = {}
    # Getting the type of 'np' (line 138)
    np_247500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 138)
    nonzero_247501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), np_247500, 'nonzero')
    # Calling nonzero(args, kwargs) (line 138)
    nonzero_call_result_247506 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), nonzero_247501, *[result_gt_247504], **kwargs_247505)
    
    # Obtaining the member '__getitem__' of a type (line 138)
    getitem___247507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), nonzero_call_result_247506, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 138)
    subscript_call_result_247508 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), getitem___247507, int_247499)
    
    # Assigning a type to the variable 'tuple_var_assignment_247088' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'tuple_var_assignment_247088', subscript_call_result_247508)
    
    # Assigning a Name to a Name (line 138):
    # Getting the type of 'tuple_var_assignment_247088' (line 138)
    tuple_var_assignment_247088_247509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'tuple_var_assignment_247088')
    # Assigning a type to the variable 'ubv' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'ubv', tuple_var_assignment_247088_247509)
    
    # Assigning a Call to a Name (line 139):
    
    # Assigning a Call to a Name (line 139):
    
    # Call to hstack(...): (line 139)
    # Processing the call arguments (line 139)
    
    # Obtaining an instance of the builtin type 'tuple' (line 139)
    tuple_247512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 139)
    # Adding element type (line 139)
    # Getting the type of 'lbv' (line 139)
    lbv_247513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 23), 'lbv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 23), tuple_247512, lbv_247513)
    # Adding element type (line 139)
    # Getting the type of 'ubv' (line 139)
    ubv_247514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 28), 'ubv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 23), tuple_247512, ubv_247514)
    
    # Processing the call keyword arguments (line 139)
    kwargs_247515 = {}
    # Getting the type of 'np' (line 139)
    np_247510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'np', False)
    # Obtaining the member 'hstack' of a type (line 139)
    hstack_247511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), np_247510, 'hstack')
    # Calling hstack(args, kwargs) (line 139)
    hstack_call_result_247516 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), hstack_247511, *[tuple_247512], **kwargs_247515)
    
    # Assigning a type to the variable 'v' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'v', hstack_call_result_247516)
    
    
    # Getting the type of 'v' (line 141)
    v_247517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'v')
    # Obtaining the member 'size' of a type (line 141)
    size_247518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 11), v_247517, 'size')
    int_247519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 20), 'int')
    # Applying the binary operator '>' (line 141)
    result_gt_247520 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 11), '>', size_247518, int_247519)
    
    # Testing the type of an if condition (line 141)
    if_condition_247521 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), result_gt_247520)
    # Assigning a type to the variable 'if_condition_247521' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_247521', if_condition_247521)
    # SSA begins for if statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 142):
    
    # Assigning a BinOp to a Name (line 142):
    
    # Call to hstack(...): (line 142)
    # Processing the call arguments (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 143)
    tuple_247524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 143)
    # Adding element type (line 143)
    
    # Obtaining the type of the subscript
    # Getting the type of 'lbv' (line 143)
    lbv_247525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'lbv', False)
    # Getting the type of 'lb_free' (line 143)
    lb_free_247526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'lb_free', False)
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___247527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), lb_free_247526, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_247528 = invoke(stypy.reporting.localization.Localization(__file__, 143, 16), getitem___247527, lbv_247525)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'lbv' (line 143)
    lbv_247529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 38), 'lbv', False)
    # Getting the type of 'x_free' (line 143)
    x_free_247530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 31), 'x_free', False)
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___247531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 31), x_free_247530, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_247532 = invoke(stypy.reporting.localization.Localization(__file__, 143, 31), getitem___247531, lbv_247529)
    
    # Applying the binary operator '-' (line 143)
    result_sub_247533 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 16), '-', subscript_call_result_247528, subscript_call_result_247532)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 16), tuple_247524, result_sub_247533)
    # Adding element type (line 143)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ubv' (line 144)
    ubv_247534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'ubv', False)
    # Getting the type of 'ub_free' (line 144)
    ub_free_247535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'ub_free', False)
    # Obtaining the member '__getitem__' of a type (line 144)
    getitem___247536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 16), ub_free_247535, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 144)
    subscript_call_result_247537 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), getitem___247536, ubv_247534)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ubv' (line 144)
    ubv_247538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 38), 'ubv', False)
    # Getting the type of 'x_free' (line 144)
    x_free_247539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 31), 'x_free', False)
    # Obtaining the member '__getitem__' of a type (line 144)
    getitem___247540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 31), x_free_247539, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 144)
    subscript_call_result_247541 = invoke(stypy.reporting.localization.Localization(__file__, 144, 31), getitem___247540, ubv_247538)
    
    # Applying the binary operator '-' (line 144)
    result_sub_247542 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 16), '-', subscript_call_result_247537, subscript_call_result_247541)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 16), tuple_247524, result_sub_247542)
    
    # Processing the call keyword arguments (line 142)
    kwargs_247543 = {}
    # Getting the type of 'np' (line 142)
    np_247522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'np', False)
    # Obtaining the member 'hstack' of a type (line 142)
    hstack_247523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 21), np_247522, 'hstack')
    # Calling hstack(args, kwargs) (line 142)
    hstack_call_result_247544 = invoke(stypy.reporting.localization.Localization(__file__, 142, 21), hstack_247523, *[tuple_247524], **kwargs_247543)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 144)
    v_247545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 50), 'v')
    # Getting the type of 'z' (line 144)
    z_247546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 48), 'z')
    # Obtaining the member '__getitem__' of a type (line 144)
    getitem___247547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 48), z_247546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 144)
    subscript_call_result_247548 = invoke(stypy.reporting.localization.Localization(__file__, 144, 48), getitem___247547, v_247545)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 144)
    v_247549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 62), 'v')
    # Getting the type of 'x_free' (line 144)
    x_free_247550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 55), 'x_free')
    # Obtaining the member '__getitem__' of a type (line 144)
    getitem___247551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 55), x_free_247550, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 144)
    subscript_call_result_247552 = invoke(stypy.reporting.localization.Localization(__file__, 144, 55), getitem___247551, v_247549)
    
    # Applying the binary operator '-' (line 144)
    result_sub_247553 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 48), '-', subscript_call_result_247548, subscript_call_result_247552)
    
    # Applying the binary operator 'div' (line 142)
    result_div_247554 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 21), 'div', hstack_call_result_247544, result_sub_247553)
    
    # Assigning a type to the variable 'alphas' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'alphas', result_div_247554)
    
    # Assigning a Call to a Name (line 146):
    
    # Assigning a Call to a Name (line 146):
    
    # Call to argmin(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'alphas' (line 146)
    alphas_247557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'alphas', False)
    # Processing the call keyword arguments (line 146)
    kwargs_247558 = {}
    # Getting the type of 'np' (line 146)
    np_247555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'np', False)
    # Obtaining the member 'argmin' of a type (line 146)
    argmin_247556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), np_247555, 'argmin')
    # Calling argmin(args, kwargs) (line 146)
    argmin_call_result_247559 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), argmin_247556, *[alphas_247557], **kwargs_247558)
    
    # Assigning a type to the variable 'i' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'i', argmin_call_result_247559)
    
    # Assigning a Subscript to a Name (line 147):
    
    # Assigning a Subscript to a Name (line 147):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 147)
    i_247560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'i')
    # Getting the type of 'v' (line 147)
    v_247561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'v')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___247562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 21), v_247561, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_247563 = invoke(stypy.reporting.localization.Localization(__file__, 147, 21), getitem___247562, i_247560)
    
    # Assigning a type to the variable 'i_free' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'i_free', subscript_call_result_247563)
    
    # Assigning a Subscript to a Name (line 148):
    
    # Assigning a Subscript to a Name (line 148):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 148)
    i_247564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'i')
    # Getting the type of 'alphas' (line 148)
    alphas_247565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'alphas')
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___247566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 20), alphas_247565, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_247567 = invoke(stypy.reporting.localization.Localization(__file__, 148, 20), getitem___247566, i_247564)
    
    # Assigning a type to the variable 'alpha' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'alpha', subscript_call_result_247567)
    
    # Getting the type of 'x_free' (line 150)
    x_free_247568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'x_free')
    int_247569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 22), 'int')
    # Getting the type of 'alpha' (line 150)
    alpha_247570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 26), 'alpha')
    # Applying the binary operator '-' (line 150)
    result_sub_247571 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 22), '-', int_247569, alpha_247570)
    
    # Applying the binary operator '*=' (line 150)
    result_imul_247572 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), '*=', x_free_247568, result_sub_247571)
    # Assigning a type to the variable 'x_free' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'x_free', result_imul_247572)
    
    
    # Getting the type of 'x_free' (line 151)
    x_free_247573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'x_free')
    # Getting the type of 'alpha' (line 151)
    alpha_247574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 22), 'alpha')
    # Getting the type of 'z' (line 151)
    z_247575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 30), 'z')
    # Applying the binary operator '*' (line 151)
    result_mul_247576 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 22), '*', alpha_247574, z_247575)
    
    # Applying the binary operator '+=' (line 151)
    result_iadd_247577 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 12), '+=', x_free_247573, result_mul_247576)
    # Assigning a type to the variable 'x_free' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'x_free', result_iadd_247577)
    
    
    
    # Getting the type of 'i' (line 153)
    i_247578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'i')
    # Getting the type of 'lbv' (line 153)
    lbv_247579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'lbv')
    # Obtaining the member 'size' of a type (line 153)
    size_247580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 19), lbv_247579, 'size')
    # Applying the binary operator '<' (line 153)
    result_lt_247581 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 15), '<', i_247578, size_247580)
    
    # Testing the type of an if condition (line 153)
    if_condition_247582 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 12), result_lt_247581)
    # Assigning a type to the variable 'if_condition_247582' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'if_condition_247582', if_condition_247582)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 154):
    
    # Assigning a Num to a Subscript (line 154):
    int_247583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 45), 'int')
    # Getting the type of 'on_bound' (line 154)
    on_bound_247584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'on_bound')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i_free' (line 154)
    i_free_247585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), 'i_free')
    # Getting the type of 'free_set' (line 154)
    free_set_247586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 25), 'free_set')
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___247587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 25), free_set_247586, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_247588 = invoke(stypy.reporting.localization.Localization(__file__, 154, 25), getitem___247587, i_free_247585)
    
    # Storing an element on a container (line 154)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 16), on_bound_247584, (subscript_call_result_247588, int_247583))
    # SSA branch for the else part of an if statement (line 153)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Subscript (line 156):
    
    # Assigning a Num to a Subscript (line 156):
    int_247589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 45), 'int')
    # Getting the type of 'on_bound' (line 156)
    on_bound_247590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'on_bound')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i_free' (line 156)
    i_free_247591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 34), 'i_free')
    # Getting the type of 'free_set' (line 156)
    free_set_247592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'free_set')
    # Obtaining the member '__getitem__' of a type (line 156)
    getitem___247593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 25), free_set_247592, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 156)
    subscript_call_result_247594 = invoke(stypy.reporting.localization.Localization(__file__, 156, 25), getitem___247593, i_free_247591)
    
    # Storing an element on a container (line 156)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 16), on_bound_247590, (subscript_call_result_247594, int_247589))
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 141)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 158):
    
    # Assigning a Name to a Name (line 158):
    # Getting the type of 'z' (line 158)
    z_247595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'z')
    # Assigning a type to the variable 'x_free' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'x_free', z_247595)
    # SSA join for if statement (line 141)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 160):
    
    # Assigning a Name to a Subscript (line 160):
    # Getting the type of 'x_free' (line 160)
    x_free_247596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 22), 'x_free')
    # Getting the type of 'x' (line 160)
    x_247597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'x')
    # Getting the type of 'free_set' (line 160)
    free_set_247598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 10), 'free_set')
    # Storing an element on a container (line 160)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 8), x_247597, (free_set_247598, x_free_247596))
    
    # Assigning a Call to a Name (line 161):
    
    # Assigning a Call to a Name (line 161):
    
    # Call to norm(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'x_free' (line 161)
    x_free_247600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 25), 'x_free', False)
    # Getting the type of 'x_free_old' (line 161)
    x_free_old_247601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 34), 'x_free_old', False)
    # Applying the binary operator '-' (line 161)
    result_sub_247602 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 25), '-', x_free_247600, x_free_old_247601)
    
    # Processing the call keyword arguments (line 161)
    kwargs_247603 = {}
    # Getting the type of 'norm' (line 161)
    norm_247599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'norm', False)
    # Calling norm(args, kwargs) (line 161)
    norm_call_result_247604 = invoke(stypy.reporting.localization.Localization(__file__, 161, 20), norm_247599, *[result_sub_247602], **kwargs_247603)
    
    # Assigning a type to the variable 'step_norm' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'step_norm', norm_call_result_247604)
    
    # Assigning a BinOp to a Name (line 163):
    
    # Assigning a BinOp to a Name (line 163):
    
    # Call to dot(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'x' (line 163)
    x_247607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 18), 'x', False)
    # Processing the call keyword arguments (line 163)
    kwargs_247608 = {}
    # Getting the type of 'A' (line 163)
    A_247605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'A', False)
    # Obtaining the member 'dot' of a type (line 163)
    dot_247606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), A_247605, 'dot')
    # Calling dot(args, kwargs) (line 163)
    dot_call_result_247609 = invoke(stypy.reporting.localization.Localization(__file__, 163, 12), dot_247606, *[x_247607], **kwargs_247608)
    
    # Getting the type of 'b' (line 163)
    b_247610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 23), 'b')
    # Applying the binary operator '-' (line 163)
    result_sub_247611 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 12), '-', dot_call_result_247609, b_247610)
    
    # Assigning a type to the variable 'r' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'r', result_sub_247611)
    
    # Assigning a BinOp to a Name (line 164):
    
    # Assigning a BinOp to a Name (line 164):
    float_247612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 19), 'float')
    
    # Call to dot(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'r' (line 164)
    r_247615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 32), 'r', False)
    # Getting the type of 'r' (line 164)
    r_247616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 35), 'r', False)
    # Processing the call keyword arguments (line 164)
    kwargs_247617 = {}
    # Getting the type of 'np' (line 164)
    np_247613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 25), 'np', False)
    # Obtaining the member 'dot' of a type (line 164)
    dot_247614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 25), np_247613, 'dot')
    # Calling dot(args, kwargs) (line 164)
    dot_call_result_247618 = invoke(stypy.reporting.localization.Localization(__file__, 164, 25), dot_247614, *[r_247615, r_247616], **kwargs_247617)
    
    # Applying the binary operator '*' (line 164)
    result_mul_247619 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 19), '*', float_247612, dot_call_result_247618)
    
    # Assigning a type to the variable 'cost_new' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'cost_new', result_mul_247619)
    
    # Assigning a BinOp to a Name (line 165):
    
    # Assigning a BinOp to a Name (line 165):
    # Getting the type of 'cost' (line 165)
    cost_247620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 22), 'cost')
    # Getting the type of 'cost_new' (line 165)
    cost_new_247621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 29), 'cost_new')
    # Applying the binary operator '-' (line 165)
    result_sub_247622 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 22), '-', cost_247620, cost_new_247621)
    
    # Assigning a type to the variable 'cost_change' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'cost_change', result_sub_247622)
    
    
    # Getting the type of 'cost_change' (line 167)
    cost_change_247623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'cost_change')
    # Getting the type of 'tol' (line 167)
    tol_247624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'tol')
    # Getting the type of 'cost' (line 167)
    cost_247625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 31), 'cost')
    # Applying the binary operator '*' (line 167)
    result_mul_247626 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 25), '*', tol_247624, cost_247625)
    
    # Applying the binary operator '<' (line 167)
    result_lt_247627 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 11), '<', cost_change_247623, result_mul_247626)
    
    # Testing the type of an if condition (line 167)
    if_condition_247628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 8), result_lt_247627)
    # Assigning a type to the variable 'if_condition_247628' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'if_condition_247628', if_condition_247628)
    # SSA begins for if statement (line 167)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 168):
    
    # Assigning a Num to a Name (line 168):
    int_247629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 33), 'int')
    # Assigning a type to the variable 'termination_status' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'termination_status', int_247629)
    # SSA join for if statement (line 167)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 169):
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'cost_new' (line 169)
    cost_new_247630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'cost_new')
    # Assigning a type to the variable 'cost' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'cost', cost_new_247630)
    
    # Assigning a Call to a Name (line 171):
    
    # Assigning a Call to a Name (line 171):
    
    # Call to dot(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'r' (line 171)
    r_247634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'r', False)
    # Processing the call keyword arguments (line 171)
    kwargs_247635 = {}
    # Getting the type of 'A' (line 171)
    A_247631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'A', False)
    # Obtaining the member 'T' of a type (line 171)
    T_247632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 12), A_247631, 'T')
    # Obtaining the member 'dot' of a type (line 171)
    dot_247633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 12), T_247632, 'dot')
    # Calling dot(args, kwargs) (line 171)
    dot_call_result_247636 = invoke(stypy.reporting.localization.Localization(__file__, 171, 12), dot_247633, *[r_247634], **kwargs_247635)
    
    # Assigning a type to the variable 'g' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'g', dot_call_result_247636)
    
    # Assigning a Call to a Name (line 172):
    
    # Assigning a Call to a Name (line 172):
    
    # Call to compute_kkt_optimality(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'g' (line 172)
    g_247638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 44), 'g', False)
    # Getting the type of 'on_bound' (line 172)
    on_bound_247639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 47), 'on_bound', False)
    # Processing the call keyword arguments (line 172)
    kwargs_247640 = {}
    # Getting the type of 'compute_kkt_optimality' (line 172)
    compute_kkt_optimality_247637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 21), 'compute_kkt_optimality', False)
    # Calling compute_kkt_optimality(args, kwargs) (line 172)
    compute_kkt_optimality_call_result_247641 = invoke(stypy.reporting.localization.Localization(__file__, 172, 21), compute_kkt_optimality_247637, *[g_247638, on_bound_247639], **kwargs_247640)
    
    # Assigning a type to the variable 'optimality' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'optimality', compute_kkt_optimality_call_result_247641)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 174)
    # Getting the type of 'termination_status' (line 174)
    termination_status_247642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 7), 'termination_status')
    # Getting the type of 'None' (line 174)
    None_247643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 29), 'None')
    
    (may_be_247644, more_types_in_union_247645) = may_be_none(termination_status_247642, None_247643)

    if may_be_247644:

        if more_types_in_union_247645:
            # Runtime conditional SSA (line 174)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 175):
        
        # Assigning a Num to a Name (line 175):
        int_247646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 29), 'int')
        # Assigning a type to the variable 'termination_status' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'termination_status', int_247646)

        if more_types_in_union_247645:
            # SSA join for if statement (line 174)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to OptimizeResult(...): (line 177)
    # Processing the call keyword arguments (line 177)
    # Getting the type of 'x' (line 178)
    x_247648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 10), 'x', False)
    keyword_247649 = x_247648
    # Getting the type of 'r' (line 178)
    r_247650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'r', False)
    keyword_247651 = r_247650
    # Getting the type of 'cost' (line 178)
    cost_247652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 25), 'cost', False)
    keyword_247653 = cost_247652
    # Getting the type of 'optimality' (line 178)
    optimality_247654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 42), 'optimality', False)
    keyword_247655 = optimality_247654
    # Getting the type of 'on_bound' (line 178)
    on_bound_247656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 66), 'on_bound', False)
    keyword_247657 = on_bound_247656
    # Getting the type of 'iteration' (line 179)
    iteration_247658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'iteration', False)
    int_247659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 24), 'int')
    # Applying the binary operator '+' (line 179)
    result_add_247660 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 12), '+', iteration_247658, int_247659)
    
    keyword_247661 = result_add_247660
    # Getting the type of 'termination_status' (line 179)
    termination_status_247662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 34), 'termination_status', False)
    keyword_247663 = termination_status_247662
    # Getting the type of 'initial_cost' (line 180)
    initial_cost_247664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'initial_cost', False)
    keyword_247665 = initial_cost_247664
    kwargs_247666 = {'status': keyword_247663, 'initial_cost': keyword_247665, 'active_mask': keyword_247657, 'cost': keyword_247653, 'optimality': keyword_247655, 'fun': keyword_247651, 'x': keyword_247649, 'nit': keyword_247661}
    # Getting the type of 'OptimizeResult' (line 177)
    OptimizeResult_247647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 177)
    OptimizeResult_call_result_247667 = invoke(stypy.reporting.localization.Localization(__file__, 177, 11), OptimizeResult_247647, *[], **kwargs_247666)
    
    # Assigning a type to the variable 'stypy_return_type' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type', OptimizeResult_call_result_247667)
    
    # ################# End of 'bvls(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bvls' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_247668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_247668)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bvls'
    return stypy_return_type_247668

# Assigning a type to the variable 'bvls' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'bvls', bvls)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
