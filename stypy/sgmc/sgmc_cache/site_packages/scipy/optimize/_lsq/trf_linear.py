
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''The adaptation of Trust Region Reflective algorithm for a linear
2: least-squares problem.'''
3: from __future__ import division, print_function, absolute_import
4: 
5: import numpy as np
6: from numpy.linalg import norm
7: from scipy.linalg import qr, solve_triangular
8: from scipy.sparse.linalg import lsmr
9: from scipy.optimize import OptimizeResult
10: 
11: from .givens_elimination import givens_elimination
12: from .common import (
13:     EPS, step_size_to_bound, find_active_constraints, in_bounds,
14:     make_strictly_feasible, build_quadratic_1d, evaluate_quadratic,
15:     minimize_quadratic_1d, CL_scaling_vector, reflective_transformation,
16:     print_header_linear, print_iteration_linear, compute_grad,
17:     regularized_lsq_operator, right_multiplied_operator)
18: 
19: 
20: def regularized_lsq_with_qr(m, n, R, QTb, perm, diag, copy_R=True):
21:     '''Solve regularized least squares using information from QR-decomposition.
22: 
23:     The initial problem is to solve the following system in a least-squares
24:     sense:
25:     ::
26: 
27:         A x = b
28:         D x = 0
29: 
30:     Where D is diagonal matrix. The method is based on QR decomposition
31:     of the form A P = Q R, where P is a column permutation matrix, Q is an
32:     orthogonal matrix and R is an upper triangular matrix.
33: 
34:     Parameters
35:     ----------
36:     m, n : int
37:         Initial shape of A.
38:     R : ndarray, shape (n, n)
39:         Upper triangular matrix from QR decomposition of A.
40:     QTb : ndarray, shape (n,)
41:         First n components of Q^T b.
42:     perm : ndarray, shape (n,)
43:         Array defining column permutation of A, such that i-th column of
44:         P is perm[i]-th column of identity matrix.
45:     diag : ndarray, shape (n,)
46:         Array containing diagonal elements of D.
47: 
48:     Returns
49:     -------
50:     x : ndarray, shape (n,)
51:         Found least-squares solution.
52:     '''
53:     if copy_R:
54:         R = R.copy()
55:     v = QTb.copy()
56: 
57:     givens_elimination(R, v, diag[perm])
58: 
59:     abs_diag_R = np.abs(np.diag(R))
60:     threshold = EPS * max(m, n) * np.max(abs_diag_R)
61:     nns, = np.nonzero(abs_diag_R > threshold)
62: 
63:     R = R[np.ix_(nns, nns)]
64:     v = v[nns]
65: 
66:     x = np.zeros(n)
67:     x[perm[nns]] = solve_triangular(R, v)
68: 
69:     return x
70: 
71: 
72: def backtracking(A, g, x, p, theta, p_dot_g, lb, ub):
73:     '''Find an appropriate step size using backtracking line search.'''
74:     alpha = 1
75:     while True:
76:         x_new, _ = reflective_transformation(x + alpha * p, lb, ub)
77:         step = x_new - x
78:         cost_change = -evaluate_quadratic(A, g, step)
79:         if cost_change > -0.1 * alpha * p_dot_g:
80:             break
81:         alpha *= 0.5
82: 
83:     active = find_active_constraints(x_new, lb, ub)
84:     if np.any(active != 0):
85:         x_new, _ = reflective_transformation(x + theta * alpha * p, lb, ub)
86:         x_new = make_strictly_feasible(x_new, lb, ub, rstep=0)
87:         step = x_new - x
88:         cost_change = -evaluate_quadratic(A, g, step)
89: 
90:     return x, step, cost_change
91: 
92: 
93: def select_step(x, A_h, g_h, c_h, p, p_h, d, lb, ub, theta):
94:     '''Select the best step according to Trust Region Reflective algorithm.'''
95:     if in_bounds(x + p, lb, ub):
96:         return p
97: 
98:     p_stride, hits = step_size_to_bound(x, p, lb, ub)
99:     r_h = np.copy(p_h)
100:     r_h[hits.astype(bool)] *= -1
101:     r = d * r_h
102: 
103:     # Restrict step, such that it hits the bound.
104:     p *= p_stride
105:     p_h *= p_stride
106:     x_on_bound = x + p
107: 
108:     # Find the step size along reflected direction.
109:     r_stride_u, _ = step_size_to_bound(x_on_bound, r, lb, ub)
110: 
111:     # Stay interior.
112:     r_stride_l = (1 - theta) * r_stride_u
113:     r_stride_u *= theta
114: 
115:     if r_stride_u > 0:
116:         a, b, c = build_quadratic_1d(A_h, g_h, r_h, s0=p_h, diag=c_h)
117:         r_stride, r_value = minimize_quadratic_1d(
118:             a, b, r_stride_l, r_stride_u, c=c)
119:         r_h = p_h + r_h * r_stride
120:         r = d * r_h
121:     else:
122:         r_value = np.inf
123: 
124:     # Now correct p_h to make it strictly interior.
125:     p_h *= theta
126:     p *= theta
127:     p_value = evaluate_quadratic(A_h, g_h, p_h, diag=c_h)
128: 
129:     ag_h = -g_h
130:     ag = d * ag_h
131:     ag_stride_u, _ = step_size_to_bound(x, ag, lb, ub)
132:     ag_stride_u *= theta
133:     a, b = build_quadratic_1d(A_h, g_h, ag_h, diag=c_h)
134:     ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride_u)
135:     ag *= ag_stride
136: 
137:     if p_value < r_value and p_value < ag_value:
138:         return p
139:     elif r_value < p_value and r_value < ag_value:
140:         return r
141:     else:
142:         return ag
143: 
144: 
145: def trf_linear(A, b, x_lsq, lb, ub, tol, lsq_solver, lsmr_tol, max_iter,
146:                verbose):
147:     m, n = A.shape
148:     x, _ = reflective_transformation(x_lsq, lb, ub)
149:     x = make_strictly_feasible(x, lb, ub, rstep=0.1)
150: 
151:     if lsq_solver == 'exact':
152:         QT, R, perm = qr(A, mode='economic', pivoting=True)
153:         QT = QT.T
154: 
155:         if m < n:
156:             R = np.vstack((R, np.zeros((n - m, n))))
157: 
158:         QTr = np.zeros(n)
159:         k = min(m, n)
160:     elif lsq_solver == 'lsmr':
161:         r_aug = np.zeros(m + n)
162:         auto_lsmr_tol = False
163:         if lsmr_tol is None:
164:             lsmr_tol = 1e-2 * tol
165:         elif lsmr_tol == 'auto':
166:             auto_lsmr_tol = True
167: 
168:     r = A.dot(x) - b
169:     g = compute_grad(A, r)
170:     cost = 0.5 * np.dot(r, r)
171:     initial_cost = cost
172: 
173:     termination_status = None
174:     step_norm = None
175:     cost_change = None
176: 
177:     if max_iter is None:
178:         max_iter = 100
179: 
180:     if verbose == 2:
181:         print_header_linear()
182: 
183:     for iteration in range(max_iter):
184:         v, dv = CL_scaling_vector(x, g, lb, ub)
185:         g_scaled = g * v
186:         g_norm = norm(g_scaled, ord=np.inf)
187:         if g_norm < tol:
188:             termination_status = 1
189: 
190:         if verbose == 2:
191:             print_iteration_linear(iteration, cost, cost_change,
192:                                    step_norm, g_norm)
193: 
194:         if termination_status is not None:
195:             break
196: 
197:         diag_h = g * dv
198:         diag_root_h = diag_h ** 0.5
199:         d = v ** 0.5
200:         g_h = d * g
201: 
202:         A_h = right_multiplied_operator(A, d)
203:         if lsq_solver == 'exact':
204:             QTr[:k] = QT.dot(r)
205:             p_h = -regularized_lsq_with_qr(m, n, R * d[perm], QTr, perm,
206:                                            diag_root_h, copy_R=False)
207:         elif lsq_solver == 'lsmr':
208:             lsmr_op = regularized_lsq_operator(A_h, diag_root_h)
209:             r_aug[:m] = r
210:             if auto_lsmr_tol:
211:                 eta = 1e-2 * min(0.5, g_norm)
212:                 lsmr_tol = max(EPS, min(0.1, eta * g_norm))
213:             p_h = -lsmr(lsmr_op, r_aug, atol=lsmr_tol, btol=lsmr_tol)[0]
214: 
215:         p = d * p_h
216: 
217:         p_dot_g = np.dot(p, g)
218:         if p_dot_g > 0:
219:             termination_status = -1
220: 
221:         theta = 1 - min(0.005, g_norm)
222:         step = select_step(x, A_h, g_h, diag_h, p, p_h, d, lb, ub, theta)
223:         cost_change = -evaluate_quadratic(A, g, step)
224: 
225:         # Perhaps almost never executed, the idea is that `p` is descent
226:         # direction thus we must find acceptable cost decrease using simple
227:         # "backtracking", otherwise algorithm's logic would break.
228:         if cost_change < 0:
229:             x, step, cost_change = backtracking(
230:                 A, g, x, p, theta, p_dot_g, lb, ub)
231:         else:
232:             x = make_strictly_feasible(x + step, lb, ub, rstep=0)
233: 
234:         step_norm = norm(step)
235:         r = A.dot(x) - b
236:         g = compute_grad(A, r)
237: 
238:         if cost_change < tol * cost:
239:             termination_status = 2
240: 
241:         cost = 0.5 * np.dot(r, r)
242: 
243:     if termination_status is None:
244:         termination_status = 0
245: 
246:     active_mask = find_active_constraints(x, lb, ub, rtol=tol)
247: 
248:     return OptimizeResult(
249:         x=x, fun=r, cost=cost, optimality=g_norm, active_mask=active_mask,
250:         nit=iteration + 1, status=termination_status,
251:         initial_cost=initial_cost)
252: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_254711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'The adaptation of Trust Region Reflective algorithm for a linear\nleast-squares problem.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_254712 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_254712) is not StypyTypeError):

    if (import_254712 != 'pyd_module'):
        __import__(import_254712)
        sys_modules_254713 = sys.modules[import_254712]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_254713.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_254712)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.linalg import norm' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_254714 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.linalg')

if (type(import_254714) is not StypyTypeError):

    if (import_254714 != 'pyd_module'):
        __import__(import_254714)
        sys_modules_254715 = sys.modules[import_254714]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.linalg', sys_modules_254715.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_254715, sys_modules_254715.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.linalg', import_254714)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.linalg import qr, solve_triangular' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_254716 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg')

if (type(import_254716) is not StypyTypeError):

    if (import_254716 != 'pyd_module'):
        __import__(import_254716)
        sys_modules_254717 = sys.modules[import_254716]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', sys_modules_254717.module_type_store, module_type_store, ['qr', 'solve_triangular'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_254717, sys_modules_254717.module_type_store, module_type_store)
    else:
        from scipy.linalg import qr, solve_triangular

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', None, module_type_store, ['qr', 'solve_triangular'], [qr, solve_triangular])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', import_254716)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.sparse.linalg import lsmr' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_254718 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.linalg')

if (type(import_254718) is not StypyTypeError):

    if (import_254718 != 'pyd_module'):
        __import__(import_254718)
        sys_modules_254719 = sys.modules[import_254718]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.linalg', sys_modules_254719.module_type_store, module_type_store, ['lsmr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_254719, sys_modules_254719.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import lsmr

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.linalg', None, module_type_store, ['lsmr'], [lsmr])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.linalg', import_254718)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.optimize import OptimizeResult' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_254720 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize')

if (type(import_254720) is not StypyTypeError):

    if (import_254720 != 'pyd_module'):
        __import__(import_254720)
        sys_modules_254721 = sys.modules[import_254720]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', sys_modules_254721.module_type_store, module_type_store, ['OptimizeResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_254721, sys_modules_254721.module_type_store, module_type_store)
    else:
        from scipy.optimize import OptimizeResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', None, module_type_store, ['OptimizeResult'], [OptimizeResult])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', import_254720)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.optimize._lsq.givens_elimination import givens_elimination' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_254722 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._lsq.givens_elimination')

if (type(import_254722) is not StypyTypeError):

    if (import_254722 != 'pyd_module'):
        __import__(import_254722)
        sys_modules_254723 = sys.modules[import_254722]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._lsq.givens_elimination', sys_modules_254723.module_type_store, module_type_store, ['givens_elimination'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_254723, sys_modules_254723.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.givens_elimination import givens_elimination

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._lsq.givens_elimination', None, module_type_store, ['givens_elimination'], [givens_elimination])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.givens_elimination' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize._lsq.givens_elimination', import_254722)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.optimize._lsq.common import EPS, step_size_to_bound, find_active_constraints, in_bounds, make_strictly_feasible, build_quadratic_1d, evaluate_quadratic, minimize_quadratic_1d, CL_scaling_vector, reflective_transformation, print_header_linear, print_iteration_linear, compute_grad, regularized_lsq_operator, right_multiplied_operator' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_254724 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._lsq.common')

if (type(import_254724) is not StypyTypeError):

    if (import_254724 != 'pyd_module'):
        __import__(import_254724)
        sys_modules_254725 = sys.modules[import_254724]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._lsq.common', sys_modules_254725.module_type_store, module_type_store, ['EPS', 'step_size_to_bound', 'find_active_constraints', 'in_bounds', 'make_strictly_feasible', 'build_quadratic_1d', 'evaluate_quadratic', 'minimize_quadratic_1d', 'CL_scaling_vector', 'reflective_transformation', 'print_header_linear', 'print_iteration_linear', 'compute_grad', 'regularized_lsq_operator', 'right_multiplied_operator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_254725, sys_modules_254725.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.common import EPS, step_size_to_bound, find_active_constraints, in_bounds, make_strictly_feasible, build_quadratic_1d, evaluate_quadratic, minimize_quadratic_1d, CL_scaling_vector, reflective_transformation, print_header_linear, print_iteration_linear, compute_grad, regularized_lsq_operator, right_multiplied_operator

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._lsq.common', None, module_type_store, ['EPS', 'step_size_to_bound', 'find_active_constraints', 'in_bounds', 'make_strictly_feasible', 'build_quadratic_1d', 'evaluate_quadratic', 'minimize_quadratic_1d', 'CL_scaling_vector', 'reflective_transformation', 'print_header_linear', 'print_iteration_linear', 'compute_grad', 'regularized_lsq_operator', 'right_multiplied_operator'], [EPS, step_size_to_bound, find_active_constraints, in_bounds, make_strictly_feasible, build_quadratic_1d, evaluate_quadratic, minimize_quadratic_1d, CL_scaling_vector, reflective_transformation, print_header_linear, print_iteration_linear, compute_grad, regularized_lsq_operator, right_multiplied_operator])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.common' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize._lsq.common', import_254724)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')


@norecursion
def regularized_lsq_with_qr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 20)
    True_254726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 61), 'True')
    defaults = [True_254726]
    # Create a new context for function 'regularized_lsq_with_qr'
    module_type_store = module_type_store.open_function_context('regularized_lsq_with_qr', 20, 0, False)
    
    # Passed parameters checking function
    regularized_lsq_with_qr.stypy_localization = localization
    regularized_lsq_with_qr.stypy_type_of_self = None
    regularized_lsq_with_qr.stypy_type_store = module_type_store
    regularized_lsq_with_qr.stypy_function_name = 'regularized_lsq_with_qr'
    regularized_lsq_with_qr.stypy_param_names_list = ['m', 'n', 'R', 'QTb', 'perm', 'diag', 'copy_R']
    regularized_lsq_with_qr.stypy_varargs_param_name = None
    regularized_lsq_with_qr.stypy_kwargs_param_name = None
    regularized_lsq_with_qr.stypy_call_defaults = defaults
    regularized_lsq_with_qr.stypy_call_varargs = varargs
    regularized_lsq_with_qr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'regularized_lsq_with_qr', ['m', 'n', 'R', 'QTb', 'perm', 'diag', 'copy_R'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'regularized_lsq_with_qr', localization, ['m', 'n', 'R', 'QTb', 'perm', 'diag', 'copy_R'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'regularized_lsq_with_qr(...)' code ##################

    str_254727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'str', 'Solve regularized least squares using information from QR-decomposition.\n\n    The initial problem is to solve the following system in a least-squares\n    sense:\n    ::\n\n        A x = b\n        D x = 0\n\n    Where D is diagonal matrix. The method is based on QR decomposition\n    of the form A P = Q R, where P is a column permutation matrix, Q is an\n    orthogonal matrix and R is an upper triangular matrix.\n\n    Parameters\n    ----------\n    m, n : int\n        Initial shape of A.\n    R : ndarray, shape (n, n)\n        Upper triangular matrix from QR decomposition of A.\n    QTb : ndarray, shape (n,)\n        First n components of Q^T b.\n    perm : ndarray, shape (n,)\n        Array defining column permutation of A, such that i-th column of\n        P is perm[i]-th column of identity matrix.\n    diag : ndarray, shape (n,)\n        Array containing diagonal elements of D.\n\n    Returns\n    -------\n    x : ndarray, shape (n,)\n        Found least-squares solution.\n    ')
    
    # Getting the type of 'copy_R' (line 53)
    copy_R_254728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 7), 'copy_R')
    # Testing the type of an if condition (line 53)
    if_condition_254729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 4), copy_R_254728)
    # Assigning a type to the variable 'if_condition_254729' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'if_condition_254729', if_condition_254729)
    # SSA begins for if statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 54):
    
    # Assigning a Call to a Name (line 54):
    
    # Call to copy(...): (line 54)
    # Processing the call keyword arguments (line 54)
    kwargs_254732 = {}
    # Getting the type of 'R' (line 54)
    R_254730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'R', False)
    # Obtaining the member 'copy' of a type (line 54)
    copy_254731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), R_254730, 'copy')
    # Calling copy(args, kwargs) (line 54)
    copy_call_result_254733 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), copy_254731, *[], **kwargs_254732)
    
    # Assigning a type to the variable 'R' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'R', copy_call_result_254733)
    # SSA join for if statement (line 53)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to copy(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_254736 = {}
    # Getting the type of 'QTb' (line 55)
    QTb_254734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'QTb', False)
    # Obtaining the member 'copy' of a type (line 55)
    copy_254735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), QTb_254734, 'copy')
    # Calling copy(args, kwargs) (line 55)
    copy_call_result_254737 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), copy_254735, *[], **kwargs_254736)
    
    # Assigning a type to the variable 'v' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'v', copy_call_result_254737)
    
    # Call to givens_elimination(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'R' (line 57)
    R_254739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'R', False)
    # Getting the type of 'v' (line 57)
    v_254740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'v', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'perm' (line 57)
    perm_254741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'perm', False)
    # Getting the type of 'diag' (line 57)
    diag_254742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 'diag', False)
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___254743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 29), diag_254742, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_254744 = invoke(stypy.reporting.localization.Localization(__file__, 57, 29), getitem___254743, perm_254741)
    
    # Processing the call keyword arguments (line 57)
    kwargs_254745 = {}
    # Getting the type of 'givens_elimination' (line 57)
    givens_elimination_254738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'givens_elimination', False)
    # Calling givens_elimination(args, kwargs) (line 57)
    givens_elimination_call_result_254746 = invoke(stypy.reporting.localization.Localization(__file__, 57, 4), givens_elimination_254738, *[R_254739, v_254740, subscript_call_result_254744], **kwargs_254745)
    
    
    # Assigning a Call to a Name (line 59):
    
    # Assigning a Call to a Name (line 59):
    
    # Call to abs(...): (line 59)
    # Processing the call arguments (line 59)
    
    # Call to diag(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'R' (line 59)
    R_254751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'R', False)
    # Processing the call keyword arguments (line 59)
    kwargs_254752 = {}
    # Getting the type of 'np' (line 59)
    np_254749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'np', False)
    # Obtaining the member 'diag' of a type (line 59)
    diag_254750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 24), np_254749, 'diag')
    # Calling diag(args, kwargs) (line 59)
    diag_call_result_254753 = invoke(stypy.reporting.localization.Localization(__file__, 59, 24), diag_254750, *[R_254751], **kwargs_254752)
    
    # Processing the call keyword arguments (line 59)
    kwargs_254754 = {}
    # Getting the type of 'np' (line 59)
    np_254747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 17), 'np', False)
    # Obtaining the member 'abs' of a type (line 59)
    abs_254748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 17), np_254747, 'abs')
    # Calling abs(args, kwargs) (line 59)
    abs_call_result_254755 = invoke(stypy.reporting.localization.Localization(__file__, 59, 17), abs_254748, *[diag_call_result_254753], **kwargs_254754)
    
    # Assigning a type to the variable 'abs_diag_R' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'abs_diag_R', abs_call_result_254755)
    
    # Assigning a BinOp to a Name (line 60):
    
    # Assigning a BinOp to a Name (line 60):
    # Getting the type of 'EPS' (line 60)
    EPS_254756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'EPS')
    
    # Call to max(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'm' (line 60)
    m_254758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), 'm', False)
    # Getting the type of 'n' (line 60)
    n_254759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'n', False)
    # Processing the call keyword arguments (line 60)
    kwargs_254760 = {}
    # Getting the type of 'max' (line 60)
    max_254757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'max', False)
    # Calling max(args, kwargs) (line 60)
    max_call_result_254761 = invoke(stypy.reporting.localization.Localization(__file__, 60, 22), max_254757, *[m_254758, n_254759], **kwargs_254760)
    
    # Applying the binary operator '*' (line 60)
    result_mul_254762 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 16), '*', EPS_254756, max_call_result_254761)
    
    
    # Call to max(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'abs_diag_R' (line 60)
    abs_diag_R_254765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 41), 'abs_diag_R', False)
    # Processing the call keyword arguments (line 60)
    kwargs_254766 = {}
    # Getting the type of 'np' (line 60)
    np_254763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'np', False)
    # Obtaining the member 'max' of a type (line 60)
    max_254764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 34), np_254763, 'max')
    # Calling max(args, kwargs) (line 60)
    max_call_result_254767 = invoke(stypy.reporting.localization.Localization(__file__, 60, 34), max_254764, *[abs_diag_R_254765], **kwargs_254766)
    
    # Applying the binary operator '*' (line 60)
    result_mul_254768 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 32), '*', result_mul_254762, max_call_result_254767)
    
    # Assigning a type to the variable 'threshold' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'threshold', result_mul_254768)
    
    # Assigning a Call to a Tuple (line 61):
    
    # Assigning a Subscript to a Name (line 61):
    
    # Obtaining the type of the subscript
    int_254769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'int')
    
    # Call to nonzero(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Getting the type of 'abs_diag_R' (line 61)
    abs_diag_R_254772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 22), 'abs_diag_R', False)
    # Getting the type of 'threshold' (line 61)
    threshold_254773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 35), 'threshold', False)
    # Applying the binary operator '>' (line 61)
    result_gt_254774 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 22), '>', abs_diag_R_254772, threshold_254773)
    
    # Processing the call keyword arguments (line 61)
    kwargs_254775 = {}
    # Getting the type of 'np' (line 61)
    np_254770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 61)
    nonzero_254771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 11), np_254770, 'nonzero')
    # Calling nonzero(args, kwargs) (line 61)
    nonzero_call_result_254776 = invoke(stypy.reporting.localization.Localization(__file__, 61, 11), nonzero_254771, *[result_gt_254774], **kwargs_254775)
    
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___254777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 4), nonzero_call_result_254776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_254778 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), getitem___254777, int_254769)
    
    # Assigning a type to the variable 'tuple_var_assignment_254679' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_254679', subscript_call_result_254778)
    
    # Assigning a Name to a Name (line 61):
    # Getting the type of 'tuple_var_assignment_254679' (line 61)
    tuple_var_assignment_254679_254779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_254679')
    # Assigning a type to the variable 'nns' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'nns', tuple_var_assignment_254679_254779)
    
    # Assigning a Subscript to a Name (line 63):
    
    # Assigning a Subscript to a Name (line 63):
    
    # Obtaining the type of the subscript
    
    # Call to ix_(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'nns' (line 63)
    nns_254782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'nns', False)
    # Getting the type of 'nns' (line 63)
    nns_254783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'nns', False)
    # Processing the call keyword arguments (line 63)
    kwargs_254784 = {}
    # Getting the type of 'np' (line 63)
    np_254780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 10), 'np', False)
    # Obtaining the member 'ix_' of a type (line 63)
    ix__254781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 10), np_254780, 'ix_')
    # Calling ix_(args, kwargs) (line 63)
    ix__call_result_254785 = invoke(stypy.reporting.localization.Localization(__file__, 63, 10), ix__254781, *[nns_254782, nns_254783], **kwargs_254784)
    
    # Getting the type of 'R' (line 63)
    R_254786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'R')
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___254787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), R_254786, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_254788 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), getitem___254787, ix__call_result_254785)
    
    # Assigning a type to the variable 'R' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'R', subscript_call_result_254788)
    
    # Assigning a Subscript to a Name (line 64):
    
    # Assigning a Subscript to a Name (line 64):
    
    # Obtaining the type of the subscript
    # Getting the type of 'nns' (line 64)
    nns_254789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 10), 'nns')
    # Getting the type of 'v' (line 64)
    v_254790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'v')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___254791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), v_254790, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_254792 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___254791, nns_254789)
    
    # Assigning a type to the variable 'v' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'v', subscript_call_result_254792)
    
    # Assigning a Call to a Name (line 66):
    
    # Assigning a Call to a Name (line 66):
    
    # Call to zeros(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'n' (line 66)
    n_254795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'n', False)
    # Processing the call keyword arguments (line 66)
    kwargs_254796 = {}
    # Getting the type of 'np' (line 66)
    np_254793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 66)
    zeros_254794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), np_254793, 'zeros')
    # Calling zeros(args, kwargs) (line 66)
    zeros_call_result_254797 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), zeros_254794, *[n_254795], **kwargs_254796)
    
    # Assigning a type to the variable 'x' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'x', zeros_call_result_254797)
    
    # Assigning a Call to a Subscript (line 67):
    
    # Assigning a Call to a Subscript (line 67):
    
    # Call to solve_triangular(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'R' (line 67)
    R_254799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 36), 'R', False)
    # Getting the type of 'v' (line 67)
    v_254800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 39), 'v', False)
    # Processing the call keyword arguments (line 67)
    kwargs_254801 = {}
    # Getting the type of 'solve_triangular' (line 67)
    solve_triangular_254798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), 'solve_triangular', False)
    # Calling solve_triangular(args, kwargs) (line 67)
    solve_triangular_call_result_254802 = invoke(stypy.reporting.localization.Localization(__file__, 67, 19), solve_triangular_254798, *[R_254799, v_254800], **kwargs_254801)
    
    # Getting the type of 'x' (line 67)
    x_254803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'x')
    
    # Obtaining the type of the subscript
    # Getting the type of 'nns' (line 67)
    nns_254804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'nns')
    # Getting the type of 'perm' (line 67)
    perm_254805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 6), 'perm')
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___254806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 6), perm_254805, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_254807 = invoke(stypy.reporting.localization.Localization(__file__, 67, 6), getitem___254806, nns_254804)
    
    # Storing an element on a container (line 67)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 4), x_254803, (subscript_call_result_254807, solve_triangular_call_result_254802))
    # Getting the type of 'x' (line 69)
    x_254808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type', x_254808)
    
    # ################# End of 'regularized_lsq_with_qr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'regularized_lsq_with_qr' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_254809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_254809)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'regularized_lsq_with_qr'
    return stypy_return_type_254809

# Assigning a type to the variable 'regularized_lsq_with_qr' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'regularized_lsq_with_qr', regularized_lsq_with_qr)

@norecursion
def backtracking(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'backtracking'
    module_type_store = module_type_store.open_function_context('backtracking', 72, 0, False)
    
    # Passed parameters checking function
    backtracking.stypy_localization = localization
    backtracking.stypy_type_of_self = None
    backtracking.stypy_type_store = module_type_store
    backtracking.stypy_function_name = 'backtracking'
    backtracking.stypy_param_names_list = ['A', 'g', 'x', 'p', 'theta', 'p_dot_g', 'lb', 'ub']
    backtracking.stypy_varargs_param_name = None
    backtracking.stypy_kwargs_param_name = None
    backtracking.stypy_call_defaults = defaults
    backtracking.stypy_call_varargs = varargs
    backtracking.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'backtracking', ['A', 'g', 'x', 'p', 'theta', 'p_dot_g', 'lb', 'ub'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'backtracking', localization, ['A', 'g', 'x', 'p', 'theta', 'p_dot_g', 'lb', 'ub'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'backtracking(...)' code ##################

    str_254810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'str', 'Find an appropriate step size using backtracking line search.')
    
    # Assigning a Num to a Name (line 74):
    
    # Assigning a Num to a Name (line 74):
    int_254811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 12), 'int')
    # Assigning a type to the variable 'alpha' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'alpha', int_254811)
    
    # Getting the type of 'True' (line 75)
    True_254812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 10), 'True')
    # Testing the type of an if condition (line 75)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 4), True_254812)
    # SSA begins for while statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 76):
    
    # Assigning a Subscript to a Name (line 76):
    
    # Obtaining the type of the subscript
    int_254813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'int')
    
    # Call to reflective_transformation(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'x' (line 76)
    x_254815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 45), 'x', False)
    # Getting the type of 'alpha' (line 76)
    alpha_254816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 49), 'alpha', False)
    # Getting the type of 'p' (line 76)
    p_254817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 57), 'p', False)
    # Applying the binary operator '*' (line 76)
    result_mul_254818 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 49), '*', alpha_254816, p_254817)
    
    # Applying the binary operator '+' (line 76)
    result_add_254819 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 45), '+', x_254815, result_mul_254818)
    
    # Getting the type of 'lb' (line 76)
    lb_254820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 60), 'lb', False)
    # Getting the type of 'ub' (line 76)
    ub_254821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 64), 'ub', False)
    # Processing the call keyword arguments (line 76)
    kwargs_254822 = {}
    # Getting the type of 'reflective_transformation' (line 76)
    reflective_transformation_254814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 76)
    reflective_transformation_call_result_254823 = invoke(stypy.reporting.localization.Localization(__file__, 76, 19), reflective_transformation_254814, *[result_add_254819, lb_254820, ub_254821], **kwargs_254822)
    
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___254824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), reflective_transformation_call_result_254823, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_254825 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), getitem___254824, int_254813)
    
    # Assigning a type to the variable 'tuple_var_assignment_254680' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_254680', subscript_call_result_254825)
    
    # Assigning a Subscript to a Name (line 76):
    
    # Obtaining the type of the subscript
    int_254826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'int')
    
    # Call to reflective_transformation(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'x' (line 76)
    x_254828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 45), 'x', False)
    # Getting the type of 'alpha' (line 76)
    alpha_254829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 49), 'alpha', False)
    # Getting the type of 'p' (line 76)
    p_254830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 57), 'p', False)
    # Applying the binary operator '*' (line 76)
    result_mul_254831 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 49), '*', alpha_254829, p_254830)
    
    # Applying the binary operator '+' (line 76)
    result_add_254832 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 45), '+', x_254828, result_mul_254831)
    
    # Getting the type of 'lb' (line 76)
    lb_254833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 60), 'lb', False)
    # Getting the type of 'ub' (line 76)
    ub_254834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 64), 'ub', False)
    # Processing the call keyword arguments (line 76)
    kwargs_254835 = {}
    # Getting the type of 'reflective_transformation' (line 76)
    reflective_transformation_254827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 76)
    reflective_transformation_call_result_254836 = invoke(stypy.reporting.localization.Localization(__file__, 76, 19), reflective_transformation_254827, *[result_add_254832, lb_254833, ub_254834], **kwargs_254835)
    
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___254837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), reflective_transformation_call_result_254836, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_254838 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), getitem___254837, int_254826)
    
    # Assigning a type to the variable 'tuple_var_assignment_254681' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_254681', subscript_call_result_254838)
    
    # Assigning a Name to a Name (line 76):
    # Getting the type of 'tuple_var_assignment_254680' (line 76)
    tuple_var_assignment_254680_254839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_254680')
    # Assigning a type to the variable 'x_new' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'x_new', tuple_var_assignment_254680_254839)
    
    # Assigning a Name to a Name (line 76):
    # Getting the type of 'tuple_var_assignment_254681' (line 76)
    tuple_var_assignment_254681_254840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_254681')
    # Assigning a type to the variable '_' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), '_', tuple_var_assignment_254681_254840)
    
    # Assigning a BinOp to a Name (line 77):
    
    # Assigning a BinOp to a Name (line 77):
    # Getting the type of 'x_new' (line 77)
    x_new_254841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'x_new')
    # Getting the type of 'x' (line 77)
    x_254842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 23), 'x')
    # Applying the binary operator '-' (line 77)
    result_sub_254843 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 15), '-', x_new_254841, x_254842)
    
    # Assigning a type to the variable 'step' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'step', result_sub_254843)
    
    # Assigning a UnaryOp to a Name (line 78):
    
    # Assigning a UnaryOp to a Name (line 78):
    
    
    # Call to evaluate_quadratic(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'A' (line 78)
    A_254845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 42), 'A', False)
    # Getting the type of 'g' (line 78)
    g_254846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 45), 'g', False)
    # Getting the type of 'step' (line 78)
    step_254847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 48), 'step', False)
    # Processing the call keyword arguments (line 78)
    kwargs_254848 = {}
    # Getting the type of 'evaluate_quadratic' (line 78)
    evaluate_quadratic_254844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'evaluate_quadratic', False)
    # Calling evaluate_quadratic(args, kwargs) (line 78)
    evaluate_quadratic_call_result_254849 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), evaluate_quadratic_254844, *[A_254845, g_254846, step_254847], **kwargs_254848)
    
    # Applying the 'usub' unary operator (line 78)
    result___neg___254850 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 22), 'usub', evaluate_quadratic_call_result_254849)
    
    # Assigning a type to the variable 'cost_change' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'cost_change', result___neg___254850)
    
    
    # Getting the type of 'cost_change' (line 79)
    cost_change_254851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'cost_change')
    float_254852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 25), 'float')
    # Getting the type of 'alpha' (line 79)
    alpha_254853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 32), 'alpha')
    # Applying the binary operator '*' (line 79)
    result_mul_254854 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 25), '*', float_254852, alpha_254853)
    
    # Getting the type of 'p_dot_g' (line 79)
    p_dot_g_254855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 40), 'p_dot_g')
    # Applying the binary operator '*' (line 79)
    result_mul_254856 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 38), '*', result_mul_254854, p_dot_g_254855)
    
    # Applying the binary operator '>' (line 79)
    result_gt_254857 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 11), '>', cost_change_254851, result_mul_254856)
    
    # Testing the type of an if condition (line 79)
    if_condition_254858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), result_gt_254857)
    # Assigning a type to the variable 'if_condition_254858' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_254858', if_condition_254858)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'alpha' (line 81)
    alpha_254859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'alpha')
    float_254860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 17), 'float')
    # Applying the binary operator '*=' (line 81)
    result_imul_254861 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 8), '*=', alpha_254859, float_254860)
    # Assigning a type to the variable 'alpha' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'alpha', result_imul_254861)
    
    # SSA join for while statement (line 75)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to find_active_constraints(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'x_new' (line 83)
    x_new_254863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 37), 'x_new', False)
    # Getting the type of 'lb' (line 83)
    lb_254864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 44), 'lb', False)
    # Getting the type of 'ub' (line 83)
    ub_254865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 48), 'ub', False)
    # Processing the call keyword arguments (line 83)
    kwargs_254866 = {}
    # Getting the type of 'find_active_constraints' (line 83)
    find_active_constraints_254862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'find_active_constraints', False)
    # Calling find_active_constraints(args, kwargs) (line 83)
    find_active_constraints_call_result_254867 = invoke(stypy.reporting.localization.Localization(__file__, 83, 13), find_active_constraints_254862, *[x_new_254863, lb_254864, ub_254865], **kwargs_254866)
    
    # Assigning a type to the variable 'active' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'active', find_active_constraints_call_result_254867)
    
    
    # Call to any(...): (line 84)
    # Processing the call arguments (line 84)
    
    # Getting the type of 'active' (line 84)
    active_254870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'active', False)
    int_254871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 24), 'int')
    # Applying the binary operator '!=' (line 84)
    result_ne_254872 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 14), '!=', active_254870, int_254871)
    
    # Processing the call keyword arguments (line 84)
    kwargs_254873 = {}
    # Getting the type of 'np' (line 84)
    np_254868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 84)
    any_254869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 7), np_254868, 'any')
    # Calling any(args, kwargs) (line 84)
    any_call_result_254874 = invoke(stypy.reporting.localization.Localization(__file__, 84, 7), any_254869, *[result_ne_254872], **kwargs_254873)
    
    # Testing the type of an if condition (line 84)
    if_condition_254875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 4), any_call_result_254874)
    # Assigning a type to the variable 'if_condition_254875' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'if_condition_254875', if_condition_254875)
    # SSA begins for if statement (line 84)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 85):
    
    # Assigning a Subscript to a Name (line 85):
    
    # Obtaining the type of the subscript
    int_254876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
    
    # Call to reflective_transformation(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'x' (line 85)
    x_254878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 45), 'x', False)
    # Getting the type of 'theta' (line 85)
    theta_254879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 49), 'theta', False)
    # Getting the type of 'alpha' (line 85)
    alpha_254880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 57), 'alpha', False)
    # Applying the binary operator '*' (line 85)
    result_mul_254881 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 49), '*', theta_254879, alpha_254880)
    
    # Getting the type of 'p' (line 85)
    p_254882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 65), 'p', False)
    # Applying the binary operator '*' (line 85)
    result_mul_254883 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 63), '*', result_mul_254881, p_254882)
    
    # Applying the binary operator '+' (line 85)
    result_add_254884 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 45), '+', x_254878, result_mul_254883)
    
    # Getting the type of 'lb' (line 85)
    lb_254885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 68), 'lb', False)
    # Getting the type of 'ub' (line 85)
    ub_254886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 72), 'ub', False)
    # Processing the call keyword arguments (line 85)
    kwargs_254887 = {}
    # Getting the type of 'reflective_transformation' (line 85)
    reflective_transformation_254877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 19), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 85)
    reflective_transformation_call_result_254888 = invoke(stypy.reporting.localization.Localization(__file__, 85, 19), reflective_transformation_254877, *[result_add_254884, lb_254885, ub_254886], **kwargs_254887)
    
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___254889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), reflective_transformation_call_result_254888, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_254890 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), getitem___254889, int_254876)
    
    # Assigning a type to the variable 'tuple_var_assignment_254682' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_254682', subscript_call_result_254890)
    
    # Assigning a Subscript to a Name (line 85):
    
    # Obtaining the type of the subscript
    int_254891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
    
    # Call to reflective_transformation(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'x' (line 85)
    x_254893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 45), 'x', False)
    # Getting the type of 'theta' (line 85)
    theta_254894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 49), 'theta', False)
    # Getting the type of 'alpha' (line 85)
    alpha_254895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 57), 'alpha', False)
    # Applying the binary operator '*' (line 85)
    result_mul_254896 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 49), '*', theta_254894, alpha_254895)
    
    # Getting the type of 'p' (line 85)
    p_254897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 65), 'p', False)
    # Applying the binary operator '*' (line 85)
    result_mul_254898 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 63), '*', result_mul_254896, p_254897)
    
    # Applying the binary operator '+' (line 85)
    result_add_254899 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 45), '+', x_254893, result_mul_254898)
    
    # Getting the type of 'lb' (line 85)
    lb_254900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 68), 'lb', False)
    # Getting the type of 'ub' (line 85)
    ub_254901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 72), 'ub', False)
    # Processing the call keyword arguments (line 85)
    kwargs_254902 = {}
    # Getting the type of 'reflective_transformation' (line 85)
    reflective_transformation_254892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 19), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 85)
    reflective_transformation_call_result_254903 = invoke(stypy.reporting.localization.Localization(__file__, 85, 19), reflective_transformation_254892, *[result_add_254899, lb_254900, ub_254901], **kwargs_254902)
    
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___254904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), reflective_transformation_call_result_254903, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_254905 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), getitem___254904, int_254891)
    
    # Assigning a type to the variable 'tuple_var_assignment_254683' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_254683', subscript_call_result_254905)
    
    # Assigning a Name to a Name (line 85):
    # Getting the type of 'tuple_var_assignment_254682' (line 85)
    tuple_var_assignment_254682_254906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_254682')
    # Assigning a type to the variable 'x_new' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'x_new', tuple_var_assignment_254682_254906)
    
    # Assigning a Name to a Name (line 85):
    # Getting the type of 'tuple_var_assignment_254683' (line 85)
    tuple_var_assignment_254683_254907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'tuple_var_assignment_254683')
    # Assigning a type to the variable '_' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), '_', tuple_var_assignment_254683_254907)
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to make_strictly_feasible(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'x_new' (line 86)
    x_new_254909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 39), 'x_new', False)
    # Getting the type of 'lb' (line 86)
    lb_254910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 46), 'lb', False)
    # Getting the type of 'ub' (line 86)
    ub_254911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 50), 'ub', False)
    # Processing the call keyword arguments (line 86)
    int_254912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 60), 'int')
    keyword_254913 = int_254912
    kwargs_254914 = {'rstep': keyword_254913}
    # Getting the type of 'make_strictly_feasible' (line 86)
    make_strictly_feasible_254908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'make_strictly_feasible', False)
    # Calling make_strictly_feasible(args, kwargs) (line 86)
    make_strictly_feasible_call_result_254915 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), make_strictly_feasible_254908, *[x_new_254909, lb_254910, ub_254911], **kwargs_254914)
    
    # Assigning a type to the variable 'x_new' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'x_new', make_strictly_feasible_call_result_254915)
    
    # Assigning a BinOp to a Name (line 87):
    
    # Assigning a BinOp to a Name (line 87):
    # Getting the type of 'x_new' (line 87)
    x_new_254916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'x_new')
    # Getting the type of 'x' (line 87)
    x_254917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'x')
    # Applying the binary operator '-' (line 87)
    result_sub_254918 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 15), '-', x_new_254916, x_254917)
    
    # Assigning a type to the variable 'step' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'step', result_sub_254918)
    
    # Assigning a UnaryOp to a Name (line 88):
    
    # Assigning a UnaryOp to a Name (line 88):
    
    
    # Call to evaluate_quadratic(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'A' (line 88)
    A_254920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 42), 'A', False)
    # Getting the type of 'g' (line 88)
    g_254921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 45), 'g', False)
    # Getting the type of 'step' (line 88)
    step_254922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 48), 'step', False)
    # Processing the call keyword arguments (line 88)
    kwargs_254923 = {}
    # Getting the type of 'evaluate_quadratic' (line 88)
    evaluate_quadratic_254919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'evaluate_quadratic', False)
    # Calling evaluate_quadratic(args, kwargs) (line 88)
    evaluate_quadratic_call_result_254924 = invoke(stypy.reporting.localization.Localization(__file__, 88, 23), evaluate_quadratic_254919, *[A_254920, g_254921, step_254922], **kwargs_254923)
    
    # Applying the 'usub' unary operator (line 88)
    result___neg___254925 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 22), 'usub', evaluate_quadratic_call_result_254924)
    
    # Assigning a type to the variable 'cost_change' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'cost_change', result___neg___254925)
    # SSA join for if statement (line 84)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 90)
    tuple_254926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 90)
    # Adding element type (line 90)
    # Getting the type of 'x' (line 90)
    x_254927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 11), tuple_254926, x_254927)
    # Adding element type (line 90)
    # Getting the type of 'step' (line 90)
    step_254928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 14), 'step')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 11), tuple_254926, step_254928)
    # Adding element type (line 90)
    # Getting the type of 'cost_change' (line 90)
    cost_change_254929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'cost_change')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 11), tuple_254926, cost_change_254929)
    
    # Assigning a type to the variable 'stypy_return_type' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type', tuple_254926)
    
    # ################# End of 'backtracking(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'backtracking' in the type store
    # Getting the type of 'stypy_return_type' (line 72)
    stypy_return_type_254930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_254930)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'backtracking'
    return stypy_return_type_254930

# Assigning a type to the variable 'backtracking' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'backtracking', backtracking)

@norecursion
def select_step(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'select_step'
    module_type_store = module_type_store.open_function_context('select_step', 93, 0, False)
    
    # Passed parameters checking function
    select_step.stypy_localization = localization
    select_step.stypy_type_of_self = None
    select_step.stypy_type_store = module_type_store
    select_step.stypy_function_name = 'select_step'
    select_step.stypy_param_names_list = ['x', 'A_h', 'g_h', 'c_h', 'p', 'p_h', 'd', 'lb', 'ub', 'theta']
    select_step.stypy_varargs_param_name = None
    select_step.stypy_kwargs_param_name = None
    select_step.stypy_call_defaults = defaults
    select_step.stypy_call_varargs = varargs
    select_step.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'select_step', ['x', 'A_h', 'g_h', 'c_h', 'p', 'p_h', 'd', 'lb', 'ub', 'theta'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'select_step', localization, ['x', 'A_h', 'g_h', 'c_h', 'p', 'p_h', 'd', 'lb', 'ub', 'theta'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'select_step(...)' code ##################

    str_254931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 4), 'str', 'Select the best step according to Trust Region Reflective algorithm.')
    
    
    # Call to in_bounds(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'x' (line 95)
    x_254933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 17), 'x', False)
    # Getting the type of 'p' (line 95)
    p_254934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'p', False)
    # Applying the binary operator '+' (line 95)
    result_add_254935 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 17), '+', x_254933, p_254934)
    
    # Getting the type of 'lb' (line 95)
    lb_254936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'lb', False)
    # Getting the type of 'ub' (line 95)
    ub_254937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'ub', False)
    # Processing the call keyword arguments (line 95)
    kwargs_254938 = {}
    # Getting the type of 'in_bounds' (line 95)
    in_bounds_254932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 7), 'in_bounds', False)
    # Calling in_bounds(args, kwargs) (line 95)
    in_bounds_call_result_254939 = invoke(stypy.reporting.localization.Localization(__file__, 95, 7), in_bounds_254932, *[result_add_254935, lb_254936, ub_254937], **kwargs_254938)
    
    # Testing the type of an if condition (line 95)
    if_condition_254940 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 4), in_bounds_call_result_254939)
    # Assigning a type to the variable 'if_condition_254940' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'if_condition_254940', if_condition_254940)
    # SSA begins for if statement (line 95)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'p' (line 96)
    p_254941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'p')
    # Assigning a type to the variable 'stypy_return_type' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'stypy_return_type', p_254941)
    # SSA join for if statement (line 95)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 98):
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_254942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'x' (line 98)
    x_254944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 40), 'x', False)
    # Getting the type of 'p' (line 98)
    p_254945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 43), 'p', False)
    # Getting the type of 'lb' (line 98)
    lb_254946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 46), 'lb', False)
    # Getting the type of 'ub' (line 98)
    ub_254947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 50), 'ub', False)
    # Processing the call keyword arguments (line 98)
    kwargs_254948 = {}
    # Getting the type of 'step_size_to_bound' (line 98)
    step_size_to_bound_254943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 98)
    step_size_to_bound_call_result_254949 = invoke(stypy.reporting.localization.Localization(__file__, 98, 21), step_size_to_bound_254943, *[x_254944, p_254945, lb_254946, ub_254947], **kwargs_254948)
    
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___254950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), step_size_to_bound_call_result_254949, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_254951 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), getitem___254950, int_254942)
    
    # Assigning a type to the variable 'tuple_var_assignment_254684' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_254684', subscript_call_result_254951)
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_254952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'x' (line 98)
    x_254954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 40), 'x', False)
    # Getting the type of 'p' (line 98)
    p_254955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 43), 'p', False)
    # Getting the type of 'lb' (line 98)
    lb_254956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 46), 'lb', False)
    # Getting the type of 'ub' (line 98)
    ub_254957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 50), 'ub', False)
    # Processing the call keyword arguments (line 98)
    kwargs_254958 = {}
    # Getting the type of 'step_size_to_bound' (line 98)
    step_size_to_bound_254953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 98)
    step_size_to_bound_call_result_254959 = invoke(stypy.reporting.localization.Localization(__file__, 98, 21), step_size_to_bound_254953, *[x_254954, p_254955, lb_254956, ub_254957], **kwargs_254958)
    
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___254960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), step_size_to_bound_call_result_254959, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_254961 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), getitem___254960, int_254952)
    
    # Assigning a type to the variable 'tuple_var_assignment_254685' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_254685', subscript_call_result_254961)
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'tuple_var_assignment_254684' (line 98)
    tuple_var_assignment_254684_254962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_254684')
    # Assigning a type to the variable 'p_stride' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'p_stride', tuple_var_assignment_254684_254962)
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'tuple_var_assignment_254685' (line 98)
    tuple_var_assignment_254685_254963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_254685')
    # Assigning a type to the variable 'hits' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'hits', tuple_var_assignment_254685_254963)
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to copy(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'p_h' (line 99)
    p_h_254966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'p_h', False)
    # Processing the call keyword arguments (line 99)
    kwargs_254967 = {}
    # Getting the type of 'np' (line 99)
    np_254964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 10), 'np', False)
    # Obtaining the member 'copy' of a type (line 99)
    copy_254965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 10), np_254964, 'copy')
    # Calling copy(args, kwargs) (line 99)
    copy_call_result_254968 = invoke(stypy.reporting.localization.Localization(__file__, 99, 10), copy_254965, *[p_h_254966], **kwargs_254967)
    
    # Assigning a type to the variable 'r_h' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'r_h', copy_call_result_254968)
    
    # Getting the type of 'r_h' (line 100)
    r_h_254969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'r_h')
    
    # Obtaining the type of the subscript
    
    # Call to astype(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'bool' (line 100)
    bool_254972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'bool', False)
    # Processing the call keyword arguments (line 100)
    kwargs_254973 = {}
    # Getting the type of 'hits' (line 100)
    hits_254970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'hits', False)
    # Obtaining the member 'astype' of a type (line 100)
    astype_254971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), hits_254970, 'astype')
    # Calling astype(args, kwargs) (line 100)
    astype_call_result_254974 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), astype_254971, *[bool_254972], **kwargs_254973)
    
    # Getting the type of 'r_h' (line 100)
    r_h_254975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'r_h')
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___254976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 4), r_h_254975, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_254977 = invoke(stypy.reporting.localization.Localization(__file__, 100, 4), getitem___254976, astype_call_result_254974)
    
    int_254978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 30), 'int')
    # Applying the binary operator '*=' (line 100)
    result_imul_254979 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 4), '*=', subscript_call_result_254977, int_254978)
    # Getting the type of 'r_h' (line 100)
    r_h_254980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'r_h')
    
    # Call to astype(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'bool' (line 100)
    bool_254983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'bool', False)
    # Processing the call keyword arguments (line 100)
    kwargs_254984 = {}
    # Getting the type of 'hits' (line 100)
    hits_254981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'hits', False)
    # Obtaining the member 'astype' of a type (line 100)
    astype_254982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), hits_254981, 'astype')
    # Calling astype(args, kwargs) (line 100)
    astype_call_result_254985 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), astype_254982, *[bool_254983], **kwargs_254984)
    
    # Storing an element on a container (line 100)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 4), r_h_254980, (astype_call_result_254985, result_imul_254979))
    
    
    # Assigning a BinOp to a Name (line 101):
    
    # Assigning a BinOp to a Name (line 101):
    # Getting the type of 'd' (line 101)
    d_254986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'd')
    # Getting the type of 'r_h' (line 101)
    r_h_254987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'r_h')
    # Applying the binary operator '*' (line 101)
    result_mul_254988 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 8), '*', d_254986, r_h_254987)
    
    # Assigning a type to the variable 'r' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'r', result_mul_254988)
    
    # Getting the type of 'p' (line 104)
    p_254989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'p')
    # Getting the type of 'p_stride' (line 104)
    p_stride_254990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 9), 'p_stride')
    # Applying the binary operator '*=' (line 104)
    result_imul_254991 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 4), '*=', p_254989, p_stride_254990)
    # Assigning a type to the variable 'p' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'p', result_imul_254991)
    
    
    # Getting the type of 'p_h' (line 105)
    p_h_254992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'p_h')
    # Getting the type of 'p_stride' (line 105)
    p_stride_254993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'p_stride')
    # Applying the binary operator '*=' (line 105)
    result_imul_254994 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 4), '*=', p_h_254992, p_stride_254993)
    # Assigning a type to the variable 'p_h' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'p_h', result_imul_254994)
    
    
    # Assigning a BinOp to a Name (line 106):
    
    # Assigning a BinOp to a Name (line 106):
    # Getting the type of 'x' (line 106)
    x_254995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'x')
    # Getting the type of 'p' (line 106)
    p_254996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'p')
    # Applying the binary operator '+' (line 106)
    result_add_254997 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 17), '+', x_254995, p_254996)
    
    # Assigning a type to the variable 'x_on_bound' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'x_on_bound', result_add_254997)
    
    # Assigning a Call to a Tuple (line 109):
    
    # Assigning a Subscript to a Name (line 109):
    
    # Obtaining the type of the subscript
    int_254998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'x_on_bound' (line 109)
    x_on_bound_255000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 39), 'x_on_bound', False)
    # Getting the type of 'r' (line 109)
    r_255001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 51), 'r', False)
    # Getting the type of 'lb' (line 109)
    lb_255002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 54), 'lb', False)
    # Getting the type of 'ub' (line 109)
    ub_255003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 58), 'ub', False)
    # Processing the call keyword arguments (line 109)
    kwargs_255004 = {}
    # Getting the type of 'step_size_to_bound' (line 109)
    step_size_to_bound_254999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 109)
    step_size_to_bound_call_result_255005 = invoke(stypy.reporting.localization.Localization(__file__, 109, 20), step_size_to_bound_254999, *[x_on_bound_255000, r_255001, lb_255002, ub_255003], **kwargs_255004)
    
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___255006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 4), step_size_to_bound_call_result_255005, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_255007 = invoke(stypy.reporting.localization.Localization(__file__, 109, 4), getitem___255006, int_254998)
    
    # Assigning a type to the variable 'tuple_var_assignment_254686' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'tuple_var_assignment_254686', subscript_call_result_255007)
    
    # Assigning a Subscript to a Name (line 109):
    
    # Obtaining the type of the subscript
    int_255008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'x_on_bound' (line 109)
    x_on_bound_255010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 39), 'x_on_bound', False)
    # Getting the type of 'r' (line 109)
    r_255011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 51), 'r', False)
    # Getting the type of 'lb' (line 109)
    lb_255012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 54), 'lb', False)
    # Getting the type of 'ub' (line 109)
    ub_255013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 58), 'ub', False)
    # Processing the call keyword arguments (line 109)
    kwargs_255014 = {}
    # Getting the type of 'step_size_to_bound' (line 109)
    step_size_to_bound_255009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 109)
    step_size_to_bound_call_result_255015 = invoke(stypy.reporting.localization.Localization(__file__, 109, 20), step_size_to_bound_255009, *[x_on_bound_255010, r_255011, lb_255012, ub_255013], **kwargs_255014)
    
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___255016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 4), step_size_to_bound_call_result_255015, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_255017 = invoke(stypy.reporting.localization.Localization(__file__, 109, 4), getitem___255016, int_255008)
    
    # Assigning a type to the variable 'tuple_var_assignment_254687' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'tuple_var_assignment_254687', subscript_call_result_255017)
    
    # Assigning a Name to a Name (line 109):
    # Getting the type of 'tuple_var_assignment_254686' (line 109)
    tuple_var_assignment_254686_255018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'tuple_var_assignment_254686')
    # Assigning a type to the variable 'r_stride_u' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'r_stride_u', tuple_var_assignment_254686_255018)
    
    # Assigning a Name to a Name (line 109):
    # Getting the type of 'tuple_var_assignment_254687' (line 109)
    tuple_var_assignment_254687_255019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'tuple_var_assignment_254687')
    # Assigning a type to the variable '_' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), '_', tuple_var_assignment_254687_255019)
    
    # Assigning a BinOp to a Name (line 112):
    
    # Assigning a BinOp to a Name (line 112):
    int_255020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 18), 'int')
    # Getting the type of 'theta' (line 112)
    theta_255021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'theta')
    # Applying the binary operator '-' (line 112)
    result_sub_255022 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 18), '-', int_255020, theta_255021)
    
    # Getting the type of 'r_stride_u' (line 112)
    r_stride_u_255023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'r_stride_u')
    # Applying the binary operator '*' (line 112)
    result_mul_255024 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 17), '*', result_sub_255022, r_stride_u_255023)
    
    # Assigning a type to the variable 'r_stride_l' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'r_stride_l', result_mul_255024)
    
    # Getting the type of 'r_stride_u' (line 113)
    r_stride_u_255025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'r_stride_u')
    # Getting the type of 'theta' (line 113)
    theta_255026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'theta')
    # Applying the binary operator '*=' (line 113)
    result_imul_255027 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 4), '*=', r_stride_u_255025, theta_255026)
    # Assigning a type to the variable 'r_stride_u' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'r_stride_u', result_imul_255027)
    
    
    
    # Getting the type of 'r_stride_u' (line 115)
    r_stride_u_255028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 7), 'r_stride_u')
    int_255029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 20), 'int')
    # Applying the binary operator '>' (line 115)
    result_gt_255030 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 7), '>', r_stride_u_255028, int_255029)
    
    # Testing the type of an if condition (line 115)
    if_condition_255031 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 4), result_gt_255030)
    # Assigning a type to the variable 'if_condition_255031' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'if_condition_255031', if_condition_255031)
    # SSA begins for if statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 116):
    
    # Assigning a Subscript to a Name (line 116):
    
    # Obtaining the type of the subscript
    int_255032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 8), 'int')
    
    # Call to build_quadratic_1d(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'A_h' (line 116)
    A_h_255034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'A_h', False)
    # Getting the type of 'g_h' (line 116)
    g_h_255035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 42), 'g_h', False)
    # Getting the type of 'r_h' (line 116)
    r_h_255036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 47), 'r_h', False)
    # Processing the call keyword arguments (line 116)
    # Getting the type of 'p_h' (line 116)
    p_h_255037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 55), 'p_h', False)
    keyword_255038 = p_h_255037
    # Getting the type of 'c_h' (line 116)
    c_h_255039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 65), 'c_h', False)
    keyword_255040 = c_h_255039
    kwargs_255041 = {'diag': keyword_255040, 's0': keyword_255038}
    # Getting the type of 'build_quadratic_1d' (line 116)
    build_quadratic_1d_255033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 116)
    build_quadratic_1d_call_result_255042 = invoke(stypy.reporting.localization.Localization(__file__, 116, 18), build_quadratic_1d_255033, *[A_h_255034, g_h_255035, r_h_255036], **kwargs_255041)
    
    # Obtaining the member '__getitem__' of a type (line 116)
    getitem___255043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), build_quadratic_1d_call_result_255042, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
    subscript_call_result_255044 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), getitem___255043, int_255032)
    
    # Assigning a type to the variable 'tuple_var_assignment_254688' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'tuple_var_assignment_254688', subscript_call_result_255044)
    
    # Assigning a Subscript to a Name (line 116):
    
    # Obtaining the type of the subscript
    int_255045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 8), 'int')
    
    # Call to build_quadratic_1d(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'A_h' (line 116)
    A_h_255047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'A_h', False)
    # Getting the type of 'g_h' (line 116)
    g_h_255048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 42), 'g_h', False)
    # Getting the type of 'r_h' (line 116)
    r_h_255049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 47), 'r_h', False)
    # Processing the call keyword arguments (line 116)
    # Getting the type of 'p_h' (line 116)
    p_h_255050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 55), 'p_h', False)
    keyword_255051 = p_h_255050
    # Getting the type of 'c_h' (line 116)
    c_h_255052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 65), 'c_h', False)
    keyword_255053 = c_h_255052
    kwargs_255054 = {'diag': keyword_255053, 's0': keyword_255051}
    # Getting the type of 'build_quadratic_1d' (line 116)
    build_quadratic_1d_255046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 116)
    build_quadratic_1d_call_result_255055 = invoke(stypy.reporting.localization.Localization(__file__, 116, 18), build_quadratic_1d_255046, *[A_h_255047, g_h_255048, r_h_255049], **kwargs_255054)
    
    # Obtaining the member '__getitem__' of a type (line 116)
    getitem___255056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), build_quadratic_1d_call_result_255055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
    subscript_call_result_255057 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), getitem___255056, int_255045)
    
    # Assigning a type to the variable 'tuple_var_assignment_254689' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'tuple_var_assignment_254689', subscript_call_result_255057)
    
    # Assigning a Subscript to a Name (line 116):
    
    # Obtaining the type of the subscript
    int_255058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 8), 'int')
    
    # Call to build_quadratic_1d(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'A_h' (line 116)
    A_h_255060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'A_h', False)
    # Getting the type of 'g_h' (line 116)
    g_h_255061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 42), 'g_h', False)
    # Getting the type of 'r_h' (line 116)
    r_h_255062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 47), 'r_h', False)
    # Processing the call keyword arguments (line 116)
    # Getting the type of 'p_h' (line 116)
    p_h_255063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 55), 'p_h', False)
    keyword_255064 = p_h_255063
    # Getting the type of 'c_h' (line 116)
    c_h_255065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 65), 'c_h', False)
    keyword_255066 = c_h_255065
    kwargs_255067 = {'diag': keyword_255066, 's0': keyword_255064}
    # Getting the type of 'build_quadratic_1d' (line 116)
    build_quadratic_1d_255059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 116)
    build_quadratic_1d_call_result_255068 = invoke(stypy.reporting.localization.Localization(__file__, 116, 18), build_quadratic_1d_255059, *[A_h_255060, g_h_255061, r_h_255062], **kwargs_255067)
    
    # Obtaining the member '__getitem__' of a type (line 116)
    getitem___255069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), build_quadratic_1d_call_result_255068, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
    subscript_call_result_255070 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), getitem___255069, int_255058)
    
    # Assigning a type to the variable 'tuple_var_assignment_254690' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'tuple_var_assignment_254690', subscript_call_result_255070)
    
    # Assigning a Name to a Name (line 116):
    # Getting the type of 'tuple_var_assignment_254688' (line 116)
    tuple_var_assignment_254688_255071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'tuple_var_assignment_254688')
    # Assigning a type to the variable 'a' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'a', tuple_var_assignment_254688_255071)
    
    # Assigning a Name to a Name (line 116):
    # Getting the type of 'tuple_var_assignment_254689' (line 116)
    tuple_var_assignment_254689_255072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'tuple_var_assignment_254689')
    # Assigning a type to the variable 'b' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'b', tuple_var_assignment_254689_255072)
    
    # Assigning a Name to a Name (line 116):
    # Getting the type of 'tuple_var_assignment_254690' (line 116)
    tuple_var_assignment_254690_255073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'tuple_var_assignment_254690')
    # Assigning a type to the variable 'c' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 14), 'c', tuple_var_assignment_254690_255073)
    
    # Assigning a Call to a Tuple (line 117):
    
    # Assigning a Subscript to a Name (line 117):
    
    # Obtaining the type of the subscript
    int_255074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 8), 'int')
    
    # Call to minimize_quadratic_1d(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'a' (line 118)
    a_255076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'a', False)
    # Getting the type of 'b' (line 118)
    b_255077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'b', False)
    # Getting the type of 'r_stride_l' (line 118)
    r_stride_l_255078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 18), 'r_stride_l', False)
    # Getting the type of 'r_stride_u' (line 118)
    r_stride_u_255079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 30), 'r_stride_u', False)
    # Processing the call keyword arguments (line 117)
    # Getting the type of 'c' (line 118)
    c_255080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 44), 'c', False)
    keyword_255081 = c_255080
    kwargs_255082 = {'c': keyword_255081}
    # Getting the type of 'minimize_quadratic_1d' (line 117)
    minimize_quadratic_1d_255075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 28), 'minimize_quadratic_1d', False)
    # Calling minimize_quadratic_1d(args, kwargs) (line 117)
    minimize_quadratic_1d_call_result_255083 = invoke(stypy.reporting.localization.Localization(__file__, 117, 28), minimize_quadratic_1d_255075, *[a_255076, b_255077, r_stride_l_255078, r_stride_u_255079], **kwargs_255082)
    
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___255084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), minimize_quadratic_1d_call_result_255083, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_255085 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), getitem___255084, int_255074)
    
    # Assigning a type to the variable 'tuple_var_assignment_254691' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_254691', subscript_call_result_255085)
    
    # Assigning a Subscript to a Name (line 117):
    
    # Obtaining the type of the subscript
    int_255086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 8), 'int')
    
    # Call to minimize_quadratic_1d(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'a' (line 118)
    a_255088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'a', False)
    # Getting the type of 'b' (line 118)
    b_255089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'b', False)
    # Getting the type of 'r_stride_l' (line 118)
    r_stride_l_255090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 18), 'r_stride_l', False)
    # Getting the type of 'r_stride_u' (line 118)
    r_stride_u_255091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 30), 'r_stride_u', False)
    # Processing the call keyword arguments (line 117)
    # Getting the type of 'c' (line 118)
    c_255092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 44), 'c', False)
    keyword_255093 = c_255092
    kwargs_255094 = {'c': keyword_255093}
    # Getting the type of 'minimize_quadratic_1d' (line 117)
    minimize_quadratic_1d_255087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 28), 'minimize_quadratic_1d', False)
    # Calling minimize_quadratic_1d(args, kwargs) (line 117)
    minimize_quadratic_1d_call_result_255095 = invoke(stypy.reporting.localization.Localization(__file__, 117, 28), minimize_quadratic_1d_255087, *[a_255088, b_255089, r_stride_l_255090, r_stride_u_255091], **kwargs_255094)
    
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___255096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), minimize_quadratic_1d_call_result_255095, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_255097 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), getitem___255096, int_255086)
    
    # Assigning a type to the variable 'tuple_var_assignment_254692' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_254692', subscript_call_result_255097)
    
    # Assigning a Name to a Name (line 117):
    # Getting the type of 'tuple_var_assignment_254691' (line 117)
    tuple_var_assignment_254691_255098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_254691')
    # Assigning a type to the variable 'r_stride' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'r_stride', tuple_var_assignment_254691_255098)
    
    # Assigning a Name to a Name (line 117):
    # Getting the type of 'tuple_var_assignment_254692' (line 117)
    tuple_var_assignment_254692_255099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_254692')
    # Assigning a type to the variable 'r_value' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'r_value', tuple_var_assignment_254692_255099)
    
    # Assigning a BinOp to a Name (line 119):
    
    # Assigning a BinOp to a Name (line 119):
    # Getting the type of 'p_h' (line 119)
    p_h_255100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 14), 'p_h')
    # Getting the type of 'r_h' (line 119)
    r_h_255101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'r_h')
    # Getting the type of 'r_stride' (line 119)
    r_stride_255102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'r_stride')
    # Applying the binary operator '*' (line 119)
    result_mul_255103 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 20), '*', r_h_255101, r_stride_255102)
    
    # Applying the binary operator '+' (line 119)
    result_add_255104 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 14), '+', p_h_255100, result_mul_255103)
    
    # Assigning a type to the variable 'r_h' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'r_h', result_add_255104)
    
    # Assigning a BinOp to a Name (line 120):
    
    # Assigning a BinOp to a Name (line 120):
    # Getting the type of 'd' (line 120)
    d_255105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'd')
    # Getting the type of 'r_h' (line 120)
    r_h_255106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'r_h')
    # Applying the binary operator '*' (line 120)
    result_mul_255107 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 12), '*', d_255105, r_h_255106)
    
    # Assigning a type to the variable 'r' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'r', result_mul_255107)
    # SSA branch for the else part of an if statement (line 115)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 122):
    
    # Assigning a Attribute to a Name (line 122):
    # Getting the type of 'np' (line 122)
    np_255108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 18), 'np')
    # Obtaining the member 'inf' of a type (line 122)
    inf_255109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 18), np_255108, 'inf')
    # Assigning a type to the variable 'r_value' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'r_value', inf_255109)
    # SSA join for if statement (line 115)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'p_h' (line 125)
    p_h_255110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'p_h')
    # Getting the type of 'theta' (line 125)
    theta_255111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'theta')
    # Applying the binary operator '*=' (line 125)
    result_imul_255112 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 4), '*=', p_h_255110, theta_255111)
    # Assigning a type to the variable 'p_h' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'p_h', result_imul_255112)
    
    
    # Getting the type of 'p' (line 126)
    p_255113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'p')
    # Getting the type of 'theta' (line 126)
    theta_255114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 9), 'theta')
    # Applying the binary operator '*=' (line 126)
    result_imul_255115 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 4), '*=', p_255113, theta_255114)
    # Assigning a type to the variable 'p' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'p', result_imul_255115)
    
    
    # Assigning a Call to a Name (line 127):
    
    # Assigning a Call to a Name (line 127):
    
    # Call to evaluate_quadratic(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'A_h' (line 127)
    A_h_255117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 33), 'A_h', False)
    # Getting the type of 'g_h' (line 127)
    g_h_255118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 38), 'g_h', False)
    # Getting the type of 'p_h' (line 127)
    p_h_255119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 43), 'p_h', False)
    # Processing the call keyword arguments (line 127)
    # Getting the type of 'c_h' (line 127)
    c_h_255120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 53), 'c_h', False)
    keyword_255121 = c_h_255120
    kwargs_255122 = {'diag': keyword_255121}
    # Getting the type of 'evaluate_quadratic' (line 127)
    evaluate_quadratic_255116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 14), 'evaluate_quadratic', False)
    # Calling evaluate_quadratic(args, kwargs) (line 127)
    evaluate_quadratic_call_result_255123 = invoke(stypy.reporting.localization.Localization(__file__, 127, 14), evaluate_quadratic_255116, *[A_h_255117, g_h_255118, p_h_255119], **kwargs_255122)
    
    # Assigning a type to the variable 'p_value' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'p_value', evaluate_quadratic_call_result_255123)
    
    # Assigning a UnaryOp to a Name (line 129):
    
    # Assigning a UnaryOp to a Name (line 129):
    
    # Getting the type of 'g_h' (line 129)
    g_h_255124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'g_h')
    # Applying the 'usub' unary operator (line 129)
    result___neg___255125 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 11), 'usub', g_h_255124)
    
    # Assigning a type to the variable 'ag_h' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'ag_h', result___neg___255125)
    
    # Assigning a BinOp to a Name (line 130):
    
    # Assigning a BinOp to a Name (line 130):
    # Getting the type of 'd' (line 130)
    d_255126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 9), 'd')
    # Getting the type of 'ag_h' (line 130)
    ag_h_255127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'ag_h')
    # Applying the binary operator '*' (line 130)
    result_mul_255128 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 9), '*', d_255126, ag_h_255127)
    
    # Assigning a type to the variable 'ag' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'ag', result_mul_255128)
    
    # Assigning a Call to a Tuple (line 131):
    
    # Assigning a Subscript to a Name (line 131):
    
    # Obtaining the type of the subscript
    int_255129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'x' (line 131)
    x_255131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 40), 'x', False)
    # Getting the type of 'ag' (line 131)
    ag_255132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 43), 'ag', False)
    # Getting the type of 'lb' (line 131)
    lb_255133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 47), 'lb', False)
    # Getting the type of 'ub' (line 131)
    ub_255134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 51), 'ub', False)
    # Processing the call keyword arguments (line 131)
    kwargs_255135 = {}
    # Getting the type of 'step_size_to_bound' (line 131)
    step_size_to_bound_255130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 131)
    step_size_to_bound_call_result_255136 = invoke(stypy.reporting.localization.Localization(__file__, 131, 21), step_size_to_bound_255130, *[x_255131, ag_255132, lb_255133, ub_255134], **kwargs_255135)
    
    # Obtaining the member '__getitem__' of a type (line 131)
    getitem___255137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 4), step_size_to_bound_call_result_255136, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 131)
    subscript_call_result_255138 = invoke(stypy.reporting.localization.Localization(__file__, 131, 4), getitem___255137, int_255129)
    
    # Assigning a type to the variable 'tuple_var_assignment_254693' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_var_assignment_254693', subscript_call_result_255138)
    
    # Assigning a Subscript to a Name (line 131):
    
    # Obtaining the type of the subscript
    int_255139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'x' (line 131)
    x_255141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 40), 'x', False)
    # Getting the type of 'ag' (line 131)
    ag_255142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 43), 'ag', False)
    # Getting the type of 'lb' (line 131)
    lb_255143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 47), 'lb', False)
    # Getting the type of 'ub' (line 131)
    ub_255144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 51), 'ub', False)
    # Processing the call keyword arguments (line 131)
    kwargs_255145 = {}
    # Getting the type of 'step_size_to_bound' (line 131)
    step_size_to_bound_255140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 131)
    step_size_to_bound_call_result_255146 = invoke(stypy.reporting.localization.Localization(__file__, 131, 21), step_size_to_bound_255140, *[x_255141, ag_255142, lb_255143, ub_255144], **kwargs_255145)
    
    # Obtaining the member '__getitem__' of a type (line 131)
    getitem___255147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 4), step_size_to_bound_call_result_255146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 131)
    subscript_call_result_255148 = invoke(stypy.reporting.localization.Localization(__file__, 131, 4), getitem___255147, int_255139)
    
    # Assigning a type to the variable 'tuple_var_assignment_254694' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_var_assignment_254694', subscript_call_result_255148)
    
    # Assigning a Name to a Name (line 131):
    # Getting the type of 'tuple_var_assignment_254693' (line 131)
    tuple_var_assignment_254693_255149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_var_assignment_254693')
    # Assigning a type to the variable 'ag_stride_u' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'ag_stride_u', tuple_var_assignment_254693_255149)
    
    # Assigning a Name to a Name (line 131):
    # Getting the type of 'tuple_var_assignment_254694' (line 131)
    tuple_var_assignment_254694_255150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_var_assignment_254694')
    # Assigning a type to the variable '_' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 17), '_', tuple_var_assignment_254694_255150)
    
    # Getting the type of 'ag_stride_u' (line 132)
    ag_stride_u_255151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'ag_stride_u')
    # Getting the type of 'theta' (line 132)
    theta_255152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'theta')
    # Applying the binary operator '*=' (line 132)
    result_imul_255153 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 4), '*=', ag_stride_u_255151, theta_255152)
    # Assigning a type to the variable 'ag_stride_u' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'ag_stride_u', result_imul_255153)
    
    
    # Assigning a Call to a Tuple (line 133):
    
    # Assigning a Subscript to a Name (line 133):
    
    # Obtaining the type of the subscript
    int_255154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 4), 'int')
    
    # Call to build_quadratic_1d(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'A_h' (line 133)
    A_h_255156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 30), 'A_h', False)
    # Getting the type of 'g_h' (line 133)
    g_h_255157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 35), 'g_h', False)
    # Getting the type of 'ag_h' (line 133)
    ag_h_255158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 40), 'ag_h', False)
    # Processing the call keyword arguments (line 133)
    # Getting the type of 'c_h' (line 133)
    c_h_255159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 51), 'c_h', False)
    keyword_255160 = c_h_255159
    kwargs_255161 = {'diag': keyword_255160}
    # Getting the type of 'build_quadratic_1d' (line 133)
    build_quadratic_1d_255155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 133)
    build_quadratic_1d_call_result_255162 = invoke(stypy.reporting.localization.Localization(__file__, 133, 11), build_quadratic_1d_255155, *[A_h_255156, g_h_255157, ag_h_255158], **kwargs_255161)
    
    # Obtaining the member '__getitem__' of a type (line 133)
    getitem___255163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 4), build_quadratic_1d_call_result_255162, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 133)
    subscript_call_result_255164 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), getitem___255163, int_255154)
    
    # Assigning a type to the variable 'tuple_var_assignment_254695' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'tuple_var_assignment_254695', subscript_call_result_255164)
    
    # Assigning a Subscript to a Name (line 133):
    
    # Obtaining the type of the subscript
    int_255165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 4), 'int')
    
    # Call to build_quadratic_1d(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'A_h' (line 133)
    A_h_255167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 30), 'A_h', False)
    # Getting the type of 'g_h' (line 133)
    g_h_255168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 35), 'g_h', False)
    # Getting the type of 'ag_h' (line 133)
    ag_h_255169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 40), 'ag_h', False)
    # Processing the call keyword arguments (line 133)
    # Getting the type of 'c_h' (line 133)
    c_h_255170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 51), 'c_h', False)
    keyword_255171 = c_h_255170
    kwargs_255172 = {'diag': keyword_255171}
    # Getting the type of 'build_quadratic_1d' (line 133)
    build_quadratic_1d_255166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 133)
    build_quadratic_1d_call_result_255173 = invoke(stypy.reporting.localization.Localization(__file__, 133, 11), build_quadratic_1d_255166, *[A_h_255167, g_h_255168, ag_h_255169], **kwargs_255172)
    
    # Obtaining the member '__getitem__' of a type (line 133)
    getitem___255174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 4), build_quadratic_1d_call_result_255173, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 133)
    subscript_call_result_255175 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), getitem___255174, int_255165)
    
    # Assigning a type to the variable 'tuple_var_assignment_254696' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'tuple_var_assignment_254696', subscript_call_result_255175)
    
    # Assigning a Name to a Name (line 133):
    # Getting the type of 'tuple_var_assignment_254695' (line 133)
    tuple_var_assignment_254695_255176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'tuple_var_assignment_254695')
    # Assigning a type to the variable 'a' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'a', tuple_var_assignment_254695_255176)
    
    # Assigning a Name to a Name (line 133):
    # Getting the type of 'tuple_var_assignment_254696' (line 133)
    tuple_var_assignment_254696_255177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'tuple_var_assignment_254696')
    # Assigning a type to the variable 'b' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 7), 'b', tuple_var_assignment_254696_255177)
    
    # Assigning a Call to a Tuple (line 134):
    
    # Assigning a Subscript to a Name (line 134):
    
    # Obtaining the type of the subscript
    int_255178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 4), 'int')
    
    # Call to minimize_quadratic_1d(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'a' (line 134)
    a_255180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 48), 'a', False)
    # Getting the type of 'b' (line 134)
    b_255181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 51), 'b', False)
    int_255182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 54), 'int')
    # Getting the type of 'ag_stride_u' (line 134)
    ag_stride_u_255183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 57), 'ag_stride_u', False)
    # Processing the call keyword arguments (line 134)
    kwargs_255184 = {}
    # Getting the type of 'minimize_quadratic_1d' (line 134)
    minimize_quadratic_1d_255179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 26), 'minimize_quadratic_1d', False)
    # Calling minimize_quadratic_1d(args, kwargs) (line 134)
    minimize_quadratic_1d_call_result_255185 = invoke(stypy.reporting.localization.Localization(__file__, 134, 26), minimize_quadratic_1d_255179, *[a_255180, b_255181, int_255182, ag_stride_u_255183], **kwargs_255184)
    
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___255186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 4), minimize_quadratic_1d_call_result_255185, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_255187 = invoke(stypy.reporting.localization.Localization(__file__, 134, 4), getitem___255186, int_255178)
    
    # Assigning a type to the variable 'tuple_var_assignment_254697' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'tuple_var_assignment_254697', subscript_call_result_255187)
    
    # Assigning a Subscript to a Name (line 134):
    
    # Obtaining the type of the subscript
    int_255188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 4), 'int')
    
    # Call to minimize_quadratic_1d(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'a' (line 134)
    a_255190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 48), 'a', False)
    # Getting the type of 'b' (line 134)
    b_255191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 51), 'b', False)
    int_255192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 54), 'int')
    # Getting the type of 'ag_stride_u' (line 134)
    ag_stride_u_255193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 57), 'ag_stride_u', False)
    # Processing the call keyword arguments (line 134)
    kwargs_255194 = {}
    # Getting the type of 'minimize_quadratic_1d' (line 134)
    minimize_quadratic_1d_255189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 26), 'minimize_quadratic_1d', False)
    # Calling minimize_quadratic_1d(args, kwargs) (line 134)
    minimize_quadratic_1d_call_result_255195 = invoke(stypy.reporting.localization.Localization(__file__, 134, 26), minimize_quadratic_1d_255189, *[a_255190, b_255191, int_255192, ag_stride_u_255193], **kwargs_255194)
    
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___255196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 4), minimize_quadratic_1d_call_result_255195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_255197 = invoke(stypy.reporting.localization.Localization(__file__, 134, 4), getitem___255196, int_255188)
    
    # Assigning a type to the variable 'tuple_var_assignment_254698' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'tuple_var_assignment_254698', subscript_call_result_255197)
    
    # Assigning a Name to a Name (line 134):
    # Getting the type of 'tuple_var_assignment_254697' (line 134)
    tuple_var_assignment_254697_255198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'tuple_var_assignment_254697')
    # Assigning a type to the variable 'ag_stride' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'ag_stride', tuple_var_assignment_254697_255198)
    
    # Assigning a Name to a Name (line 134):
    # Getting the type of 'tuple_var_assignment_254698' (line 134)
    tuple_var_assignment_254698_255199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'tuple_var_assignment_254698')
    # Assigning a type to the variable 'ag_value' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'ag_value', tuple_var_assignment_254698_255199)
    
    # Getting the type of 'ag' (line 135)
    ag_255200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'ag')
    # Getting the type of 'ag_stride' (line 135)
    ag_stride_255201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 10), 'ag_stride')
    # Applying the binary operator '*=' (line 135)
    result_imul_255202 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 4), '*=', ag_255200, ag_stride_255201)
    # Assigning a type to the variable 'ag' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'ag', result_imul_255202)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'p_value' (line 137)
    p_value_255203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 7), 'p_value')
    # Getting the type of 'r_value' (line 137)
    r_value_255204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 17), 'r_value')
    # Applying the binary operator '<' (line 137)
    result_lt_255205 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 7), '<', p_value_255203, r_value_255204)
    
    
    # Getting the type of 'p_value' (line 137)
    p_value_255206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'p_value')
    # Getting the type of 'ag_value' (line 137)
    ag_value_255207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 39), 'ag_value')
    # Applying the binary operator '<' (line 137)
    result_lt_255208 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 29), '<', p_value_255206, ag_value_255207)
    
    # Applying the binary operator 'and' (line 137)
    result_and_keyword_255209 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 7), 'and', result_lt_255205, result_lt_255208)
    
    # Testing the type of an if condition (line 137)
    if_condition_255210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 4), result_and_keyword_255209)
    # Assigning a type to the variable 'if_condition_255210' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'if_condition_255210', if_condition_255210)
    # SSA begins for if statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'p' (line 138)
    p_255211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'p')
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type', p_255211)
    # SSA branch for the else part of an if statement (line 137)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'r_value' (line 139)
    r_value_255212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 9), 'r_value')
    # Getting the type of 'p_value' (line 139)
    p_value_255213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'p_value')
    # Applying the binary operator '<' (line 139)
    result_lt_255214 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 9), '<', r_value_255212, p_value_255213)
    
    
    # Getting the type of 'r_value' (line 139)
    r_value_255215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 31), 'r_value')
    # Getting the type of 'ag_value' (line 139)
    ag_value_255216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 41), 'ag_value')
    # Applying the binary operator '<' (line 139)
    result_lt_255217 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 31), '<', r_value_255215, ag_value_255216)
    
    # Applying the binary operator 'and' (line 139)
    result_and_keyword_255218 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 9), 'and', result_lt_255214, result_lt_255217)
    
    # Testing the type of an if condition (line 139)
    if_condition_255219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 9), result_and_keyword_255218)
    # Assigning a type to the variable 'if_condition_255219' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 9), 'if_condition_255219', if_condition_255219)
    # SSA begins for if statement (line 139)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'r' (line 140)
    r_255220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'stypy_return_type', r_255220)
    # SSA branch for the else part of an if statement (line 139)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'ag' (line 142)
    ag_255221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'ag')
    # Assigning a type to the variable 'stypy_return_type' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stypy_return_type', ag_255221)
    # SSA join for if statement (line 139)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'select_step(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'select_step' in the type store
    # Getting the type of 'stypy_return_type' (line 93)
    stypy_return_type_255222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_255222)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'select_step'
    return stypy_return_type_255222

# Assigning a type to the variable 'select_step' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'select_step', select_step)

@norecursion
def trf_linear(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'trf_linear'
    module_type_store = module_type_store.open_function_context('trf_linear', 145, 0, False)
    
    # Passed parameters checking function
    trf_linear.stypy_localization = localization
    trf_linear.stypy_type_of_self = None
    trf_linear.stypy_type_store = module_type_store
    trf_linear.stypy_function_name = 'trf_linear'
    trf_linear.stypy_param_names_list = ['A', 'b', 'x_lsq', 'lb', 'ub', 'tol', 'lsq_solver', 'lsmr_tol', 'max_iter', 'verbose']
    trf_linear.stypy_varargs_param_name = None
    trf_linear.stypy_kwargs_param_name = None
    trf_linear.stypy_call_defaults = defaults
    trf_linear.stypy_call_varargs = varargs
    trf_linear.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'trf_linear', ['A', 'b', 'x_lsq', 'lb', 'ub', 'tol', 'lsq_solver', 'lsmr_tol', 'max_iter', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'trf_linear', localization, ['A', 'b', 'x_lsq', 'lb', 'ub', 'tol', 'lsq_solver', 'lsmr_tol', 'max_iter', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'trf_linear(...)' code ##################

    
    # Assigning a Attribute to a Tuple (line 147):
    
    # Assigning a Subscript to a Name (line 147):
    
    # Obtaining the type of the subscript
    int_255223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 4), 'int')
    # Getting the type of 'A' (line 147)
    A_255224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), 'A')
    # Obtaining the member 'shape' of a type (line 147)
    shape_255225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 11), A_255224, 'shape')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___255226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 4), shape_255225, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_255227 = invoke(stypy.reporting.localization.Localization(__file__, 147, 4), getitem___255226, int_255223)
    
    # Assigning a type to the variable 'tuple_var_assignment_254699' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'tuple_var_assignment_254699', subscript_call_result_255227)
    
    # Assigning a Subscript to a Name (line 147):
    
    # Obtaining the type of the subscript
    int_255228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 4), 'int')
    # Getting the type of 'A' (line 147)
    A_255229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), 'A')
    # Obtaining the member 'shape' of a type (line 147)
    shape_255230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 11), A_255229, 'shape')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___255231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 4), shape_255230, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_255232 = invoke(stypy.reporting.localization.Localization(__file__, 147, 4), getitem___255231, int_255228)
    
    # Assigning a type to the variable 'tuple_var_assignment_254700' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'tuple_var_assignment_254700', subscript_call_result_255232)
    
    # Assigning a Name to a Name (line 147):
    # Getting the type of 'tuple_var_assignment_254699' (line 147)
    tuple_var_assignment_254699_255233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'tuple_var_assignment_254699')
    # Assigning a type to the variable 'm' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'm', tuple_var_assignment_254699_255233)
    
    # Assigning a Name to a Name (line 147):
    # Getting the type of 'tuple_var_assignment_254700' (line 147)
    tuple_var_assignment_254700_255234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'tuple_var_assignment_254700')
    # Assigning a type to the variable 'n' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 7), 'n', tuple_var_assignment_254700_255234)
    
    # Assigning a Call to a Tuple (line 148):
    
    # Assigning a Subscript to a Name (line 148):
    
    # Obtaining the type of the subscript
    int_255235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 4), 'int')
    
    # Call to reflective_transformation(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'x_lsq' (line 148)
    x_lsq_255237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 37), 'x_lsq', False)
    # Getting the type of 'lb' (line 148)
    lb_255238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 44), 'lb', False)
    # Getting the type of 'ub' (line 148)
    ub_255239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 48), 'ub', False)
    # Processing the call keyword arguments (line 148)
    kwargs_255240 = {}
    # Getting the type of 'reflective_transformation' (line 148)
    reflective_transformation_255236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 148)
    reflective_transformation_call_result_255241 = invoke(stypy.reporting.localization.Localization(__file__, 148, 11), reflective_transformation_255236, *[x_lsq_255237, lb_255238, ub_255239], **kwargs_255240)
    
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___255242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 4), reflective_transformation_call_result_255241, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_255243 = invoke(stypy.reporting.localization.Localization(__file__, 148, 4), getitem___255242, int_255235)
    
    # Assigning a type to the variable 'tuple_var_assignment_254701' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_254701', subscript_call_result_255243)
    
    # Assigning a Subscript to a Name (line 148):
    
    # Obtaining the type of the subscript
    int_255244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 4), 'int')
    
    # Call to reflective_transformation(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'x_lsq' (line 148)
    x_lsq_255246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 37), 'x_lsq', False)
    # Getting the type of 'lb' (line 148)
    lb_255247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 44), 'lb', False)
    # Getting the type of 'ub' (line 148)
    ub_255248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 48), 'ub', False)
    # Processing the call keyword arguments (line 148)
    kwargs_255249 = {}
    # Getting the type of 'reflective_transformation' (line 148)
    reflective_transformation_255245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'reflective_transformation', False)
    # Calling reflective_transformation(args, kwargs) (line 148)
    reflective_transformation_call_result_255250 = invoke(stypy.reporting.localization.Localization(__file__, 148, 11), reflective_transformation_255245, *[x_lsq_255246, lb_255247, ub_255248], **kwargs_255249)
    
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___255251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 4), reflective_transformation_call_result_255250, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_255252 = invoke(stypy.reporting.localization.Localization(__file__, 148, 4), getitem___255251, int_255244)
    
    # Assigning a type to the variable 'tuple_var_assignment_254702' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_254702', subscript_call_result_255252)
    
    # Assigning a Name to a Name (line 148):
    # Getting the type of 'tuple_var_assignment_254701' (line 148)
    tuple_var_assignment_254701_255253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_254701')
    # Assigning a type to the variable 'x' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'x', tuple_var_assignment_254701_255253)
    
    # Assigning a Name to a Name (line 148):
    # Getting the type of 'tuple_var_assignment_254702' (line 148)
    tuple_var_assignment_254702_255254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_254702')
    # Assigning a type to the variable '_' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), '_', tuple_var_assignment_254702_255254)
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to make_strictly_feasible(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'x' (line 149)
    x_255256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 31), 'x', False)
    # Getting the type of 'lb' (line 149)
    lb_255257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 34), 'lb', False)
    # Getting the type of 'ub' (line 149)
    ub_255258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 38), 'ub', False)
    # Processing the call keyword arguments (line 149)
    float_255259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 48), 'float')
    keyword_255260 = float_255259
    kwargs_255261 = {'rstep': keyword_255260}
    # Getting the type of 'make_strictly_feasible' (line 149)
    make_strictly_feasible_255255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'make_strictly_feasible', False)
    # Calling make_strictly_feasible(args, kwargs) (line 149)
    make_strictly_feasible_call_result_255262 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), make_strictly_feasible_255255, *[x_255256, lb_255257, ub_255258], **kwargs_255261)
    
    # Assigning a type to the variable 'x' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'x', make_strictly_feasible_call_result_255262)
    
    
    # Getting the type of 'lsq_solver' (line 151)
    lsq_solver_255263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 7), 'lsq_solver')
    str_255264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 21), 'str', 'exact')
    # Applying the binary operator '==' (line 151)
    result_eq_255265 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 7), '==', lsq_solver_255263, str_255264)
    
    # Testing the type of an if condition (line 151)
    if_condition_255266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 4), result_eq_255265)
    # Assigning a type to the variable 'if_condition_255266' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'if_condition_255266', if_condition_255266)
    # SSA begins for if statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 152):
    
    # Assigning a Subscript to a Name (line 152):
    
    # Obtaining the type of the subscript
    int_255267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 8), 'int')
    
    # Call to qr(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'A' (line 152)
    A_255269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'A', False)
    # Processing the call keyword arguments (line 152)
    str_255270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 33), 'str', 'economic')
    keyword_255271 = str_255270
    # Getting the type of 'True' (line 152)
    True_255272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 54), 'True', False)
    keyword_255273 = True_255272
    kwargs_255274 = {'pivoting': keyword_255273, 'mode': keyword_255271}
    # Getting the type of 'qr' (line 152)
    qr_255268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'qr', False)
    # Calling qr(args, kwargs) (line 152)
    qr_call_result_255275 = invoke(stypy.reporting.localization.Localization(__file__, 152, 22), qr_255268, *[A_255269], **kwargs_255274)
    
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___255276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), qr_call_result_255275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_255277 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), getitem___255276, int_255267)
    
    # Assigning a type to the variable 'tuple_var_assignment_254703' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_var_assignment_254703', subscript_call_result_255277)
    
    # Assigning a Subscript to a Name (line 152):
    
    # Obtaining the type of the subscript
    int_255278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 8), 'int')
    
    # Call to qr(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'A' (line 152)
    A_255280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'A', False)
    # Processing the call keyword arguments (line 152)
    str_255281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 33), 'str', 'economic')
    keyword_255282 = str_255281
    # Getting the type of 'True' (line 152)
    True_255283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 54), 'True', False)
    keyword_255284 = True_255283
    kwargs_255285 = {'pivoting': keyword_255284, 'mode': keyword_255282}
    # Getting the type of 'qr' (line 152)
    qr_255279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'qr', False)
    # Calling qr(args, kwargs) (line 152)
    qr_call_result_255286 = invoke(stypy.reporting.localization.Localization(__file__, 152, 22), qr_255279, *[A_255280], **kwargs_255285)
    
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___255287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), qr_call_result_255286, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_255288 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), getitem___255287, int_255278)
    
    # Assigning a type to the variable 'tuple_var_assignment_254704' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_var_assignment_254704', subscript_call_result_255288)
    
    # Assigning a Subscript to a Name (line 152):
    
    # Obtaining the type of the subscript
    int_255289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 8), 'int')
    
    # Call to qr(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'A' (line 152)
    A_255291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'A', False)
    # Processing the call keyword arguments (line 152)
    str_255292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 33), 'str', 'economic')
    keyword_255293 = str_255292
    # Getting the type of 'True' (line 152)
    True_255294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 54), 'True', False)
    keyword_255295 = True_255294
    kwargs_255296 = {'pivoting': keyword_255295, 'mode': keyword_255293}
    # Getting the type of 'qr' (line 152)
    qr_255290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'qr', False)
    # Calling qr(args, kwargs) (line 152)
    qr_call_result_255297 = invoke(stypy.reporting.localization.Localization(__file__, 152, 22), qr_255290, *[A_255291], **kwargs_255296)
    
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___255298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), qr_call_result_255297, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_255299 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), getitem___255298, int_255289)
    
    # Assigning a type to the variable 'tuple_var_assignment_254705' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_var_assignment_254705', subscript_call_result_255299)
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'tuple_var_assignment_254703' (line 152)
    tuple_var_assignment_254703_255300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_var_assignment_254703')
    # Assigning a type to the variable 'QT' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'QT', tuple_var_assignment_254703_255300)
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'tuple_var_assignment_254704' (line 152)
    tuple_var_assignment_254704_255301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_var_assignment_254704')
    # Assigning a type to the variable 'R' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'R', tuple_var_assignment_254704_255301)
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'tuple_var_assignment_254705' (line 152)
    tuple_var_assignment_254705_255302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_var_assignment_254705')
    # Assigning a type to the variable 'perm' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'perm', tuple_var_assignment_254705_255302)
    
    # Assigning a Attribute to a Name (line 153):
    
    # Assigning a Attribute to a Name (line 153):
    # Getting the type of 'QT' (line 153)
    QT_255303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 13), 'QT')
    # Obtaining the member 'T' of a type (line 153)
    T_255304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 13), QT_255303, 'T')
    # Assigning a type to the variable 'QT' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'QT', T_255304)
    
    
    # Getting the type of 'm' (line 155)
    m_255305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'm')
    # Getting the type of 'n' (line 155)
    n_255306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'n')
    # Applying the binary operator '<' (line 155)
    result_lt_255307 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 11), '<', m_255305, n_255306)
    
    # Testing the type of an if condition (line 155)
    if_condition_255308 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 8), result_lt_255307)
    # Assigning a type to the variable 'if_condition_255308' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'if_condition_255308', if_condition_255308)
    # SSA begins for if statement (line 155)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 156):
    
    # Assigning a Call to a Name (line 156):
    
    # Call to vstack(...): (line 156)
    # Processing the call arguments (line 156)
    
    # Obtaining an instance of the builtin type 'tuple' (line 156)
    tuple_255311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 156)
    # Adding element type (line 156)
    # Getting the type of 'R' (line 156)
    R_255312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 27), 'R', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 27), tuple_255311, R_255312)
    # Adding element type (line 156)
    
    # Call to zeros(...): (line 156)
    # Processing the call arguments (line 156)
    
    # Obtaining an instance of the builtin type 'tuple' (line 156)
    tuple_255315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 156)
    # Adding element type (line 156)
    # Getting the type of 'n' (line 156)
    n_255316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 40), 'n', False)
    # Getting the type of 'm' (line 156)
    m_255317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 44), 'm', False)
    # Applying the binary operator '-' (line 156)
    result_sub_255318 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 40), '-', n_255316, m_255317)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 40), tuple_255315, result_sub_255318)
    # Adding element type (line 156)
    # Getting the type of 'n' (line 156)
    n_255319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 47), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 40), tuple_255315, n_255319)
    
    # Processing the call keyword arguments (line 156)
    kwargs_255320 = {}
    # Getting the type of 'np' (line 156)
    np_255313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 30), 'np', False)
    # Obtaining the member 'zeros' of a type (line 156)
    zeros_255314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 30), np_255313, 'zeros')
    # Calling zeros(args, kwargs) (line 156)
    zeros_call_result_255321 = invoke(stypy.reporting.localization.Localization(__file__, 156, 30), zeros_255314, *[tuple_255315], **kwargs_255320)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 27), tuple_255311, zeros_call_result_255321)
    
    # Processing the call keyword arguments (line 156)
    kwargs_255322 = {}
    # Getting the type of 'np' (line 156)
    np_255309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'np', False)
    # Obtaining the member 'vstack' of a type (line 156)
    vstack_255310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 16), np_255309, 'vstack')
    # Calling vstack(args, kwargs) (line 156)
    vstack_call_result_255323 = invoke(stypy.reporting.localization.Localization(__file__, 156, 16), vstack_255310, *[tuple_255311], **kwargs_255322)
    
    # Assigning a type to the variable 'R' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'R', vstack_call_result_255323)
    # SSA join for if statement (line 155)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to zeros(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'n' (line 158)
    n_255326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 23), 'n', False)
    # Processing the call keyword arguments (line 158)
    kwargs_255327 = {}
    # Getting the type of 'np' (line 158)
    np_255324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 14), 'np', False)
    # Obtaining the member 'zeros' of a type (line 158)
    zeros_255325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 14), np_255324, 'zeros')
    # Calling zeros(args, kwargs) (line 158)
    zeros_call_result_255328 = invoke(stypy.reporting.localization.Localization(__file__, 158, 14), zeros_255325, *[n_255326], **kwargs_255327)
    
    # Assigning a type to the variable 'QTr' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'QTr', zeros_call_result_255328)
    
    # Assigning a Call to a Name (line 159):
    
    # Assigning a Call to a Name (line 159):
    
    # Call to min(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'm' (line 159)
    m_255330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'm', False)
    # Getting the type of 'n' (line 159)
    n_255331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'n', False)
    # Processing the call keyword arguments (line 159)
    kwargs_255332 = {}
    # Getting the type of 'min' (line 159)
    min_255329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'min', False)
    # Calling min(args, kwargs) (line 159)
    min_call_result_255333 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), min_255329, *[m_255330, n_255331], **kwargs_255332)
    
    # Assigning a type to the variable 'k' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'k', min_call_result_255333)
    # SSA branch for the else part of an if statement (line 151)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'lsq_solver' (line 160)
    lsq_solver_255334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 9), 'lsq_solver')
    str_255335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 23), 'str', 'lsmr')
    # Applying the binary operator '==' (line 160)
    result_eq_255336 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 9), '==', lsq_solver_255334, str_255335)
    
    # Testing the type of an if condition (line 160)
    if_condition_255337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 9), result_eq_255336)
    # Assigning a type to the variable 'if_condition_255337' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 9), 'if_condition_255337', if_condition_255337)
    # SSA begins for if statement (line 160)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 161):
    
    # Assigning a Call to a Name (line 161):
    
    # Call to zeros(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'm' (line 161)
    m_255340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 25), 'm', False)
    # Getting the type of 'n' (line 161)
    n_255341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 29), 'n', False)
    # Applying the binary operator '+' (line 161)
    result_add_255342 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 25), '+', m_255340, n_255341)
    
    # Processing the call keyword arguments (line 161)
    kwargs_255343 = {}
    # Getting the type of 'np' (line 161)
    np_255338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'np', False)
    # Obtaining the member 'zeros' of a type (line 161)
    zeros_255339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), np_255338, 'zeros')
    # Calling zeros(args, kwargs) (line 161)
    zeros_call_result_255344 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), zeros_255339, *[result_add_255342], **kwargs_255343)
    
    # Assigning a type to the variable 'r_aug' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'r_aug', zeros_call_result_255344)
    
    # Assigning a Name to a Name (line 162):
    
    # Assigning a Name to a Name (line 162):
    # Getting the type of 'False' (line 162)
    False_255345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 24), 'False')
    # Assigning a type to the variable 'auto_lsmr_tol' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'auto_lsmr_tol', False_255345)
    
    # Type idiom detected: calculating its left and rigth part (line 163)
    # Getting the type of 'lsmr_tol' (line 163)
    lsmr_tol_255346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'lsmr_tol')
    # Getting the type of 'None' (line 163)
    None_255347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 23), 'None')
    
    (may_be_255348, more_types_in_union_255349) = may_be_none(lsmr_tol_255346, None_255347)

    if may_be_255348:

        if more_types_in_union_255349:
            # Runtime conditional SSA (line 163)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 164):
        
        # Assigning a BinOp to a Name (line 164):
        float_255350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'float')
        # Getting the type of 'tol' (line 164)
        tol_255351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 30), 'tol')
        # Applying the binary operator '*' (line 164)
        result_mul_255352 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 23), '*', float_255350, tol_255351)
        
        # Assigning a type to the variable 'lsmr_tol' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'lsmr_tol', result_mul_255352)

        if more_types_in_union_255349:
            # Runtime conditional SSA for else branch (line 163)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_255348) or more_types_in_union_255349):
        
        
        # Getting the type of 'lsmr_tol' (line 165)
        lsmr_tol_255353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 13), 'lsmr_tol')
        str_255354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 25), 'str', 'auto')
        # Applying the binary operator '==' (line 165)
        result_eq_255355 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 13), '==', lsmr_tol_255353, str_255354)
        
        # Testing the type of an if condition (line 165)
        if_condition_255356 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 13), result_eq_255355)
        # Assigning a type to the variable 'if_condition_255356' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 13), 'if_condition_255356', if_condition_255356)
        # SSA begins for if statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 166):
        
        # Assigning a Name to a Name (line 166):
        # Getting the type of 'True' (line 166)
        True_255357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'True')
        # Assigning a type to the variable 'auto_lsmr_tol' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'auto_lsmr_tol', True_255357)
        # SSA join for if statement (line 165)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_255348 and more_types_in_union_255349):
            # SSA join for if statement (line 163)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 160)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 151)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 168):
    
    # Assigning a BinOp to a Name (line 168):
    
    # Call to dot(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'x' (line 168)
    x_255360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 14), 'x', False)
    # Processing the call keyword arguments (line 168)
    kwargs_255361 = {}
    # Getting the type of 'A' (line 168)
    A_255358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'A', False)
    # Obtaining the member 'dot' of a type (line 168)
    dot_255359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), A_255358, 'dot')
    # Calling dot(args, kwargs) (line 168)
    dot_call_result_255362 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), dot_255359, *[x_255360], **kwargs_255361)
    
    # Getting the type of 'b' (line 168)
    b_255363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'b')
    # Applying the binary operator '-' (line 168)
    result_sub_255364 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 8), '-', dot_call_result_255362, b_255363)
    
    # Assigning a type to the variable 'r' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'r', result_sub_255364)
    
    # Assigning a Call to a Name (line 169):
    
    # Assigning a Call to a Name (line 169):
    
    # Call to compute_grad(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'A' (line 169)
    A_255366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'A', False)
    # Getting the type of 'r' (line 169)
    r_255367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'r', False)
    # Processing the call keyword arguments (line 169)
    kwargs_255368 = {}
    # Getting the type of 'compute_grad' (line 169)
    compute_grad_255365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'compute_grad', False)
    # Calling compute_grad(args, kwargs) (line 169)
    compute_grad_call_result_255369 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), compute_grad_255365, *[A_255366, r_255367], **kwargs_255368)
    
    # Assigning a type to the variable 'g' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'g', compute_grad_call_result_255369)
    
    # Assigning a BinOp to a Name (line 170):
    
    # Assigning a BinOp to a Name (line 170):
    float_255370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 11), 'float')
    
    # Call to dot(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'r' (line 170)
    r_255373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'r', False)
    # Getting the type of 'r' (line 170)
    r_255374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'r', False)
    # Processing the call keyword arguments (line 170)
    kwargs_255375 = {}
    # Getting the type of 'np' (line 170)
    np_255371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 17), 'np', False)
    # Obtaining the member 'dot' of a type (line 170)
    dot_255372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 17), np_255371, 'dot')
    # Calling dot(args, kwargs) (line 170)
    dot_call_result_255376 = invoke(stypy.reporting.localization.Localization(__file__, 170, 17), dot_255372, *[r_255373, r_255374], **kwargs_255375)
    
    # Applying the binary operator '*' (line 170)
    result_mul_255377 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 11), '*', float_255370, dot_call_result_255376)
    
    # Assigning a type to the variable 'cost' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'cost', result_mul_255377)
    
    # Assigning a Name to a Name (line 171):
    
    # Assigning a Name to a Name (line 171):
    # Getting the type of 'cost' (line 171)
    cost_255378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 19), 'cost')
    # Assigning a type to the variable 'initial_cost' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'initial_cost', cost_255378)
    
    # Assigning a Name to a Name (line 173):
    
    # Assigning a Name to a Name (line 173):
    # Getting the type of 'None' (line 173)
    None_255379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 25), 'None')
    # Assigning a type to the variable 'termination_status' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'termination_status', None_255379)
    
    # Assigning a Name to a Name (line 174):
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'None' (line 174)
    None_255380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'None')
    # Assigning a type to the variable 'step_norm' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'step_norm', None_255380)
    
    # Assigning a Name to a Name (line 175):
    
    # Assigning a Name to a Name (line 175):
    # Getting the type of 'None' (line 175)
    None_255381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 18), 'None')
    # Assigning a type to the variable 'cost_change' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'cost_change', None_255381)
    
    # Type idiom detected: calculating its left and rigth part (line 177)
    # Getting the type of 'max_iter' (line 177)
    max_iter_255382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 7), 'max_iter')
    # Getting the type of 'None' (line 177)
    None_255383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'None')
    
    (may_be_255384, more_types_in_union_255385) = may_be_none(max_iter_255382, None_255383)

    if may_be_255384:

        if more_types_in_union_255385:
            # Runtime conditional SSA (line 177)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 178):
        
        # Assigning a Num to a Name (line 178):
        int_255386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 19), 'int')
        # Assigning a type to the variable 'max_iter' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'max_iter', int_255386)

        if more_types_in_union_255385:
            # SSA join for if statement (line 177)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'verbose' (line 180)
    verbose_255387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 7), 'verbose')
    int_255388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 18), 'int')
    # Applying the binary operator '==' (line 180)
    result_eq_255389 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 7), '==', verbose_255387, int_255388)
    
    # Testing the type of an if condition (line 180)
    if_condition_255390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 4), result_eq_255389)
    # Assigning a type to the variable 'if_condition_255390' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'if_condition_255390', if_condition_255390)
    # SSA begins for if statement (line 180)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print_header_linear(...): (line 181)
    # Processing the call keyword arguments (line 181)
    kwargs_255392 = {}
    # Getting the type of 'print_header_linear' (line 181)
    print_header_linear_255391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'print_header_linear', False)
    # Calling print_header_linear(args, kwargs) (line 181)
    print_header_linear_call_result_255393 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), print_header_linear_255391, *[], **kwargs_255392)
    
    # SSA join for if statement (line 180)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'max_iter' (line 183)
    max_iter_255395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 27), 'max_iter', False)
    # Processing the call keyword arguments (line 183)
    kwargs_255396 = {}
    # Getting the type of 'range' (line 183)
    range_255394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'range', False)
    # Calling range(args, kwargs) (line 183)
    range_call_result_255397 = invoke(stypy.reporting.localization.Localization(__file__, 183, 21), range_255394, *[max_iter_255395], **kwargs_255396)
    
    # Testing the type of a for loop iterable (line 183)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 183, 4), range_call_result_255397)
    # Getting the type of the for loop variable (line 183)
    for_loop_var_255398 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 183, 4), range_call_result_255397)
    # Assigning a type to the variable 'iteration' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'iteration', for_loop_var_255398)
    # SSA begins for a for statement (line 183)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 184):
    
    # Assigning a Subscript to a Name (line 184):
    
    # Obtaining the type of the subscript
    int_255399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 8), 'int')
    
    # Call to CL_scaling_vector(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'x' (line 184)
    x_255401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'x', False)
    # Getting the type of 'g' (line 184)
    g_255402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 37), 'g', False)
    # Getting the type of 'lb' (line 184)
    lb_255403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 40), 'lb', False)
    # Getting the type of 'ub' (line 184)
    ub_255404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 44), 'ub', False)
    # Processing the call keyword arguments (line 184)
    kwargs_255405 = {}
    # Getting the type of 'CL_scaling_vector' (line 184)
    CL_scaling_vector_255400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'CL_scaling_vector', False)
    # Calling CL_scaling_vector(args, kwargs) (line 184)
    CL_scaling_vector_call_result_255406 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), CL_scaling_vector_255400, *[x_255401, g_255402, lb_255403, ub_255404], **kwargs_255405)
    
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___255407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), CL_scaling_vector_call_result_255406, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_255408 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), getitem___255407, int_255399)
    
    # Assigning a type to the variable 'tuple_var_assignment_254706' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_var_assignment_254706', subscript_call_result_255408)
    
    # Assigning a Subscript to a Name (line 184):
    
    # Obtaining the type of the subscript
    int_255409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 8), 'int')
    
    # Call to CL_scaling_vector(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'x' (line 184)
    x_255411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'x', False)
    # Getting the type of 'g' (line 184)
    g_255412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 37), 'g', False)
    # Getting the type of 'lb' (line 184)
    lb_255413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 40), 'lb', False)
    # Getting the type of 'ub' (line 184)
    ub_255414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 44), 'ub', False)
    # Processing the call keyword arguments (line 184)
    kwargs_255415 = {}
    # Getting the type of 'CL_scaling_vector' (line 184)
    CL_scaling_vector_255410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'CL_scaling_vector', False)
    # Calling CL_scaling_vector(args, kwargs) (line 184)
    CL_scaling_vector_call_result_255416 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), CL_scaling_vector_255410, *[x_255411, g_255412, lb_255413, ub_255414], **kwargs_255415)
    
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___255417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), CL_scaling_vector_call_result_255416, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_255418 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), getitem___255417, int_255409)
    
    # Assigning a type to the variable 'tuple_var_assignment_254707' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_var_assignment_254707', subscript_call_result_255418)
    
    # Assigning a Name to a Name (line 184):
    # Getting the type of 'tuple_var_assignment_254706' (line 184)
    tuple_var_assignment_254706_255419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_var_assignment_254706')
    # Assigning a type to the variable 'v' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'v', tuple_var_assignment_254706_255419)
    
    # Assigning a Name to a Name (line 184):
    # Getting the type of 'tuple_var_assignment_254707' (line 184)
    tuple_var_assignment_254707_255420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_var_assignment_254707')
    # Assigning a type to the variable 'dv' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'dv', tuple_var_assignment_254707_255420)
    
    # Assigning a BinOp to a Name (line 185):
    
    # Assigning a BinOp to a Name (line 185):
    # Getting the type of 'g' (line 185)
    g_255421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'g')
    # Getting the type of 'v' (line 185)
    v_255422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'v')
    # Applying the binary operator '*' (line 185)
    result_mul_255423 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 19), '*', g_255421, v_255422)
    
    # Assigning a type to the variable 'g_scaled' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'g_scaled', result_mul_255423)
    
    # Assigning a Call to a Name (line 186):
    
    # Assigning a Call to a Name (line 186):
    
    # Call to norm(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'g_scaled' (line 186)
    g_scaled_255425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 22), 'g_scaled', False)
    # Processing the call keyword arguments (line 186)
    # Getting the type of 'np' (line 186)
    np_255426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 36), 'np', False)
    # Obtaining the member 'inf' of a type (line 186)
    inf_255427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 36), np_255426, 'inf')
    keyword_255428 = inf_255427
    kwargs_255429 = {'ord': keyword_255428}
    # Getting the type of 'norm' (line 186)
    norm_255424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 17), 'norm', False)
    # Calling norm(args, kwargs) (line 186)
    norm_call_result_255430 = invoke(stypy.reporting.localization.Localization(__file__, 186, 17), norm_255424, *[g_scaled_255425], **kwargs_255429)
    
    # Assigning a type to the variable 'g_norm' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'g_norm', norm_call_result_255430)
    
    
    # Getting the type of 'g_norm' (line 187)
    g_norm_255431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'g_norm')
    # Getting the type of 'tol' (line 187)
    tol_255432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 20), 'tol')
    # Applying the binary operator '<' (line 187)
    result_lt_255433 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 11), '<', g_norm_255431, tol_255432)
    
    # Testing the type of an if condition (line 187)
    if_condition_255434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 8), result_lt_255433)
    # Assigning a type to the variable 'if_condition_255434' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'if_condition_255434', if_condition_255434)
    # SSA begins for if statement (line 187)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 188):
    
    # Assigning a Num to a Name (line 188):
    int_255435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 33), 'int')
    # Assigning a type to the variable 'termination_status' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'termination_status', int_255435)
    # SSA join for if statement (line 187)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbose' (line 190)
    verbose_255436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'verbose')
    int_255437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 22), 'int')
    # Applying the binary operator '==' (line 190)
    result_eq_255438 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 11), '==', verbose_255436, int_255437)
    
    # Testing the type of an if condition (line 190)
    if_condition_255439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 8), result_eq_255438)
    # Assigning a type to the variable 'if_condition_255439' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'if_condition_255439', if_condition_255439)
    # SSA begins for if statement (line 190)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print_iteration_linear(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'iteration' (line 191)
    iteration_255441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 35), 'iteration', False)
    # Getting the type of 'cost' (line 191)
    cost_255442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 46), 'cost', False)
    # Getting the type of 'cost_change' (line 191)
    cost_change_255443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 52), 'cost_change', False)
    # Getting the type of 'step_norm' (line 192)
    step_norm_255444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 35), 'step_norm', False)
    # Getting the type of 'g_norm' (line 192)
    g_norm_255445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 46), 'g_norm', False)
    # Processing the call keyword arguments (line 191)
    kwargs_255446 = {}
    # Getting the type of 'print_iteration_linear' (line 191)
    print_iteration_linear_255440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'print_iteration_linear', False)
    # Calling print_iteration_linear(args, kwargs) (line 191)
    print_iteration_linear_call_result_255447 = invoke(stypy.reporting.localization.Localization(__file__, 191, 12), print_iteration_linear_255440, *[iteration_255441, cost_255442, cost_change_255443, step_norm_255444, g_norm_255445], **kwargs_255446)
    
    # SSA join for if statement (line 190)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 194)
    # Getting the type of 'termination_status' (line 194)
    termination_status_255448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'termination_status')
    # Getting the type of 'None' (line 194)
    None_255449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 37), 'None')
    
    (may_be_255450, more_types_in_union_255451) = may_not_be_none(termination_status_255448, None_255449)

    if may_be_255450:

        if more_types_in_union_255451:
            # Runtime conditional SSA (line 194)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_255451:
            # SSA join for if statement (line 194)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 197):
    
    # Assigning a BinOp to a Name (line 197):
    # Getting the type of 'g' (line 197)
    g_255452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 17), 'g')
    # Getting the type of 'dv' (line 197)
    dv_255453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 21), 'dv')
    # Applying the binary operator '*' (line 197)
    result_mul_255454 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 17), '*', g_255452, dv_255453)
    
    # Assigning a type to the variable 'diag_h' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'diag_h', result_mul_255454)
    
    # Assigning a BinOp to a Name (line 198):
    
    # Assigning a BinOp to a Name (line 198):
    # Getting the type of 'diag_h' (line 198)
    diag_h_255455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 22), 'diag_h')
    float_255456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 32), 'float')
    # Applying the binary operator '**' (line 198)
    result_pow_255457 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 22), '**', diag_h_255455, float_255456)
    
    # Assigning a type to the variable 'diag_root_h' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'diag_root_h', result_pow_255457)
    
    # Assigning a BinOp to a Name (line 199):
    
    # Assigning a BinOp to a Name (line 199):
    # Getting the type of 'v' (line 199)
    v_255458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'v')
    float_255459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 17), 'float')
    # Applying the binary operator '**' (line 199)
    result_pow_255460 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 12), '**', v_255458, float_255459)
    
    # Assigning a type to the variable 'd' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'd', result_pow_255460)
    
    # Assigning a BinOp to a Name (line 200):
    
    # Assigning a BinOp to a Name (line 200):
    # Getting the type of 'd' (line 200)
    d_255461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 14), 'd')
    # Getting the type of 'g' (line 200)
    g_255462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), 'g')
    # Applying the binary operator '*' (line 200)
    result_mul_255463 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 14), '*', d_255461, g_255462)
    
    # Assigning a type to the variable 'g_h' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'g_h', result_mul_255463)
    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Call to a Name (line 202):
    
    # Call to right_multiplied_operator(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'A' (line 202)
    A_255465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 40), 'A', False)
    # Getting the type of 'd' (line 202)
    d_255466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 43), 'd', False)
    # Processing the call keyword arguments (line 202)
    kwargs_255467 = {}
    # Getting the type of 'right_multiplied_operator' (line 202)
    right_multiplied_operator_255464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 14), 'right_multiplied_operator', False)
    # Calling right_multiplied_operator(args, kwargs) (line 202)
    right_multiplied_operator_call_result_255468 = invoke(stypy.reporting.localization.Localization(__file__, 202, 14), right_multiplied_operator_255464, *[A_255465, d_255466], **kwargs_255467)
    
    # Assigning a type to the variable 'A_h' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'A_h', right_multiplied_operator_call_result_255468)
    
    
    # Getting the type of 'lsq_solver' (line 203)
    lsq_solver_255469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'lsq_solver')
    str_255470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 25), 'str', 'exact')
    # Applying the binary operator '==' (line 203)
    result_eq_255471 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 11), '==', lsq_solver_255469, str_255470)
    
    # Testing the type of an if condition (line 203)
    if_condition_255472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 8), result_eq_255471)
    # Assigning a type to the variable 'if_condition_255472' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'if_condition_255472', if_condition_255472)
    # SSA begins for if statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 204):
    
    # Assigning a Call to a Subscript (line 204):
    
    # Call to dot(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'r' (line 204)
    r_255475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 29), 'r', False)
    # Processing the call keyword arguments (line 204)
    kwargs_255476 = {}
    # Getting the type of 'QT' (line 204)
    QT_255473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 22), 'QT', False)
    # Obtaining the member 'dot' of a type (line 204)
    dot_255474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 22), QT_255473, 'dot')
    # Calling dot(args, kwargs) (line 204)
    dot_call_result_255477 = invoke(stypy.reporting.localization.Localization(__file__, 204, 22), dot_255474, *[r_255475], **kwargs_255476)
    
    # Getting the type of 'QTr' (line 204)
    QTr_255478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'QTr')
    # Getting the type of 'k' (line 204)
    k_255479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 17), 'k')
    slice_255480 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 204, 12), None, k_255479, None)
    # Storing an element on a container (line 204)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 12), QTr_255478, (slice_255480, dot_call_result_255477))
    
    # Assigning a UnaryOp to a Name (line 205):
    
    # Assigning a UnaryOp to a Name (line 205):
    
    
    # Call to regularized_lsq_with_qr(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'm' (line 205)
    m_255482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 43), 'm', False)
    # Getting the type of 'n' (line 205)
    n_255483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 46), 'n', False)
    # Getting the type of 'R' (line 205)
    R_255484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 49), 'R', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'perm' (line 205)
    perm_255485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 55), 'perm', False)
    # Getting the type of 'd' (line 205)
    d_255486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 53), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___255487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 53), d_255486, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_255488 = invoke(stypy.reporting.localization.Localization(__file__, 205, 53), getitem___255487, perm_255485)
    
    # Applying the binary operator '*' (line 205)
    result_mul_255489 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 49), '*', R_255484, subscript_call_result_255488)
    
    # Getting the type of 'QTr' (line 205)
    QTr_255490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 62), 'QTr', False)
    # Getting the type of 'perm' (line 205)
    perm_255491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 67), 'perm', False)
    # Getting the type of 'diag_root_h' (line 206)
    diag_root_h_255492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 43), 'diag_root_h', False)
    # Processing the call keyword arguments (line 205)
    # Getting the type of 'False' (line 206)
    False_255493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 63), 'False', False)
    keyword_255494 = False_255493
    kwargs_255495 = {'copy_R': keyword_255494}
    # Getting the type of 'regularized_lsq_with_qr' (line 205)
    regularized_lsq_with_qr_255481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'regularized_lsq_with_qr', False)
    # Calling regularized_lsq_with_qr(args, kwargs) (line 205)
    regularized_lsq_with_qr_call_result_255496 = invoke(stypy.reporting.localization.Localization(__file__, 205, 19), regularized_lsq_with_qr_255481, *[m_255482, n_255483, result_mul_255489, QTr_255490, perm_255491, diag_root_h_255492], **kwargs_255495)
    
    # Applying the 'usub' unary operator (line 205)
    result___neg___255497 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 18), 'usub', regularized_lsq_with_qr_call_result_255496)
    
    # Assigning a type to the variable 'p_h' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'p_h', result___neg___255497)
    # SSA branch for the else part of an if statement (line 203)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'lsq_solver' (line 207)
    lsq_solver_255498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 13), 'lsq_solver')
    str_255499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 27), 'str', 'lsmr')
    # Applying the binary operator '==' (line 207)
    result_eq_255500 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 13), '==', lsq_solver_255498, str_255499)
    
    # Testing the type of an if condition (line 207)
    if_condition_255501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 13), result_eq_255500)
    # Assigning a type to the variable 'if_condition_255501' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 13), 'if_condition_255501', if_condition_255501)
    # SSA begins for if statement (line 207)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 208):
    
    # Assigning a Call to a Name (line 208):
    
    # Call to regularized_lsq_operator(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'A_h' (line 208)
    A_h_255503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 47), 'A_h', False)
    # Getting the type of 'diag_root_h' (line 208)
    diag_root_h_255504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 52), 'diag_root_h', False)
    # Processing the call keyword arguments (line 208)
    kwargs_255505 = {}
    # Getting the type of 'regularized_lsq_operator' (line 208)
    regularized_lsq_operator_255502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 22), 'regularized_lsq_operator', False)
    # Calling regularized_lsq_operator(args, kwargs) (line 208)
    regularized_lsq_operator_call_result_255506 = invoke(stypy.reporting.localization.Localization(__file__, 208, 22), regularized_lsq_operator_255502, *[A_h_255503, diag_root_h_255504], **kwargs_255505)
    
    # Assigning a type to the variable 'lsmr_op' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'lsmr_op', regularized_lsq_operator_call_result_255506)
    
    # Assigning a Name to a Subscript (line 209):
    
    # Assigning a Name to a Subscript (line 209):
    # Getting the type of 'r' (line 209)
    r_255507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 24), 'r')
    # Getting the type of 'r_aug' (line 209)
    r_aug_255508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'r_aug')
    # Getting the type of 'm' (line 209)
    m_255509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 19), 'm')
    slice_255510 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 209, 12), None, m_255509, None)
    # Storing an element on a container (line 209)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 12), r_aug_255508, (slice_255510, r_255507))
    
    # Getting the type of 'auto_lsmr_tol' (line 210)
    auto_lsmr_tol_255511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 15), 'auto_lsmr_tol')
    # Testing the type of an if condition (line 210)
    if_condition_255512 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 12), auto_lsmr_tol_255511)
    # Assigning a type to the variable 'if_condition_255512' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'if_condition_255512', if_condition_255512)
    # SSA begins for if statement (line 210)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 211):
    
    # Assigning a BinOp to a Name (line 211):
    float_255513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 22), 'float')
    
    # Call to min(...): (line 211)
    # Processing the call arguments (line 211)
    float_255515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 33), 'float')
    # Getting the type of 'g_norm' (line 211)
    g_norm_255516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 38), 'g_norm', False)
    # Processing the call keyword arguments (line 211)
    kwargs_255517 = {}
    # Getting the type of 'min' (line 211)
    min_255514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 29), 'min', False)
    # Calling min(args, kwargs) (line 211)
    min_call_result_255518 = invoke(stypy.reporting.localization.Localization(__file__, 211, 29), min_255514, *[float_255515, g_norm_255516], **kwargs_255517)
    
    # Applying the binary operator '*' (line 211)
    result_mul_255519 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 22), '*', float_255513, min_call_result_255518)
    
    # Assigning a type to the variable 'eta' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'eta', result_mul_255519)
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Call to max(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'EPS' (line 212)
    EPS_255521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'EPS', False)
    
    # Call to min(...): (line 212)
    # Processing the call arguments (line 212)
    float_255523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 40), 'float')
    # Getting the type of 'eta' (line 212)
    eta_255524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 45), 'eta', False)
    # Getting the type of 'g_norm' (line 212)
    g_norm_255525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 51), 'g_norm', False)
    # Applying the binary operator '*' (line 212)
    result_mul_255526 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 45), '*', eta_255524, g_norm_255525)
    
    # Processing the call keyword arguments (line 212)
    kwargs_255527 = {}
    # Getting the type of 'min' (line 212)
    min_255522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'min', False)
    # Calling min(args, kwargs) (line 212)
    min_call_result_255528 = invoke(stypy.reporting.localization.Localization(__file__, 212, 36), min_255522, *[float_255523, result_mul_255526], **kwargs_255527)
    
    # Processing the call keyword arguments (line 212)
    kwargs_255529 = {}
    # Getting the type of 'max' (line 212)
    max_255520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 27), 'max', False)
    # Calling max(args, kwargs) (line 212)
    max_call_result_255530 = invoke(stypy.reporting.localization.Localization(__file__, 212, 27), max_255520, *[EPS_255521, min_call_result_255528], **kwargs_255529)
    
    # Assigning a type to the variable 'lsmr_tol' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'lsmr_tol', max_call_result_255530)
    # SSA join for if statement (line 210)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a UnaryOp to a Name (line 213):
    
    # Assigning a UnaryOp to a Name (line 213):
    
    
    # Obtaining the type of the subscript
    int_255531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 70), 'int')
    
    # Call to lsmr(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'lsmr_op' (line 213)
    lsmr_op_255533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'lsmr_op', False)
    # Getting the type of 'r_aug' (line 213)
    r_aug_255534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 33), 'r_aug', False)
    # Processing the call keyword arguments (line 213)
    # Getting the type of 'lsmr_tol' (line 213)
    lsmr_tol_255535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 45), 'lsmr_tol', False)
    keyword_255536 = lsmr_tol_255535
    # Getting the type of 'lsmr_tol' (line 213)
    lsmr_tol_255537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 60), 'lsmr_tol', False)
    keyword_255538 = lsmr_tol_255537
    kwargs_255539 = {'btol': keyword_255538, 'atol': keyword_255536}
    # Getting the type of 'lsmr' (line 213)
    lsmr_255532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'lsmr', False)
    # Calling lsmr(args, kwargs) (line 213)
    lsmr_call_result_255540 = invoke(stypy.reporting.localization.Localization(__file__, 213, 19), lsmr_255532, *[lsmr_op_255533, r_aug_255534], **kwargs_255539)
    
    # Obtaining the member '__getitem__' of a type (line 213)
    getitem___255541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 19), lsmr_call_result_255540, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 213)
    subscript_call_result_255542 = invoke(stypy.reporting.localization.Localization(__file__, 213, 19), getitem___255541, int_255531)
    
    # Applying the 'usub' unary operator (line 213)
    result___neg___255543 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 18), 'usub', subscript_call_result_255542)
    
    # Assigning a type to the variable 'p_h' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'p_h', result___neg___255543)
    # SSA join for if statement (line 207)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 203)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 215):
    
    # Assigning a BinOp to a Name (line 215):
    # Getting the type of 'd' (line 215)
    d_255544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'd')
    # Getting the type of 'p_h' (line 215)
    p_h_255545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'p_h')
    # Applying the binary operator '*' (line 215)
    result_mul_255546 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 12), '*', d_255544, p_h_255545)
    
    # Assigning a type to the variable 'p' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'p', result_mul_255546)
    
    # Assigning a Call to a Name (line 217):
    
    # Assigning a Call to a Name (line 217):
    
    # Call to dot(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'p' (line 217)
    p_255549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 25), 'p', False)
    # Getting the type of 'g' (line 217)
    g_255550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 28), 'g', False)
    # Processing the call keyword arguments (line 217)
    kwargs_255551 = {}
    # Getting the type of 'np' (line 217)
    np_255547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 217)
    dot_255548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 18), np_255547, 'dot')
    # Calling dot(args, kwargs) (line 217)
    dot_call_result_255552 = invoke(stypy.reporting.localization.Localization(__file__, 217, 18), dot_255548, *[p_255549, g_255550], **kwargs_255551)
    
    # Assigning a type to the variable 'p_dot_g' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'p_dot_g', dot_call_result_255552)
    
    
    # Getting the type of 'p_dot_g' (line 218)
    p_dot_g_255553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'p_dot_g')
    int_255554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 21), 'int')
    # Applying the binary operator '>' (line 218)
    result_gt_255555 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 11), '>', p_dot_g_255553, int_255554)
    
    # Testing the type of an if condition (line 218)
    if_condition_255556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 8), result_gt_255555)
    # Assigning a type to the variable 'if_condition_255556' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'if_condition_255556', if_condition_255556)
    # SSA begins for if statement (line 218)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 219):
    
    # Assigning a Num to a Name (line 219):
    int_255557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 33), 'int')
    # Assigning a type to the variable 'termination_status' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'termination_status', int_255557)
    # SSA join for if statement (line 218)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 221):
    
    # Assigning a BinOp to a Name (line 221):
    int_255558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 16), 'int')
    
    # Call to min(...): (line 221)
    # Processing the call arguments (line 221)
    float_255560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 24), 'float')
    # Getting the type of 'g_norm' (line 221)
    g_norm_255561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 31), 'g_norm', False)
    # Processing the call keyword arguments (line 221)
    kwargs_255562 = {}
    # Getting the type of 'min' (line 221)
    min_255559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'min', False)
    # Calling min(args, kwargs) (line 221)
    min_call_result_255563 = invoke(stypy.reporting.localization.Localization(__file__, 221, 20), min_255559, *[float_255560, g_norm_255561], **kwargs_255562)
    
    # Applying the binary operator '-' (line 221)
    result_sub_255564 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 16), '-', int_255558, min_call_result_255563)
    
    # Assigning a type to the variable 'theta' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'theta', result_sub_255564)
    
    # Assigning a Call to a Name (line 222):
    
    # Assigning a Call to a Name (line 222):
    
    # Call to select_step(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'x' (line 222)
    x_255566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 27), 'x', False)
    # Getting the type of 'A_h' (line 222)
    A_h_255567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 30), 'A_h', False)
    # Getting the type of 'g_h' (line 222)
    g_h_255568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 35), 'g_h', False)
    # Getting the type of 'diag_h' (line 222)
    diag_h_255569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 40), 'diag_h', False)
    # Getting the type of 'p' (line 222)
    p_255570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 48), 'p', False)
    # Getting the type of 'p_h' (line 222)
    p_h_255571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 51), 'p_h', False)
    # Getting the type of 'd' (line 222)
    d_255572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 56), 'd', False)
    # Getting the type of 'lb' (line 222)
    lb_255573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 59), 'lb', False)
    # Getting the type of 'ub' (line 222)
    ub_255574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 63), 'ub', False)
    # Getting the type of 'theta' (line 222)
    theta_255575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 67), 'theta', False)
    # Processing the call keyword arguments (line 222)
    kwargs_255576 = {}
    # Getting the type of 'select_step' (line 222)
    select_step_255565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 15), 'select_step', False)
    # Calling select_step(args, kwargs) (line 222)
    select_step_call_result_255577 = invoke(stypy.reporting.localization.Localization(__file__, 222, 15), select_step_255565, *[x_255566, A_h_255567, g_h_255568, diag_h_255569, p_255570, p_h_255571, d_255572, lb_255573, ub_255574, theta_255575], **kwargs_255576)
    
    # Assigning a type to the variable 'step' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'step', select_step_call_result_255577)
    
    # Assigning a UnaryOp to a Name (line 223):
    
    # Assigning a UnaryOp to a Name (line 223):
    
    
    # Call to evaluate_quadratic(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'A' (line 223)
    A_255579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), 'A', False)
    # Getting the type of 'g' (line 223)
    g_255580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 45), 'g', False)
    # Getting the type of 'step' (line 223)
    step_255581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 48), 'step', False)
    # Processing the call keyword arguments (line 223)
    kwargs_255582 = {}
    # Getting the type of 'evaluate_quadratic' (line 223)
    evaluate_quadratic_255578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'evaluate_quadratic', False)
    # Calling evaluate_quadratic(args, kwargs) (line 223)
    evaluate_quadratic_call_result_255583 = invoke(stypy.reporting.localization.Localization(__file__, 223, 23), evaluate_quadratic_255578, *[A_255579, g_255580, step_255581], **kwargs_255582)
    
    # Applying the 'usub' unary operator (line 223)
    result___neg___255584 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 22), 'usub', evaluate_quadratic_call_result_255583)
    
    # Assigning a type to the variable 'cost_change' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'cost_change', result___neg___255584)
    
    
    # Getting the type of 'cost_change' (line 228)
    cost_change_255585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'cost_change')
    int_255586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 25), 'int')
    # Applying the binary operator '<' (line 228)
    result_lt_255587 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 11), '<', cost_change_255585, int_255586)
    
    # Testing the type of an if condition (line 228)
    if_condition_255588 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 8), result_lt_255587)
    # Assigning a type to the variable 'if_condition_255588' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'if_condition_255588', if_condition_255588)
    # SSA begins for if statement (line 228)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 229):
    
    # Assigning a Subscript to a Name (line 229):
    
    # Obtaining the type of the subscript
    int_255589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 12), 'int')
    
    # Call to backtracking(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'A' (line 230)
    A_255591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'A', False)
    # Getting the type of 'g' (line 230)
    g_255592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 19), 'g', False)
    # Getting the type of 'x' (line 230)
    x_255593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 22), 'x', False)
    # Getting the type of 'p' (line 230)
    p_255594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 'p', False)
    # Getting the type of 'theta' (line 230)
    theta_255595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 28), 'theta', False)
    # Getting the type of 'p_dot_g' (line 230)
    p_dot_g_255596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 35), 'p_dot_g', False)
    # Getting the type of 'lb' (line 230)
    lb_255597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 44), 'lb', False)
    # Getting the type of 'ub' (line 230)
    ub_255598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 48), 'ub', False)
    # Processing the call keyword arguments (line 229)
    kwargs_255599 = {}
    # Getting the type of 'backtracking' (line 229)
    backtracking_255590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 35), 'backtracking', False)
    # Calling backtracking(args, kwargs) (line 229)
    backtracking_call_result_255600 = invoke(stypy.reporting.localization.Localization(__file__, 229, 35), backtracking_255590, *[A_255591, g_255592, x_255593, p_255594, theta_255595, p_dot_g_255596, lb_255597, ub_255598], **kwargs_255599)
    
    # Obtaining the member '__getitem__' of a type (line 229)
    getitem___255601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), backtracking_call_result_255600, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 229)
    subscript_call_result_255602 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), getitem___255601, int_255589)
    
    # Assigning a type to the variable 'tuple_var_assignment_254708' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'tuple_var_assignment_254708', subscript_call_result_255602)
    
    # Assigning a Subscript to a Name (line 229):
    
    # Obtaining the type of the subscript
    int_255603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 12), 'int')
    
    # Call to backtracking(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'A' (line 230)
    A_255605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'A', False)
    # Getting the type of 'g' (line 230)
    g_255606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 19), 'g', False)
    # Getting the type of 'x' (line 230)
    x_255607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 22), 'x', False)
    # Getting the type of 'p' (line 230)
    p_255608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 'p', False)
    # Getting the type of 'theta' (line 230)
    theta_255609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 28), 'theta', False)
    # Getting the type of 'p_dot_g' (line 230)
    p_dot_g_255610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 35), 'p_dot_g', False)
    # Getting the type of 'lb' (line 230)
    lb_255611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 44), 'lb', False)
    # Getting the type of 'ub' (line 230)
    ub_255612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 48), 'ub', False)
    # Processing the call keyword arguments (line 229)
    kwargs_255613 = {}
    # Getting the type of 'backtracking' (line 229)
    backtracking_255604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 35), 'backtracking', False)
    # Calling backtracking(args, kwargs) (line 229)
    backtracking_call_result_255614 = invoke(stypy.reporting.localization.Localization(__file__, 229, 35), backtracking_255604, *[A_255605, g_255606, x_255607, p_255608, theta_255609, p_dot_g_255610, lb_255611, ub_255612], **kwargs_255613)
    
    # Obtaining the member '__getitem__' of a type (line 229)
    getitem___255615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), backtracking_call_result_255614, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 229)
    subscript_call_result_255616 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), getitem___255615, int_255603)
    
    # Assigning a type to the variable 'tuple_var_assignment_254709' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'tuple_var_assignment_254709', subscript_call_result_255616)
    
    # Assigning a Subscript to a Name (line 229):
    
    # Obtaining the type of the subscript
    int_255617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 12), 'int')
    
    # Call to backtracking(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'A' (line 230)
    A_255619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'A', False)
    # Getting the type of 'g' (line 230)
    g_255620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 19), 'g', False)
    # Getting the type of 'x' (line 230)
    x_255621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 22), 'x', False)
    # Getting the type of 'p' (line 230)
    p_255622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 'p', False)
    # Getting the type of 'theta' (line 230)
    theta_255623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 28), 'theta', False)
    # Getting the type of 'p_dot_g' (line 230)
    p_dot_g_255624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 35), 'p_dot_g', False)
    # Getting the type of 'lb' (line 230)
    lb_255625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 44), 'lb', False)
    # Getting the type of 'ub' (line 230)
    ub_255626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 48), 'ub', False)
    # Processing the call keyword arguments (line 229)
    kwargs_255627 = {}
    # Getting the type of 'backtracking' (line 229)
    backtracking_255618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 35), 'backtracking', False)
    # Calling backtracking(args, kwargs) (line 229)
    backtracking_call_result_255628 = invoke(stypy.reporting.localization.Localization(__file__, 229, 35), backtracking_255618, *[A_255619, g_255620, x_255621, p_255622, theta_255623, p_dot_g_255624, lb_255625, ub_255626], **kwargs_255627)
    
    # Obtaining the member '__getitem__' of a type (line 229)
    getitem___255629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), backtracking_call_result_255628, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 229)
    subscript_call_result_255630 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), getitem___255629, int_255617)
    
    # Assigning a type to the variable 'tuple_var_assignment_254710' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'tuple_var_assignment_254710', subscript_call_result_255630)
    
    # Assigning a Name to a Name (line 229):
    # Getting the type of 'tuple_var_assignment_254708' (line 229)
    tuple_var_assignment_254708_255631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'tuple_var_assignment_254708')
    # Assigning a type to the variable 'x' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'x', tuple_var_assignment_254708_255631)
    
    # Assigning a Name to a Name (line 229):
    # Getting the type of 'tuple_var_assignment_254709' (line 229)
    tuple_var_assignment_254709_255632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'tuple_var_assignment_254709')
    # Assigning a type to the variable 'step' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'step', tuple_var_assignment_254709_255632)
    
    # Assigning a Name to a Name (line 229):
    # Getting the type of 'tuple_var_assignment_254710' (line 229)
    tuple_var_assignment_254710_255633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'tuple_var_assignment_254710')
    # Assigning a type to the variable 'cost_change' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 21), 'cost_change', tuple_var_assignment_254710_255633)
    # SSA branch for the else part of an if statement (line 228)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 232):
    
    # Assigning a Call to a Name (line 232):
    
    # Call to make_strictly_feasible(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'x' (line 232)
    x_255635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 39), 'x', False)
    # Getting the type of 'step' (line 232)
    step_255636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 43), 'step', False)
    # Applying the binary operator '+' (line 232)
    result_add_255637 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 39), '+', x_255635, step_255636)
    
    # Getting the type of 'lb' (line 232)
    lb_255638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 49), 'lb', False)
    # Getting the type of 'ub' (line 232)
    ub_255639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 53), 'ub', False)
    # Processing the call keyword arguments (line 232)
    int_255640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 63), 'int')
    keyword_255641 = int_255640
    kwargs_255642 = {'rstep': keyword_255641}
    # Getting the type of 'make_strictly_feasible' (line 232)
    make_strictly_feasible_255634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'make_strictly_feasible', False)
    # Calling make_strictly_feasible(args, kwargs) (line 232)
    make_strictly_feasible_call_result_255643 = invoke(stypy.reporting.localization.Localization(__file__, 232, 16), make_strictly_feasible_255634, *[result_add_255637, lb_255638, ub_255639], **kwargs_255642)
    
    # Assigning a type to the variable 'x' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'x', make_strictly_feasible_call_result_255643)
    # SSA join for if statement (line 228)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 234):
    
    # Assigning a Call to a Name (line 234):
    
    # Call to norm(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'step' (line 234)
    step_255645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 25), 'step', False)
    # Processing the call keyword arguments (line 234)
    kwargs_255646 = {}
    # Getting the type of 'norm' (line 234)
    norm_255644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 20), 'norm', False)
    # Calling norm(args, kwargs) (line 234)
    norm_call_result_255647 = invoke(stypy.reporting.localization.Localization(__file__, 234, 20), norm_255644, *[step_255645], **kwargs_255646)
    
    # Assigning a type to the variable 'step_norm' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'step_norm', norm_call_result_255647)
    
    # Assigning a BinOp to a Name (line 235):
    
    # Assigning a BinOp to a Name (line 235):
    
    # Call to dot(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'x' (line 235)
    x_255650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 18), 'x', False)
    # Processing the call keyword arguments (line 235)
    kwargs_255651 = {}
    # Getting the type of 'A' (line 235)
    A_255648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'A', False)
    # Obtaining the member 'dot' of a type (line 235)
    dot_255649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), A_255648, 'dot')
    # Calling dot(args, kwargs) (line 235)
    dot_call_result_255652 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), dot_255649, *[x_255650], **kwargs_255651)
    
    # Getting the type of 'b' (line 235)
    b_255653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 23), 'b')
    # Applying the binary operator '-' (line 235)
    result_sub_255654 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 12), '-', dot_call_result_255652, b_255653)
    
    # Assigning a type to the variable 'r' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'r', result_sub_255654)
    
    # Assigning a Call to a Name (line 236):
    
    # Assigning a Call to a Name (line 236):
    
    # Call to compute_grad(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'A' (line 236)
    A_255656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 25), 'A', False)
    # Getting the type of 'r' (line 236)
    r_255657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 28), 'r', False)
    # Processing the call keyword arguments (line 236)
    kwargs_255658 = {}
    # Getting the type of 'compute_grad' (line 236)
    compute_grad_255655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'compute_grad', False)
    # Calling compute_grad(args, kwargs) (line 236)
    compute_grad_call_result_255659 = invoke(stypy.reporting.localization.Localization(__file__, 236, 12), compute_grad_255655, *[A_255656, r_255657], **kwargs_255658)
    
    # Assigning a type to the variable 'g' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'g', compute_grad_call_result_255659)
    
    
    # Getting the type of 'cost_change' (line 238)
    cost_change_255660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'cost_change')
    # Getting the type of 'tol' (line 238)
    tol_255661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 25), 'tol')
    # Getting the type of 'cost' (line 238)
    cost_255662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 31), 'cost')
    # Applying the binary operator '*' (line 238)
    result_mul_255663 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 25), '*', tol_255661, cost_255662)
    
    # Applying the binary operator '<' (line 238)
    result_lt_255664 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 11), '<', cost_change_255660, result_mul_255663)
    
    # Testing the type of an if condition (line 238)
    if_condition_255665 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), result_lt_255664)
    # Assigning a type to the variable 'if_condition_255665' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_255665', if_condition_255665)
    # SSA begins for if statement (line 238)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 239):
    
    # Assigning a Num to a Name (line 239):
    int_255666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 33), 'int')
    # Assigning a type to the variable 'termination_status' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'termination_status', int_255666)
    # SSA join for if statement (line 238)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 241):
    
    # Assigning a BinOp to a Name (line 241):
    float_255667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 15), 'float')
    
    # Call to dot(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'r' (line 241)
    r_255670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 28), 'r', False)
    # Getting the type of 'r' (line 241)
    r_255671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 31), 'r', False)
    # Processing the call keyword arguments (line 241)
    kwargs_255672 = {}
    # Getting the type of 'np' (line 241)
    np_255668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'np', False)
    # Obtaining the member 'dot' of a type (line 241)
    dot_255669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 21), np_255668, 'dot')
    # Calling dot(args, kwargs) (line 241)
    dot_call_result_255673 = invoke(stypy.reporting.localization.Localization(__file__, 241, 21), dot_255669, *[r_255670, r_255671], **kwargs_255672)
    
    # Applying the binary operator '*' (line 241)
    result_mul_255674 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 15), '*', float_255667, dot_call_result_255673)
    
    # Assigning a type to the variable 'cost' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'cost', result_mul_255674)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 243)
    # Getting the type of 'termination_status' (line 243)
    termination_status_255675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 7), 'termination_status')
    # Getting the type of 'None' (line 243)
    None_255676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 29), 'None')
    
    (may_be_255677, more_types_in_union_255678) = may_be_none(termination_status_255675, None_255676)

    if may_be_255677:

        if more_types_in_union_255678:
            # Runtime conditional SSA (line 243)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 244):
        
        # Assigning a Num to a Name (line 244):
        int_255679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 29), 'int')
        # Assigning a type to the variable 'termination_status' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'termination_status', int_255679)

        if more_types_in_union_255678:
            # SSA join for if statement (line 243)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 246):
    
    # Assigning a Call to a Name (line 246):
    
    # Call to find_active_constraints(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'x' (line 246)
    x_255681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 42), 'x', False)
    # Getting the type of 'lb' (line 246)
    lb_255682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 45), 'lb', False)
    # Getting the type of 'ub' (line 246)
    ub_255683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 49), 'ub', False)
    # Processing the call keyword arguments (line 246)
    # Getting the type of 'tol' (line 246)
    tol_255684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 58), 'tol', False)
    keyword_255685 = tol_255684
    kwargs_255686 = {'rtol': keyword_255685}
    # Getting the type of 'find_active_constraints' (line 246)
    find_active_constraints_255680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 18), 'find_active_constraints', False)
    # Calling find_active_constraints(args, kwargs) (line 246)
    find_active_constraints_call_result_255687 = invoke(stypy.reporting.localization.Localization(__file__, 246, 18), find_active_constraints_255680, *[x_255681, lb_255682, ub_255683], **kwargs_255686)
    
    # Assigning a type to the variable 'active_mask' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'active_mask', find_active_constraints_call_result_255687)
    
    # Call to OptimizeResult(...): (line 248)
    # Processing the call keyword arguments (line 248)
    # Getting the type of 'x' (line 249)
    x_255689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 10), 'x', False)
    keyword_255690 = x_255689
    # Getting the type of 'r' (line 249)
    r_255691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 17), 'r', False)
    keyword_255692 = r_255691
    # Getting the type of 'cost' (line 249)
    cost_255693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 25), 'cost', False)
    keyword_255694 = cost_255693
    # Getting the type of 'g_norm' (line 249)
    g_norm_255695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 42), 'g_norm', False)
    keyword_255696 = g_norm_255695
    # Getting the type of 'active_mask' (line 249)
    active_mask_255697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 62), 'active_mask', False)
    keyword_255698 = active_mask_255697
    # Getting the type of 'iteration' (line 250)
    iteration_255699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'iteration', False)
    int_255700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 24), 'int')
    # Applying the binary operator '+' (line 250)
    result_add_255701 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 12), '+', iteration_255699, int_255700)
    
    keyword_255702 = result_add_255701
    # Getting the type of 'termination_status' (line 250)
    termination_status_255703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 34), 'termination_status', False)
    keyword_255704 = termination_status_255703
    # Getting the type of 'initial_cost' (line 251)
    initial_cost_255705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 21), 'initial_cost', False)
    keyword_255706 = initial_cost_255705
    kwargs_255707 = {'status': keyword_255704, 'initial_cost': keyword_255706, 'active_mask': keyword_255698, 'cost': keyword_255694, 'optimality': keyword_255696, 'fun': keyword_255692, 'x': keyword_255690, 'nit': keyword_255702}
    # Getting the type of 'OptimizeResult' (line 248)
    OptimizeResult_255688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 11), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 248)
    OptimizeResult_call_result_255708 = invoke(stypy.reporting.localization.Localization(__file__, 248, 11), OptimizeResult_255688, *[], **kwargs_255707)
    
    # Assigning a type to the variable 'stypy_return_type' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type', OptimizeResult_call_result_255708)
    
    # ################# End of 'trf_linear(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'trf_linear' in the type store
    # Getting the type of 'stypy_return_type' (line 145)
    stypy_return_type_255709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_255709)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'trf_linear'
    return stypy_return_type_255709

# Assigning a type to the variable 'trf_linear' (line 145)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'trf_linear', trf_linear)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
