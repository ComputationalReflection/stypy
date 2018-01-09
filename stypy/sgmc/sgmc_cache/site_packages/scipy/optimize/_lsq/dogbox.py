
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: dogleg algorithm with rectangular trust regions for least-squares minimization.
3: 
4: The description of the algorithm can be found in [Voglis]_. The algorithm does
5: trust-region iterations, but the shape of trust regions is rectangular as
6: opposed to conventional elliptical. The intersection of a trust region and
7: an initial feasible region is again some rectangle. Thus on each iteration a
8: bound-constrained quadratic optimization problem is solved.
9: 
10: A quadratic problem is solved by well-known dogleg approach, where the
11: function is minimized along piecewise-linear "dogleg" path [NumOpt]_,
12: Chapter 4. If Jacobian is not rank-deficient then the function is decreasing
13: along this path, and optimization amounts to simply following along this
14: path as long as a point stays within the bounds. A constrained Cauchy step
15: (along the anti-gradient) is considered for safety in rank deficient cases,
16: in this situations the convergence might be slow.
17: 
18: If during iterations some variable hit the initial bound and the component
19: of anti-gradient points outside the feasible region, then a next dogleg step
20: won't make any progress. At this state such variables satisfy first-order
21: optimality conditions and they are excluded before computing a next dogleg
22: step.
23: 
24: Gauss-Newton step can be computed exactly by `numpy.linalg.lstsq` (for dense
25: Jacobian matrices) or by iterative procedure `scipy.sparse.linalg.lsmr` (for
26: dense and sparse matrices, or Jacobian being LinearOperator). The second
27: option allows to solve very large problems (up to couple of millions of
28: residuals on a regular PC), provided the Jacobian matrix is sufficiently
29: sparse. But note that dogbox is not very good for solving problems with
30: large number of constraints, because of variables exclusion-inclusion on each
31: iteration (a required number of function evaluations might be high or accuracy
32: of a solution will be poor), thus its large-scale usage is probably limited
33: to unconstrained problems.
34: 
35: References
36: ----------
37: .. [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region Dogleg
38:             Approach for Unconstrained and Bound Constrained Nonlinear
39:             Optimization", WSEAS International Conference on Applied
40:             Mathematics, Corfu, Greece, 2004.
41: .. [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization, 2nd edition".
42: '''
43: from __future__ import division, print_function, absolute_import
44: 
45: import numpy as np
46: from numpy.linalg import lstsq, norm
47: 
48: from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsmr
49: from scipy.optimize import OptimizeResult
50: from scipy._lib.six import string_types
51: 
52: from .common import (
53:     step_size_to_bound, in_bounds, update_tr_radius, evaluate_quadratic,
54:     build_quadratic_1d, minimize_quadratic_1d, compute_grad,
55:     compute_jac_scale, check_termination, scale_for_robust_loss_function,
56:     print_header_nonlinear, print_iteration_nonlinear)
57: 
58: 
59: def lsmr_operator(Jop, d, active_set):
60:     '''Compute LinearOperator to use in LSMR by dogbox algorithm.
61: 
62:     `active_set` mask is used to excluded active variables from computations
63:     of matrix-vector products.
64:     '''
65:     m, n = Jop.shape
66: 
67:     def matvec(x):
68:         x_free = x.ravel().copy()
69:         x_free[active_set] = 0
70:         return Jop.matvec(x * d)
71: 
72:     def rmatvec(x):
73:         r = d * Jop.rmatvec(x)
74:         r[active_set] = 0
75:         return r
76: 
77:     return LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=float)
78: 
79: 
80: def find_intersection(x, tr_bounds, lb, ub):
81:     '''Find intersection of trust-region bounds and initial bounds.
82: 
83:     Returns
84:     -------
85:     lb_total, ub_total : ndarray with shape of x
86:         Lower and upper bounds of the intersection region.
87:     orig_l, orig_u : ndarray of bool with shape of x
88:         True means that an original bound is taken as a corresponding bound
89:         in the intersection region.
90:     tr_l, tr_u : ndarray of bool with shape of x
91:         True means that a trust-region bound is taken as a corresponding bound
92:         in the intersection region.
93:     '''
94:     lb_centered = lb - x
95:     ub_centered = ub - x
96: 
97:     lb_total = np.maximum(lb_centered, -tr_bounds)
98:     ub_total = np.minimum(ub_centered, tr_bounds)
99: 
100:     orig_l = np.equal(lb_total, lb_centered)
101:     orig_u = np.equal(ub_total, ub_centered)
102: 
103:     tr_l = np.equal(lb_total, -tr_bounds)
104:     tr_u = np.equal(ub_total, tr_bounds)
105: 
106:     return lb_total, ub_total, orig_l, orig_u, tr_l, tr_u
107: 
108: 
109: def dogleg_step(x, newton_step, g, a, b, tr_bounds, lb, ub):
110:     '''Find dogleg step in a rectangular region.
111: 
112:     Returns
113:     -------
114:     step : ndarray, shape (n,)
115:         Computed dogleg step.
116:     bound_hits : ndarray of int, shape (n,)
117:         Each component shows whether a corresponding variable hits the
118:         initial bound after the step is taken:
119:             *  0 - a variable doesn't hit the bound.
120:             * -1 - lower bound is hit.
121:             *  1 - upper bound is hit.
122:     tr_hit : bool
123:         Whether the step hit the boundary of the trust-region.
124:     '''
125:     lb_total, ub_total, orig_l, orig_u, tr_l, tr_u = find_intersection(
126:         x, tr_bounds, lb, ub
127:     )
128:     bound_hits = np.zeros_like(x, dtype=int)
129: 
130:     if in_bounds(newton_step, lb_total, ub_total):
131:         return newton_step, bound_hits, False
132: 
133:     to_bounds, _ = step_size_to_bound(np.zeros_like(x), -g, lb_total, ub_total)
134: 
135:     # The classical dogleg algorithm would check if Cauchy step fits into
136:     # the bounds, and just return it constrained version if not. But in a
137:     # rectangular trust region it makes sense to try to improve constrained
138:     # Cauchy step too. Thus we don't distinguish these two cases.
139: 
140:     cauchy_step = -minimize_quadratic_1d(a, b, 0, to_bounds)[0] * g
141: 
142:     step_diff = newton_step - cauchy_step
143:     step_size, hits = step_size_to_bound(cauchy_step, step_diff,
144:                                          lb_total, ub_total)
145:     bound_hits[(hits < 0) & orig_l] = -1
146:     bound_hits[(hits > 0) & orig_u] = 1
147:     tr_hit = np.any((hits < 0) & tr_l | (hits > 0) & tr_u)
148: 
149:     return cauchy_step + step_size * step_diff, bound_hits, tr_hit
150: 
151: 
152: def dogbox(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale,
153:            loss_function, tr_solver, tr_options, verbose):
154:     f = f0
155:     f_true = f.copy()
156:     nfev = 1
157: 
158:     J = J0
159:     njev = 1
160: 
161:     if loss_function is not None:
162:         rho = loss_function(f)
163:         cost = 0.5 * np.sum(rho[0])
164:         J, f = scale_for_robust_loss_function(J, f, rho)
165:     else:
166:         cost = 0.5 * np.dot(f, f)
167: 
168:     g = compute_grad(J, f)
169: 
170:     jac_scale = isinstance(x_scale, string_types) and x_scale == 'jac'
171:     if jac_scale:
172:         scale, scale_inv = compute_jac_scale(J)
173:     else:
174:         scale, scale_inv = x_scale, 1 / x_scale
175: 
176:     Delta = norm(x0 * scale_inv, ord=np.inf)
177:     if Delta == 0:
178:         Delta = 1.0
179: 
180:     on_bound = np.zeros_like(x0, dtype=int)
181:     on_bound[np.equal(x0, lb)] = -1
182:     on_bound[np.equal(x0, ub)] = 1
183: 
184:     x = x0
185:     step = np.empty_like(x0)
186: 
187:     if max_nfev is None:
188:         max_nfev = x0.size * 100
189: 
190:     termination_status = None
191:     iteration = 0
192:     step_norm = None
193:     actual_reduction = None
194: 
195:     if verbose == 2:
196:         print_header_nonlinear()
197: 
198:     while True:
199:         active_set = on_bound * g < 0
200:         free_set = ~active_set
201: 
202:         g_free = g[free_set]
203:         g_full = g.copy()
204:         g[active_set] = 0
205: 
206:         g_norm = norm(g, ord=np.inf)
207:         if g_norm < gtol:
208:             termination_status = 1
209: 
210:         if verbose == 2:
211:             print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
212:                                       step_norm, g_norm)
213: 
214:         if termination_status is not None or nfev == max_nfev:
215:             break
216: 
217:         x_free = x[free_set]
218:         lb_free = lb[free_set]
219:         ub_free = ub[free_set]
220:         scale_free = scale[free_set]
221: 
222:         # Compute (Gauss-)Newton and build quadratic model for Cauchy step.
223:         if tr_solver == 'exact':
224:             J_free = J[:, free_set]
225:             newton_step = lstsq(J_free, -f, rcond=-1)[0]
226: 
227:             # Coefficients for the quadratic model along the anti-gradient.
228:             a, b = build_quadratic_1d(J_free, g_free, -g_free)
229:         elif tr_solver == 'lsmr':
230:             Jop = aslinearoperator(J)
231: 
232:             # We compute lsmr step in scaled variables and then
233:             # transform back to normal variables, if lsmr would give exact lsq
234:             # solution this would be equivalent to not doing any
235:             # transformations, but from experience it's better this way.
236: 
237:             # We pass active_set to make computations as if we selected
238:             # the free subset of J columns, but without actually doing any
239:             # slicing, which is expensive for sparse matrices and impossible
240:             # for LinearOperator.
241: 
242:             lsmr_op = lsmr_operator(Jop, scale, active_set)
243:             newton_step = -lsmr(lsmr_op, f, **tr_options)[0][free_set]
244:             newton_step *= scale_free
245: 
246:             # Components of g for active variables were zeroed, so this call
247:             # is correct and equivalent to using J_free and g_free.
248:             a, b = build_quadratic_1d(Jop, g, -g)
249: 
250:         actual_reduction = -1.0
251:         while actual_reduction <= 0 and nfev < max_nfev:
252:             tr_bounds = Delta * scale_free
253: 
254:             step_free, on_bound_free, tr_hit = dogleg_step(
255:                 x_free, newton_step, g_free, a, b, tr_bounds, lb_free, ub_free)
256: 
257:             step.fill(0.0)
258:             step[free_set] = step_free
259: 
260:             if tr_solver == 'exact':
261:                 predicted_reduction = -evaluate_quadratic(J_free, g_free,
262:                                                           step_free)
263:             elif tr_solver == 'lsmr':
264:                 predicted_reduction = -evaluate_quadratic(Jop, g, step)
265: 
266:             x_new = x + step
267:             f_new = fun(x_new)
268:             nfev += 1
269: 
270:             step_h_norm = norm(step * scale_inv, ord=np.inf)
271: 
272:             if not np.all(np.isfinite(f_new)):
273:                 Delta = 0.25 * step_h_norm
274:                 continue
275: 
276:             # Usual trust-region step quality estimation.
277:             if loss_function is not None:
278:                 cost_new = loss_function(f_new, cost_only=True)
279:             else:
280:                 cost_new = 0.5 * np.dot(f_new, f_new)
281:             actual_reduction = cost - cost_new
282: 
283:             Delta, ratio = update_tr_radius(
284:                 Delta, actual_reduction, predicted_reduction,
285:                 step_h_norm, tr_hit
286:             )
287: 
288:             step_norm = norm(step)
289:             termination_status = check_termination(
290:                 actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
291: 
292:             if termination_status is not None:
293:                 break
294: 
295:         if actual_reduction > 0:
296:             on_bound[free_set] = on_bound_free
297: 
298:             x = x_new
299:             # Set variables exactly at the boundary.
300:             mask = on_bound == -1
301:             x[mask] = lb[mask]
302:             mask = on_bound == 1
303:             x[mask] = ub[mask]
304: 
305:             f = f_new
306:             f_true = f.copy()
307: 
308:             cost = cost_new
309: 
310:             J = jac(x, f)
311:             njev += 1
312: 
313:             if loss_function is not None:
314:                 rho = loss_function(f)
315:                 J, f = scale_for_robust_loss_function(J, f, rho)
316: 
317:             g = compute_grad(J, f)
318: 
319:             if jac_scale:
320:                 scale, scale_inv = compute_jac_scale(J, scale_inv)
321:         else:
322:             step_norm = 0
323:             actual_reduction = 0
324: 
325:         iteration += 1
326: 
327:     if termination_status is None:
328:         termination_status = 0
329: 
330:     return OptimizeResult(
331:         x=x, cost=cost, fun=f_true, jac=J, grad=g_full, optimality=g_norm,
332:         active_mask=on_bound, nfev=nfev, njev=njev, status=termination_status)
333: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_249553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, (-1)), 'str', '\ndogleg algorithm with rectangular trust regions for least-squares minimization.\n\nThe description of the algorithm can be found in [Voglis]_. The algorithm does\ntrust-region iterations, but the shape of trust regions is rectangular as\nopposed to conventional elliptical. The intersection of a trust region and\nan initial feasible region is again some rectangle. Thus on each iteration a\nbound-constrained quadratic optimization problem is solved.\n\nA quadratic problem is solved by well-known dogleg approach, where the\nfunction is minimized along piecewise-linear "dogleg" path [NumOpt]_,\nChapter 4. If Jacobian is not rank-deficient then the function is decreasing\nalong this path, and optimization amounts to simply following along this\npath as long as a point stays within the bounds. A constrained Cauchy step\n(along the anti-gradient) is considered for safety in rank deficient cases,\nin this situations the convergence might be slow.\n\nIf during iterations some variable hit the initial bound and the component\nof anti-gradient points outside the feasible region, then a next dogleg step\nwon\'t make any progress. At this state such variables satisfy first-order\noptimality conditions and they are excluded before computing a next dogleg\nstep.\n\nGauss-Newton step can be computed exactly by `numpy.linalg.lstsq` (for dense\nJacobian matrices) or by iterative procedure `scipy.sparse.linalg.lsmr` (for\ndense and sparse matrices, or Jacobian being LinearOperator). The second\noption allows to solve very large problems (up to couple of millions of\nresiduals on a regular PC), provided the Jacobian matrix is sufficiently\nsparse. But note that dogbox is not very good for solving problems with\nlarge number of constraints, because of variables exclusion-inclusion on each\niteration (a required number of function evaluations might be high or accuracy\nof a solution will be poor), thus its large-scale usage is probably limited\nto unconstrained problems.\n\nReferences\n----------\n.. [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region Dogleg\n            Approach for Unconstrained and Bound Constrained Nonlinear\n            Optimization", WSEAS International Conference on Applied\n            Mathematics, Corfu, Greece, 2004.\n.. [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization, 2nd edition".\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 45, 0))

# 'import numpy' statement (line 45)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_249554 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'numpy')

if (type(import_249554) is not StypyTypeError):

    if (import_249554 != 'pyd_module'):
        __import__(import_249554)
        sys_modules_249555 = sys.modules[import_249554]
        import_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'np', sys_modules_249555.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'numpy', import_249554)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 46, 0))

# 'from numpy.linalg import lstsq, norm' statement (line 46)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_249556 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 46, 0), 'numpy.linalg')

if (type(import_249556) is not StypyTypeError):

    if (import_249556 != 'pyd_module'):
        __import__(import_249556)
        sys_modules_249557 = sys.modules[import_249556]
        import_from_module(stypy.reporting.localization.Localization(__file__, 46, 0), 'numpy.linalg', sys_modules_249557.module_type_store, module_type_store, ['lstsq', 'norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 46, 0), __file__, sys_modules_249557, sys_modules_249557.module_type_store, module_type_store)
    else:
        from numpy.linalg import lstsq, norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 46, 0), 'numpy.linalg', None, module_type_store, ['lstsq', 'norm'], [lstsq, norm])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'numpy.linalg', import_249556)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 48, 0))

# 'from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsmr' statement (line 48)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_249558 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'scipy.sparse.linalg')

if (type(import_249558) is not StypyTypeError):

    if (import_249558 != 'pyd_module'):
        __import__(import_249558)
        sys_modules_249559 = sys.modules[import_249558]
        import_from_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'scipy.sparse.linalg', sys_modules_249559.module_type_store, module_type_store, ['LinearOperator', 'aslinearoperator', 'lsmr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 48, 0), __file__, sys_modules_249559, sys_modules_249559.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsmr

        import_from_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'scipy.sparse.linalg', None, module_type_store, ['LinearOperator', 'aslinearoperator', 'lsmr'], [LinearOperator, aslinearoperator, lsmr])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'scipy.sparse.linalg', import_249558)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 49, 0))

# 'from scipy.optimize import OptimizeResult' statement (line 49)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_249560 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'scipy.optimize')

if (type(import_249560) is not StypyTypeError):

    if (import_249560 != 'pyd_module'):
        __import__(import_249560)
        sys_modules_249561 = sys.modules[import_249560]
        import_from_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'scipy.optimize', sys_modules_249561.module_type_store, module_type_store, ['OptimizeResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 49, 0), __file__, sys_modules_249561, sys_modules_249561.module_type_store, module_type_store)
    else:
        from scipy.optimize import OptimizeResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'scipy.optimize', None, module_type_store, ['OptimizeResult'], [OptimizeResult])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'scipy.optimize', import_249560)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 50, 0))

# 'from scipy._lib.six import string_types' statement (line 50)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_249562 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'scipy._lib.six')

if (type(import_249562) is not StypyTypeError):

    if (import_249562 != 'pyd_module'):
        __import__(import_249562)
        sys_modules_249563 = sys.modules[import_249562]
        import_from_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'scipy._lib.six', sys_modules_249563.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 50, 0), __file__, sys_modules_249563, sys_modules_249563.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'scipy._lib.six', import_249562)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 52, 0))

# 'from scipy.optimize._lsq.common import step_size_to_bound, in_bounds, update_tr_radius, evaluate_quadratic, build_quadratic_1d, minimize_quadratic_1d, compute_grad, compute_jac_scale, check_termination, scale_for_robust_loss_function, print_header_nonlinear, print_iteration_nonlinear' statement (line 52)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_249564 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'scipy.optimize._lsq.common')

if (type(import_249564) is not StypyTypeError):

    if (import_249564 != 'pyd_module'):
        __import__(import_249564)
        sys_modules_249565 = sys.modules[import_249564]
        import_from_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'scipy.optimize._lsq.common', sys_modules_249565.module_type_store, module_type_store, ['step_size_to_bound', 'in_bounds', 'update_tr_radius', 'evaluate_quadratic', 'build_quadratic_1d', 'minimize_quadratic_1d', 'compute_grad', 'compute_jac_scale', 'check_termination', 'scale_for_robust_loss_function', 'print_header_nonlinear', 'print_iteration_nonlinear'])
        nest_module(stypy.reporting.localization.Localization(__file__, 52, 0), __file__, sys_modules_249565, sys_modules_249565.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.common import step_size_to_bound, in_bounds, update_tr_radius, evaluate_quadratic, build_quadratic_1d, minimize_quadratic_1d, compute_grad, compute_jac_scale, check_termination, scale_for_robust_loss_function, print_header_nonlinear, print_iteration_nonlinear

        import_from_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'scipy.optimize._lsq.common', None, module_type_store, ['step_size_to_bound', 'in_bounds', 'update_tr_radius', 'evaluate_quadratic', 'build_quadratic_1d', 'minimize_quadratic_1d', 'compute_grad', 'compute_jac_scale', 'check_termination', 'scale_for_robust_loss_function', 'print_header_nonlinear', 'print_iteration_nonlinear'], [step_size_to_bound, in_bounds, update_tr_radius, evaluate_quadratic, build_quadratic_1d, minimize_quadratic_1d, compute_grad, compute_jac_scale, check_termination, scale_for_robust_loss_function, print_header_nonlinear, print_iteration_nonlinear])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.common' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'scipy.optimize._lsq.common', import_249564)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')


@norecursion
def lsmr_operator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lsmr_operator'
    module_type_store = module_type_store.open_function_context('lsmr_operator', 59, 0, False)
    
    # Passed parameters checking function
    lsmr_operator.stypy_localization = localization
    lsmr_operator.stypy_type_of_self = None
    lsmr_operator.stypy_type_store = module_type_store
    lsmr_operator.stypy_function_name = 'lsmr_operator'
    lsmr_operator.stypy_param_names_list = ['Jop', 'd', 'active_set']
    lsmr_operator.stypy_varargs_param_name = None
    lsmr_operator.stypy_kwargs_param_name = None
    lsmr_operator.stypy_call_defaults = defaults
    lsmr_operator.stypy_call_varargs = varargs
    lsmr_operator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lsmr_operator', ['Jop', 'd', 'active_set'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lsmr_operator', localization, ['Jop', 'd', 'active_set'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lsmr_operator(...)' code ##################

    str_249566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', 'Compute LinearOperator to use in LSMR by dogbox algorithm.\n\n    `active_set` mask is used to excluded active variables from computations\n    of matrix-vector products.\n    ')
    
    # Assigning a Attribute to a Tuple (line 65):
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_249567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    # Getting the type of 'Jop' (line 65)
    Jop_249568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'Jop')
    # Obtaining the member 'shape' of a type (line 65)
    shape_249569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), Jop_249568, 'shape')
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___249570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), shape_249569, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_249571 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___249570, int_249567)
    
    # Assigning a type to the variable 'tuple_var_assignment_249522' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_249522', subscript_call_result_249571)
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_249572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    # Getting the type of 'Jop' (line 65)
    Jop_249573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'Jop')
    # Obtaining the member 'shape' of a type (line 65)
    shape_249574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), Jop_249573, 'shape')
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___249575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), shape_249574, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_249576 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___249575, int_249572)
    
    # Assigning a type to the variable 'tuple_var_assignment_249523' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_249523', subscript_call_result_249576)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_249522' (line 65)
    tuple_var_assignment_249522_249577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_249522')
    # Assigning a type to the variable 'm' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'm', tuple_var_assignment_249522_249577)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_249523' (line 65)
    tuple_var_assignment_249523_249578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_249523')
    # Assigning a type to the variable 'n' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 7), 'n', tuple_var_assignment_249523_249578)

    @norecursion
    def matvec(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matvec'
        module_type_store = module_type_store.open_function_context('matvec', 67, 4, False)
        
        # Passed parameters checking function
        matvec.stypy_localization = localization
        matvec.stypy_type_of_self = None
        matvec.stypy_type_store = module_type_store
        matvec.stypy_function_name = 'matvec'
        matvec.stypy_param_names_list = ['x']
        matvec.stypy_varargs_param_name = None
        matvec.stypy_kwargs_param_name = None
        matvec.stypy_call_defaults = defaults
        matvec.stypy_call_varargs = varargs
        matvec.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'matvec(...)' code ##################

        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to copy(...): (line 68)
        # Processing the call keyword arguments (line 68)
        kwargs_249584 = {}
        
        # Call to ravel(...): (line 68)
        # Processing the call keyword arguments (line 68)
        kwargs_249581 = {}
        # Getting the type of 'x' (line 68)
        x_249579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'x', False)
        # Obtaining the member 'ravel' of a type (line 68)
        ravel_249580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 17), x_249579, 'ravel')
        # Calling ravel(args, kwargs) (line 68)
        ravel_call_result_249582 = invoke(stypy.reporting.localization.Localization(__file__, 68, 17), ravel_249580, *[], **kwargs_249581)
        
        # Obtaining the member 'copy' of a type (line 68)
        copy_249583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 17), ravel_call_result_249582, 'copy')
        # Calling copy(args, kwargs) (line 68)
        copy_call_result_249585 = invoke(stypy.reporting.localization.Localization(__file__, 68, 17), copy_249583, *[], **kwargs_249584)
        
        # Assigning a type to the variable 'x_free' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'x_free', copy_call_result_249585)
        
        # Assigning a Num to a Subscript (line 69):
        
        # Assigning a Num to a Subscript (line 69):
        int_249586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 29), 'int')
        # Getting the type of 'x_free' (line 69)
        x_free_249587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'x_free')
        # Getting the type of 'active_set' (line 69)
        active_set_249588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'active_set')
        # Storing an element on a container (line 69)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 8), x_free_249587, (active_set_249588, int_249586))
        
        # Call to matvec(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'x' (line 70)
        x_249591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 26), 'x', False)
        # Getting the type of 'd' (line 70)
        d_249592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 30), 'd', False)
        # Applying the binary operator '*' (line 70)
        result_mul_249593 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 26), '*', x_249591, d_249592)
        
        # Processing the call keyword arguments (line 70)
        kwargs_249594 = {}
        # Getting the type of 'Jop' (line 70)
        Jop_249589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'Jop', False)
        # Obtaining the member 'matvec' of a type (line 70)
        matvec_249590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 15), Jop_249589, 'matvec')
        # Calling matvec(args, kwargs) (line 70)
        matvec_call_result_249595 = invoke(stypy.reporting.localization.Localization(__file__, 70, 15), matvec_249590, *[result_mul_249593], **kwargs_249594)
        
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type', matvec_call_result_249595)
        
        # ################# End of 'matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_249596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_249596)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matvec'
        return stypy_return_type_249596

    # Assigning a type to the variable 'matvec' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'matvec', matvec)

    @norecursion
    def rmatvec(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'rmatvec'
        module_type_store = module_type_store.open_function_context('rmatvec', 72, 4, False)
        
        # Passed parameters checking function
        rmatvec.stypy_localization = localization
        rmatvec.stypy_type_of_self = None
        rmatvec.stypy_type_store = module_type_store
        rmatvec.stypy_function_name = 'rmatvec'
        rmatvec.stypy_param_names_list = ['x']
        rmatvec.stypy_varargs_param_name = None
        rmatvec.stypy_kwargs_param_name = None
        rmatvec.stypy_call_defaults = defaults
        rmatvec.stypy_call_varargs = varargs
        rmatvec.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rmatvec(...)' code ##################

        
        # Assigning a BinOp to a Name (line 73):
        
        # Assigning a BinOp to a Name (line 73):
        # Getting the type of 'd' (line 73)
        d_249597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'd')
        
        # Call to rmatvec(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'x' (line 73)
        x_249600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 28), 'x', False)
        # Processing the call keyword arguments (line 73)
        kwargs_249601 = {}
        # Getting the type of 'Jop' (line 73)
        Jop_249598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'Jop', False)
        # Obtaining the member 'rmatvec' of a type (line 73)
        rmatvec_249599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), Jop_249598, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 73)
        rmatvec_call_result_249602 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), rmatvec_249599, *[x_249600], **kwargs_249601)
        
        # Applying the binary operator '*' (line 73)
        result_mul_249603 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 12), '*', d_249597, rmatvec_call_result_249602)
        
        # Assigning a type to the variable 'r' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'r', result_mul_249603)
        
        # Assigning a Num to a Subscript (line 74):
        
        # Assigning a Num to a Subscript (line 74):
        int_249604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'int')
        # Getting the type of 'r' (line 74)
        r_249605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'r')
        # Getting the type of 'active_set' (line 74)
        active_set_249606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 10), 'active_set')
        # Storing an element on a container (line 74)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 8), r_249605, (active_set_249606, int_249604))
        # Getting the type of 'r' (line 75)
        r_249607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'stypy_return_type', r_249607)
        
        # ################# End of 'rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_249608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_249608)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rmatvec'
        return stypy_return_type_249608

    # Assigning a type to the variable 'rmatvec' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'rmatvec', rmatvec)
    
    # Call to LinearOperator(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Obtaining an instance of the builtin type 'tuple' (line 77)
    tuple_249610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 77)
    # Adding element type (line 77)
    # Getting the type of 'm' (line 77)
    m_249611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 27), tuple_249610, m_249611)
    # Adding element type (line 77)
    # Getting the type of 'n' (line 77)
    n_249612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 30), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 27), tuple_249610, n_249612)
    
    # Processing the call keyword arguments (line 77)
    # Getting the type of 'matvec' (line 77)
    matvec_249613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 41), 'matvec', False)
    keyword_249614 = matvec_249613
    # Getting the type of 'rmatvec' (line 77)
    rmatvec_249615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 57), 'rmatvec', False)
    keyword_249616 = rmatvec_249615
    # Getting the type of 'float' (line 77)
    float_249617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 72), 'float', False)
    keyword_249618 = float_249617
    kwargs_249619 = {'dtype': keyword_249618, 'rmatvec': keyword_249616, 'matvec': keyword_249614}
    # Getting the type of 'LinearOperator' (line 77)
    LinearOperator_249609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'LinearOperator', False)
    # Calling LinearOperator(args, kwargs) (line 77)
    LinearOperator_call_result_249620 = invoke(stypy.reporting.localization.Localization(__file__, 77, 11), LinearOperator_249609, *[tuple_249610], **kwargs_249619)
    
    # Assigning a type to the variable 'stypy_return_type' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type', LinearOperator_call_result_249620)
    
    # ################# End of 'lsmr_operator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lsmr_operator' in the type store
    # Getting the type of 'stypy_return_type' (line 59)
    stypy_return_type_249621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249621)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lsmr_operator'
    return stypy_return_type_249621

# Assigning a type to the variable 'lsmr_operator' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'lsmr_operator', lsmr_operator)

@norecursion
def find_intersection(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_intersection'
    module_type_store = module_type_store.open_function_context('find_intersection', 80, 0, False)
    
    # Passed parameters checking function
    find_intersection.stypy_localization = localization
    find_intersection.stypy_type_of_self = None
    find_intersection.stypy_type_store = module_type_store
    find_intersection.stypy_function_name = 'find_intersection'
    find_intersection.stypy_param_names_list = ['x', 'tr_bounds', 'lb', 'ub']
    find_intersection.stypy_varargs_param_name = None
    find_intersection.stypy_kwargs_param_name = None
    find_intersection.stypy_call_defaults = defaults
    find_intersection.stypy_call_varargs = varargs
    find_intersection.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_intersection', ['x', 'tr_bounds', 'lb', 'ub'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_intersection', localization, ['x', 'tr_bounds', 'lb', 'ub'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_intersection(...)' code ##################

    str_249622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, (-1)), 'str', 'Find intersection of trust-region bounds and initial bounds.\n\n    Returns\n    -------\n    lb_total, ub_total : ndarray with shape of x\n        Lower and upper bounds of the intersection region.\n    orig_l, orig_u : ndarray of bool with shape of x\n        True means that an original bound is taken as a corresponding bound\n        in the intersection region.\n    tr_l, tr_u : ndarray of bool with shape of x\n        True means that a trust-region bound is taken as a corresponding bound\n        in the intersection region.\n    ')
    
    # Assigning a BinOp to a Name (line 94):
    
    # Assigning a BinOp to a Name (line 94):
    # Getting the type of 'lb' (line 94)
    lb_249623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'lb')
    # Getting the type of 'x' (line 94)
    x_249624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'x')
    # Applying the binary operator '-' (line 94)
    result_sub_249625 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 18), '-', lb_249623, x_249624)
    
    # Assigning a type to the variable 'lb_centered' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'lb_centered', result_sub_249625)
    
    # Assigning a BinOp to a Name (line 95):
    
    # Assigning a BinOp to a Name (line 95):
    # Getting the type of 'ub' (line 95)
    ub_249626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'ub')
    # Getting the type of 'x' (line 95)
    x_249627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'x')
    # Applying the binary operator '-' (line 95)
    result_sub_249628 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 18), '-', ub_249626, x_249627)
    
    # Assigning a type to the variable 'ub_centered' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'ub_centered', result_sub_249628)
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to maximum(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'lb_centered' (line 97)
    lb_centered_249631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'lb_centered', False)
    
    # Getting the type of 'tr_bounds' (line 97)
    tr_bounds_249632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'tr_bounds', False)
    # Applying the 'usub' unary operator (line 97)
    result___neg___249633 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 39), 'usub', tr_bounds_249632)
    
    # Processing the call keyword arguments (line 97)
    kwargs_249634 = {}
    # Getting the type of 'np' (line 97)
    np_249629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'np', False)
    # Obtaining the member 'maximum' of a type (line 97)
    maximum_249630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), np_249629, 'maximum')
    # Calling maximum(args, kwargs) (line 97)
    maximum_call_result_249635 = invoke(stypy.reporting.localization.Localization(__file__, 97, 15), maximum_249630, *[lb_centered_249631, result___neg___249633], **kwargs_249634)
    
    # Assigning a type to the variable 'lb_total' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'lb_total', maximum_call_result_249635)
    
    # Assigning a Call to a Name (line 98):
    
    # Assigning a Call to a Name (line 98):
    
    # Call to minimum(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'ub_centered' (line 98)
    ub_centered_249638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'ub_centered', False)
    # Getting the type of 'tr_bounds' (line 98)
    tr_bounds_249639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 39), 'tr_bounds', False)
    # Processing the call keyword arguments (line 98)
    kwargs_249640 = {}
    # Getting the type of 'np' (line 98)
    np_249636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'np', False)
    # Obtaining the member 'minimum' of a type (line 98)
    minimum_249637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 15), np_249636, 'minimum')
    # Calling minimum(args, kwargs) (line 98)
    minimum_call_result_249641 = invoke(stypy.reporting.localization.Localization(__file__, 98, 15), minimum_249637, *[ub_centered_249638, tr_bounds_249639], **kwargs_249640)
    
    # Assigning a type to the variable 'ub_total' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'ub_total', minimum_call_result_249641)
    
    # Assigning a Call to a Name (line 100):
    
    # Assigning a Call to a Name (line 100):
    
    # Call to equal(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'lb_total' (line 100)
    lb_total_249644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'lb_total', False)
    # Getting the type of 'lb_centered' (line 100)
    lb_centered_249645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 32), 'lb_centered', False)
    # Processing the call keyword arguments (line 100)
    kwargs_249646 = {}
    # Getting the type of 'np' (line 100)
    np_249642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'np', False)
    # Obtaining the member 'equal' of a type (line 100)
    equal_249643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 13), np_249642, 'equal')
    # Calling equal(args, kwargs) (line 100)
    equal_call_result_249647 = invoke(stypy.reporting.localization.Localization(__file__, 100, 13), equal_249643, *[lb_total_249644, lb_centered_249645], **kwargs_249646)
    
    # Assigning a type to the variable 'orig_l' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'orig_l', equal_call_result_249647)
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to equal(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'ub_total' (line 101)
    ub_total_249650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'ub_total', False)
    # Getting the type of 'ub_centered' (line 101)
    ub_centered_249651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 32), 'ub_centered', False)
    # Processing the call keyword arguments (line 101)
    kwargs_249652 = {}
    # Getting the type of 'np' (line 101)
    np_249648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), 'np', False)
    # Obtaining the member 'equal' of a type (line 101)
    equal_249649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 13), np_249648, 'equal')
    # Calling equal(args, kwargs) (line 101)
    equal_call_result_249653 = invoke(stypy.reporting.localization.Localization(__file__, 101, 13), equal_249649, *[ub_total_249650, ub_centered_249651], **kwargs_249652)
    
    # Assigning a type to the variable 'orig_u' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'orig_u', equal_call_result_249653)
    
    # Assigning a Call to a Name (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to equal(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'lb_total' (line 103)
    lb_total_249656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'lb_total', False)
    
    # Getting the type of 'tr_bounds' (line 103)
    tr_bounds_249657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 31), 'tr_bounds', False)
    # Applying the 'usub' unary operator (line 103)
    result___neg___249658 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 30), 'usub', tr_bounds_249657)
    
    # Processing the call keyword arguments (line 103)
    kwargs_249659 = {}
    # Getting the type of 'np' (line 103)
    np_249654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'np', False)
    # Obtaining the member 'equal' of a type (line 103)
    equal_249655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 11), np_249654, 'equal')
    # Calling equal(args, kwargs) (line 103)
    equal_call_result_249660 = invoke(stypy.reporting.localization.Localization(__file__, 103, 11), equal_249655, *[lb_total_249656, result___neg___249658], **kwargs_249659)
    
    # Assigning a type to the variable 'tr_l' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'tr_l', equal_call_result_249660)
    
    # Assigning a Call to a Name (line 104):
    
    # Assigning a Call to a Name (line 104):
    
    # Call to equal(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'ub_total' (line 104)
    ub_total_249663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'ub_total', False)
    # Getting the type of 'tr_bounds' (line 104)
    tr_bounds_249664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'tr_bounds', False)
    # Processing the call keyword arguments (line 104)
    kwargs_249665 = {}
    # Getting the type of 'np' (line 104)
    np_249661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'np', False)
    # Obtaining the member 'equal' of a type (line 104)
    equal_249662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), np_249661, 'equal')
    # Calling equal(args, kwargs) (line 104)
    equal_call_result_249666 = invoke(stypy.reporting.localization.Localization(__file__, 104, 11), equal_249662, *[ub_total_249663, tr_bounds_249664], **kwargs_249665)
    
    # Assigning a type to the variable 'tr_u' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'tr_u', equal_call_result_249666)
    
    # Obtaining an instance of the builtin type 'tuple' (line 106)
    tuple_249667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 106)
    # Adding element type (line 106)
    # Getting the type of 'lb_total' (line 106)
    lb_total_249668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'lb_total')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 11), tuple_249667, lb_total_249668)
    # Adding element type (line 106)
    # Getting the type of 'ub_total' (line 106)
    ub_total_249669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'ub_total')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 11), tuple_249667, ub_total_249669)
    # Adding element type (line 106)
    # Getting the type of 'orig_l' (line 106)
    orig_l_249670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 31), 'orig_l')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 11), tuple_249667, orig_l_249670)
    # Adding element type (line 106)
    # Getting the type of 'orig_u' (line 106)
    orig_u_249671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'orig_u')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 11), tuple_249667, orig_u_249671)
    # Adding element type (line 106)
    # Getting the type of 'tr_l' (line 106)
    tr_l_249672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 47), 'tr_l')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 11), tuple_249667, tr_l_249672)
    # Adding element type (line 106)
    # Getting the type of 'tr_u' (line 106)
    tr_u_249673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 53), 'tr_u')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 11), tuple_249667, tr_u_249673)
    
    # Assigning a type to the variable 'stypy_return_type' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type', tuple_249667)
    
    # ################# End of 'find_intersection(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_intersection' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_249674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249674)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_intersection'
    return stypy_return_type_249674

# Assigning a type to the variable 'find_intersection' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'find_intersection', find_intersection)

@norecursion
def dogleg_step(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dogleg_step'
    module_type_store = module_type_store.open_function_context('dogleg_step', 109, 0, False)
    
    # Passed parameters checking function
    dogleg_step.stypy_localization = localization
    dogleg_step.stypy_type_of_self = None
    dogleg_step.stypy_type_store = module_type_store
    dogleg_step.stypy_function_name = 'dogleg_step'
    dogleg_step.stypy_param_names_list = ['x', 'newton_step', 'g', 'a', 'b', 'tr_bounds', 'lb', 'ub']
    dogleg_step.stypy_varargs_param_name = None
    dogleg_step.stypy_kwargs_param_name = None
    dogleg_step.stypy_call_defaults = defaults
    dogleg_step.stypy_call_varargs = varargs
    dogleg_step.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dogleg_step', ['x', 'newton_step', 'g', 'a', 'b', 'tr_bounds', 'lb', 'ub'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dogleg_step', localization, ['x', 'newton_step', 'g', 'a', 'b', 'tr_bounds', 'lb', 'ub'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dogleg_step(...)' code ##################

    str_249675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, (-1)), 'str', "Find dogleg step in a rectangular region.\n\n    Returns\n    -------\n    step : ndarray, shape (n,)\n        Computed dogleg step.\n    bound_hits : ndarray of int, shape (n,)\n        Each component shows whether a corresponding variable hits the\n        initial bound after the step is taken:\n            *  0 - a variable doesn't hit the bound.\n            * -1 - lower bound is hit.\n            *  1 - upper bound is hit.\n    tr_hit : bool\n        Whether the step hit the boundary of the trust-region.\n    ")
    
    # Assigning a Call to a Tuple (line 125):
    
    # Assigning a Subscript to a Name (line 125):
    
    # Obtaining the type of the subscript
    int_249676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 4), 'int')
    
    # Call to find_intersection(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'x' (line 126)
    x_249678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'x', False)
    # Getting the type of 'tr_bounds' (line 126)
    tr_bounds_249679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'tr_bounds', False)
    # Getting the type of 'lb' (line 126)
    lb_249680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'lb', False)
    # Getting the type of 'ub' (line 126)
    ub_249681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 26), 'ub', False)
    # Processing the call keyword arguments (line 125)
    kwargs_249682 = {}
    # Getting the type of 'find_intersection' (line 125)
    find_intersection_249677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 53), 'find_intersection', False)
    # Calling find_intersection(args, kwargs) (line 125)
    find_intersection_call_result_249683 = invoke(stypy.reporting.localization.Localization(__file__, 125, 53), find_intersection_249677, *[x_249678, tr_bounds_249679, lb_249680, ub_249681], **kwargs_249682)
    
    # Obtaining the member '__getitem__' of a type (line 125)
    getitem___249684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 4), find_intersection_call_result_249683, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 125)
    subscript_call_result_249685 = invoke(stypy.reporting.localization.Localization(__file__, 125, 4), getitem___249684, int_249676)
    
    # Assigning a type to the variable 'tuple_var_assignment_249524' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_249524', subscript_call_result_249685)
    
    # Assigning a Subscript to a Name (line 125):
    
    # Obtaining the type of the subscript
    int_249686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 4), 'int')
    
    # Call to find_intersection(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'x' (line 126)
    x_249688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'x', False)
    # Getting the type of 'tr_bounds' (line 126)
    tr_bounds_249689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'tr_bounds', False)
    # Getting the type of 'lb' (line 126)
    lb_249690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'lb', False)
    # Getting the type of 'ub' (line 126)
    ub_249691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 26), 'ub', False)
    # Processing the call keyword arguments (line 125)
    kwargs_249692 = {}
    # Getting the type of 'find_intersection' (line 125)
    find_intersection_249687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 53), 'find_intersection', False)
    # Calling find_intersection(args, kwargs) (line 125)
    find_intersection_call_result_249693 = invoke(stypy.reporting.localization.Localization(__file__, 125, 53), find_intersection_249687, *[x_249688, tr_bounds_249689, lb_249690, ub_249691], **kwargs_249692)
    
    # Obtaining the member '__getitem__' of a type (line 125)
    getitem___249694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 4), find_intersection_call_result_249693, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 125)
    subscript_call_result_249695 = invoke(stypy.reporting.localization.Localization(__file__, 125, 4), getitem___249694, int_249686)
    
    # Assigning a type to the variable 'tuple_var_assignment_249525' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_249525', subscript_call_result_249695)
    
    # Assigning a Subscript to a Name (line 125):
    
    # Obtaining the type of the subscript
    int_249696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 4), 'int')
    
    # Call to find_intersection(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'x' (line 126)
    x_249698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'x', False)
    # Getting the type of 'tr_bounds' (line 126)
    tr_bounds_249699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'tr_bounds', False)
    # Getting the type of 'lb' (line 126)
    lb_249700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'lb', False)
    # Getting the type of 'ub' (line 126)
    ub_249701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 26), 'ub', False)
    # Processing the call keyword arguments (line 125)
    kwargs_249702 = {}
    # Getting the type of 'find_intersection' (line 125)
    find_intersection_249697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 53), 'find_intersection', False)
    # Calling find_intersection(args, kwargs) (line 125)
    find_intersection_call_result_249703 = invoke(stypy.reporting.localization.Localization(__file__, 125, 53), find_intersection_249697, *[x_249698, tr_bounds_249699, lb_249700, ub_249701], **kwargs_249702)
    
    # Obtaining the member '__getitem__' of a type (line 125)
    getitem___249704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 4), find_intersection_call_result_249703, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 125)
    subscript_call_result_249705 = invoke(stypy.reporting.localization.Localization(__file__, 125, 4), getitem___249704, int_249696)
    
    # Assigning a type to the variable 'tuple_var_assignment_249526' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_249526', subscript_call_result_249705)
    
    # Assigning a Subscript to a Name (line 125):
    
    # Obtaining the type of the subscript
    int_249706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 4), 'int')
    
    # Call to find_intersection(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'x' (line 126)
    x_249708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'x', False)
    # Getting the type of 'tr_bounds' (line 126)
    tr_bounds_249709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'tr_bounds', False)
    # Getting the type of 'lb' (line 126)
    lb_249710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'lb', False)
    # Getting the type of 'ub' (line 126)
    ub_249711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 26), 'ub', False)
    # Processing the call keyword arguments (line 125)
    kwargs_249712 = {}
    # Getting the type of 'find_intersection' (line 125)
    find_intersection_249707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 53), 'find_intersection', False)
    # Calling find_intersection(args, kwargs) (line 125)
    find_intersection_call_result_249713 = invoke(stypy.reporting.localization.Localization(__file__, 125, 53), find_intersection_249707, *[x_249708, tr_bounds_249709, lb_249710, ub_249711], **kwargs_249712)
    
    # Obtaining the member '__getitem__' of a type (line 125)
    getitem___249714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 4), find_intersection_call_result_249713, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 125)
    subscript_call_result_249715 = invoke(stypy.reporting.localization.Localization(__file__, 125, 4), getitem___249714, int_249706)
    
    # Assigning a type to the variable 'tuple_var_assignment_249527' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_249527', subscript_call_result_249715)
    
    # Assigning a Subscript to a Name (line 125):
    
    # Obtaining the type of the subscript
    int_249716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 4), 'int')
    
    # Call to find_intersection(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'x' (line 126)
    x_249718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'x', False)
    # Getting the type of 'tr_bounds' (line 126)
    tr_bounds_249719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'tr_bounds', False)
    # Getting the type of 'lb' (line 126)
    lb_249720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'lb', False)
    # Getting the type of 'ub' (line 126)
    ub_249721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 26), 'ub', False)
    # Processing the call keyword arguments (line 125)
    kwargs_249722 = {}
    # Getting the type of 'find_intersection' (line 125)
    find_intersection_249717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 53), 'find_intersection', False)
    # Calling find_intersection(args, kwargs) (line 125)
    find_intersection_call_result_249723 = invoke(stypy.reporting.localization.Localization(__file__, 125, 53), find_intersection_249717, *[x_249718, tr_bounds_249719, lb_249720, ub_249721], **kwargs_249722)
    
    # Obtaining the member '__getitem__' of a type (line 125)
    getitem___249724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 4), find_intersection_call_result_249723, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 125)
    subscript_call_result_249725 = invoke(stypy.reporting.localization.Localization(__file__, 125, 4), getitem___249724, int_249716)
    
    # Assigning a type to the variable 'tuple_var_assignment_249528' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_249528', subscript_call_result_249725)
    
    # Assigning a Subscript to a Name (line 125):
    
    # Obtaining the type of the subscript
    int_249726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 4), 'int')
    
    # Call to find_intersection(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'x' (line 126)
    x_249728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'x', False)
    # Getting the type of 'tr_bounds' (line 126)
    tr_bounds_249729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'tr_bounds', False)
    # Getting the type of 'lb' (line 126)
    lb_249730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'lb', False)
    # Getting the type of 'ub' (line 126)
    ub_249731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 26), 'ub', False)
    # Processing the call keyword arguments (line 125)
    kwargs_249732 = {}
    # Getting the type of 'find_intersection' (line 125)
    find_intersection_249727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 53), 'find_intersection', False)
    # Calling find_intersection(args, kwargs) (line 125)
    find_intersection_call_result_249733 = invoke(stypy.reporting.localization.Localization(__file__, 125, 53), find_intersection_249727, *[x_249728, tr_bounds_249729, lb_249730, ub_249731], **kwargs_249732)
    
    # Obtaining the member '__getitem__' of a type (line 125)
    getitem___249734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 4), find_intersection_call_result_249733, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 125)
    subscript_call_result_249735 = invoke(stypy.reporting.localization.Localization(__file__, 125, 4), getitem___249734, int_249726)
    
    # Assigning a type to the variable 'tuple_var_assignment_249529' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_249529', subscript_call_result_249735)
    
    # Assigning a Name to a Name (line 125):
    # Getting the type of 'tuple_var_assignment_249524' (line 125)
    tuple_var_assignment_249524_249736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_249524')
    # Assigning a type to the variable 'lb_total' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'lb_total', tuple_var_assignment_249524_249736)
    
    # Assigning a Name to a Name (line 125):
    # Getting the type of 'tuple_var_assignment_249525' (line 125)
    tuple_var_assignment_249525_249737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_249525')
    # Assigning a type to the variable 'ub_total' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 14), 'ub_total', tuple_var_assignment_249525_249737)
    
    # Assigning a Name to a Name (line 125):
    # Getting the type of 'tuple_var_assignment_249526' (line 125)
    tuple_var_assignment_249526_249738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_249526')
    # Assigning a type to the variable 'orig_l' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'orig_l', tuple_var_assignment_249526_249738)
    
    # Assigning a Name to a Name (line 125):
    # Getting the type of 'tuple_var_assignment_249527' (line 125)
    tuple_var_assignment_249527_249739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_249527')
    # Assigning a type to the variable 'orig_u' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 32), 'orig_u', tuple_var_assignment_249527_249739)
    
    # Assigning a Name to a Name (line 125):
    # Getting the type of 'tuple_var_assignment_249528' (line 125)
    tuple_var_assignment_249528_249740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_249528')
    # Assigning a type to the variable 'tr_l' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 40), 'tr_l', tuple_var_assignment_249528_249740)
    
    # Assigning a Name to a Name (line 125):
    # Getting the type of 'tuple_var_assignment_249529' (line 125)
    tuple_var_assignment_249529_249741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_249529')
    # Assigning a type to the variable 'tr_u' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 46), 'tr_u', tuple_var_assignment_249529_249741)
    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to zeros_like(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'x' (line 128)
    x_249744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 31), 'x', False)
    # Processing the call keyword arguments (line 128)
    # Getting the type of 'int' (line 128)
    int_249745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 40), 'int', False)
    keyword_249746 = int_249745
    kwargs_249747 = {'dtype': keyword_249746}
    # Getting the type of 'np' (line 128)
    np_249742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 128)
    zeros_like_249743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 17), np_249742, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 128)
    zeros_like_call_result_249748 = invoke(stypy.reporting.localization.Localization(__file__, 128, 17), zeros_like_249743, *[x_249744], **kwargs_249747)
    
    # Assigning a type to the variable 'bound_hits' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'bound_hits', zeros_like_call_result_249748)
    
    
    # Call to in_bounds(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'newton_step' (line 130)
    newton_step_249750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 17), 'newton_step', False)
    # Getting the type of 'lb_total' (line 130)
    lb_total_249751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 30), 'lb_total', False)
    # Getting the type of 'ub_total' (line 130)
    ub_total_249752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 40), 'ub_total', False)
    # Processing the call keyword arguments (line 130)
    kwargs_249753 = {}
    # Getting the type of 'in_bounds' (line 130)
    in_bounds_249749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 7), 'in_bounds', False)
    # Calling in_bounds(args, kwargs) (line 130)
    in_bounds_call_result_249754 = invoke(stypy.reporting.localization.Localization(__file__, 130, 7), in_bounds_249749, *[newton_step_249750, lb_total_249751, ub_total_249752], **kwargs_249753)
    
    # Testing the type of an if condition (line 130)
    if_condition_249755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 4), in_bounds_call_result_249754)
    # Assigning a type to the variable 'if_condition_249755' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'if_condition_249755', if_condition_249755)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 131)
    tuple_249756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 131)
    # Adding element type (line 131)
    # Getting the type of 'newton_step' (line 131)
    newton_step_249757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'newton_step')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 15), tuple_249756, newton_step_249757)
    # Adding element type (line 131)
    # Getting the type of 'bound_hits' (line 131)
    bound_hits_249758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 'bound_hits')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 15), tuple_249756, bound_hits_249758)
    # Adding element type (line 131)
    # Getting the type of 'False' (line 131)
    False_249759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 40), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 15), tuple_249756, False_249759)
    
    # Assigning a type to the variable 'stypy_return_type' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', tuple_249756)
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 133):
    
    # Assigning a Subscript to a Name (line 133):
    
    # Obtaining the type of the subscript
    int_249760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 133)
    # Processing the call arguments (line 133)
    
    # Call to zeros_like(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'x' (line 133)
    x_249764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 52), 'x', False)
    # Processing the call keyword arguments (line 133)
    kwargs_249765 = {}
    # Getting the type of 'np' (line 133)
    np_249762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 38), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 133)
    zeros_like_249763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 38), np_249762, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 133)
    zeros_like_call_result_249766 = invoke(stypy.reporting.localization.Localization(__file__, 133, 38), zeros_like_249763, *[x_249764], **kwargs_249765)
    
    
    # Getting the type of 'g' (line 133)
    g_249767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 57), 'g', False)
    # Applying the 'usub' unary operator (line 133)
    result___neg___249768 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 56), 'usub', g_249767)
    
    # Getting the type of 'lb_total' (line 133)
    lb_total_249769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 60), 'lb_total', False)
    # Getting the type of 'ub_total' (line 133)
    ub_total_249770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 70), 'ub_total', False)
    # Processing the call keyword arguments (line 133)
    kwargs_249771 = {}
    # Getting the type of 'step_size_to_bound' (line 133)
    step_size_to_bound_249761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 133)
    step_size_to_bound_call_result_249772 = invoke(stypy.reporting.localization.Localization(__file__, 133, 19), step_size_to_bound_249761, *[zeros_like_call_result_249766, result___neg___249768, lb_total_249769, ub_total_249770], **kwargs_249771)
    
    # Obtaining the member '__getitem__' of a type (line 133)
    getitem___249773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 4), step_size_to_bound_call_result_249772, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 133)
    subscript_call_result_249774 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), getitem___249773, int_249760)
    
    # Assigning a type to the variable 'tuple_var_assignment_249530' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'tuple_var_assignment_249530', subscript_call_result_249774)
    
    # Assigning a Subscript to a Name (line 133):
    
    # Obtaining the type of the subscript
    int_249775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 133)
    # Processing the call arguments (line 133)
    
    # Call to zeros_like(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'x' (line 133)
    x_249779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 52), 'x', False)
    # Processing the call keyword arguments (line 133)
    kwargs_249780 = {}
    # Getting the type of 'np' (line 133)
    np_249777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 38), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 133)
    zeros_like_249778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 38), np_249777, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 133)
    zeros_like_call_result_249781 = invoke(stypy.reporting.localization.Localization(__file__, 133, 38), zeros_like_249778, *[x_249779], **kwargs_249780)
    
    
    # Getting the type of 'g' (line 133)
    g_249782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 57), 'g', False)
    # Applying the 'usub' unary operator (line 133)
    result___neg___249783 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 56), 'usub', g_249782)
    
    # Getting the type of 'lb_total' (line 133)
    lb_total_249784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 60), 'lb_total', False)
    # Getting the type of 'ub_total' (line 133)
    ub_total_249785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 70), 'ub_total', False)
    # Processing the call keyword arguments (line 133)
    kwargs_249786 = {}
    # Getting the type of 'step_size_to_bound' (line 133)
    step_size_to_bound_249776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 133)
    step_size_to_bound_call_result_249787 = invoke(stypy.reporting.localization.Localization(__file__, 133, 19), step_size_to_bound_249776, *[zeros_like_call_result_249781, result___neg___249783, lb_total_249784, ub_total_249785], **kwargs_249786)
    
    # Obtaining the member '__getitem__' of a type (line 133)
    getitem___249788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 4), step_size_to_bound_call_result_249787, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 133)
    subscript_call_result_249789 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), getitem___249788, int_249775)
    
    # Assigning a type to the variable 'tuple_var_assignment_249531' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'tuple_var_assignment_249531', subscript_call_result_249789)
    
    # Assigning a Name to a Name (line 133):
    # Getting the type of 'tuple_var_assignment_249530' (line 133)
    tuple_var_assignment_249530_249790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'tuple_var_assignment_249530')
    # Assigning a type to the variable 'to_bounds' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'to_bounds', tuple_var_assignment_249530_249790)
    
    # Assigning a Name to a Name (line 133):
    # Getting the type of 'tuple_var_assignment_249531' (line 133)
    tuple_var_assignment_249531_249791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'tuple_var_assignment_249531')
    # Assigning a type to the variable '_' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), '_', tuple_var_assignment_249531_249791)
    
    # Assigning a BinOp to a Name (line 140):
    
    # Assigning a BinOp to a Name (line 140):
    
    
    # Obtaining the type of the subscript
    int_249792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 61), 'int')
    
    # Call to minimize_quadratic_1d(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'a' (line 140)
    a_249794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 41), 'a', False)
    # Getting the type of 'b' (line 140)
    b_249795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 44), 'b', False)
    int_249796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 47), 'int')
    # Getting the type of 'to_bounds' (line 140)
    to_bounds_249797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 50), 'to_bounds', False)
    # Processing the call keyword arguments (line 140)
    kwargs_249798 = {}
    # Getting the type of 'minimize_quadratic_1d' (line 140)
    minimize_quadratic_1d_249793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 19), 'minimize_quadratic_1d', False)
    # Calling minimize_quadratic_1d(args, kwargs) (line 140)
    minimize_quadratic_1d_call_result_249799 = invoke(stypy.reporting.localization.Localization(__file__, 140, 19), minimize_quadratic_1d_249793, *[a_249794, b_249795, int_249796, to_bounds_249797], **kwargs_249798)
    
    # Obtaining the member '__getitem__' of a type (line 140)
    getitem___249800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 19), minimize_quadratic_1d_call_result_249799, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 140)
    subscript_call_result_249801 = invoke(stypy.reporting.localization.Localization(__file__, 140, 19), getitem___249800, int_249792)
    
    # Applying the 'usub' unary operator (line 140)
    result___neg___249802 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 18), 'usub', subscript_call_result_249801)
    
    # Getting the type of 'g' (line 140)
    g_249803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 66), 'g')
    # Applying the binary operator '*' (line 140)
    result_mul_249804 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 18), '*', result___neg___249802, g_249803)
    
    # Assigning a type to the variable 'cauchy_step' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'cauchy_step', result_mul_249804)
    
    # Assigning a BinOp to a Name (line 142):
    
    # Assigning a BinOp to a Name (line 142):
    # Getting the type of 'newton_step' (line 142)
    newton_step_249805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'newton_step')
    # Getting the type of 'cauchy_step' (line 142)
    cauchy_step_249806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), 'cauchy_step')
    # Applying the binary operator '-' (line 142)
    result_sub_249807 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 16), '-', newton_step_249805, cauchy_step_249806)
    
    # Assigning a type to the variable 'step_diff' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'step_diff', result_sub_249807)
    
    # Assigning a Call to a Tuple (line 143):
    
    # Assigning a Subscript to a Name (line 143):
    
    # Obtaining the type of the subscript
    int_249808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'cauchy_step' (line 143)
    cauchy_step_249810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 41), 'cauchy_step', False)
    # Getting the type of 'step_diff' (line 143)
    step_diff_249811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 54), 'step_diff', False)
    # Getting the type of 'lb_total' (line 144)
    lb_total_249812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 41), 'lb_total', False)
    # Getting the type of 'ub_total' (line 144)
    ub_total_249813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 51), 'ub_total', False)
    # Processing the call keyword arguments (line 143)
    kwargs_249814 = {}
    # Getting the type of 'step_size_to_bound' (line 143)
    step_size_to_bound_249809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 22), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 143)
    step_size_to_bound_call_result_249815 = invoke(stypy.reporting.localization.Localization(__file__, 143, 22), step_size_to_bound_249809, *[cauchy_step_249810, step_diff_249811, lb_total_249812, ub_total_249813], **kwargs_249814)
    
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___249816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 4), step_size_to_bound_call_result_249815, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_249817 = invoke(stypy.reporting.localization.Localization(__file__, 143, 4), getitem___249816, int_249808)
    
    # Assigning a type to the variable 'tuple_var_assignment_249532' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_249532', subscript_call_result_249817)
    
    # Assigning a Subscript to a Name (line 143):
    
    # Obtaining the type of the subscript
    int_249818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 4), 'int')
    
    # Call to step_size_to_bound(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'cauchy_step' (line 143)
    cauchy_step_249820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 41), 'cauchy_step', False)
    # Getting the type of 'step_diff' (line 143)
    step_diff_249821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 54), 'step_diff', False)
    # Getting the type of 'lb_total' (line 144)
    lb_total_249822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 41), 'lb_total', False)
    # Getting the type of 'ub_total' (line 144)
    ub_total_249823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 51), 'ub_total', False)
    # Processing the call keyword arguments (line 143)
    kwargs_249824 = {}
    # Getting the type of 'step_size_to_bound' (line 143)
    step_size_to_bound_249819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 22), 'step_size_to_bound', False)
    # Calling step_size_to_bound(args, kwargs) (line 143)
    step_size_to_bound_call_result_249825 = invoke(stypy.reporting.localization.Localization(__file__, 143, 22), step_size_to_bound_249819, *[cauchy_step_249820, step_diff_249821, lb_total_249822, ub_total_249823], **kwargs_249824)
    
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___249826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 4), step_size_to_bound_call_result_249825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_249827 = invoke(stypy.reporting.localization.Localization(__file__, 143, 4), getitem___249826, int_249818)
    
    # Assigning a type to the variable 'tuple_var_assignment_249533' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_249533', subscript_call_result_249827)
    
    # Assigning a Name to a Name (line 143):
    # Getting the type of 'tuple_var_assignment_249532' (line 143)
    tuple_var_assignment_249532_249828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_249532')
    # Assigning a type to the variable 'step_size' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'step_size', tuple_var_assignment_249532_249828)
    
    # Assigning a Name to a Name (line 143):
    # Getting the type of 'tuple_var_assignment_249533' (line 143)
    tuple_var_assignment_249533_249829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'tuple_var_assignment_249533')
    # Assigning a type to the variable 'hits' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'hits', tuple_var_assignment_249533_249829)
    
    # Assigning a Num to a Subscript (line 145):
    
    # Assigning a Num to a Subscript (line 145):
    int_249830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 38), 'int')
    # Getting the type of 'bound_hits' (line 145)
    bound_hits_249831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'bound_hits')
    
    # Getting the type of 'hits' (line 145)
    hits_249832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'hits')
    int_249833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 23), 'int')
    # Applying the binary operator '<' (line 145)
    result_lt_249834 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 16), '<', hits_249832, int_249833)
    
    # Getting the type of 'orig_l' (line 145)
    orig_l_249835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 28), 'orig_l')
    # Applying the binary operator '&' (line 145)
    result_and__249836 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 15), '&', result_lt_249834, orig_l_249835)
    
    # Storing an element on a container (line 145)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 4), bound_hits_249831, (result_and__249836, int_249830))
    
    # Assigning a Num to a Subscript (line 146):
    
    # Assigning a Num to a Subscript (line 146):
    int_249837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 38), 'int')
    # Getting the type of 'bound_hits' (line 146)
    bound_hits_249838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'bound_hits')
    
    # Getting the type of 'hits' (line 146)
    hits_249839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'hits')
    int_249840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 23), 'int')
    # Applying the binary operator '>' (line 146)
    result_gt_249841 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 16), '>', hits_249839, int_249840)
    
    # Getting the type of 'orig_u' (line 146)
    orig_u_249842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 28), 'orig_u')
    # Applying the binary operator '&' (line 146)
    result_and__249843 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 15), '&', result_gt_249841, orig_u_249842)
    
    # Storing an element on a container (line 146)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 4), bound_hits_249838, (result_and__249843, int_249837))
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 147):
    
    # Call to any(...): (line 147)
    # Processing the call arguments (line 147)
    
    # Getting the type of 'hits' (line 147)
    hits_249846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'hits', False)
    int_249847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 28), 'int')
    # Applying the binary operator '<' (line 147)
    result_lt_249848 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 21), '<', hits_249846, int_249847)
    
    # Getting the type of 'tr_l' (line 147)
    tr_l_249849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 33), 'tr_l', False)
    # Applying the binary operator '&' (line 147)
    result_and__249850 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 20), '&', result_lt_249848, tr_l_249849)
    
    
    # Getting the type of 'hits' (line 147)
    hits_249851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 41), 'hits', False)
    int_249852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 48), 'int')
    # Applying the binary operator '>' (line 147)
    result_gt_249853 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 41), '>', hits_249851, int_249852)
    
    # Getting the type of 'tr_u' (line 147)
    tr_u_249854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 53), 'tr_u', False)
    # Applying the binary operator '&' (line 147)
    result_and__249855 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 40), '&', result_gt_249853, tr_u_249854)
    
    # Applying the binary operator '|' (line 147)
    result_or__249856 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 20), '|', result_and__249850, result_and__249855)
    
    # Processing the call keyword arguments (line 147)
    kwargs_249857 = {}
    # Getting the type of 'np' (line 147)
    np_249844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 13), 'np', False)
    # Obtaining the member 'any' of a type (line 147)
    any_249845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 13), np_249844, 'any')
    # Calling any(args, kwargs) (line 147)
    any_call_result_249858 = invoke(stypy.reporting.localization.Localization(__file__, 147, 13), any_249845, *[result_or__249856], **kwargs_249857)
    
    # Assigning a type to the variable 'tr_hit' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'tr_hit', any_call_result_249858)
    
    # Obtaining an instance of the builtin type 'tuple' (line 149)
    tuple_249859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 149)
    # Adding element type (line 149)
    # Getting the type of 'cauchy_step' (line 149)
    cauchy_step_249860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'cauchy_step')
    # Getting the type of 'step_size' (line 149)
    step_size_249861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'step_size')
    # Getting the type of 'step_diff' (line 149)
    step_diff_249862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'step_diff')
    # Applying the binary operator '*' (line 149)
    result_mul_249863 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 25), '*', step_size_249861, step_diff_249862)
    
    # Applying the binary operator '+' (line 149)
    result_add_249864 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), '+', cauchy_step_249860, result_mul_249863)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 11), tuple_249859, result_add_249864)
    # Adding element type (line 149)
    # Getting the type of 'bound_hits' (line 149)
    bound_hits_249865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 48), 'bound_hits')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 11), tuple_249859, bound_hits_249865)
    # Adding element type (line 149)
    # Getting the type of 'tr_hit' (line 149)
    tr_hit_249866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 60), 'tr_hit')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 11), tuple_249859, tr_hit_249866)
    
    # Assigning a type to the variable 'stypy_return_type' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type', tuple_249859)
    
    # ################# End of 'dogleg_step(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dogleg_step' in the type store
    # Getting the type of 'stypy_return_type' (line 109)
    stypy_return_type_249867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_249867)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dogleg_step'
    return stypy_return_type_249867

# Assigning a type to the variable 'dogleg_step' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'dogleg_step', dogleg_step)

@norecursion
def dogbox(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dogbox'
    module_type_store = module_type_store.open_function_context('dogbox', 152, 0, False)
    
    # Passed parameters checking function
    dogbox.stypy_localization = localization
    dogbox.stypy_type_of_self = None
    dogbox.stypy_type_store = module_type_store
    dogbox.stypy_function_name = 'dogbox'
    dogbox.stypy_param_names_list = ['fun', 'jac', 'x0', 'f0', 'J0', 'lb', 'ub', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'loss_function', 'tr_solver', 'tr_options', 'verbose']
    dogbox.stypy_varargs_param_name = None
    dogbox.stypy_kwargs_param_name = None
    dogbox.stypy_call_defaults = defaults
    dogbox.stypy_call_varargs = varargs
    dogbox.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dogbox', ['fun', 'jac', 'x0', 'f0', 'J0', 'lb', 'ub', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'loss_function', 'tr_solver', 'tr_options', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dogbox', localization, ['fun', 'jac', 'x0', 'f0', 'J0', 'lb', 'ub', 'ftol', 'xtol', 'gtol', 'max_nfev', 'x_scale', 'loss_function', 'tr_solver', 'tr_options', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dogbox(...)' code ##################

    
    # Assigning a Name to a Name (line 154):
    
    # Assigning a Name to a Name (line 154):
    # Getting the type of 'f0' (line 154)
    f0_249868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'f0')
    # Assigning a type to the variable 'f' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'f', f0_249868)
    
    # Assigning a Call to a Name (line 155):
    
    # Assigning a Call to a Name (line 155):
    
    # Call to copy(...): (line 155)
    # Processing the call keyword arguments (line 155)
    kwargs_249871 = {}
    # Getting the type of 'f' (line 155)
    f_249869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 13), 'f', False)
    # Obtaining the member 'copy' of a type (line 155)
    copy_249870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 13), f_249869, 'copy')
    # Calling copy(args, kwargs) (line 155)
    copy_call_result_249872 = invoke(stypy.reporting.localization.Localization(__file__, 155, 13), copy_249870, *[], **kwargs_249871)
    
    # Assigning a type to the variable 'f_true' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'f_true', copy_call_result_249872)
    
    # Assigning a Num to a Name (line 156):
    
    # Assigning a Num to a Name (line 156):
    int_249873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 11), 'int')
    # Assigning a type to the variable 'nfev' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'nfev', int_249873)
    
    # Assigning a Name to a Name (line 158):
    
    # Assigning a Name to a Name (line 158):
    # Getting the type of 'J0' (line 158)
    J0_249874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'J0')
    # Assigning a type to the variable 'J' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'J', J0_249874)
    
    # Assigning a Num to a Name (line 159):
    
    # Assigning a Num to a Name (line 159):
    int_249875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 11), 'int')
    # Assigning a type to the variable 'njev' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'njev', int_249875)
    
    # Type idiom detected: calculating its left and rigth part (line 161)
    # Getting the type of 'loss_function' (line 161)
    loss_function_249876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'loss_function')
    # Getting the type of 'None' (line 161)
    None_249877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'None')
    
    (may_be_249878, more_types_in_union_249879) = may_not_be_none(loss_function_249876, None_249877)

    if may_be_249878:

        if more_types_in_union_249879:
            # Runtime conditional SSA (line 161)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 162):
        
        # Assigning a Call to a Name (line 162):
        
        # Call to loss_function(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'f' (line 162)
        f_249881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 28), 'f', False)
        # Processing the call keyword arguments (line 162)
        kwargs_249882 = {}
        # Getting the type of 'loss_function' (line 162)
        loss_function_249880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 14), 'loss_function', False)
        # Calling loss_function(args, kwargs) (line 162)
        loss_function_call_result_249883 = invoke(stypy.reporting.localization.Localization(__file__, 162, 14), loss_function_249880, *[f_249881], **kwargs_249882)
        
        # Assigning a type to the variable 'rho' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'rho', loss_function_call_result_249883)
        
        # Assigning a BinOp to a Name (line 163):
        
        # Assigning a BinOp to a Name (line 163):
        float_249884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 15), 'float')
        
        # Call to sum(...): (line 163)
        # Processing the call arguments (line 163)
        
        # Obtaining the type of the subscript
        int_249887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 32), 'int')
        # Getting the type of 'rho' (line 163)
        rho_249888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'rho', False)
        # Obtaining the member '__getitem__' of a type (line 163)
        getitem___249889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 28), rho_249888, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 163)
        subscript_call_result_249890 = invoke(stypy.reporting.localization.Localization(__file__, 163, 28), getitem___249889, int_249887)
        
        # Processing the call keyword arguments (line 163)
        kwargs_249891 = {}
        # Getting the type of 'np' (line 163)
        np_249885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 21), 'np', False)
        # Obtaining the member 'sum' of a type (line 163)
        sum_249886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 21), np_249885, 'sum')
        # Calling sum(args, kwargs) (line 163)
        sum_call_result_249892 = invoke(stypy.reporting.localization.Localization(__file__, 163, 21), sum_249886, *[subscript_call_result_249890], **kwargs_249891)
        
        # Applying the binary operator '*' (line 163)
        result_mul_249893 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 15), '*', float_249884, sum_call_result_249892)
        
        # Assigning a type to the variable 'cost' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'cost', result_mul_249893)
        
        # Assigning a Call to a Tuple (line 164):
        
        # Assigning a Subscript to a Name (line 164):
        
        # Obtaining the type of the subscript
        int_249894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 8), 'int')
        
        # Call to scale_for_robust_loss_function(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'J' (line 164)
        J_249896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 46), 'J', False)
        # Getting the type of 'f' (line 164)
        f_249897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 49), 'f', False)
        # Getting the type of 'rho' (line 164)
        rho_249898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 52), 'rho', False)
        # Processing the call keyword arguments (line 164)
        kwargs_249899 = {}
        # Getting the type of 'scale_for_robust_loss_function' (line 164)
        scale_for_robust_loss_function_249895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'scale_for_robust_loss_function', False)
        # Calling scale_for_robust_loss_function(args, kwargs) (line 164)
        scale_for_robust_loss_function_call_result_249900 = invoke(stypy.reporting.localization.Localization(__file__, 164, 15), scale_for_robust_loss_function_249895, *[J_249896, f_249897, rho_249898], **kwargs_249899)
        
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___249901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), scale_for_robust_loss_function_call_result_249900, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_249902 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), getitem___249901, int_249894)
        
        # Assigning a type to the variable 'tuple_var_assignment_249534' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_249534', subscript_call_result_249902)
        
        # Assigning a Subscript to a Name (line 164):
        
        # Obtaining the type of the subscript
        int_249903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 8), 'int')
        
        # Call to scale_for_robust_loss_function(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'J' (line 164)
        J_249905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 46), 'J', False)
        # Getting the type of 'f' (line 164)
        f_249906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 49), 'f', False)
        # Getting the type of 'rho' (line 164)
        rho_249907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 52), 'rho', False)
        # Processing the call keyword arguments (line 164)
        kwargs_249908 = {}
        # Getting the type of 'scale_for_robust_loss_function' (line 164)
        scale_for_robust_loss_function_249904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'scale_for_robust_loss_function', False)
        # Calling scale_for_robust_loss_function(args, kwargs) (line 164)
        scale_for_robust_loss_function_call_result_249909 = invoke(stypy.reporting.localization.Localization(__file__, 164, 15), scale_for_robust_loss_function_249904, *[J_249905, f_249906, rho_249907], **kwargs_249908)
        
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___249910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), scale_for_robust_loss_function_call_result_249909, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_249911 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), getitem___249910, int_249903)
        
        # Assigning a type to the variable 'tuple_var_assignment_249535' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_249535', subscript_call_result_249911)
        
        # Assigning a Name to a Name (line 164):
        # Getting the type of 'tuple_var_assignment_249534' (line 164)
        tuple_var_assignment_249534_249912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_249534')
        # Assigning a type to the variable 'J' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'J', tuple_var_assignment_249534_249912)
        
        # Assigning a Name to a Name (line 164):
        # Getting the type of 'tuple_var_assignment_249535' (line 164)
        tuple_var_assignment_249535_249913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_249535')
        # Assigning a type to the variable 'f' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'f', tuple_var_assignment_249535_249913)

        if more_types_in_union_249879:
            # Runtime conditional SSA for else branch (line 161)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_249878) or more_types_in_union_249879):
        
        # Assigning a BinOp to a Name (line 166):
        
        # Assigning a BinOp to a Name (line 166):
        float_249914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 15), 'float')
        
        # Call to dot(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'f' (line 166)
        f_249917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'f', False)
        # Getting the type of 'f' (line 166)
        f_249918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 31), 'f', False)
        # Processing the call keyword arguments (line 166)
        kwargs_249919 = {}
        # Getting the type of 'np' (line 166)
        np_249915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'np', False)
        # Obtaining the member 'dot' of a type (line 166)
        dot_249916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 21), np_249915, 'dot')
        # Calling dot(args, kwargs) (line 166)
        dot_call_result_249920 = invoke(stypy.reporting.localization.Localization(__file__, 166, 21), dot_249916, *[f_249917, f_249918], **kwargs_249919)
        
        # Applying the binary operator '*' (line 166)
        result_mul_249921 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 15), '*', float_249914, dot_call_result_249920)
        
        # Assigning a type to the variable 'cost' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'cost', result_mul_249921)

        if (may_be_249878 and more_types_in_union_249879):
            # SSA join for if statement (line 161)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to compute_grad(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'J' (line 168)
    J_249923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'J', False)
    # Getting the type of 'f' (line 168)
    f_249924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'f', False)
    # Processing the call keyword arguments (line 168)
    kwargs_249925 = {}
    # Getting the type of 'compute_grad' (line 168)
    compute_grad_249922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'compute_grad', False)
    # Calling compute_grad(args, kwargs) (line 168)
    compute_grad_call_result_249926 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), compute_grad_249922, *[J_249923, f_249924], **kwargs_249925)
    
    # Assigning a type to the variable 'g' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'g', compute_grad_call_result_249926)
    
    # Assigning a BoolOp to a Name (line 170):
    
    # Assigning a BoolOp to a Name (line 170):
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'x_scale' (line 170)
    x_scale_249928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'x_scale', False)
    # Getting the type of 'string_types' (line 170)
    string_types_249929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 36), 'string_types', False)
    # Processing the call keyword arguments (line 170)
    kwargs_249930 = {}
    # Getting the type of 'isinstance' (line 170)
    isinstance_249927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 170)
    isinstance_call_result_249931 = invoke(stypy.reporting.localization.Localization(__file__, 170, 16), isinstance_249927, *[x_scale_249928, string_types_249929], **kwargs_249930)
    
    
    # Getting the type of 'x_scale' (line 170)
    x_scale_249932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 54), 'x_scale')
    str_249933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 65), 'str', 'jac')
    # Applying the binary operator '==' (line 170)
    result_eq_249934 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 54), '==', x_scale_249932, str_249933)
    
    # Applying the binary operator 'and' (line 170)
    result_and_keyword_249935 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 16), 'and', isinstance_call_result_249931, result_eq_249934)
    
    # Assigning a type to the variable 'jac_scale' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'jac_scale', result_and_keyword_249935)
    
    # Getting the type of 'jac_scale' (line 171)
    jac_scale_249936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 7), 'jac_scale')
    # Testing the type of an if condition (line 171)
    if_condition_249937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 4), jac_scale_249936)
    # Assigning a type to the variable 'if_condition_249937' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'if_condition_249937', if_condition_249937)
    # SSA begins for if statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 172):
    
    # Assigning a Subscript to a Name (line 172):
    
    # Obtaining the type of the subscript
    int_249938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 8), 'int')
    
    # Call to compute_jac_scale(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'J' (line 172)
    J_249940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 45), 'J', False)
    # Processing the call keyword arguments (line 172)
    kwargs_249941 = {}
    # Getting the type of 'compute_jac_scale' (line 172)
    compute_jac_scale_249939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 27), 'compute_jac_scale', False)
    # Calling compute_jac_scale(args, kwargs) (line 172)
    compute_jac_scale_call_result_249942 = invoke(stypy.reporting.localization.Localization(__file__, 172, 27), compute_jac_scale_249939, *[J_249940], **kwargs_249941)
    
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___249943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), compute_jac_scale_call_result_249942, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_249944 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), getitem___249943, int_249938)
    
    # Assigning a type to the variable 'tuple_var_assignment_249536' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'tuple_var_assignment_249536', subscript_call_result_249944)
    
    # Assigning a Subscript to a Name (line 172):
    
    # Obtaining the type of the subscript
    int_249945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 8), 'int')
    
    # Call to compute_jac_scale(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'J' (line 172)
    J_249947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 45), 'J', False)
    # Processing the call keyword arguments (line 172)
    kwargs_249948 = {}
    # Getting the type of 'compute_jac_scale' (line 172)
    compute_jac_scale_249946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 27), 'compute_jac_scale', False)
    # Calling compute_jac_scale(args, kwargs) (line 172)
    compute_jac_scale_call_result_249949 = invoke(stypy.reporting.localization.Localization(__file__, 172, 27), compute_jac_scale_249946, *[J_249947], **kwargs_249948)
    
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___249950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), compute_jac_scale_call_result_249949, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_249951 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), getitem___249950, int_249945)
    
    # Assigning a type to the variable 'tuple_var_assignment_249537' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'tuple_var_assignment_249537', subscript_call_result_249951)
    
    # Assigning a Name to a Name (line 172):
    # Getting the type of 'tuple_var_assignment_249536' (line 172)
    tuple_var_assignment_249536_249952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'tuple_var_assignment_249536')
    # Assigning a type to the variable 'scale' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'scale', tuple_var_assignment_249536_249952)
    
    # Assigning a Name to a Name (line 172):
    # Getting the type of 'tuple_var_assignment_249537' (line 172)
    tuple_var_assignment_249537_249953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'tuple_var_assignment_249537')
    # Assigning a type to the variable 'scale_inv' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'scale_inv', tuple_var_assignment_249537_249953)
    # SSA branch for the else part of an if statement (line 171)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Tuple (line 174):
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'x_scale' (line 174)
    x_scale_249954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'x_scale')
    # Assigning a type to the variable 'tuple_assignment_249538' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_assignment_249538', x_scale_249954)
    
    # Assigning a BinOp to a Name (line 174):
    int_249955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 36), 'int')
    # Getting the type of 'x_scale' (line 174)
    x_scale_249956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 40), 'x_scale')
    # Applying the binary operator 'div' (line 174)
    result_div_249957 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 36), 'div', int_249955, x_scale_249956)
    
    # Assigning a type to the variable 'tuple_assignment_249539' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_assignment_249539', result_div_249957)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_assignment_249538' (line 174)
    tuple_assignment_249538_249958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_assignment_249538')
    # Assigning a type to the variable 'scale' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'scale', tuple_assignment_249538_249958)
    
    # Assigning a Name to a Name (line 174):
    # Getting the type of 'tuple_assignment_249539' (line 174)
    tuple_assignment_249539_249959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_assignment_249539')
    # Assigning a type to the variable 'scale_inv' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'scale_inv', tuple_assignment_249539_249959)
    # SSA join for if statement (line 171)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 176):
    
    # Assigning a Call to a Name (line 176):
    
    # Call to norm(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'x0' (line 176)
    x0_249961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 17), 'x0', False)
    # Getting the type of 'scale_inv' (line 176)
    scale_inv_249962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 22), 'scale_inv', False)
    # Applying the binary operator '*' (line 176)
    result_mul_249963 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 17), '*', x0_249961, scale_inv_249962)
    
    # Processing the call keyword arguments (line 176)
    # Getting the type of 'np' (line 176)
    np_249964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 37), 'np', False)
    # Obtaining the member 'inf' of a type (line 176)
    inf_249965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 37), np_249964, 'inf')
    keyword_249966 = inf_249965
    kwargs_249967 = {'ord': keyword_249966}
    # Getting the type of 'norm' (line 176)
    norm_249960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'norm', False)
    # Calling norm(args, kwargs) (line 176)
    norm_call_result_249968 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), norm_249960, *[result_mul_249963], **kwargs_249967)
    
    # Assigning a type to the variable 'Delta' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'Delta', norm_call_result_249968)
    
    
    # Getting the type of 'Delta' (line 177)
    Delta_249969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 7), 'Delta')
    int_249970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 16), 'int')
    # Applying the binary operator '==' (line 177)
    result_eq_249971 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 7), '==', Delta_249969, int_249970)
    
    # Testing the type of an if condition (line 177)
    if_condition_249972 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 4), result_eq_249971)
    # Assigning a type to the variable 'if_condition_249972' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'if_condition_249972', if_condition_249972)
    # SSA begins for if statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 178):
    
    # Assigning a Num to a Name (line 178):
    float_249973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 16), 'float')
    # Assigning a type to the variable 'Delta' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'Delta', float_249973)
    # SSA join for if statement (line 177)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to zeros_like(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'x0' (line 180)
    x0_249976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 29), 'x0', False)
    # Processing the call keyword arguments (line 180)
    # Getting the type of 'int' (line 180)
    int_249977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 39), 'int', False)
    keyword_249978 = int_249977
    kwargs_249979 = {'dtype': keyword_249978}
    # Getting the type of 'np' (line 180)
    np_249974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 180)
    zeros_like_249975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 15), np_249974, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 180)
    zeros_like_call_result_249980 = invoke(stypy.reporting.localization.Localization(__file__, 180, 15), zeros_like_249975, *[x0_249976], **kwargs_249979)
    
    # Assigning a type to the variable 'on_bound' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'on_bound', zeros_like_call_result_249980)
    
    # Assigning a Num to a Subscript (line 181):
    
    # Assigning a Num to a Subscript (line 181):
    int_249981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 33), 'int')
    # Getting the type of 'on_bound' (line 181)
    on_bound_249982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'on_bound')
    
    # Call to equal(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'x0' (line 181)
    x0_249985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 22), 'x0', False)
    # Getting the type of 'lb' (line 181)
    lb_249986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 26), 'lb', False)
    # Processing the call keyword arguments (line 181)
    kwargs_249987 = {}
    # Getting the type of 'np' (line 181)
    np_249983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 13), 'np', False)
    # Obtaining the member 'equal' of a type (line 181)
    equal_249984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 13), np_249983, 'equal')
    # Calling equal(args, kwargs) (line 181)
    equal_call_result_249988 = invoke(stypy.reporting.localization.Localization(__file__, 181, 13), equal_249984, *[x0_249985, lb_249986], **kwargs_249987)
    
    # Storing an element on a container (line 181)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 4), on_bound_249982, (equal_call_result_249988, int_249981))
    
    # Assigning a Num to a Subscript (line 182):
    
    # Assigning a Num to a Subscript (line 182):
    int_249989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 33), 'int')
    # Getting the type of 'on_bound' (line 182)
    on_bound_249990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'on_bound')
    
    # Call to equal(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'x0' (line 182)
    x0_249993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'x0', False)
    # Getting the type of 'ub' (line 182)
    ub_249994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 26), 'ub', False)
    # Processing the call keyword arguments (line 182)
    kwargs_249995 = {}
    # Getting the type of 'np' (line 182)
    np_249991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 13), 'np', False)
    # Obtaining the member 'equal' of a type (line 182)
    equal_249992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 13), np_249991, 'equal')
    # Calling equal(args, kwargs) (line 182)
    equal_call_result_249996 = invoke(stypy.reporting.localization.Localization(__file__, 182, 13), equal_249992, *[x0_249993, ub_249994], **kwargs_249995)
    
    # Storing an element on a container (line 182)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 4), on_bound_249990, (equal_call_result_249996, int_249989))
    
    # Assigning a Name to a Name (line 184):
    
    # Assigning a Name to a Name (line 184):
    # Getting the type of 'x0' (line 184)
    x0_249997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'x0')
    # Assigning a type to the variable 'x' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'x', x0_249997)
    
    # Assigning a Call to a Name (line 185):
    
    # Assigning a Call to a Name (line 185):
    
    # Call to empty_like(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'x0' (line 185)
    x0_250000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 25), 'x0', False)
    # Processing the call keyword arguments (line 185)
    kwargs_250001 = {}
    # Getting the type of 'np' (line 185)
    np_249998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 185)
    empty_like_249999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 11), np_249998, 'empty_like')
    # Calling empty_like(args, kwargs) (line 185)
    empty_like_call_result_250002 = invoke(stypy.reporting.localization.Localization(__file__, 185, 11), empty_like_249999, *[x0_250000], **kwargs_250001)
    
    # Assigning a type to the variable 'step' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'step', empty_like_call_result_250002)
    
    # Type idiom detected: calculating its left and rigth part (line 187)
    # Getting the type of 'max_nfev' (line 187)
    max_nfev_250003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 7), 'max_nfev')
    # Getting the type of 'None' (line 187)
    None_250004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 'None')
    
    (may_be_250005, more_types_in_union_250006) = may_be_none(max_nfev_250003, None_250004)

    if may_be_250005:

        if more_types_in_union_250006:
            # Runtime conditional SSA (line 187)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 188):
        
        # Assigning a BinOp to a Name (line 188):
        # Getting the type of 'x0' (line 188)
        x0_250007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 19), 'x0')
        # Obtaining the member 'size' of a type (line 188)
        size_250008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 19), x0_250007, 'size')
        int_250009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 29), 'int')
        # Applying the binary operator '*' (line 188)
        result_mul_250010 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 19), '*', size_250008, int_250009)
        
        # Assigning a type to the variable 'max_nfev' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'max_nfev', result_mul_250010)

        if more_types_in_union_250006:
            # SSA join for if statement (line 187)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Name (line 190):
    
    # Assigning a Name to a Name (line 190):
    # Getting the type of 'None' (line 190)
    None_250011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), 'None')
    # Assigning a type to the variable 'termination_status' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'termination_status', None_250011)
    
    # Assigning a Num to a Name (line 191):
    
    # Assigning a Num to a Name (line 191):
    int_250012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 16), 'int')
    # Assigning a type to the variable 'iteration' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'iteration', int_250012)
    
    # Assigning a Name to a Name (line 192):
    
    # Assigning a Name to a Name (line 192):
    # Getting the type of 'None' (line 192)
    None_250013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'None')
    # Assigning a type to the variable 'step_norm' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'step_norm', None_250013)
    
    # Assigning a Name to a Name (line 193):
    
    # Assigning a Name to a Name (line 193):
    # Getting the type of 'None' (line 193)
    None_250014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'None')
    # Assigning a type to the variable 'actual_reduction' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'actual_reduction', None_250014)
    
    
    # Getting the type of 'verbose' (line 195)
    verbose_250015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 7), 'verbose')
    int_250016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 18), 'int')
    # Applying the binary operator '==' (line 195)
    result_eq_250017 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 7), '==', verbose_250015, int_250016)
    
    # Testing the type of an if condition (line 195)
    if_condition_250018 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 4), result_eq_250017)
    # Assigning a type to the variable 'if_condition_250018' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'if_condition_250018', if_condition_250018)
    # SSA begins for if statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print_header_nonlinear(...): (line 196)
    # Processing the call keyword arguments (line 196)
    kwargs_250020 = {}
    # Getting the type of 'print_header_nonlinear' (line 196)
    print_header_nonlinear_250019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'print_header_nonlinear', False)
    # Calling print_header_nonlinear(args, kwargs) (line 196)
    print_header_nonlinear_call_result_250021 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), print_header_nonlinear_250019, *[], **kwargs_250020)
    
    # SSA join for if statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'True' (line 198)
    True_250022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 10), 'True')
    # Testing the type of an if condition (line 198)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 4), True_250022)
    # SSA begins for while statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Compare to a Name (line 199):
    
    # Assigning a Compare to a Name (line 199):
    
    # Getting the type of 'on_bound' (line 199)
    on_bound_250023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'on_bound')
    # Getting the type of 'g' (line 199)
    g_250024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 32), 'g')
    # Applying the binary operator '*' (line 199)
    result_mul_250025 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 21), '*', on_bound_250023, g_250024)
    
    int_250026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 36), 'int')
    # Applying the binary operator '<' (line 199)
    result_lt_250027 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 21), '<', result_mul_250025, int_250026)
    
    # Assigning a type to the variable 'active_set' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'active_set', result_lt_250027)
    
    # Assigning a UnaryOp to a Name (line 200):
    
    # Assigning a UnaryOp to a Name (line 200):
    
    # Getting the type of 'active_set' (line 200)
    active_set_250028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'active_set')
    # Applying the '~' unary operator (line 200)
    result_inv_250029 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 19), '~', active_set_250028)
    
    # Assigning a type to the variable 'free_set' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'free_set', result_inv_250029)
    
    # Assigning a Subscript to a Name (line 202):
    
    # Assigning a Subscript to a Name (line 202):
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 202)
    free_set_250030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'free_set')
    # Getting the type of 'g' (line 202)
    g_250031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 17), 'g')
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___250032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 17), g_250031, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_250033 = invoke(stypy.reporting.localization.Localization(__file__, 202, 17), getitem___250032, free_set_250030)
    
    # Assigning a type to the variable 'g_free' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'g_free', subscript_call_result_250033)
    
    # Assigning a Call to a Name (line 203):
    
    # Assigning a Call to a Name (line 203):
    
    # Call to copy(...): (line 203)
    # Processing the call keyword arguments (line 203)
    kwargs_250036 = {}
    # Getting the type of 'g' (line 203)
    g_250034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 17), 'g', False)
    # Obtaining the member 'copy' of a type (line 203)
    copy_250035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 17), g_250034, 'copy')
    # Calling copy(args, kwargs) (line 203)
    copy_call_result_250037 = invoke(stypy.reporting.localization.Localization(__file__, 203, 17), copy_250035, *[], **kwargs_250036)
    
    # Assigning a type to the variable 'g_full' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'g_full', copy_call_result_250037)
    
    # Assigning a Num to a Subscript (line 204):
    
    # Assigning a Num to a Subscript (line 204):
    int_250038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 24), 'int')
    # Getting the type of 'g' (line 204)
    g_250039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'g')
    # Getting the type of 'active_set' (line 204)
    active_set_250040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 10), 'active_set')
    # Storing an element on a container (line 204)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 8), g_250039, (active_set_250040, int_250038))
    
    # Assigning a Call to a Name (line 206):
    
    # Assigning a Call to a Name (line 206):
    
    # Call to norm(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'g' (line 206)
    g_250042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 22), 'g', False)
    # Processing the call keyword arguments (line 206)
    # Getting the type of 'np' (line 206)
    np_250043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 29), 'np', False)
    # Obtaining the member 'inf' of a type (line 206)
    inf_250044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 29), np_250043, 'inf')
    keyword_250045 = inf_250044
    kwargs_250046 = {'ord': keyword_250045}
    # Getting the type of 'norm' (line 206)
    norm_250041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'norm', False)
    # Calling norm(args, kwargs) (line 206)
    norm_call_result_250047 = invoke(stypy.reporting.localization.Localization(__file__, 206, 17), norm_250041, *[g_250042], **kwargs_250046)
    
    # Assigning a type to the variable 'g_norm' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'g_norm', norm_call_result_250047)
    
    
    # Getting the type of 'g_norm' (line 207)
    g_norm_250048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'g_norm')
    # Getting the type of 'gtol' (line 207)
    gtol_250049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 20), 'gtol')
    # Applying the binary operator '<' (line 207)
    result_lt_250050 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 11), '<', g_norm_250048, gtol_250049)
    
    # Testing the type of an if condition (line 207)
    if_condition_250051 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 8), result_lt_250050)
    # Assigning a type to the variable 'if_condition_250051' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'if_condition_250051', if_condition_250051)
    # SSA begins for if statement (line 207)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 208):
    
    # Assigning a Num to a Name (line 208):
    int_250052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 33), 'int')
    # Assigning a type to the variable 'termination_status' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'termination_status', int_250052)
    # SSA join for if statement (line 207)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbose' (line 210)
    verbose_250053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'verbose')
    int_250054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 22), 'int')
    # Applying the binary operator '==' (line 210)
    result_eq_250055 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 11), '==', verbose_250053, int_250054)
    
    # Testing the type of an if condition (line 210)
    if_condition_250056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 8), result_eq_250055)
    # Assigning a type to the variable 'if_condition_250056' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'if_condition_250056', if_condition_250056)
    # SSA begins for if statement (line 210)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print_iteration_nonlinear(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'iteration' (line 211)
    iteration_250058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 38), 'iteration', False)
    # Getting the type of 'nfev' (line 211)
    nfev_250059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 49), 'nfev', False)
    # Getting the type of 'cost' (line 211)
    cost_250060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 55), 'cost', False)
    # Getting the type of 'actual_reduction' (line 211)
    actual_reduction_250061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 61), 'actual_reduction', False)
    # Getting the type of 'step_norm' (line 212)
    step_norm_250062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 38), 'step_norm', False)
    # Getting the type of 'g_norm' (line 212)
    g_norm_250063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 49), 'g_norm', False)
    # Processing the call keyword arguments (line 211)
    kwargs_250064 = {}
    # Getting the type of 'print_iteration_nonlinear' (line 211)
    print_iteration_nonlinear_250057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'print_iteration_nonlinear', False)
    # Calling print_iteration_nonlinear(args, kwargs) (line 211)
    print_iteration_nonlinear_call_result_250065 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), print_iteration_nonlinear_250057, *[iteration_250058, nfev_250059, cost_250060, actual_reduction_250061, step_norm_250062, g_norm_250063], **kwargs_250064)
    
    # SSA join for if statement (line 210)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'termination_status' (line 214)
    termination_status_250066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'termination_status')
    # Getting the type of 'None' (line 214)
    None_250067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 37), 'None')
    # Applying the binary operator 'isnot' (line 214)
    result_is_not_250068 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 11), 'isnot', termination_status_250066, None_250067)
    
    
    # Getting the type of 'nfev' (line 214)
    nfev_250069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 45), 'nfev')
    # Getting the type of 'max_nfev' (line 214)
    max_nfev_250070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 53), 'max_nfev')
    # Applying the binary operator '==' (line 214)
    result_eq_250071 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 45), '==', nfev_250069, max_nfev_250070)
    
    # Applying the binary operator 'or' (line 214)
    result_or_keyword_250072 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 11), 'or', result_is_not_250068, result_eq_250071)
    
    # Testing the type of an if condition (line 214)
    if_condition_250073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 8), result_or_keyword_250072)
    # Assigning a type to the variable 'if_condition_250073' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'if_condition_250073', if_condition_250073)
    # SSA begins for if statement (line 214)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 214)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 217):
    
    # Assigning a Subscript to a Name (line 217):
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 217)
    free_set_250074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 19), 'free_set')
    # Getting the type of 'x' (line 217)
    x_250075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 17), 'x')
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___250076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 17), x_250075, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_250077 = invoke(stypy.reporting.localization.Localization(__file__, 217, 17), getitem___250076, free_set_250074)
    
    # Assigning a type to the variable 'x_free' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'x_free', subscript_call_result_250077)
    
    # Assigning a Subscript to a Name (line 218):
    
    # Assigning a Subscript to a Name (line 218):
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 218)
    free_set_250078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 21), 'free_set')
    # Getting the type of 'lb' (line 218)
    lb_250079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 18), 'lb')
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___250080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 18), lb_250079, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_250081 = invoke(stypy.reporting.localization.Localization(__file__, 218, 18), getitem___250080, free_set_250078)
    
    # Assigning a type to the variable 'lb_free' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'lb_free', subscript_call_result_250081)
    
    # Assigning a Subscript to a Name (line 219):
    
    # Assigning a Subscript to a Name (line 219):
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 219)
    free_set_250082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'free_set')
    # Getting the type of 'ub' (line 219)
    ub_250083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 18), 'ub')
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___250084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 18), ub_250083, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_250085 = invoke(stypy.reporting.localization.Localization(__file__, 219, 18), getitem___250084, free_set_250082)
    
    # Assigning a type to the variable 'ub_free' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'ub_free', subscript_call_result_250085)
    
    # Assigning a Subscript to a Name (line 220):
    
    # Assigning a Subscript to a Name (line 220):
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 220)
    free_set_250086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 27), 'free_set')
    # Getting the type of 'scale' (line 220)
    scale_250087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'scale')
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___250088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 21), scale_250087, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_250089 = invoke(stypy.reporting.localization.Localization(__file__, 220, 21), getitem___250088, free_set_250086)
    
    # Assigning a type to the variable 'scale_free' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'scale_free', subscript_call_result_250089)
    
    
    # Getting the type of 'tr_solver' (line 223)
    tr_solver_250090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'tr_solver')
    str_250091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 24), 'str', 'exact')
    # Applying the binary operator '==' (line 223)
    result_eq_250092 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 11), '==', tr_solver_250090, str_250091)
    
    # Testing the type of an if condition (line 223)
    if_condition_250093 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 8), result_eq_250092)
    # Assigning a type to the variable 'if_condition_250093' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'if_condition_250093', if_condition_250093)
    # SSA begins for if statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 224):
    
    # Assigning a Subscript to a Name (line 224):
    
    # Obtaining the type of the subscript
    slice_250094 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 224, 21), None, None, None)
    # Getting the type of 'free_set' (line 224)
    free_set_250095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 26), 'free_set')
    # Getting the type of 'J' (line 224)
    J_250096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 21), 'J')
    # Obtaining the member '__getitem__' of a type (line 224)
    getitem___250097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 21), J_250096, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 224)
    subscript_call_result_250098 = invoke(stypy.reporting.localization.Localization(__file__, 224, 21), getitem___250097, (slice_250094, free_set_250095))
    
    # Assigning a type to the variable 'J_free' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'J_free', subscript_call_result_250098)
    
    # Assigning a Subscript to a Name (line 225):
    
    # Assigning a Subscript to a Name (line 225):
    
    # Obtaining the type of the subscript
    int_250099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 54), 'int')
    
    # Call to lstsq(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'J_free' (line 225)
    J_free_250101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 32), 'J_free', False)
    
    # Getting the type of 'f' (line 225)
    f_250102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 41), 'f', False)
    # Applying the 'usub' unary operator (line 225)
    result___neg___250103 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 40), 'usub', f_250102)
    
    # Processing the call keyword arguments (line 225)
    int_250104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 50), 'int')
    keyword_250105 = int_250104
    kwargs_250106 = {'rcond': keyword_250105}
    # Getting the type of 'lstsq' (line 225)
    lstsq_250100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 26), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 225)
    lstsq_call_result_250107 = invoke(stypy.reporting.localization.Localization(__file__, 225, 26), lstsq_250100, *[J_free_250101, result___neg___250103], **kwargs_250106)
    
    # Obtaining the member '__getitem__' of a type (line 225)
    getitem___250108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 26), lstsq_call_result_250107, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 225)
    subscript_call_result_250109 = invoke(stypy.reporting.localization.Localization(__file__, 225, 26), getitem___250108, int_250099)
    
    # Assigning a type to the variable 'newton_step' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'newton_step', subscript_call_result_250109)
    
    # Assigning a Call to a Tuple (line 228):
    
    # Assigning a Subscript to a Name (line 228):
    
    # Obtaining the type of the subscript
    int_250110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 12), 'int')
    
    # Call to build_quadratic_1d(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'J_free' (line 228)
    J_free_250112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 38), 'J_free', False)
    # Getting the type of 'g_free' (line 228)
    g_free_250113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 46), 'g_free', False)
    
    # Getting the type of 'g_free' (line 228)
    g_free_250114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 55), 'g_free', False)
    # Applying the 'usub' unary operator (line 228)
    result___neg___250115 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 54), 'usub', g_free_250114)
    
    # Processing the call keyword arguments (line 228)
    kwargs_250116 = {}
    # Getting the type of 'build_quadratic_1d' (line 228)
    build_quadratic_1d_250111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 228)
    build_quadratic_1d_call_result_250117 = invoke(stypy.reporting.localization.Localization(__file__, 228, 19), build_quadratic_1d_250111, *[J_free_250112, g_free_250113, result___neg___250115], **kwargs_250116)
    
    # Obtaining the member '__getitem__' of a type (line 228)
    getitem___250118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), build_quadratic_1d_call_result_250117, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 228)
    subscript_call_result_250119 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), getitem___250118, int_250110)
    
    # Assigning a type to the variable 'tuple_var_assignment_249540' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'tuple_var_assignment_249540', subscript_call_result_250119)
    
    # Assigning a Subscript to a Name (line 228):
    
    # Obtaining the type of the subscript
    int_250120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 12), 'int')
    
    # Call to build_quadratic_1d(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'J_free' (line 228)
    J_free_250122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 38), 'J_free', False)
    # Getting the type of 'g_free' (line 228)
    g_free_250123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 46), 'g_free', False)
    
    # Getting the type of 'g_free' (line 228)
    g_free_250124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 55), 'g_free', False)
    # Applying the 'usub' unary operator (line 228)
    result___neg___250125 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 54), 'usub', g_free_250124)
    
    # Processing the call keyword arguments (line 228)
    kwargs_250126 = {}
    # Getting the type of 'build_quadratic_1d' (line 228)
    build_quadratic_1d_250121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 228)
    build_quadratic_1d_call_result_250127 = invoke(stypy.reporting.localization.Localization(__file__, 228, 19), build_quadratic_1d_250121, *[J_free_250122, g_free_250123, result___neg___250125], **kwargs_250126)
    
    # Obtaining the member '__getitem__' of a type (line 228)
    getitem___250128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), build_quadratic_1d_call_result_250127, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 228)
    subscript_call_result_250129 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), getitem___250128, int_250120)
    
    # Assigning a type to the variable 'tuple_var_assignment_249541' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'tuple_var_assignment_249541', subscript_call_result_250129)
    
    # Assigning a Name to a Name (line 228):
    # Getting the type of 'tuple_var_assignment_249540' (line 228)
    tuple_var_assignment_249540_250130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'tuple_var_assignment_249540')
    # Assigning a type to the variable 'a' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'a', tuple_var_assignment_249540_250130)
    
    # Assigning a Name to a Name (line 228):
    # Getting the type of 'tuple_var_assignment_249541' (line 228)
    tuple_var_assignment_249541_250131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'tuple_var_assignment_249541')
    # Assigning a type to the variable 'b' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'b', tuple_var_assignment_249541_250131)
    # SSA branch for the else part of an if statement (line 223)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'tr_solver' (line 229)
    tr_solver_250132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 13), 'tr_solver')
    str_250133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 26), 'str', 'lsmr')
    # Applying the binary operator '==' (line 229)
    result_eq_250134 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 13), '==', tr_solver_250132, str_250133)
    
    # Testing the type of an if condition (line 229)
    if_condition_250135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 13), result_eq_250134)
    # Assigning a type to the variable 'if_condition_250135' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 13), 'if_condition_250135', if_condition_250135)
    # SSA begins for if statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 230):
    
    # Assigning a Call to a Name (line 230):
    
    # Call to aslinearoperator(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'J' (line 230)
    J_250137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 35), 'J', False)
    # Processing the call keyword arguments (line 230)
    kwargs_250138 = {}
    # Getting the type of 'aslinearoperator' (line 230)
    aslinearoperator_250136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 18), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 230)
    aslinearoperator_call_result_250139 = invoke(stypy.reporting.localization.Localization(__file__, 230, 18), aslinearoperator_250136, *[J_250137], **kwargs_250138)
    
    # Assigning a type to the variable 'Jop' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'Jop', aslinearoperator_call_result_250139)
    
    # Assigning a Call to a Name (line 242):
    
    # Assigning a Call to a Name (line 242):
    
    # Call to lsmr_operator(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'Jop' (line 242)
    Jop_250141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 36), 'Jop', False)
    # Getting the type of 'scale' (line 242)
    scale_250142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 41), 'scale', False)
    # Getting the type of 'active_set' (line 242)
    active_set_250143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 48), 'active_set', False)
    # Processing the call keyword arguments (line 242)
    kwargs_250144 = {}
    # Getting the type of 'lsmr_operator' (line 242)
    lsmr_operator_250140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 22), 'lsmr_operator', False)
    # Calling lsmr_operator(args, kwargs) (line 242)
    lsmr_operator_call_result_250145 = invoke(stypy.reporting.localization.Localization(__file__, 242, 22), lsmr_operator_250140, *[Jop_250141, scale_250142, active_set_250143], **kwargs_250144)
    
    # Assigning a type to the variable 'lsmr_op' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'lsmr_op', lsmr_operator_call_result_250145)
    
    # Assigning a UnaryOp to a Name (line 243):
    
    # Assigning a UnaryOp to a Name (line 243):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'free_set' (line 243)
    free_set_250146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 61), 'free_set')
    
    # Obtaining the type of the subscript
    int_250147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 58), 'int')
    
    # Call to lsmr(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'lsmr_op' (line 243)
    lsmr_op_250149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 32), 'lsmr_op', False)
    # Getting the type of 'f' (line 243)
    f_250150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 41), 'f', False)
    # Processing the call keyword arguments (line 243)
    # Getting the type of 'tr_options' (line 243)
    tr_options_250151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 46), 'tr_options', False)
    kwargs_250152 = {'tr_options_250151': tr_options_250151}
    # Getting the type of 'lsmr' (line 243)
    lsmr_250148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 27), 'lsmr', False)
    # Calling lsmr(args, kwargs) (line 243)
    lsmr_call_result_250153 = invoke(stypy.reporting.localization.Localization(__file__, 243, 27), lsmr_250148, *[lsmr_op_250149, f_250150], **kwargs_250152)
    
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___250154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 27), lsmr_call_result_250153, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_250155 = invoke(stypy.reporting.localization.Localization(__file__, 243, 27), getitem___250154, int_250147)
    
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___250156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 27), subscript_call_result_250155, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_250157 = invoke(stypy.reporting.localization.Localization(__file__, 243, 27), getitem___250156, free_set_250146)
    
    # Applying the 'usub' unary operator (line 243)
    result___neg___250158 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 26), 'usub', subscript_call_result_250157)
    
    # Assigning a type to the variable 'newton_step' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'newton_step', result___neg___250158)
    
    # Getting the type of 'newton_step' (line 244)
    newton_step_250159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'newton_step')
    # Getting the type of 'scale_free' (line 244)
    scale_free_250160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 27), 'scale_free')
    # Applying the binary operator '*=' (line 244)
    result_imul_250161 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 12), '*=', newton_step_250159, scale_free_250160)
    # Assigning a type to the variable 'newton_step' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'newton_step', result_imul_250161)
    
    
    # Assigning a Call to a Tuple (line 248):
    
    # Assigning a Subscript to a Name (line 248):
    
    # Obtaining the type of the subscript
    int_250162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 12), 'int')
    
    # Call to build_quadratic_1d(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'Jop' (line 248)
    Jop_250164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 38), 'Jop', False)
    # Getting the type of 'g' (line 248)
    g_250165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 43), 'g', False)
    
    # Getting the type of 'g' (line 248)
    g_250166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 47), 'g', False)
    # Applying the 'usub' unary operator (line 248)
    result___neg___250167 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 46), 'usub', g_250166)
    
    # Processing the call keyword arguments (line 248)
    kwargs_250168 = {}
    # Getting the type of 'build_quadratic_1d' (line 248)
    build_quadratic_1d_250163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 19), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 248)
    build_quadratic_1d_call_result_250169 = invoke(stypy.reporting.localization.Localization(__file__, 248, 19), build_quadratic_1d_250163, *[Jop_250164, g_250165, result___neg___250167], **kwargs_250168)
    
    # Obtaining the member '__getitem__' of a type (line 248)
    getitem___250170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), build_quadratic_1d_call_result_250169, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 248)
    subscript_call_result_250171 = invoke(stypy.reporting.localization.Localization(__file__, 248, 12), getitem___250170, int_250162)
    
    # Assigning a type to the variable 'tuple_var_assignment_249542' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'tuple_var_assignment_249542', subscript_call_result_250171)
    
    # Assigning a Subscript to a Name (line 248):
    
    # Obtaining the type of the subscript
    int_250172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 12), 'int')
    
    # Call to build_quadratic_1d(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'Jop' (line 248)
    Jop_250174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 38), 'Jop', False)
    # Getting the type of 'g' (line 248)
    g_250175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 43), 'g', False)
    
    # Getting the type of 'g' (line 248)
    g_250176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 47), 'g', False)
    # Applying the 'usub' unary operator (line 248)
    result___neg___250177 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 46), 'usub', g_250176)
    
    # Processing the call keyword arguments (line 248)
    kwargs_250178 = {}
    # Getting the type of 'build_quadratic_1d' (line 248)
    build_quadratic_1d_250173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 19), 'build_quadratic_1d', False)
    # Calling build_quadratic_1d(args, kwargs) (line 248)
    build_quadratic_1d_call_result_250179 = invoke(stypy.reporting.localization.Localization(__file__, 248, 19), build_quadratic_1d_250173, *[Jop_250174, g_250175, result___neg___250177], **kwargs_250178)
    
    # Obtaining the member '__getitem__' of a type (line 248)
    getitem___250180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), build_quadratic_1d_call_result_250179, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 248)
    subscript_call_result_250181 = invoke(stypy.reporting.localization.Localization(__file__, 248, 12), getitem___250180, int_250172)
    
    # Assigning a type to the variable 'tuple_var_assignment_249543' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'tuple_var_assignment_249543', subscript_call_result_250181)
    
    # Assigning a Name to a Name (line 248):
    # Getting the type of 'tuple_var_assignment_249542' (line 248)
    tuple_var_assignment_249542_250182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'tuple_var_assignment_249542')
    # Assigning a type to the variable 'a' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'a', tuple_var_assignment_249542_250182)
    
    # Assigning a Name to a Name (line 248):
    # Getting the type of 'tuple_var_assignment_249543' (line 248)
    tuple_var_assignment_249543_250183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'tuple_var_assignment_249543')
    # Assigning a type to the variable 'b' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 15), 'b', tuple_var_assignment_249543_250183)
    # SSA join for if statement (line 229)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 250):
    
    # Assigning a Num to a Name (line 250):
    float_250184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 27), 'float')
    # Assigning a type to the variable 'actual_reduction' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'actual_reduction', float_250184)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'actual_reduction' (line 251)
    actual_reduction_250185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 14), 'actual_reduction')
    int_250186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 34), 'int')
    # Applying the binary operator '<=' (line 251)
    result_le_250187 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 14), '<=', actual_reduction_250185, int_250186)
    
    
    # Getting the type of 'nfev' (line 251)
    nfev_250188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 40), 'nfev')
    # Getting the type of 'max_nfev' (line 251)
    max_nfev_250189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 47), 'max_nfev')
    # Applying the binary operator '<' (line 251)
    result_lt_250190 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 40), '<', nfev_250188, max_nfev_250189)
    
    # Applying the binary operator 'and' (line 251)
    result_and_keyword_250191 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 14), 'and', result_le_250187, result_lt_250190)
    
    # Testing the type of an if condition (line 251)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 8), result_and_keyword_250191)
    # SSA begins for while statement (line 251)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 252):
    
    # Assigning a BinOp to a Name (line 252):
    # Getting the type of 'Delta' (line 252)
    Delta_250192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 24), 'Delta')
    # Getting the type of 'scale_free' (line 252)
    scale_free_250193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 32), 'scale_free')
    # Applying the binary operator '*' (line 252)
    result_mul_250194 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 24), '*', Delta_250192, scale_free_250193)
    
    # Assigning a type to the variable 'tr_bounds' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'tr_bounds', result_mul_250194)
    
    # Assigning a Call to a Tuple (line 254):
    
    # Assigning a Subscript to a Name (line 254):
    
    # Obtaining the type of the subscript
    int_250195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 12), 'int')
    
    # Call to dogleg_step(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'x_free' (line 255)
    x_free_250197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'x_free', False)
    # Getting the type of 'newton_step' (line 255)
    newton_step_250198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'newton_step', False)
    # Getting the type of 'g_free' (line 255)
    g_free_250199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 37), 'g_free', False)
    # Getting the type of 'a' (line 255)
    a_250200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 45), 'a', False)
    # Getting the type of 'b' (line 255)
    b_250201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 48), 'b', False)
    # Getting the type of 'tr_bounds' (line 255)
    tr_bounds_250202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 51), 'tr_bounds', False)
    # Getting the type of 'lb_free' (line 255)
    lb_free_250203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 62), 'lb_free', False)
    # Getting the type of 'ub_free' (line 255)
    ub_free_250204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 71), 'ub_free', False)
    # Processing the call keyword arguments (line 254)
    kwargs_250205 = {}
    # Getting the type of 'dogleg_step' (line 254)
    dogleg_step_250196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 47), 'dogleg_step', False)
    # Calling dogleg_step(args, kwargs) (line 254)
    dogleg_step_call_result_250206 = invoke(stypy.reporting.localization.Localization(__file__, 254, 47), dogleg_step_250196, *[x_free_250197, newton_step_250198, g_free_250199, a_250200, b_250201, tr_bounds_250202, lb_free_250203, ub_free_250204], **kwargs_250205)
    
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___250207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 12), dogleg_step_call_result_250206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_250208 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), getitem___250207, int_250195)
    
    # Assigning a type to the variable 'tuple_var_assignment_249544' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'tuple_var_assignment_249544', subscript_call_result_250208)
    
    # Assigning a Subscript to a Name (line 254):
    
    # Obtaining the type of the subscript
    int_250209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 12), 'int')
    
    # Call to dogleg_step(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'x_free' (line 255)
    x_free_250211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'x_free', False)
    # Getting the type of 'newton_step' (line 255)
    newton_step_250212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'newton_step', False)
    # Getting the type of 'g_free' (line 255)
    g_free_250213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 37), 'g_free', False)
    # Getting the type of 'a' (line 255)
    a_250214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 45), 'a', False)
    # Getting the type of 'b' (line 255)
    b_250215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 48), 'b', False)
    # Getting the type of 'tr_bounds' (line 255)
    tr_bounds_250216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 51), 'tr_bounds', False)
    # Getting the type of 'lb_free' (line 255)
    lb_free_250217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 62), 'lb_free', False)
    # Getting the type of 'ub_free' (line 255)
    ub_free_250218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 71), 'ub_free', False)
    # Processing the call keyword arguments (line 254)
    kwargs_250219 = {}
    # Getting the type of 'dogleg_step' (line 254)
    dogleg_step_250210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 47), 'dogleg_step', False)
    # Calling dogleg_step(args, kwargs) (line 254)
    dogleg_step_call_result_250220 = invoke(stypy.reporting.localization.Localization(__file__, 254, 47), dogleg_step_250210, *[x_free_250211, newton_step_250212, g_free_250213, a_250214, b_250215, tr_bounds_250216, lb_free_250217, ub_free_250218], **kwargs_250219)
    
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___250221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 12), dogleg_step_call_result_250220, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_250222 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), getitem___250221, int_250209)
    
    # Assigning a type to the variable 'tuple_var_assignment_249545' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'tuple_var_assignment_249545', subscript_call_result_250222)
    
    # Assigning a Subscript to a Name (line 254):
    
    # Obtaining the type of the subscript
    int_250223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 12), 'int')
    
    # Call to dogleg_step(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'x_free' (line 255)
    x_free_250225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'x_free', False)
    # Getting the type of 'newton_step' (line 255)
    newton_step_250226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'newton_step', False)
    # Getting the type of 'g_free' (line 255)
    g_free_250227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 37), 'g_free', False)
    # Getting the type of 'a' (line 255)
    a_250228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 45), 'a', False)
    # Getting the type of 'b' (line 255)
    b_250229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 48), 'b', False)
    # Getting the type of 'tr_bounds' (line 255)
    tr_bounds_250230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 51), 'tr_bounds', False)
    # Getting the type of 'lb_free' (line 255)
    lb_free_250231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 62), 'lb_free', False)
    # Getting the type of 'ub_free' (line 255)
    ub_free_250232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 71), 'ub_free', False)
    # Processing the call keyword arguments (line 254)
    kwargs_250233 = {}
    # Getting the type of 'dogleg_step' (line 254)
    dogleg_step_250224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 47), 'dogleg_step', False)
    # Calling dogleg_step(args, kwargs) (line 254)
    dogleg_step_call_result_250234 = invoke(stypy.reporting.localization.Localization(__file__, 254, 47), dogleg_step_250224, *[x_free_250225, newton_step_250226, g_free_250227, a_250228, b_250229, tr_bounds_250230, lb_free_250231, ub_free_250232], **kwargs_250233)
    
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___250235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 12), dogleg_step_call_result_250234, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_250236 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), getitem___250235, int_250223)
    
    # Assigning a type to the variable 'tuple_var_assignment_249546' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'tuple_var_assignment_249546', subscript_call_result_250236)
    
    # Assigning a Name to a Name (line 254):
    # Getting the type of 'tuple_var_assignment_249544' (line 254)
    tuple_var_assignment_249544_250237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'tuple_var_assignment_249544')
    # Assigning a type to the variable 'step_free' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'step_free', tuple_var_assignment_249544_250237)
    
    # Assigning a Name to a Name (line 254):
    # Getting the type of 'tuple_var_assignment_249545' (line 254)
    tuple_var_assignment_249545_250238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'tuple_var_assignment_249545')
    # Assigning a type to the variable 'on_bound_free' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 23), 'on_bound_free', tuple_var_assignment_249545_250238)
    
    # Assigning a Name to a Name (line 254):
    # Getting the type of 'tuple_var_assignment_249546' (line 254)
    tuple_var_assignment_249546_250239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'tuple_var_assignment_249546')
    # Assigning a type to the variable 'tr_hit' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 38), 'tr_hit', tuple_var_assignment_249546_250239)
    
    # Call to fill(...): (line 257)
    # Processing the call arguments (line 257)
    float_250242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 22), 'float')
    # Processing the call keyword arguments (line 257)
    kwargs_250243 = {}
    # Getting the type of 'step' (line 257)
    step_250240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'step', False)
    # Obtaining the member 'fill' of a type (line 257)
    fill_250241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), step_250240, 'fill')
    # Calling fill(args, kwargs) (line 257)
    fill_call_result_250244 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), fill_250241, *[float_250242], **kwargs_250243)
    
    
    # Assigning a Name to a Subscript (line 258):
    
    # Assigning a Name to a Subscript (line 258):
    # Getting the type of 'step_free' (line 258)
    step_free_250245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 29), 'step_free')
    # Getting the type of 'step' (line 258)
    step_250246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'step')
    # Getting the type of 'free_set' (line 258)
    free_set_250247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 17), 'free_set')
    # Storing an element on a container (line 258)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 12), step_250246, (free_set_250247, step_free_250245))
    
    
    # Getting the type of 'tr_solver' (line 260)
    tr_solver_250248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'tr_solver')
    str_250249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 28), 'str', 'exact')
    # Applying the binary operator '==' (line 260)
    result_eq_250250 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 15), '==', tr_solver_250248, str_250249)
    
    # Testing the type of an if condition (line 260)
    if_condition_250251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 12), result_eq_250250)
    # Assigning a type to the variable 'if_condition_250251' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'if_condition_250251', if_condition_250251)
    # SSA begins for if statement (line 260)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a UnaryOp to a Name (line 261):
    
    # Assigning a UnaryOp to a Name (line 261):
    
    
    # Call to evaluate_quadratic(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'J_free' (line 261)
    J_free_250253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 58), 'J_free', False)
    # Getting the type of 'g_free' (line 261)
    g_free_250254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 66), 'g_free', False)
    # Getting the type of 'step_free' (line 262)
    step_free_250255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 58), 'step_free', False)
    # Processing the call keyword arguments (line 261)
    kwargs_250256 = {}
    # Getting the type of 'evaluate_quadratic' (line 261)
    evaluate_quadratic_250252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 39), 'evaluate_quadratic', False)
    # Calling evaluate_quadratic(args, kwargs) (line 261)
    evaluate_quadratic_call_result_250257 = invoke(stypy.reporting.localization.Localization(__file__, 261, 39), evaluate_quadratic_250252, *[J_free_250253, g_free_250254, step_free_250255], **kwargs_250256)
    
    # Applying the 'usub' unary operator (line 261)
    result___neg___250258 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 38), 'usub', evaluate_quadratic_call_result_250257)
    
    # Assigning a type to the variable 'predicted_reduction' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 16), 'predicted_reduction', result___neg___250258)
    # SSA branch for the else part of an if statement (line 260)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'tr_solver' (line 263)
    tr_solver_250259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 17), 'tr_solver')
    str_250260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 30), 'str', 'lsmr')
    # Applying the binary operator '==' (line 263)
    result_eq_250261 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 17), '==', tr_solver_250259, str_250260)
    
    # Testing the type of an if condition (line 263)
    if_condition_250262 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 17), result_eq_250261)
    # Assigning a type to the variable 'if_condition_250262' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 17), 'if_condition_250262', if_condition_250262)
    # SSA begins for if statement (line 263)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a UnaryOp to a Name (line 264):
    
    # Assigning a UnaryOp to a Name (line 264):
    
    
    # Call to evaluate_quadratic(...): (line 264)
    # Processing the call arguments (line 264)
    # Getting the type of 'Jop' (line 264)
    Jop_250264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 58), 'Jop', False)
    # Getting the type of 'g' (line 264)
    g_250265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 63), 'g', False)
    # Getting the type of 'step' (line 264)
    step_250266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 66), 'step', False)
    # Processing the call keyword arguments (line 264)
    kwargs_250267 = {}
    # Getting the type of 'evaluate_quadratic' (line 264)
    evaluate_quadratic_250263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 39), 'evaluate_quadratic', False)
    # Calling evaluate_quadratic(args, kwargs) (line 264)
    evaluate_quadratic_call_result_250268 = invoke(stypy.reporting.localization.Localization(__file__, 264, 39), evaluate_quadratic_250263, *[Jop_250264, g_250265, step_250266], **kwargs_250267)
    
    # Applying the 'usub' unary operator (line 264)
    result___neg___250269 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 38), 'usub', evaluate_quadratic_call_result_250268)
    
    # Assigning a type to the variable 'predicted_reduction' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'predicted_reduction', result___neg___250269)
    # SSA join for if statement (line 263)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 260)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 266):
    
    # Assigning a BinOp to a Name (line 266):
    # Getting the type of 'x' (line 266)
    x_250270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 20), 'x')
    # Getting the type of 'step' (line 266)
    step_250271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 24), 'step')
    # Applying the binary operator '+' (line 266)
    result_add_250272 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 20), '+', x_250270, step_250271)
    
    # Assigning a type to the variable 'x_new' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'x_new', result_add_250272)
    
    # Assigning a Call to a Name (line 267):
    
    # Assigning a Call to a Name (line 267):
    
    # Call to fun(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'x_new' (line 267)
    x_new_250274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 24), 'x_new', False)
    # Processing the call keyword arguments (line 267)
    kwargs_250275 = {}
    # Getting the type of 'fun' (line 267)
    fun_250273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 20), 'fun', False)
    # Calling fun(args, kwargs) (line 267)
    fun_call_result_250276 = invoke(stypy.reporting.localization.Localization(__file__, 267, 20), fun_250273, *[x_new_250274], **kwargs_250275)
    
    # Assigning a type to the variable 'f_new' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'f_new', fun_call_result_250276)
    
    # Getting the type of 'nfev' (line 268)
    nfev_250277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'nfev')
    int_250278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 20), 'int')
    # Applying the binary operator '+=' (line 268)
    result_iadd_250279 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 12), '+=', nfev_250277, int_250278)
    # Assigning a type to the variable 'nfev' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'nfev', result_iadd_250279)
    
    
    # Assigning a Call to a Name (line 270):
    
    # Assigning a Call to a Name (line 270):
    
    # Call to norm(...): (line 270)
    # Processing the call arguments (line 270)
    # Getting the type of 'step' (line 270)
    step_250281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 31), 'step', False)
    # Getting the type of 'scale_inv' (line 270)
    scale_inv_250282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 38), 'scale_inv', False)
    # Applying the binary operator '*' (line 270)
    result_mul_250283 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 31), '*', step_250281, scale_inv_250282)
    
    # Processing the call keyword arguments (line 270)
    # Getting the type of 'np' (line 270)
    np_250284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 53), 'np', False)
    # Obtaining the member 'inf' of a type (line 270)
    inf_250285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 53), np_250284, 'inf')
    keyword_250286 = inf_250285
    kwargs_250287 = {'ord': keyword_250286}
    # Getting the type of 'norm' (line 270)
    norm_250280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 26), 'norm', False)
    # Calling norm(args, kwargs) (line 270)
    norm_call_result_250288 = invoke(stypy.reporting.localization.Localization(__file__, 270, 26), norm_250280, *[result_mul_250283], **kwargs_250287)
    
    # Assigning a type to the variable 'step_h_norm' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'step_h_norm', norm_call_result_250288)
    
    
    
    # Call to all(...): (line 272)
    # Processing the call arguments (line 272)
    
    # Call to isfinite(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'f_new' (line 272)
    f_new_250293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 38), 'f_new', False)
    # Processing the call keyword arguments (line 272)
    kwargs_250294 = {}
    # Getting the type of 'np' (line 272)
    np_250291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 26), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 272)
    isfinite_250292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 26), np_250291, 'isfinite')
    # Calling isfinite(args, kwargs) (line 272)
    isfinite_call_result_250295 = invoke(stypy.reporting.localization.Localization(__file__, 272, 26), isfinite_250292, *[f_new_250293], **kwargs_250294)
    
    # Processing the call keyword arguments (line 272)
    kwargs_250296 = {}
    # Getting the type of 'np' (line 272)
    np_250289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 19), 'np', False)
    # Obtaining the member 'all' of a type (line 272)
    all_250290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 19), np_250289, 'all')
    # Calling all(args, kwargs) (line 272)
    all_call_result_250297 = invoke(stypy.reporting.localization.Localization(__file__, 272, 19), all_250290, *[isfinite_call_result_250295], **kwargs_250296)
    
    # Applying the 'not' unary operator (line 272)
    result_not__250298 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 15), 'not', all_call_result_250297)
    
    # Testing the type of an if condition (line 272)
    if_condition_250299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 12), result_not__250298)
    # Assigning a type to the variable 'if_condition_250299' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'if_condition_250299', if_condition_250299)
    # SSA begins for if statement (line 272)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 273):
    
    # Assigning a BinOp to a Name (line 273):
    float_250300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 24), 'float')
    # Getting the type of 'step_h_norm' (line 273)
    step_h_norm_250301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 31), 'step_h_norm')
    # Applying the binary operator '*' (line 273)
    result_mul_250302 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 24), '*', float_250300, step_h_norm_250301)
    
    # Assigning a type to the variable 'Delta' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'Delta', result_mul_250302)
    # SSA join for if statement (line 272)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 277)
    # Getting the type of 'loss_function' (line 277)
    loss_function_250303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'loss_function')
    # Getting the type of 'None' (line 277)
    None_250304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 36), 'None')
    
    (may_be_250305, more_types_in_union_250306) = may_not_be_none(loss_function_250303, None_250304)

    if may_be_250305:

        if more_types_in_union_250306:
            # Runtime conditional SSA (line 277)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to loss_function(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'f_new' (line 278)
        f_new_250308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 41), 'f_new', False)
        # Processing the call keyword arguments (line 278)
        # Getting the type of 'True' (line 278)
        True_250309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 58), 'True', False)
        keyword_250310 = True_250309
        kwargs_250311 = {'cost_only': keyword_250310}
        # Getting the type of 'loss_function' (line 278)
        loss_function_250307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 27), 'loss_function', False)
        # Calling loss_function(args, kwargs) (line 278)
        loss_function_call_result_250312 = invoke(stypy.reporting.localization.Localization(__file__, 278, 27), loss_function_250307, *[f_new_250308], **kwargs_250311)
        
        # Assigning a type to the variable 'cost_new' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'cost_new', loss_function_call_result_250312)

        if more_types_in_union_250306:
            # Runtime conditional SSA for else branch (line 277)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_250305) or more_types_in_union_250306):
        
        # Assigning a BinOp to a Name (line 280):
        
        # Assigning a BinOp to a Name (line 280):
        float_250313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 27), 'float')
        
        # Call to dot(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'f_new' (line 280)
        f_new_250316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 40), 'f_new', False)
        # Getting the type of 'f_new' (line 280)
        f_new_250317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 47), 'f_new', False)
        # Processing the call keyword arguments (line 280)
        kwargs_250318 = {}
        # Getting the type of 'np' (line 280)
        np_250314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 33), 'np', False)
        # Obtaining the member 'dot' of a type (line 280)
        dot_250315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 33), np_250314, 'dot')
        # Calling dot(args, kwargs) (line 280)
        dot_call_result_250319 = invoke(stypy.reporting.localization.Localization(__file__, 280, 33), dot_250315, *[f_new_250316, f_new_250317], **kwargs_250318)
        
        # Applying the binary operator '*' (line 280)
        result_mul_250320 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 27), '*', float_250313, dot_call_result_250319)
        
        # Assigning a type to the variable 'cost_new' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'cost_new', result_mul_250320)

        if (may_be_250305 and more_types_in_union_250306):
            # SSA join for if statement (line 277)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 281):
    
    # Assigning a BinOp to a Name (line 281):
    # Getting the type of 'cost' (line 281)
    cost_250321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 31), 'cost')
    # Getting the type of 'cost_new' (line 281)
    cost_new_250322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 38), 'cost_new')
    # Applying the binary operator '-' (line 281)
    result_sub_250323 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 31), '-', cost_250321, cost_new_250322)
    
    # Assigning a type to the variable 'actual_reduction' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'actual_reduction', result_sub_250323)
    
    # Assigning a Call to a Tuple (line 283):
    
    # Assigning a Subscript to a Name (line 283):
    
    # Obtaining the type of the subscript
    int_250324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 12), 'int')
    
    # Call to update_tr_radius(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'Delta' (line 284)
    Delta_250326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'Delta', False)
    # Getting the type of 'actual_reduction' (line 284)
    actual_reduction_250327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 23), 'actual_reduction', False)
    # Getting the type of 'predicted_reduction' (line 284)
    predicted_reduction_250328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 41), 'predicted_reduction', False)
    # Getting the type of 'step_h_norm' (line 285)
    step_h_norm_250329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 16), 'step_h_norm', False)
    # Getting the type of 'tr_hit' (line 285)
    tr_hit_250330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 29), 'tr_hit', False)
    # Processing the call keyword arguments (line 283)
    kwargs_250331 = {}
    # Getting the type of 'update_tr_radius' (line 283)
    update_tr_radius_250325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'update_tr_radius', False)
    # Calling update_tr_radius(args, kwargs) (line 283)
    update_tr_radius_call_result_250332 = invoke(stypy.reporting.localization.Localization(__file__, 283, 27), update_tr_radius_250325, *[Delta_250326, actual_reduction_250327, predicted_reduction_250328, step_h_norm_250329, tr_hit_250330], **kwargs_250331)
    
    # Obtaining the member '__getitem__' of a type (line 283)
    getitem___250333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 12), update_tr_radius_call_result_250332, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 283)
    subscript_call_result_250334 = invoke(stypy.reporting.localization.Localization(__file__, 283, 12), getitem___250333, int_250324)
    
    # Assigning a type to the variable 'tuple_var_assignment_249547' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'tuple_var_assignment_249547', subscript_call_result_250334)
    
    # Assigning a Subscript to a Name (line 283):
    
    # Obtaining the type of the subscript
    int_250335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 12), 'int')
    
    # Call to update_tr_radius(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'Delta' (line 284)
    Delta_250337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'Delta', False)
    # Getting the type of 'actual_reduction' (line 284)
    actual_reduction_250338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 23), 'actual_reduction', False)
    # Getting the type of 'predicted_reduction' (line 284)
    predicted_reduction_250339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 41), 'predicted_reduction', False)
    # Getting the type of 'step_h_norm' (line 285)
    step_h_norm_250340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 16), 'step_h_norm', False)
    # Getting the type of 'tr_hit' (line 285)
    tr_hit_250341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 29), 'tr_hit', False)
    # Processing the call keyword arguments (line 283)
    kwargs_250342 = {}
    # Getting the type of 'update_tr_radius' (line 283)
    update_tr_radius_250336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'update_tr_radius', False)
    # Calling update_tr_radius(args, kwargs) (line 283)
    update_tr_radius_call_result_250343 = invoke(stypy.reporting.localization.Localization(__file__, 283, 27), update_tr_radius_250336, *[Delta_250337, actual_reduction_250338, predicted_reduction_250339, step_h_norm_250340, tr_hit_250341], **kwargs_250342)
    
    # Obtaining the member '__getitem__' of a type (line 283)
    getitem___250344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 12), update_tr_radius_call_result_250343, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 283)
    subscript_call_result_250345 = invoke(stypy.reporting.localization.Localization(__file__, 283, 12), getitem___250344, int_250335)
    
    # Assigning a type to the variable 'tuple_var_assignment_249548' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'tuple_var_assignment_249548', subscript_call_result_250345)
    
    # Assigning a Name to a Name (line 283):
    # Getting the type of 'tuple_var_assignment_249547' (line 283)
    tuple_var_assignment_249547_250346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'tuple_var_assignment_249547')
    # Assigning a type to the variable 'Delta' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'Delta', tuple_var_assignment_249547_250346)
    
    # Assigning a Name to a Name (line 283):
    # Getting the type of 'tuple_var_assignment_249548' (line 283)
    tuple_var_assignment_249548_250347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'tuple_var_assignment_249548')
    # Assigning a type to the variable 'ratio' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 'ratio', tuple_var_assignment_249548_250347)
    
    # Assigning a Call to a Name (line 288):
    
    # Assigning a Call to a Name (line 288):
    
    # Call to norm(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'step' (line 288)
    step_250349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 29), 'step', False)
    # Processing the call keyword arguments (line 288)
    kwargs_250350 = {}
    # Getting the type of 'norm' (line 288)
    norm_250348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'norm', False)
    # Calling norm(args, kwargs) (line 288)
    norm_call_result_250351 = invoke(stypy.reporting.localization.Localization(__file__, 288, 24), norm_250348, *[step_250349], **kwargs_250350)
    
    # Assigning a type to the variable 'step_norm' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'step_norm', norm_call_result_250351)
    
    # Assigning a Call to a Name (line 289):
    
    # Assigning a Call to a Name (line 289):
    
    # Call to check_termination(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'actual_reduction' (line 290)
    actual_reduction_250353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'actual_reduction', False)
    # Getting the type of 'cost' (line 290)
    cost_250354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 34), 'cost', False)
    # Getting the type of 'step_norm' (line 290)
    step_norm_250355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 40), 'step_norm', False)
    
    # Call to norm(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'x' (line 290)
    x_250357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 56), 'x', False)
    # Processing the call keyword arguments (line 290)
    kwargs_250358 = {}
    # Getting the type of 'norm' (line 290)
    norm_250356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 51), 'norm', False)
    # Calling norm(args, kwargs) (line 290)
    norm_call_result_250359 = invoke(stypy.reporting.localization.Localization(__file__, 290, 51), norm_250356, *[x_250357], **kwargs_250358)
    
    # Getting the type of 'ratio' (line 290)
    ratio_250360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 60), 'ratio', False)
    # Getting the type of 'ftol' (line 290)
    ftol_250361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 67), 'ftol', False)
    # Getting the type of 'xtol' (line 290)
    xtol_250362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 73), 'xtol', False)
    # Processing the call keyword arguments (line 289)
    kwargs_250363 = {}
    # Getting the type of 'check_termination' (line 289)
    check_termination_250352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 33), 'check_termination', False)
    # Calling check_termination(args, kwargs) (line 289)
    check_termination_call_result_250364 = invoke(stypy.reporting.localization.Localization(__file__, 289, 33), check_termination_250352, *[actual_reduction_250353, cost_250354, step_norm_250355, norm_call_result_250359, ratio_250360, ftol_250361, xtol_250362], **kwargs_250363)
    
    # Assigning a type to the variable 'termination_status' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'termination_status', check_termination_call_result_250364)
    
    # Type idiom detected: calculating its left and rigth part (line 292)
    # Getting the type of 'termination_status' (line 292)
    termination_status_250365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'termination_status')
    # Getting the type of 'None' (line 292)
    None_250366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 41), 'None')
    
    (may_be_250367, more_types_in_union_250368) = may_not_be_none(termination_status_250365, None_250366)

    if may_be_250367:

        if more_types_in_union_250368:
            # Runtime conditional SSA (line 292)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_250368:
            # SSA join for if statement (line 292)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for while statement (line 251)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'actual_reduction' (line 295)
    actual_reduction_250369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 11), 'actual_reduction')
    int_250370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 30), 'int')
    # Applying the binary operator '>' (line 295)
    result_gt_250371 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 11), '>', actual_reduction_250369, int_250370)
    
    # Testing the type of an if condition (line 295)
    if_condition_250372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 8), result_gt_250371)
    # Assigning a type to the variable 'if_condition_250372' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'if_condition_250372', if_condition_250372)
    # SSA begins for if statement (line 295)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 296):
    
    # Assigning a Name to a Subscript (line 296):
    # Getting the type of 'on_bound_free' (line 296)
    on_bound_free_250373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 33), 'on_bound_free')
    # Getting the type of 'on_bound' (line 296)
    on_bound_250374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'on_bound')
    # Getting the type of 'free_set' (line 296)
    free_set_250375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 21), 'free_set')
    # Storing an element on a container (line 296)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 12), on_bound_250374, (free_set_250375, on_bound_free_250373))
    
    # Assigning a Name to a Name (line 298):
    
    # Assigning a Name to a Name (line 298):
    # Getting the type of 'x_new' (line 298)
    x_new_250376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'x_new')
    # Assigning a type to the variable 'x' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'x', x_new_250376)
    
    # Assigning a Compare to a Name (line 300):
    
    # Assigning a Compare to a Name (line 300):
    
    # Getting the type of 'on_bound' (line 300)
    on_bound_250377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 19), 'on_bound')
    int_250378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 31), 'int')
    # Applying the binary operator '==' (line 300)
    result_eq_250379 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 19), '==', on_bound_250377, int_250378)
    
    # Assigning a type to the variable 'mask' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'mask', result_eq_250379)
    
    # Assigning a Subscript to a Subscript (line 301):
    
    # Assigning a Subscript to a Subscript (line 301):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 301)
    mask_250380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'mask')
    # Getting the type of 'lb' (line 301)
    lb_250381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 22), 'lb')
    # Obtaining the member '__getitem__' of a type (line 301)
    getitem___250382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 22), lb_250381, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 301)
    subscript_call_result_250383 = invoke(stypy.reporting.localization.Localization(__file__, 301, 22), getitem___250382, mask_250380)
    
    # Getting the type of 'x' (line 301)
    x_250384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'x')
    # Getting the type of 'mask' (line 301)
    mask_250385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 14), 'mask')
    # Storing an element on a container (line 301)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), x_250384, (mask_250385, subscript_call_result_250383))
    
    # Assigning a Compare to a Name (line 302):
    
    # Assigning a Compare to a Name (line 302):
    
    # Getting the type of 'on_bound' (line 302)
    on_bound_250386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 19), 'on_bound')
    int_250387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 31), 'int')
    # Applying the binary operator '==' (line 302)
    result_eq_250388 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 19), '==', on_bound_250386, int_250387)
    
    # Assigning a type to the variable 'mask' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'mask', result_eq_250388)
    
    # Assigning a Subscript to a Subscript (line 303):
    
    # Assigning a Subscript to a Subscript (line 303):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 303)
    mask_250389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 25), 'mask')
    # Getting the type of 'ub' (line 303)
    ub_250390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 22), 'ub')
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___250391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 22), ub_250390, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_250392 = invoke(stypy.reporting.localization.Localization(__file__, 303, 22), getitem___250391, mask_250389)
    
    # Getting the type of 'x' (line 303)
    x_250393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'x')
    # Getting the type of 'mask' (line 303)
    mask_250394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 14), 'mask')
    # Storing an element on a container (line 303)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 12), x_250393, (mask_250394, subscript_call_result_250392))
    
    # Assigning a Name to a Name (line 305):
    
    # Assigning a Name to a Name (line 305):
    # Getting the type of 'f_new' (line 305)
    f_new_250395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'f_new')
    # Assigning a type to the variable 'f' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'f', f_new_250395)
    
    # Assigning a Call to a Name (line 306):
    
    # Assigning a Call to a Name (line 306):
    
    # Call to copy(...): (line 306)
    # Processing the call keyword arguments (line 306)
    kwargs_250398 = {}
    # Getting the type of 'f' (line 306)
    f_250396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 21), 'f', False)
    # Obtaining the member 'copy' of a type (line 306)
    copy_250397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 21), f_250396, 'copy')
    # Calling copy(args, kwargs) (line 306)
    copy_call_result_250399 = invoke(stypy.reporting.localization.Localization(__file__, 306, 21), copy_250397, *[], **kwargs_250398)
    
    # Assigning a type to the variable 'f_true' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'f_true', copy_call_result_250399)
    
    # Assigning a Name to a Name (line 308):
    
    # Assigning a Name to a Name (line 308):
    # Getting the type of 'cost_new' (line 308)
    cost_new_250400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 19), 'cost_new')
    # Assigning a type to the variable 'cost' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'cost', cost_new_250400)
    
    # Assigning a Call to a Name (line 310):
    
    # Assigning a Call to a Name (line 310):
    
    # Call to jac(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'x' (line 310)
    x_250402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 20), 'x', False)
    # Getting the type of 'f' (line 310)
    f_250403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 23), 'f', False)
    # Processing the call keyword arguments (line 310)
    kwargs_250404 = {}
    # Getting the type of 'jac' (line 310)
    jac_250401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'jac', False)
    # Calling jac(args, kwargs) (line 310)
    jac_call_result_250405 = invoke(stypy.reporting.localization.Localization(__file__, 310, 16), jac_250401, *[x_250402, f_250403], **kwargs_250404)
    
    # Assigning a type to the variable 'J' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'J', jac_call_result_250405)
    
    # Getting the type of 'njev' (line 311)
    njev_250406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'njev')
    int_250407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 20), 'int')
    # Applying the binary operator '+=' (line 311)
    result_iadd_250408 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 12), '+=', njev_250406, int_250407)
    # Assigning a type to the variable 'njev' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'njev', result_iadd_250408)
    
    
    # Type idiom detected: calculating its left and rigth part (line 313)
    # Getting the type of 'loss_function' (line 313)
    loss_function_250409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'loss_function')
    # Getting the type of 'None' (line 313)
    None_250410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 36), 'None')
    
    (may_be_250411, more_types_in_union_250412) = may_not_be_none(loss_function_250409, None_250410)

    if may_be_250411:

        if more_types_in_union_250412:
            # Runtime conditional SSA (line 313)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 314):
        
        # Assigning a Call to a Name (line 314):
        
        # Call to loss_function(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'f' (line 314)
        f_250414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 36), 'f', False)
        # Processing the call keyword arguments (line 314)
        kwargs_250415 = {}
        # Getting the type of 'loss_function' (line 314)
        loss_function_250413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 22), 'loss_function', False)
        # Calling loss_function(args, kwargs) (line 314)
        loss_function_call_result_250416 = invoke(stypy.reporting.localization.Localization(__file__, 314, 22), loss_function_250413, *[f_250414], **kwargs_250415)
        
        # Assigning a type to the variable 'rho' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'rho', loss_function_call_result_250416)
        
        # Assigning a Call to a Tuple (line 315):
        
        # Assigning a Subscript to a Name (line 315):
        
        # Obtaining the type of the subscript
        int_250417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 16), 'int')
        
        # Call to scale_for_robust_loss_function(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'J' (line 315)
        J_250419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 54), 'J', False)
        # Getting the type of 'f' (line 315)
        f_250420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 57), 'f', False)
        # Getting the type of 'rho' (line 315)
        rho_250421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 60), 'rho', False)
        # Processing the call keyword arguments (line 315)
        kwargs_250422 = {}
        # Getting the type of 'scale_for_robust_loss_function' (line 315)
        scale_for_robust_loss_function_250418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'scale_for_robust_loss_function', False)
        # Calling scale_for_robust_loss_function(args, kwargs) (line 315)
        scale_for_robust_loss_function_call_result_250423 = invoke(stypy.reporting.localization.Localization(__file__, 315, 23), scale_for_robust_loss_function_250418, *[J_250419, f_250420, rho_250421], **kwargs_250422)
        
        # Obtaining the member '__getitem__' of a type (line 315)
        getitem___250424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 16), scale_for_robust_loss_function_call_result_250423, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 315)
        subscript_call_result_250425 = invoke(stypy.reporting.localization.Localization(__file__, 315, 16), getitem___250424, int_250417)
        
        # Assigning a type to the variable 'tuple_var_assignment_249549' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'tuple_var_assignment_249549', subscript_call_result_250425)
        
        # Assigning a Subscript to a Name (line 315):
        
        # Obtaining the type of the subscript
        int_250426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 16), 'int')
        
        # Call to scale_for_robust_loss_function(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'J' (line 315)
        J_250428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 54), 'J', False)
        # Getting the type of 'f' (line 315)
        f_250429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 57), 'f', False)
        # Getting the type of 'rho' (line 315)
        rho_250430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 60), 'rho', False)
        # Processing the call keyword arguments (line 315)
        kwargs_250431 = {}
        # Getting the type of 'scale_for_robust_loss_function' (line 315)
        scale_for_robust_loss_function_250427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'scale_for_robust_loss_function', False)
        # Calling scale_for_robust_loss_function(args, kwargs) (line 315)
        scale_for_robust_loss_function_call_result_250432 = invoke(stypy.reporting.localization.Localization(__file__, 315, 23), scale_for_robust_loss_function_250427, *[J_250428, f_250429, rho_250430], **kwargs_250431)
        
        # Obtaining the member '__getitem__' of a type (line 315)
        getitem___250433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 16), scale_for_robust_loss_function_call_result_250432, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 315)
        subscript_call_result_250434 = invoke(stypy.reporting.localization.Localization(__file__, 315, 16), getitem___250433, int_250426)
        
        # Assigning a type to the variable 'tuple_var_assignment_249550' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'tuple_var_assignment_249550', subscript_call_result_250434)
        
        # Assigning a Name to a Name (line 315):
        # Getting the type of 'tuple_var_assignment_249549' (line 315)
        tuple_var_assignment_249549_250435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'tuple_var_assignment_249549')
        # Assigning a type to the variable 'J' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'J', tuple_var_assignment_249549_250435)
        
        # Assigning a Name to a Name (line 315):
        # Getting the type of 'tuple_var_assignment_249550' (line 315)
        tuple_var_assignment_249550_250436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'tuple_var_assignment_249550')
        # Assigning a type to the variable 'f' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 19), 'f', tuple_var_assignment_249550_250436)

        if more_types_in_union_250412:
            # SSA join for if statement (line 313)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 317):
    
    # Assigning a Call to a Name (line 317):
    
    # Call to compute_grad(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 'J' (line 317)
    J_250438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 29), 'J', False)
    # Getting the type of 'f' (line 317)
    f_250439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 32), 'f', False)
    # Processing the call keyword arguments (line 317)
    kwargs_250440 = {}
    # Getting the type of 'compute_grad' (line 317)
    compute_grad_250437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 16), 'compute_grad', False)
    # Calling compute_grad(args, kwargs) (line 317)
    compute_grad_call_result_250441 = invoke(stypy.reporting.localization.Localization(__file__, 317, 16), compute_grad_250437, *[J_250438, f_250439], **kwargs_250440)
    
    # Assigning a type to the variable 'g' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'g', compute_grad_call_result_250441)
    
    # Getting the type of 'jac_scale' (line 319)
    jac_scale_250442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 15), 'jac_scale')
    # Testing the type of an if condition (line 319)
    if_condition_250443 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 12), jac_scale_250442)
    # Assigning a type to the variable 'if_condition_250443' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'if_condition_250443', if_condition_250443)
    # SSA begins for if statement (line 319)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 320):
    
    # Assigning a Subscript to a Name (line 320):
    
    # Obtaining the type of the subscript
    int_250444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 16), 'int')
    
    # Call to compute_jac_scale(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'J' (line 320)
    J_250446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 53), 'J', False)
    # Getting the type of 'scale_inv' (line 320)
    scale_inv_250447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 56), 'scale_inv', False)
    # Processing the call keyword arguments (line 320)
    kwargs_250448 = {}
    # Getting the type of 'compute_jac_scale' (line 320)
    compute_jac_scale_250445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 35), 'compute_jac_scale', False)
    # Calling compute_jac_scale(args, kwargs) (line 320)
    compute_jac_scale_call_result_250449 = invoke(stypy.reporting.localization.Localization(__file__, 320, 35), compute_jac_scale_250445, *[J_250446, scale_inv_250447], **kwargs_250448)
    
    # Obtaining the member '__getitem__' of a type (line 320)
    getitem___250450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 16), compute_jac_scale_call_result_250449, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 320)
    subscript_call_result_250451 = invoke(stypy.reporting.localization.Localization(__file__, 320, 16), getitem___250450, int_250444)
    
    # Assigning a type to the variable 'tuple_var_assignment_249551' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'tuple_var_assignment_249551', subscript_call_result_250451)
    
    # Assigning a Subscript to a Name (line 320):
    
    # Obtaining the type of the subscript
    int_250452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 16), 'int')
    
    # Call to compute_jac_scale(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'J' (line 320)
    J_250454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 53), 'J', False)
    # Getting the type of 'scale_inv' (line 320)
    scale_inv_250455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 56), 'scale_inv', False)
    # Processing the call keyword arguments (line 320)
    kwargs_250456 = {}
    # Getting the type of 'compute_jac_scale' (line 320)
    compute_jac_scale_250453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 35), 'compute_jac_scale', False)
    # Calling compute_jac_scale(args, kwargs) (line 320)
    compute_jac_scale_call_result_250457 = invoke(stypy.reporting.localization.Localization(__file__, 320, 35), compute_jac_scale_250453, *[J_250454, scale_inv_250455], **kwargs_250456)
    
    # Obtaining the member '__getitem__' of a type (line 320)
    getitem___250458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 16), compute_jac_scale_call_result_250457, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 320)
    subscript_call_result_250459 = invoke(stypy.reporting.localization.Localization(__file__, 320, 16), getitem___250458, int_250452)
    
    # Assigning a type to the variable 'tuple_var_assignment_249552' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'tuple_var_assignment_249552', subscript_call_result_250459)
    
    # Assigning a Name to a Name (line 320):
    # Getting the type of 'tuple_var_assignment_249551' (line 320)
    tuple_var_assignment_249551_250460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'tuple_var_assignment_249551')
    # Assigning a type to the variable 'scale' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'scale', tuple_var_assignment_249551_250460)
    
    # Assigning a Name to a Name (line 320):
    # Getting the type of 'tuple_var_assignment_249552' (line 320)
    tuple_var_assignment_249552_250461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'tuple_var_assignment_249552')
    # Assigning a type to the variable 'scale_inv' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 23), 'scale_inv', tuple_var_assignment_249552_250461)
    # SSA join for if statement (line 319)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 295)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 322):
    
    # Assigning a Num to a Name (line 322):
    int_250462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 24), 'int')
    # Assigning a type to the variable 'step_norm' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'step_norm', int_250462)
    
    # Assigning a Num to a Name (line 323):
    
    # Assigning a Num to a Name (line 323):
    int_250463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 31), 'int')
    # Assigning a type to the variable 'actual_reduction' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'actual_reduction', int_250463)
    # SSA join for if statement (line 295)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'iteration' (line 325)
    iteration_250464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'iteration')
    int_250465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 21), 'int')
    # Applying the binary operator '+=' (line 325)
    result_iadd_250466 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 8), '+=', iteration_250464, int_250465)
    # Assigning a type to the variable 'iteration' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'iteration', result_iadd_250466)
    
    # SSA join for while statement (line 198)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 327)
    # Getting the type of 'termination_status' (line 327)
    termination_status_250467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 7), 'termination_status')
    # Getting the type of 'None' (line 327)
    None_250468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 29), 'None')
    
    (may_be_250469, more_types_in_union_250470) = may_be_none(termination_status_250467, None_250468)

    if may_be_250469:

        if more_types_in_union_250470:
            # Runtime conditional SSA (line 327)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 328):
        
        # Assigning a Num to a Name (line 328):
        int_250471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 29), 'int')
        # Assigning a type to the variable 'termination_status' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'termination_status', int_250471)

        if more_types_in_union_250470:
            # SSA join for if statement (line 327)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to OptimizeResult(...): (line 330)
    # Processing the call keyword arguments (line 330)
    # Getting the type of 'x' (line 331)
    x_250473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 10), 'x', False)
    keyword_250474 = x_250473
    # Getting the type of 'cost' (line 331)
    cost_250475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 18), 'cost', False)
    keyword_250476 = cost_250475
    # Getting the type of 'f_true' (line 331)
    f_true_250477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 28), 'f_true', False)
    keyword_250478 = f_true_250477
    # Getting the type of 'J' (line 331)
    J_250479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 40), 'J', False)
    keyword_250480 = J_250479
    # Getting the type of 'g_full' (line 331)
    g_full_250481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 48), 'g_full', False)
    keyword_250482 = g_full_250481
    # Getting the type of 'g_norm' (line 331)
    g_norm_250483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 67), 'g_norm', False)
    keyword_250484 = g_norm_250483
    # Getting the type of 'on_bound' (line 332)
    on_bound_250485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 20), 'on_bound', False)
    keyword_250486 = on_bound_250485
    # Getting the type of 'nfev' (line 332)
    nfev_250487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 35), 'nfev', False)
    keyword_250488 = nfev_250487
    # Getting the type of 'njev' (line 332)
    njev_250489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 46), 'njev', False)
    keyword_250490 = njev_250489
    # Getting the type of 'termination_status' (line 332)
    termination_status_250491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 59), 'termination_status', False)
    keyword_250492 = termination_status_250491
    kwargs_250493 = {'status': keyword_250492, 'njev': keyword_250490, 'nfev': keyword_250488, 'active_mask': keyword_250486, 'cost': keyword_250476, 'optimality': keyword_250484, 'fun': keyword_250478, 'x': keyword_250474, 'grad': keyword_250482, 'jac': keyword_250480}
    # Getting the type of 'OptimizeResult' (line 330)
    OptimizeResult_250472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 330)
    OptimizeResult_call_result_250494 = invoke(stypy.reporting.localization.Localization(__file__, 330, 11), OptimizeResult_250472, *[], **kwargs_250493)
    
    # Assigning a type to the variable 'stypy_return_type' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type', OptimizeResult_call_result_250494)
    
    # ################# End of 'dogbox(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dogbox' in the type store
    # Getting the type of 'stypy_return_type' (line 152)
    stypy_return_type_250495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_250495)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dogbox'
    return stypy_return_type_250495

# Assigning a type to the variable 'dogbox' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'dogbox', dogbox)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
