
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: =====================================================
3: Optimization and root finding (:mod:`scipy.optimize`)
4: =====================================================
5: 
6: .. currentmodule:: scipy.optimize
7: 
8: Optimization
9: ============
10: 
11: Local Optimization
12: ------------------
13: 
14: .. autosummary::
15:    :toctree: generated/
16: 
17:    minimize - Unified interface for minimizers of multivariate functions
18:    minimize_scalar - Unified interface for minimizers of univariate functions
19:    OptimizeResult - The optimization result returned by some optimizers
20:    OptimizeWarning - The optimization encountered problems
21: 
22: The `minimize` function supports the following methods:
23: 
24: .. toctree::
25: 
26:    optimize.minimize-neldermead
27:    optimize.minimize-powell
28:    optimize.minimize-cg
29:    optimize.minimize-bfgs
30:    optimize.minimize-newtoncg
31:    optimize.minimize-lbfgsb
32:    optimize.minimize-tnc
33:    optimize.minimize-cobyla
34:    optimize.minimize-slsqp
35:    optimize.minimize-dogleg
36:    optimize.minimize-trustncg
37:    optimize.minimize-trustkrylov
38:    optimize.minimize-trustexact
39: 
40: The `minimize_scalar` function supports the following methods:
41: 
42: .. toctree::
43: 
44:    optimize.minimize_scalar-brent
45:    optimize.minimize_scalar-bounded
46:    optimize.minimize_scalar-golden
47: 
48: The specific optimization method interfaces below in this subsection are
49: not recommended for use in new scripts; all of these methods are accessible
50: via a newer, more consistent interface provided by the functions above.
51: 
52: General-purpose multivariate methods:
53: 
54: .. autosummary::
55:    :toctree: generated/
56: 
57:    fmin - Nelder-Mead Simplex algorithm
58:    fmin_powell - Powell's (modified) level set method
59:    fmin_cg - Non-linear (Polak-Ribiere) conjugate gradient algorithm
60:    fmin_bfgs - Quasi-Newton method (Broydon-Fletcher-Goldfarb-Shanno)
61:    fmin_ncg - Line-search Newton Conjugate Gradient
62: 
63: Constrained multivariate methods:
64: 
65: .. autosummary::
66:    :toctree: generated/
67: 
68:    fmin_l_bfgs_b - Zhu, Byrd, and Nocedal's constrained optimizer
69:    fmin_tnc - Truncated Newton code
70:    fmin_cobyla - Constrained optimization by linear approximation
71:    fmin_slsqp - Minimization using sequential least-squares programming
72:    differential_evolution - stochastic minimization using differential evolution
73: 
74: Univariate (scalar) minimization methods:
75: 
76: .. autosummary::
77:    :toctree: generated/
78: 
79:    fminbound - Bounded minimization of a scalar function
80:    brent - 1-D function minimization using Brent method
81:    golden - 1-D function minimization using Golden Section method
82: 
83: Equation (Local) Minimizers
84: ---------------------------
85: 
86: .. autosummary::
87:    :toctree: generated/
88: 
89:    leastsq - Minimize the sum of squares of M equations in N unknowns
90:    least_squares - Feature-rich least-squares minimization.
91:    nnls - Linear least-squares problem with non-negativity constraint
92:    lsq_linear - Linear least-squares problem with bound constraints
93: 
94: Global Optimization
95: -------------------
96: 
97: .. autosummary::
98:    :toctree: generated/
99: 
100:    basinhopping - Basinhopping stochastic optimizer
101:    brute - Brute force searching optimizer
102:    differential_evolution - stochastic minimization using differential evolution
103: 
104: Rosenbrock function
105: -------------------
106: 
107: .. autosummary::
108:    :toctree: generated/
109: 
110:    rosen - The Rosenbrock function.
111:    rosen_der - The derivative of the Rosenbrock function.
112:    rosen_hess - The Hessian matrix of the Rosenbrock function.
113:    rosen_hess_prod - Product of the Rosenbrock Hessian with a vector.
114: 
115: Fitting
116: =======
117: 
118: .. autosummary::
119:    :toctree: generated/
120: 
121:    curve_fit -- Fit curve to a set of points
122: 
123: Root finding
124: ============
125: 
126: Scalar functions
127: ----------------
128: .. autosummary::
129:    :toctree: generated/
130: 
131:    brentq - quadratic interpolation Brent method
132:    brenth - Brent method, modified by Harris with hyperbolic extrapolation
133:    ridder - Ridder's method
134:    bisect - Bisection method
135:    newton - Secant method or Newton's method
136: 
137: Fixed point finding:
138: 
139: .. autosummary::
140:    :toctree: generated/
141: 
142:    fixed_point - Single-variable fixed-point solver
143: 
144: Multidimensional
145: ----------------
146: 
147: General nonlinear solvers:
148: 
149: .. autosummary::
150:    :toctree: generated/
151: 
152:    root - Unified interface for nonlinear solvers of multivariate functions
153:    fsolve - Non-linear multi-variable equation solver
154:    broyden1 - Broyden's first method
155:    broyden2 - Broyden's second method
156: 
157: The `root` function supports the following methods:
158: 
159: .. toctree::
160: 
161:    optimize.root-hybr
162:    optimize.root-lm
163:    optimize.root-broyden1
164:    optimize.root-broyden2
165:    optimize.root-anderson
166:    optimize.root-linearmixing
167:    optimize.root-diagbroyden
168:    optimize.root-excitingmixing
169:    optimize.root-krylov
170:    optimize.root-dfsane
171: 
172: Large-scale nonlinear solvers:
173: 
174: .. autosummary::
175:    :toctree: generated/
176: 
177:    newton_krylov
178:    anderson
179: 
180: Simple iterations:
181: 
182: .. autosummary::
183:    :toctree: generated/
184: 
185:    excitingmixing
186:    linearmixing
187:    diagbroyden
188: 
189: :mod:`Additional information on the nonlinear solvers <scipy.optimize.nonlin>`
190: 
191: Linear Programming
192: ==================
193: 
194: General linear programming solver:
195: 
196: .. autosummary::
197:    :toctree: generated/
198: 
199:    linprog -- Unified interface for minimizers of linear programming problems
200: 
201: The `linprog` function supports the following methods:
202: 
203: .. toctree::
204: 
205:    optimize.linprog-simplex
206:    optimize.linprog-interior-point
207: 
208: The simplex method supports callback functions, such as:
209:     
210: .. autosummary::
211:    :toctree: generated/
212:    
213:    linprog_verbose_callback -- Sample callback function for linprog (simplex)
214: 
215: Assignment problems:
216: 
217: .. autosummary::
218:    :toctree: generated/
219: 
220:    linear_sum_assignment -- Solves the linear-sum assignment problem
221: 
222: Utilities
223: =========
224: 
225: .. autosummary::
226:    :toctree: generated/
227: 
228:    approx_fprime - Approximate the gradient of a scalar function
229:    bracket - Bracket a minimum, given two starting points
230:    check_grad - Check the supplied derivative using finite differences
231:    line_search - Return a step that satisfies the strong Wolfe conditions
232: 
233:    show_options - Show specific options optimization solvers
234:    LbfgsInvHessProduct - Linear operator for L-BFGS approximate inverse Hessian
235: 
236: '''
237: 
238: from __future__ import division, print_function, absolute_import
239: 
240: from .optimize import *
241: from ._minimize import *
242: from ._root import *
243: from .minpack import *
244: from .zeros import *
245: from .lbfgsb import fmin_l_bfgs_b, LbfgsInvHessProduct
246: from .tnc import fmin_tnc
247: from .cobyla import fmin_cobyla
248: from .nonlin import *
249: from .slsqp import fmin_slsqp
250: from .nnls import nnls
251: from ._basinhopping import basinhopping
252: from ._linprog import linprog, linprog_verbose_callback
253: from ._hungarian import linear_sum_assignment
254: from ._differentialevolution import differential_evolution
255: from ._lsq import least_squares, lsq_linear
256: 
257: __all__ = [s for s in dir() if not s.startswith('_')]
258: 
259: from scipy._lib._testutils import PytestTester
260: test = PytestTester(__name__)
261: del PytestTester
262: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_204736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, (-1)), 'str', "\n=====================================================\nOptimization and root finding (:mod:`scipy.optimize`)\n=====================================================\n\n.. currentmodule:: scipy.optimize\n\nOptimization\n============\n\nLocal Optimization\n------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   minimize - Unified interface for minimizers of multivariate functions\n   minimize_scalar - Unified interface for minimizers of univariate functions\n   OptimizeResult - The optimization result returned by some optimizers\n   OptimizeWarning - The optimization encountered problems\n\nThe `minimize` function supports the following methods:\n\n.. toctree::\n\n   optimize.minimize-neldermead\n   optimize.minimize-powell\n   optimize.minimize-cg\n   optimize.minimize-bfgs\n   optimize.minimize-newtoncg\n   optimize.minimize-lbfgsb\n   optimize.minimize-tnc\n   optimize.minimize-cobyla\n   optimize.minimize-slsqp\n   optimize.minimize-dogleg\n   optimize.minimize-trustncg\n   optimize.minimize-trustkrylov\n   optimize.minimize-trustexact\n\nThe `minimize_scalar` function supports the following methods:\n\n.. toctree::\n\n   optimize.minimize_scalar-brent\n   optimize.minimize_scalar-bounded\n   optimize.minimize_scalar-golden\n\nThe specific optimization method interfaces below in this subsection are\nnot recommended for use in new scripts; all of these methods are accessible\nvia a newer, more consistent interface provided by the functions above.\n\nGeneral-purpose multivariate methods:\n\n.. autosummary::\n   :toctree: generated/\n\n   fmin - Nelder-Mead Simplex algorithm\n   fmin_powell - Powell's (modified) level set method\n   fmin_cg - Non-linear (Polak-Ribiere) conjugate gradient algorithm\n   fmin_bfgs - Quasi-Newton method (Broydon-Fletcher-Goldfarb-Shanno)\n   fmin_ncg - Line-search Newton Conjugate Gradient\n\nConstrained multivariate methods:\n\n.. autosummary::\n   :toctree: generated/\n\n   fmin_l_bfgs_b - Zhu, Byrd, and Nocedal's constrained optimizer\n   fmin_tnc - Truncated Newton code\n   fmin_cobyla - Constrained optimization by linear approximation\n   fmin_slsqp - Minimization using sequential least-squares programming\n   differential_evolution - stochastic minimization using differential evolution\n\nUnivariate (scalar) minimization methods:\n\n.. autosummary::\n   :toctree: generated/\n\n   fminbound - Bounded minimization of a scalar function\n   brent - 1-D function minimization using Brent method\n   golden - 1-D function minimization using Golden Section method\n\nEquation (Local) Minimizers\n---------------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   leastsq - Minimize the sum of squares of M equations in N unknowns\n   least_squares - Feature-rich least-squares minimization.\n   nnls - Linear least-squares problem with non-negativity constraint\n   lsq_linear - Linear least-squares problem with bound constraints\n\nGlobal Optimization\n-------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   basinhopping - Basinhopping stochastic optimizer\n   brute - Brute force searching optimizer\n   differential_evolution - stochastic minimization using differential evolution\n\nRosenbrock function\n-------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   rosen - The Rosenbrock function.\n   rosen_der - The derivative of the Rosenbrock function.\n   rosen_hess - The Hessian matrix of the Rosenbrock function.\n   rosen_hess_prod - Product of the Rosenbrock Hessian with a vector.\n\nFitting\n=======\n\n.. autosummary::\n   :toctree: generated/\n\n   curve_fit -- Fit curve to a set of points\n\nRoot finding\n============\n\nScalar functions\n----------------\n.. autosummary::\n   :toctree: generated/\n\n   brentq - quadratic interpolation Brent method\n   brenth - Brent method, modified by Harris with hyperbolic extrapolation\n   ridder - Ridder's method\n   bisect - Bisection method\n   newton - Secant method or Newton's method\n\nFixed point finding:\n\n.. autosummary::\n   :toctree: generated/\n\n   fixed_point - Single-variable fixed-point solver\n\nMultidimensional\n----------------\n\nGeneral nonlinear solvers:\n\n.. autosummary::\n   :toctree: generated/\n\n   root - Unified interface for nonlinear solvers of multivariate functions\n   fsolve - Non-linear multi-variable equation solver\n   broyden1 - Broyden's first method\n   broyden2 - Broyden's second method\n\nThe `root` function supports the following methods:\n\n.. toctree::\n\n   optimize.root-hybr\n   optimize.root-lm\n   optimize.root-broyden1\n   optimize.root-broyden2\n   optimize.root-anderson\n   optimize.root-linearmixing\n   optimize.root-diagbroyden\n   optimize.root-excitingmixing\n   optimize.root-krylov\n   optimize.root-dfsane\n\nLarge-scale nonlinear solvers:\n\n.. autosummary::\n   :toctree: generated/\n\n   newton_krylov\n   anderson\n\nSimple iterations:\n\n.. autosummary::\n   :toctree: generated/\n\n   excitingmixing\n   linearmixing\n   diagbroyden\n\n:mod:`Additional information on the nonlinear solvers <scipy.optimize.nonlin>`\n\nLinear Programming\n==================\n\nGeneral linear programming solver:\n\n.. autosummary::\n   :toctree: generated/\n\n   linprog -- Unified interface for minimizers of linear programming problems\n\nThe `linprog` function supports the following methods:\n\n.. toctree::\n\n   optimize.linprog-simplex\n   optimize.linprog-interior-point\n\nThe simplex method supports callback functions, such as:\n    \n.. autosummary::\n   :toctree: generated/\n   \n   linprog_verbose_callback -- Sample callback function for linprog (simplex)\n\nAssignment problems:\n\n.. autosummary::\n   :toctree: generated/\n\n   linear_sum_assignment -- Solves the linear-sum assignment problem\n\nUtilities\n=========\n\n.. autosummary::\n   :toctree: generated/\n\n   approx_fprime - Approximate the gradient of a scalar function\n   bracket - Bracket a minimum, given two starting points\n   check_grad - Check the supplied derivative using finite differences\n   line_search - Return a step that satisfies the strong Wolfe conditions\n\n   show_options - Show specific options optimization solvers\n   LbfgsInvHessProduct - Linear operator for L-BFGS approximate inverse Hessian\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 240, 0))

# 'from scipy.optimize.optimize import ' statement (line 240)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204737 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 240, 0), 'scipy.optimize.optimize')

if (type(import_204737) is not StypyTypeError):

    if (import_204737 != 'pyd_module'):
        __import__(import_204737)
        sys_modules_204738 = sys.modules[import_204737]
        import_from_module(stypy.reporting.localization.Localization(__file__, 240, 0), 'scipy.optimize.optimize', sys_modules_204738.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 240, 0), __file__, sys_modules_204738, sys_modules_204738.module_type_store, module_type_store)
    else:
        from scipy.optimize.optimize import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 240, 0), 'scipy.optimize.optimize', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.optimize.optimize' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 0), 'scipy.optimize.optimize', import_204737)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 241, 0))

# 'from scipy.optimize._minimize import ' statement (line 241)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204739 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 241, 0), 'scipy.optimize._minimize')

if (type(import_204739) is not StypyTypeError):

    if (import_204739 != 'pyd_module'):
        __import__(import_204739)
        sys_modules_204740 = sys.modules[import_204739]
        import_from_module(stypy.reporting.localization.Localization(__file__, 241, 0), 'scipy.optimize._minimize', sys_modules_204740.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 241, 0), __file__, sys_modules_204740, sys_modules_204740.module_type_store, module_type_store)
    else:
        from scipy.optimize._minimize import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 241, 0), 'scipy.optimize._minimize', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.optimize._minimize' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), 'scipy.optimize._minimize', import_204739)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 242, 0))

# 'from scipy.optimize._root import ' statement (line 242)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204741 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 242, 0), 'scipy.optimize._root')

if (type(import_204741) is not StypyTypeError):

    if (import_204741 != 'pyd_module'):
        __import__(import_204741)
        sys_modules_204742 = sys.modules[import_204741]
        import_from_module(stypy.reporting.localization.Localization(__file__, 242, 0), 'scipy.optimize._root', sys_modules_204742.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 242, 0), __file__, sys_modules_204742, sys_modules_204742.module_type_store, module_type_store)
    else:
        from scipy.optimize._root import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 242, 0), 'scipy.optimize._root', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.optimize._root' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), 'scipy.optimize._root', import_204741)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 243, 0))

# 'from scipy.optimize.minpack import ' statement (line 243)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204743 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 243, 0), 'scipy.optimize.minpack')

if (type(import_204743) is not StypyTypeError):

    if (import_204743 != 'pyd_module'):
        __import__(import_204743)
        sys_modules_204744 = sys.modules[import_204743]
        import_from_module(stypy.reporting.localization.Localization(__file__, 243, 0), 'scipy.optimize.minpack', sys_modules_204744.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 243, 0), __file__, sys_modules_204744, sys_modules_204744.module_type_store, module_type_store)
    else:
        from scipy.optimize.minpack import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 243, 0), 'scipy.optimize.minpack', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.optimize.minpack' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'scipy.optimize.minpack', import_204743)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 244, 0))

# 'from scipy.optimize.zeros import ' statement (line 244)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204745 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 244, 0), 'scipy.optimize.zeros')

if (type(import_204745) is not StypyTypeError):

    if (import_204745 != 'pyd_module'):
        __import__(import_204745)
        sys_modules_204746 = sys.modules[import_204745]
        import_from_module(stypy.reporting.localization.Localization(__file__, 244, 0), 'scipy.optimize.zeros', sys_modules_204746.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 244, 0), __file__, sys_modules_204746, sys_modules_204746.module_type_store, module_type_store)
    else:
        from scipy.optimize.zeros import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 244, 0), 'scipy.optimize.zeros', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.optimize.zeros' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'scipy.optimize.zeros', import_204745)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 245, 0))

# 'from scipy.optimize.lbfgsb import fmin_l_bfgs_b, LbfgsInvHessProduct' statement (line 245)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204747 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 245, 0), 'scipy.optimize.lbfgsb')

if (type(import_204747) is not StypyTypeError):

    if (import_204747 != 'pyd_module'):
        __import__(import_204747)
        sys_modules_204748 = sys.modules[import_204747]
        import_from_module(stypy.reporting.localization.Localization(__file__, 245, 0), 'scipy.optimize.lbfgsb', sys_modules_204748.module_type_store, module_type_store, ['fmin_l_bfgs_b', 'LbfgsInvHessProduct'])
        nest_module(stypy.reporting.localization.Localization(__file__, 245, 0), __file__, sys_modules_204748, sys_modules_204748.module_type_store, module_type_store)
    else:
        from scipy.optimize.lbfgsb import fmin_l_bfgs_b, LbfgsInvHessProduct

        import_from_module(stypy.reporting.localization.Localization(__file__, 245, 0), 'scipy.optimize.lbfgsb', None, module_type_store, ['fmin_l_bfgs_b', 'LbfgsInvHessProduct'], [fmin_l_bfgs_b, LbfgsInvHessProduct])

else:
    # Assigning a type to the variable 'scipy.optimize.lbfgsb' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 0), 'scipy.optimize.lbfgsb', import_204747)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 246, 0))

# 'from scipy.optimize.tnc import fmin_tnc' statement (line 246)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204749 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 246, 0), 'scipy.optimize.tnc')

if (type(import_204749) is not StypyTypeError):

    if (import_204749 != 'pyd_module'):
        __import__(import_204749)
        sys_modules_204750 = sys.modules[import_204749]
        import_from_module(stypy.reporting.localization.Localization(__file__, 246, 0), 'scipy.optimize.tnc', sys_modules_204750.module_type_store, module_type_store, ['fmin_tnc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 246, 0), __file__, sys_modules_204750, sys_modules_204750.module_type_store, module_type_store)
    else:
        from scipy.optimize.tnc import fmin_tnc

        import_from_module(stypy.reporting.localization.Localization(__file__, 246, 0), 'scipy.optimize.tnc', None, module_type_store, ['fmin_tnc'], [fmin_tnc])

else:
    # Assigning a type to the variable 'scipy.optimize.tnc' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 0), 'scipy.optimize.tnc', import_204749)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 247, 0))

# 'from scipy.optimize.cobyla import fmin_cobyla' statement (line 247)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204751 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 247, 0), 'scipy.optimize.cobyla')

if (type(import_204751) is not StypyTypeError):

    if (import_204751 != 'pyd_module'):
        __import__(import_204751)
        sys_modules_204752 = sys.modules[import_204751]
        import_from_module(stypy.reporting.localization.Localization(__file__, 247, 0), 'scipy.optimize.cobyla', sys_modules_204752.module_type_store, module_type_store, ['fmin_cobyla'])
        nest_module(stypy.reporting.localization.Localization(__file__, 247, 0), __file__, sys_modules_204752, sys_modules_204752.module_type_store, module_type_store)
    else:
        from scipy.optimize.cobyla import fmin_cobyla

        import_from_module(stypy.reporting.localization.Localization(__file__, 247, 0), 'scipy.optimize.cobyla', None, module_type_store, ['fmin_cobyla'], [fmin_cobyla])

else:
    # Assigning a type to the variable 'scipy.optimize.cobyla' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 0), 'scipy.optimize.cobyla', import_204751)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 248, 0))

# 'from scipy.optimize.nonlin import ' statement (line 248)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204753 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 248, 0), 'scipy.optimize.nonlin')

if (type(import_204753) is not StypyTypeError):

    if (import_204753 != 'pyd_module'):
        __import__(import_204753)
        sys_modules_204754 = sys.modules[import_204753]
        import_from_module(stypy.reporting.localization.Localization(__file__, 248, 0), 'scipy.optimize.nonlin', sys_modules_204754.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 248, 0), __file__, sys_modules_204754, sys_modules_204754.module_type_store, module_type_store)
    else:
        from scipy.optimize.nonlin import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 248, 0), 'scipy.optimize.nonlin', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.optimize.nonlin' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), 'scipy.optimize.nonlin', import_204753)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 249, 0))

# 'from scipy.optimize.slsqp import fmin_slsqp' statement (line 249)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204755 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 249, 0), 'scipy.optimize.slsqp')

if (type(import_204755) is not StypyTypeError):

    if (import_204755 != 'pyd_module'):
        __import__(import_204755)
        sys_modules_204756 = sys.modules[import_204755]
        import_from_module(stypy.reporting.localization.Localization(__file__, 249, 0), 'scipy.optimize.slsqp', sys_modules_204756.module_type_store, module_type_store, ['fmin_slsqp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 249, 0), __file__, sys_modules_204756, sys_modules_204756.module_type_store, module_type_store)
    else:
        from scipy.optimize.slsqp import fmin_slsqp

        import_from_module(stypy.reporting.localization.Localization(__file__, 249, 0), 'scipy.optimize.slsqp', None, module_type_store, ['fmin_slsqp'], [fmin_slsqp])

else:
    # Assigning a type to the variable 'scipy.optimize.slsqp' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'scipy.optimize.slsqp', import_204755)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 250, 0))

# 'from scipy.optimize.nnls import nnls' statement (line 250)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204757 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 250, 0), 'scipy.optimize.nnls')

if (type(import_204757) is not StypyTypeError):

    if (import_204757 != 'pyd_module'):
        __import__(import_204757)
        sys_modules_204758 = sys.modules[import_204757]
        import_from_module(stypy.reporting.localization.Localization(__file__, 250, 0), 'scipy.optimize.nnls', sys_modules_204758.module_type_store, module_type_store, ['nnls'])
        nest_module(stypy.reporting.localization.Localization(__file__, 250, 0), __file__, sys_modules_204758, sys_modules_204758.module_type_store, module_type_store)
    else:
        from scipy.optimize.nnls import nnls

        import_from_module(stypy.reporting.localization.Localization(__file__, 250, 0), 'scipy.optimize.nnls', None, module_type_store, ['nnls'], [nnls])

else:
    # Assigning a type to the variable 'scipy.optimize.nnls' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), 'scipy.optimize.nnls', import_204757)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 251, 0))

# 'from scipy.optimize._basinhopping import basinhopping' statement (line 251)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204759 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 251, 0), 'scipy.optimize._basinhopping')

if (type(import_204759) is not StypyTypeError):

    if (import_204759 != 'pyd_module'):
        __import__(import_204759)
        sys_modules_204760 = sys.modules[import_204759]
        import_from_module(stypy.reporting.localization.Localization(__file__, 251, 0), 'scipy.optimize._basinhopping', sys_modules_204760.module_type_store, module_type_store, ['basinhopping'])
        nest_module(stypy.reporting.localization.Localization(__file__, 251, 0), __file__, sys_modules_204760, sys_modules_204760.module_type_store, module_type_store)
    else:
        from scipy.optimize._basinhopping import basinhopping

        import_from_module(stypy.reporting.localization.Localization(__file__, 251, 0), 'scipy.optimize._basinhopping', None, module_type_store, ['basinhopping'], [basinhopping])

else:
    # Assigning a type to the variable 'scipy.optimize._basinhopping' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 0), 'scipy.optimize._basinhopping', import_204759)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 252, 0))

# 'from scipy.optimize._linprog import linprog, linprog_verbose_callback' statement (line 252)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204761 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 252, 0), 'scipy.optimize._linprog')

if (type(import_204761) is not StypyTypeError):

    if (import_204761 != 'pyd_module'):
        __import__(import_204761)
        sys_modules_204762 = sys.modules[import_204761]
        import_from_module(stypy.reporting.localization.Localization(__file__, 252, 0), 'scipy.optimize._linprog', sys_modules_204762.module_type_store, module_type_store, ['linprog', 'linprog_verbose_callback'])
        nest_module(stypy.reporting.localization.Localization(__file__, 252, 0), __file__, sys_modules_204762, sys_modules_204762.module_type_store, module_type_store)
    else:
        from scipy.optimize._linprog import linprog, linprog_verbose_callback

        import_from_module(stypy.reporting.localization.Localization(__file__, 252, 0), 'scipy.optimize._linprog', None, module_type_store, ['linprog', 'linprog_verbose_callback'], [linprog, linprog_verbose_callback])

else:
    # Assigning a type to the variable 'scipy.optimize._linprog' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 0), 'scipy.optimize._linprog', import_204761)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 253, 0))

# 'from scipy.optimize._hungarian import linear_sum_assignment' statement (line 253)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204763 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 253, 0), 'scipy.optimize._hungarian')

if (type(import_204763) is not StypyTypeError):

    if (import_204763 != 'pyd_module'):
        __import__(import_204763)
        sys_modules_204764 = sys.modules[import_204763]
        import_from_module(stypy.reporting.localization.Localization(__file__, 253, 0), 'scipy.optimize._hungarian', sys_modules_204764.module_type_store, module_type_store, ['linear_sum_assignment'])
        nest_module(stypy.reporting.localization.Localization(__file__, 253, 0), __file__, sys_modules_204764, sys_modules_204764.module_type_store, module_type_store)
    else:
        from scipy.optimize._hungarian import linear_sum_assignment

        import_from_module(stypy.reporting.localization.Localization(__file__, 253, 0), 'scipy.optimize._hungarian', None, module_type_store, ['linear_sum_assignment'], [linear_sum_assignment])

else:
    # Assigning a type to the variable 'scipy.optimize._hungarian' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 0), 'scipy.optimize._hungarian', import_204763)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 254, 0))

# 'from scipy.optimize._differentialevolution import differential_evolution' statement (line 254)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204765 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 254, 0), 'scipy.optimize._differentialevolution')

if (type(import_204765) is not StypyTypeError):

    if (import_204765 != 'pyd_module'):
        __import__(import_204765)
        sys_modules_204766 = sys.modules[import_204765]
        import_from_module(stypy.reporting.localization.Localization(__file__, 254, 0), 'scipy.optimize._differentialevolution', sys_modules_204766.module_type_store, module_type_store, ['differential_evolution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 254, 0), __file__, sys_modules_204766, sys_modules_204766.module_type_store, module_type_store)
    else:
        from scipy.optimize._differentialevolution import differential_evolution

        import_from_module(stypy.reporting.localization.Localization(__file__, 254, 0), 'scipy.optimize._differentialevolution', None, module_type_store, ['differential_evolution'], [differential_evolution])

else:
    # Assigning a type to the variable 'scipy.optimize._differentialevolution' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 0), 'scipy.optimize._differentialevolution', import_204765)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 255, 0))

# 'from scipy.optimize._lsq import least_squares, lsq_linear' statement (line 255)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204767 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 255, 0), 'scipy.optimize._lsq')

if (type(import_204767) is not StypyTypeError):

    if (import_204767 != 'pyd_module'):
        __import__(import_204767)
        sys_modules_204768 = sys.modules[import_204767]
        import_from_module(stypy.reporting.localization.Localization(__file__, 255, 0), 'scipy.optimize._lsq', sys_modules_204768.module_type_store, module_type_store, ['least_squares', 'lsq_linear'])
        nest_module(stypy.reporting.localization.Localization(__file__, 255, 0), __file__, sys_modules_204768, sys_modules_204768.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq import least_squares, lsq_linear

        import_from_module(stypy.reporting.localization.Localization(__file__, 255, 0), 'scipy.optimize._lsq', None, module_type_store, ['least_squares', 'lsq_linear'], [least_squares, lsq_linear])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 0), 'scipy.optimize._lsq', import_204767)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a ListComp to a Name (line 257):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 257)
# Processing the call keyword arguments (line 257)
kwargs_204777 = {}
# Getting the type of 'dir' (line 257)
dir_204776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 22), 'dir', False)
# Calling dir(args, kwargs) (line 257)
dir_call_result_204778 = invoke(stypy.reporting.localization.Localization(__file__, 257, 22), dir_204776, *[], **kwargs_204777)

comprehension_204779 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 11), dir_call_result_204778)
# Assigning a type to the variable 's' (line 257)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 's', comprehension_204779)


# Call to startswith(...): (line 257)
# Processing the call arguments (line 257)
str_204772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 48), 'str', '_')
# Processing the call keyword arguments (line 257)
kwargs_204773 = {}
# Getting the type of 's' (line 257)
s_204770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 257)
startswith_204771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 35), s_204770, 'startswith')
# Calling startswith(args, kwargs) (line 257)
startswith_call_result_204774 = invoke(stypy.reporting.localization.Localization(__file__, 257, 35), startswith_204771, *[str_204772], **kwargs_204773)

# Applying the 'not' unary operator (line 257)
result_not__204775 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 31), 'not', startswith_call_result_204774)

# Getting the type of 's' (line 257)
s_204769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 's')
list_204780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 11), list_204780, s_204769)
# Assigning a type to the variable '__all__' (line 257)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), '__all__', list_204780)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 259, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 259)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204781 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 259, 0), 'scipy._lib._testutils')

if (type(import_204781) is not StypyTypeError):

    if (import_204781 != 'pyd_module'):
        __import__(import_204781)
        sys_modules_204782 = sys.modules[import_204781]
        import_from_module(stypy.reporting.localization.Localization(__file__, 259, 0), 'scipy._lib._testutils', sys_modules_204782.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 259, 0), __file__, sys_modules_204782, sys_modules_204782.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 259, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), 'scipy._lib._testutils', import_204781)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a Call to a Name (line 260):

# Call to PytestTester(...): (line 260)
# Processing the call arguments (line 260)
# Getting the type of '__name__' (line 260)
name___204784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 20), '__name__', False)
# Processing the call keyword arguments (line 260)
kwargs_204785 = {}
# Getting the type of 'PytestTester' (line 260)
PytestTester_204783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 260)
PytestTester_call_result_204786 = invoke(stypy.reporting.localization.Localization(__file__, 260, 7), PytestTester_204783, *[name___204784], **kwargs_204785)

# Assigning a type to the variable 'test' (line 260)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 0), 'test', PytestTester_call_result_204786)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 261, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
