
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Interface to Constrained Optimization By Linear Approximation
3: 
4: Functions
5: ---------
6: .. autosummary::
7:    :toctree: generated/
8: 
9:     fmin_cobyla
10: 
11: '''
12: 
13: from __future__ import division, print_function, absolute_import
14: 
15: import numpy as np
16: from scipy._lib.six import callable
17: from scipy.optimize import _cobyla
18: from .optimize import OptimizeResult, _check_unknown_options
19: try:
20:     from itertools import izip
21: except ImportError:
22:     izip = zip
23: 
24: 
25: __all__ = ['fmin_cobyla']
26: 
27: 
28: def fmin_cobyla(func, x0, cons, args=(), consargs=None, rhobeg=1.0,
29:                 rhoend=1e-4, maxfun=1000, disp=None, catol=2e-4):
30:     '''
31:     Minimize a function using the Constrained Optimization BY Linear
32:     Approximation (COBYLA) method. This method wraps a FORTRAN
33:     implementation of the algorithm.
34: 
35:     Parameters
36:     ----------
37:     func : callable
38:         Function to minimize. In the form func(x, \\*args).
39:     x0 : ndarray
40:         Initial guess.
41:     cons : sequence
42:         Constraint functions; must all be ``>=0`` (a single function
43:         if only 1 constraint). Each function takes the parameters `x`
44:         as its first argument, and it can return either a single number or
45:         an array or list of numbers.
46:     args : tuple, optional
47:         Extra arguments to pass to function.
48:     consargs : tuple, optional
49:         Extra arguments to pass to constraint functions (default of None means
50:         use same extra arguments as those passed to func).
51:         Use ``()`` for no extra arguments.
52:     rhobeg : float, optional
53:         Reasonable initial changes to the variables.
54:     rhoend : float, optional
55:         Final accuracy in the optimization (not precisely guaranteed). This
56:         is a lower bound on the size of the trust region.
57:     disp : {0, 1, 2, 3}, optional
58:         Controls the frequency of output; 0 implies no output.
59:     maxfun : int, optional
60:         Maximum number of function evaluations.
61:     catol : float, optional
62:         Absolute tolerance for constraint violations.
63: 
64:     Returns
65:     -------
66:     x : ndarray
67:         The argument that minimises `f`.
68: 
69:     See also
70:     --------
71:     minimize: Interface to minimization algorithms for multivariate
72:         functions. See the 'COBYLA' `method` in particular.
73: 
74:     Notes
75:     -----
76:     This algorithm is based on linear approximations to the objective
77:     function and each constraint. We briefly describe the algorithm.
78: 
79:     Suppose the function is being minimized over k variables. At the
80:     jth iteration the algorithm has k+1 points v_1, ..., v_(k+1),
81:     an approximate solution x_j, and a radius RHO_j.
82:     (i.e. linear plus a constant) approximations to the objective
83:     function and constraint functions such that their function values
84:     agree with the linear approximation on the k+1 points v_1,.., v_(k+1).
85:     This gives a linear program to solve (where the linear approximations
86:     of the constraint functions are constrained to be non-negative).
87: 
88:     However the linear approximations are likely only good
89:     approximations near the current simplex, so the linear program is
90:     given the further requirement that the solution, which
91:     will become x_(j+1), must be within RHO_j from x_j. RHO_j only
92:     decreases, never increases. The initial RHO_j is rhobeg and the
93:     final RHO_j is rhoend. In this way COBYLA's iterations behave
94:     like a trust region algorithm.
95: 
96:     Additionally, the linear program may be inconsistent, or the
97:     approximation may give poor improvement. For details about
98:     how these issues are resolved, as well as how the points v_i are
99:     updated, refer to the source code or the references below.
100: 
101: 
102:     References
103:     ----------
104:     Powell M.J.D. (1994), "A direct search optimization method that models
105:     the objective and constraint functions by linear interpolation.", in
106:     Advances in Optimization and Numerical Analysis, eds. S. Gomez and
107:     J-P Hennart, Kluwer Academic (Dordrecht), pp. 51-67
108: 
109:     Powell M.J.D. (1998), "Direct search algorithms for optimization
110:     calculations", Acta Numerica 7, 287-336
111: 
112:     Powell M.J.D. (2007), "A view of algorithms for optimization without
113:     derivatives", Cambridge University Technical Report DAMTP 2007/NA03
114: 
115: 
116:     Examples
117:     --------
118:     Minimize the objective function f(x,y) = x*y subject
119:     to the constraints x**2 + y**2 < 1 and y > 0::
120: 
121:         >>> def objective(x):
122:         ...     return x[0]*x[1]
123:         ...
124:         >>> def constr1(x):
125:         ...     return 1 - (x[0]**2 + x[1]**2)
126:         ...
127:         >>> def constr2(x):
128:         ...     return x[1]
129:         ...
130:         >>> from scipy.optimize import fmin_cobyla
131:         >>> fmin_cobyla(objective, [0.0, 0.1], [constr1, constr2], rhoend=1e-7)
132:         array([-0.70710685,  0.70710671])
133: 
134:     The exact solution is (-sqrt(2)/2, sqrt(2)/2).
135: 
136: 
137: 
138:     '''
139:     err = "cons must be a sequence of callable functions or a single"\
140:           " callable function."
141:     try:
142:         len(cons)
143:     except TypeError:
144:         if callable(cons):
145:             cons = [cons]
146:         else:
147:             raise TypeError(err)
148:     else:
149:         for thisfunc in cons:
150:             if not callable(thisfunc):
151:                 raise TypeError(err)
152: 
153:     if consargs is None:
154:         consargs = args
155: 
156:     # build constraints
157:     con = tuple({'type': 'ineq', 'fun': c, 'args': consargs} for c in cons)
158: 
159:     # options
160:     opts = {'rhobeg': rhobeg,
161:             'tol': rhoend,
162:             'disp': disp,
163:             'maxiter': maxfun,
164:             'catol': catol}
165: 
166:     sol = _minimize_cobyla(func, x0, args, constraints=con,
167:                            **opts)
168:     if disp and not sol['success']:
169:         print("COBYLA failed to find a solution: %s" % (sol.message,))
170:     return sol['x']
171: 
172: 
173: def _minimize_cobyla(fun, x0, args=(), constraints=(),
174:                      rhobeg=1.0, tol=1e-4, maxiter=1000,
175:                      disp=False, catol=2e-4, **unknown_options):
176:     '''
177:     Minimize a scalar function of one or more variables using the
178:     Constrained Optimization BY Linear Approximation (COBYLA) algorithm.
179: 
180:     Options
181:     -------
182:     rhobeg : float
183:         Reasonable initial changes to the variables.
184:     tol : float
185:         Final accuracy in the optimization (not precisely guaranteed).
186:         This is a lower bound on the size of the trust region.
187:     disp : bool
188:         Set to True to print convergence messages. If False,
189:         `verbosity` is ignored as set to 0.
190:     maxiter : int
191:         Maximum number of function evaluations.
192:     catol : float
193:         Tolerance (absolute) for constraint violations
194: 
195:     '''
196:     _check_unknown_options(unknown_options)
197:     maxfun = maxiter
198:     rhoend = tol
199:     if not disp:
200:         iprint = 0
201: 
202:     # check constraints
203:     if isinstance(constraints, dict):
204:         constraints = (constraints, )
205: 
206:     for ic, con in enumerate(constraints):
207:         # check type
208:         try:
209:             ctype = con['type'].lower()
210:         except KeyError:
211:             raise KeyError('Constraint %d has no type defined.' % ic)
212:         except TypeError:
213:             raise TypeError('Constraints must be defined using a '
214:                             'dictionary.')
215:         except AttributeError:
216:             raise TypeError("Constraint's type must be a string.")
217:         else:
218:             if ctype != 'ineq':
219:                 raise ValueError("Constraints of type '%s' not handled by "
220:                                  "COBYLA." % con['type'])
221: 
222:         # check function
223:         if 'fun' not in con:
224:             raise KeyError('Constraint %d has no function defined.' % ic)
225: 
226:         # check extra arguments
227:         if 'args' not in con:
228:             con['args'] = ()
229: 
230:     # m is the total number of constraint values
231:     # it takes into account that some constraints may be vector-valued
232:     cons_lengths = []
233:     for c in constraints:
234:         f = c['fun'](x0, *c['args'])
235:         try:
236:             cons_length = len(f)
237:         except TypeError:
238:             cons_length = 1
239:         cons_lengths.append(cons_length)
240:     m = sum(cons_lengths)
241: 
242:     def calcfc(x, con):
243:         f = fun(x, *args)
244:         i = 0
245:         for size, c in izip(cons_lengths, constraints):
246:             con[i: i + size] = c['fun'](x, *c['args'])
247:             i += size
248:         return f
249: 
250:     info = np.zeros(4, np.float64)
251:     xopt, info = _cobyla.minimize(calcfc, m=m, x=np.copy(x0), rhobeg=rhobeg,
252:                                   rhoend=rhoend, iprint=iprint, maxfun=maxfun,
253:                                   dinfo=info)
254: 
255:     if info[3] > catol:
256:         # Check constraint violation
257:         info[0] = 4
258: 
259:     return OptimizeResult(x=xopt,
260:                           status=int(info[0]),
261:                           success=info[0] == 1,
262:                           message={1: 'Optimization terminated successfully.',
263:                                    2: 'Maximum number of function evaluations has '
264:                                       'been exceeded.',
265:                                    3: 'Rounding errors are becoming damaging in '
266:                                       'COBYLA subroutine.',
267:                                    4: 'Did not converge to a solution satisfying '
268:                                       'the constraints. See `maxcv` for magnitude '
269:                                       'of violation.'
270:                                    }.get(info[0], 'Unknown exit status.'),
271:                           nfev=int(info[1]),
272:                           fun=info[2],
273:                           maxcv=info[3])
274: 
275: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_167260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\nInterface to Constrained Optimization By Linear Approximation\n\nFunctions\n---------\n.. autosummary::\n   :toctree: generated/\n\n    fmin_cobyla\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import numpy' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_167261 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy')

if (type(import_167261) is not StypyTypeError):

    if (import_167261 != 'pyd_module'):
        __import__(import_167261)
        sys_modules_167262 = sys.modules[import_167261]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'np', sys_modules_167262.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy', import_167261)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy._lib.six import callable' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_167263 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib.six')

if (type(import_167263) is not StypyTypeError):

    if (import_167263 != 'pyd_module'):
        __import__(import_167263)
        sys_modules_167264 = sys.modules[import_167263]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib.six', sys_modules_167264.module_type_store, module_type_store, ['callable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_167264, sys_modules_167264.module_type_store, module_type_store)
    else:
        from scipy._lib.six import callable

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib.six', None, module_type_store, ['callable'], [callable])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib.six', import_167263)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.optimize import _cobyla' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_167265 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize')

if (type(import_167265) is not StypyTypeError):

    if (import_167265 != 'pyd_module'):
        __import__(import_167265)
        sys_modules_167266 = sys.modules[import_167265]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize', sys_modules_167266.module_type_store, module_type_store, ['_cobyla'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_167266, sys_modules_167266.module_type_store, module_type_store)
    else:
        from scipy.optimize import _cobyla

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize', None, module_type_store, ['_cobyla'], [_cobyla])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize', import_167265)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.optimize.optimize import OptimizeResult, _check_unknown_options' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_167267 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize.optimize')

if (type(import_167267) is not StypyTypeError):

    if (import_167267 != 'pyd_module'):
        __import__(import_167267)
        sys_modules_167268 = sys.modules[import_167267]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize.optimize', sys_modules_167268.module_type_store, module_type_store, ['OptimizeResult', '_check_unknown_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_167268, sys_modules_167268.module_type_store, module_type_store)
    else:
        from scipy.optimize.optimize import OptimizeResult, _check_unknown_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize.optimize', None, module_type_store, ['OptimizeResult', '_check_unknown_options'], [OptimizeResult, _check_unknown_options])

else:
    # Assigning a type to the variable 'scipy.optimize.optimize' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize.optimize', import_167267)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')



# SSA begins for try-except statement (line 19)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 4))

# 'from itertools import izip' statement (line 20)
try:
    from itertools import izip

except:
    izip = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 20, 4), 'itertools', None, module_type_store, ['izip'], [izip])

# SSA branch for the except part of a try statement (line 19)
# SSA branch for the except 'ImportError' branch of a try statement (line 19)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 22):

# Assigning a Name to a Name (line 22):
# Getting the type of 'zip' (line 22)
zip_167269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'zip')
# Assigning a type to the variable 'izip' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'izip', zip_167269)
# SSA join for try-except statement (line 19)
module_type_store = module_type_store.join_ssa_context()


# Assigning a List to a Name (line 25):

# Assigning a List to a Name (line 25):
__all__ = ['fmin_cobyla']
module_type_store.set_exportable_members(['fmin_cobyla'])

# Obtaining an instance of the builtin type 'list' (line 25)
list_167270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
str_167271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'str', 'fmin_cobyla')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_167270, str_167271)

# Assigning a type to the variable '__all__' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), '__all__', list_167270)

@norecursion
def fmin_cobyla(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 28)
    tuple_167272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 28)
    
    # Getting the type of 'None' (line 28)
    None_167273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 50), 'None')
    float_167274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 63), 'float')
    float_167275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 23), 'float')
    int_167276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 36), 'int')
    # Getting the type of 'None' (line 29)
    None_167277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 47), 'None')
    float_167278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 59), 'float')
    defaults = [tuple_167272, None_167273, float_167274, float_167275, int_167276, None_167277, float_167278]
    # Create a new context for function 'fmin_cobyla'
    module_type_store = module_type_store.open_function_context('fmin_cobyla', 28, 0, False)
    
    # Passed parameters checking function
    fmin_cobyla.stypy_localization = localization
    fmin_cobyla.stypy_type_of_self = None
    fmin_cobyla.stypy_type_store = module_type_store
    fmin_cobyla.stypy_function_name = 'fmin_cobyla'
    fmin_cobyla.stypy_param_names_list = ['func', 'x0', 'cons', 'args', 'consargs', 'rhobeg', 'rhoend', 'maxfun', 'disp', 'catol']
    fmin_cobyla.stypy_varargs_param_name = None
    fmin_cobyla.stypy_kwargs_param_name = None
    fmin_cobyla.stypy_call_defaults = defaults
    fmin_cobyla.stypy_call_varargs = varargs
    fmin_cobyla.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fmin_cobyla', ['func', 'x0', 'cons', 'args', 'consargs', 'rhobeg', 'rhoend', 'maxfun', 'disp', 'catol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fmin_cobyla', localization, ['func', 'x0', 'cons', 'args', 'consargs', 'rhobeg', 'rhoend', 'maxfun', 'disp', 'catol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fmin_cobyla(...)' code ##################

    str_167279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, (-1)), 'str', '\n    Minimize a function using the Constrained Optimization BY Linear\n    Approximation (COBYLA) method. This method wraps a FORTRAN\n    implementation of the algorithm.\n\n    Parameters\n    ----------\n    func : callable\n        Function to minimize. In the form func(x, \\*args).\n    x0 : ndarray\n        Initial guess.\n    cons : sequence\n        Constraint functions; must all be ``>=0`` (a single function\n        if only 1 constraint). Each function takes the parameters `x`\n        as its first argument, and it can return either a single number or\n        an array or list of numbers.\n    args : tuple, optional\n        Extra arguments to pass to function.\n    consargs : tuple, optional\n        Extra arguments to pass to constraint functions (default of None means\n        use same extra arguments as those passed to func).\n        Use ``()`` for no extra arguments.\n    rhobeg : float, optional\n        Reasonable initial changes to the variables.\n    rhoend : float, optional\n        Final accuracy in the optimization (not precisely guaranteed). This\n        is a lower bound on the size of the trust region.\n    disp : {0, 1, 2, 3}, optional\n        Controls the frequency of output; 0 implies no output.\n    maxfun : int, optional\n        Maximum number of function evaluations.\n    catol : float, optional\n        Absolute tolerance for constraint violations.\n\n    Returns\n    -------\n    x : ndarray\n        The argument that minimises `f`.\n\n    See also\n    --------\n    minimize: Interface to minimization algorithms for multivariate\n        functions. See the \'COBYLA\' `method` in particular.\n\n    Notes\n    -----\n    This algorithm is based on linear approximations to the objective\n    function and each constraint. We briefly describe the algorithm.\n\n    Suppose the function is being minimized over k variables. At the\n    jth iteration the algorithm has k+1 points v_1, ..., v_(k+1),\n    an approximate solution x_j, and a radius RHO_j.\n    (i.e. linear plus a constant) approximations to the objective\n    function and constraint functions such that their function values\n    agree with the linear approximation on the k+1 points v_1,.., v_(k+1).\n    This gives a linear program to solve (where the linear approximations\n    of the constraint functions are constrained to be non-negative).\n\n    However the linear approximations are likely only good\n    approximations near the current simplex, so the linear program is\n    given the further requirement that the solution, which\n    will become x_(j+1), must be within RHO_j from x_j. RHO_j only\n    decreases, never increases. The initial RHO_j is rhobeg and the\n    final RHO_j is rhoend. In this way COBYLA\'s iterations behave\n    like a trust region algorithm.\n\n    Additionally, the linear program may be inconsistent, or the\n    approximation may give poor improvement. For details about\n    how these issues are resolved, as well as how the points v_i are\n    updated, refer to the source code or the references below.\n\n\n    References\n    ----------\n    Powell M.J.D. (1994), "A direct search optimization method that models\n    the objective and constraint functions by linear interpolation.", in\n    Advances in Optimization and Numerical Analysis, eds. S. Gomez and\n    J-P Hennart, Kluwer Academic (Dordrecht), pp. 51-67\n\n    Powell M.J.D. (1998), "Direct search algorithms for optimization\n    calculations", Acta Numerica 7, 287-336\n\n    Powell M.J.D. (2007), "A view of algorithms for optimization without\n    derivatives", Cambridge University Technical Report DAMTP 2007/NA03\n\n\n    Examples\n    --------\n    Minimize the objective function f(x,y) = x*y subject\n    to the constraints x**2 + y**2 < 1 and y > 0::\n\n        >>> def objective(x):\n        ...     return x[0]*x[1]\n        ...\n        >>> def constr1(x):\n        ...     return 1 - (x[0]**2 + x[1]**2)\n        ...\n        >>> def constr2(x):\n        ...     return x[1]\n        ...\n        >>> from scipy.optimize import fmin_cobyla\n        >>> fmin_cobyla(objective, [0.0, 0.1], [constr1, constr2], rhoend=1e-7)\n        array([-0.70710685,  0.70710671])\n\n    The exact solution is (-sqrt(2)/2, sqrt(2)/2).\n\n\n\n    ')
    
    # Assigning a Str to a Name (line 139):
    
    # Assigning a Str to a Name (line 139):
    str_167280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 10), 'str', 'cons must be a sequence of callable functions or a single callable function.')
    # Assigning a type to the variable 'err' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'err', str_167280)
    
    
    # SSA begins for try-except statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to len(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'cons' (line 142)
    cons_167282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'cons', False)
    # Processing the call keyword arguments (line 142)
    kwargs_167283 = {}
    # Getting the type of 'len' (line 142)
    len_167281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'len', False)
    # Calling len(args, kwargs) (line 142)
    len_call_result_167284 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), len_167281, *[cons_167282], **kwargs_167283)
    
    # SSA branch for the except part of a try statement (line 141)
    # SSA branch for the except 'TypeError' branch of a try statement (line 141)
    module_type_store.open_ssa_branch('except')
    
    
    # Call to callable(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'cons' (line 144)
    cons_167286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'cons', False)
    # Processing the call keyword arguments (line 144)
    kwargs_167287 = {}
    # Getting the type of 'callable' (line 144)
    callable_167285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'callable', False)
    # Calling callable(args, kwargs) (line 144)
    callable_call_result_167288 = invoke(stypy.reporting.localization.Localization(__file__, 144, 11), callable_167285, *[cons_167286], **kwargs_167287)
    
    # Testing the type of an if condition (line 144)
    if_condition_167289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 8), callable_call_result_167288)
    # Assigning a type to the variable 'if_condition_167289' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'if_condition_167289', if_condition_167289)
    # SSA begins for if statement (line 144)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 145):
    
    # Assigning a List to a Name (line 145):
    
    # Obtaining an instance of the builtin type 'list' (line 145)
    list_167290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 145)
    # Adding element type (line 145)
    # Getting the type of 'cons' (line 145)
    cons_167291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 20), 'cons')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 19), list_167290, cons_167291)
    
    # Assigning a type to the variable 'cons' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'cons', list_167290)
    # SSA branch for the else part of an if statement (line 144)
    module_type_store.open_ssa_branch('else')
    
    # Call to TypeError(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'err' (line 147)
    err_167293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 28), 'err', False)
    # Processing the call keyword arguments (line 147)
    kwargs_167294 = {}
    # Getting the type of 'TypeError' (line 147)
    TypeError_167292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 147)
    TypeError_call_result_167295 = invoke(stypy.reporting.localization.Localization(__file__, 147, 18), TypeError_167292, *[err_167293], **kwargs_167294)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 147, 12), TypeError_call_result_167295, 'raise parameter', BaseException)
    # SSA join for if statement (line 144)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else branch of a try statement (line 141)
    module_type_store.open_ssa_branch('except else')
    
    # Getting the type of 'cons' (line 149)
    cons_167296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'cons')
    # Testing the type of a for loop iterable (line 149)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 149, 8), cons_167296)
    # Getting the type of the for loop variable (line 149)
    for_loop_var_167297 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 149, 8), cons_167296)
    # Assigning a type to the variable 'thisfunc' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'thisfunc', for_loop_var_167297)
    # SSA begins for a for statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to callable(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'thisfunc' (line 150)
    thisfunc_167299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 28), 'thisfunc', False)
    # Processing the call keyword arguments (line 150)
    kwargs_167300 = {}
    # Getting the type of 'callable' (line 150)
    callable_167298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 'callable', False)
    # Calling callable(args, kwargs) (line 150)
    callable_call_result_167301 = invoke(stypy.reporting.localization.Localization(__file__, 150, 19), callable_167298, *[thisfunc_167299], **kwargs_167300)
    
    # Applying the 'not' unary operator (line 150)
    result_not__167302 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 15), 'not', callable_call_result_167301)
    
    # Testing the type of an if condition (line 150)
    if_condition_167303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 12), result_not__167302)
    # Assigning a type to the variable 'if_condition_167303' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'if_condition_167303', if_condition_167303)
    # SSA begins for if statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'err' (line 151)
    err_167305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 32), 'err', False)
    # Processing the call keyword arguments (line 151)
    kwargs_167306 = {}
    # Getting the type of 'TypeError' (line 151)
    TypeError_167304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 22), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 151)
    TypeError_call_result_167307 = invoke(stypy.reporting.localization.Localization(__file__, 151, 22), TypeError_167304, *[err_167305], **kwargs_167306)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 151, 16), TypeError_call_result_167307, 'raise parameter', BaseException)
    # SSA join for if statement (line 150)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 141)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 153)
    # Getting the type of 'consargs' (line 153)
    consargs_167308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 7), 'consargs')
    # Getting the type of 'None' (line 153)
    None_167309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'None')
    
    (may_be_167310, more_types_in_union_167311) = may_be_none(consargs_167308, None_167309)

    if may_be_167310:

        if more_types_in_union_167311:
            # Runtime conditional SSA (line 153)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 154):
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'args' (line 154)
        args_167312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 'args')
        # Assigning a type to the variable 'consargs' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'consargs', args_167312)

        if more_types_in_union_167311:
            # SSA join for if statement (line 153)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 157):
    
    # Assigning a Call to a Name (line 157):
    
    # Call to tuple(...): (line 157)
    # Processing the call arguments (line 157)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 157, 16, True)
    # Calculating comprehension expression
    # Getting the type of 'cons' (line 157)
    cons_167321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 70), 'cons', False)
    comprehension_167322 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 16), cons_167321)
    # Assigning a type to the variable 'c' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'c', comprehension_167322)
    
    # Obtaining an instance of the builtin type 'dict' (line 157)
    dict_167314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 157)
    # Adding element type (key, value) (line 157)
    str_167315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 17), 'str', 'type')
    str_167316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 25), 'str', 'ineq')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 16), dict_167314, (str_167315, str_167316))
    # Adding element type (key, value) (line 157)
    str_167317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 33), 'str', 'fun')
    # Getting the type of 'c' (line 157)
    c_167318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 40), 'c', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 16), dict_167314, (str_167317, c_167318))
    # Adding element type (key, value) (line 157)
    str_167319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 43), 'str', 'args')
    # Getting the type of 'consargs' (line 157)
    consargs_167320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 51), 'consargs', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 16), dict_167314, (str_167319, consargs_167320))
    
    list_167323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 16), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 16), list_167323, dict_167314)
    # Processing the call keyword arguments (line 157)
    kwargs_167324 = {}
    # Getting the type of 'tuple' (line 157)
    tuple_167313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 10), 'tuple', False)
    # Calling tuple(args, kwargs) (line 157)
    tuple_call_result_167325 = invoke(stypy.reporting.localization.Localization(__file__, 157, 10), tuple_167313, *[list_167323], **kwargs_167324)
    
    # Assigning a type to the variable 'con' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'con', tuple_call_result_167325)
    
    # Assigning a Dict to a Name (line 160):
    
    # Assigning a Dict to a Name (line 160):
    
    # Obtaining an instance of the builtin type 'dict' (line 160)
    dict_167326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 160)
    # Adding element type (key, value) (line 160)
    str_167327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 12), 'str', 'rhobeg')
    # Getting the type of 'rhobeg' (line 160)
    rhobeg_167328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 22), 'rhobeg')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 11), dict_167326, (str_167327, rhobeg_167328))
    # Adding element type (key, value) (line 160)
    str_167329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 12), 'str', 'tol')
    # Getting the type of 'rhoend' (line 161)
    rhoend_167330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'rhoend')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 11), dict_167326, (str_167329, rhoend_167330))
    # Adding element type (key, value) (line 160)
    str_167331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 12), 'str', 'disp')
    # Getting the type of 'disp' (line 162)
    disp_167332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'disp')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 11), dict_167326, (str_167331, disp_167332))
    # Adding element type (key, value) (line 160)
    str_167333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 12), 'str', 'maxiter')
    # Getting the type of 'maxfun' (line 163)
    maxfun_167334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 23), 'maxfun')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 11), dict_167326, (str_167333, maxfun_167334))
    # Adding element type (key, value) (line 160)
    str_167335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 12), 'str', 'catol')
    # Getting the type of 'catol' (line 164)
    catol_167336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'catol')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 11), dict_167326, (str_167335, catol_167336))
    
    # Assigning a type to the variable 'opts' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'opts', dict_167326)
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to _minimize_cobyla(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'func' (line 166)
    func_167338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 27), 'func', False)
    # Getting the type of 'x0' (line 166)
    x0_167339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 33), 'x0', False)
    # Getting the type of 'args' (line 166)
    args_167340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 37), 'args', False)
    # Processing the call keyword arguments (line 166)
    # Getting the type of 'con' (line 166)
    con_167341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 55), 'con', False)
    keyword_167342 = con_167341
    # Getting the type of 'opts' (line 167)
    opts_167343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 29), 'opts', False)
    kwargs_167344 = {'opts_167343': opts_167343, 'constraints': keyword_167342}
    # Getting the type of '_minimize_cobyla' (line 166)
    _minimize_cobyla_167337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 10), '_minimize_cobyla', False)
    # Calling _minimize_cobyla(args, kwargs) (line 166)
    _minimize_cobyla_call_result_167345 = invoke(stypy.reporting.localization.Localization(__file__, 166, 10), _minimize_cobyla_167337, *[func_167338, x0_167339, args_167340], **kwargs_167344)
    
    # Assigning a type to the variable 'sol' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'sol', _minimize_cobyla_call_result_167345)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'disp' (line 168)
    disp_167346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 7), 'disp')
    
    
    # Obtaining the type of the subscript
    str_167347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 24), 'str', 'success')
    # Getting the type of 'sol' (line 168)
    sol_167348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'sol')
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___167349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 20), sol_167348, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 168)
    subscript_call_result_167350 = invoke(stypy.reporting.localization.Localization(__file__, 168, 20), getitem___167349, str_167347)
    
    # Applying the 'not' unary operator (line 168)
    result_not__167351 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 16), 'not', subscript_call_result_167350)
    
    # Applying the binary operator 'and' (line 168)
    result_and_keyword_167352 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 7), 'and', disp_167346, result_not__167351)
    
    # Testing the type of an if condition (line 168)
    if_condition_167353 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 4), result_and_keyword_167352)
    # Assigning a type to the variable 'if_condition_167353' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'if_condition_167353', if_condition_167353)
    # SSA begins for if statement (line 168)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 169)
    # Processing the call arguments (line 169)
    str_167355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 14), 'str', 'COBYLA failed to find a solution: %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 169)
    tuple_167356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 169)
    # Adding element type (line 169)
    # Getting the type of 'sol' (line 169)
    sol_167357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 56), 'sol', False)
    # Obtaining the member 'message' of a type (line 169)
    message_167358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 56), sol_167357, 'message')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 56), tuple_167356, message_167358)
    
    # Applying the binary operator '%' (line 169)
    result_mod_167359 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 14), '%', str_167355, tuple_167356)
    
    # Processing the call keyword arguments (line 169)
    kwargs_167360 = {}
    # Getting the type of 'print' (line 169)
    print_167354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'print', False)
    # Calling print(args, kwargs) (line 169)
    print_call_result_167361 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), print_167354, *[result_mod_167359], **kwargs_167360)
    
    # SSA join for if statement (line 168)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    str_167362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 15), 'str', 'x')
    # Getting the type of 'sol' (line 170)
    sol_167363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'sol')
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___167364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 11), sol_167363, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_167365 = invoke(stypy.reporting.localization.Localization(__file__, 170, 11), getitem___167364, str_167362)
    
    # Assigning a type to the variable 'stypy_return_type' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type', subscript_call_result_167365)
    
    # ################# End of 'fmin_cobyla(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fmin_cobyla' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_167366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_167366)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fmin_cobyla'
    return stypy_return_type_167366

# Assigning a type to the variable 'fmin_cobyla' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'fmin_cobyla', fmin_cobyla)

@norecursion
def _minimize_cobyla(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 173)
    tuple_167367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 173)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 173)
    tuple_167368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 173)
    
    float_167369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 28), 'float')
    float_167370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 37), 'float')
    int_167371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 51), 'int')
    # Getting the type of 'False' (line 175)
    False_167372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 26), 'False')
    float_167373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 39), 'float')
    defaults = [tuple_167367, tuple_167368, float_167369, float_167370, int_167371, False_167372, float_167373]
    # Create a new context for function '_minimize_cobyla'
    module_type_store = module_type_store.open_function_context('_minimize_cobyla', 173, 0, False)
    
    # Passed parameters checking function
    _minimize_cobyla.stypy_localization = localization
    _minimize_cobyla.stypy_type_of_self = None
    _minimize_cobyla.stypy_type_store = module_type_store
    _minimize_cobyla.stypy_function_name = '_minimize_cobyla'
    _minimize_cobyla.stypy_param_names_list = ['fun', 'x0', 'args', 'constraints', 'rhobeg', 'tol', 'maxiter', 'disp', 'catol']
    _minimize_cobyla.stypy_varargs_param_name = None
    _minimize_cobyla.stypy_kwargs_param_name = 'unknown_options'
    _minimize_cobyla.stypy_call_defaults = defaults
    _minimize_cobyla.stypy_call_varargs = varargs
    _minimize_cobyla.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_minimize_cobyla', ['fun', 'x0', 'args', 'constraints', 'rhobeg', 'tol', 'maxiter', 'disp', 'catol'], None, 'unknown_options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_minimize_cobyla', localization, ['fun', 'x0', 'args', 'constraints', 'rhobeg', 'tol', 'maxiter', 'disp', 'catol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_minimize_cobyla(...)' code ##################

    str_167374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, (-1)), 'str', '\n    Minimize a scalar function of one or more variables using the\n    Constrained Optimization BY Linear Approximation (COBYLA) algorithm.\n\n    Options\n    -------\n    rhobeg : float\n        Reasonable initial changes to the variables.\n    tol : float\n        Final accuracy in the optimization (not precisely guaranteed).\n        This is a lower bound on the size of the trust region.\n    disp : bool\n        Set to True to print convergence messages. If False,\n        `verbosity` is ignored as set to 0.\n    maxiter : int\n        Maximum number of function evaluations.\n    catol : float\n        Tolerance (absolute) for constraint violations\n\n    ')
    
    # Call to _check_unknown_options(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'unknown_options' (line 196)
    unknown_options_167376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 27), 'unknown_options', False)
    # Processing the call keyword arguments (line 196)
    kwargs_167377 = {}
    # Getting the type of '_check_unknown_options' (line 196)
    _check_unknown_options_167375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), '_check_unknown_options', False)
    # Calling _check_unknown_options(args, kwargs) (line 196)
    _check_unknown_options_call_result_167378 = invoke(stypy.reporting.localization.Localization(__file__, 196, 4), _check_unknown_options_167375, *[unknown_options_167376], **kwargs_167377)
    
    
    # Assigning a Name to a Name (line 197):
    
    # Assigning a Name to a Name (line 197):
    # Getting the type of 'maxiter' (line 197)
    maxiter_167379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 13), 'maxiter')
    # Assigning a type to the variable 'maxfun' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'maxfun', maxiter_167379)
    
    # Assigning a Name to a Name (line 198):
    
    # Assigning a Name to a Name (line 198):
    # Getting the type of 'tol' (line 198)
    tol_167380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 13), 'tol')
    # Assigning a type to the variable 'rhoend' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'rhoend', tol_167380)
    
    
    # Getting the type of 'disp' (line 199)
    disp_167381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'disp')
    # Applying the 'not' unary operator (line 199)
    result_not__167382 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 7), 'not', disp_167381)
    
    # Testing the type of an if condition (line 199)
    if_condition_167383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 4), result_not__167382)
    # Assigning a type to the variable 'if_condition_167383' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'if_condition_167383', if_condition_167383)
    # SSA begins for if statement (line 199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 200):
    
    # Assigning a Num to a Name (line 200):
    int_167384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 17), 'int')
    # Assigning a type to the variable 'iprint' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'iprint', int_167384)
    # SSA join for if statement (line 199)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 203)
    # Getting the type of 'dict' (line 203)
    dict_167385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 31), 'dict')
    # Getting the type of 'constraints' (line 203)
    constraints_167386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 18), 'constraints')
    
    (may_be_167387, more_types_in_union_167388) = may_be_subtype(dict_167385, constraints_167386)

    if may_be_167387:

        if more_types_in_union_167388:
            # Runtime conditional SSA (line 203)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'constraints' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'constraints', remove_not_subtype_from_union(constraints_167386, dict))
        
        # Assigning a Tuple to a Name (line 204):
        
        # Assigning a Tuple to a Name (line 204):
        
        # Obtaining an instance of the builtin type 'tuple' (line 204)
        tuple_167389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 204)
        # Adding element type (line 204)
        # Getting the type of 'constraints' (line 204)
        constraints_167390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 23), 'constraints')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 23), tuple_167389, constraints_167390)
        
        # Assigning a type to the variable 'constraints' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'constraints', tuple_167389)

        if more_types_in_union_167388:
            # SSA join for if statement (line 203)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to enumerate(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'constraints' (line 206)
    constraints_167392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 29), 'constraints', False)
    # Processing the call keyword arguments (line 206)
    kwargs_167393 = {}
    # Getting the type of 'enumerate' (line 206)
    enumerate_167391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 206)
    enumerate_call_result_167394 = invoke(stypy.reporting.localization.Localization(__file__, 206, 19), enumerate_167391, *[constraints_167392], **kwargs_167393)
    
    # Testing the type of a for loop iterable (line 206)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 206, 4), enumerate_call_result_167394)
    # Getting the type of the for loop variable (line 206)
    for_loop_var_167395 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 206, 4), enumerate_call_result_167394)
    # Assigning a type to the variable 'ic' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'ic', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 4), for_loop_var_167395))
    # Assigning a type to the variable 'con' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'con', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 4), for_loop_var_167395))
    # SSA begins for a for statement (line 206)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 208)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to lower(...): (line 209)
    # Processing the call keyword arguments (line 209)
    kwargs_167401 = {}
    
    # Obtaining the type of the subscript
    str_167396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 24), 'str', 'type')
    # Getting the type of 'con' (line 209)
    con_167397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'con', False)
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___167398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 20), con_167397, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_167399 = invoke(stypy.reporting.localization.Localization(__file__, 209, 20), getitem___167398, str_167396)
    
    # Obtaining the member 'lower' of a type (line 209)
    lower_167400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 20), subscript_call_result_167399, 'lower')
    # Calling lower(args, kwargs) (line 209)
    lower_call_result_167402 = invoke(stypy.reporting.localization.Localization(__file__, 209, 20), lower_167400, *[], **kwargs_167401)
    
    # Assigning a type to the variable 'ctype' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'ctype', lower_call_result_167402)
    # SSA branch for the except part of a try statement (line 208)
    # SSA branch for the except 'KeyError' branch of a try statement (line 208)
    module_type_store.open_ssa_branch('except')
    
    # Call to KeyError(...): (line 211)
    # Processing the call arguments (line 211)
    str_167404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 27), 'str', 'Constraint %d has no type defined.')
    # Getting the type of 'ic' (line 211)
    ic_167405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 66), 'ic', False)
    # Applying the binary operator '%' (line 211)
    result_mod_167406 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 27), '%', str_167404, ic_167405)
    
    # Processing the call keyword arguments (line 211)
    kwargs_167407 = {}
    # Getting the type of 'KeyError' (line 211)
    KeyError_167403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'KeyError', False)
    # Calling KeyError(args, kwargs) (line 211)
    KeyError_call_result_167408 = invoke(stypy.reporting.localization.Localization(__file__, 211, 18), KeyError_167403, *[result_mod_167406], **kwargs_167407)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 211, 12), KeyError_call_result_167408, 'raise parameter', BaseException)
    # SSA branch for the except 'TypeError' branch of a try statement (line 208)
    module_type_store.open_ssa_branch('except')
    
    # Call to TypeError(...): (line 213)
    # Processing the call arguments (line 213)
    str_167410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 28), 'str', 'Constraints must be defined using a dictionary.')
    # Processing the call keyword arguments (line 213)
    kwargs_167411 = {}
    # Getting the type of 'TypeError' (line 213)
    TypeError_167409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 213)
    TypeError_call_result_167412 = invoke(stypy.reporting.localization.Localization(__file__, 213, 18), TypeError_167409, *[str_167410], **kwargs_167411)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 213, 12), TypeError_call_result_167412, 'raise parameter', BaseException)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 208)
    module_type_store.open_ssa_branch('except')
    
    # Call to TypeError(...): (line 216)
    # Processing the call arguments (line 216)
    str_167414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 28), 'str', "Constraint's type must be a string.")
    # Processing the call keyword arguments (line 216)
    kwargs_167415 = {}
    # Getting the type of 'TypeError' (line 216)
    TypeError_167413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 216)
    TypeError_call_result_167416 = invoke(stypy.reporting.localization.Localization(__file__, 216, 18), TypeError_167413, *[str_167414], **kwargs_167415)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 216, 12), TypeError_call_result_167416, 'raise parameter', BaseException)
    # SSA branch for the else branch of a try statement (line 208)
    module_type_store.open_ssa_branch('except else')
    
    
    # Getting the type of 'ctype' (line 218)
    ctype_167417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'ctype')
    str_167418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 24), 'str', 'ineq')
    # Applying the binary operator '!=' (line 218)
    result_ne_167419 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 15), '!=', ctype_167417, str_167418)
    
    # Testing the type of an if condition (line 218)
    if_condition_167420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 12), result_ne_167419)
    # Assigning a type to the variable 'if_condition_167420' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'if_condition_167420', if_condition_167420)
    # SSA begins for if statement (line 218)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 219)
    # Processing the call arguments (line 219)
    str_167422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 33), 'str', "Constraints of type '%s' not handled by COBYLA.")
    
    # Obtaining the type of the subscript
    str_167423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 49), 'str', 'type')
    # Getting the type of 'con' (line 220)
    con_167424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 45), 'con', False)
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___167425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 45), con_167424, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_167426 = invoke(stypy.reporting.localization.Localization(__file__, 220, 45), getitem___167425, str_167423)
    
    # Applying the binary operator '%' (line 219)
    result_mod_167427 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 33), '%', str_167422, subscript_call_result_167426)
    
    # Processing the call keyword arguments (line 219)
    kwargs_167428 = {}
    # Getting the type of 'ValueError' (line 219)
    ValueError_167421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 219)
    ValueError_call_result_167429 = invoke(stypy.reporting.localization.Localization(__file__, 219, 22), ValueError_167421, *[result_mod_167427], **kwargs_167428)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 219, 16), ValueError_call_result_167429, 'raise parameter', BaseException)
    # SSA join for if statement (line 218)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 208)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_167430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 11), 'str', 'fun')
    # Getting the type of 'con' (line 223)
    con_167431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 24), 'con')
    # Applying the binary operator 'notin' (line 223)
    result_contains_167432 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 11), 'notin', str_167430, con_167431)
    
    # Testing the type of an if condition (line 223)
    if_condition_167433 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 8), result_contains_167432)
    # Assigning a type to the variable 'if_condition_167433' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'if_condition_167433', if_condition_167433)
    # SSA begins for if statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to KeyError(...): (line 224)
    # Processing the call arguments (line 224)
    str_167435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 27), 'str', 'Constraint %d has no function defined.')
    # Getting the type of 'ic' (line 224)
    ic_167436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 70), 'ic', False)
    # Applying the binary operator '%' (line 224)
    result_mod_167437 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 27), '%', str_167435, ic_167436)
    
    # Processing the call keyword arguments (line 224)
    kwargs_167438 = {}
    # Getting the type of 'KeyError' (line 224)
    KeyError_167434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 18), 'KeyError', False)
    # Calling KeyError(args, kwargs) (line 224)
    KeyError_call_result_167439 = invoke(stypy.reporting.localization.Localization(__file__, 224, 18), KeyError_167434, *[result_mod_167437], **kwargs_167438)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 224, 12), KeyError_call_result_167439, 'raise parameter', BaseException)
    # SSA join for if statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_167440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 11), 'str', 'args')
    # Getting the type of 'con' (line 227)
    con_167441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'con')
    # Applying the binary operator 'notin' (line 227)
    result_contains_167442 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 11), 'notin', str_167440, con_167441)
    
    # Testing the type of an if condition (line 227)
    if_condition_167443 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 8), result_contains_167442)
    # Assigning a type to the variable 'if_condition_167443' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'if_condition_167443', if_condition_167443)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Subscript (line 228):
    
    # Assigning a Tuple to a Subscript (line 228):
    
    # Obtaining an instance of the builtin type 'tuple' (line 228)
    tuple_167444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 228)
    
    # Getting the type of 'con' (line 228)
    con_167445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'con')
    str_167446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 16), 'str', 'args')
    # Storing an element on a container (line 228)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), con_167445, (str_167446, tuple_167444))
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 232):
    
    # Assigning a List to a Name (line 232):
    
    # Obtaining an instance of the builtin type 'list' (line 232)
    list_167447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 232)
    
    # Assigning a type to the variable 'cons_lengths' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'cons_lengths', list_167447)
    
    # Getting the type of 'constraints' (line 233)
    constraints_167448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 13), 'constraints')
    # Testing the type of a for loop iterable (line 233)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 233, 4), constraints_167448)
    # Getting the type of the for loop variable (line 233)
    for_loop_var_167449 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 233, 4), constraints_167448)
    # Assigning a type to the variable 'c' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'c', for_loop_var_167449)
    # SSA begins for a for statement (line 233)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 234):
    
    # Assigning a Call to a Name (line 234):
    
    # Call to (...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'x0' (line 234)
    x0_167454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 21), 'x0', False)
    
    # Obtaining the type of the subscript
    str_167455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 28), 'str', 'args')
    # Getting the type of 'c' (line 234)
    c_167456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 26), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 234)
    getitem___167457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 26), c_167456, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 234)
    subscript_call_result_167458 = invoke(stypy.reporting.localization.Localization(__file__, 234, 26), getitem___167457, str_167455)
    
    # Processing the call keyword arguments (line 234)
    kwargs_167459 = {}
    
    # Obtaining the type of the subscript
    str_167450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 14), 'str', 'fun')
    # Getting the type of 'c' (line 234)
    c_167451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 234)
    getitem___167452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), c_167451, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 234)
    subscript_call_result_167453 = invoke(stypy.reporting.localization.Localization(__file__, 234, 12), getitem___167452, str_167450)
    
    # Calling (args, kwargs) (line 234)
    _call_result_167460 = invoke(stypy.reporting.localization.Localization(__file__, 234, 12), subscript_call_result_167453, *[x0_167454, subscript_call_result_167458], **kwargs_167459)
    
    # Assigning a type to the variable 'f' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'f', _call_result_167460)
    
    
    # SSA begins for try-except statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 236):
    
    # Assigning a Call to a Name (line 236):
    
    # Call to len(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'f' (line 236)
    f_167462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 30), 'f', False)
    # Processing the call keyword arguments (line 236)
    kwargs_167463 = {}
    # Getting the type of 'len' (line 236)
    len_167461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 26), 'len', False)
    # Calling len(args, kwargs) (line 236)
    len_call_result_167464 = invoke(stypy.reporting.localization.Localization(__file__, 236, 26), len_167461, *[f_167462], **kwargs_167463)
    
    # Assigning a type to the variable 'cons_length' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'cons_length', len_call_result_167464)
    # SSA branch for the except part of a try statement (line 235)
    # SSA branch for the except 'TypeError' branch of a try statement (line 235)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Num to a Name (line 238):
    
    # Assigning a Num to a Name (line 238):
    int_167465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 26), 'int')
    # Assigning a type to the variable 'cons_length' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'cons_length', int_167465)
    # SSA join for try-except statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'cons_length' (line 239)
    cons_length_167468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 28), 'cons_length', False)
    # Processing the call keyword arguments (line 239)
    kwargs_167469 = {}
    # Getting the type of 'cons_lengths' (line 239)
    cons_lengths_167466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'cons_lengths', False)
    # Obtaining the member 'append' of a type (line 239)
    append_167467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), cons_lengths_167466, 'append')
    # Calling append(args, kwargs) (line 239)
    append_call_result_167470 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), append_167467, *[cons_length_167468], **kwargs_167469)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 240):
    
    # Assigning a Call to a Name (line 240):
    
    # Call to sum(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'cons_lengths' (line 240)
    cons_lengths_167472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'cons_lengths', False)
    # Processing the call keyword arguments (line 240)
    kwargs_167473 = {}
    # Getting the type of 'sum' (line 240)
    sum_167471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'sum', False)
    # Calling sum(args, kwargs) (line 240)
    sum_call_result_167474 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), sum_167471, *[cons_lengths_167472], **kwargs_167473)
    
    # Assigning a type to the variable 'm' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'm', sum_call_result_167474)

    @norecursion
    def calcfc(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'calcfc'
        module_type_store = module_type_store.open_function_context('calcfc', 242, 4, False)
        
        # Passed parameters checking function
        calcfc.stypy_localization = localization
        calcfc.stypy_type_of_self = None
        calcfc.stypy_type_store = module_type_store
        calcfc.stypy_function_name = 'calcfc'
        calcfc.stypy_param_names_list = ['x', 'con']
        calcfc.stypy_varargs_param_name = None
        calcfc.stypy_kwargs_param_name = None
        calcfc.stypy_call_defaults = defaults
        calcfc.stypy_call_varargs = varargs
        calcfc.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'calcfc', ['x', 'con'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'calcfc', localization, ['x', 'con'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'calcfc(...)' code ##################

        
        # Assigning a Call to a Name (line 243):
        
        # Assigning a Call to a Name (line 243):
        
        # Call to fun(...): (line 243)
        # Processing the call arguments (line 243)
        # Getting the type of 'x' (line 243)
        x_167476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'x', False)
        # Getting the type of 'args' (line 243)
        args_167477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'args', False)
        # Processing the call keyword arguments (line 243)
        kwargs_167478 = {}
        # Getting the type of 'fun' (line 243)
        fun_167475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'fun', False)
        # Calling fun(args, kwargs) (line 243)
        fun_call_result_167479 = invoke(stypy.reporting.localization.Localization(__file__, 243, 12), fun_167475, *[x_167476, args_167477], **kwargs_167478)
        
        # Assigning a type to the variable 'f' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'f', fun_call_result_167479)
        
        # Assigning a Num to a Name (line 244):
        
        # Assigning a Num to a Name (line 244):
        int_167480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 12), 'int')
        # Assigning a type to the variable 'i' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'i', int_167480)
        
        
        # Call to izip(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'cons_lengths' (line 245)
        cons_lengths_167482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 28), 'cons_lengths', False)
        # Getting the type of 'constraints' (line 245)
        constraints_167483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 42), 'constraints', False)
        # Processing the call keyword arguments (line 245)
        kwargs_167484 = {}
        # Getting the type of 'izip' (line 245)
        izip_167481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 23), 'izip', False)
        # Calling izip(args, kwargs) (line 245)
        izip_call_result_167485 = invoke(stypy.reporting.localization.Localization(__file__, 245, 23), izip_167481, *[cons_lengths_167482, constraints_167483], **kwargs_167484)
        
        # Testing the type of a for loop iterable (line 245)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 245, 8), izip_call_result_167485)
        # Getting the type of the for loop variable (line 245)
        for_loop_var_167486 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 245, 8), izip_call_result_167485)
        # Assigning a type to the variable 'size' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'size', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 8), for_loop_var_167486))
        # Assigning a type to the variable 'c' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 8), for_loop_var_167486))
        # SSA begins for a for statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Subscript (line 246):
        
        # Assigning a Call to a Subscript (line 246):
        
        # Call to (...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'x' (line 246)
        x_167491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 40), 'x', False)
        
        # Obtaining the type of the subscript
        str_167492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 46), 'str', 'args')
        # Getting the type of 'c' (line 246)
        c_167493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 44), 'c', False)
        # Obtaining the member '__getitem__' of a type (line 246)
        getitem___167494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 44), c_167493, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 246)
        subscript_call_result_167495 = invoke(stypy.reporting.localization.Localization(__file__, 246, 44), getitem___167494, str_167492)
        
        # Processing the call keyword arguments (line 246)
        kwargs_167496 = {}
        
        # Obtaining the type of the subscript
        str_167487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 33), 'str', 'fun')
        # Getting the type of 'c' (line 246)
        c_167488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 31), 'c', False)
        # Obtaining the member '__getitem__' of a type (line 246)
        getitem___167489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 31), c_167488, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 246)
        subscript_call_result_167490 = invoke(stypy.reporting.localization.Localization(__file__, 246, 31), getitem___167489, str_167487)
        
        # Calling (args, kwargs) (line 246)
        _call_result_167497 = invoke(stypy.reporting.localization.Localization(__file__, 246, 31), subscript_call_result_167490, *[x_167491, subscript_call_result_167495], **kwargs_167496)
        
        # Getting the type of 'con' (line 246)
        con_167498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'con')
        # Getting the type of 'i' (line 246)
        i_167499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'i')
        # Getting the type of 'i' (line 246)
        i_167500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 'i')
        # Getting the type of 'size' (line 246)
        size_167501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 23), 'size')
        # Applying the binary operator '+' (line 246)
        result_add_167502 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 19), '+', i_167500, size_167501)
        
        slice_167503 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 246, 12), i_167499, result_add_167502, None)
        # Storing an element on a container (line 246)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 12), con_167498, (slice_167503, _call_result_167497))
        
        # Getting the type of 'i' (line 247)
        i_167504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'i')
        # Getting the type of 'size' (line 247)
        size_167505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 17), 'size')
        # Applying the binary operator '+=' (line 247)
        result_iadd_167506 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 12), '+=', i_167504, size_167505)
        # Assigning a type to the variable 'i' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'i', result_iadd_167506)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'f' (line 248)
        f_167507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 15), 'f')
        # Assigning a type to the variable 'stypy_return_type' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'stypy_return_type', f_167507)
        
        # ################# End of 'calcfc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'calcfc' in the type store
        # Getting the type of 'stypy_return_type' (line 242)
        stypy_return_type_167508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_167508)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'calcfc'
        return stypy_return_type_167508

    # Assigning a type to the variable 'calcfc' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'calcfc', calcfc)
    
    # Assigning a Call to a Name (line 250):
    
    # Assigning a Call to a Name (line 250):
    
    # Call to zeros(...): (line 250)
    # Processing the call arguments (line 250)
    int_167511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 20), 'int')
    # Getting the type of 'np' (line 250)
    np_167512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 23), 'np', False)
    # Obtaining the member 'float64' of a type (line 250)
    float64_167513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 23), np_167512, 'float64')
    # Processing the call keyword arguments (line 250)
    kwargs_167514 = {}
    # Getting the type of 'np' (line 250)
    np_167509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 11), 'np', False)
    # Obtaining the member 'zeros' of a type (line 250)
    zeros_167510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 11), np_167509, 'zeros')
    # Calling zeros(args, kwargs) (line 250)
    zeros_call_result_167515 = invoke(stypy.reporting.localization.Localization(__file__, 250, 11), zeros_167510, *[int_167511, float64_167513], **kwargs_167514)
    
    # Assigning a type to the variable 'info' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'info', zeros_call_result_167515)
    
    # Assigning a Call to a Tuple (line 251):
    
    # Assigning a Subscript to a Name (line 251):
    
    # Obtaining the type of the subscript
    int_167516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 4), 'int')
    
    # Call to minimize(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'calcfc' (line 251)
    calcfc_167519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 34), 'calcfc', False)
    # Processing the call keyword arguments (line 251)
    # Getting the type of 'm' (line 251)
    m_167520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 44), 'm', False)
    keyword_167521 = m_167520
    
    # Call to copy(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'x0' (line 251)
    x0_167524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 57), 'x0', False)
    # Processing the call keyword arguments (line 251)
    kwargs_167525 = {}
    # Getting the type of 'np' (line 251)
    np_167522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 49), 'np', False)
    # Obtaining the member 'copy' of a type (line 251)
    copy_167523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 49), np_167522, 'copy')
    # Calling copy(args, kwargs) (line 251)
    copy_call_result_167526 = invoke(stypy.reporting.localization.Localization(__file__, 251, 49), copy_167523, *[x0_167524], **kwargs_167525)
    
    keyword_167527 = copy_call_result_167526
    # Getting the type of 'rhobeg' (line 251)
    rhobeg_167528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 69), 'rhobeg', False)
    keyword_167529 = rhobeg_167528
    # Getting the type of 'rhoend' (line 252)
    rhoend_167530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 41), 'rhoend', False)
    keyword_167531 = rhoend_167530
    # Getting the type of 'iprint' (line 252)
    iprint_167532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 56), 'iprint', False)
    keyword_167533 = iprint_167532
    # Getting the type of 'maxfun' (line 252)
    maxfun_167534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 71), 'maxfun', False)
    keyword_167535 = maxfun_167534
    # Getting the type of 'info' (line 253)
    info_167536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 40), 'info', False)
    keyword_167537 = info_167536
    kwargs_167538 = {'rhoend': keyword_167531, 'dinfo': keyword_167537, 'iprint': keyword_167533, 'm': keyword_167521, 'rhobeg': keyword_167529, 'x': keyword_167527, 'maxfun': keyword_167535}
    # Getting the type of '_cobyla' (line 251)
    _cobyla_167517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 17), '_cobyla', False)
    # Obtaining the member 'minimize' of a type (line 251)
    minimize_167518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 17), _cobyla_167517, 'minimize')
    # Calling minimize(args, kwargs) (line 251)
    minimize_call_result_167539 = invoke(stypy.reporting.localization.Localization(__file__, 251, 17), minimize_167518, *[calcfc_167519], **kwargs_167538)
    
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___167540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 4), minimize_call_result_167539, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_167541 = invoke(stypy.reporting.localization.Localization(__file__, 251, 4), getitem___167540, int_167516)
    
    # Assigning a type to the variable 'tuple_var_assignment_167258' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'tuple_var_assignment_167258', subscript_call_result_167541)
    
    # Assigning a Subscript to a Name (line 251):
    
    # Obtaining the type of the subscript
    int_167542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 4), 'int')
    
    # Call to minimize(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'calcfc' (line 251)
    calcfc_167545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 34), 'calcfc', False)
    # Processing the call keyword arguments (line 251)
    # Getting the type of 'm' (line 251)
    m_167546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 44), 'm', False)
    keyword_167547 = m_167546
    
    # Call to copy(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'x0' (line 251)
    x0_167550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 57), 'x0', False)
    # Processing the call keyword arguments (line 251)
    kwargs_167551 = {}
    # Getting the type of 'np' (line 251)
    np_167548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 49), 'np', False)
    # Obtaining the member 'copy' of a type (line 251)
    copy_167549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 49), np_167548, 'copy')
    # Calling copy(args, kwargs) (line 251)
    copy_call_result_167552 = invoke(stypy.reporting.localization.Localization(__file__, 251, 49), copy_167549, *[x0_167550], **kwargs_167551)
    
    keyword_167553 = copy_call_result_167552
    # Getting the type of 'rhobeg' (line 251)
    rhobeg_167554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 69), 'rhobeg', False)
    keyword_167555 = rhobeg_167554
    # Getting the type of 'rhoend' (line 252)
    rhoend_167556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 41), 'rhoend', False)
    keyword_167557 = rhoend_167556
    # Getting the type of 'iprint' (line 252)
    iprint_167558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 56), 'iprint', False)
    keyword_167559 = iprint_167558
    # Getting the type of 'maxfun' (line 252)
    maxfun_167560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 71), 'maxfun', False)
    keyword_167561 = maxfun_167560
    # Getting the type of 'info' (line 253)
    info_167562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 40), 'info', False)
    keyword_167563 = info_167562
    kwargs_167564 = {'rhoend': keyword_167557, 'dinfo': keyword_167563, 'iprint': keyword_167559, 'm': keyword_167547, 'rhobeg': keyword_167555, 'x': keyword_167553, 'maxfun': keyword_167561}
    # Getting the type of '_cobyla' (line 251)
    _cobyla_167543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 17), '_cobyla', False)
    # Obtaining the member 'minimize' of a type (line 251)
    minimize_167544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 17), _cobyla_167543, 'minimize')
    # Calling minimize(args, kwargs) (line 251)
    minimize_call_result_167565 = invoke(stypy.reporting.localization.Localization(__file__, 251, 17), minimize_167544, *[calcfc_167545], **kwargs_167564)
    
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___167566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 4), minimize_call_result_167565, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_167567 = invoke(stypy.reporting.localization.Localization(__file__, 251, 4), getitem___167566, int_167542)
    
    # Assigning a type to the variable 'tuple_var_assignment_167259' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'tuple_var_assignment_167259', subscript_call_result_167567)
    
    # Assigning a Name to a Name (line 251):
    # Getting the type of 'tuple_var_assignment_167258' (line 251)
    tuple_var_assignment_167258_167568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'tuple_var_assignment_167258')
    # Assigning a type to the variable 'xopt' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'xopt', tuple_var_assignment_167258_167568)
    
    # Assigning a Name to a Name (line 251):
    # Getting the type of 'tuple_var_assignment_167259' (line 251)
    tuple_var_assignment_167259_167569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'tuple_var_assignment_167259')
    # Assigning a type to the variable 'info' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 10), 'info', tuple_var_assignment_167259_167569)
    
    
    
    # Obtaining the type of the subscript
    int_167570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 12), 'int')
    # Getting the type of 'info' (line 255)
    info_167571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 7), 'info')
    # Obtaining the member '__getitem__' of a type (line 255)
    getitem___167572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 7), info_167571, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 255)
    subscript_call_result_167573 = invoke(stypy.reporting.localization.Localization(__file__, 255, 7), getitem___167572, int_167570)
    
    # Getting the type of 'catol' (line 255)
    catol_167574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 17), 'catol')
    # Applying the binary operator '>' (line 255)
    result_gt_167575 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 7), '>', subscript_call_result_167573, catol_167574)
    
    # Testing the type of an if condition (line 255)
    if_condition_167576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 4), result_gt_167575)
    # Assigning a type to the variable 'if_condition_167576' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'if_condition_167576', if_condition_167576)
    # SSA begins for if statement (line 255)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 257):
    
    # Assigning a Num to a Subscript (line 257):
    int_167577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 18), 'int')
    # Getting the type of 'info' (line 257)
    info_167578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'info')
    int_167579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 13), 'int')
    # Storing an element on a container (line 257)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), info_167578, (int_167579, int_167577))
    # SSA join for if statement (line 255)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to OptimizeResult(...): (line 259)
    # Processing the call keyword arguments (line 259)
    # Getting the type of 'xopt' (line 259)
    xopt_167581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'xopt', False)
    keyword_167582 = xopt_167581
    
    # Call to int(...): (line 260)
    # Processing the call arguments (line 260)
    
    # Obtaining the type of the subscript
    int_167584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 42), 'int')
    # Getting the type of 'info' (line 260)
    info_167585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 37), 'info', False)
    # Obtaining the member '__getitem__' of a type (line 260)
    getitem___167586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 37), info_167585, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 260)
    subscript_call_result_167587 = invoke(stypy.reporting.localization.Localization(__file__, 260, 37), getitem___167586, int_167584)
    
    # Processing the call keyword arguments (line 260)
    kwargs_167588 = {}
    # Getting the type of 'int' (line 260)
    int_167583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 33), 'int', False)
    # Calling int(args, kwargs) (line 260)
    int_call_result_167589 = invoke(stypy.reporting.localization.Localization(__file__, 260, 33), int_167583, *[subscript_call_result_167587], **kwargs_167588)
    
    keyword_167590 = int_call_result_167589
    
    
    # Obtaining the type of the subscript
    int_167591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 39), 'int')
    # Getting the type of 'info' (line 261)
    info_167592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 34), 'info', False)
    # Obtaining the member '__getitem__' of a type (line 261)
    getitem___167593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 34), info_167592, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 261)
    subscript_call_result_167594 = invoke(stypy.reporting.localization.Localization(__file__, 261, 34), getitem___167593, int_167591)
    
    int_167595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 45), 'int')
    # Applying the binary operator '==' (line 261)
    result_eq_167596 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 34), '==', subscript_call_result_167594, int_167595)
    
    keyword_167597 = result_eq_167596
    
    # Call to get(...): (line 262)
    # Processing the call arguments (line 262)
    
    # Obtaining the type of the subscript
    int_167608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 46), 'int')
    # Getting the type of 'info' (line 270)
    info_167609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 41), 'info', False)
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___167610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 41), info_167609, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_167611 = invoke(stypy.reporting.localization.Localization(__file__, 270, 41), getitem___167610, int_167608)
    
    str_167612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 50), 'str', 'Unknown exit status.')
    # Processing the call keyword arguments (line 262)
    kwargs_167613 = {}
    
    # Obtaining an instance of the builtin type 'dict' (line 262)
    dict_167598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 34), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 262)
    # Adding element type (key, value) (line 262)
    int_167599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 35), 'int')
    str_167600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 38), 'str', 'Optimization terminated successfully.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 34), dict_167598, (int_167599, str_167600))
    # Adding element type (key, value) (line 262)
    int_167601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 35), 'int')
    str_167602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 38), 'str', 'Maximum number of function evaluations has been exceeded.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 34), dict_167598, (int_167601, str_167602))
    # Adding element type (key, value) (line 262)
    int_167603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 35), 'int')
    str_167604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 38), 'str', 'Rounding errors are becoming damaging in COBYLA subroutine.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 34), dict_167598, (int_167603, str_167604))
    # Adding element type (key, value) (line 262)
    int_167605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 35), 'int')
    str_167606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 38), 'str', 'Did not converge to a solution satisfying the constraints. See `maxcv` for magnitude of violation.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 34), dict_167598, (int_167605, str_167606))
    
    # Obtaining the member 'get' of a type (line 262)
    get_167607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 34), dict_167598, 'get')
    # Calling get(args, kwargs) (line 262)
    get_call_result_167614 = invoke(stypy.reporting.localization.Localization(__file__, 262, 34), get_167607, *[subscript_call_result_167611, str_167612], **kwargs_167613)
    
    keyword_167615 = get_call_result_167614
    
    # Call to int(...): (line 271)
    # Processing the call arguments (line 271)
    
    # Obtaining the type of the subscript
    int_167617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 40), 'int')
    # Getting the type of 'info' (line 271)
    info_167618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 35), 'info', False)
    # Obtaining the member '__getitem__' of a type (line 271)
    getitem___167619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 35), info_167618, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 271)
    subscript_call_result_167620 = invoke(stypy.reporting.localization.Localization(__file__, 271, 35), getitem___167619, int_167617)
    
    # Processing the call keyword arguments (line 271)
    kwargs_167621 = {}
    # Getting the type of 'int' (line 271)
    int_167616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 31), 'int', False)
    # Calling int(args, kwargs) (line 271)
    int_call_result_167622 = invoke(stypy.reporting.localization.Localization(__file__, 271, 31), int_167616, *[subscript_call_result_167620], **kwargs_167621)
    
    keyword_167623 = int_call_result_167622
    
    # Obtaining the type of the subscript
    int_167624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 35), 'int')
    # Getting the type of 'info' (line 272)
    info_167625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 30), 'info', False)
    # Obtaining the member '__getitem__' of a type (line 272)
    getitem___167626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 30), info_167625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 272)
    subscript_call_result_167627 = invoke(stypy.reporting.localization.Localization(__file__, 272, 30), getitem___167626, int_167624)
    
    keyword_167628 = subscript_call_result_167627
    
    # Obtaining the type of the subscript
    int_167629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 37), 'int')
    # Getting the type of 'info' (line 273)
    info_167630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 32), 'info', False)
    # Obtaining the member '__getitem__' of a type (line 273)
    getitem___167631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 32), info_167630, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 273)
    subscript_call_result_167632 = invoke(stypy.reporting.localization.Localization(__file__, 273, 32), getitem___167631, int_167629)
    
    keyword_167633 = subscript_call_result_167632
    kwargs_167634 = {'status': keyword_167590, 'maxcv': keyword_167633, 'success': keyword_167597, 'nfev': keyword_167623, 'fun': keyword_167628, 'x': keyword_167582, 'message': keyword_167615}
    # Getting the type of 'OptimizeResult' (line 259)
    OptimizeResult_167580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 259)
    OptimizeResult_call_result_167635 = invoke(stypy.reporting.localization.Localization(__file__, 259, 11), OptimizeResult_167580, *[], **kwargs_167634)
    
    # Assigning a type to the variable 'stypy_return_type' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'stypy_return_type', OptimizeResult_call_result_167635)
    
    # ################# End of '_minimize_cobyla(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_minimize_cobyla' in the type store
    # Getting the type of 'stypy_return_type' (line 173)
    stypy_return_type_167636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_167636)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_minimize_cobyla'
    return stypy_return_type_167636

# Assigning a type to the variable '_minimize_cobyla' (line 173)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), '_minimize_cobyla', _minimize_cobyla)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
