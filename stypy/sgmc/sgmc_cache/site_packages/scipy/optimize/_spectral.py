
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Spectral Algorithm for Nonlinear Equations
3: '''
4: from __future__ import division, absolute_import, print_function
5: 
6: import collections
7: 
8: import numpy as np
9: from scipy.optimize import OptimizeResult
10: from scipy.optimize.optimize import _check_unknown_options
11: from .linesearch import _nonmonotone_line_search_cruz, _nonmonotone_line_search_cheng
12: 
13: class _NoConvergence(Exception):
14:     pass
15: 
16: 
17: def _root_df_sane(func, x0, args=(), ftol=1e-8, fatol=1e-300, maxfev=1000,
18:                   fnorm=None, callback=None, disp=False, M=10, eta_strategy=None,
19:                   sigma_eps=1e-10, sigma_0=1.0, line_search='cruz', **unknown_options):
20:     r'''
21:     Solve nonlinear equation with the DF-SANE method
22: 
23:     Options
24:     -------
25:     ftol : float, optional
26:         Relative norm tolerance.
27:     fatol : float, optional
28:         Absolute norm tolerance.
29:         Algorithm terminates when ``||func(x)|| < fatol + ftol ||func(x_0)||``.
30:     fnorm : callable, optional
31:         Norm to use in the convergence check. If None, 2-norm is used.
32:     maxfev : int, optional
33:         Maximum number of function evaluations.
34:     disp : bool, optional
35:         Whether to print convergence process to stdout.
36:     eta_strategy : callable, optional
37:         Choice of the ``eta_k`` parameter, which gives slack for growth
38:         of ``||F||**2``.  Called as ``eta_k = eta_strategy(k, x, F)`` with
39:         `k` the iteration number, `x` the current iterate and `F` the current
40:         residual. Should satisfy ``eta_k > 0`` and ``sum(eta, k=0..inf) < inf``.
41:         Default: ``||F||**2 / (1 + k)**2``.
42:     sigma_eps : float, optional
43:         The spectral coefficient is constrained to ``sigma_eps < sigma < 1/sigma_eps``.
44:         Default: 1e-10
45:     sigma_0 : float, optional
46:         Initial spectral coefficient.
47:         Default: 1.0
48:     M : int, optional
49:         Number of iterates to include in the nonmonotonic line search.
50:         Default: 10
51:     line_search : {'cruz', 'cheng'}
52:         Type of line search to employ. 'cruz' is the original one defined in
53:         [Martinez & Raydan. Math. Comp. 75, 1429 (2006)], 'cheng' is
54:         a modified search defined in [Cheng & Li. IMA J. Numer. Anal. 29, 814 (2009)].
55:         Default: 'cruz'
56: 
57:     References
58:     ----------
59:     .. [1] "Spectral residual method without gradient information for solving
60:            large-scale nonlinear systems of equations." W. La Cruz,
61:            J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).
62:     .. [2] W. La Cruz, Opt. Meth. Software, 29, 24 (2014).
63:     .. [3] W. Cheng, D.-H. Li. IMA J. Numer. Anal. **29**, 814 (2009).
64: 
65:     '''
66:     _check_unknown_options(unknown_options)
67: 
68:     if line_search not in ('cheng', 'cruz'):
69:         raise ValueError("Invalid value %r for 'line_search'" % (line_search,))
70: 
71:     nexp = 2
72: 
73:     if eta_strategy is None:
74:         # Different choice from [1], as their eta is not invariant
75:         # vs. scaling of F.
76:         def eta_strategy(k, x, F):
77:             # Obtain squared 2-norm of the initial residual from the outer scope
78:             return f_0 / (1 + k)**2
79: 
80:     if fnorm is None:
81:         def fnorm(F):
82:             # Obtain squared 2-norm of the current residual from the outer scope
83:             return f_k**(1.0/nexp)
84: 
85:     def fmerit(F):
86:         return np.linalg.norm(F)**nexp
87: 
88:     nfev = [0]
89:     f, x_k, x_shape, f_k, F_k, is_complex = _wrap_func(func, x0, fmerit, nfev, maxfev, args)
90: 
91:     k = 0
92:     f_0 = f_k
93:     sigma_k = sigma_0
94: 
95:     F_0_norm = fnorm(F_k)
96: 
97:     # For the 'cruz' line search
98:     prev_fs = collections.deque([f_k], M)
99: 
100:     # For the 'cheng' line search
101:     Q = 1.0
102:     C = f_0
103: 
104:     converged = False
105:     message = "too many function evaluations required"
106: 
107:     while True:
108:         F_k_norm = fnorm(F_k)
109: 
110:         if disp:
111:             print("iter %d: ||F|| = %g, sigma = %g" % (k, F_k_norm, sigma_k))
112: 
113:         if callback is not None:
114:             callback(x_k, F_k)
115: 
116:         if F_k_norm < ftol * F_0_norm + fatol:
117:             # Converged!
118:             message = "successful convergence"
119:             converged = True
120:             break
121: 
122:         # Control spectral parameter, from [2]
123:         if abs(sigma_k) > 1/sigma_eps:
124:             sigma_k = 1/sigma_eps * np.sign(sigma_k)
125:         elif abs(sigma_k) < sigma_eps:
126:             sigma_k = sigma_eps
127: 
128:         # Line search direction
129:         d = -sigma_k * F_k
130: 
131:         # Nonmonotone line search
132:         eta = eta_strategy(k, x_k, F_k)
133:         try:
134:             if line_search == 'cruz':
135:                 alpha, xp, fp, Fp = _nonmonotone_line_search_cruz(f, x_k, d, prev_fs, eta=eta)
136:             elif line_search == 'cheng':
137:                 alpha, xp, fp, Fp, C, Q = _nonmonotone_line_search_cheng(f, x_k, d, f_k, C, Q, eta=eta)
138:         except _NoConvergence:
139:             break
140: 
141:         # Update spectral parameter
142:         s_k = xp - x_k
143:         y_k = Fp - F_k
144:         sigma_k = np.vdot(s_k, s_k) / np.vdot(s_k, y_k)
145: 
146:         # Take step
147:         x_k = xp
148:         F_k = Fp
149:         f_k = fp
150: 
151:         # Store function value
152:         if line_search == 'cruz':
153:             prev_fs.append(fp)
154: 
155:         k += 1
156: 
157:     x = _wrap_result(x_k, is_complex, shape=x_shape)
158:     F = _wrap_result(F_k, is_complex)
159: 
160:     result = OptimizeResult(x=x, success=converged,
161:                             message=message,
162:                             fun=F, nfev=nfev[0], nit=k)
163: 
164:     return result
165: 
166: 
167: def _wrap_func(func, x0, fmerit, nfev_list, maxfev, args=()):
168:     '''
169:     Wrap a function and an initial value so that (i) complex values
170:     are wrapped to reals, and (ii) value for a merit function
171:     fmerit(x, f) is computed at the same time, (iii) iteration count
172:     is maintained and an exception is raised if it is exceeded.
173: 
174:     Parameters
175:     ----------
176:     func : callable
177:         Function to wrap
178:     x0 : ndarray
179:         Initial value
180:     fmerit : callable
181:         Merit function fmerit(f) for computing merit value from residual.
182:     nfev_list : list
183:         List to store number of evaluations in. Should be [0] in the beginning.
184:     maxfev : int
185:         Maximum number of evaluations before _NoConvergence is raised.
186:     args : tuple
187:         Extra arguments to func
188: 
189:     Returns
190:     -------
191:     wrap_func : callable
192:         Wrapped function, to be called as
193:         ``F, fp = wrap_func(x0)``
194:     x0_wrap : ndarray of float
195:         Wrapped initial value; raveled to 1D and complex
196:         values mapped to reals.
197:     x0_shape : tuple
198:         Shape of the initial value array
199:     f : float
200:         Merit function at F
201:     F : ndarray of float
202:         Residual at x0_wrap
203:     is_complex : bool
204:         Whether complex values were mapped to reals
205: 
206:     '''
207:     x0 = np.asarray(x0)
208:     x0_shape = x0.shape
209:     F = np.asarray(func(x0, *args)).ravel()
210:     is_complex = np.iscomplexobj(x0) or np.iscomplexobj(F)
211:     x0 = x0.ravel()
212: 
213:     nfev_list[0] = 1
214: 
215:     if is_complex:
216:         def wrap_func(x):
217:             if nfev_list[0] >= maxfev:
218:                 raise _NoConvergence()
219:             nfev_list[0] += 1
220:             z = _real2complex(x).reshape(x0_shape)
221:             v = np.asarray(func(z, *args)).ravel()
222:             F = _complex2real(v)
223:             f = fmerit(F)
224:             return f, F
225: 
226:         x0 = _complex2real(x0)
227:         F = _complex2real(F)
228:     else:
229:         def wrap_func(x):
230:             if nfev_list[0] >= maxfev:
231:                 raise _NoConvergence()
232:             nfev_list[0] += 1
233:             x = x.reshape(x0_shape)
234:             F = np.asarray(func(x, *args)).ravel()
235:             f = fmerit(F)
236:             return f, F
237: 
238:     return wrap_func, x0, x0_shape, fmerit(F), F, is_complex
239: 
240: 
241: def _wrap_result(result, is_complex, shape=None):
242:     '''
243:     Convert from real to complex and reshape result arrays.
244:     '''
245:     if is_complex:
246:         z = _real2complex(result)
247:     else:
248:         z = result
249:     if shape is not None:
250:         z = z.reshape(shape)
251:     return z
252: 
253: 
254: def _real2complex(x):
255:     return np.ascontiguousarray(x, dtype=float).view(np.complex128)
256: 
257: 
258: def _complex2real(z):
259:     return np.ascontiguousarray(z, dtype=complex).view(np.float64)
260: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_201787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nSpectral Algorithm for Nonlinear Equations\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import collections' statement (line 6)
import collections

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'collections', collections, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_201788 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_201788) is not StypyTypeError):

    if (import_201788 != 'pyd_module'):
        __import__(import_201788)
        sys_modules_201789 = sys.modules[import_201788]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_201789.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_201788)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.optimize import OptimizeResult' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_201790 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize')

if (type(import_201790) is not StypyTypeError):

    if (import_201790 != 'pyd_module'):
        __import__(import_201790)
        sys_modules_201791 = sys.modules[import_201790]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', sys_modules_201791.module_type_store, module_type_store, ['OptimizeResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_201791, sys_modules_201791.module_type_store, module_type_store)
    else:
        from scipy.optimize import OptimizeResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', None, module_type_store, ['OptimizeResult'], [OptimizeResult])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', import_201790)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.optimize.optimize import _check_unknown_options' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_201792 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize.optimize')

if (type(import_201792) is not StypyTypeError):

    if (import_201792 != 'pyd_module'):
        __import__(import_201792)
        sys_modules_201793 = sys.modules[import_201792]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize.optimize', sys_modules_201793.module_type_store, module_type_store, ['_check_unknown_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_201793, sys_modules_201793.module_type_store, module_type_store)
    else:
        from scipy.optimize.optimize import _check_unknown_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize.optimize', None, module_type_store, ['_check_unknown_options'], [_check_unknown_options])

else:
    # Assigning a type to the variable 'scipy.optimize.optimize' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize.optimize', import_201792)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.optimize.linesearch import _nonmonotone_line_search_cruz, _nonmonotone_line_search_cheng' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_201794 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize.linesearch')

if (type(import_201794) is not StypyTypeError):

    if (import_201794 != 'pyd_module'):
        __import__(import_201794)
        sys_modules_201795 = sys.modules[import_201794]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize.linesearch', sys_modules_201795.module_type_store, module_type_store, ['_nonmonotone_line_search_cruz', '_nonmonotone_line_search_cheng'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_201795, sys_modules_201795.module_type_store, module_type_store)
    else:
        from scipy.optimize.linesearch import _nonmonotone_line_search_cruz, _nonmonotone_line_search_cheng

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize.linesearch', None, module_type_store, ['_nonmonotone_line_search_cruz', '_nonmonotone_line_search_cheng'], [_nonmonotone_line_search_cruz, _nonmonotone_line_search_cheng])

else:
    # Assigning a type to the variable 'scipy.optimize.linesearch' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize.linesearch', import_201794)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

# Declaration of the '_NoConvergence' class
# Getting the type of 'Exception' (line 13)
Exception_201796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 21), 'Exception')

class _NoConvergence(Exception_201796, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 0, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_NoConvergence.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable '_NoConvergence' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '_NoConvergence', _NoConvergence)

@norecursion
def _root_df_sane(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 17)
    tuple_201797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 17)
    
    float_201798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 42), 'float')
    float_201799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 54), 'float')
    int_201800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 69), 'int')
    # Getting the type of 'None' (line 18)
    None_201801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'None')
    # Getting the type of 'None' (line 18)
    None_201802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 39), 'None')
    # Getting the type of 'False' (line 18)
    False_201803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 50), 'False')
    int_201804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 59), 'int')
    # Getting the type of 'None' (line 18)
    None_201805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 76), 'None')
    float_201806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 28), 'float')
    float_201807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 43), 'float')
    str_201808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 60), 'str', 'cruz')
    defaults = [tuple_201797, float_201798, float_201799, int_201800, None_201801, None_201802, False_201803, int_201804, None_201805, float_201806, float_201807, str_201808]
    # Create a new context for function '_root_df_sane'
    module_type_store = module_type_store.open_function_context('_root_df_sane', 17, 0, False)
    
    # Passed parameters checking function
    _root_df_sane.stypy_localization = localization
    _root_df_sane.stypy_type_of_self = None
    _root_df_sane.stypy_type_store = module_type_store
    _root_df_sane.stypy_function_name = '_root_df_sane'
    _root_df_sane.stypy_param_names_list = ['func', 'x0', 'args', 'ftol', 'fatol', 'maxfev', 'fnorm', 'callback', 'disp', 'M', 'eta_strategy', 'sigma_eps', 'sigma_0', 'line_search']
    _root_df_sane.stypy_varargs_param_name = None
    _root_df_sane.stypy_kwargs_param_name = 'unknown_options'
    _root_df_sane.stypy_call_defaults = defaults
    _root_df_sane.stypy_call_varargs = varargs
    _root_df_sane.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_root_df_sane', ['func', 'x0', 'args', 'ftol', 'fatol', 'maxfev', 'fnorm', 'callback', 'disp', 'M', 'eta_strategy', 'sigma_eps', 'sigma_0', 'line_search'], None, 'unknown_options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_root_df_sane', localization, ['func', 'x0', 'args', 'ftol', 'fatol', 'maxfev', 'fnorm', 'callback', 'disp', 'M', 'eta_strategy', 'sigma_eps', 'sigma_0', 'line_search'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_root_df_sane(...)' code ##################

    str_201809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, (-1)), 'str', '\n    Solve nonlinear equation with the DF-SANE method\n\n    Options\n    -------\n    ftol : float, optional\n        Relative norm tolerance.\n    fatol : float, optional\n        Absolute norm tolerance.\n        Algorithm terminates when ``||func(x)|| < fatol + ftol ||func(x_0)||``.\n    fnorm : callable, optional\n        Norm to use in the convergence check. If None, 2-norm is used.\n    maxfev : int, optional\n        Maximum number of function evaluations.\n    disp : bool, optional\n        Whether to print convergence process to stdout.\n    eta_strategy : callable, optional\n        Choice of the ``eta_k`` parameter, which gives slack for growth\n        of ``||F||**2``.  Called as ``eta_k = eta_strategy(k, x, F)`` with\n        `k` the iteration number, `x` the current iterate and `F` the current\n        residual. Should satisfy ``eta_k > 0`` and ``sum(eta, k=0..inf) < inf``.\n        Default: ``||F||**2 / (1 + k)**2``.\n    sigma_eps : float, optional\n        The spectral coefficient is constrained to ``sigma_eps < sigma < 1/sigma_eps``.\n        Default: 1e-10\n    sigma_0 : float, optional\n        Initial spectral coefficient.\n        Default: 1.0\n    M : int, optional\n        Number of iterates to include in the nonmonotonic line search.\n        Default: 10\n    line_search : {\'cruz\', \'cheng\'}\n        Type of line search to employ. \'cruz\' is the original one defined in\n        [Martinez & Raydan. Math. Comp. 75, 1429 (2006)], \'cheng\' is\n        a modified search defined in [Cheng & Li. IMA J. Numer. Anal. 29, 814 (2009)].\n        Default: \'cruz\'\n\n    References\n    ----------\n    .. [1] "Spectral residual method without gradient information for solving\n           large-scale nonlinear systems of equations." W. La Cruz,\n           J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).\n    .. [2] W. La Cruz, Opt. Meth. Software, 29, 24 (2014).\n    .. [3] W. Cheng, D.-H. Li. IMA J. Numer. Anal. **29**, 814 (2009).\n\n    ')
    
    # Call to _check_unknown_options(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'unknown_options' (line 66)
    unknown_options_201811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 27), 'unknown_options', False)
    # Processing the call keyword arguments (line 66)
    kwargs_201812 = {}
    # Getting the type of '_check_unknown_options' (line 66)
    _check_unknown_options_201810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), '_check_unknown_options', False)
    # Calling _check_unknown_options(args, kwargs) (line 66)
    _check_unknown_options_call_result_201813 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), _check_unknown_options_201810, *[unknown_options_201811], **kwargs_201812)
    
    
    
    # Getting the type of 'line_search' (line 68)
    line_search_201814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 7), 'line_search')
    
    # Obtaining an instance of the builtin type 'tuple' (line 68)
    tuple_201815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 68)
    # Adding element type (line 68)
    str_201816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 27), 'str', 'cheng')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 27), tuple_201815, str_201816)
    # Adding element type (line 68)
    str_201817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 36), 'str', 'cruz')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 27), tuple_201815, str_201817)
    
    # Applying the binary operator 'notin' (line 68)
    result_contains_201818 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 7), 'notin', line_search_201814, tuple_201815)
    
    # Testing the type of an if condition (line 68)
    if_condition_201819 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 4), result_contains_201818)
    # Assigning a type to the variable 'if_condition_201819' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'if_condition_201819', if_condition_201819)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 69)
    # Processing the call arguments (line 69)
    str_201821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 25), 'str', "Invalid value %r for 'line_search'")
    
    # Obtaining an instance of the builtin type 'tuple' (line 69)
    tuple_201822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 65), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 69)
    # Adding element type (line 69)
    # Getting the type of 'line_search' (line 69)
    line_search_201823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 65), 'line_search', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 65), tuple_201822, line_search_201823)
    
    # Applying the binary operator '%' (line 69)
    result_mod_201824 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 25), '%', str_201821, tuple_201822)
    
    # Processing the call keyword arguments (line 69)
    kwargs_201825 = {}
    # Getting the type of 'ValueError' (line 69)
    ValueError_201820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 69)
    ValueError_call_result_201826 = invoke(stypy.reporting.localization.Localization(__file__, 69, 14), ValueError_201820, *[result_mod_201824], **kwargs_201825)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 69, 8), ValueError_call_result_201826, 'raise parameter', BaseException)
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 71):
    
    # Assigning a Num to a Name (line 71):
    int_201827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 11), 'int')
    # Assigning a type to the variable 'nexp' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'nexp', int_201827)
    
    # Type idiom detected: calculating its left and rigth part (line 73)
    # Getting the type of 'eta_strategy' (line 73)
    eta_strategy_201828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 7), 'eta_strategy')
    # Getting the type of 'None' (line 73)
    None_201829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'None')
    
    (may_be_201830, more_types_in_union_201831) = may_be_none(eta_strategy_201828, None_201829)

    if may_be_201830:

        if more_types_in_union_201831:
            # Runtime conditional SSA (line 73)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def eta_strategy(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'eta_strategy'
            module_type_store = module_type_store.open_function_context('eta_strategy', 76, 8, False)
            
            # Passed parameters checking function
            eta_strategy.stypy_localization = localization
            eta_strategy.stypy_type_of_self = None
            eta_strategy.stypy_type_store = module_type_store
            eta_strategy.stypy_function_name = 'eta_strategy'
            eta_strategy.stypy_param_names_list = ['k', 'x', 'F']
            eta_strategy.stypy_varargs_param_name = None
            eta_strategy.stypy_kwargs_param_name = None
            eta_strategy.stypy_call_defaults = defaults
            eta_strategy.stypy_call_varargs = varargs
            eta_strategy.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'eta_strategy', ['k', 'x', 'F'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'eta_strategy', localization, ['k', 'x', 'F'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'eta_strategy(...)' code ##################

            # Getting the type of 'f_0' (line 78)
            f_0_201832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'f_0')
            int_201833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'int')
            # Getting the type of 'k' (line 78)
            k_201834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'k')
            # Applying the binary operator '+' (line 78)
            result_add_201835 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 26), '+', int_201833, k_201834)
            
            int_201836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 34), 'int')
            # Applying the binary operator '**' (line 78)
            result_pow_201837 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 25), '**', result_add_201835, int_201836)
            
            # Applying the binary operator 'div' (line 78)
            result_div_201838 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 19), 'div', f_0_201832, result_pow_201837)
            
            # Assigning a type to the variable 'stypy_return_type' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'stypy_return_type', result_div_201838)
            
            # ################# End of 'eta_strategy(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'eta_strategy' in the type store
            # Getting the type of 'stypy_return_type' (line 76)
            stypy_return_type_201839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_201839)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'eta_strategy'
            return stypy_return_type_201839

        # Assigning a type to the variable 'eta_strategy' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'eta_strategy', eta_strategy)

        if more_types_in_union_201831:
            # SSA join for if statement (line 73)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 80)
    # Getting the type of 'fnorm' (line 80)
    fnorm_201840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 7), 'fnorm')
    # Getting the type of 'None' (line 80)
    None_201841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'None')
    
    (may_be_201842, more_types_in_union_201843) = may_be_none(fnorm_201840, None_201841)

    if may_be_201842:

        if more_types_in_union_201843:
            # Runtime conditional SSA (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def fnorm(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fnorm'
            module_type_store = module_type_store.open_function_context('fnorm', 81, 8, False)
            
            # Passed parameters checking function
            fnorm.stypy_localization = localization
            fnorm.stypy_type_of_self = None
            fnorm.stypy_type_store = module_type_store
            fnorm.stypy_function_name = 'fnorm'
            fnorm.stypy_param_names_list = ['F']
            fnorm.stypy_varargs_param_name = None
            fnorm.stypy_kwargs_param_name = None
            fnorm.stypy_call_defaults = defaults
            fnorm.stypy_call_varargs = varargs
            fnorm.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fnorm', ['F'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fnorm', localization, ['F'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fnorm(...)' code ##################

            # Getting the type of 'f_k' (line 83)
            f_k_201844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'f_k')
            float_201845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 25), 'float')
            # Getting the type of 'nexp' (line 83)
            nexp_201846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 29), 'nexp')
            # Applying the binary operator 'div' (line 83)
            result_div_201847 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 25), 'div', float_201845, nexp_201846)
            
            # Applying the binary operator '**' (line 83)
            result_pow_201848 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 19), '**', f_k_201844, result_div_201847)
            
            # Assigning a type to the variable 'stypy_return_type' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'stypy_return_type', result_pow_201848)
            
            # ################# End of 'fnorm(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fnorm' in the type store
            # Getting the type of 'stypy_return_type' (line 81)
            stypy_return_type_201849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_201849)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fnorm'
            return stypy_return_type_201849

        # Assigning a type to the variable 'fnorm' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'fnorm', fnorm)

        if more_types_in_union_201843:
            # SSA join for if statement (line 80)
            module_type_store = module_type_store.join_ssa_context()


    

    @norecursion
    def fmerit(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fmerit'
        module_type_store = module_type_store.open_function_context('fmerit', 85, 4, False)
        
        # Passed parameters checking function
        fmerit.stypy_localization = localization
        fmerit.stypy_type_of_self = None
        fmerit.stypy_type_store = module_type_store
        fmerit.stypy_function_name = 'fmerit'
        fmerit.stypy_param_names_list = ['F']
        fmerit.stypy_varargs_param_name = None
        fmerit.stypy_kwargs_param_name = None
        fmerit.stypy_call_defaults = defaults
        fmerit.stypy_call_varargs = varargs
        fmerit.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fmerit', ['F'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fmerit', localization, ['F'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fmerit(...)' code ##################

        
        # Call to norm(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'F' (line 86)
        F_201853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 30), 'F', False)
        # Processing the call keyword arguments (line 86)
        kwargs_201854 = {}
        # Getting the type of 'np' (line 86)
        np_201850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'np', False)
        # Obtaining the member 'linalg' of a type (line 86)
        linalg_201851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), np_201850, 'linalg')
        # Obtaining the member 'norm' of a type (line 86)
        norm_201852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), linalg_201851, 'norm')
        # Calling norm(args, kwargs) (line 86)
        norm_call_result_201855 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), norm_201852, *[F_201853], **kwargs_201854)
        
        # Getting the type of 'nexp' (line 86)
        nexp_201856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 34), 'nexp')
        # Applying the binary operator '**' (line 86)
        result_pow_201857 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 15), '**', norm_call_result_201855, nexp_201856)
        
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type', result_pow_201857)
        
        # ################# End of 'fmerit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fmerit' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_201858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_201858)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fmerit'
        return stypy_return_type_201858

    # Assigning a type to the variable 'fmerit' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'fmerit', fmerit)
    
    # Assigning a List to a Name (line 88):
    
    # Assigning a List to a Name (line 88):
    
    # Obtaining an instance of the builtin type 'list' (line 88)
    list_201859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 88)
    # Adding element type (line 88)
    int_201860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 11), list_201859, int_201860)
    
    # Assigning a type to the variable 'nfev' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'nfev', list_201859)
    
    # Assigning a Call to a Tuple (line 89):
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_201861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'int')
    
    # Call to _wrap_func(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'func' (line 89)
    func_201863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 55), 'func', False)
    # Getting the type of 'x0' (line 89)
    x0_201864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 61), 'x0', False)
    # Getting the type of 'fmerit' (line 89)
    fmerit_201865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 65), 'fmerit', False)
    # Getting the type of 'nfev' (line 89)
    nfev_201866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 73), 'nfev', False)
    # Getting the type of 'maxfev' (line 89)
    maxfev_201867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 79), 'maxfev', False)
    # Getting the type of 'args' (line 89)
    args_201868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 87), 'args', False)
    # Processing the call keyword arguments (line 89)
    kwargs_201869 = {}
    # Getting the type of '_wrap_func' (line 89)
    _wrap_func_201862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 44), '_wrap_func', False)
    # Calling _wrap_func(args, kwargs) (line 89)
    _wrap_func_call_result_201870 = invoke(stypy.reporting.localization.Localization(__file__, 89, 44), _wrap_func_201862, *[func_201863, x0_201864, fmerit_201865, nfev_201866, maxfev_201867, args_201868], **kwargs_201869)
    
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___201871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), _wrap_func_call_result_201870, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_201872 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), getitem___201871, int_201861)
    
    # Assigning a type to the variable 'tuple_var_assignment_201771' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_201771', subscript_call_result_201872)
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_201873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'int')
    
    # Call to _wrap_func(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'func' (line 89)
    func_201875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 55), 'func', False)
    # Getting the type of 'x0' (line 89)
    x0_201876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 61), 'x0', False)
    # Getting the type of 'fmerit' (line 89)
    fmerit_201877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 65), 'fmerit', False)
    # Getting the type of 'nfev' (line 89)
    nfev_201878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 73), 'nfev', False)
    # Getting the type of 'maxfev' (line 89)
    maxfev_201879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 79), 'maxfev', False)
    # Getting the type of 'args' (line 89)
    args_201880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 87), 'args', False)
    # Processing the call keyword arguments (line 89)
    kwargs_201881 = {}
    # Getting the type of '_wrap_func' (line 89)
    _wrap_func_201874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 44), '_wrap_func', False)
    # Calling _wrap_func(args, kwargs) (line 89)
    _wrap_func_call_result_201882 = invoke(stypy.reporting.localization.Localization(__file__, 89, 44), _wrap_func_201874, *[func_201875, x0_201876, fmerit_201877, nfev_201878, maxfev_201879, args_201880], **kwargs_201881)
    
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___201883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), _wrap_func_call_result_201882, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_201884 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), getitem___201883, int_201873)
    
    # Assigning a type to the variable 'tuple_var_assignment_201772' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_201772', subscript_call_result_201884)
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_201885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'int')
    
    # Call to _wrap_func(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'func' (line 89)
    func_201887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 55), 'func', False)
    # Getting the type of 'x0' (line 89)
    x0_201888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 61), 'x0', False)
    # Getting the type of 'fmerit' (line 89)
    fmerit_201889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 65), 'fmerit', False)
    # Getting the type of 'nfev' (line 89)
    nfev_201890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 73), 'nfev', False)
    # Getting the type of 'maxfev' (line 89)
    maxfev_201891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 79), 'maxfev', False)
    # Getting the type of 'args' (line 89)
    args_201892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 87), 'args', False)
    # Processing the call keyword arguments (line 89)
    kwargs_201893 = {}
    # Getting the type of '_wrap_func' (line 89)
    _wrap_func_201886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 44), '_wrap_func', False)
    # Calling _wrap_func(args, kwargs) (line 89)
    _wrap_func_call_result_201894 = invoke(stypy.reporting.localization.Localization(__file__, 89, 44), _wrap_func_201886, *[func_201887, x0_201888, fmerit_201889, nfev_201890, maxfev_201891, args_201892], **kwargs_201893)
    
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___201895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), _wrap_func_call_result_201894, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_201896 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), getitem___201895, int_201885)
    
    # Assigning a type to the variable 'tuple_var_assignment_201773' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_201773', subscript_call_result_201896)
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_201897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'int')
    
    # Call to _wrap_func(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'func' (line 89)
    func_201899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 55), 'func', False)
    # Getting the type of 'x0' (line 89)
    x0_201900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 61), 'x0', False)
    # Getting the type of 'fmerit' (line 89)
    fmerit_201901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 65), 'fmerit', False)
    # Getting the type of 'nfev' (line 89)
    nfev_201902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 73), 'nfev', False)
    # Getting the type of 'maxfev' (line 89)
    maxfev_201903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 79), 'maxfev', False)
    # Getting the type of 'args' (line 89)
    args_201904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 87), 'args', False)
    # Processing the call keyword arguments (line 89)
    kwargs_201905 = {}
    # Getting the type of '_wrap_func' (line 89)
    _wrap_func_201898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 44), '_wrap_func', False)
    # Calling _wrap_func(args, kwargs) (line 89)
    _wrap_func_call_result_201906 = invoke(stypy.reporting.localization.Localization(__file__, 89, 44), _wrap_func_201898, *[func_201899, x0_201900, fmerit_201901, nfev_201902, maxfev_201903, args_201904], **kwargs_201905)
    
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___201907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), _wrap_func_call_result_201906, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_201908 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), getitem___201907, int_201897)
    
    # Assigning a type to the variable 'tuple_var_assignment_201774' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_201774', subscript_call_result_201908)
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_201909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'int')
    
    # Call to _wrap_func(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'func' (line 89)
    func_201911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 55), 'func', False)
    # Getting the type of 'x0' (line 89)
    x0_201912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 61), 'x0', False)
    # Getting the type of 'fmerit' (line 89)
    fmerit_201913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 65), 'fmerit', False)
    # Getting the type of 'nfev' (line 89)
    nfev_201914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 73), 'nfev', False)
    # Getting the type of 'maxfev' (line 89)
    maxfev_201915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 79), 'maxfev', False)
    # Getting the type of 'args' (line 89)
    args_201916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 87), 'args', False)
    # Processing the call keyword arguments (line 89)
    kwargs_201917 = {}
    # Getting the type of '_wrap_func' (line 89)
    _wrap_func_201910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 44), '_wrap_func', False)
    # Calling _wrap_func(args, kwargs) (line 89)
    _wrap_func_call_result_201918 = invoke(stypy.reporting.localization.Localization(__file__, 89, 44), _wrap_func_201910, *[func_201911, x0_201912, fmerit_201913, nfev_201914, maxfev_201915, args_201916], **kwargs_201917)
    
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___201919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), _wrap_func_call_result_201918, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_201920 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), getitem___201919, int_201909)
    
    # Assigning a type to the variable 'tuple_var_assignment_201775' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_201775', subscript_call_result_201920)
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_201921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'int')
    
    # Call to _wrap_func(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'func' (line 89)
    func_201923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 55), 'func', False)
    # Getting the type of 'x0' (line 89)
    x0_201924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 61), 'x0', False)
    # Getting the type of 'fmerit' (line 89)
    fmerit_201925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 65), 'fmerit', False)
    # Getting the type of 'nfev' (line 89)
    nfev_201926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 73), 'nfev', False)
    # Getting the type of 'maxfev' (line 89)
    maxfev_201927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 79), 'maxfev', False)
    # Getting the type of 'args' (line 89)
    args_201928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 87), 'args', False)
    # Processing the call keyword arguments (line 89)
    kwargs_201929 = {}
    # Getting the type of '_wrap_func' (line 89)
    _wrap_func_201922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 44), '_wrap_func', False)
    # Calling _wrap_func(args, kwargs) (line 89)
    _wrap_func_call_result_201930 = invoke(stypy.reporting.localization.Localization(__file__, 89, 44), _wrap_func_201922, *[func_201923, x0_201924, fmerit_201925, nfev_201926, maxfev_201927, args_201928], **kwargs_201929)
    
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___201931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), _wrap_func_call_result_201930, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_201932 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), getitem___201931, int_201921)
    
    # Assigning a type to the variable 'tuple_var_assignment_201776' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_201776', subscript_call_result_201932)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_201771' (line 89)
    tuple_var_assignment_201771_201933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_201771')
    # Assigning a type to the variable 'f' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'f', tuple_var_assignment_201771_201933)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_201772' (line 89)
    tuple_var_assignment_201772_201934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_201772')
    # Assigning a type to the variable 'x_k' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 7), 'x_k', tuple_var_assignment_201772_201934)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_201773' (line 89)
    tuple_var_assignment_201773_201935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_201773')
    # Assigning a type to the variable 'x_shape' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'x_shape', tuple_var_assignment_201773_201935)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_201774' (line 89)
    tuple_var_assignment_201774_201936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_201774')
    # Assigning a type to the variable 'f_k' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 21), 'f_k', tuple_var_assignment_201774_201936)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_201775' (line 89)
    tuple_var_assignment_201775_201937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_201775')
    # Assigning a type to the variable 'F_k' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'F_k', tuple_var_assignment_201775_201937)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_201776' (line 89)
    tuple_var_assignment_201776_201938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_201776')
    # Assigning a type to the variable 'is_complex' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 31), 'is_complex', tuple_var_assignment_201776_201938)
    
    # Assigning a Num to a Name (line 91):
    
    # Assigning a Num to a Name (line 91):
    int_201939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
    # Assigning a type to the variable 'k' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'k', int_201939)
    
    # Assigning a Name to a Name (line 92):
    
    # Assigning a Name to a Name (line 92):
    # Getting the type of 'f_k' (line 92)
    f_k_201940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 10), 'f_k')
    # Assigning a type to the variable 'f_0' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'f_0', f_k_201940)
    
    # Assigning a Name to a Name (line 93):
    
    # Assigning a Name to a Name (line 93):
    # Getting the type of 'sigma_0' (line 93)
    sigma_0_201941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 14), 'sigma_0')
    # Assigning a type to the variable 'sigma_k' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'sigma_k', sigma_0_201941)
    
    # Assigning a Call to a Name (line 95):
    
    # Assigning a Call to a Name (line 95):
    
    # Call to fnorm(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'F_k' (line 95)
    F_k_201943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'F_k', False)
    # Processing the call keyword arguments (line 95)
    kwargs_201944 = {}
    # Getting the type of 'fnorm' (line 95)
    fnorm_201942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'fnorm', False)
    # Calling fnorm(args, kwargs) (line 95)
    fnorm_call_result_201945 = invoke(stypy.reporting.localization.Localization(__file__, 95, 15), fnorm_201942, *[F_k_201943], **kwargs_201944)
    
    # Assigning a type to the variable 'F_0_norm' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'F_0_norm', fnorm_call_result_201945)
    
    # Assigning a Call to a Name (line 98):
    
    # Assigning a Call to a Name (line 98):
    
    # Call to deque(...): (line 98)
    # Processing the call arguments (line 98)
    
    # Obtaining an instance of the builtin type 'list' (line 98)
    list_201948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 98)
    # Adding element type (line 98)
    # Getting the type of 'f_k' (line 98)
    f_k_201949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'f_k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 32), list_201948, f_k_201949)
    
    # Getting the type of 'M' (line 98)
    M_201950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 39), 'M', False)
    # Processing the call keyword arguments (line 98)
    kwargs_201951 = {}
    # Getting the type of 'collections' (line 98)
    collections_201946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'collections', False)
    # Obtaining the member 'deque' of a type (line 98)
    deque_201947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 14), collections_201946, 'deque')
    # Calling deque(args, kwargs) (line 98)
    deque_call_result_201952 = invoke(stypy.reporting.localization.Localization(__file__, 98, 14), deque_201947, *[list_201948, M_201950], **kwargs_201951)
    
    # Assigning a type to the variable 'prev_fs' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'prev_fs', deque_call_result_201952)
    
    # Assigning a Num to a Name (line 101):
    
    # Assigning a Num to a Name (line 101):
    float_201953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'float')
    # Assigning a type to the variable 'Q' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'Q', float_201953)
    
    # Assigning a Name to a Name (line 102):
    
    # Assigning a Name to a Name (line 102):
    # Getting the type of 'f_0' (line 102)
    f_0_201954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'f_0')
    # Assigning a type to the variable 'C' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'C', f_0_201954)
    
    # Assigning a Name to a Name (line 104):
    
    # Assigning a Name to a Name (line 104):
    # Getting the type of 'False' (line 104)
    False_201955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'False')
    # Assigning a type to the variable 'converged' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'converged', False_201955)
    
    # Assigning a Str to a Name (line 105):
    
    # Assigning a Str to a Name (line 105):
    str_201956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 14), 'str', 'too many function evaluations required')
    # Assigning a type to the variable 'message' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'message', str_201956)
    
    # Getting the type of 'True' (line 107)
    True_201957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 10), 'True')
    # Testing the type of an if condition (line 107)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 4), True_201957)
    # SSA begins for while statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 108):
    
    # Assigning a Call to a Name (line 108):
    
    # Call to fnorm(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'F_k' (line 108)
    F_k_201959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'F_k', False)
    # Processing the call keyword arguments (line 108)
    kwargs_201960 = {}
    # Getting the type of 'fnorm' (line 108)
    fnorm_201958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'fnorm', False)
    # Calling fnorm(args, kwargs) (line 108)
    fnorm_call_result_201961 = invoke(stypy.reporting.localization.Localization(__file__, 108, 19), fnorm_201958, *[F_k_201959], **kwargs_201960)
    
    # Assigning a type to the variable 'F_k_norm' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'F_k_norm', fnorm_call_result_201961)
    
    # Getting the type of 'disp' (line 110)
    disp_201962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'disp')
    # Testing the type of an if condition (line 110)
    if_condition_201963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 8), disp_201962)
    # Assigning a type to the variable 'if_condition_201963' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'if_condition_201963', if_condition_201963)
    # SSA begins for if statement (line 110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 111)
    # Processing the call arguments (line 111)
    str_201965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 18), 'str', 'iter %d: ||F|| = %g, sigma = %g')
    
    # Obtaining an instance of the builtin type 'tuple' (line 111)
    tuple_201966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 55), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 111)
    # Adding element type (line 111)
    # Getting the type of 'k' (line 111)
    k_201967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 55), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 55), tuple_201966, k_201967)
    # Adding element type (line 111)
    # Getting the type of 'F_k_norm' (line 111)
    F_k_norm_201968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 58), 'F_k_norm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 55), tuple_201966, F_k_norm_201968)
    # Adding element type (line 111)
    # Getting the type of 'sigma_k' (line 111)
    sigma_k_201969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 68), 'sigma_k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 55), tuple_201966, sigma_k_201969)
    
    # Applying the binary operator '%' (line 111)
    result_mod_201970 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 18), '%', str_201965, tuple_201966)
    
    # Processing the call keyword arguments (line 111)
    kwargs_201971 = {}
    # Getting the type of 'print' (line 111)
    print_201964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'print', False)
    # Calling print(args, kwargs) (line 111)
    print_call_result_201972 = invoke(stypy.reporting.localization.Localization(__file__, 111, 12), print_201964, *[result_mod_201970], **kwargs_201971)
    
    # SSA join for if statement (line 110)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 113)
    # Getting the type of 'callback' (line 113)
    callback_201973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'callback')
    # Getting the type of 'None' (line 113)
    None_201974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'None')
    
    (may_be_201975, more_types_in_union_201976) = may_not_be_none(callback_201973, None_201974)

    if may_be_201975:

        if more_types_in_union_201976:
            # Runtime conditional SSA (line 113)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to callback(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'x_k' (line 114)
        x_k_201978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'x_k', False)
        # Getting the type of 'F_k' (line 114)
        F_k_201979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 26), 'F_k', False)
        # Processing the call keyword arguments (line 114)
        kwargs_201980 = {}
        # Getting the type of 'callback' (line 114)
        callback_201977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'callback', False)
        # Calling callback(args, kwargs) (line 114)
        callback_call_result_201981 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), callback_201977, *[x_k_201978, F_k_201979], **kwargs_201980)
        

        if more_types_in_union_201976:
            # SSA join for if statement (line 113)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'F_k_norm' (line 116)
    F_k_norm_201982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'F_k_norm')
    # Getting the type of 'ftol' (line 116)
    ftol_201983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'ftol')
    # Getting the type of 'F_0_norm' (line 116)
    F_0_norm_201984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 29), 'F_0_norm')
    # Applying the binary operator '*' (line 116)
    result_mul_201985 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 22), '*', ftol_201983, F_0_norm_201984)
    
    # Getting the type of 'fatol' (line 116)
    fatol_201986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 40), 'fatol')
    # Applying the binary operator '+' (line 116)
    result_add_201987 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 22), '+', result_mul_201985, fatol_201986)
    
    # Applying the binary operator '<' (line 116)
    result_lt_201988 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 11), '<', F_k_norm_201982, result_add_201987)
    
    # Testing the type of an if condition (line 116)
    if_condition_201989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 8), result_lt_201988)
    # Assigning a type to the variable 'if_condition_201989' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'if_condition_201989', if_condition_201989)
    # SSA begins for if statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 118):
    
    # Assigning a Str to a Name (line 118):
    str_201990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 22), 'str', 'successful convergence')
    # Assigning a type to the variable 'message' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'message', str_201990)
    
    # Assigning a Name to a Name (line 119):
    
    # Assigning a Name to a Name (line 119):
    # Getting the type of 'True' (line 119)
    True_201991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'True')
    # Assigning a type to the variable 'converged' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'converged', True_201991)
    # SSA join for if statement (line 116)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to abs(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'sigma_k' (line 123)
    sigma_k_201993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'sigma_k', False)
    # Processing the call keyword arguments (line 123)
    kwargs_201994 = {}
    # Getting the type of 'abs' (line 123)
    abs_201992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 123)
    abs_call_result_201995 = invoke(stypy.reporting.localization.Localization(__file__, 123, 11), abs_201992, *[sigma_k_201993], **kwargs_201994)
    
    int_201996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 26), 'int')
    # Getting the type of 'sigma_eps' (line 123)
    sigma_eps_201997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'sigma_eps')
    # Applying the binary operator 'div' (line 123)
    result_div_201998 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 26), 'div', int_201996, sigma_eps_201997)
    
    # Applying the binary operator '>' (line 123)
    result_gt_201999 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 11), '>', abs_call_result_201995, result_div_201998)
    
    # Testing the type of an if condition (line 123)
    if_condition_202000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 8), result_gt_201999)
    # Assigning a type to the variable 'if_condition_202000' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'if_condition_202000', if_condition_202000)
    # SSA begins for if statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 124):
    
    # Assigning a BinOp to a Name (line 124):
    int_202001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 22), 'int')
    # Getting the type of 'sigma_eps' (line 124)
    sigma_eps_202002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 24), 'sigma_eps')
    # Applying the binary operator 'div' (line 124)
    result_div_202003 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 22), 'div', int_202001, sigma_eps_202002)
    
    
    # Call to sign(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'sigma_k' (line 124)
    sigma_k_202006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 44), 'sigma_k', False)
    # Processing the call keyword arguments (line 124)
    kwargs_202007 = {}
    # Getting the type of 'np' (line 124)
    np_202004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 36), 'np', False)
    # Obtaining the member 'sign' of a type (line 124)
    sign_202005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 36), np_202004, 'sign')
    # Calling sign(args, kwargs) (line 124)
    sign_call_result_202008 = invoke(stypy.reporting.localization.Localization(__file__, 124, 36), sign_202005, *[sigma_k_202006], **kwargs_202007)
    
    # Applying the binary operator '*' (line 124)
    result_mul_202009 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 34), '*', result_div_202003, sign_call_result_202008)
    
    # Assigning a type to the variable 'sigma_k' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'sigma_k', result_mul_202009)
    # SSA branch for the else part of an if statement (line 123)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to abs(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'sigma_k' (line 125)
    sigma_k_202011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'sigma_k', False)
    # Processing the call keyword arguments (line 125)
    kwargs_202012 = {}
    # Getting the type of 'abs' (line 125)
    abs_202010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'abs', False)
    # Calling abs(args, kwargs) (line 125)
    abs_call_result_202013 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), abs_202010, *[sigma_k_202011], **kwargs_202012)
    
    # Getting the type of 'sigma_eps' (line 125)
    sigma_eps_202014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'sigma_eps')
    # Applying the binary operator '<' (line 125)
    result_lt_202015 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 13), '<', abs_call_result_202013, sigma_eps_202014)
    
    # Testing the type of an if condition (line 125)
    if_condition_202016 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 13), result_lt_202015)
    # Assigning a type to the variable 'if_condition_202016' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'if_condition_202016', if_condition_202016)
    # SSA begins for if statement (line 125)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 126):
    
    # Assigning a Name to a Name (line 126):
    # Getting the type of 'sigma_eps' (line 126)
    sigma_eps_202017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'sigma_eps')
    # Assigning a type to the variable 'sigma_k' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'sigma_k', sigma_eps_202017)
    # SSA join for if statement (line 125)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 123)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 129):
    
    # Assigning a BinOp to a Name (line 129):
    
    # Getting the type of 'sigma_k' (line 129)
    sigma_k_202018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 13), 'sigma_k')
    # Applying the 'usub' unary operator (line 129)
    result___neg___202019 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 12), 'usub', sigma_k_202018)
    
    # Getting the type of 'F_k' (line 129)
    F_k_202020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'F_k')
    # Applying the binary operator '*' (line 129)
    result_mul_202021 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 12), '*', result___neg___202019, F_k_202020)
    
    # Assigning a type to the variable 'd' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'd', result_mul_202021)
    
    # Assigning a Call to a Name (line 132):
    
    # Assigning a Call to a Name (line 132):
    
    # Call to eta_strategy(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'k' (line 132)
    k_202023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'k', False)
    # Getting the type of 'x_k' (line 132)
    x_k_202024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 30), 'x_k', False)
    # Getting the type of 'F_k' (line 132)
    F_k_202025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 35), 'F_k', False)
    # Processing the call keyword arguments (line 132)
    kwargs_202026 = {}
    # Getting the type of 'eta_strategy' (line 132)
    eta_strategy_202022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'eta_strategy', False)
    # Calling eta_strategy(args, kwargs) (line 132)
    eta_strategy_call_result_202027 = invoke(stypy.reporting.localization.Localization(__file__, 132, 14), eta_strategy_202022, *[k_202023, x_k_202024, F_k_202025], **kwargs_202026)
    
    # Assigning a type to the variable 'eta' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'eta', eta_strategy_call_result_202027)
    
    
    # SSA begins for try-except statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Getting the type of 'line_search' (line 134)
    line_search_202028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'line_search')
    str_202029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 30), 'str', 'cruz')
    # Applying the binary operator '==' (line 134)
    result_eq_202030 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), '==', line_search_202028, str_202029)
    
    # Testing the type of an if condition (line 134)
    if_condition_202031 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 12), result_eq_202030)
    # Assigning a type to the variable 'if_condition_202031' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'if_condition_202031', if_condition_202031)
    # SSA begins for if statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 135):
    
    # Assigning a Subscript to a Name (line 135):
    
    # Obtaining the type of the subscript
    int_202032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 16), 'int')
    
    # Call to _nonmonotone_line_search_cruz(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'f' (line 135)
    f_202034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 66), 'f', False)
    # Getting the type of 'x_k' (line 135)
    x_k_202035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 69), 'x_k', False)
    # Getting the type of 'd' (line 135)
    d_202036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 74), 'd', False)
    # Getting the type of 'prev_fs' (line 135)
    prev_fs_202037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 77), 'prev_fs', False)
    # Processing the call keyword arguments (line 135)
    # Getting the type of 'eta' (line 135)
    eta_202038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 90), 'eta', False)
    keyword_202039 = eta_202038
    kwargs_202040 = {'eta': keyword_202039}
    # Getting the type of '_nonmonotone_line_search_cruz' (line 135)
    _nonmonotone_line_search_cruz_202033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), '_nonmonotone_line_search_cruz', False)
    # Calling _nonmonotone_line_search_cruz(args, kwargs) (line 135)
    _nonmonotone_line_search_cruz_call_result_202041 = invoke(stypy.reporting.localization.Localization(__file__, 135, 36), _nonmonotone_line_search_cruz_202033, *[f_202034, x_k_202035, d_202036, prev_fs_202037], **kwargs_202040)
    
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___202042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), _nonmonotone_line_search_cruz_call_result_202041, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_202043 = invoke(stypy.reporting.localization.Localization(__file__, 135, 16), getitem___202042, int_202032)
    
    # Assigning a type to the variable 'tuple_var_assignment_201777' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'tuple_var_assignment_201777', subscript_call_result_202043)
    
    # Assigning a Subscript to a Name (line 135):
    
    # Obtaining the type of the subscript
    int_202044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 16), 'int')
    
    # Call to _nonmonotone_line_search_cruz(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'f' (line 135)
    f_202046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 66), 'f', False)
    # Getting the type of 'x_k' (line 135)
    x_k_202047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 69), 'x_k', False)
    # Getting the type of 'd' (line 135)
    d_202048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 74), 'd', False)
    # Getting the type of 'prev_fs' (line 135)
    prev_fs_202049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 77), 'prev_fs', False)
    # Processing the call keyword arguments (line 135)
    # Getting the type of 'eta' (line 135)
    eta_202050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 90), 'eta', False)
    keyword_202051 = eta_202050
    kwargs_202052 = {'eta': keyword_202051}
    # Getting the type of '_nonmonotone_line_search_cruz' (line 135)
    _nonmonotone_line_search_cruz_202045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), '_nonmonotone_line_search_cruz', False)
    # Calling _nonmonotone_line_search_cruz(args, kwargs) (line 135)
    _nonmonotone_line_search_cruz_call_result_202053 = invoke(stypy.reporting.localization.Localization(__file__, 135, 36), _nonmonotone_line_search_cruz_202045, *[f_202046, x_k_202047, d_202048, prev_fs_202049], **kwargs_202052)
    
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___202054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), _nonmonotone_line_search_cruz_call_result_202053, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_202055 = invoke(stypy.reporting.localization.Localization(__file__, 135, 16), getitem___202054, int_202044)
    
    # Assigning a type to the variable 'tuple_var_assignment_201778' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'tuple_var_assignment_201778', subscript_call_result_202055)
    
    # Assigning a Subscript to a Name (line 135):
    
    # Obtaining the type of the subscript
    int_202056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 16), 'int')
    
    # Call to _nonmonotone_line_search_cruz(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'f' (line 135)
    f_202058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 66), 'f', False)
    # Getting the type of 'x_k' (line 135)
    x_k_202059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 69), 'x_k', False)
    # Getting the type of 'd' (line 135)
    d_202060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 74), 'd', False)
    # Getting the type of 'prev_fs' (line 135)
    prev_fs_202061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 77), 'prev_fs', False)
    # Processing the call keyword arguments (line 135)
    # Getting the type of 'eta' (line 135)
    eta_202062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 90), 'eta', False)
    keyword_202063 = eta_202062
    kwargs_202064 = {'eta': keyword_202063}
    # Getting the type of '_nonmonotone_line_search_cruz' (line 135)
    _nonmonotone_line_search_cruz_202057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), '_nonmonotone_line_search_cruz', False)
    # Calling _nonmonotone_line_search_cruz(args, kwargs) (line 135)
    _nonmonotone_line_search_cruz_call_result_202065 = invoke(stypy.reporting.localization.Localization(__file__, 135, 36), _nonmonotone_line_search_cruz_202057, *[f_202058, x_k_202059, d_202060, prev_fs_202061], **kwargs_202064)
    
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___202066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), _nonmonotone_line_search_cruz_call_result_202065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_202067 = invoke(stypy.reporting.localization.Localization(__file__, 135, 16), getitem___202066, int_202056)
    
    # Assigning a type to the variable 'tuple_var_assignment_201779' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'tuple_var_assignment_201779', subscript_call_result_202067)
    
    # Assigning a Subscript to a Name (line 135):
    
    # Obtaining the type of the subscript
    int_202068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 16), 'int')
    
    # Call to _nonmonotone_line_search_cruz(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'f' (line 135)
    f_202070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 66), 'f', False)
    # Getting the type of 'x_k' (line 135)
    x_k_202071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 69), 'x_k', False)
    # Getting the type of 'd' (line 135)
    d_202072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 74), 'd', False)
    # Getting the type of 'prev_fs' (line 135)
    prev_fs_202073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 77), 'prev_fs', False)
    # Processing the call keyword arguments (line 135)
    # Getting the type of 'eta' (line 135)
    eta_202074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 90), 'eta', False)
    keyword_202075 = eta_202074
    kwargs_202076 = {'eta': keyword_202075}
    # Getting the type of '_nonmonotone_line_search_cruz' (line 135)
    _nonmonotone_line_search_cruz_202069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), '_nonmonotone_line_search_cruz', False)
    # Calling _nonmonotone_line_search_cruz(args, kwargs) (line 135)
    _nonmonotone_line_search_cruz_call_result_202077 = invoke(stypy.reporting.localization.Localization(__file__, 135, 36), _nonmonotone_line_search_cruz_202069, *[f_202070, x_k_202071, d_202072, prev_fs_202073], **kwargs_202076)
    
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___202078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), _nonmonotone_line_search_cruz_call_result_202077, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_202079 = invoke(stypy.reporting.localization.Localization(__file__, 135, 16), getitem___202078, int_202068)
    
    # Assigning a type to the variable 'tuple_var_assignment_201780' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'tuple_var_assignment_201780', subscript_call_result_202079)
    
    # Assigning a Name to a Name (line 135):
    # Getting the type of 'tuple_var_assignment_201777' (line 135)
    tuple_var_assignment_201777_202080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'tuple_var_assignment_201777')
    # Assigning a type to the variable 'alpha' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'alpha', tuple_var_assignment_201777_202080)
    
    # Assigning a Name to a Name (line 135):
    # Getting the type of 'tuple_var_assignment_201778' (line 135)
    tuple_var_assignment_201778_202081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'tuple_var_assignment_201778')
    # Assigning a type to the variable 'xp' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'xp', tuple_var_assignment_201778_202081)
    
    # Assigning a Name to a Name (line 135):
    # Getting the type of 'tuple_var_assignment_201779' (line 135)
    tuple_var_assignment_201779_202082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'tuple_var_assignment_201779')
    # Assigning a type to the variable 'fp' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 27), 'fp', tuple_var_assignment_201779_202082)
    
    # Assigning a Name to a Name (line 135):
    # Getting the type of 'tuple_var_assignment_201780' (line 135)
    tuple_var_assignment_201780_202083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'tuple_var_assignment_201780')
    # Assigning a type to the variable 'Fp' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'Fp', tuple_var_assignment_201780_202083)
    # SSA branch for the else part of an if statement (line 134)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'line_search' (line 136)
    line_search_202084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 17), 'line_search')
    str_202085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 32), 'str', 'cheng')
    # Applying the binary operator '==' (line 136)
    result_eq_202086 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 17), '==', line_search_202084, str_202085)
    
    # Testing the type of an if condition (line 136)
    if_condition_202087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 17), result_eq_202086)
    # Assigning a type to the variable 'if_condition_202087' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 17), 'if_condition_202087', if_condition_202087)
    # SSA begins for if statement (line 136)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 137):
    
    # Assigning a Subscript to a Name (line 137):
    
    # Obtaining the type of the subscript
    int_202088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 16), 'int')
    
    # Call to _nonmonotone_line_search_cheng(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'f' (line 137)
    f_202090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 73), 'f', False)
    # Getting the type of 'x_k' (line 137)
    x_k_202091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 76), 'x_k', False)
    # Getting the type of 'd' (line 137)
    d_202092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 81), 'd', False)
    # Getting the type of 'f_k' (line 137)
    f_k_202093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 84), 'f_k', False)
    # Getting the type of 'C' (line 137)
    C_202094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 89), 'C', False)
    # Getting the type of 'Q' (line 137)
    Q_202095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 92), 'Q', False)
    # Processing the call keyword arguments (line 137)
    # Getting the type of 'eta' (line 137)
    eta_202096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 99), 'eta', False)
    keyword_202097 = eta_202096
    kwargs_202098 = {'eta': keyword_202097}
    # Getting the type of '_nonmonotone_line_search_cheng' (line 137)
    _nonmonotone_line_search_cheng_202089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), '_nonmonotone_line_search_cheng', False)
    # Calling _nonmonotone_line_search_cheng(args, kwargs) (line 137)
    _nonmonotone_line_search_cheng_call_result_202099 = invoke(stypy.reporting.localization.Localization(__file__, 137, 42), _nonmonotone_line_search_cheng_202089, *[f_202090, x_k_202091, d_202092, f_k_202093, C_202094, Q_202095], **kwargs_202098)
    
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___202100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), _nonmonotone_line_search_cheng_call_result_202099, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_202101 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), getitem___202100, int_202088)
    
    # Assigning a type to the variable 'tuple_var_assignment_201781' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tuple_var_assignment_201781', subscript_call_result_202101)
    
    # Assigning a Subscript to a Name (line 137):
    
    # Obtaining the type of the subscript
    int_202102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 16), 'int')
    
    # Call to _nonmonotone_line_search_cheng(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'f' (line 137)
    f_202104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 73), 'f', False)
    # Getting the type of 'x_k' (line 137)
    x_k_202105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 76), 'x_k', False)
    # Getting the type of 'd' (line 137)
    d_202106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 81), 'd', False)
    # Getting the type of 'f_k' (line 137)
    f_k_202107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 84), 'f_k', False)
    # Getting the type of 'C' (line 137)
    C_202108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 89), 'C', False)
    # Getting the type of 'Q' (line 137)
    Q_202109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 92), 'Q', False)
    # Processing the call keyword arguments (line 137)
    # Getting the type of 'eta' (line 137)
    eta_202110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 99), 'eta', False)
    keyword_202111 = eta_202110
    kwargs_202112 = {'eta': keyword_202111}
    # Getting the type of '_nonmonotone_line_search_cheng' (line 137)
    _nonmonotone_line_search_cheng_202103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), '_nonmonotone_line_search_cheng', False)
    # Calling _nonmonotone_line_search_cheng(args, kwargs) (line 137)
    _nonmonotone_line_search_cheng_call_result_202113 = invoke(stypy.reporting.localization.Localization(__file__, 137, 42), _nonmonotone_line_search_cheng_202103, *[f_202104, x_k_202105, d_202106, f_k_202107, C_202108, Q_202109], **kwargs_202112)
    
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___202114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), _nonmonotone_line_search_cheng_call_result_202113, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_202115 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), getitem___202114, int_202102)
    
    # Assigning a type to the variable 'tuple_var_assignment_201782' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tuple_var_assignment_201782', subscript_call_result_202115)
    
    # Assigning a Subscript to a Name (line 137):
    
    # Obtaining the type of the subscript
    int_202116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 16), 'int')
    
    # Call to _nonmonotone_line_search_cheng(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'f' (line 137)
    f_202118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 73), 'f', False)
    # Getting the type of 'x_k' (line 137)
    x_k_202119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 76), 'x_k', False)
    # Getting the type of 'd' (line 137)
    d_202120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 81), 'd', False)
    # Getting the type of 'f_k' (line 137)
    f_k_202121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 84), 'f_k', False)
    # Getting the type of 'C' (line 137)
    C_202122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 89), 'C', False)
    # Getting the type of 'Q' (line 137)
    Q_202123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 92), 'Q', False)
    # Processing the call keyword arguments (line 137)
    # Getting the type of 'eta' (line 137)
    eta_202124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 99), 'eta', False)
    keyword_202125 = eta_202124
    kwargs_202126 = {'eta': keyword_202125}
    # Getting the type of '_nonmonotone_line_search_cheng' (line 137)
    _nonmonotone_line_search_cheng_202117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), '_nonmonotone_line_search_cheng', False)
    # Calling _nonmonotone_line_search_cheng(args, kwargs) (line 137)
    _nonmonotone_line_search_cheng_call_result_202127 = invoke(stypy.reporting.localization.Localization(__file__, 137, 42), _nonmonotone_line_search_cheng_202117, *[f_202118, x_k_202119, d_202120, f_k_202121, C_202122, Q_202123], **kwargs_202126)
    
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___202128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), _nonmonotone_line_search_cheng_call_result_202127, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_202129 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), getitem___202128, int_202116)
    
    # Assigning a type to the variable 'tuple_var_assignment_201783' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tuple_var_assignment_201783', subscript_call_result_202129)
    
    # Assigning a Subscript to a Name (line 137):
    
    # Obtaining the type of the subscript
    int_202130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 16), 'int')
    
    # Call to _nonmonotone_line_search_cheng(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'f' (line 137)
    f_202132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 73), 'f', False)
    # Getting the type of 'x_k' (line 137)
    x_k_202133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 76), 'x_k', False)
    # Getting the type of 'd' (line 137)
    d_202134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 81), 'd', False)
    # Getting the type of 'f_k' (line 137)
    f_k_202135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 84), 'f_k', False)
    # Getting the type of 'C' (line 137)
    C_202136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 89), 'C', False)
    # Getting the type of 'Q' (line 137)
    Q_202137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 92), 'Q', False)
    # Processing the call keyword arguments (line 137)
    # Getting the type of 'eta' (line 137)
    eta_202138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 99), 'eta', False)
    keyword_202139 = eta_202138
    kwargs_202140 = {'eta': keyword_202139}
    # Getting the type of '_nonmonotone_line_search_cheng' (line 137)
    _nonmonotone_line_search_cheng_202131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), '_nonmonotone_line_search_cheng', False)
    # Calling _nonmonotone_line_search_cheng(args, kwargs) (line 137)
    _nonmonotone_line_search_cheng_call_result_202141 = invoke(stypy.reporting.localization.Localization(__file__, 137, 42), _nonmonotone_line_search_cheng_202131, *[f_202132, x_k_202133, d_202134, f_k_202135, C_202136, Q_202137], **kwargs_202140)
    
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___202142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), _nonmonotone_line_search_cheng_call_result_202141, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_202143 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), getitem___202142, int_202130)
    
    # Assigning a type to the variable 'tuple_var_assignment_201784' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tuple_var_assignment_201784', subscript_call_result_202143)
    
    # Assigning a Subscript to a Name (line 137):
    
    # Obtaining the type of the subscript
    int_202144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 16), 'int')
    
    # Call to _nonmonotone_line_search_cheng(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'f' (line 137)
    f_202146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 73), 'f', False)
    # Getting the type of 'x_k' (line 137)
    x_k_202147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 76), 'x_k', False)
    # Getting the type of 'd' (line 137)
    d_202148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 81), 'd', False)
    # Getting the type of 'f_k' (line 137)
    f_k_202149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 84), 'f_k', False)
    # Getting the type of 'C' (line 137)
    C_202150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 89), 'C', False)
    # Getting the type of 'Q' (line 137)
    Q_202151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 92), 'Q', False)
    # Processing the call keyword arguments (line 137)
    # Getting the type of 'eta' (line 137)
    eta_202152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 99), 'eta', False)
    keyword_202153 = eta_202152
    kwargs_202154 = {'eta': keyword_202153}
    # Getting the type of '_nonmonotone_line_search_cheng' (line 137)
    _nonmonotone_line_search_cheng_202145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), '_nonmonotone_line_search_cheng', False)
    # Calling _nonmonotone_line_search_cheng(args, kwargs) (line 137)
    _nonmonotone_line_search_cheng_call_result_202155 = invoke(stypy.reporting.localization.Localization(__file__, 137, 42), _nonmonotone_line_search_cheng_202145, *[f_202146, x_k_202147, d_202148, f_k_202149, C_202150, Q_202151], **kwargs_202154)
    
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___202156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), _nonmonotone_line_search_cheng_call_result_202155, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_202157 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), getitem___202156, int_202144)
    
    # Assigning a type to the variable 'tuple_var_assignment_201785' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tuple_var_assignment_201785', subscript_call_result_202157)
    
    # Assigning a Subscript to a Name (line 137):
    
    # Obtaining the type of the subscript
    int_202158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 16), 'int')
    
    # Call to _nonmonotone_line_search_cheng(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'f' (line 137)
    f_202160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 73), 'f', False)
    # Getting the type of 'x_k' (line 137)
    x_k_202161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 76), 'x_k', False)
    # Getting the type of 'd' (line 137)
    d_202162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 81), 'd', False)
    # Getting the type of 'f_k' (line 137)
    f_k_202163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 84), 'f_k', False)
    # Getting the type of 'C' (line 137)
    C_202164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 89), 'C', False)
    # Getting the type of 'Q' (line 137)
    Q_202165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 92), 'Q', False)
    # Processing the call keyword arguments (line 137)
    # Getting the type of 'eta' (line 137)
    eta_202166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 99), 'eta', False)
    keyword_202167 = eta_202166
    kwargs_202168 = {'eta': keyword_202167}
    # Getting the type of '_nonmonotone_line_search_cheng' (line 137)
    _nonmonotone_line_search_cheng_202159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), '_nonmonotone_line_search_cheng', False)
    # Calling _nonmonotone_line_search_cheng(args, kwargs) (line 137)
    _nonmonotone_line_search_cheng_call_result_202169 = invoke(stypy.reporting.localization.Localization(__file__, 137, 42), _nonmonotone_line_search_cheng_202159, *[f_202160, x_k_202161, d_202162, f_k_202163, C_202164, Q_202165], **kwargs_202168)
    
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___202170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), _nonmonotone_line_search_cheng_call_result_202169, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_202171 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), getitem___202170, int_202158)
    
    # Assigning a type to the variable 'tuple_var_assignment_201786' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tuple_var_assignment_201786', subscript_call_result_202171)
    
    # Assigning a Name to a Name (line 137):
    # Getting the type of 'tuple_var_assignment_201781' (line 137)
    tuple_var_assignment_201781_202172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tuple_var_assignment_201781')
    # Assigning a type to the variable 'alpha' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'alpha', tuple_var_assignment_201781_202172)
    
    # Assigning a Name to a Name (line 137):
    # Getting the type of 'tuple_var_assignment_201782' (line 137)
    tuple_var_assignment_201782_202173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tuple_var_assignment_201782')
    # Assigning a type to the variable 'xp' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 'xp', tuple_var_assignment_201782_202173)
    
    # Assigning a Name to a Name (line 137):
    # Getting the type of 'tuple_var_assignment_201783' (line 137)
    tuple_var_assignment_201783_202174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tuple_var_assignment_201783')
    # Assigning a type to the variable 'fp' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'fp', tuple_var_assignment_201783_202174)
    
    # Assigning a Name to a Name (line 137):
    # Getting the type of 'tuple_var_assignment_201784' (line 137)
    tuple_var_assignment_201784_202175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tuple_var_assignment_201784')
    # Assigning a type to the variable 'Fp' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 31), 'Fp', tuple_var_assignment_201784_202175)
    
    # Assigning a Name to a Name (line 137):
    # Getting the type of 'tuple_var_assignment_201785' (line 137)
    tuple_var_assignment_201785_202176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tuple_var_assignment_201785')
    # Assigning a type to the variable 'C' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 35), 'C', tuple_var_assignment_201785_202176)
    
    # Assigning a Name to a Name (line 137):
    # Getting the type of 'tuple_var_assignment_201786' (line 137)
    tuple_var_assignment_201786_202177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tuple_var_assignment_201786')
    # Assigning a type to the variable 'Q' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 38), 'Q', tuple_var_assignment_201786_202177)
    # SSA join for if statement (line 136)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 134)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 133)
    # SSA branch for the except '_NoConvergence' branch of a try statement (line 133)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 133)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 142):
    
    # Assigning a BinOp to a Name (line 142):
    # Getting the type of 'xp' (line 142)
    xp_202178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 14), 'xp')
    # Getting the type of 'x_k' (line 142)
    x_k_202179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'x_k')
    # Applying the binary operator '-' (line 142)
    result_sub_202180 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 14), '-', xp_202178, x_k_202179)
    
    # Assigning a type to the variable 's_k' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 's_k', result_sub_202180)
    
    # Assigning a BinOp to a Name (line 143):
    
    # Assigning a BinOp to a Name (line 143):
    # Getting the type of 'Fp' (line 143)
    Fp_202181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 14), 'Fp')
    # Getting the type of 'F_k' (line 143)
    F_k_202182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 19), 'F_k')
    # Applying the binary operator '-' (line 143)
    result_sub_202183 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 14), '-', Fp_202181, F_k_202182)
    
    # Assigning a type to the variable 'y_k' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'y_k', result_sub_202183)
    
    # Assigning a BinOp to a Name (line 144):
    
    # Assigning a BinOp to a Name (line 144):
    
    # Call to vdot(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 's_k' (line 144)
    s_k_202186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 's_k', False)
    # Getting the type of 's_k' (line 144)
    s_k_202187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 31), 's_k', False)
    # Processing the call keyword arguments (line 144)
    kwargs_202188 = {}
    # Getting the type of 'np' (line 144)
    np_202184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 18), 'np', False)
    # Obtaining the member 'vdot' of a type (line 144)
    vdot_202185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 18), np_202184, 'vdot')
    # Calling vdot(args, kwargs) (line 144)
    vdot_call_result_202189 = invoke(stypy.reporting.localization.Localization(__file__, 144, 18), vdot_202185, *[s_k_202186, s_k_202187], **kwargs_202188)
    
    
    # Call to vdot(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 's_k' (line 144)
    s_k_202192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 46), 's_k', False)
    # Getting the type of 'y_k' (line 144)
    y_k_202193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 51), 'y_k', False)
    # Processing the call keyword arguments (line 144)
    kwargs_202194 = {}
    # Getting the type of 'np' (line 144)
    np_202190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 38), 'np', False)
    # Obtaining the member 'vdot' of a type (line 144)
    vdot_202191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 38), np_202190, 'vdot')
    # Calling vdot(args, kwargs) (line 144)
    vdot_call_result_202195 = invoke(stypy.reporting.localization.Localization(__file__, 144, 38), vdot_202191, *[s_k_202192, y_k_202193], **kwargs_202194)
    
    # Applying the binary operator 'div' (line 144)
    result_div_202196 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 18), 'div', vdot_call_result_202189, vdot_call_result_202195)
    
    # Assigning a type to the variable 'sigma_k' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'sigma_k', result_div_202196)
    
    # Assigning a Name to a Name (line 147):
    
    # Assigning a Name to a Name (line 147):
    # Getting the type of 'xp' (line 147)
    xp_202197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 'xp')
    # Assigning a type to the variable 'x_k' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'x_k', xp_202197)
    
    # Assigning a Name to a Name (line 148):
    
    # Assigning a Name to a Name (line 148):
    # Getting the type of 'Fp' (line 148)
    Fp_202198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 14), 'Fp')
    # Assigning a type to the variable 'F_k' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'F_k', Fp_202198)
    
    # Assigning a Name to a Name (line 149):
    
    # Assigning a Name to a Name (line 149):
    # Getting the type of 'fp' (line 149)
    fp_202199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 14), 'fp')
    # Assigning a type to the variable 'f_k' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'f_k', fp_202199)
    
    
    # Getting the type of 'line_search' (line 152)
    line_search_202200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'line_search')
    str_202201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 26), 'str', 'cruz')
    # Applying the binary operator '==' (line 152)
    result_eq_202202 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 11), '==', line_search_202200, str_202201)
    
    # Testing the type of an if condition (line 152)
    if_condition_202203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 8), result_eq_202202)
    # Assigning a type to the variable 'if_condition_202203' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'if_condition_202203', if_condition_202203)
    # SSA begins for if statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'fp' (line 153)
    fp_202206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 27), 'fp', False)
    # Processing the call keyword arguments (line 153)
    kwargs_202207 = {}
    # Getting the type of 'prev_fs' (line 153)
    prev_fs_202204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'prev_fs', False)
    # Obtaining the member 'append' of a type (line 153)
    append_202205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 12), prev_fs_202204, 'append')
    # Calling append(args, kwargs) (line 153)
    append_call_result_202208 = invoke(stypy.reporting.localization.Localization(__file__, 153, 12), append_202205, *[fp_202206], **kwargs_202207)
    
    # SSA join for if statement (line 152)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'k' (line 155)
    k_202209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'k')
    int_202210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 13), 'int')
    # Applying the binary operator '+=' (line 155)
    result_iadd_202211 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 8), '+=', k_202209, int_202210)
    # Assigning a type to the variable 'k' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'k', result_iadd_202211)
    
    # SSA join for while statement (line 107)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 157):
    
    # Assigning a Call to a Name (line 157):
    
    # Call to _wrap_result(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'x_k' (line 157)
    x_k_202213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'x_k', False)
    # Getting the type of 'is_complex' (line 157)
    is_complex_202214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 26), 'is_complex', False)
    # Processing the call keyword arguments (line 157)
    # Getting the type of 'x_shape' (line 157)
    x_shape_202215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 44), 'x_shape', False)
    keyword_202216 = x_shape_202215
    kwargs_202217 = {'shape': keyword_202216}
    # Getting the type of '_wrap_result' (line 157)
    _wrap_result_202212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), '_wrap_result', False)
    # Calling _wrap_result(args, kwargs) (line 157)
    _wrap_result_call_result_202218 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), _wrap_result_202212, *[x_k_202213, is_complex_202214], **kwargs_202217)
    
    # Assigning a type to the variable 'x' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'x', _wrap_result_call_result_202218)
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to _wrap_result(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'F_k' (line 158)
    F_k_202220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'F_k', False)
    # Getting the type of 'is_complex' (line 158)
    is_complex_202221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 26), 'is_complex', False)
    # Processing the call keyword arguments (line 158)
    kwargs_202222 = {}
    # Getting the type of '_wrap_result' (line 158)
    _wrap_result_202219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), '_wrap_result', False)
    # Calling _wrap_result(args, kwargs) (line 158)
    _wrap_result_call_result_202223 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), _wrap_result_202219, *[F_k_202220, is_complex_202221], **kwargs_202222)
    
    # Assigning a type to the variable 'F' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'F', _wrap_result_call_result_202223)
    
    # Assigning a Call to a Name (line 160):
    
    # Assigning a Call to a Name (line 160):
    
    # Call to OptimizeResult(...): (line 160)
    # Processing the call keyword arguments (line 160)
    # Getting the type of 'x' (line 160)
    x_202225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 30), 'x', False)
    keyword_202226 = x_202225
    # Getting the type of 'converged' (line 160)
    converged_202227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 41), 'converged', False)
    keyword_202228 = converged_202227
    # Getting the type of 'message' (line 161)
    message_202229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 36), 'message', False)
    keyword_202230 = message_202229
    # Getting the type of 'F' (line 162)
    F_202231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'F', False)
    keyword_202232 = F_202231
    
    # Obtaining the type of the subscript
    int_202233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 45), 'int')
    # Getting the type of 'nfev' (line 162)
    nfev_202234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 40), 'nfev', False)
    # Obtaining the member '__getitem__' of a type (line 162)
    getitem___202235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 40), nfev_202234, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 162)
    subscript_call_result_202236 = invoke(stypy.reporting.localization.Localization(__file__, 162, 40), getitem___202235, int_202233)
    
    keyword_202237 = subscript_call_result_202236
    # Getting the type of 'k' (line 162)
    k_202238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 53), 'k', False)
    keyword_202239 = k_202238
    kwargs_202240 = {'success': keyword_202228, 'nfev': keyword_202237, 'fun': keyword_202232, 'x': keyword_202226, 'message': keyword_202230, 'nit': keyword_202239}
    # Getting the type of 'OptimizeResult' (line 160)
    OptimizeResult_202224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 13), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 160)
    OptimizeResult_call_result_202241 = invoke(stypy.reporting.localization.Localization(__file__, 160, 13), OptimizeResult_202224, *[], **kwargs_202240)
    
    # Assigning a type to the variable 'result' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'result', OptimizeResult_call_result_202241)
    # Getting the type of 'result' (line 164)
    result_202242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type', result_202242)
    
    # ################# End of '_root_df_sane(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_root_df_sane' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_202243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_202243)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_root_df_sane'
    return stypy_return_type_202243

# Assigning a type to the variable '_root_df_sane' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '_root_df_sane', _root_df_sane)

@norecursion
def _wrap_func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 167)
    tuple_202244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 167)
    
    defaults = [tuple_202244]
    # Create a new context for function '_wrap_func'
    module_type_store = module_type_store.open_function_context('_wrap_func', 167, 0, False)
    
    # Passed parameters checking function
    _wrap_func.stypy_localization = localization
    _wrap_func.stypy_type_of_self = None
    _wrap_func.stypy_type_store = module_type_store
    _wrap_func.stypy_function_name = '_wrap_func'
    _wrap_func.stypy_param_names_list = ['func', 'x0', 'fmerit', 'nfev_list', 'maxfev', 'args']
    _wrap_func.stypy_varargs_param_name = None
    _wrap_func.stypy_kwargs_param_name = None
    _wrap_func.stypy_call_defaults = defaults
    _wrap_func.stypy_call_varargs = varargs
    _wrap_func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_wrap_func', ['func', 'x0', 'fmerit', 'nfev_list', 'maxfev', 'args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_wrap_func', localization, ['func', 'x0', 'fmerit', 'nfev_list', 'maxfev', 'args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_wrap_func(...)' code ##################

    str_202245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, (-1)), 'str', '\n    Wrap a function and an initial value so that (i) complex values\n    are wrapped to reals, and (ii) value for a merit function\n    fmerit(x, f) is computed at the same time, (iii) iteration count\n    is maintained and an exception is raised if it is exceeded.\n\n    Parameters\n    ----------\n    func : callable\n        Function to wrap\n    x0 : ndarray\n        Initial value\n    fmerit : callable\n        Merit function fmerit(f) for computing merit value from residual.\n    nfev_list : list\n        List to store number of evaluations in. Should be [0] in the beginning.\n    maxfev : int\n        Maximum number of evaluations before _NoConvergence is raised.\n    args : tuple\n        Extra arguments to func\n\n    Returns\n    -------\n    wrap_func : callable\n        Wrapped function, to be called as\n        ``F, fp = wrap_func(x0)``\n    x0_wrap : ndarray of float\n        Wrapped initial value; raveled to 1D and complex\n        values mapped to reals.\n    x0_shape : tuple\n        Shape of the initial value array\n    f : float\n        Merit function at F\n    F : ndarray of float\n        Residual at x0_wrap\n    is_complex : bool\n        Whether complex values were mapped to reals\n\n    ')
    
    # Assigning a Call to a Name (line 207):
    
    # Assigning a Call to a Name (line 207):
    
    # Call to asarray(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'x0' (line 207)
    x0_202248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 20), 'x0', False)
    # Processing the call keyword arguments (line 207)
    kwargs_202249 = {}
    # Getting the type of 'np' (line 207)
    np_202246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 9), 'np', False)
    # Obtaining the member 'asarray' of a type (line 207)
    asarray_202247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 9), np_202246, 'asarray')
    # Calling asarray(args, kwargs) (line 207)
    asarray_call_result_202250 = invoke(stypy.reporting.localization.Localization(__file__, 207, 9), asarray_202247, *[x0_202248], **kwargs_202249)
    
    # Assigning a type to the variable 'x0' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'x0', asarray_call_result_202250)
    
    # Assigning a Attribute to a Name (line 208):
    
    # Assigning a Attribute to a Name (line 208):
    # Getting the type of 'x0' (line 208)
    x0_202251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'x0')
    # Obtaining the member 'shape' of a type (line 208)
    shape_202252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 15), x0_202251, 'shape')
    # Assigning a type to the variable 'x0_shape' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'x0_shape', shape_202252)
    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to ravel(...): (line 209)
    # Processing the call keyword arguments (line 209)
    kwargs_202263 = {}
    
    # Call to asarray(...): (line 209)
    # Processing the call arguments (line 209)
    
    # Call to func(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'x0' (line 209)
    x0_202256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 24), 'x0', False)
    # Getting the type of 'args' (line 209)
    args_202257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 29), 'args', False)
    # Processing the call keyword arguments (line 209)
    kwargs_202258 = {}
    # Getting the type of 'func' (line 209)
    func_202255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 19), 'func', False)
    # Calling func(args, kwargs) (line 209)
    func_call_result_202259 = invoke(stypy.reporting.localization.Localization(__file__, 209, 19), func_202255, *[x0_202256, args_202257], **kwargs_202258)
    
    # Processing the call keyword arguments (line 209)
    kwargs_202260 = {}
    # Getting the type of 'np' (line 209)
    np_202253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 209)
    asarray_202254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), np_202253, 'asarray')
    # Calling asarray(args, kwargs) (line 209)
    asarray_call_result_202261 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), asarray_202254, *[func_call_result_202259], **kwargs_202260)
    
    # Obtaining the member 'ravel' of a type (line 209)
    ravel_202262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), asarray_call_result_202261, 'ravel')
    # Calling ravel(args, kwargs) (line 209)
    ravel_call_result_202264 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), ravel_202262, *[], **kwargs_202263)
    
    # Assigning a type to the variable 'F' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'F', ravel_call_result_202264)
    
    # Assigning a BoolOp to a Name (line 210):
    
    # Assigning a BoolOp to a Name (line 210):
    
    # Evaluating a boolean operation
    
    # Call to iscomplexobj(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'x0' (line 210)
    x0_202267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 33), 'x0', False)
    # Processing the call keyword arguments (line 210)
    kwargs_202268 = {}
    # Getting the type of 'np' (line 210)
    np_202265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 17), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 210)
    iscomplexobj_202266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 17), np_202265, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 210)
    iscomplexobj_call_result_202269 = invoke(stypy.reporting.localization.Localization(__file__, 210, 17), iscomplexobj_202266, *[x0_202267], **kwargs_202268)
    
    
    # Call to iscomplexobj(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'F' (line 210)
    F_202272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 56), 'F', False)
    # Processing the call keyword arguments (line 210)
    kwargs_202273 = {}
    # Getting the type of 'np' (line 210)
    np_202270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 40), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 210)
    iscomplexobj_202271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 40), np_202270, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 210)
    iscomplexobj_call_result_202274 = invoke(stypy.reporting.localization.Localization(__file__, 210, 40), iscomplexobj_202271, *[F_202272], **kwargs_202273)
    
    # Applying the binary operator 'or' (line 210)
    result_or_keyword_202275 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 17), 'or', iscomplexobj_call_result_202269, iscomplexobj_call_result_202274)
    
    # Assigning a type to the variable 'is_complex' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'is_complex', result_or_keyword_202275)
    
    # Assigning a Call to a Name (line 211):
    
    # Assigning a Call to a Name (line 211):
    
    # Call to ravel(...): (line 211)
    # Processing the call keyword arguments (line 211)
    kwargs_202278 = {}
    # Getting the type of 'x0' (line 211)
    x0_202276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 9), 'x0', False)
    # Obtaining the member 'ravel' of a type (line 211)
    ravel_202277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 9), x0_202276, 'ravel')
    # Calling ravel(args, kwargs) (line 211)
    ravel_call_result_202279 = invoke(stypy.reporting.localization.Localization(__file__, 211, 9), ravel_202277, *[], **kwargs_202278)
    
    # Assigning a type to the variable 'x0' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'x0', ravel_call_result_202279)
    
    # Assigning a Num to a Subscript (line 213):
    
    # Assigning a Num to a Subscript (line 213):
    int_202280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 19), 'int')
    # Getting the type of 'nfev_list' (line 213)
    nfev_list_202281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'nfev_list')
    int_202282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 14), 'int')
    # Storing an element on a container (line 213)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 4), nfev_list_202281, (int_202282, int_202280))
    
    # Getting the type of 'is_complex' (line 215)
    is_complex_202283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 7), 'is_complex')
    # Testing the type of an if condition (line 215)
    if_condition_202284 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 4), is_complex_202283)
    # Assigning a type to the variable 'if_condition_202284' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'if_condition_202284', if_condition_202284)
    # SSA begins for if statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

    @norecursion
    def wrap_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wrap_func'
        module_type_store = module_type_store.open_function_context('wrap_func', 216, 8, False)
        
        # Passed parameters checking function
        wrap_func.stypy_localization = localization
        wrap_func.stypy_type_of_self = None
        wrap_func.stypy_type_store = module_type_store
        wrap_func.stypy_function_name = 'wrap_func'
        wrap_func.stypy_param_names_list = ['x']
        wrap_func.stypy_varargs_param_name = None
        wrap_func.stypy_kwargs_param_name = None
        wrap_func.stypy_call_defaults = defaults
        wrap_func.stypy_call_varargs = varargs
        wrap_func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'wrap_func', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wrap_func', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wrap_func(...)' code ##################

        
        
        
        # Obtaining the type of the subscript
        int_202285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 25), 'int')
        # Getting the type of 'nfev_list' (line 217)
        nfev_list_202286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'nfev_list')
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___202287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 15), nfev_list_202286, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_202288 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), getitem___202287, int_202285)
        
        # Getting the type of 'maxfev' (line 217)
        maxfev_202289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 31), 'maxfev')
        # Applying the binary operator '>=' (line 217)
        result_ge_202290 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 15), '>=', subscript_call_result_202288, maxfev_202289)
        
        # Testing the type of an if condition (line 217)
        if_condition_202291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 12), result_ge_202290)
        # Assigning a type to the variable 'if_condition_202291' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'if_condition_202291', if_condition_202291)
        # SSA begins for if statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _NoConvergence(...): (line 218)
        # Processing the call keyword arguments (line 218)
        kwargs_202293 = {}
        # Getting the type of '_NoConvergence' (line 218)
        _NoConvergence_202292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), '_NoConvergence', False)
        # Calling _NoConvergence(args, kwargs) (line 218)
        _NoConvergence_call_result_202294 = invoke(stypy.reporting.localization.Localization(__file__, 218, 22), _NoConvergence_202292, *[], **kwargs_202293)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 218, 16), _NoConvergence_call_result_202294, 'raise parameter', BaseException)
        # SSA join for if statement (line 217)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'nfev_list' (line 219)
        nfev_list_202295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'nfev_list')
        
        # Obtaining the type of the subscript
        int_202296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 22), 'int')
        # Getting the type of 'nfev_list' (line 219)
        nfev_list_202297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'nfev_list')
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___202298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 12), nfev_list_202297, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_202299 = invoke(stypy.reporting.localization.Localization(__file__, 219, 12), getitem___202298, int_202296)
        
        int_202300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 28), 'int')
        # Applying the binary operator '+=' (line 219)
        result_iadd_202301 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 12), '+=', subscript_call_result_202299, int_202300)
        # Getting the type of 'nfev_list' (line 219)
        nfev_list_202302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'nfev_list')
        int_202303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 22), 'int')
        # Storing an element on a container (line 219)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 12), nfev_list_202302, (int_202303, result_iadd_202301))
        
        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to reshape(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'x0_shape' (line 220)
        x0_shape_202309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 41), 'x0_shape', False)
        # Processing the call keyword arguments (line 220)
        kwargs_202310 = {}
        
        # Call to _real2complex(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'x' (line 220)
        x_202305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 30), 'x', False)
        # Processing the call keyword arguments (line 220)
        kwargs_202306 = {}
        # Getting the type of '_real2complex' (line 220)
        _real2complex_202304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), '_real2complex', False)
        # Calling _real2complex(args, kwargs) (line 220)
        _real2complex_call_result_202307 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), _real2complex_202304, *[x_202305], **kwargs_202306)
        
        # Obtaining the member 'reshape' of a type (line 220)
        reshape_202308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 16), _real2complex_call_result_202307, 'reshape')
        # Calling reshape(args, kwargs) (line 220)
        reshape_call_result_202311 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), reshape_202308, *[x0_shape_202309], **kwargs_202310)
        
        # Assigning a type to the variable 'z' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'z', reshape_call_result_202311)
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to ravel(...): (line 221)
        # Processing the call keyword arguments (line 221)
        kwargs_202322 = {}
        
        # Call to asarray(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Call to func(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'z' (line 221)
        z_202315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 32), 'z', False)
        # Getting the type of 'args' (line 221)
        args_202316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 36), 'args', False)
        # Processing the call keyword arguments (line 221)
        kwargs_202317 = {}
        # Getting the type of 'func' (line 221)
        func_202314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 27), 'func', False)
        # Calling func(args, kwargs) (line 221)
        func_call_result_202318 = invoke(stypy.reporting.localization.Localization(__file__, 221, 27), func_202314, *[z_202315, args_202316], **kwargs_202317)
        
        # Processing the call keyword arguments (line 221)
        kwargs_202319 = {}
        # Getting the type of 'np' (line 221)
        np_202312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'np', False)
        # Obtaining the member 'asarray' of a type (line 221)
        asarray_202313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), np_202312, 'asarray')
        # Calling asarray(args, kwargs) (line 221)
        asarray_call_result_202320 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), asarray_202313, *[func_call_result_202318], **kwargs_202319)
        
        # Obtaining the member 'ravel' of a type (line 221)
        ravel_202321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), asarray_call_result_202320, 'ravel')
        # Calling ravel(args, kwargs) (line 221)
        ravel_call_result_202323 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), ravel_202321, *[], **kwargs_202322)
        
        # Assigning a type to the variable 'v' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'v', ravel_call_result_202323)
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to _complex2real(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'v' (line 222)
        v_202325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 30), 'v', False)
        # Processing the call keyword arguments (line 222)
        kwargs_202326 = {}
        # Getting the type of '_complex2real' (line 222)
        _complex2real_202324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), '_complex2real', False)
        # Calling _complex2real(args, kwargs) (line 222)
        _complex2real_call_result_202327 = invoke(stypy.reporting.localization.Localization(__file__, 222, 16), _complex2real_202324, *[v_202325], **kwargs_202326)
        
        # Assigning a type to the variable 'F' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'F', _complex2real_call_result_202327)
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to fmerit(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'F' (line 223)
        F_202329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'F', False)
        # Processing the call keyword arguments (line 223)
        kwargs_202330 = {}
        # Getting the type of 'fmerit' (line 223)
        fmerit_202328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'fmerit', False)
        # Calling fmerit(args, kwargs) (line 223)
        fmerit_call_result_202331 = invoke(stypy.reporting.localization.Localization(__file__, 223, 16), fmerit_202328, *[F_202329], **kwargs_202330)
        
        # Assigning a type to the variable 'f' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'f', fmerit_call_result_202331)
        
        # Obtaining an instance of the builtin type 'tuple' (line 224)
        tuple_202332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 224)
        # Adding element type (line 224)
        # Getting the type of 'f' (line 224)
        f_202333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 19), tuple_202332, f_202333)
        # Adding element type (line 224)
        # Getting the type of 'F' (line 224)
        F_202334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 22), 'F')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 19), tuple_202332, F_202334)
        
        # Assigning a type to the variable 'stypy_return_type' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'stypy_return_type', tuple_202332)
        
        # ################# End of 'wrap_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wrap_func' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_202335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_202335)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wrap_func'
        return stypy_return_type_202335

    # Assigning a type to the variable 'wrap_func' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'wrap_func', wrap_func)
    
    # Assigning a Call to a Name (line 226):
    
    # Assigning a Call to a Name (line 226):
    
    # Call to _complex2real(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'x0' (line 226)
    x0_202337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'x0', False)
    # Processing the call keyword arguments (line 226)
    kwargs_202338 = {}
    # Getting the type of '_complex2real' (line 226)
    _complex2real_202336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 13), '_complex2real', False)
    # Calling _complex2real(args, kwargs) (line 226)
    _complex2real_call_result_202339 = invoke(stypy.reporting.localization.Localization(__file__, 226, 13), _complex2real_202336, *[x0_202337], **kwargs_202338)
    
    # Assigning a type to the variable 'x0' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'x0', _complex2real_call_result_202339)
    
    # Assigning a Call to a Name (line 227):
    
    # Assigning a Call to a Name (line 227):
    
    # Call to _complex2real(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'F' (line 227)
    F_202341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 26), 'F', False)
    # Processing the call keyword arguments (line 227)
    kwargs_202342 = {}
    # Getting the type of '_complex2real' (line 227)
    _complex2real_202340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), '_complex2real', False)
    # Calling _complex2real(args, kwargs) (line 227)
    _complex2real_call_result_202343 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), _complex2real_202340, *[F_202341], **kwargs_202342)
    
    # Assigning a type to the variable 'F' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'F', _complex2real_call_result_202343)
    # SSA branch for the else part of an if statement (line 215)
    module_type_store.open_ssa_branch('else')

    @norecursion
    def wrap_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wrap_func'
        module_type_store = module_type_store.open_function_context('wrap_func', 229, 8, False)
        
        # Passed parameters checking function
        wrap_func.stypy_localization = localization
        wrap_func.stypy_type_of_self = None
        wrap_func.stypy_type_store = module_type_store
        wrap_func.stypy_function_name = 'wrap_func'
        wrap_func.stypy_param_names_list = ['x']
        wrap_func.stypy_varargs_param_name = None
        wrap_func.stypy_kwargs_param_name = None
        wrap_func.stypy_call_defaults = defaults
        wrap_func.stypy_call_varargs = varargs
        wrap_func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'wrap_func', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wrap_func', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wrap_func(...)' code ##################

        
        
        
        # Obtaining the type of the subscript
        int_202344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 25), 'int')
        # Getting the type of 'nfev_list' (line 230)
        nfev_list_202345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'nfev_list')
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___202346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 15), nfev_list_202345, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 230)
        subscript_call_result_202347 = invoke(stypy.reporting.localization.Localization(__file__, 230, 15), getitem___202346, int_202344)
        
        # Getting the type of 'maxfev' (line 230)
        maxfev_202348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 31), 'maxfev')
        # Applying the binary operator '>=' (line 230)
        result_ge_202349 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 15), '>=', subscript_call_result_202347, maxfev_202348)
        
        # Testing the type of an if condition (line 230)
        if_condition_202350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 12), result_ge_202349)
        # Assigning a type to the variable 'if_condition_202350' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'if_condition_202350', if_condition_202350)
        # SSA begins for if statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _NoConvergence(...): (line 231)
        # Processing the call keyword arguments (line 231)
        kwargs_202352 = {}
        # Getting the type of '_NoConvergence' (line 231)
        _NoConvergence_202351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 22), '_NoConvergence', False)
        # Calling _NoConvergence(args, kwargs) (line 231)
        _NoConvergence_call_result_202353 = invoke(stypy.reporting.localization.Localization(__file__, 231, 22), _NoConvergence_202351, *[], **kwargs_202352)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 231, 16), _NoConvergence_call_result_202353, 'raise parameter', BaseException)
        # SSA join for if statement (line 230)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'nfev_list' (line 232)
        nfev_list_202354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'nfev_list')
        
        # Obtaining the type of the subscript
        int_202355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 22), 'int')
        # Getting the type of 'nfev_list' (line 232)
        nfev_list_202356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'nfev_list')
        # Obtaining the member '__getitem__' of a type (line 232)
        getitem___202357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), nfev_list_202356, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 232)
        subscript_call_result_202358 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), getitem___202357, int_202355)
        
        int_202359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 28), 'int')
        # Applying the binary operator '+=' (line 232)
        result_iadd_202360 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 12), '+=', subscript_call_result_202358, int_202359)
        # Getting the type of 'nfev_list' (line 232)
        nfev_list_202361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'nfev_list')
        int_202362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 22), 'int')
        # Storing an element on a container (line 232)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 12), nfev_list_202361, (int_202362, result_iadd_202360))
        
        
        # Assigning a Call to a Name (line 233):
        
        # Assigning a Call to a Name (line 233):
        
        # Call to reshape(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'x0_shape' (line 233)
        x0_shape_202365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 26), 'x0_shape', False)
        # Processing the call keyword arguments (line 233)
        kwargs_202366 = {}
        # Getting the type of 'x' (line 233)
        x_202363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'x', False)
        # Obtaining the member 'reshape' of a type (line 233)
        reshape_202364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 16), x_202363, 'reshape')
        # Calling reshape(args, kwargs) (line 233)
        reshape_call_result_202367 = invoke(stypy.reporting.localization.Localization(__file__, 233, 16), reshape_202364, *[x0_shape_202365], **kwargs_202366)
        
        # Assigning a type to the variable 'x' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'x', reshape_call_result_202367)
        
        # Assigning a Call to a Name (line 234):
        
        # Assigning a Call to a Name (line 234):
        
        # Call to ravel(...): (line 234)
        # Processing the call keyword arguments (line 234)
        kwargs_202378 = {}
        
        # Call to asarray(...): (line 234)
        # Processing the call arguments (line 234)
        
        # Call to func(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'x' (line 234)
        x_202371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 32), 'x', False)
        # Getting the type of 'args' (line 234)
        args_202372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 36), 'args', False)
        # Processing the call keyword arguments (line 234)
        kwargs_202373 = {}
        # Getting the type of 'func' (line 234)
        func_202370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 27), 'func', False)
        # Calling func(args, kwargs) (line 234)
        func_call_result_202374 = invoke(stypy.reporting.localization.Localization(__file__, 234, 27), func_202370, *[x_202371, args_202372], **kwargs_202373)
        
        # Processing the call keyword arguments (line 234)
        kwargs_202375 = {}
        # Getting the type of 'np' (line 234)
        np_202368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'np', False)
        # Obtaining the member 'asarray' of a type (line 234)
        asarray_202369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 16), np_202368, 'asarray')
        # Calling asarray(args, kwargs) (line 234)
        asarray_call_result_202376 = invoke(stypy.reporting.localization.Localization(__file__, 234, 16), asarray_202369, *[func_call_result_202374], **kwargs_202375)
        
        # Obtaining the member 'ravel' of a type (line 234)
        ravel_202377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 16), asarray_call_result_202376, 'ravel')
        # Calling ravel(args, kwargs) (line 234)
        ravel_call_result_202379 = invoke(stypy.reporting.localization.Localization(__file__, 234, 16), ravel_202377, *[], **kwargs_202378)
        
        # Assigning a type to the variable 'F' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'F', ravel_call_result_202379)
        
        # Assigning a Call to a Name (line 235):
        
        # Assigning a Call to a Name (line 235):
        
        # Call to fmerit(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'F' (line 235)
        F_202381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 23), 'F', False)
        # Processing the call keyword arguments (line 235)
        kwargs_202382 = {}
        # Getting the type of 'fmerit' (line 235)
        fmerit_202380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'fmerit', False)
        # Calling fmerit(args, kwargs) (line 235)
        fmerit_call_result_202383 = invoke(stypy.reporting.localization.Localization(__file__, 235, 16), fmerit_202380, *[F_202381], **kwargs_202382)
        
        # Assigning a type to the variable 'f' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'f', fmerit_call_result_202383)
        
        # Obtaining an instance of the builtin type 'tuple' (line 236)
        tuple_202384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 236)
        # Adding element type (line 236)
        # Getting the type of 'f' (line 236)
        f_202385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 19), 'f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 19), tuple_202384, f_202385)
        # Adding element type (line 236)
        # Getting the type of 'F' (line 236)
        F_202386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 22), 'F')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 19), tuple_202384, F_202386)
        
        # Assigning a type to the variable 'stypy_return_type' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'stypy_return_type', tuple_202384)
        
        # ################# End of 'wrap_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wrap_func' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_202387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_202387)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wrap_func'
        return stypy_return_type_202387

    # Assigning a type to the variable 'wrap_func' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'wrap_func', wrap_func)
    # SSA join for if statement (line 215)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 238)
    tuple_202388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 238)
    # Adding element type (line 238)
    # Getting the type of 'wrap_func' (line 238)
    wrap_func_202389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'wrap_func')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 11), tuple_202388, wrap_func_202389)
    # Adding element type (line 238)
    # Getting the type of 'x0' (line 238)
    x0_202390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 22), 'x0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 11), tuple_202388, x0_202390)
    # Adding element type (line 238)
    # Getting the type of 'x0_shape' (line 238)
    x0_shape_202391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 26), 'x0_shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 11), tuple_202388, x0_shape_202391)
    # Adding element type (line 238)
    
    # Call to fmerit(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'F' (line 238)
    F_202393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 43), 'F', False)
    # Processing the call keyword arguments (line 238)
    kwargs_202394 = {}
    # Getting the type of 'fmerit' (line 238)
    fmerit_202392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 36), 'fmerit', False)
    # Calling fmerit(args, kwargs) (line 238)
    fmerit_call_result_202395 = invoke(stypy.reporting.localization.Localization(__file__, 238, 36), fmerit_202392, *[F_202393], **kwargs_202394)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 11), tuple_202388, fmerit_call_result_202395)
    # Adding element type (line 238)
    # Getting the type of 'F' (line 238)
    F_202396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 47), 'F')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 11), tuple_202388, F_202396)
    # Adding element type (line 238)
    # Getting the type of 'is_complex' (line 238)
    is_complex_202397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 50), 'is_complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 11), tuple_202388, is_complex_202397)
    
    # Assigning a type to the variable 'stypy_return_type' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'stypy_return_type', tuple_202388)
    
    # ################# End of '_wrap_func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_wrap_func' in the type store
    # Getting the type of 'stypy_return_type' (line 167)
    stypy_return_type_202398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_202398)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_wrap_func'
    return stypy_return_type_202398

# Assigning a type to the variable '_wrap_func' (line 167)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), '_wrap_func', _wrap_func)

@norecursion
def _wrap_result(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 241)
    None_202399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 43), 'None')
    defaults = [None_202399]
    # Create a new context for function '_wrap_result'
    module_type_store = module_type_store.open_function_context('_wrap_result', 241, 0, False)
    
    # Passed parameters checking function
    _wrap_result.stypy_localization = localization
    _wrap_result.stypy_type_of_self = None
    _wrap_result.stypy_type_store = module_type_store
    _wrap_result.stypy_function_name = '_wrap_result'
    _wrap_result.stypy_param_names_list = ['result', 'is_complex', 'shape']
    _wrap_result.stypy_varargs_param_name = None
    _wrap_result.stypy_kwargs_param_name = None
    _wrap_result.stypy_call_defaults = defaults
    _wrap_result.stypy_call_varargs = varargs
    _wrap_result.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_wrap_result', ['result', 'is_complex', 'shape'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_wrap_result', localization, ['result', 'is_complex', 'shape'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_wrap_result(...)' code ##################

    str_202400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, (-1)), 'str', '\n    Convert from real to complex and reshape result arrays.\n    ')
    
    # Getting the type of 'is_complex' (line 245)
    is_complex_202401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 7), 'is_complex')
    # Testing the type of an if condition (line 245)
    if_condition_202402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 4), is_complex_202401)
    # Assigning a type to the variable 'if_condition_202402' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'if_condition_202402', if_condition_202402)
    # SSA begins for if statement (line 245)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 246):
    
    # Assigning a Call to a Name (line 246):
    
    # Call to _real2complex(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'result' (line 246)
    result_202404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 26), 'result', False)
    # Processing the call keyword arguments (line 246)
    kwargs_202405 = {}
    # Getting the type of '_real2complex' (line 246)
    _real2complex_202403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), '_real2complex', False)
    # Calling _real2complex(args, kwargs) (line 246)
    _real2complex_call_result_202406 = invoke(stypy.reporting.localization.Localization(__file__, 246, 12), _real2complex_202403, *[result_202404], **kwargs_202405)
    
    # Assigning a type to the variable 'z' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'z', _real2complex_call_result_202406)
    # SSA branch for the else part of an if statement (line 245)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 248):
    
    # Assigning a Name to a Name (line 248):
    # Getting the type of 'result' (line 248)
    result_202407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'result')
    # Assigning a type to the variable 'z' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'z', result_202407)
    # SSA join for if statement (line 245)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 249)
    # Getting the type of 'shape' (line 249)
    shape_202408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'shape')
    # Getting the type of 'None' (line 249)
    None_202409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'None')
    
    (may_be_202410, more_types_in_union_202411) = may_not_be_none(shape_202408, None_202409)

    if may_be_202410:

        if more_types_in_union_202411:
            # Runtime conditional SSA (line 249)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to reshape(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'shape' (line 250)
        shape_202414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 22), 'shape', False)
        # Processing the call keyword arguments (line 250)
        kwargs_202415 = {}
        # Getting the type of 'z' (line 250)
        z_202412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'z', False)
        # Obtaining the member 'reshape' of a type (line 250)
        reshape_202413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 12), z_202412, 'reshape')
        # Calling reshape(args, kwargs) (line 250)
        reshape_call_result_202416 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), reshape_202413, *[shape_202414], **kwargs_202415)
        
        # Assigning a type to the variable 'z' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'z', reshape_call_result_202416)

        if more_types_in_union_202411:
            # SSA join for if statement (line 249)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'z' (line 251)
    z_202417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'z')
    # Assigning a type to the variable 'stypy_return_type' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'stypy_return_type', z_202417)
    
    # ################# End of '_wrap_result(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_wrap_result' in the type store
    # Getting the type of 'stypy_return_type' (line 241)
    stypy_return_type_202418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_202418)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_wrap_result'
    return stypy_return_type_202418

# Assigning a type to the variable '_wrap_result' (line 241)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), '_wrap_result', _wrap_result)

@norecursion
def _real2complex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_real2complex'
    module_type_store = module_type_store.open_function_context('_real2complex', 254, 0, False)
    
    # Passed parameters checking function
    _real2complex.stypy_localization = localization
    _real2complex.stypy_type_of_self = None
    _real2complex.stypy_type_store = module_type_store
    _real2complex.stypy_function_name = '_real2complex'
    _real2complex.stypy_param_names_list = ['x']
    _real2complex.stypy_varargs_param_name = None
    _real2complex.stypy_kwargs_param_name = None
    _real2complex.stypy_call_defaults = defaults
    _real2complex.stypy_call_varargs = varargs
    _real2complex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_real2complex', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_real2complex', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_real2complex(...)' code ##################

    
    # Call to view(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'np' (line 255)
    np_202427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 53), 'np', False)
    # Obtaining the member 'complex128' of a type (line 255)
    complex128_202428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 53), np_202427, 'complex128')
    # Processing the call keyword arguments (line 255)
    kwargs_202429 = {}
    
    # Call to ascontiguousarray(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'x' (line 255)
    x_202421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 32), 'x', False)
    # Processing the call keyword arguments (line 255)
    # Getting the type of 'float' (line 255)
    float_202422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 41), 'float', False)
    keyword_202423 = float_202422
    kwargs_202424 = {'dtype': keyword_202423}
    # Getting the type of 'np' (line 255)
    np_202419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 11), 'np', False)
    # Obtaining the member 'ascontiguousarray' of a type (line 255)
    ascontiguousarray_202420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 11), np_202419, 'ascontiguousarray')
    # Calling ascontiguousarray(args, kwargs) (line 255)
    ascontiguousarray_call_result_202425 = invoke(stypy.reporting.localization.Localization(__file__, 255, 11), ascontiguousarray_202420, *[x_202421], **kwargs_202424)
    
    # Obtaining the member 'view' of a type (line 255)
    view_202426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 11), ascontiguousarray_call_result_202425, 'view')
    # Calling view(args, kwargs) (line 255)
    view_call_result_202430 = invoke(stypy.reporting.localization.Localization(__file__, 255, 11), view_202426, *[complex128_202428], **kwargs_202429)
    
    # Assigning a type to the variable 'stypy_return_type' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'stypy_return_type', view_call_result_202430)
    
    # ################# End of '_real2complex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_real2complex' in the type store
    # Getting the type of 'stypy_return_type' (line 254)
    stypy_return_type_202431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_202431)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_real2complex'
    return stypy_return_type_202431

# Assigning a type to the variable '_real2complex' (line 254)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 0), '_real2complex', _real2complex)

@norecursion
def _complex2real(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_complex2real'
    module_type_store = module_type_store.open_function_context('_complex2real', 258, 0, False)
    
    # Passed parameters checking function
    _complex2real.stypy_localization = localization
    _complex2real.stypy_type_of_self = None
    _complex2real.stypy_type_store = module_type_store
    _complex2real.stypy_function_name = '_complex2real'
    _complex2real.stypy_param_names_list = ['z']
    _complex2real.stypy_varargs_param_name = None
    _complex2real.stypy_kwargs_param_name = None
    _complex2real.stypy_call_defaults = defaults
    _complex2real.stypy_call_varargs = varargs
    _complex2real.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_complex2real', ['z'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_complex2real', localization, ['z'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_complex2real(...)' code ##################

    
    # Call to view(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'np' (line 259)
    np_202440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 55), 'np', False)
    # Obtaining the member 'float64' of a type (line 259)
    float64_202441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 55), np_202440, 'float64')
    # Processing the call keyword arguments (line 259)
    kwargs_202442 = {}
    
    # Call to ascontiguousarray(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'z' (line 259)
    z_202434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 32), 'z', False)
    # Processing the call keyword arguments (line 259)
    # Getting the type of 'complex' (line 259)
    complex_202435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 41), 'complex', False)
    keyword_202436 = complex_202435
    kwargs_202437 = {'dtype': keyword_202436}
    # Getting the type of 'np' (line 259)
    np_202432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'np', False)
    # Obtaining the member 'ascontiguousarray' of a type (line 259)
    ascontiguousarray_202433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 11), np_202432, 'ascontiguousarray')
    # Calling ascontiguousarray(args, kwargs) (line 259)
    ascontiguousarray_call_result_202438 = invoke(stypy.reporting.localization.Localization(__file__, 259, 11), ascontiguousarray_202433, *[z_202434], **kwargs_202437)
    
    # Obtaining the member 'view' of a type (line 259)
    view_202439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 11), ascontiguousarray_call_result_202438, 'view')
    # Calling view(args, kwargs) (line 259)
    view_call_result_202443 = invoke(stypy.reporting.localization.Localization(__file__, 259, 11), view_202439, *[float64_202441], **kwargs_202442)
    
    # Assigning a type to the variable 'stypy_return_type' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'stypy_return_type', view_call_result_202443)
    
    # ################# End of '_complex2real(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_complex2real' in the type store
    # Getting the type of 'stypy_return_type' (line 258)
    stypy_return_type_202444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_202444)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_complex2real'
    return stypy_return_type_202444

# Assigning a type to the variable '_complex2real' (line 258)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 0), '_complex2real', _complex2real)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
