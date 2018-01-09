
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import numpy as np
2: from scipy.integrate import ode
3: from .common import validate_tol, warn_extraneous
4: from .base import OdeSolver, DenseOutput
5: 
6: 
7: class LSODA(OdeSolver):
8:     '''Adams/BDF method with automatic stiffness detection and switching.
9: 
10:     This is a wrapper to the Fortran solver from ODEPACK [1]_. It switches
11:     automatically between the nonstiff Adams method and the stiff BDF method.
12:     The method was originally detailed in [2]_.
13: 
14:     Parameters
15:     ----------
16:     fun : callable
17:         Right-hand side of the system. The calling signature is ``fun(t, y)``.
18:         Here ``t`` is a scalar and there are two options for ndarray ``y``.
19:         It can either have shape (n,), then ``fun`` must return array_like with
20:         shape (n,). Or alternatively it can have shape (n, k), then ``fun``
21:         must return array_like with shape (n, k), i.e. each column
22:         corresponds to a single column in ``y``. The choice between the two
23:         options is determined by `vectorized` argument (see below). The
24:         vectorized implementation allows faster approximation of the Jacobian
25:         by finite differences.
26:     t0 : float
27:         Initial time.
28:     y0 : array_like, shape (n,)
29:         Initial state.
30:     t_bound : float
31:         Boundary time --- the integration won't continue beyond it. It also
32:         determines the direction of the integration.
33:     first_step : float or None, optional
34:         Initial step size. Default is ``None`` which means that the algorithm
35:         should choose.
36:     min_step : float, optional
37:         Minimum allowed step size. Default is 0.0, i.e. the step is not
38:         bounded and determined solely by the solver.
39:     max_step : float, optional
40:         Maximum allowed step size. Default is ``np.inf``, i.e. the step is not
41:         bounded and determined solely by the solver.
42:     rtol, atol : float and array_like, optional
43:         Relative and absolute tolerances. The solver keeps the local error
44:         estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
45:         relative accuracy (number of correct digits). But if a component of `y`
46:         is approximately below `atol` then the error only needs to fall within
47:         the same `atol` threshold, and the number of correct digits is not
48:         guaranteed. If components of y have different scales, it might be
49:         beneficial to set different `atol` values for different components by
50:         passing array_like with shape (n,) for `atol`. Default values are
51:         1e-3 for `rtol` and 1e-6 for `atol`.
52:     jac : None or callable, optional
53:         Jacobian matrix of the right-hand side of the system with respect to
54:         ``y``. The Jacobian matrix has shape (n, n) and its element (i, j) is
55:         equal to ``d f_i / d y_j``. The function will be called as
56:         ``jac(t, y)``. If None (default), then the Jacobian will be
57:         approximated by finite differences. It is generally recommended to
58:         provide the Jacobian rather than relying on a finite difference
59:         approximation.
60:     lband, uband : int or None, optional
61:         Jacobian band width:
62:         ``jac[i, j] != 0 only for i - lband <= j <= i + uband``. Setting these
63:         requires your jac routine to return the Jacobian in the packed format:
64:         the returned array must have ``n`` columns and ``uband + lband + 1``
65:         rows in which Jacobian diagonals are written. Specifically
66:         ``jac_packed[uband + i - j , j] = jac[i, j]``. The same format is used
67:         in `scipy.linalg.solve_banded` (check for an illustration).
68:         These parameters can be also used with ``jac=None`` to reduce the
69:         number of Jacobian elements estimated by finite differences.
70:     vectorized : bool, optional
71:         Whether `fun` is implemented in a vectorized fashion. A vectorized
72:         implementation offers no advantages for this solver. Default is False.
73: 
74:     Attributes
75:     ----------
76:     n : int
77:         Number of equations.
78:     status : string
79:         Current status of the solver: 'running', 'finished' or 'failed'.
80:     t_bound : float
81:         Boundary time.
82:     direction : float
83:         Integration direction: +1 or -1.
84:     t : float
85:         Current time.
86:     y : ndarray
87:         Current state.
88:     t_old : float
89:         Previous time. None if no steps were made yet.
90:     nfev : int
91:         Number of the system's rhs evaluations.
92:     njev : int
93:         Number of the Jacobian evaluations.
94: 
95:     References
96:     ----------
97:     .. [1] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
98:            Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
99:            pp. 55-64, 1983.
100:     .. [2] L. Petzold, "Automatic selection of methods for solving stiff and
101:            nonstiff systems of ordinary differential equations", SIAM Journal
102:            on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,
103:            1983.
104:     '''
105:     def __init__(self, fun, t0, y0, t_bound, first_step=None, min_step=0.0,
106:                  max_step=np.inf, rtol=1e-3, atol=1e-6, jac=None, lband=None,
107:                  uband=None, vectorized=False, **extraneous):
108:         warn_extraneous(extraneous)
109:         super(LSODA, self).__init__(fun, t0, y0, t_bound, vectorized)
110: 
111:         if first_step is None:
112:             first_step = 0  # LSODA value for automatic selection.
113:         elif first_step <= 0:
114:             raise ValueError("`first_step` must be positive or None.")
115: 
116:         if max_step == np.inf:
117:             max_step = 0  # LSODA value for infinity.
118:         elif max_step <= 0:
119:             raise ValueError("`max_step` must be positive.")
120: 
121:         if min_step < 0:
122:             raise ValueError("`min_step` must be nonnegative.")
123: 
124:         rtol, atol = validate_tol(rtol, atol, self.n)
125: 
126:         if jac is None:  # No lambda as PEP8 insists.
127:             def jac():
128:                 return None
129: 
130:         solver = ode(self.fun, jac)
131:         solver.set_integrator('lsoda', rtol=rtol, atol=atol, max_step=max_step,
132:                               min_step=min_step, first_step=first_step,
133:                               lband=lband, uband=uband)
134:         solver.set_initial_value(y0, t0)
135: 
136:         # Inject t_bound into rwork array as needed for itask=5.
137:         solver._integrator.rwork[0] = self.t_bound
138:         solver._integrator.call_args[4] = solver._integrator.rwork
139: 
140:         self._lsoda_solver = solver
141: 
142:     def _step_impl(self):
143:         solver = self._lsoda_solver
144:         integrator = solver._integrator
145: 
146:         # From lsoda.step and lsoda.integrate itask=5 means take a single
147:         # step and do not go past t_bound.
148:         itask = integrator.call_args[2]
149:         integrator.call_args[2] = 5
150:         solver._y, solver.t = integrator.run(
151:             solver.f, solver.jac, solver._y, solver.t,
152:             self.t_bound, solver.f_params, solver.jac_params)
153:         integrator.call_args[2] = itask
154: 
155:         if solver.successful():
156:             self.t = solver.t
157:             self.y = solver._y
158:             # From LSODA Fortran source njev is equal to nlu.
159:             self.njev = integrator.iwork[12]
160:             self.nlu = integrator.iwork[12]
161:             return True, None
162:         else:
163:             return False, 'Unexpected istate in LSODA.'
164: 
165:     def _dense_output_impl(self):
166:         iwork = self._lsoda_solver._integrator.iwork
167:         rwork = self._lsoda_solver._integrator.rwork
168: 
169:         order = iwork[14]
170:         h = rwork[11]
171:         yh = np.reshape(rwork[20:20 + (order + 1) * self.n],
172:                         (self.n, order + 1), order='F').copy()
173: 
174:         return LsodaDenseOutput(self.t_old, self.t, h, order, yh)
175: 
176: 
177: class LsodaDenseOutput(DenseOutput):
178:     def __init__(self, t_old, t, h, order, yh):
179:         super(LsodaDenseOutput, self).__init__(t_old, t)
180:         self.h = h
181:         self.yh = yh
182:         self.p = np.arange(order + 1)
183: 
184:     def _call_impl(self, t):
185:         if t.ndim == 0:
186:             x = ((t - self.t) / self.h) ** self.p
187:         else:
188:             x = ((t - self.t) / self.h) ** self.p[:, None]
189: 
190:         return np.dot(self.yh, x)
191: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import numpy' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_56350 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy')

if (type(import_56350) is not StypyTypeError):

    if (import_56350 != 'pyd_module'):
        __import__(import_56350)
        sys_modules_56351 = sys.modules[import_56350]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'np', sys_modules_56351.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy', import_56350)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from scipy.integrate import ode' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_56352 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy.integrate')

if (type(import_56352) is not StypyTypeError):

    if (import_56352 != 'pyd_module'):
        __import__(import_56352)
        sys_modules_56353 = sys.modules[import_56352]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy.integrate', sys_modules_56353.module_type_store, module_type_store, ['ode'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_56353, sys_modules_56353.module_type_store, module_type_store)
    else:
        from scipy.integrate import ode

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy.integrate', None, module_type_store, ['ode'], [ode])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy.integrate', import_56352)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy.integrate._ivp.common import validate_tol, warn_extraneous' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_56354 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.integrate._ivp.common')

if (type(import_56354) is not StypyTypeError):

    if (import_56354 != 'pyd_module'):
        __import__(import_56354)
        sys_modules_56355 = sys.modules[import_56354]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.integrate._ivp.common', sys_modules_56355.module_type_store, module_type_store, ['validate_tol', 'warn_extraneous'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_56355, sys_modules_56355.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.common import validate_tol, warn_extraneous

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.integrate._ivp.common', None, module_type_store, ['validate_tol', 'warn_extraneous'], [validate_tol, warn_extraneous])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.common' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.integrate._ivp.common', import_56354)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.integrate._ivp.base import OdeSolver, DenseOutput' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_56356 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.base')

if (type(import_56356) is not StypyTypeError):

    if (import_56356 != 'pyd_module'):
        __import__(import_56356)
        sys_modules_56357 = sys.modules[import_56356]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.base', sys_modules_56357.module_type_store, module_type_store, ['OdeSolver', 'DenseOutput'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_56357, sys_modules_56357.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.base import OdeSolver, DenseOutput

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.base', None, module_type_store, ['OdeSolver', 'DenseOutput'], [OdeSolver, DenseOutput])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.base' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.base', import_56356)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

# Declaration of the 'LSODA' class
# Getting the type of 'OdeSolver' (line 7)
OdeSolver_56358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'OdeSolver')

class LSODA(OdeSolver_56358, ):
    str_56359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, (-1)), 'str', 'Adams/BDF method with automatic stiffness detection and switching.\n\n    This is a wrapper to the Fortran solver from ODEPACK [1]_. It switches\n    automatically between the nonstiff Adams method and the stiff BDF method.\n    The method was originally detailed in [2]_.\n\n    Parameters\n    ----------\n    fun : callable\n        Right-hand side of the system. The calling signature is ``fun(t, y)``.\n        Here ``t`` is a scalar and there are two options for ndarray ``y``.\n        It can either have shape (n,), then ``fun`` must return array_like with\n        shape (n,). Or alternatively it can have shape (n, k), then ``fun``\n        must return array_like with shape (n, k), i.e. each column\n        corresponds to a single column in ``y``. The choice between the two\n        options is determined by `vectorized` argument (see below). The\n        vectorized implementation allows faster approximation of the Jacobian\n        by finite differences.\n    t0 : float\n        Initial time.\n    y0 : array_like, shape (n,)\n        Initial state.\n    t_bound : float\n        Boundary time --- the integration won\'t continue beyond it. It also\n        determines the direction of the integration.\n    first_step : float or None, optional\n        Initial step size. Default is ``None`` which means that the algorithm\n        should choose.\n    min_step : float, optional\n        Minimum allowed step size. Default is 0.0, i.e. the step is not\n        bounded and determined solely by the solver.\n    max_step : float, optional\n        Maximum allowed step size. Default is ``np.inf``, i.e. the step is not\n        bounded and determined solely by the solver.\n    rtol, atol : float and array_like, optional\n        Relative and absolute tolerances. The solver keeps the local error\n        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a\n        relative accuracy (number of correct digits). But if a component of `y`\n        is approximately below `atol` then the error only needs to fall within\n        the same `atol` threshold, and the number of correct digits is not\n        guaranteed. If components of y have different scales, it might be\n        beneficial to set different `atol` values for different components by\n        passing array_like with shape (n,) for `atol`. Default values are\n        1e-3 for `rtol` and 1e-6 for `atol`.\n    jac : None or callable, optional\n        Jacobian matrix of the right-hand side of the system with respect to\n        ``y``. The Jacobian matrix has shape (n, n) and its element (i, j) is\n        equal to ``d f_i / d y_j``. The function will be called as\n        ``jac(t, y)``. If None (default), then the Jacobian will be\n        approximated by finite differences. It is generally recommended to\n        provide the Jacobian rather than relying on a finite difference\n        approximation.\n    lband, uband : int or None, optional\n        Jacobian band width:\n        ``jac[i, j] != 0 only for i - lband <= j <= i + uband``. Setting these\n        requires your jac routine to return the Jacobian in the packed format:\n        the returned array must have ``n`` columns and ``uband + lband + 1``\n        rows in which Jacobian diagonals are written. Specifically\n        ``jac_packed[uband + i - j , j] = jac[i, j]``. The same format is used\n        in `scipy.linalg.solve_banded` (check for an illustration).\n        These parameters can be also used with ``jac=None`` to reduce the\n        number of Jacobian elements estimated by finite differences.\n    vectorized : bool, optional\n        Whether `fun` is implemented in a vectorized fashion. A vectorized\n        implementation offers no advantages for this solver. Default is False.\n\n    Attributes\n    ----------\n    n : int\n        Number of equations.\n    status : string\n        Current status of the solver: \'running\', \'finished\' or \'failed\'.\n    t_bound : float\n        Boundary time.\n    direction : float\n        Integration direction: +1 or -1.\n    t : float\n        Current time.\n    y : ndarray\n        Current state.\n    t_old : float\n        Previous time. None if no steps were made yet.\n    nfev : int\n        Number of the system\'s rhs evaluations.\n    njev : int\n        Number of the Jacobian evaluations.\n\n    References\n    ----------\n    .. [1] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE\n           Solvers," IMACS Transactions on Scientific Computation, Vol 1.,\n           pp. 55-64, 1983.\n    .. [2] L. Petzold, "Automatic selection of methods for solving stiff and\n           nonstiff systems of ordinary differential equations", SIAM Journal\n           on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,\n           1983.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 105)
        None_56360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 56), 'None')
        float_56361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 71), 'float')
        # Getting the type of 'np' (line 106)
        np_56362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'np')
        # Obtaining the member 'inf' of a type (line 106)
        inf_56363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 26), np_56362, 'inf')
        float_56364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 39), 'float')
        float_56365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 50), 'float')
        # Getting the type of 'None' (line 106)
        None_56366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 60), 'None')
        # Getting the type of 'None' (line 106)
        None_56367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 72), 'None')
        # Getting the type of 'None' (line 107)
        None_56368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 23), 'None')
        # Getting the type of 'False' (line 107)
        False_56369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 40), 'False')
        defaults = [None_56360, float_56361, inf_56363, float_56364, float_56365, None_56366, None_56367, None_56368, False_56369]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LSODA.__init__', ['fun', 't0', 'y0', 't_bound', 'first_step', 'min_step', 'max_step', 'rtol', 'atol', 'jac', 'lband', 'uband', 'vectorized'], None, 'extraneous', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['fun', 't0', 'y0', 't_bound', 'first_step', 'min_step', 'max_step', 'rtol', 'atol', 'jac', 'lband', 'uband', 'vectorized'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to warn_extraneous(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'extraneous' (line 108)
        extraneous_56371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'extraneous', False)
        # Processing the call keyword arguments (line 108)
        kwargs_56372 = {}
        # Getting the type of 'warn_extraneous' (line 108)
        warn_extraneous_56370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'warn_extraneous', False)
        # Calling warn_extraneous(args, kwargs) (line 108)
        warn_extraneous_call_result_56373 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), warn_extraneous_56370, *[extraneous_56371], **kwargs_56372)
        
        
        # Call to __init__(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'fun' (line 109)
        fun_56380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 36), 'fun', False)
        # Getting the type of 't0' (line 109)
        t0_56381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 41), 't0', False)
        # Getting the type of 'y0' (line 109)
        y0_56382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 45), 'y0', False)
        # Getting the type of 't_bound' (line 109)
        t_bound_56383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 49), 't_bound', False)
        # Getting the type of 'vectorized' (line 109)
        vectorized_56384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 58), 'vectorized', False)
        # Processing the call keyword arguments (line 109)
        kwargs_56385 = {}
        
        # Call to super(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'LSODA' (line 109)
        LSODA_56375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 14), 'LSODA', False)
        # Getting the type of 'self' (line 109)
        self_56376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'self', False)
        # Processing the call keyword arguments (line 109)
        kwargs_56377 = {}
        # Getting the type of 'super' (line 109)
        super_56374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'super', False)
        # Calling super(args, kwargs) (line 109)
        super_call_result_56378 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), super_56374, *[LSODA_56375, self_56376], **kwargs_56377)
        
        # Obtaining the member '__init__' of a type (line 109)
        init___56379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), super_call_result_56378, '__init__')
        # Calling __init__(args, kwargs) (line 109)
        init___call_result_56386 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), init___56379, *[fun_56380, t0_56381, y0_56382, t_bound_56383, vectorized_56384], **kwargs_56385)
        
        
        # Type idiom detected: calculating its left and rigth part (line 111)
        # Getting the type of 'first_step' (line 111)
        first_step_56387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'first_step')
        # Getting the type of 'None' (line 111)
        None_56388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'None')
        
        (may_be_56389, more_types_in_union_56390) = may_be_none(first_step_56387, None_56388)

        if may_be_56389:

            if more_types_in_union_56390:
                # Runtime conditional SSA (line 111)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 112):
            
            # Assigning a Num to a Name (line 112):
            int_56391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 25), 'int')
            # Assigning a type to the variable 'first_step' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'first_step', int_56391)

            if more_types_in_union_56390:
                # Runtime conditional SSA for else branch (line 111)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_56389) or more_types_in_union_56390):
            
            
            # Getting the type of 'first_step' (line 113)
            first_step_56392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'first_step')
            int_56393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 27), 'int')
            # Applying the binary operator '<=' (line 113)
            result_le_56394 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 13), '<=', first_step_56392, int_56393)
            
            # Testing the type of an if condition (line 113)
            if_condition_56395 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 13), result_le_56394)
            # Assigning a type to the variable 'if_condition_56395' (line 113)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'if_condition_56395', if_condition_56395)
            # SSA begins for if statement (line 113)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 114)
            # Processing the call arguments (line 114)
            str_56397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 29), 'str', '`first_step` must be positive or None.')
            # Processing the call keyword arguments (line 114)
            kwargs_56398 = {}
            # Getting the type of 'ValueError' (line 114)
            ValueError_56396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 114)
            ValueError_call_result_56399 = invoke(stypy.reporting.localization.Localization(__file__, 114, 18), ValueError_56396, *[str_56397], **kwargs_56398)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 114, 12), ValueError_call_result_56399, 'raise parameter', BaseException)
            # SSA join for if statement (line 113)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_56389 and more_types_in_union_56390):
                # SSA join for if statement (line 111)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'max_step' (line 116)
        max_step_56400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'max_step')
        # Getting the type of 'np' (line 116)
        np_56401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'np')
        # Obtaining the member 'inf' of a type (line 116)
        inf_56402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 23), np_56401, 'inf')
        # Applying the binary operator '==' (line 116)
        result_eq_56403 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 11), '==', max_step_56400, inf_56402)
        
        # Testing the type of an if condition (line 116)
        if_condition_56404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 8), result_eq_56403)
        # Assigning a type to the variable 'if_condition_56404' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'if_condition_56404', if_condition_56404)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 117):
        
        # Assigning a Num to a Name (line 117):
        int_56405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'int')
        # Assigning a type to the variable 'max_step' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'max_step', int_56405)
        # SSA branch for the else part of an if statement (line 116)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'max_step' (line 118)
        max_step_56406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 13), 'max_step')
        int_56407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 25), 'int')
        # Applying the binary operator '<=' (line 118)
        result_le_56408 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 13), '<=', max_step_56406, int_56407)
        
        # Testing the type of an if condition (line 118)
        if_condition_56409 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 13), result_le_56408)
        # Assigning a type to the variable 'if_condition_56409' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 13), 'if_condition_56409', if_condition_56409)
        # SSA begins for if statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 119)
        # Processing the call arguments (line 119)
        str_56411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 29), 'str', '`max_step` must be positive.')
        # Processing the call keyword arguments (line 119)
        kwargs_56412 = {}
        # Getting the type of 'ValueError' (line 119)
        ValueError_56410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 119)
        ValueError_call_result_56413 = invoke(stypy.reporting.localization.Localization(__file__, 119, 18), ValueError_56410, *[str_56411], **kwargs_56412)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 119, 12), ValueError_call_result_56413, 'raise parameter', BaseException)
        # SSA join for if statement (line 118)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'min_step' (line 121)
        min_step_56414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'min_step')
        int_56415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 22), 'int')
        # Applying the binary operator '<' (line 121)
        result_lt_56416 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 11), '<', min_step_56414, int_56415)
        
        # Testing the type of an if condition (line 121)
        if_condition_56417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 8), result_lt_56416)
        # Assigning a type to the variable 'if_condition_56417' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'if_condition_56417', if_condition_56417)
        # SSA begins for if statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 122)
        # Processing the call arguments (line 122)
        str_56419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 29), 'str', '`min_step` must be nonnegative.')
        # Processing the call keyword arguments (line 122)
        kwargs_56420 = {}
        # Getting the type of 'ValueError' (line 122)
        ValueError_56418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 122)
        ValueError_call_result_56421 = invoke(stypy.reporting.localization.Localization(__file__, 122, 18), ValueError_56418, *[str_56419], **kwargs_56420)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 122, 12), ValueError_call_result_56421, 'raise parameter', BaseException)
        # SSA join for if statement (line 121)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 124):
        
        # Assigning a Subscript to a Name (line 124):
        
        # Obtaining the type of the subscript
        int_56422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 8), 'int')
        
        # Call to validate_tol(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'rtol' (line 124)
        rtol_56424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 34), 'rtol', False)
        # Getting the type of 'atol' (line 124)
        atol_56425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'atol', False)
        # Getting the type of 'self' (line 124)
        self_56426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 46), 'self', False)
        # Obtaining the member 'n' of a type (line 124)
        n_56427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 46), self_56426, 'n')
        # Processing the call keyword arguments (line 124)
        kwargs_56428 = {}
        # Getting the type of 'validate_tol' (line 124)
        validate_tol_56423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'validate_tol', False)
        # Calling validate_tol(args, kwargs) (line 124)
        validate_tol_call_result_56429 = invoke(stypy.reporting.localization.Localization(__file__, 124, 21), validate_tol_56423, *[rtol_56424, atol_56425, n_56427], **kwargs_56428)
        
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___56430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), validate_tol_call_result_56429, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_56431 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), getitem___56430, int_56422)
        
        # Assigning a type to the variable 'tuple_var_assignment_56346' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'tuple_var_assignment_56346', subscript_call_result_56431)
        
        # Assigning a Subscript to a Name (line 124):
        
        # Obtaining the type of the subscript
        int_56432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 8), 'int')
        
        # Call to validate_tol(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'rtol' (line 124)
        rtol_56434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 34), 'rtol', False)
        # Getting the type of 'atol' (line 124)
        atol_56435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'atol', False)
        # Getting the type of 'self' (line 124)
        self_56436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 46), 'self', False)
        # Obtaining the member 'n' of a type (line 124)
        n_56437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 46), self_56436, 'n')
        # Processing the call keyword arguments (line 124)
        kwargs_56438 = {}
        # Getting the type of 'validate_tol' (line 124)
        validate_tol_56433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'validate_tol', False)
        # Calling validate_tol(args, kwargs) (line 124)
        validate_tol_call_result_56439 = invoke(stypy.reporting.localization.Localization(__file__, 124, 21), validate_tol_56433, *[rtol_56434, atol_56435, n_56437], **kwargs_56438)
        
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___56440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), validate_tol_call_result_56439, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_56441 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), getitem___56440, int_56432)
        
        # Assigning a type to the variable 'tuple_var_assignment_56347' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'tuple_var_assignment_56347', subscript_call_result_56441)
        
        # Assigning a Name to a Name (line 124):
        # Getting the type of 'tuple_var_assignment_56346' (line 124)
        tuple_var_assignment_56346_56442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'tuple_var_assignment_56346')
        # Assigning a type to the variable 'rtol' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'rtol', tuple_var_assignment_56346_56442)
        
        # Assigning a Name to a Name (line 124):
        # Getting the type of 'tuple_var_assignment_56347' (line 124)
        tuple_var_assignment_56347_56443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'tuple_var_assignment_56347')
        # Assigning a type to the variable 'atol' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 14), 'atol', tuple_var_assignment_56347_56443)
        
        # Type idiom detected: calculating its left and rigth part (line 126)
        # Getting the type of 'jac' (line 126)
        jac_56444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'jac')
        # Getting the type of 'None' (line 126)
        None_56445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 18), 'None')
        
        (may_be_56446, more_types_in_union_56447) = may_be_none(jac_56444, None_56445)

        if may_be_56446:

            if more_types_in_union_56447:
                # Runtime conditional SSA (line 126)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store


            @norecursion
            def jac(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'jac'
                module_type_store = module_type_store.open_function_context('jac', 127, 12, False)
                
                # Passed parameters checking function
                jac.stypy_localization = localization
                jac.stypy_type_of_self = None
                jac.stypy_type_store = module_type_store
                jac.stypy_function_name = 'jac'
                jac.stypy_param_names_list = []
                jac.stypy_varargs_param_name = None
                jac.stypy_kwargs_param_name = None
                jac.stypy_call_defaults = defaults
                jac.stypy_call_varargs = varargs
                jac.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'jac', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'jac', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'jac(...)' code ##################

                # Getting the type of 'None' (line 128)
                None_56448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'None')
                # Assigning a type to the variable 'stypy_return_type' (line 128)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'stypy_return_type', None_56448)
                
                # ################# End of 'jac(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'jac' in the type store
                # Getting the type of 'stypy_return_type' (line 127)
                stypy_return_type_56449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_56449)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'jac'
                return stypy_return_type_56449

            # Assigning a type to the variable 'jac' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'jac', jac)

            if more_types_in_union_56447:
                # SSA join for if statement (line 126)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to ode(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'self' (line 130)
        self_56451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'self', False)
        # Obtaining the member 'fun' of a type (line 130)
        fun_56452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 21), self_56451, 'fun')
        # Getting the type of 'jac' (line 130)
        jac_56453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 31), 'jac', False)
        # Processing the call keyword arguments (line 130)
        kwargs_56454 = {}
        # Getting the type of 'ode' (line 130)
        ode_56450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 17), 'ode', False)
        # Calling ode(args, kwargs) (line 130)
        ode_call_result_56455 = invoke(stypy.reporting.localization.Localization(__file__, 130, 17), ode_56450, *[fun_56452, jac_56453], **kwargs_56454)
        
        # Assigning a type to the variable 'solver' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'solver', ode_call_result_56455)
        
        # Call to set_integrator(...): (line 131)
        # Processing the call arguments (line 131)
        str_56458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 30), 'str', 'lsoda')
        # Processing the call keyword arguments (line 131)
        # Getting the type of 'rtol' (line 131)
        rtol_56459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 44), 'rtol', False)
        keyword_56460 = rtol_56459
        # Getting the type of 'atol' (line 131)
        atol_56461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 55), 'atol', False)
        keyword_56462 = atol_56461
        # Getting the type of 'max_step' (line 131)
        max_step_56463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 70), 'max_step', False)
        keyword_56464 = max_step_56463
        # Getting the type of 'min_step' (line 132)
        min_step_56465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 39), 'min_step', False)
        keyword_56466 = min_step_56465
        # Getting the type of 'first_step' (line 132)
        first_step_56467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 60), 'first_step', False)
        keyword_56468 = first_step_56467
        # Getting the type of 'lband' (line 133)
        lband_56469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 'lband', False)
        keyword_56470 = lband_56469
        # Getting the type of 'uband' (line 133)
        uband_56471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 49), 'uband', False)
        keyword_56472 = uband_56471
        kwargs_56473 = {'max_step': keyword_56464, 'lband': keyword_56470, 'uband': keyword_56472, 'min_step': keyword_56466, 'first_step': keyword_56468, 'rtol': keyword_56460, 'atol': keyword_56462}
        # Getting the type of 'solver' (line 131)
        solver_56456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'solver', False)
        # Obtaining the member 'set_integrator' of a type (line 131)
        set_integrator_56457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), solver_56456, 'set_integrator')
        # Calling set_integrator(args, kwargs) (line 131)
        set_integrator_call_result_56474 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), set_integrator_56457, *[str_56458], **kwargs_56473)
        
        
        # Call to set_initial_value(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'y0' (line 134)
        y0_56477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 33), 'y0', False)
        # Getting the type of 't0' (line 134)
        t0_56478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 37), 't0', False)
        # Processing the call keyword arguments (line 134)
        kwargs_56479 = {}
        # Getting the type of 'solver' (line 134)
        solver_56475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'solver', False)
        # Obtaining the member 'set_initial_value' of a type (line 134)
        set_initial_value_56476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), solver_56475, 'set_initial_value')
        # Calling set_initial_value(args, kwargs) (line 134)
        set_initial_value_call_result_56480 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), set_initial_value_56476, *[y0_56477, t0_56478], **kwargs_56479)
        
        
        # Assigning a Attribute to a Subscript (line 137):
        
        # Assigning a Attribute to a Subscript (line 137):
        # Getting the type of 'self' (line 137)
        self_56481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 38), 'self')
        # Obtaining the member 't_bound' of a type (line 137)
        t_bound_56482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 38), self_56481, 't_bound')
        # Getting the type of 'solver' (line 137)
        solver_56483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'solver')
        # Obtaining the member '_integrator' of a type (line 137)
        _integrator_56484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), solver_56483, '_integrator')
        # Obtaining the member 'rwork' of a type (line 137)
        rwork_56485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), _integrator_56484, 'rwork')
        int_56486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 33), 'int')
        # Storing an element on a container (line 137)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 8), rwork_56485, (int_56486, t_bound_56482))
        
        # Assigning a Attribute to a Subscript (line 138):
        
        # Assigning a Attribute to a Subscript (line 138):
        # Getting the type of 'solver' (line 138)
        solver_56487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 42), 'solver')
        # Obtaining the member '_integrator' of a type (line 138)
        _integrator_56488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 42), solver_56487, '_integrator')
        # Obtaining the member 'rwork' of a type (line 138)
        rwork_56489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 42), _integrator_56488, 'rwork')
        # Getting the type of 'solver' (line 138)
        solver_56490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'solver')
        # Obtaining the member '_integrator' of a type (line 138)
        _integrator_56491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), solver_56490, '_integrator')
        # Obtaining the member 'call_args' of a type (line 138)
        call_args_56492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), _integrator_56491, 'call_args')
        int_56493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 37), 'int')
        # Storing an element on a container (line 138)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 8), call_args_56492, (int_56493, rwork_56489))
        
        # Assigning a Name to a Attribute (line 140):
        
        # Assigning a Name to a Attribute (line 140):
        # Getting the type of 'solver' (line 140)
        solver_56494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 29), 'solver')
        # Getting the type of 'self' (line 140)
        self_56495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'self')
        # Setting the type of the member '_lsoda_solver' of a type (line 140)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), self_56495, '_lsoda_solver', solver_56494)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _step_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_step_impl'
        module_type_store = module_type_store.open_function_context('_step_impl', 142, 4, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LSODA._step_impl.__dict__.__setitem__('stypy_localization', localization)
        LSODA._step_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LSODA._step_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        LSODA._step_impl.__dict__.__setitem__('stypy_function_name', 'LSODA._step_impl')
        LSODA._step_impl.__dict__.__setitem__('stypy_param_names_list', [])
        LSODA._step_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        LSODA._step_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LSODA._step_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        LSODA._step_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        LSODA._step_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LSODA._step_impl.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LSODA._step_impl', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_step_impl', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_step_impl(...)' code ##################

        
        # Assigning a Attribute to a Name (line 143):
        
        # Assigning a Attribute to a Name (line 143):
        # Getting the type of 'self' (line 143)
        self_56496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 17), 'self')
        # Obtaining the member '_lsoda_solver' of a type (line 143)
        _lsoda_solver_56497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 17), self_56496, '_lsoda_solver')
        # Assigning a type to the variable 'solver' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'solver', _lsoda_solver_56497)
        
        # Assigning a Attribute to a Name (line 144):
        
        # Assigning a Attribute to a Name (line 144):
        # Getting the type of 'solver' (line 144)
        solver_56498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'solver')
        # Obtaining the member '_integrator' of a type (line 144)
        _integrator_56499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 21), solver_56498, '_integrator')
        # Assigning a type to the variable 'integrator' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'integrator', _integrator_56499)
        
        # Assigning a Subscript to a Name (line 148):
        
        # Assigning a Subscript to a Name (line 148):
        
        # Obtaining the type of the subscript
        int_56500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 37), 'int')
        # Getting the type of 'integrator' (line 148)
        integrator_56501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'integrator')
        # Obtaining the member 'call_args' of a type (line 148)
        call_args_56502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 16), integrator_56501, 'call_args')
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___56503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 16), call_args_56502, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_56504 = invoke(stypy.reporting.localization.Localization(__file__, 148, 16), getitem___56503, int_56500)
        
        # Assigning a type to the variable 'itask' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'itask', subscript_call_result_56504)
        
        # Assigning a Num to a Subscript (line 149):
        
        # Assigning a Num to a Subscript (line 149):
        int_56505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 34), 'int')
        # Getting the type of 'integrator' (line 149)
        integrator_56506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'integrator')
        # Obtaining the member 'call_args' of a type (line 149)
        call_args_56507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), integrator_56506, 'call_args')
        int_56508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 29), 'int')
        # Storing an element on a container (line 149)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 8), call_args_56507, (int_56508, int_56505))
        
        # Assigning a Call to a Tuple (line 150):
        
        # Assigning a Subscript to a Name (line 150):
        
        # Obtaining the type of the subscript
        int_56509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 8), 'int')
        
        # Call to run(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'solver' (line 151)
        solver_56512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'solver', False)
        # Obtaining the member 'f' of a type (line 151)
        f_56513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), solver_56512, 'f')
        # Getting the type of 'solver' (line 151)
        solver_56514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 22), 'solver', False)
        # Obtaining the member 'jac' of a type (line 151)
        jac_56515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 22), solver_56514, 'jac')
        # Getting the type of 'solver' (line 151)
        solver_56516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'solver', False)
        # Obtaining the member '_y' of a type (line 151)
        _y_56517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 34), solver_56516, '_y')
        # Getting the type of 'solver' (line 151)
        solver_56518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 45), 'solver', False)
        # Obtaining the member 't' of a type (line 151)
        t_56519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 45), solver_56518, 't')
        # Getting the type of 'self' (line 152)
        self_56520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'self', False)
        # Obtaining the member 't_bound' of a type (line 152)
        t_bound_56521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), self_56520, 't_bound')
        # Getting the type of 'solver' (line 152)
        solver_56522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 26), 'solver', False)
        # Obtaining the member 'f_params' of a type (line 152)
        f_params_56523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 26), solver_56522, 'f_params')
        # Getting the type of 'solver' (line 152)
        solver_56524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 43), 'solver', False)
        # Obtaining the member 'jac_params' of a type (line 152)
        jac_params_56525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 43), solver_56524, 'jac_params')
        # Processing the call keyword arguments (line 150)
        kwargs_56526 = {}
        # Getting the type of 'integrator' (line 150)
        integrator_56510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'integrator', False)
        # Obtaining the member 'run' of a type (line 150)
        run_56511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 30), integrator_56510, 'run')
        # Calling run(args, kwargs) (line 150)
        run_call_result_56527 = invoke(stypy.reporting.localization.Localization(__file__, 150, 30), run_56511, *[f_56513, jac_56515, _y_56517, t_56519, t_bound_56521, f_params_56523, jac_params_56525], **kwargs_56526)
        
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___56528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), run_call_result_56527, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_56529 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), getitem___56528, int_56509)
        
        # Assigning a type to the variable 'tuple_var_assignment_56348' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'tuple_var_assignment_56348', subscript_call_result_56529)
        
        # Assigning a Subscript to a Name (line 150):
        
        # Obtaining the type of the subscript
        int_56530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 8), 'int')
        
        # Call to run(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'solver' (line 151)
        solver_56533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'solver', False)
        # Obtaining the member 'f' of a type (line 151)
        f_56534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), solver_56533, 'f')
        # Getting the type of 'solver' (line 151)
        solver_56535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 22), 'solver', False)
        # Obtaining the member 'jac' of a type (line 151)
        jac_56536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 22), solver_56535, 'jac')
        # Getting the type of 'solver' (line 151)
        solver_56537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'solver', False)
        # Obtaining the member '_y' of a type (line 151)
        _y_56538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 34), solver_56537, '_y')
        # Getting the type of 'solver' (line 151)
        solver_56539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 45), 'solver', False)
        # Obtaining the member 't' of a type (line 151)
        t_56540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 45), solver_56539, 't')
        # Getting the type of 'self' (line 152)
        self_56541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'self', False)
        # Obtaining the member 't_bound' of a type (line 152)
        t_bound_56542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), self_56541, 't_bound')
        # Getting the type of 'solver' (line 152)
        solver_56543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 26), 'solver', False)
        # Obtaining the member 'f_params' of a type (line 152)
        f_params_56544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 26), solver_56543, 'f_params')
        # Getting the type of 'solver' (line 152)
        solver_56545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 43), 'solver', False)
        # Obtaining the member 'jac_params' of a type (line 152)
        jac_params_56546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 43), solver_56545, 'jac_params')
        # Processing the call keyword arguments (line 150)
        kwargs_56547 = {}
        # Getting the type of 'integrator' (line 150)
        integrator_56531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'integrator', False)
        # Obtaining the member 'run' of a type (line 150)
        run_56532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 30), integrator_56531, 'run')
        # Calling run(args, kwargs) (line 150)
        run_call_result_56548 = invoke(stypy.reporting.localization.Localization(__file__, 150, 30), run_56532, *[f_56534, jac_56536, _y_56538, t_56540, t_bound_56542, f_params_56544, jac_params_56546], **kwargs_56547)
        
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___56549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), run_call_result_56548, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_56550 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), getitem___56549, int_56530)
        
        # Assigning a type to the variable 'tuple_var_assignment_56349' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'tuple_var_assignment_56349', subscript_call_result_56550)
        
        # Assigning a Name to a Attribute (line 150):
        # Getting the type of 'tuple_var_assignment_56348' (line 150)
        tuple_var_assignment_56348_56551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'tuple_var_assignment_56348')
        # Getting the type of 'solver' (line 150)
        solver_56552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'solver')
        # Setting the type of the member '_y' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), solver_56552, '_y', tuple_var_assignment_56348_56551)
        
        # Assigning a Name to a Attribute (line 150):
        # Getting the type of 'tuple_var_assignment_56349' (line 150)
        tuple_var_assignment_56349_56553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'tuple_var_assignment_56349')
        # Getting the type of 'solver' (line 150)
        solver_56554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 'solver')
        # Setting the type of the member 't' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 19), solver_56554, 't', tuple_var_assignment_56349_56553)
        
        # Assigning a Name to a Subscript (line 153):
        
        # Assigning a Name to a Subscript (line 153):
        # Getting the type of 'itask' (line 153)
        itask_56555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), 'itask')
        # Getting the type of 'integrator' (line 153)
        integrator_56556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'integrator')
        # Obtaining the member 'call_args' of a type (line 153)
        call_args_56557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), integrator_56556, 'call_args')
        int_56558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 29), 'int')
        # Storing an element on a container (line 153)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 8), call_args_56557, (int_56558, itask_56555))
        
        
        # Call to successful(...): (line 155)
        # Processing the call keyword arguments (line 155)
        kwargs_56561 = {}
        # Getting the type of 'solver' (line 155)
        solver_56559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'solver', False)
        # Obtaining the member 'successful' of a type (line 155)
        successful_56560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 11), solver_56559, 'successful')
        # Calling successful(args, kwargs) (line 155)
        successful_call_result_56562 = invoke(stypy.reporting.localization.Localization(__file__, 155, 11), successful_56560, *[], **kwargs_56561)
        
        # Testing the type of an if condition (line 155)
        if_condition_56563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 8), successful_call_result_56562)
        # Assigning a type to the variable 'if_condition_56563' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'if_condition_56563', if_condition_56563)
        # SSA begins for if statement (line 155)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 156):
        
        # Assigning a Attribute to a Attribute (line 156):
        # Getting the type of 'solver' (line 156)
        solver_56564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'solver')
        # Obtaining the member 't' of a type (line 156)
        t_56565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 21), solver_56564, 't')
        # Getting the type of 'self' (line 156)
        self_56566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'self')
        # Setting the type of the member 't' of a type (line 156)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 12), self_56566, 't', t_56565)
        
        # Assigning a Attribute to a Attribute (line 157):
        
        # Assigning a Attribute to a Attribute (line 157):
        # Getting the type of 'solver' (line 157)
        solver_56567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'solver')
        # Obtaining the member '_y' of a type (line 157)
        _y_56568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 21), solver_56567, '_y')
        # Getting the type of 'self' (line 157)
        self_56569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'self')
        # Setting the type of the member 'y' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), self_56569, 'y', _y_56568)
        
        # Assigning a Subscript to a Attribute (line 159):
        
        # Assigning a Subscript to a Attribute (line 159):
        
        # Obtaining the type of the subscript
        int_56570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 41), 'int')
        # Getting the type of 'integrator' (line 159)
        integrator_56571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'integrator')
        # Obtaining the member 'iwork' of a type (line 159)
        iwork_56572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 24), integrator_56571, 'iwork')
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___56573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 24), iwork_56572, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_56574 = invoke(stypy.reporting.localization.Localization(__file__, 159, 24), getitem___56573, int_56570)
        
        # Getting the type of 'self' (line 159)
        self_56575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'self')
        # Setting the type of the member 'njev' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), self_56575, 'njev', subscript_call_result_56574)
        
        # Assigning a Subscript to a Attribute (line 160):
        
        # Assigning a Subscript to a Attribute (line 160):
        
        # Obtaining the type of the subscript
        int_56576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 40), 'int')
        # Getting the type of 'integrator' (line 160)
        integrator_56577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 23), 'integrator')
        # Obtaining the member 'iwork' of a type (line 160)
        iwork_56578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 23), integrator_56577, 'iwork')
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___56579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 23), iwork_56578, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_56580 = invoke(stypy.reporting.localization.Localization(__file__, 160, 23), getitem___56579, int_56576)
        
        # Getting the type of 'self' (line 160)
        self_56581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'self')
        # Setting the type of the member 'nlu' of a type (line 160)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), self_56581, 'nlu', subscript_call_result_56580)
        
        # Obtaining an instance of the builtin type 'tuple' (line 161)
        tuple_56582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 161)
        # Adding element type (line 161)
        # Getting the type of 'True' (line 161)
        True_56583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 19), tuple_56582, True_56583)
        # Adding element type (line 161)
        # Getting the type of 'None' (line 161)
        None_56584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 25), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 19), tuple_56582, None_56584)
        
        # Assigning a type to the variable 'stypy_return_type' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'stypy_return_type', tuple_56582)
        # SSA branch for the else part of an if statement (line 155)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'tuple' (line 163)
        tuple_56585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 163)
        # Adding element type (line 163)
        # Getting the type of 'False' (line 163)
        False_56586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 19), tuple_56585, False_56586)
        # Adding element type (line 163)
        str_56587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 26), 'str', 'Unexpected istate in LSODA.')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 19), tuple_56585, str_56587)
        
        # Assigning a type to the variable 'stypy_return_type' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'stypy_return_type', tuple_56585)
        # SSA join for if statement (line 155)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_step_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_step_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 142)
        stypy_return_type_56588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56588)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_step_impl'
        return stypy_return_type_56588


    @norecursion
    def _dense_output_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dense_output_impl'
        module_type_store = module_type_store.open_function_context('_dense_output_impl', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LSODA._dense_output_impl.__dict__.__setitem__('stypy_localization', localization)
        LSODA._dense_output_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LSODA._dense_output_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        LSODA._dense_output_impl.__dict__.__setitem__('stypy_function_name', 'LSODA._dense_output_impl')
        LSODA._dense_output_impl.__dict__.__setitem__('stypy_param_names_list', [])
        LSODA._dense_output_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        LSODA._dense_output_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LSODA._dense_output_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        LSODA._dense_output_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        LSODA._dense_output_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LSODA._dense_output_impl.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LSODA._dense_output_impl', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_dense_output_impl', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_dense_output_impl(...)' code ##################

        
        # Assigning a Attribute to a Name (line 166):
        
        # Assigning a Attribute to a Name (line 166):
        # Getting the type of 'self' (line 166)
        self_56589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'self')
        # Obtaining the member '_lsoda_solver' of a type (line 166)
        _lsoda_solver_56590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 16), self_56589, '_lsoda_solver')
        # Obtaining the member '_integrator' of a type (line 166)
        _integrator_56591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 16), _lsoda_solver_56590, '_integrator')
        # Obtaining the member 'iwork' of a type (line 166)
        iwork_56592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 16), _integrator_56591, 'iwork')
        # Assigning a type to the variable 'iwork' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'iwork', iwork_56592)
        
        # Assigning a Attribute to a Name (line 167):
        
        # Assigning a Attribute to a Name (line 167):
        # Getting the type of 'self' (line 167)
        self_56593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'self')
        # Obtaining the member '_lsoda_solver' of a type (line 167)
        _lsoda_solver_56594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 16), self_56593, '_lsoda_solver')
        # Obtaining the member '_integrator' of a type (line 167)
        _integrator_56595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 16), _lsoda_solver_56594, '_integrator')
        # Obtaining the member 'rwork' of a type (line 167)
        rwork_56596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 16), _integrator_56595, 'rwork')
        # Assigning a type to the variable 'rwork' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'rwork', rwork_56596)
        
        # Assigning a Subscript to a Name (line 169):
        
        # Assigning a Subscript to a Name (line 169):
        
        # Obtaining the type of the subscript
        int_56597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 22), 'int')
        # Getting the type of 'iwork' (line 169)
        iwork_56598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'iwork')
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___56599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 16), iwork_56598, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_56600 = invoke(stypy.reporting.localization.Localization(__file__, 169, 16), getitem___56599, int_56597)
        
        # Assigning a type to the variable 'order' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'order', subscript_call_result_56600)
        
        # Assigning a Subscript to a Name (line 170):
        
        # Assigning a Subscript to a Name (line 170):
        
        # Obtaining the type of the subscript
        int_56601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 18), 'int')
        # Getting the type of 'rwork' (line 170)
        rwork_56602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'rwork')
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___56603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 12), rwork_56602, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_56604 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), getitem___56603, int_56601)
        
        # Assigning a type to the variable 'h' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'h', subscript_call_result_56604)
        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Call to copy(...): (line 171)
        # Processing the call keyword arguments (line 171)
        kwargs_56631 = {}
        
        # Call to reshape(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Obtaining the type of the subscript
        int_56607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'int')
        int_56608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 33), 'int')
        # Getting the type of 'order' (line 171)
        order_56609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 39), 'order', False)
        int_56610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 47), 'int')
        # Applying the binary operator '+' (line 171)
        result_add_56611 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 39), '+', order_56609, int_56610)
        
        # Getting the type of 'self' (line 171)
        self_56612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 52), 'self', False)
        # Obtaining the member 'n' of a type (line 171)
        n_56613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 52), self_56612, 'n')
        # Applying the binary operator '*' (line 171)
        result_mul_56614 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 38), '*', result_add_56611, n_56613)
        
        # Applying the binary operator '+' (line 171)
        result_add_56615 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 33), '+', int_56608, result_mul_56614)
        
        slice_56616 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 171, 24), int_56607, result_add_56615, None)
        # Getting the type of 'rwork' (line 171)
        rwork_56617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 24), 'rwork', False)
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___56618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 24), rwork_56617, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_56619 = invoke(stypy.reporting.localization.Localization(__file__, 171, 24), getitem___56618, slice_56616)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 172)
        tuple_56620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 172)
        # Adding element type (line 172)
        # Getting the type of 'self' (line 172)
        self_56621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 'self', False)
        # Obtaining the member 'n' of a type (line 172)
        n_56622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), self_56621, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 25), tuple_56620, n_56622)
        # Adding element type (line 172)
        # Getting the type of 'order' (line 172)
        order_56623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 33), 'order', False)
        int_56624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 41), 'int')
        # Applying the binary operator '+' (line 172)
        result_add_56625 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 33), '+', order_56623, int_56624)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 25), tuple_56620, result_add_56625)
        
        # Processing the call keyword arguments (line 171)
        str_56626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 51), 'str', 'F')
        keyword_56627 = str_56626
        kwargs_56628 = {'order': keyword_56627}
        # Getting the type of 'np' (line 171)
        np_56605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 13), 'np', False)
        # Obtaining the member 'reshape' of a type (line 171)
        reshape_56606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 13), np_56605, 'reshape')
        # Calling reshape(args, kwargs) (line 171)
        reshape_call_result_56629 = invoke(stypy.reporting.localization.Localization(__file__, 171, 13), reshape_56606, *[subscript_call_result_56619, tuple_56620], **kwargs_56628)
        
        # Obtaining the member 'copy' of a type (line 171)
        copy_56630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 13), reshape_call_result_56629, 'copy')
        # Calling copy(args, kwargs) (line 171)
        copy_call_result_56632 = invoke(stypy.reporting.localization.Localization(__file__, 171, 13), copy_56630, *[], **kwargs_56631)
        
        # Assigning a type to the variable 'yh' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'yh', copy_call_result_56632)
        
        # Call to LsodaDenseOutput(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'self' (line 174)
        self_56634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'self', False)
        # Obtaining the member 't_old' of a type (line 174)
        t_old_56635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 32), self_56634, 't_old')
        # Getting the type of 'self' (line 174)
        self_56636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 44), 'self', False)
        # Obtaining the member 't' of a type (line 174)
        t_56637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 44), self_56636, 't')
        # Getting the type of 'h' (line 174)
        h_56638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 52), 'h', False)
        # Getting the type of 'order' (line 174)
        order_56639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 55), 'order', False)
        # Getting the type of 'yh' (line 174)
        yh_56640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 62), 'yh', False)
        # Processing the call keyword arguments (line 174)
        kwargs_56641 = {}
        # Getting the type of 'LsodaDenseOutput' (line 174)
        LsodaDenseOutput_56633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'LsodaDenseOutput', False)
        # Calling LsodaDenseOutput(args, kwargs) (line 174)
        LsodaDenseOutput_call_result_56642 = invoke(stypy.reporting.localization.Localization(__file__, 174, 15), LsodaDenseOutput_56633, *[t_old_56635, t_56637, h_56638, order_56639, yh_56640], **kwargs_56641)
        
        # Assigning a type to the variable 'stypy_return_type' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'stypy_return_type', LsodaDenseOutput_call_result_56642)
        
        # ################# End of '_dense_output_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dense_output_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 165)
        stypy_return_type_56643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56643)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dense_output_impl'
        return stypy_return_type_56643


# Assigning a type to the variable 'LSODA' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'LSODA', LSODA)
# Declaration of the 'LsodaDenseOutput' class
# Getting the type of 'DenseOutput' (line 177)
DenseOutput_56644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 23), 'DenseOutput')

class LsodaDenseOutput(DenseOutput_56644, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LsodaDenseOutput.__init__', ['t_old', 't', 'h', 'order', 'yh'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['t_old', 't', 'h', 'order', 'yh'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 't_old' (line 179)
        t_old_56651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 47), 't_old', False)
        # Getting the type of 't' (line 179)
        t_56652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 54), 't', False)
        # Processing the call keyword arguments (line 179)
        kwargs_56653 = {}
        
        # Call to super(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'LsodaDenseOutput' (line 179)
        LsodaDenseOutput_56646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 14), 'LsodaDenseOutput', False)
        # Getting the type of 'self' (line 179)
        self_56647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 32), 'self', False)
        # Processing the call keyword arguments (line 179)
        kwargs_56648 = {}
        # Getting the type of 'super' (line 179)
        super_56645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'super', False)
        # Calling super(args, kwargs) (line 179)
        super_call_result_56649 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), super_56645, *[LsodaDenseOutput_56646, self_56647], **kwargs_56648)
        
        # Obtaining the member '__init__' of a type (line 179)
        init___56650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), super_call_result_56649, '__init__')
        # Calling __init__(args, kwargs) (line 179)
        init___call_result_56654 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), init___56650, *[t_old_56651, t_56652], **kwargs_56653)
        
        
        # Assigning a Name to a Attribute (line 180):
        
        # Assigning a Name to a Attribute (line 180):
        # Getting the type of 'h' (line 180)
        h_56655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'h')
        # Getting the type of 'self' (line 180)
        self_56656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'self')
        # Setting the type of the member 'h' of a type (line 180)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), self_56656, 'h', h_56655)
        
        # Assigning a Name to a Attribute (line 181):
        
        # Assigning a Name to a Attribute (line 181):
        # Getting the type of 'yh' (line 181)
        yh_56657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 18), 'yh')
        # Getting the type of 'self' (line 181)
        self_56658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self')
        # Setting the type of the member 'yh' of a type (line 181)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_56658, 'yh', yh_56657)
        
        # Assigning a Call to a Attribute (line 182):
        
        # Assigning a Call to a Attribute (line 182):
        
        # Call to arange(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'order' (line 182)
        order_56661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 27), 'order', False)
        int_56662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 35), 'int')
        # Applying the binary operator '+' (line 182)
        result_add_56663 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 27), '+', order_56661, int_56662)
        
        # Processing the call keyword arguments (line 182)
        kwargs_56664 = {}
        # Getting the type of 'np' (line 182)
        np_56659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 182)
        arange_56660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 17), np_56659, 'arange')
        # Calling arange(args, kwargs) (line 182)
        arange_call_result_56665 = invoke(stypy.reporting.localization.Localization(__file__, 182, 17), arange_56660, *[result_add_56663], **kwargs_56664)
        
        # Getting the type of 'self' (line 182)
        self_56666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'self')
        # Setting the type of the member 'p' of a type (line 182)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), self_56666, 'p', arange_call_result_56665)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _call_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_call_impl'
        module_type_store = module_type_store.open_function_context('_call_impl', 184, 4, False)
        # Assigning a type to the variable 'self' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LsodaDenseOutput._call_impl.__dict__.__setitem__('stypy_localization', localization)
        LsodaDenseOutput._call_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LsodaDenseOutput._call_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        LsodaDenseOutput._call_impl.__dict__.__setitem__('stypy_function_name', 'LsodaDenseOutput._call_impl')
        LsodaDenseOutput._call_impl.__dict__.__setitem__('stypy_param_names_list', ['t'])
        LsodaDenseOutput._call_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        LsodaDenseOutput._call_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LsodaDenseOutput._call_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        LsodaDenseOutput._call_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        LsodaDenseOutput._call_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LsodaDenseOutput._call_impl.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LsodaDenseOutput._call_impl', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_call_impl', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_call_impl(...)' code ##################

        
        
        # Getting the type of 't' (line 185)
        t_56667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 't')
        # Obtaining the member 'ndim' of a type (line 185)
        ndim_56668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 11), t_56667, 'ndim')
        int_56669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 21), 'int')
        # Applying the binary operator '==' (line 185)
        result_eq_56670 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 11), '==', ndim_56668, int_56669)
        
        # Testing the type of an if condition (line 185)
        if_condition_56671 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 8), result_eq_56670)
        # Assigning a type to the variable 'if_condition_56671' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'if_condition_56671', if_condition_56671)
        # SSA begins for if statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 186):
        
        # Assigning a BinOp to a Name (line 186):
        # Getting the type of 't' (line 186)
        t_56672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 18), 't')
        # Getting the type of 'self' (line 186)
        self_56673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 22), 'self')
        # Obtaining the member 't' of a type (line 186)
        t_56674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 22), self_56673, 't')
        # Applying the binary operator '-' (line 186)
        result_sub_56675 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 18), '-', t_56672, t_56674)
        
        # Getting the type of 'self' (line 186)
        self_56676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 32), 'self')
        # Obtaining the member 'h' of a type (line 186)
        h_56677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 32), self_56676, 'h')
        # Applying the binary operator 'div' (line 186)
        result_div_56678 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 17), 'div', result_sub_56675, h_56677)
        
        # Getting the type of 'self' (line 186)
        self_56679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 43), 'self')
        # Obtaining the member 'p' of a type (line 186)
        p_56680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 43), self_56679, 'p')
        # Applying the binary operator '**' (line 186)
        result_pow_56681 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 16), '**', result_div_56678, p_56680)
        
        # Assigning a type to the variable 'x' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'x', result_pow_56681)
        # SSA branch for the else part of an if statement (line 185)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 188):
        
        # Assigning a BinOp to a Name (line 188):
        # Getting the type of 't' (line 188)
        t_56682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 18), 't')
        # Getting the type of 'self' (line 188)
        self_56683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 22), 'self')
        # Obtaining the member 't' of a type (line 188)
        t_56684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 22), self_56683, 't')
        # Applying the binary operator '-' (line 188)
        result_sub_56685 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 18), '-', t_56682, t_56684)
        
        # Getting the type of 'self' (line 188)
        self_56686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 32), 'self')
        # Obtaining the member 'h' of a type (line 188)
        h_56687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 32), self_56686, 'h')
        # Applying the binary operator 'div' (line 188)
        result_div_56688 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 17), 'div', result_sub_56685, h_56687)
        
        
        # Obtaining the type of the subscript
        slice_56689 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 188, 43), None, None, None)
        # Getting the type of 'None' (line 188)
        None_56690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 53), 'None')
        # Getting the type of 'self' (line 188)
        self_56691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 43), 'self')
        # Obtaining the member 'p' of a type (line 188)
        p_56692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 43), self_56691, 'p')
        # Obtaining the member '__getitem__' of a type (line 188)
        getitem___56693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 43), p_56692, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 188)
        subscript_call_result_56694 = invoke(stypy.reporting.localization.Localization(__file__, 188, 43), getitem___56693, (slice_56689, None_56690))
        
        # Applying the binary operator '**' (line 188)
        result_pow_56695 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 16), '**', result_div_56688, subscript_call_result_56694)
        
        # Assigning a type to the variable 'x' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'x', result_pow_56695)
        # SSA join for if statement (line 185)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to dot(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'self' (line 190)
        self_56698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 22), 'self', False)
        # Obtaining the member 'yh' of a type (line 190)
        yh_56699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 22), self_56698, 'yh')
        # Getting the type of 'x' (line 190)
        x_56700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 31), 'x', False)
        # Processing the call keyword arguments (line 190)
        kwargs_56701 = {}
        # Getting the type of 'np' (line 190)
        np_56696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 15), 'np', False)
        # Obtaining the member 'dot' of a type (line 190)
        dot_56697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 15), np_56696, 'dot')
        # Calling dot(args, kwargs) (line 190)
        dot_call_result_56702 = invoke(stypy.reporting.localization.Localization(__file__, 190, 15), dot_56697, *[yh_56699, x_56700], **kwargs_56701)
        
        # Assigning a type to the variable 'stypy_return_type' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'stypy_return_type', dot_call_result_56702)
        
        # ################# End of '_call_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_call_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_56703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56703)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_call_impl'
        return stypy_return_type_56703


# Assigning a type to the variable 'LsodaDenseOutput' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'LsodaDenseOutput', LsodaDenseOutput)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
