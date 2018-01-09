
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Author: Travis Oliphant
2: from __future__ import division, print_function, absolute_import
3: 
4: __all__ = ['odeint']
5: 
6: from . import _odepack
7: from copy import copy
8: import warnings
9: 
10: class ODEintWarning(Warning):
11:     pass
12: 
13: _msgs = {2: "Integration successful.",
14:          1: "Nothing was done; the integration time was 0.",
15:          -1: "Excess work done on this call (perhaps wrong Dfun type).",
16:          -2: "Excess accuracy requested (tolerances too small).",
17:          -3: "Illegal input detected (internal error).",
18:          -4: "Repeated error test failures (internal error).",
19:          -5: "Repeated convergence failures (perhaps bad Jacobian or tolerances).",
20:          -6: "Error weight became zero during problem.",
21:          -7: "Internal workspace insufficient to finish (internal error)."
22:          }
23: 
24: 
25: def odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0,
26:            ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0,
27:            hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12,
28:            mxords=5, printmessg=0):
29:     '''
30:     Integrate a system of ordinary differential equations.
31: 
32:     Solve a system of ordinary differential equations using lsoda from the
33:     FORTRAN library odepack.
34: 
35:     Solves the initial value problem for stiff or non-stiff systems
36:     of first order ode-s::
37: 
38:         dy/dt = func(y, t0, ...)
39: 
40:     where y can be a vector.
41: 
42:     *Note*: The first two arguments of ``func(y, t0, ...)`` are in the
43:     opposite order of the arguments in the system definition function used
44:     by the `scipy.integrate.ode` class.
45: 
46:     Parameters
47:     ----------
48:     func : callable(y, t0, ...)
49:         Computes the derivative of y at t0.
50:     y0 : array
51:         Initial condition on y (can be a vector).
52:     t : array
53:         A sequence of time points for which to solve for y.  The initial
54:         value point should be the first element of this sequence.
55:     args : tuple, optional
56:         Extra arguments to pass to function.
57:     Dfun : callable(y, t0, ...)
58:         Gradient (Jacobian) of `func`.
59:     col_deriv : bool, optional
60:         True if `Dfun` defines derivatives down columns (faster),
61:         otherwise `Dfun` should define derivatives across rows.
62:     full_output : bool, optional
63:         True if to return a dictionary of optional outputs as the second output
64:     printmessg : bool, optional
65:         Whether to print the convergence message
66: 
67:     Returns
68:     -------
69:     y : array, shape (len(t), len(y0))
70:         Array containing the value of y for each desired time in t,
71:         with the initial value `y0` in the first row.
72:     infodict : dict, only returned if full_output == True
73:         Dictionary containing additional output information
74: 
75:         =======  ============================================================
76:         key      meaning
77:         =======  ============================================================
78:         'hu'     vector of step sizes successfully used for each time step.
79:         'tcur'   vector with the value of t reached for each time step.
80:                  (will always be at least as large as the input times).
81:         'tolsf'  vector of tolerance scale factors, greater than 1.0,
82:                  computed when a request for too much accuracy was detected.
83:         'tsw'    value of t at the time of the last method switch
84:                  (given for each time step)
85:         'nst'    cumulative number of time steps
86:         'nfe'    cumulative number of function evaluations for each time step
87:         'nje'    cumulative number of jacobian evaluations for each time step
88:         'nqu'    a vector of method orders for each successful step.
89:         'imxer'  index of the component of largest magnitude in the
90:                  weighted local error vector (e / ewt) on an error return, -1
91:                  otherwise.
92:         'lenrw'  the length of the double work array required.
93:         'leniw'  the length of integer work array required.
94:         'mused'  a vector of method indicators for each successful time step:
95:                  1: adams (nonstiff), 2: bdf (stiff)
96:         =======  ============================================================
97: 
98:     Other Parameters
99:     ----------------
100:     ml, mu : int, optional
101:         If either of these are not None or non-negative, then the
102:         Jacobian is assumed to be banded.  These give the number of
103:         lower and upper non-zero diagonals in this banded matrix.
104:         For the banded case, `Dfun` should return a matrix whose
105:         rows contain the non-zero bands (starting with the lowest diagonal).
106:         Thus, the return matrix `jac` from `Dfun` should have shape
107:         ``(ml + mu + 1, len(y0))`` when ``ml >=0`` or ``mu >=0``.
108:         The data in `jac` must be stored such that ``jac[i - j + mu, j]``
109:         holds the derivative of the `i`th equation with respect to the `j`th
110:         state variable.  If `col_deriv` is True, the transpose of this
111:         `jac` must be returned.
112:     rtol, atol : float, optional
113:         The input parameters `rtol` and `atol` determine the error
114:         control performed by the solver.  The solver will control the
115:         vector, e, of estimated local errors in y, according to an
116:         inequality of the form ``max-norm of (e / ewt) <= 1``,
117:         where ewt is a vector of positive error weights computed as
118:         ``ewt = rtol * abs(y) + atol``.
119:         rtol and atol can be either vectors the same length as y or scalars.
120:         Defaults to 1.49012e-8.
121:     tcrit : ndarray, optional
122:         Vector of critical points (e.g. singularities) where integration
123:         care should be taken.
124:     h0 : float, (0: solver-determined), optional
125:         The step size to be attempted on the first step.
126:     hmax : float, (0: solver-determined), optional
127:         The maximum absolute step size allowed.
128:     hmin : float, (0: solver-determined), optional
129:         The minimum absolute step size allowed.
130:     ixpr : bool, optional
131:         Whether to generate extra printing at method switches.
132:     mxstep : int, (0: solver-determined), optional
133:         Maximum number of (internally defined) steps allowed for each
134:         integration point in t.
135:     mxhnil : int, (0: solver-determined), optional
136:         Maximum number of messages printed.
137:     mxordn : int, (0: solver-determined), optional
138:         Maximum order to be allowed for the non-stiff (Adams) method.
139:     mxords : int, (0: solver-determined), optional
140:         Maximum order to be allowed for the stiff (BDF) method.
141: 
142:     See Also
143:     --------
144:     ode : a more object-oriented integrator based on VODE.
145:     quad : for finding the area under a curve.
146: 
147:     Examples
148:     --------
149:     The second order differential equation for the angle `theta` of a
150:     pendulum acted on by gravity with friction can be written::
151: 
152:         theta''(t) + b*theta'(t) + c*sin(theta(t)) = 0
153: 
154:     where `b` and `c` are positive constants, and a prime (') denotes a
155:     derivative.  To solve this equation with `odeint`, we must first convert
156:     it to a system of first order equations.  By defining the angular
157:     velocity ``omega(t) = theta'(t)``, we obtain the system::
158: 
159:         theta'(t) = omega(t)
160:         omega'(t) = -b*omega(t) - c*sin(theta(t))
161: 
162:     Let `y` be the vector [`theta`, `omega`].  We implement this system
163:     in python as:
164: 
165:     >>> def pend(y, t, b, c):
166:     ...     theta, omega = y
167:     ...     dydt = [omega, -b*omega - c*np.sin(theta)]
168:     ...     return dydt
169:     ...
170:     
171:     We assume the constants are `b` = 0.25 and `c` = 5.0:
172: 
173:     >>> b = 0.25
174:     >>> c = 5.0
175: 
176:     For initial conditions, we assume the pendulum is nearly vertical
177:     with `theta(0)` = `pi` - 0.1, and it initially at rest, so
178:     `omega(0)` = 0.  Then the vector of initial conditions is
179: 
180:     >>> y0 = [np.pi - 0.1, 0.0]
181: 
182:     We generate a solution 101 evenly spaced samples in the interval
183:     0 <= `t` <= 10.  So our array of times is:
184: 
185:     >>> t = np.linspace(0, 10, 101)
186: 
187:     Call `odeint` to generate the solution.  To pass the parameters
188:     `b` and `c` to `pend`, we give them to `odeint` using the `args`
189:     argument.
190: 
191:     >>> from scipy.integrate import odeint
192:     >>> sol = odeint(pend, y0, t, args=(b, c))
193: 
194:     The solution is an array with shape (101, 2).  The first column
195:     is `theta(t)`, and the second is `omega(t)`.  The following code
196:     plots both components.
197: 
198:     >>> import matplotlib.pyplot as plt
199:     >>> plt.plot(t, sol[:, 0], 'b', label='theta(t)')
200:     >>> plt.plot(t, sol[:, 1], 'g', label='omega(t)')
201:     >>> plt.legend(loc='best')
202:     >>> plt.xlabel('t')
203:     >>> plt.grid()
204:     >>> plt.show()
205:     '''
206: 
207:     if ml is None:
208:         ml = -1  # changed to zero inside function call
209:     if mu is None:
210:         mu = -1  # changed to zero inside function call
211:     t = copy(t)
212:     y0 = copy(y0)
213:     output = _odepack.odeint(func, y0, t, args, Dfun, col_deriv, ml, mu,
214:                              full_output, rtol, atol, tcrit, h0, hmax, hmin,
215:                              ixpr, mxstep, mxhnil, mxordn, mxords)
216:     if output[-1] < 0:
217:         warning_msg = _msgs[output[-1]] + " Run with full_output = 1 to get quantitative information."
218:         warnings.warn(warning_msg, ODEintWarning)
219:     elif printmessg:
220:         warning_msg = _msgs[output[-1]]
221:         warnings.warn(warning_msg, ODEintWarning)
222: 
223:     if full_output:
224:         output[1]['message'] = _msgs[output[-1]]
225: 
226:     output = output[:-1]
227:     if len(output) == 1:
228:         return output[0]
229:     else:
230:         return output
231: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 4):
__all__ = ['odeint']
module_type_store.set_exportable_members(['odeint'])

# Obtaining an instance of the builtin type 'list' (line 4)
list_28884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 4)
# Adding element type (line 4)
str_28885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'str', 'odeint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 10), list_28884, str_28885)

# Assigning a type to the variable '__all__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__all__', list_28884)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.integrate import _odepack' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_28886 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.integrate')

if (type(import_28886) is not StypyTypeError):

    if (import_28886 != 'pyd_module'):
        __import__(import_28886)
        sys_modules_28887 = sys.modules[import_28886]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.integrate', sys_modules_28887.module_type_store, module_type_store, ['_odepack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_28887, sys_modules_28887.module_type_store, module_type_store)
    else:
        from scipy.integrate import _odepack

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.integrate', None, module_type_store, ['_odepack'], [_odepack])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.integrate', import_28886)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from copy import copy' statement (line 7)
try:
    from copy import copy

except:
    copy = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'copy', None, module_type_store, ['copy'], [copy])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import warnings' statement (line 8)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'warnings', warnings, module_type_store)

# Declaration of the 'ODEintWarning' class
# Getting the type of 'Warning' (line 10)
Warning_28888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 20), 'Warning')

class ODEintWarning(Warning_28888, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 0, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ODEintWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ODEintWarning' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'ODEintWarning', ODEintWarning)

# Assigning a Dict to a Name (line 13):

# Obtaining an instance of the builtin type 'dict' (line 13)
dict_28889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 8), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 13)
# Adding element type (key, value) (line 13)
int_28890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 9), 'int')
str_28891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'str', 'Integration successful.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 8), dict_28889, (int_28890, str_28891))
# Adding element type (key, value) (line 13)
int_28892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 9), 'int')
str_28893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'str', 'Nothing was done; the integration time was 0.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 8), dict_28889, (int_28892, str_28893))
# Adding element type (key, value) (line 13)
int_28894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 9), 'int')
str_28895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'str', 'Excess work done on this call (perhaps wrong Dfun type).')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 8), dict_28889, (int_28894, str_28895))
# Adding element type (key, value) (line 13)
int_28896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 9), 'int')
str_28897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'str', 'Excess accuracy requested (tolerances too small).')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 8), dict_28889, (int_28896, str_28897))
# Adding element type (key, value) (line 13)
int_28898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 9), 'int')
str_28899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'str', 'Illegal input detected (internal error).')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 8), dict_28889, (int_28898, str_28899))
# Adding element type (key, value) (line 13)
int_28900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 9), 'int')
str_28901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 13), 'str', 'Repeated error test failures (internal error).')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 8), dict_28889, (int_28900, str_28901))
# Adding element type (key, value) (line 13)
int_28902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 9), 'int')
str_28903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 13), 'str', 'Repeated convergence failures (perhaps bad Jacobian or tolerances).')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 8), dict_28889, (int_28902, str_28903))
# Adding element type (key, value) (line 13)
int_28904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 9), 'int')
str_28905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 13), 'str', 'Error weight became zero during problem.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 8), dict_28889, (int_28904, str_28905))
# Adding element type (key, value) (line 13)
int_28906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'int')
str_28907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 13), 'str', 'Internal workspace insufficient to finish (internal error).')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 8), dict_28889, (int_28906, str_28907))

# Assigning a type to the variable '_msgs' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '_msgs', dict_28889)

@norecursion
def odeint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 25)
    tuple_28908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 25)
    
    # Getting the type of 'None' (line 25)
    None_28909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 38), 'None')
    int_28910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 54), 'int')
    int_28911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 69), 'int')
    # Getting the type of 'None' (line 26)
    None_28912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'None')
    # Getting the type of 'None' (line 26)
    None_28913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'None')
    # Getting the type of 'None' (line 26)
    None_28914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 34), 'None')
    # Getting the type of 'None' (line 26)
    None_28915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 45), 'None')
    # Getting the type of 'None' (line 26)
    None_28916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 57), 'None')
    float_28917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 66), 'float')
    float_28918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'float')
    float_28919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'float')
    int_28920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 36), 'int')
    int_28921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 46), 'int')
    int_28922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 56), 'int')
    int_28923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 66), 'int')
    int_28924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 18), 'int')
    int_28925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 32), 'int')
    defaults = [tuple_28908, None_28909, int_28910, int_28911, None_28912, None_28913, None_28914, None_28915, None_28916, float_28917, float_28918, float_28919, int_28920, int_28921, int_28922, int_28923, int_28924, int_28925]
    # Create a new context for function 'odeint'
    module_type_store = module_type_store.open_function_context('odeint', 25, 0, False)
    
    # Passed parameters checking function
    odeint.stypy_localization = localization
    odeint.stypy_type_of_self = None
    odeint.stypy_type_store = module_type_store
    odeint.stypy_function_name = 'odeint'
    odeint.stypy_param_names_list = ['func', 'y0', 't', 'args', 'Dfun', 'col_deriv', 'full_output', 'ml', 'mu', 'rtol', 'atol', 'tcrit', 'h0', 'hmax', 'hmin', 'ixpr', 'mxstep', 'mxhnil', 'mxordn', 'mxords', 'printmessg']
    odeint.stypy_varargs_param_name = None
    odeint.stypy_kwargs_param_name = None
    odeint.stypy_call_defaults = defaults
    odeint.stypy_call_varargs = varargs
    odeint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'odeint', ['func', 'y0', 't', 'args', 'Dfun', 'col_deriv', 'full_output', 'ml', 'mu', 'rtol', 'atol', 'tcrit', 'h0', 'hmax', 'hmin', 'ixpr', 'mxstep', 'mxhnil', 'mxordn', 'mxords', 'printmessg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'odeint', localization, ['func', 'y0', 't', 'args', 'Dfun', 'col_deriv', 'full_output', 'ml', 'mu', 'rtol', 'atol', 'tcrit', 'h0', 'hmax', 'hmin', 'ixpr', 'mxstep', 'mxhnil', 'mxordn', 'mxords', 'printmessg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'odeint(...)' code ##################

    str_28926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, (-1)), 'str', "\n    Integrate a system of ordinary differential equations.\n\n    Solve a system of ordinary differential equations using lsoda from the\n    FORTRAN library odepack.\n\n    Solves the initial value problem for stiff or non-stiff systems\n    of first order ode-s::\n\n        dy/dt = func(y, t0, ...)\n\n    where y can be a vector.\n\n    *Note*: The first two arguments of ``func(y, t0, ...)`` are in the\n    opposite order of the arguments in the system definition function used\n    by the `scipy.integrate.ode` class.\n\n    Parameters\n    ----------\n    func : callable(y, t0, ...)\n        Computes the derivative of y at t0.\n    y0 : array\n        Initial condition on y (can be a vector).\n    t : array\n        A sequence of time points for which to solve for y.  The initial\n        value point should be the first element of this sequence.\n    args : tuple, optional\n        Extra arguments to pass to function.\n    Dfun : callable(y, t0, ...)\n        Gradient (Jacobian) of `func`.\n    col_deriv : bool, optional\n        True if `Dfun` defines derivatives down columns (faster),\n        otherwise `Dfun` should define derivatives across rows.\n    full_output : bool, optional\n        True if to return a dictionary of optional outputs as the second output\n    printmessg : bool, optional\n        Whether to print the convergence message\n\n    Returns\n    -------\n    y : array, shape (len(t), len(y0))\n        Array containing the value of y for each desired time in t,\n        with the initial value `y0` in the first row.\n    infodict : dict, only returned if full_output == True\n        Dictionary containing additional output information\n\n        =======  ============================================================\n        key      meaning\n        =======  ============================================================\n        'hu'     vector of step sizes successfully used for each time step.\n        'tcur'   vector with the value of t reached for each time step.\n                 (will always be at least as large as the input times).\n        'tolsf'  vector of tolerance scale factors, greater than 1.0,\n                 computed when a request for too much accuracy was detected.\n        'tsw'    value of t at the time of the last method switch\n                 (given for each time step)\n        'nst'    cumulative number of time steps\n        'nfe'    cumulative number of function evaluations for each time step\n        'nje'    cumulative number of jacobian evaluations for each time step\n        'nqu'    a vector of method orders for each successful step.\n        'imxer'  index of the component of largest magnitude in the\n                 weighted local error vector (e / ewt) on an error return, -1\n                 otherwise.\n        'lenrw'  the length of the double work array required.\n        'leniw'  the length of integer work array required.\n        'mused'  a vector of method indicators for each successful time step:\n                 1: adams (nonstiff), 2: bdf (stiff)\n        =======  ============================================================\n\n    Other Parameters\n    ----------------\n    ml, mu : int, optional\n        If either of these are not None or non-negative, then the\n        Jacobian is assumed to be banded.  These give the number of\n        lower and upper non-zero diagonals in this banded matrix.\n        For the banded case, `Dfun` should return a matrix whose\n        rows contain the non-zero bands (starting with the lowest diagonal).\n        Thus, the return matrix `jac` from `Dfun` should have shape\n        ``(ml + mu + 1, len(y0))`` when ``ml >=0`` or ``mu >=0``.\n        The data in `jac` must be stored such that ``jac[i - j + mu, j]``\n        holds the derivative of the `i`th equation with respect to the `j`th\n        state variable.  If `col_deriv` is True, the transpose of this\n        `jac` must be returned.\n    rtol, atol : float, optional\n        The input parameters `rtol` and `atol` determine the error\n        control performed by the solver.  The solver will control the\n        vector, e, of estimated local errors in y, according to an\n        inequality of the form ``max-norm of (e / ewt) <= 1``,\n        where ewt is a vector of positive error weights computed as\n        ``ewt = rtol * abs(y) + atol``.\n        rtol and atol can be either vectors the same length as y or scalars.\n        Defaults to 1.49012e-8.\n    tcrit : ndarray, optional\n        Vector of critical points (e.g. singularities) where integration\n        care should be taken.\n    h0 : float, (0: solver-determined), optional\n        The step size to be attempted on the first step.\n    hmax : float, (0: solver-determined), optional\n        The maximum absolute step size allowed.\n    hmin : float, (0: solver-determined), optional\n        The minimum absolute step size allowed.\n    ixpr : bool, optional\n        Whether to generate extra printing at method switches.\n    mxstep : int, (0: solver-determined), optional\n        Maximum number of (internally defined) steps allowed for each\n        integration point in t.\n    mxhnil : int, (0: solver-determined), optional\n        Maximum number of messages printed.\n    mxordn : int, (0: solver-determined), optional\n        Maximum order to be allowed for the non-stiff (Adams) method.\n    mxords : int, (0: solver-determined), optional\n        Maximum order to be allowed for the stiff (BDF) method.\n\n    See Also\n    --------\n    ode : a more object-oriented integrator based on VODE.\n    quad : for finding the area under a curve.\n\n    Examples\n    --------\n    The second order differential equation for the angle `theta` of a\n    pendulum acted on by gravity with friction can be written::\n\n        theta''(t) + b*theta'(t) + c*sin(theta(t)) = 0\n\n    where `b` and `c` are positive constants, and a prime (') denotes a\n    derivative.  To solve this equation with `odeint`, we must first convert\n    it to a system of first order equations.  By defining the angular\n    velocity ``omega(t) = theta'(t)``, we obtain the system::\n\n        theta'(t) = omega(t)\n        omega'(t) = -b*omega(t) - c*sin(theta(t))\n\n    Let `y` be the vector [`theta`, `omega`].  We implement this system\n    in python as:\n\n    >>> def pend(y, t, b, c):\n    ...     theta, omega = y\n    ...     dydt = [omega, -b*omega - c*np.sin(theta)]\n    ...     return dydt\n    ...\n    \n    We assume the constants are `b` = 0.25 and `c` = 5.0:\n\n    >>> b = 0.25\n    >>> c = 5.0\n\n    For initial conditions, we assume the pendulum is nearly vertical\n    with `theta(0)` = `pi` - 0.1, and it initially at rest, so\n    `omega(0)` = 0.  Then the vector of initial conditions is\n\n    >>> y0 = [np.pi - 0.1, 0.0]\n\n    We generate a solution 101 evenly spaced samples in the interval\n    0 <= `t` <= 10.  So our array of times is:\n\n    >>> t = np.linspace(0, 10, 101)\n\n    Call `odeint` to generate the solution.  To pass the parameters\n    `b` and `c` to `pend`, we give them to `odeint` using the `args`\n    argument.\n\n    >>> from scipy.integrate import odeint\n    >>> sol = odeint(pend, y0, t, args=(b, c))\n\n    The solution is an array with shape (101, 2).  The first column\n    is `theta(t)`, and the second is `omega(t)`.  The following code\n    plots both components.\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(t, sol[:, 0], 'b', label='theta(t)')\n    >>> plt.plot(t, sol[:, 1], 'g', label='omega(t)')\n    >>> plt.legend(loc='best')\n    >>> plt.xlabel('t')\n    >>> plt.grid()\n    >>> plt.show()\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 207)
    # Getting the type of 'ml' (line 207)
    ml_28927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 7), 'ml')
    # Getting the type of 'None' (line 207)
    None_28928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 13), 'None')
    
    (may_be_28929, more_types_in_union_28930) = may_be_none(ml_28927, None_28928)

    if may_be_28929:

        if more_types_in_union_28930:
            # Runtime conditional SSA (line 207)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 208):
        int_28931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 13), 'int')
        # Assigning a type to the variable 'ml' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'ml', int_28931)

        if more_types_in_union_28930:
            # SSA join for if statement (line 207)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 209)
    # Getting the type of 'mu' (line 209)
    mu_28932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 7), 'mu')
    # Getting the type of 'None' (line 209)
    None_28933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), 'None')
    
    (may_be_28934, more_types_in_union_28935) = may_be_none(mu_28932, None_28933)

    if may_be_28934:

        if more_types_in_union_28935:
            # Runtime conditional SSA (line 209)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 210):
        int_28936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 13), 'int')
        # Assigning a type to the variable 'mu' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'mu', int_28936)

        if more_types_in_union_28935:
            # SSA join for if statement (line 209)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 211):
    
    # Call to copy(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 't' (line 211)
    t_28938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 13), 't', False)
    # Processing the call keyword arguments (line 211)
    kwargs_28939 = {}
    # Getting the type of 'copy' (line 211)
    copy_28937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'copy', False)
    # Calling copy(args, kwargs) (line 211)
    copy_call_result_28940 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), copy_28937, *[t_28938], **kwargs_28939)
    
    # Assigning a type to the variable 't' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 't', copy_call_result_28940)
    
    # Assigning a Call to a Name (line 212):
    
    # Call to copy(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'y0' (line 212)
    y0_28942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 14), 'y0', False)
    # Processing the call keyword arguments (line 212)
    kwargs_28943 = {}
    # Getting the type of 'copy' (line 212)
    copy_28941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 9), 'copy', False)
    # Calling copy(args, kwargs) (line 212)
    copy_call_result_28944 = invoke(stypy.reporting.localization.Localization(__file__, 212, 9), copy_28941, *[y0_28942], **kwargs_28943)
    
    # Assigning a type to the variable 'y0' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'y0', copy_call_result_28944)
    
    # Assigning a Call to a Name (line 213):
    
    # Call to odeint(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'func' (line 213)
    func_28947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 29), 'func', False)
    # Getting the type of 'y0' (line 213)
    y0_28948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 35), 'y0', False)
    # Getting the type of 't' (line 213)
    t_28949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 39), 't', False)
    # Getting the type of 'args' (line 213)
    args_28950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 42), 'args', False)
    # Getting the type of 'Dfun' (line 213)
    Dfun_28951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 48), 'Dfun', False)
    # Getting the type of 'col_deriv' (line 213)
    col_deriv_28952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 54), 'col_deriv', False)
    # Getting the type of 'ml' (line 213)
    ml_28953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 65), 'ml', False)
    # Getting the type of 'mu' (line 213)
    mu_28954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 69), 'mu', False)
    # Getting the type of 'full_output' (line 214)
    full_output_28955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 29), 'full_output', False)
    # Getting the type of 'rtol' (line 214)
    rtol_28956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 42), 'rtol', False)
    # Getting the type of 'atol' (line 214)
    atol_28957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 48), 'atol', False)
    # Getting the type of 'tcrit' (line 214)
    tcrit_28958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 54), 'tcrit', False)
    # Getting the type of 'h0' (line 214)
    h0_28959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 61), 'h0', False)
    # Getting the type of 'hmax' (line 214)
    hmax_28960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 65), 'hmax', False)
    # Getting the type of 'hmin' (line 214)
    hmin_28961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 71), 'hmin', False)
    # Getting the type of 'ixpr' (line 215)
    ixpr_28962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 29), 'ixpr', False)
    # Getting the type of 'mxstep' (line 215)
    mxstep_28963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 35), 'mxstep', False)
    # Getting the type of 'mxhnil' (line 215)
    mxhnil_28964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 43), 'mxhnil', False)
    # Getting the type of 'mxordn' (line 215)
    mxordn_28965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 51), 'mxordn', False)
    # Getting the type of 'mxords' (line 215)
    mxords_28966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 59), 'mxords', False)
    # Processing the call keyword arguments (line 213)
    kwargs_28967 = {}
    # Getting the type of '_odepack' (line 213)
    _odepack_28945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), '_odepack', False)
    # Obtaining the member 'odeint' of a type (line 213)
    odeint_28946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 13), _odepack_28945, 'odeint')
    # Calling odeint(args, kwargs) (line 213)
    odeint_call_result_28968 = invoke(stypy.reporting.localization.Localization(__file__, 213, 13), odeint_28946, *[func_28947, y0_28948, t_28949, args_28950, Dfun_28951, col_deriv_28952, ml_28953, mu_28954, full_output_28955, rtol_28956, atol_28957, tcrit_28958, h0_28959, hmax_28960, hmin_28961, ixpr_28962, mxstep_28963, mxhnil_28964, mxordn_28965, mxords_28966], **kwargs_28967)
    
    # Assigning a type to the variable 'output' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'output', odeint_call_result_28968)
    
    
    
    # Obtaining the type of the subscript
    int_28969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 14), 'int')
    # Getting the type of 'output' (line 216)
    output_28970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 7), 'output')
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___28971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 7), output_28970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_28972 = invoke(stypy.reporting.localization.Localization(__file__, 216, 7), getitem___28971, int_28969)
    
    int_28973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 20), 'int')
    # Applying the binary operator '<' (line 216)
    result_lt_28974 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 7), '<', subscript_call_result_28972, int_28973)
    
    # Testing the type of an if condition (line 216)
    if_condition_28975 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 216, 4), result_lt_28974)
    # Assigning a type to the variable 'if_condition_28975' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'if_condition_28975', if_condition_28975)
    # SSA begins for if statement (line 216)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 217):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_28976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 35), 'int')
    # Getting the type of 'output' (line 217)
    output_28977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 28), 'output')
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___28978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 28), output_28977, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_28979 = invoke(stypy.reporting.localization.Localization(__file__, 217, 28), getitem___28978, int_28976)
    
    # Getting the type of '_msgs' (line 217)
    _msgs_28980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 22), '_msgs')
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___28981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 22), _msgs_28980, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_28982 = invoke(stypy.reporting.localization.Localization(__file__, 217, 22), getitem___28981, subscript_call_result_28979)
    
    str_28983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 42), 'str', ' Run with full_output = 1 to get quantitative information.')
    # Applying the binary operator '+' (line 217)
    result_add_28984 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 22), '+', subscript_call_result_28982, str_28983)
    
    # Assigning a type to the variable 'warning_msg' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'warning_msg', result_add_28984)
    
    # Call to warn(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'warning_msg' (line 218)
    warning_msg_28987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'warning_msg', False)
    # Getting the type of 'ODEintWarning' (line 218)
    ODEintWarning_28988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 35), 'ODEintWarning', False)
    # Processing the call keyword arguments (line 218)
    kwargs_28989 = {}
    # Getting the type of 'warnings' (line 218)
    warnings_28985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 218)
    warn_28986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), warnings_28985, 'warn')
    # Calling warn(args, kwargs) (line 218)
    warn_call_result_28990 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), warn_28986, *[warning_msg_28987, ODEintWarning_28988], **kwargs_28989)
    
    # SSA branch for the else part of an if statement (line 216)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'printmessg' (line 219)
    printmessg_28991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 9), 'printmessg')
    # Testing the type of an if condition (line 219)
    if_condition_28992 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 9), printmessg_28991)
    # Assigning a type to the variable 'if_condition_28992' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 9), 'if_condition_28992', if_condition_28992)
    # SSA begins for if statement (line 219)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 220):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_28993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 35), 'int')
    # Getting the type of 'output' (line 220)
    output_28994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'output')
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___28995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 28), output_28994, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_28996 = invoke(stypy.reporting.localization.Localization(__file__, 220, 28), getitem___28995, int_28993)
    
    # Getting the type of '_msgs' (line 220)
    _msgs_28997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 22), '_msgs')
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___28998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 22), _msgs_28997, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_28999 = invoke(stypy.reporting.localization.Localization(__file__, 220, 22), getitem___28998, subscript_call_result_28996)
    
    # Assigning a type to the variable 'warning_msg' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'warning_msg', subscript_call_result_28999)
    
    # Call to warn(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'warning_msg' (line 221)
    warning_msg_29002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 22), 'warning_msg', False)
    # Getting the type of 'ODEintWarning' (line 221)
    ODEintWarning_29003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 35), 'ODEintWarning', False)
    # Processing the call keyword arguments (line 221)
    kwargs_29004 = {}
    # Getting the type of 'warnings' (line 221)
    warnings_29000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 221)
    warn_29001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), warnings_29000, 'warn')
    # Calling warn(args, kwargs) (line 221)
    warn_call_result_29005 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), warn_29001, *[warning_msg_29002, ODEintWarning_29003], **kwargs_29004)
    
    # SSA join for if statement (line 219)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 216)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'full_output' (line 223)
    full_output_29006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 7), 'full_output')
    # Testing the type of an if condition (line 223)
    if_condition_29007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 4), full_output_29006)
    # Assigning a type to the variable 'if_condition_29007' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'if_condition_29007', if_condition_29007)
    # SSA begins for if statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 224):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_29008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 44), 'int')
    # Getting the type of 'output' (line 224)
    output_29009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 37), 'output')
    # Obtaining the member '__getitem__' of a type (line 224)
    getitem___29010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 37), output_29009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 224)
    subscript_call_result_29011 = invoke(stypy.reporting.localization.Localization(__file__, 224, 37), getitem___29010, int_29008)
    
    # Getting the type of '_msgs' (line 224)
    _msgs_29012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), '_msgs')
    # Obtaining the member '__getitem__' of a type (line 224)
    getitem___29013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 31), _msgs_29012, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 224)
    subscript_call_result_29014 = invoke(stypy.reporting.localization.Localization(__file__, 224, 31), getitem___29013, subscript_call_result_29011)
    
    
    # Obtaining the type of the subscript
    int_29015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 15), 'int')
    # Getting the type of 'output' (line 224)
    output_29016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'output')
    # Obtaining the member '__getitem__' of a type (line 224)
    getitem___29017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), output_29016, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 224)
    subscript_call_result_29018 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), getitem___29017, int_29015)
    
    str_29019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 18), 'str', 'message')
    # Storing an element on a container (line 224)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 8), subscript_call_result_29018, (str_29019, subscript_call_result_29014))
    # SSA join for if statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_29020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 21), 'int')
    slice_29021 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 226, 13), None, int_29020, None)
    # Getting the type of 'output' (line 226)
    output_29022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 13), 'output')
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___29023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 13), output_29022, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_29024 = invoke(stypy.reporting.localization.Localization(__file__, 226, 13), getitem___29023, slice_29021)
    
    # Assigning a type to the variable 'output' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'output', subscript_call_result_29024)
    
    
    
    # Call to len(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'output' (line 227)
    output_29026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'output', False)
    # Processing the call keyword arguments (line 227)
    kwargs_29027 = {}
    # Getting the type of 'len' (line 227)
    len_29025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 7), 'len', False)
    # Calling len(args, kwargs) (line 227)
    len_call_result_29028 = invoke(stypy.reporting.localization.Localization(__file__, 227, 7), len_29025, *[output_29026], **kwargs_29027)
    
    int_29029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 22), 'int')
    # Applying the binary operator '==' (line 227)
    result_eq_29030 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 7), '==', len_call_result_29028, int_29029)
    
    # Testing the type of an if condition (line 227)
    if_condition_29031 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 4), result_eq_29030)
    # Assigning a type to the variable 'if_condition_29031' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'if_condition_29031', if_condition_29031)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_29032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 22), 'int')
    # Getting the type of 'output' (line 228)
    output_29033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'output')
    # Obtaining the member '__getitem__' of a type (line 228)
    getitem___29034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 15), output_29033, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 228)
    subscript_call_result_29035 = invoke(stypy.reporting.localization.Localization(__file__, 228, 15), getitem___29034, int_29032)
    
    # Assigning a type to the variable 'stypy_return_type' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'stypy_return_type', subscript_call_result_29035)
    # SSA branch for the else part of an if statement (line 227)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'output' (line 230)
    output_29036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'output')
    # Assigning a type to the variable 'stypy_return_type' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'stypy_return_type', output_29036)
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'odeint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'odeint' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_29037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29037)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'odeint'
    return stypy_return_type_29037

# Assigning a type to the variable 'odeint' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'odeint', odeint)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
