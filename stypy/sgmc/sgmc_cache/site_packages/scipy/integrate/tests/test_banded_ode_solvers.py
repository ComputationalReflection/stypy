
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: from __future__ import division, print_function, absolute_import
3: 
4: import itertools
5: import numpy as np
6: from numpy.testing import assert_allclose
7: from scipy.integrate import ode
8: 
9: 
10: def _band_count(a):
11:     '''Returns ml and mu, the lower and upper band sizes of a.'''
12:     nrows, ncols = a.shape
13:     ml = 0
14:     for k in range(-nrows+1, 0):
15:         if np.diag(a, k).any():
16:             ml = -k
17:             break
18:     mu = 0
19:     for k in range(nrows-1, 0, -1):
20:         if np.diag(a, k).any():
21:             mu = k
22:             break
23:     return ml, mu
24: 
25: 
26: def _linear_func(t, y, a):
27:     '''Linear system dy/dt = a * y'''
28:     return a.dot(y)
29: 
30: 
31: def _linear_jac(t, y, a):
32:     '''Jacobian of a * y is a.'''
33:     return a
34: 
35: 
36: def _linear_banded_jac(t, y, a):
37:     '''Banded Jacobian.'''
38:     ml, mu = _band_count(a)
39:     bjac = []
40:     for k in range(mu, 0, -1):
41:         bjac.append(np.r_[[0] * k, np.diag(a, k)])
42:     bjac.append(np.diag(a))
43:     for k in range(-1, -ml-1, -1):
44:         bjac.append(np.r_[np.diag(a, k), [0] * (-k)])
45:     return bjac
46: 
47: 
48: def _solve_linear_sys(a, y0, tend=1, dt=0.1,
49:                       solver=None, method='bdf', use_jac=True,
50:                       with_jacobian=False, banded=False):
51:     '''Use scipy.integrate.ode to solve a linear system of ODEs.
52: 
53:     a : square ndarray
54:         Matrix of the linear system to be solved.
55:     y0 : ndarray
56:         Initial condition
57:     tend : float
58:         Stop time.
59:     dt : float
60:         Step size of the output.
61:     solver : str
62:         If not None, this must be "vode", "lsoda" or "zvode".
63:     method : str
64:         Either "bdf" or "adams".
65:     use_jac : bool
66:         Determines if the jacobian function is passed to ode().
67:     with_jacobian : bool
68:         Passed to ode.set_integrator().
69:     banded : bool
70:         Determines whether a banded or full jacobian is used.
71:         If `banded` is True, `lband` and `uband` are determined by the
72:         values in `a`.
73:     '''
74:     if banded:
75:         lband, uband = _band_count(a)
76:     else:
77:         lband = None
78:         uband = None
79: 
80:     if use_jac:
81:         if banded:
82:             r = ode(_linear_func, _linear_banded_jac)
83:         else:
84:             r = ode(_linear_func, _linear_jac)
85:     else:
86:         r = ode(_linear_func)
87: 
88:     if solver is None:
89:         if np.iscomplexobj(a):
90:             solver = "zvode"
91:         else:
92:             solver = "vode"
93: 
94:     r.set_integrator(solver,
95:                      with_jacobian=with_jacobian,
96:                      method=method,
97:                      lband=lband, uband=uband,
98:                      rtol=1e-9, atol=1e-10,
99:                      )
100:     t0 = 0
101:     r.set_initial_value(y0, t0)
102:     r.set_f_params(a)
103:     r.set_jac_params(a)
104: 
105:     t = [t0]
106:     y = [y0]
107:     while r.successful() and r.t < tend:
108:         r.integrate(r.t + dt)
109:         t.append(r.t)
110:         y.append(r.y)
111: 
112:     t = np.array(t)
113:     y = np.array(y)
114:     return t, y
115: 
116: 
117: def _analytical_solution(a, y0, t):
118:     '''
119:     Analytical solution to the linear differential equations dy/dt = a*y.
120: 
121:     The solution is only valid if `a` is diagonalizable.
122: 
123:     Returns a 2-d array with shape (len(t), len(y0)).
124:     '''
125:     lam, v = np.linalg.eig(a)
126:     c = np.linalg.solve(v, y0)
127:     e = c * np.exp(lam * t.reshape(-1, 1))
128:     sol = e.dot(v.T)
129:     return sol
130: 
131: 
132: def test_banded_ode_solvers():
133:     # Test the "lsoda", "vode" and "zvode" solvers of the `ode` class
134:     # with a system that has a banded Jacobian matrix.
135: 
136:     t_exact = np.linspace(0, 1.0, 5)
137: 
138:     # --- Real arrays for testing the "lsoda" and "vode" solvers ---
139: 
140:     # lband = 2, uband = 1:
141:     a_real = np.array([[-0.6, 0.1, 0.0, 0.0, 0.0],
142:                        [0.2, -0.5, 0.9, 0.0, 0.0],
143:                        [0.1, 0.1, -0.4, 0.1, 0.0],
144:                        [0.0, 0.3, -0.1, -0.9, -0.3],
145:                        [0.0, 0.0, 0.1, 0.1, -0.7]])
146: 
147:     # lband = 0, uband = 1:
148:     a_real_upper = np.triu(a_real)
149: 
150:     # lband = 2, uband = 0:
151:     a_real_lower = np.tril(a_real)
152: 
153:     # lband = 0, uband = 0:
154:     a_real_diag = np.triu(a_real_lower)
155: 
156:     real_matrices = [a_real, a_real_upper, a_real_lower, a_real_diag]
157:     real_solutions = []
158: 
159:     for a in real_matrices:
160:         y0 = np.arange(1, a.shape[0] + 1)
161:         y_exact = _analytical_solution(a, y0, t_exact)
162:         real_solutions.append((y0, t_exact, y_exact))
163: 
164:     def check_real(idx, solver, meth, use_jac, with_jac, banded):
165:         a = real_matrices[idx]
166:         y0, t_exact, y_exact = real_solutions[idx]
167:         t, y = _solve_linear_sys(a, y0,
168:                                  tend=t_exact[-1],
169:                                  dt=t_exact[1] - t_exact[0],
170:                                  solver=solver,
171:                                  method=meth,
172:                                  use_jac=use_jac,
173:                                  with_jacobian=with_jac,
174:                                  banded=banded)
175:         assert_allclose(t, t_exact)
176:         assert_allclose(y, y_exact)
177: 
178:     for idx in range(len(real_matrices)):
179:         p = [['vode', 'lsoda'],  # solver
180:              ['bdf', 'adams'],   # method
181:              [False, True],      # use_jac
182:              [False, True],      # with_jacobian
183:              [False, True]]      # banded
184:         for solver, meth, use_jac, with_jac, banded in itertools.product(*p):
185:             check_real(idx, solver, meth, use_jac, with_jac, banded)
186: 
187:     # --- Complex arrays for testing the "zvode" solver ---
188: 
189:     # complex, lband = 2, uband = 1:
190:     a_complex = a_real - 0.5j * a_real
191: 
192:     # complex, lband = 0, uband = 0:
193:     a_complex_diag = np.diag(np.diag(a_complex))
194: 
195:     complex_matrices = [a_complex, a_complex_diag]
196:     complex_solutions = []
197: 
198:     for a in complex_matrices:
199:         y0 = np.arange(1, a.shape[0] + 1) + 1j
200:         y_exact = _analytical_solution(a, y0, t_exact)
201:         complex_solutions.append((y0, t_exact, y_exact))
202: 
203:     def check_complex(idx, solver, meth, use_jac, with_jac, banded):
204:         a = complex_matrices[idx]
205:         y0, t_exact, y_exact = complex_solutions[idx]
206:         t, y = _solve_linear_sys(a, y0,
207:                                  tend=t_exact[-1],
208:                                  dt=t_exact[1] - t_exact[0],
209:                                  solver=solver,
210:                                  method=meth,
211:                                  use_jac=use_jac,
212:                                  with_jacobian=with_jac,
213:                                  banded=banded)
214:         assert_allclose(t, t_exact)
215:         assert_allclose(y, y_exact)
216: 
217:     for idx in range(len(complex_matrices)):
218:         p = [['bdf', 'adams'],   # method
219:              [False, True],      # use_jac
220:              [False, True],      # with_jacobian
221:              [False, True]]      # banded
222:         for meth, use_jac, with_jac, banded in itertools.product(*p):
223:             check_complex(idx, "zvode", meth, use_jac, with_jac, banded)
224: 
225: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import itertools' statement (line 4)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_38017 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_38017) is not StypyTypeError):

    if (import_38017 != 'pyd_module'):
        __import__(import_38017)
        sys_modules_38018 = sys.modules[import_38017]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_38018.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_38017)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_allclose' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_38019 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_38019) is not StypyTypeError):

    if (import_38019 != 'pyd_module'):
        __import__(import_38019)
        sys_modules_38020 = sys.modules[import_38019]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_38020.module_type_store, module_type_store, ['assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_38020, sys_modules_38020.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_allclose'], [assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_38019)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.integrate import ode' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_38021 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate')

if (type(import_38021) is not StypyTypeError):

    if (import_38021 != 'pyd_module'):
        __import__(import_38021)
        sys_modules_38022 = sys.modules[import_38021]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate', sys_modules_38022.module_type_store, module_type_store, ['ode'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_38022, sys_modules_38022.module_type_store, module_type_store)
    else:
        from scipy.integrate import ode

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate', None, module_type_store, ['ode'], [ode])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate', import_38021)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')


@norecursion
def _band_count(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_band_count'
    module_type_store = module_type_store.open_function_context('_band_count', 10, 0, False)
    
    # Passed parameters checking function
    _band_count.stypy_localization = localization
    _band_count.stypy_type_of_self = None
    _band_count.stypy_type_store = module_type_store
    _band_count.stypy_function_name = '_band_count'
    _band_count.stypy_param_names_list = ['a']
    _band_count.stypy_varargs_param_name = None
    _band_count.stypy_kwargs_param_name = None
    _band_count.stypy_call_defaults = defaults
    _band_count.stypy_call_varargs = varargs
    _band_count.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_band_count', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_band_count', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_band_count(...)' code ##################

    str_38023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 4), 'str', 'Returns ml and mu, the lower and upper band sizes of a.')
    
    # Assigning a Attribute to a Tuple (line 12):
    
    # Assigning a Subscript to a Name (line 12):
    
    # Obtaining the type of the subscript
    int_38024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'int')
    # Getting the type of 'a' (line 12)
    a_38025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'a')
    # Obtaining the member 'shape' of a type (line 12)
    shape_38026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 19), a_38025, 'shape')
    # Obtaining the member '__getitem__' of a type (line 12)
    getitem___38027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), shape_38026, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 12)
    subscript_call_result_38028 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), getitem___38027, int_38024)
    
    # Assigning a type to the variable 'tuple_var_assignment_37999' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'tuple_var_assignment_37999', subscript_call_result_38028)
    
    # Assigning a Subscript to a Name (line 12):
    
    # Obtaining the type of the subscript
    int_38029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'int')
    # Getting the type of 'a' (line 12)
    a_38030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'a')
    # Obtaining the member 'shape' of a type (line 12)
    shape_38031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 19), a_38030, 'shape')
    # Obtaining the member '__getitem__' of a type (line 12)
    getitem___38032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), shape_38031, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 12)
    subscript_call_result_38033 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), getitem___38032, int_38029)
    
    # Assigning a type to the variable 'tuple_var_assignment_38000' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'tuple_var_assignment_38000', subscript_call_result_38033)
    
    # Assigning a Name to a Name (line 12):
    # Getting the type of 'tuple_var_assignment_37999' (line 12)
    tuple_var_assignment_37999_38034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'tuple_var_assignment_37999')
    # Assigning a type to the variable 'nrows' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'nrows', tuple_var_assignment_37999_38034)
    
    # Assigning a Name to a Name (line 12):
    # Getting the type of 'tuple_var_assignment_38000' (line 12)
    tuple_var_assignment_38000_38035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'tuple_var_assignment_38000')
    # Assigning a type to the variable 'ncols' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'ncols', tuple_var_assignment_38000_38035)
    
    # Assigning a Num to a Name (line 13):
    
    # Assigning a Num to a Name (line 13):
    int_38036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 9), 'int')
    # Assigning a type to the variable 'ml' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ml', int_38036)
    
    
    # Call to range(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Getting the type of 'nrows' (line 14)
    nrows_38038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 20), 'nrows', False)
    # Applying the 'usub' unary operator (line 14)
    result___neg___38039 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 19), 'usub', nrows_38038)
    
    int_38040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 26), 'int')
    # Applying the binary operator '+' (line 14)
    result_add_38041 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 19), '+', result___neg___38039, int_38040)
    
    int_38042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 29), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_38043 = {}
    # Getting the type of 'range' (line 14)
    range_38037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 13), 'range', False)
    # Calling range(args, kwargs) (line 14)
    range_call_result_38044 = invoke(stypy.reporting.localization.Localization(__file__, 14, 13), range_38037, *[result_add_38041, int_38042], **kwargs_38043)
    
    # Testing the type of a for loop iterable (line 14)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 14, 4), range_call_result_38044)
    # Getting the type of the for loop variable (line 14)
    for_loop_var_38045 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 14, 4), range_call_result_38044)
    # Assigning a type to the variable 'k' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'k', for_loop_var_38045)
    # SSA begins for a for statement (line 14)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to any(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_38053 = {}
    
    # Call to diag(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'a' (line 15)
    a_38048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'a', False)
    # Getting the type of 'k' (line 15)
    k_38049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'k', False)
    # Processing the call keyword arguments (line 15)
    kwargs_38050 = {}
    # Getting the type of 'np' (line 15)
    np_38046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'np', False)
    # Obtaining the member 'diag' of a type (line 15)
    diag_38047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 11), np_38046, 'diag')
    # Calling diag(args, kwargs) (line 15)
    diag_call_result_38051 = invoke(stypy.reporting.localization.Localization(__file__, 15, 11), diag_38047, *[a_38048, k_38049], **kwargs_38050)
    
    # Obtaining the member 'any' of a type (line 15)
    any_38052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 11), diag_call_result_38051, 'any')
    # Calling any(args, kwargs) (line 15)
    any_call_result_38054 = invoke(stypy.reporting.localization.Localization(__file__, 15, 11), any_38052, *[], **kwargs_38053)
    
    # Testing the type of an if condition (line 15)
    if_condition_38055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 8), any_call_result_38054)
    # Assigning a type to the variable 'if_condition_38055' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'if_condition_38055', if_condition_38055)
    # SSA begins for if statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a UnaryOp to a Name (line 16):
    
    # Assigning a UnaryOp to a Name (line 16):
    
    # Getting the type of 'k' (line 16)
    k_38056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'k')
    # Applying the 'usub' unary operator (line 16)
    result___neg___38057 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 17), 'usub', k_38056)
    
    # Assigning a type to the variable 'ml' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'ml', result___neg___38057)
    # SSA join for if statement (line 15)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 18):
    
    # Assigning a Num to a Name (line 18):
    int_38058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 9), 'int')
    # Assigning a type to the variable 'mu' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'mu', int_38058)
    
    
    # Call to range(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'nrows' (line 19)
    nrows_38060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'nrows', False)
    int_38061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'int')
    # Applying the binary operator '-' (line 19)
    result_sub_38062 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 19), '-', nrows_38060, int_38061)
    
    int_38063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 28), 'int')
    int_38064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'int')
    # Processing the call keyword arguments (line 19)
    kwargs_38065 = {}
    # Getting the type of 'range' (line 19)
    range_38059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'range', False)
    # Calling range(args, kwargs) (line 19)
    range_call_result_38066 = invoke(stypy.reporting.localization.Localization(__file__, 19, 13), range_38059, *[result_sub_38062, int_38063, int_38064], **kwargs_38065)
    
    # Testing the type of a for loop iterable (line 19)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 4), range_call_result_38066)
    # Getting the type of the for loop variable (line 19)
    for_loop_var_38067 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 4), range_call_result_38066)
    # Assigning a type to the variable 'k' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'k', for_loop_var_38067)
    # SSA begins for a for statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to any(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_38075 = {}
    
    # Call to diag(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'a' (line 20)
    a_38070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'a', False)
    # Getting the type of 'k' (line 20)
    k_38071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'k', False)
    # Processing the call keyword arguments (line 20)
    kwargs_38072 = {}
    # Getting the type of 'np' (line 20)
    np_38068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'np', False)
    # Obtaining the member 'diag' of a type (line 20)
    diag_38069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 11), np_38068, 'diag')
    # Calling diag(args, kwargs) (line 20)
    diag_call_result_38073 = invoke(stypy.reporting.localization.Localization(__file__, 20, 11), diag_38069, *[a_38070, k_38071], **kwargs_38072)
    
    # Obtaining the member 'any' of a type (line 20)
    any_38074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 11), diag_call_result_38073, 'any')
    # Calling any(args, kwargs) (line 20)
    any_call_result_38076 = invoke(stypy.reporting.localization.Localization(__file__, 20, 11), any_38074, *[], **kwargs_38075)
    
    # Testing the type of an if condition (line 20)
    if_condition_38077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 8), any_call_result_38076)
    # Assigning a type to the variable 'if_condition_38077' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'if_condition_38077', if_condition_38077)
    # SSA begins for if statement (line 20)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 21):
    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'k' (line 21)
    k_38078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'k')
    # Assigning a type to the variable 'mu' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'mu', k_38078)
    # SSA join for if statement (line 20)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 23)
    tuple_38079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 23)
    # Adding element type (line 23)
    # Getting the type of 'ml' (line 23)
    ml_38080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'ml')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_38079, ml_38080)
    # Adding element type (line 23)
    # Getting the type of 'mu' (line 23)
    mu_38081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'mu')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 11), tuple_38079, mu_38081)
    
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type', tuple_38079)
    
    # ################# End of '_band_count(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_band_count' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_38082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38082)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_band_count'
    return stypy_return_type_38082

# Assigning a type to the variable '_band_count' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '_band_count', _band_count)

@norecursion
def _linear_func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_linear_func'
    module_type_store = module_type_store.open_function_context('_linear_func', 26, 0, False)
    
    # Passed parameters checking function
    _linear_func.stypy_localization = localization
    _linear_func.stypy_type_of_self = None
    _linear_func.stypy_type_store = module_type_store
    _linear_func.stypy_function_name = '_linear_func'
    _linear_func.stypy_param_names_list = ['t', 'y', 'a']
    _linear_func.stypy_varargs_param_name = None
    _linear_func.stypy_kwargs_param_name = None
    _linear_func.stypy_call_defaults = defaults
    _linear_func.stypy_call_varargs = varargs
    _linear_func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_linear_func', ['t', 'y', 'a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_linear_func', localization, ['t', 'y', 'a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_linear_func(...)' code ##################

    str_38083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'str', 'Linear system dy/dt = a * y')
    
    # Call to dot(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'y' (line 28)
    y_38086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'y', False)
    # Processing the call keyword arguments (line 28)
    kwargs_38087 = {}
    # Getting the type of 'a' (line 28)
    a_38084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'a', False)
    # Obtaining the member 'dot' of a type (line 28)
    dot_38085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 11), a_38084, 'dot')
    # Calling dot(args, kwargs) (line 28)
    dot_call_result_38088 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), dot_38085, *[y_38086], **kwargs_38087)
    
    # Assigning a type to the variable 'stypy_return_type' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type', dot_call_result_38088)
    
    # ################# End of '_linear_func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_linear_func' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_38089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38089)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_linear_func'
    return stypy_return_type_38089

# Assigning a type to the variable '_linear_func' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '_linear_func', _linear_func)

@norecursion
def _linear_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_linear_jac'
    module_type_store = module_type_store.open_function_context('_linear_jac', 31, 0, False)
    
    # Passed parameters checking function
    _linear_jac.stypy_localization = localization
    _linear_jac.stypy_type_of_self = None
    _linear_jac.stypy_type_store = module_type_store
    _linear_jac.stypy_function_name = '_linear_jac'
    _linear_jac.stypy_param_names_list = ['t', 'y', 'a']
    _linear_jac.stypy_varargs_param_name = None
    _linear_jac.stypy_kwargs_param_name = None
    _linear_jac.stypy_call_defaults = defaults
    _linear_jac.stypy_call_varargs = varargs
    _linear_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_linear_jac', ['t', 'y', 'a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_linear_jac', localization, ['t', 'y', 'a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_linear_jac(...)' code ##################

    str_38090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'str', 'Jacobian of a * y is a.')
    # Getting the type of 'a' (line 33)
    a_38091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type', a_38091)
    
    # ################# End of '_linear_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_linear_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_38092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38092)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_linear_jac'
    return stypy_return_type_38092

# Assigning a type to the variable '_linear_jac' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), '_linear_jac', _linear_jac)

@norecursion
def _linear_banded_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_linear_banded_jac'
    module_type_store = module_type_store.open_function_context('_linear_banded_jac', 36, 0, False)
    
    # Passed parameters checking function
    _linear_banded_jac.stypy_localization = localization
    _linear_banded_jac.stypy_type_of_self = None
    _linear_banded_jac.stypy_type_store = module_type_store
    _linear_banded_jac.stypy_function_name = '_linear_banded_jac'
    _linear_banded_jac.stypy_param_names_list = ['t', 'y', 'a']
    _linear_banded_jac.stypy_varargs_param_name = None
    _linear_banded_jac.stypy_kwargs_param_name = None
    _linear_banded_jac.stypy_call_defaults = defaults
    _linear_banded_jac.stypy_call_varargs = varargs
    _linear_banded_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_linear_banded_jac', ['t', 'y', 'a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_linear_banded_jac', localization, ['t', 'y', 'a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_linear_banded_jac(...)' code ##################

    str_38093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'str', 'Banded Jacobian.')
    
    # Assigning a Call to a Tuple (line 38):
    
    # Assigning a Subscript to a Name (line 38):
    
    # Obtaining the type of the subscript
    int_38094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'int')
    
    # Call to _band_count(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'a' (line 38)
    a_38096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'a', False)
    # Processing the call keyword arguments (line 38)
    kwargs_38097 = {}
    # Getting the type of '_band_count' (line 38)
    _band_count_38095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), '_band_count', False)
    # Calling _band_count(args, kwargs) (line 38)
    _band_count_call_result_38098 = invoke(stypy.reporting.localization.Localization(__file__, 38, 13), _band_count_38095, *[a_38096], **kwargs_38097)
    
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___38099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), _band_count_call_result_38098, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_38100 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), getitem___38099, int_38094)
    
    # Assigning a type to the variable 'tuple_var_assignment_38001' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'tuple_var_assignment_38001', subscript_call_result_38100)
    
    # Assigning a Subscript to a Name (line 38):
    
    # Obtaining the type of the subscript
    int_38101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'int')
    
    # Call to _band_count(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'a' (line 38)
    a_38103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'a', False)
    # Processing the call keyword arguments (line 38)
    kwargs_38104 = {}
    # Getting the type of '_band_count' (line 38)
    _band_count_38102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), '_band_count', False)
    # Calling _band_count(args, kwargs) (line 38)
    _band_count_call_result_38105 = invoke(stypy.reporting.localization.Localization(__file__, 38, 13), _band_count_38102, *[a_38103], **kwargs_38104)
    
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___38106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), _band_count_call_result_38105, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_38107 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), getitem___38106, int_38101)
    
    # Assigning a type to the variable 'tuple_var_assignment_38002' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'tuple_var_assignment_38002', subscript_call_result_38107)
    
    # Assigning a Name to a Name (line 38):
    # Getting the type of 'tuple_var_assignment_38001' (line 38)
    tuple_var_assignment_38001_38108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'tuple_var_assignment_38001')
    # Assigning a type to the variable 'ml' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'ml', tuple_var_assignment_38001_38108)
    
    # Assigning a Name to a Name (line 38):
    # Getting the type of 'tuple_var_assignment_38002' (line 38)
    tuple_var_assignment_38002_38109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'tuple_var_assignment_38002')
    # Assigning a type to the variable 'mu' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'mu', tuple_var_assignment_38002_38109)
    
    # Assigning a List to a Name (line 39):
    
    # Assigning a List to a Name (line 39):
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_38110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    
    # Assigning a type to the variable 'bjac' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'bjac', list_38110)
    
    
    # Call to range(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'mu' (line 40)
    mu_38112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'mu', False)
    int_38113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'int')
    int_38114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'int')
    # Processing the call keyword arguments (line 40)
    kwargs_38115 = {}
    # Getting the type of 'range' (line 40)
    range_38111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'range', False)
    # Calling range(args, kwargs) (line 40)
    range_call_result_38116 = invoke(stypy.reporting.localization.Localization(__file__, 40, 13), range_38111, *[mu_38112, int_38113, int_38114], **kwargs_38115)
    
    # Testing the type of a for loop iterable (line 40)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 4), range_call_result_38116)
    # Getting the type of the for loop variable (line 40)
    for_loop_var_38117 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 4), range_call_result_38116)
    # Assigning a type to the variable 'k' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'k', for_loop_var_38117)
    # SSA begins for a for statement (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 41)
    tuple_38120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 41)
    # Adding element type (line 41)
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_38121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    int_38122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 26), list_38121, int_38122)
    
    # Getting the type of 'k' (line 41)
    k_38123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 32), 'k', False)
    # Applying the binary operator '*' (line 41)
    result_mul_38124 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 26), '*', list_38121, k_38123)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 26), tuple_38120, result_mul_38124)
    # Adding element type (line 41)
    
    # Call to diag(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'a' (line 41)
    a_38127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 43), 'a', False)
    # Getting the type of 'k' (line 41)
    k_38128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 46), 'k', False)
    # Processing the call keyword arguments (line 41)
    kwargs_38129 = {}
    # Getting the type of 'np' (line 41)
    np_38125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 35), 'np', False)
    # Obtaining the member 'diag' of a type (line 41)
    diag_38126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 35), np_38125, 'diag')
    # Calling diag(args, kwargs) (line 41)
    diag_call_result_38130 = invoke(stypy.reporting.localization.Localization(__file__, 41, 35), diag_38126, *[a_38127, k_38128], **kwargs_38129)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 26), tuple_38120, diag_call_result_38130)
    
    # Getting the type of 'np' (line 41)
    np_38131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'np', False)
    # Obtaining the member 'r_' of a type (line 41)
    r__38132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 20), np_38131, 'r_')
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___38133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 20), r__38132, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_38134 = invoke(stypy.reporting.localization.Localization(__file__, 41, 20), getitem___38133, tuple_38120)
    
    # Processing the call keyword arguments (line 41)
    kwargs_38135 = {}
    # Getting the type of 'bjac' (line 41)
    bjac_38118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'bjac', False)
    # Obtaining the member 'append' of a type (line 41)
    append_38119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), bjac_38118, 'append')
    # Calling append(args, kwargs) (line 41)
    append_call_result_38136 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), append_38119, *[subscript_call_result_38134], **kwargs_38135)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Call to diag(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'a' (line 42)
    a_38141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'a', False)
    # Processing the call keyword arguments (line 42)
    kwargs_38142 = {}
    # Getting the type of 'np' (line 42)
    np_38139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'np', False)
    # Obtaining the member 'diag' of a type (line 42)
    diag_38140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 16), np_38139, 'diag')
    # Calling diag(args, kwargs) (line 42)
    diag_call_result_38143 = invoke(stypy.reporting.localization.Localization(__file__, 42, 16), diag_38140, *[a_38141], **kwargs_38142)
    
    # Processing the call keyword arguments (line 42)
    kwargs_38144 = {}
    # Getting the type of 'bjac' (line 42)
    bjac_38137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'bjac', False)
    # Obtaining the member 'append' of a type (line 42)
    append_38138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 4), bjac_38137, 'append')
    # Calling append(args, kwargs) (line 42)
    append_call_result_38145 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), append_38138, *[diag_call_result_38143], **kwargs_38144)
    
    
    
    # Call to range(...): (line 43)
    # Processing the call arguments (line 43)
    int_38147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'int')
    
    # Getting the type of 'ml' (line 43)
    ml_38148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'ml', False)
    # Applying the 'usub' unary operator (line 43)
    result___neg___38149 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 23), 'usub', ml_38148)
    
    int_38150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 27), 'int')
    # Applying the binary operator '-' (line 43)
    result_sub_38151 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 23), '-', result___neg___38149, int_38150)
    
    int_38152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 30), 'int')
    # Processing the call keyword arguments (line 43)
    kwargs_38153 = {}
    # Getting the type of 'range' (line 43)
    range_38146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'range', False)
    # Calling range(args, kwargs) (line 43)
    range_call_result_38154 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), range_38146, *[int_38147, result_sub_38151, int_38152], **kwargs_38153)
    
    # Testing the type of a for loop iterable (line 43)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 4), range_call_result_38154)
    # Getting the type of the for loop variable (line 43)
    for_loop_var_38155 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 4), range_call_result_38154)
    # Assigning a type to the variable 'k' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'k', for_loop_var_38155)
    # SSA begins for a for statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_38158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    
    # Call to diag(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'a' (line 44)
    a_38161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 34), 'a', False)
    # Getting the type of 'k' (line 44)
    k_38162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 37), 'k', False)
    # Processing the call keyword arguments (line 44)
    kwargs_38163 = {}
    # Getting the type of 'np' (line 44)
    np_38159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 26), 'np', False)
    # Obtaining the member 'diag' of a type (line 44)
    diag_38160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 26), np_38159, 'diag')
    # Calling diag(args, kwargs) (line 44)
    diag_call_result_38164 = invoke(stypy.reporting.localization.Localization(__file__, 44, 26), diag_38160, *[a_38161, k_38162], **kwargs_38163)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 26), tuple_38158, diag_call_result_38164)
    # Adding element type (line 44)
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_38165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    int_38166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 41), list_38165, int_38166)
    
    
    # Getting the type of 'k' (line 44)
    k_38167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 49), 'k', False)
    # Applying the 'usub' unary operator (line 44)
    result___neg___38168 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 48), 'usub', k_38167)
    
    # Applying the binary operator '*' (line 44)
    result_mul_38169 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 41), '*', list_38165, result___neg___38168)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 26), tuple_38158, result_mul_38169)
    
    # Getting the type of 'np' (line 44)
    np_38170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'np', False)
    # Obtaining the member 'r_' of a type (line 44)
    r__38171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 20), np_38170, 'r_')
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___38172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 20), r__38171, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_38173 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), getitem___38172, tuple_38158)
    
    # Processing the call keyword arguments (line 44)
    kwargs_38174 = {}
    # Getting the type of 'bjac' (line 44)
    bjac_38156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'bjac', False)
    # Obtaining the member 'append' of a type (line 44)
    append_38157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), bjac_38156, 'append')
    # Calling append(args, kwargs) (line 44)
    append_call_result_38175 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), append_38157, *[subscript_call_result_38173], **kwargs_38174)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'bjac' (line 45)
    bjac_38176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'bjac')
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type', bjac_38176)
    
    # ################# End of '_linear_banded_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_linear_banded_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_38177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38177)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_linear_banded_jac'
    return stypy_return_type_38177

# Assigning a type to the variable '_linear_banded_jac' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), '_linear_banded_jac', _linear_banded_jac)

@norecursion
def _solve_linear_sys(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_38178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 34), 'int')
    float_38179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 40), 'float')
    # Getting the type of 'None' (line 49)
    None_38180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 29), 'None')
    str_38181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 42), 'str', 'bdf')
    # Getting the type of 'True' (line 49)
    True_38182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 57), 'True')
    # Getting the type of 'False' (line 50)
    False_38183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 36), 'False')
    # Getting the type of 'False' (line 50)
    False_38184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 50), 'False')
    defaults = [int_38178, float_38179, None_38180, str_38181, True_38182, False_38183, False_38184]
    # Create a new context for function '_solve_linear_sys'
    module_type_store = module_type_store.open_function_context('_solve_linear_sys', 48, 0, False)
    
    # Passed parameters checking function
    _solve_linear_sys.stypy_localization = localization
    _solve_linear_sys.stypy_type_of_self = None
    _solve_linear_sys.stypy_type_store = module_type_store
    _solve_linear_sys.stypy_function_name = '_solve_linear_sys'
    _solve_linear_sys.stypy_param_names_list = ['a', 'y0', 'tend', 'dt', 'solver', 'method', 'use_jac', 'with_jacobian', 'banded']
    _solve_linear_sys.stypy_varargs_param_name = None
    _solve_linear_sys.stypy_kwargs_param_name = None
    _solve_linear_sys.stypy_call_defaults = defaults
    _solve_linear_sys.stypy_call_varargs = varargs
    _solve_linear_sys.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_solve_linear_sys', ['a', 'y0', 'tend', 'dt', 'solver', 'method', 'use_jac', 'with_jacobian', 'banded'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_solve_linear_sys', localization, ['a', 'y0', 'tend', 'dt', 'solver', 'method', 'use_jac', 'with_jacobian', 'banded'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_solve_linear_sys(...)' code ##################

    str_38185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, (-1)), 'str', 'Use scipy.integrate.ode to solve a linear system of ODEs.\n\n    a : square ndarray\n        Matrix of the linear system to be solved.\n    y0 : ndarray\n        Initial condition\n    tend : float\n        Stop time.\n    dt : float\n        Step size of the output.\n    solver : str\n        If not None, this must be "vode", "lsoda" or "zvode".\n    method : str\n        Either "bdf" or "adams".\n    use_jac : bool\n        Determines if the jacobian function is passed to ode().\n    with_jacobian : bool\n        Passed to ode.set_integrator().\n    banded : bool\n        Determines whether a banded or full jacobian is used.\n        If `banded` is True, `lband` and `uband` are determined by the\n        values in `a`.\n    ')
    
    # Getting the type of 'banded' (line 74)
    banded_38186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 7), 'banded')
    # Testing the type of an if condition (line 74)
    if_condition_38187 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 4), banded_38186)
    # Assigning a type to the variable 'if_condition_38187' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'if_condition_38187', if_condition_38187)
    # SSA begins for if statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 75):
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    int_38188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'int')
    
    # Call to _band_count(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'a' (line 75)
    a_38190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 35), 'a', False)
    # Processing the call keyword arguments (line 75)
    kwargs_38191 = {}
    # Getting the type of '_band_count' (line 75)
    _band_count_38189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), '_band_count', False)
    # Calling _band_count(args, kwargs) (line 75)
    _band_count_call_result_38192 = invoke(stypy.reporting.localization.Localization(__file__, 75, 23), _band_count_38189, *[a_38190], **kwargs_38191)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___38193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), _band_count_call_result_38192, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_38194 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), getitem___38193, int_38188)
    
    # Assigning a type to the variable 'tuple_var_assignment_38003' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'tuple_var_assignment_38003', subscript_call_result_38194)
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    int_38195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'int')
    
    # Call to _band_count(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'a' (line 75)
    a_38197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 35), 'a', False)
    # Processing the call keyword arguments (line 75)
    kwargs_38198 = {}
    # Getting the type of '_band_count' (line 75)
    _band_count_38196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), '_band_count', False)
    # Calling _band_count(args, kwargs) (line 75)
    _band_count_call_result_38199 = invoke(stypy.reporting.localization.Localization(__file__, 75, 23), _band_count_38196, *[a_38197], **kwargs_38198)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___38200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), _band_count_call_result_38199, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_38201 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), getitem___38200, int_38195)
    
    # Assigning a type to the variable 'tuple_var_assignment_38004' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'tuple_var_assignment_38004', subscript_call_result_38201)
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'tuple_var_assignment_38003' (line 75)
    tuple_var_assignment_38003_38202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'tuple_var_assignment_38003')
    # Assigning a type to the variable 'lband' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'lband', tuple_var_assignment_38003_38202)
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'tuple_var_assignment_38004' (line 75)
    tuple_var_assignment_38004_38203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'tuple_var_assignment_38004')
    # Assigning a type to the variable 'uband' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'uband', tuple_var_assignment_38004_38203)
    # SSA branch for the else part of an if statement (line 74)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 77):
    
    # Assigning a Name to a Name (line 77):
    # Getting the type of 'None' (line 77)
    None_38204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'None')
    # Assigning a type to the variable 'lband' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'lband', None_38204)
    
    # Assigning a Name to a Name (line 78):
    
    # Assigning a Name to a Name (line 78):
    # Getting the type of 'None' (line 78)
    None_38205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'None')
    # Assigning a type to the variable 'uband' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'uband', None_38205)
    # SSA join for if statement (line 74)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'use_jac' (line 80)
    use_jac_38206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 7), 'use_jac')
    # Testing the type of an if condition (line 80)
    if_condition_38207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 4), use_jac_38206)
    # Assigning a type to the variable 'if_condition_38207' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'if_condition_38207', if_condition_38207)
    # SSA begins for if statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'banded' (line 81)
    banded_38208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'banded')
    # Testing the type of an if condition (line 81)
    if_condition_38209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), banded_38208)
    # Assigning a type to the variable 'if_condition_38209' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_38209', if_condition_38209)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 82):
    
    # Assigning a Call to a Name (line 82):
    
    # Call to ode(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of '_linear_func' (line 82)
    _linear_func_38211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), '_linear_func', False)
    # Getting the type of '_linear_banded_jac' (line 82)
    _linear_banded_jac_38212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 34), '_linear_banded_jac', False)
    # Processing the call keyword arguments (line 82)
    kwargs_38213 = {}
    # Getting the type of 'ode' (line 82)
    ode_38210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'ode', False)
    # Calling ode(args, kwargs) (line 82)
    ode_call_result_38214 = invoke(stypy.reporting.localization.Localization(__file__, 82, 16), ode_38210, *[_linear_func_38211, _linear_banded_jac_38212], **kwargs_38213)
    
    # Assigning a type to the variable 'r' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'r', ode_call_result_38214)
    # SSA branch for the else part of an if statement (line 81)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 84):
    
    # Assigning a Call to a Name (line 84):
    
    # Call to ode(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of '_linear_func' (line 84)
    _linear_func_38216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), '_linear_func', False)
    # Getting the type of '_linear_jac' (line 84)
    _linear_jac_38217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 34), '_linear_jac', False)
    # Processing the call keyword arguments (line 84)
    kwargs_38218 = {}
    # Getting the type of 'ode' (line 84)
    ode_38215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'ode', False)
    # Calling ode(args, kwargs) (line 84)
    ode_call_result_38219 = invoke(stypy.reporting.localization.Localization(__file__, 84, 16), ode_38215, *[_linear_func_38216, _linear_jac_38217], **kwargs_38218)
    
    # Assigning a type to the variable 'r' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'r', ode_call_result_38219)
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 80)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to ode(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of '_linear_func' (line 86)
    _linear_func_38221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), '_linear_func', False)
    # Processing the call keyword arguments (line 86)
    kwargs_38222 = {}
    # Getting the type of 'ode' (line 86)
    ode_38220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'ode', False)
    # Calling ode(args, kwargs) (line 86)
    ode_call_result_38223 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), ode_38220, *[_linear_func_38221], **kwargs_38222)
    
    # Assigning a type to the variable 'r' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'r', ode_call_result_38223)
    # SSA join for if statement (line 80)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 88)
    # Getting the type of 'solver' (line 88)
    solver_38224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 7), 'solver')
    # Getting the type of 'None' (line 88)
    None_38225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'None')
    
    (may_be_38226, more_types_in_union_38227) = may_be_none(solver_38224, None_38225)

    if may_be_38226:

        if more_types_in_union_38227:
            # Runtime conditional SSA (line 88)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Call to iscomplexobj(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'a' (line 89)
        a_38230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'a', False)
        # Processing the call keyword arguments (line 89)
        kwargs_38231 = {}
        # Getting the type of 'np' (line 89)
        np_38228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'np', False)
        # Obtaining the member 'iscomplexobj' of a type (line 89)
        iscomplexobj_38229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 11), np_38228, 'iscomplexobj')
        # Calling iscomplexobj(args, kwargs) (line 89)
        iscomplexobj_call_result_38232 = invoke(stypy.reporting.localization.Localization(__file__, 89, 11), iscomplexobj_38229, *[a_38230], **kwargs_38231)
        
        # Testing the type of an if condition (line 89)
        if_condition_38233 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 8), iscomplexobj_call_result_38232)
        # Assigning a type to the variable 'if_condition_38233' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'if_condition_38233', if_condition_38233)
        # SSA begins for if statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 90):
        
        # Assigning a Str to a Name (line 90):
        str_38234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', 'zvode')
        # Assigning a type to the variable 'solver' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'solver', str_38234)
        # SSA branch for the else part of an if statement (line 89)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 92):
        
        # Assigning a Str to a Name (line 92):
        str_38235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 21), 'str', 'vode')
        # Assigning a type to the variable 'solver' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'solver', str_38235)
        # SSA join for if statement (line 89)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_38227:
            # SSA join for if statement (line 88)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to set_integrator(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'solver' (line 94)
    solver_38238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'solver', False)
    # Processing the call keyword arguments (line 94)
    # Getting the type of 'with_jacobian' (line 95)
    with_jacobian_38239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 35), 'with_jacobian', False)
    keyword_38240 = with_jacobian_38239
    # Getting the type of 'method' (line 96)
    method_38241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 28), 'method', False)
    keyword_38242 = method_38241
    # Getting the type of 'lband' (line 97)
    lband_38243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 27), 'lband', False)
    keyword_38244 = lband_38243
    # Getting the type of 'uband' (line 97)
    uband_38245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'uband', False)
    keyword_38246 = uband_38245
    float_38247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 26), 'float')
    keyword_38248 = float_38247
    float_38249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 37), 'float')
    keyword_38250 = float_38249
    kwargs_38251 = {'lband': keyword_38244, 'uband': keyword_38246, 'with_jacobian': keyword_38240, 'rtol': keyword_38248, 'atol': keyword_38250, 'method': keyword_38242}
    # Getting the type of 'r' (line 94)
    r_38236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'r', False)
    # Obtaining the member 'set_integrator' of a type (line 94)
    set_integrator_38237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 4), r_38236, 'set_integrator')
    # Calling set_integrator(args, kwargs) (line 94)
    set_integrator_call_result_38252 = invoke(stypy.reporting.localization.Localization(__file__, 94, 4), set_integrator_38237, *[solver_38238], **kwargs_38251)
    
    
    # Assigning a Num to a Name (line 100):
    
    # Assigning a Num to a Name (line 100):
    int_38253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 9), 'int')
    # Assigning a type to the variable 't0' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 't0', int_38253)
    
    # Call to set_initial_value(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'y0' (line 101)
    y0_38256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'y0', False)
    # Getting the type of 't0' (line 101)
    t0_38257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 't0', False)
    # Processing the call keyword arguments (line 101)
    kwargs_38258 = {}
    # Getting the type of 'r' (line 101)
    r_38254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'r', False)
    # Obtaining the member 'set_initial_value' of a type (line 101)
    set_initial_value_38255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), r_38254, 'set_initial_value')
    # Calling set_initial_value(args, kwargs) (line 101)
    set_initial_value_call_result_38259 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), set_initial_value_38255, *[y0_38256, t0_38257], **kwargs_38258)
    
    
    # Call to set_f_params(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'a' (line 102)
    a_38262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'a', False)
    # Processing the call keyword arguments (line 102)
    kwargs_38263 = {}
    # Getting the type of 'r' (line 102)
    r_38260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'r', False)
    # Obtaining the member 'set_f_params' of a type (line 102)
    set_f_params_38261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 4), r_38260, 'set_f_params')
    # Calling set_f_params(args, kwargs) (line 102)
    set_f_params_call_result_38264 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), set_f_params_38261, *[a_38262], **kwargs_38263)
    
    
    # Call to set_jac_params(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'a' (line 103)
    a_38267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 21), 'a', False)
    # Processing the call keyword arguments (line 103)
    kwargs_38268 = {}
    # Getting the type of 'r' (line 103)
    r_38265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'r', False)
    # Obtaining the member 'set_jac_params' of a type (line 103)
    set_jac_params_38266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 4), r_38265, 'set_jac_params')
    # Calling set_jac_params(args, kwargs) (line 103)
    set_jac_params_call_result_38269 = invoke(stypy.reporting.localization.Localization(__file__, 103, 4), set_jac_params_38266, *[a_38267], **kwargs_38268)
    
    
    # Assigning a List to a Name (line 105):
    
    # Assigning a List to a Name (line 105):
    
    # Obtaining an instance of the builtin type 'list' (line 105)
    list_38270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 105)
    # Adding element type (line 105)
    # Getting the type of 't0' (line 105)
    t0_38271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 9), 't0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 8), list_38270, t0_38271)
    
    # Assigning a type to the variable 't' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 't', list_38270)
    
    # Assigning a List to a Name (line 106):
    
    # Assigning a List to a Name (line 106):
    
    # Obtaining an instance of the builtin type 'list' (line 106)
    list_38272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 106)
    # Adding element type (line 106)
    # Getting the type of 'y0' (line 106)
    y0_38273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 9), 'y0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 8), list_38272, y0_38273)
    
    # Assigning a type to the variable 'y' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'y', list_38272)
    
    
    # Evaluating a boolean operation
    
    # Call to successful(...): (line 107)
    # Processing the call keyword arguments (line 107)
    kwargs_38276 = {}
    # Getting the type of 'r' (line 107)
    r_38274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 10), 'r', False)
    # Obtaining the member 'successful' of a type (line 107)
    successful_38275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 10), r_38274, 'successful')
    # Calling successful(args, kwargs) (line 107)
    successful_call_result_38277 = invoke(stypy.reporting.localization.Localization(__file__, 107, 10), successful_38275, *[], **kwargs_38276)
    
    
    # Getting the type of 'r' (line 107)
    r_38278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'r')
    # Obtaining the member 't' of a type (line 107)
    t_38279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 29), r_38278, 't')
    # Getting the type of 'tend' (line 107)
    tend_38280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 35), 'tend')
    # Applying the binary operator '<' (line 107)
    result_lt_38281 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 29), '<', t_38279, tend_38280)
    
    # Applying the binary operator 'and' (line 107)
    result_and_keyword_38282 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 10), 'and', successful_call_result_38277, result_lt_38281)
    
    # Testing the type of an if condition (line 107)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 4), result_and_keyword_38282)
    # SSA begins for while statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Call to integrate(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'r' (line 108)
    r_38285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'r', False)
    # Obtaining the member 't' of a type (line 108)
    t_38286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 20), r_38285, 't')
    # Getting the type of 'dt' (line 108)
    dt_38287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 26), 'dt', False)
    # Applying the binary operator '+' (line 108)
    result_add_38288 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 20), '+', t_38286, dt_38287)
    
    # Processing the call keyword arguments (line 108)
    kwargs_38289 = {}
    # Getting the type of 'r' (line 108)
    r_38283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'r', False)
    # Obtaining the member 'integrate' of a type (line 108)
    integrate_38284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), r_38283, 'integrate')
    # Calling integrate(args, kwargs) (line 108)
    integrate_call_result_38290 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), integrate_38284, *[result_add_38288], **kwargs_38289)
    
    
    # Call to append(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'r' (line 109)
    r_38293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'r', False)
    # Obtaining the member 't' of a type (line 109)
    t_38294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 17), r_38293, 't')
    # Processing the call keyword arguments (line 109)
    kwargs_38295 = {}
    # Getting the type of 't' (line 109)
    t_38291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 't', False)
    # Obtaining the member 'append' of a type (line 109)
    append_38292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), t_38291, 'append')
    # Calling append(args, kwargs) (line 109)
    append_call_result_38296 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), append_38292, *[t_38294], **kwargs_38295)
    
    
    # Call to append(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'r' (line 110)
    r_38299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'r', False)
    # Obtaining the member 'y' of a type (line 110)
    y_38300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 17), r_38299, 'y')
    # Processing the call keyword arguments (line 110)
    kwargs_38301 = {}
    # Getting the type of 'y' (line 110)
    y_38297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'y', False)
    # Obtaining the member 'append' of a type (line 110)
    append_38298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), y_38297, 'append')
    # Calling append(args, kwargs) (line 110)
    append_call_result_38302 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), append_38298, *[y_38300], **kwargs_38301)
    
    # SSA join for while statement (line 107)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 112):
    
    # Assigning a Call to a Name (line 112):
    
    # Call to array(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 't' (line 112)
    t_38305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 't', False)
    # Processing the call keyword arguments (line 112)
    kwargs_38306 = {}
    # Getting the type of 'np' (line 112)
    np_38303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 112)
    array_38304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), np_38303, 'array')
    # Calling array(args, kwargs) (line 112)
    array_call_result_38307 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), array_38304, *[t_38305], **kwargs_38306)
    
    # Assigning a type to the variable 't' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 't', array_call_result_38307)
    
    # Assigning a Call to a Name (line 113):
    
    # Assigning a Call to a Name (line 113):
    
    # Call to array(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'y' (line 113)
    y_38310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 17), 'y', False)
    # Processing the call keyword arguments (line 113)
    kwargs_38311 = {}
    # Getting the type of 'np' (line 113)
    np_38308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 113)
    array_38309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), np_38308, 'array')
    # Calling array(args, kwargs) (line 113)
    array_call_result_38312 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), array_38309, *[y_38310], **kwargs_38311)
    
    # Assigning a type to the variable 'y' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'y', array_call_result_38312)
    
    # Obtaining an instance of the builtin type 'tuple' (line 114)
    tuple_38313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 114)
    # Adding element type (line 114)
    # Getting the type of 't' (line 114)
    t_38314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 't')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 11), tuple_38313, t_38314)
    # Adding element type (line 114)
    # Getting the type of 'y' (line 114)
    y_38315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 11), tuple_38313, y_38315)
    
    # Assigning a type to the variable 'stypy_return_type' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type', tuple_38313)
    
    # ################# End of '_solve_linear_sys(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_solve_linear_sys' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_38316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38316)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_solve_linear_sys'
    return stypy_return_type_38316

# Assigning a type to the variable '_solve_linear_sys' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), '_solve_linear_sys', _solve_linear_sys)

@norecursion
def _analytical_solution(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_analytical_solution'
    module_type_store = module_type_store.open_function_context('_analytical_solution', 117, 0, False)
    
    # Passed parameters checking function
    _analytical_solution.stypy_localization = localization
    _analytical_solution.stypy_type_of_self = None
    _analytical_solution.stypy_type_store = module_type_store
    _analytical_solution.stypy_function_name = '_analytical_solution'
    _analytical_solution.stypy_param_names_list = ['a', 'y0', 't']
    _analytical_solution.stypy_varargs_param_name = None
    _analytical_solution.stypy_kwargs_param_name = None
    _analytical_solution.stypy_call_defaults = defaults
    _analytical_solution.stypy_call_varargs = varargs
    _analytical_solution.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_analytical_solution', ['a', 'y0', 't'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_analytical_solution', localization, ['a', 'y0', 't'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_analytical_solution(...)' code ##################

    str_38317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, (-1)), 'str', '\n    Analytical solution to the linear differential equations dy/dt = a*y.\n\n    The solution is only valid if `a` is diagonalizable.\n\n    Returns a 2-d array with shape (len(t), len(y0)).\n    ')
    
    # Assigning a Call to a Tuple (line 125):
    
    # Assigning a Subscript to a Name (line 125):
    
    # Obtaining the type of the subscript
    int_38318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 4), 'int')
    
    # Call to eig(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'a' (line 125)
    a_38322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'a', False)
    # Processing the call keyword arguments (line 125)
    kwargs_38323 = {}
    # Getting the type of 'np' (line 125)
    np_38319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'np', False)
    # Obtaining the member 'linalg' of a type (line 125)
    linalg_38320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), np_38319, 'linalg')
    # Obtaining the member 'eig' of a type (line 125)
    eig_38321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), linalg_38320, 'eig')
    # Calling eig(args, kwargs) (line 125)
    eig_call_result_38324 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), eig_38321, *[a_38322], **kwargs_38323)
    
    # Obtaining the member '__getitem__' of a type (line 125)
    getitem___38325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 4), eig_call_result_38324, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 125)
    subscript_call_result_38326 = invoke(stypy.reporting.localization.Localization(__file__, 125, 4), getitem___38325, int_38318)
    
    # Assigning a type to the variable 'tuple_var_assignment_38005' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_38005', subscript_call_result_38326)
    
    # Assigning a Subscript to a Name (line 125):
    
    # Obtaining the type of the subscript
    int_38327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 4), 'int')
    
    # Call to eig(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'a' (line 125)
    a_38331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'a', False)
    # Processing the call keyword arguments (line 125)
    kwargs_38332 = {}
    # Getting the type of 'np' (line 125)
    np_38328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'np', False)
    # Obtaining the member 'linalg' of a type (line 125)
    linalg_38329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), np_38328, 'linalg')
    # Obtaining the member 'eig' of a type (line 125)
    eig_38330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), linalg_38329, 'eig')
    # Calling eig(args, kwargs) (line 125)
    eig_call_result_38333 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), eig_38330, *[a_38331], **kwargs_38332)
    
    # Obtaining the member '__getitem__' of a type (line 125)
    getitem___38334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 4), eig_call_result_38333, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 125)
    subscript_call_result_38335 = invoke(stypy.reporting.localization.Localization(__file__, 125, 4), getitem___38334, int_38327)
    
    # Assigning a type to the variable 'tuple_var_assignment_38006' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_38006', subscript_call_result_38335)
    
    # Assigning a Name to a Name (line 125):
    # Getting the type of 'tuple_var_assignment_38005' (line 125)
    tuple_var_assignment_38005_38336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_38005')
    # Assigning a type to the variable 'lam' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'lam', tuple_var_assignment_38005_38336)
    
    # Assigning a Name to a Name (line 125):
    # Getting the type of 'tuple_var_assignment_38006' (line 125)
    tuple_var_assignment_38006_38337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_var_assignment_38006')
    # Assigning a type to the variable 'v' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 9), 'v', tuple_var_assignment_38006_38337)
    
    # Assigning a Call to a Name (line 126):
    
    # Assigning a Call to a Name (line 126):
    
    # Call to solve(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'v' (line 126)
    v_38341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'v', False)
    # Getting the type of 'y0' (line 126)
    y0_38342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 27), 'y0', False)
    # Processing the call keyword arguments (line 126)
    kwargs_38343 = {}
    # Getting the type of 'np' (line 126)
    np_38338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'np', False)
    # Obtaining the member 'linalg' of a type (line 126)
    linalg_38339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), np_38338, 'linalg')
    # Obtaining the member 'solve' of a type (line 126)
    solve_38340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), linalg_38339, 'solve')
    # Calling solve(args, kwargs) (line 126)
    solve_call_result_38344 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), solve_38340, *[v_38341, y0_38342], **kwargs_38343)
    
    # Assigning a type to the variable 'c' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'c', solve_call_result_38344)
    
    # Assigning a BinOp to a Name (line 127):
    
    # Assigning a BinOp to a Name (line 127):
    # Getting the type of 'c' (line 127)
    c_38345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'c')
    
    # Call to exp(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'lam' (line 127)
    lam_38348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'lam', False)
    
    # Call to reshape(...): (line 127)
    # Processing the call arguments (line 127)
    int_38351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 35), 'int')
    int_38352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 39), 'int')
    # Processing the call keyword arguments (line 127)
    kwargs_38353 = {}
    # Getting the type of 't' (line 127)
    t_38349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 25), 't', False)
    # Obtaining the member 'reshape' of a type (line 127)
    reshape_38350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 25), t_38349, 'reshape')
    # Calling reshape(args, kwargs) (line 127)
    reshape_call_result_38354 = invoke(stypy.reporting.localization.Localization(__file__, 127, 25), reshape_38350, *[int_38351, int_38352], **kwargs_38353)
    
    # Applying the binary operator '*' (line 127)
    result_mul_38355 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 19), '*', lam_38348, reshape_call_result_38354)
    
    # Processing the call keyword arguments (line 127)
    kwargs_38356 = {}
    # Getting the type of 'np' (line 127)
    np_38346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'np', False)
    # Obtaining the member 'exp' of a type (line 127)
    exp_38347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), np_38346, 'exp')
    # Calling exp(args, kwargs) (line 127)
    exp_call_result_38357 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), exp_38347, *[result_mul_38355], **kwargs_38356)
    
    # Applying the binary operator '*' (line 127)
    result_mul_38358 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 8), '*', c_38345, exp_call_result_38357)
    
    # Assigning a type to the variable 'e' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'e', result_mul_38358)
    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to dot(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'v' (line 128)
    v_38361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'v', False)
    # Obtaining the member 'T' of a type (line 128)
    T_38362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 16), v_38361, 'T')
    # Processing the call keyword arguments (line 128)
    kwargs_38363 = {}
    # Getting the type of 'e' (line 128)
    e_38359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 10), 'e', False)
    # Obtaining the member 'dot' of a type (line 128)
    dot_38360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 10), e_38359, 'dot')
    # Calling dot(args, kwargs) (line 128)
    dot_call_result_38364 = invoke(stypy.reporting.localization.Localization(__file__, 128, 10), dot_38360, *[T_38362], **kwargs_38363)
    
    # Assigning a type to the variable 'sol' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'sol', dot_call_result_38364)
    # Getting the type of 'sol' (line 129)
    sol_38365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'sol')
    # Assigning a type to the variable 'stypy_return_type' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type', sol_38365)
    
    # ################# End of '_analytical_solution(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_analytical_solution' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_38366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38366)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_analytical_solution'
    return stypy_return_type_38366

# Assigning a type to the variable '_analytical_solution' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), '_analytical_solution', _analytical_solution)

@norecursion
def test_banded_ode_solvers(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_banded_ode_solvers'
    module_type_store = module_type_store.open_function_context('test_banded_ode_solvers', 132, 0, False)
    
    # Passed parameters checking function
    test_banded_ode_solvers.stypy_localization = localization
    test_banded_ode_solvers.stypy_type_of_self = None
    test_banded_ode_solvers.stypy_type_store = module_type_store
    test_banded_ode_solvers.stypy_function_name = 'test_banded_ode_solvers'
    test_banded_ode_solvers.stypy_param_names_list = []
    test_banded_ode_solvers.stypy_varargs_param_name = None
    test_banded_ode_solvers.stypy_kwargs_param_name = None
    test_banded_ode_solvers.stypy_call_defaults = defaults
    test_banded_ode_solvers.stypy_call_varargs = varargs
    test_banded_ode_solvers.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_banded_ode_solvers', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_banded_ode_solvers', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_banded_ode_solvers(...)' code ##################

    
    # Assigning a Call to a Name (line 136):
    
    # Assigning a Call to a Name (line 136):
    
    # Call to linspace(...): (line 136)
    # Processing the call arguments (line 136)
    int_38369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 26), 'int')
    float_38370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 29), 'float')
    int_38371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 34), 'int')
    # Processing the call keyword arguments (line 136)
    kwargs_38372 = {}
    # Getting the type of 'np' (line 136)
    np_38367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 14), 'np', False)
    # Obtaining the member 'linspace' of a type (line 136)
    linspace_38368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 14), np_38367, 'linspace')
    # Calling linspace(args, kwargs) (line 136)
    linspace_call_result_38373 = invoke(stypy.reporting.localization.Localization(__file__, 136, 14), linspace_38368, *[int_38369, float_38370, int_38371], **kwargs_38372)
    
    # Assigning a type to the variable 't_exact' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 't_exact', linspace_call_result_38373)
    
    # Assigning a Call to a Name (line 141):
    
    # Assigning a Call to a Name (line 141):
    
    # Call to array(...): (line 141)
    # Processing the call arguments (line 141)
    
    # Obtaining an instance of the builtin type 'list' (line 141)
    list_38376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 141)
    # Adding element type (line 141)
    
    # Obtaining an instance of the builtin type 'list' (line 141)
    list_38377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 141)
    # Adding element type (line 141)
    float_38378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 23), list_38377, float_38378)
    # Adding element type (line 141)
    float_38379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 23), list_38377, float_38379)
    # Adding element type (line 141)
    float_38380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 23), list_38377, float_38380)
    # Adding element type (line 141)
    float_38381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 23), list_38377, float_38381)
    # Adding element type (line 141)
    float_38382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 45), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 23), list_38377, float_38382)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 22), list_38376, list_38377)
    # Adding element type (line 141)
    
    # Obtaining an instance of the builtin type 'list' (line 142)
    list_38383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 142)
    # Adding element type (line 142)
    float_38384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 23), list_38383, float_38384)
    # Adding element type (line 142)
    float_38385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 23), list_38383, float_38385)
    # Adding element type (line 142)
    float_38386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 23), list_38383, float_38386)
    # Adding element type (line 142)
    float_38387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 23), list_38383, float_38387)
    # Adding element type (line 142)
    float_38388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 45), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 23), list_38383, float_38388)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 22), list_38376, list_38383)
    # Adding element type (line 141)
    
    # Obtaining an instance of the builtin type 'list' (line 143)
    list_38389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 143)
    # Adding element type (line 143)
    float_38390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 23), list_38389, float_38390)
    # Adding element type (line 143)
    float_38391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 23), list_38389, float_38391)
    # Adding element type (line 143)
    float_38392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 23), list_38389, float_38392)
    # Adding element type (line 143)
    float_38393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 23), list_38389, float_38393)
    # Adding element type (line 143)
    float_38394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 45), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 23), list_38389, float_38394)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 22), list_38376, list_38389)
    # Adding element type (line 141)
    
    # Obtaining an instance of the builtin type 'list' (line 144)
    list_38395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 144)
    # Adding element type (line 144)
    float_38396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 23), list_38395, float_38396)
    # Adding element type (line 144)
    float_38397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 23), list_38395, float_38397)
    # Adding element type (line 144)
    float_38398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 23), list_38395, float_38398)
    # Adding element type (line 144)
    float_38399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 23), list_38395, float_38399)
    # Adding element type (line 144)
    float_38400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 46), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 23), list_38395, float_38400)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 22), list_38376, list_38395)
    # Adding element type (line 141)
    
    # Obtaining an instance of the builtin type 'list' (line 145)
    list_38401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 145)
    # Adding element type (line 145)
    float_38402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 23), list_38401, float_38402)
    # Adding element type (line 145)
    float_38403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 23), list_38401, float_38403)
    # Adding element type (line 145)
    float_38404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 23), list_38401, float_38404)
    # Adding element type (line 145)
    float_38405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 23), list_38401, float_38405)
    # Adding element type (line 145)
    float_38406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 44), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 23), list_38401, float_38406)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 22), list_38376, list_38401)
    
    # Processing the call keyword arguments (line 141)
    kwargs_38407 = {}
    # Getting the type of 'np' (line 141)
    np_38374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 141)
    array_38375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 13), np_38374, 'array')
    # Calling array(args, kwargs) (line 141)
    array_call_result_38408 = invoke(stypy.reporting.localization.Localization(__file__, 141, 13), array_38375, *[list_38376], **kwargs_38407)
    
    # Assigning a type to the variable 'a_real' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'a_real', array_call_result_38408)
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to triu(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'a_real' (line 148)
    a_real_38411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'a_real', False)
    # Processing the call keyword arguments (line 148)
    kwargs_38412 = {}
    # Getting the type of 'np' (line 148)
    np_38409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'np', False)
    # Obtaining the member 'triu' of a type (line 148)
    triu_38410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 19), np_38409, 'triu')
    # Calling triu(args, kwargs) (line 148)
    triu_call_result_38413 = invoke(stypy.reporting.localization.Localization(__file__, 148, 19), triu_38410, *[a_real_38411], **kwargs_38412)
    
    # Assigning a type to the variable 'a_real_upper' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'a_real_upper', triu_call_result_38413)
    
    # Assigning a Call to a Name (line 151):
    
    # Assigning a Call to a Name (line 151):
    
    # Call to tril(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'a_real' (line 151)
    a_real_38416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 27), 'a_real', False)
    # Processing the call keyword arguments (line 151)
    kwargs_38417 = {}
    # Getting the type of 'np' (line 151)
    np_38414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'np', False)
    # Obtaining the member 'tril' of a type (line 151)
    tril_38415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 19), np_38414, 'tril')
    # Calling tril(args, kwargs) (line 151)
    tril_call_result_38418 = invoke(stypy.reporting.localization.Localization(__file__, 151, 19), tril_38415, *[a_real_38416], **kwargs_38417)
    
    # Assigning a type to the variable 'a_real_lower' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'a_real_lower', tril_call_result_38418)
    
    # Assigning a Call to a Name (line 154):
    
    # Assigning a Call to a Name (line 154):
    
    # Call to triu(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'a_real_lower' (line 154)
    a_real_lower_38421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'a_real_lower', False)
    # Processing the call keyword arguments (line 154)
    kwargs_38422 = {}
    # Getting the type of 'np' (line 154)
    np_38419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 18), 'np', False)
    # Obtaining the member 'triu' of a type (line 154)
    triu_38420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 18), np_38419, 'triu')
    # Calling triu(args, kwargs) (line 154)
    triu_call_result_38423 = invoke(stypy.reporting.localization.Localization(__file__, 154, 18), triu_38420, *[a_real_lower_38421], **kwargs_38422)
    
    # Assigning a type to the variable 'a_real_diag' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'a_real_diag', triu_call_result_38423)
    
    # Assigning a List to a Name (line 156):
    
    # Assigning a List to a Name (line 156):
    
    # Obtaining an instance of the builtin type 'list' (line 156)
    list_38424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 156)
    # Adding element type (line 156)
    # Getting the type of 'a_real' (line 156)
    a_real_38425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'a_real')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 20), list_38424, a_real_38425)
    # Adding element type (line 156)
    # Getting the type of 'a_real_upper' (line 156)
    a_real_upper_38426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 29), 'a_real_upper')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 20), list_38424, a_real_upper_38426)
    # Adding element type (line 156)
    # Getting the type of 'a_real_lower' (line 156)
    a_real_lower_38427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 43), 'a_real_lower')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 20), list_38424, a_real_lower_38427)
    # Adding element type (line 156)
    # Getting the type of 'a_real_diag' (line 156)
    a_real_diag_38428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 57), 'a_real_diag')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 20), list_38424, a_real_diag_38428)
    
    # Assigning a type to the variable 'real_matrices' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'real_matrices', list_38424)
    
    # Assigning a List to a Name (line 157):
    
    # Assigning a List to a Name (line 157):
    
    # Obtaining an instance of the builtin type 'list' (line 157)
    list_38429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 157)
    
    # Assigning a type to the variable 'real_solutions' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'real_solutions', list_38429)
    
    # Getting the type of 'real_matrices' (line 159)
    real_matrices_38430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'real_matrices')
    # Testing the type of a for loop iterable (line 159)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 159, 4), real_matrices_38430)
    # Getting the type of the for loop variable (line 159)
    for_loop_var_38431 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 159, 4), real_matrices_38430)
    # Assigning a type to the variable 'a' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'a', for_loop_var_38431)
    # SSA begins for a for statement (line 159)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 160):
    
    # Assigning a Call to a Name (line 160):
    
    # Call to arange(...): (line 160)
    # Processing the call arguments (line 160)
    int_38434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 23), 'int')
    
    # Obtaining the type of the subscript
    int_38435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 34), 'int')
    # Getting the type of 'a' (line 160)
    a_38436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'a', False)
    # Obtaining the member 'shape' of a type (line 160)
    shape_38437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 26), a_38436, 'shape')
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___38438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 26), shape_38437, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_38439 = invoke(stypy.reporting.localization.Localization(__file__, 160, 26), getitem___38438, int_38435)
    
    int_38440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 39), 'int')
    # Applying the binary operator '+' (line 160)
    result_add_38441 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 26), '+', subscript_call_result_38439, int_38440)
    
    # Processing the call keyword arguments (line 160)
    kwargs_38442 = {}
    # Getting the type of 'np' (line 160)
    np_38432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 13), 'np', False)
    # Obtaining the member 'arange' of a type (line 160)
    arange_38433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 13), np_38432, 'arange')
    # Calling arange(args, kwargs) (line 160)
    arange_call_result_38443 = invoke(stypy.reporting.localization.Localization(__file__, 160, 13), arange_38433, *[int_38434, result_add_38441], **kwargs_38442)
    
    # Assigning a type to the variable 'y0' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'y0', arange_call_result_38443)
    
    # Assigning a Call to a Name (line 161):
    
    # Assigning a Call to a Name (line 161):
    
    # Call to _analytical_solution(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'a' (line 161)
    a_38445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 39), 'a', False)
    # Getting the type of 'y0' (line 161)
    y0_38446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 42), 'y0', False)
    # Getting the type of 't_exact' (line 161)
    t_exact_38447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 46), 't_exact', False)
    # Processing the call keyword arguments (line 161)
    kwargs_38448 = {}
    # Getting the type of '_analytical_solution' (line 161)
    _analytical_solution_38444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 18), '_analytical_solution', False)
    # Calling _analytical_solution(args, kwargs) (line 161)
    _analytical_solution_call_result_38449 = invoke(stypy.reporting.localization.Localization(__file__, 161, 18), _analytical_solution_38444, *[a_38445, y0_38446, t_exact_38447], **kwargs_38448)
    
    # Assigning a type to the variable 'y_exact' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'y_exact', _analytical_solution_call_result_38449)
    
    # Call to append(...): (line 162)
    # Processing the call arguments (line 162)
    
    # Obtaining an instance of the builtin type 'tuple' (line 162)
    tuple_38452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 162)
    # Adding element type (line 162)
    # Getting the type of 'y0' (line 162)
    y0_38453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 31), 'y0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 31), tuple_38452, y0_38453)
    # Adding element type (line 162)
    # Getting the type of 't_exact' (line 162)
    t_exact_38454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 35), 't_exact', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 31), tuple_38452, t_exact_38454)
    # Adding element type (line 162)
    # Getting the type of 'y_exact' (line 162)
    y_exact_38455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 44), 'y_exact', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 31), tuple_38452, y_exact_38455)
    
    # Processing the call keyword arguments (line 162)
    kwargs_38456 = {}
    # Getting the type of 'real_solutions' (line 162)
    real_solutions_38450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'real_solutions', False)
    # Obtaining the member 'append' of a type (line 162)
    append_38451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), real_solutions_38450, 'append')
    # Calling append(args, kwargs) (line 162)
    append_call_result_38457 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), append_38451, *[tuple_38452], **kwargs_38456)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def check_real(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_real'
        module_type_store = module_type_store.open_function_context('check_real', 164, 4, False)
        
        # Passed parameters checking function
        check_real.stypy_localization = localization
        check_real.stypy_type_of_self = None
        check_real.stypy_type_store = module_type_store
        check_real.stypy_function_name = 'check_real'
        check_real.stypy_param_names_list = ['idx', 'solver', 'meth', 'use_jac', 'with_jac', 'banded']
        check_real.stypy_varargs_param_name = None
        check_real.stypy_kwargs_param_name = None
        check_real.stypy_call_defaults = defaults
        check_real.stypy_call_varargs = varargs
        check_real.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check_real', ['idx', 'solver', 'meth', 'use_jac', 'with_jac', 'banded'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_real', localization, ['idx', 'solver', 'meth', 'use_jac', 'with_jac', 'banded'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_real(...)' code ##################

        
        # Assigning a Subscript to a Name (line 165):
        
        # Assigning a Subscript to a Name (line 165):
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 165)
        idx_38458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 26), 'idx')
        # Getting the type of 'real_matrices' (line 165)
        real_matrices_38459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'real_matrices')
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___38460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), real_matrices_38459, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_38461 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), getitem___38460, idx_38458)
        
        # Assigning a type to the variable 'a' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'a', subscript_call_result_38461)
        
        # Assigning a Subscript to a Tuple (line 166):
        
        # Assigning a Subscript to a Name (line 166):
        
        # Obtaining the type of the subscript
        int_38462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 166)
        idx_38463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 46), 'idx')
        # Getting the type of 'real_solutions' (line 166)
        real_solutions_38464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 31), 'real_solutions')
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___38465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 31), real_solutions_38464, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_38466 = invoke(stypy.reporting.localization.Localization(__file__, 166, 31), getitem___38465, idx_38463)
        
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___38467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), subscript_call_result_38466, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_38468 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___38467, int_38462)
        
        # Assigning a type to the variable 'tuple_var_assignment_38007' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_38007', subscript_call_result_38468)
        
        # Assigning a Subscript to a Name (line 166):
        
        # Obtaining the type of the subscript
        int_38469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 166)
        idx_38470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 46), 'idx')
        # Getting the type of 'real_solutions' (line 166)
        real_solutions_38471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 31), 'real_solutions')
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___38472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 31), real_solutions_38471, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_38473 = invoke(stypy.reporting.localization.Localization(__file__, 166, 31), getitem___38472, idx_38470)
        
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___38474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), subscript_call_result_38473, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_38475 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___38474, int_38469)
        
        # Assigning a type to the variable 'tuple_var_assignment_38008' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_38008', subscript_call_result_38475)
        
        # Assigning a Subscript to a Name (line 166):
        
        # Obtaining the type of the subscript
        int_38476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 166)
        idx_38477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 46), 'idx')
        # Getting the type of 'real_solutions' (line 166)
        real_solutions_38478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 31), 'real_solutions')
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___38479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 31), real_solutions_38478, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_38480 = invoke(stypy.reporting.localization.Localization(__file__, 166, 31), getitem___38479, idx_38477)
        
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___38481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), subscript_call_result_38480, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_38482 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___38481, int_38476)
        
        # Assigning a type to the variable 'tuple_var_assignment_38009' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_38009', subscript_call_result_38482)
        
        # Assigning a Name to a Name (line 166):
        # Getting the type of 'tuple_var_assignment_38007' (line 166)
        tuple_var_assignment_38007_38483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_38007')
        # Assigning a type to the variable 'y0' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'y0', tuple_var_assignment_38007_38483)
        
        # Assigning a Name to a Name (line 166):
        # Getting the type of 'tuple_var_assignment_38008' (line 166)
        tuple_var_assignment_38008_38484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_38008')
        # Assigning a type to the variable 't_exact' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 't_exact', tuple_var_assignment_38008_38484)
        
        # Assigning a Name to a Name (line 166):
        # Getting the type of 'tuple_var_assignment_38009' (line 166)
        tuple_var_assignment_38009_38485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_38009')
        # Assigning a type to the variable 'y_exact' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'y_exact', tuple_var_assignment_38009_38485)
        
        # Assigning a Call to a Tuple (line 167):
        
        # Assigning a Subscript to a Name (line 167):
        
        # Obtaining the type of the subscript
        int_38486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 8), 'int')
        
        # Call to _solve_linear_sys(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'a' (line 167)
        a_38488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'a', False)
        # Getting the type of 'y0' (line 167)
        y0_38489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 36), 'y0', False)
        # Processing the call keyword arguments (line 167)
        
        # Obtaining the type of the subscript
        int_38490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 46), 'int')
        # Getting the type of 't_exact' (line 168)
        t_exact_38491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 38), 't_exact', False)
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___38492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 38), t_exact_38491, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_38493 = invoke(stypy.reporting.localization.Localization(__file__, 168, 38), getitem___38492, int_38490)
        
        keyword_38494 = subscript_call_result_38493
        
        # Obtaining the type of the subscript
        int_38495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 44), 'int')
        # Getting the type of 't_exact' (line 169)
        t_exact_38496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 36), 't_exact', False)
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___38497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 36), t_exact_38496, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_38498 = invoke(stypy.reporting.localization.Localization(__file__, 169, 36), getitem___38497, int_38495)
        
        
        # Obtaining the type of the subscript
        int_38499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 57), 'int')
        # Getting the type of 't_exact' (line 169)
        t_exact_38500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 49), 't_exact', False)
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___38501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 49), t_exact_38500, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_38502 = invoke(stypy.reporting.localization.Localization(__file__, 169, 49), getitem___38501, int_38499)
        
        # Applying the binary operator '-' (line 169)
        result_sub_38503 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 36), '-', subscript_call_result_38498, subscript_call_result_38502)
        
        keyword_38504 = result_sub_38503
        # Getting the type of 'solver' (line 170)
        solver_38505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 40), 'solver', False)
        keyword_38506 = solver_38505
        # Getting the type of 'meth' (line 171)
        meth_38507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 40), 'meth', False)
        keyword_38508 = meth_38507
        # Getting the type of 'use_jac' (line 172)
        use_jac_38509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 41), 'use_jac', False)
        keyword_38510 = use_jac_38509
        # Getting the type of 'with_jac' (line 173)
        with_jac_38511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 47), 'with_jac', False)
        keyword_38512 = with_jac_38511
        # Getting the type of 'banded' (line 174)
        banded_38513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 40), 'banded', False)
        keyword_38514 = banded_38513
        kwargs_38515 = {'solver': keyword_38506, 'banded': keyword_38514, 'tend': keyword_38494, 'with_jacobian': keyword_38512, 'use_jac': keyword_38510, 'dt': keyword_38504, 'method': keyword_38508}
        # Getting the type of '_solve_linear_sys' (line 167)
        _solve_linear_sys_38487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), '_solve_linear_sys', False)
        # Calling _solve_linear_sys(args, kwargs) (line 167)
        _solve_linear_sys_call_result_38516 = invoke(stypy.reporting.localization.Localization(__file__, 167, 15), _solve_linear_sys_38487, *[a_38488, y0_38489], **kwargs_38515)
        
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___38517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), _solve_linear_sys_call_result_38516, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_38518 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), getitem___38517, int_38486)
        
        # Assigning a type to the variable 'tuple_var_assignment_38010' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_38010', subscript_call_result_38518)
        
        # Assigning a Subscript to a Name (line 167):
        
        # Obtaining the type of the subscript
        int_38519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 8), 'int')
        
        # Call to _solve_linear_sys(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'a' (line 167)
        a_38521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'a', False)
        # Getting the type of 'y0' (line 167)
        y0_38522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 36), 'y0', False)
        # Processing the call keyword arguments (line 167)
        
        # Obtaining the type of the subscript
        int_38523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 46), 'int')
        # Getting the type of 't_exact' (line 168)
        t_exact_38524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 38), 't_exact', False)
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___38525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 38), t_exact_38524, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_38526 = invoke(stypy.reporting.localization.Localization(__file__, 168, 38), getitem___38525, int_38523)
        
        keyword_38527 = subscript_call_result_38526
        
        # Obtaining the type of the subscript
        int_38528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 44), 'int')
        # Getting the type of 't_exact' (line 169)
        t_exact_38529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 36), 't_exact', False)
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___38530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 36), t_exact_38529, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_38531 = invoke(stypy.reporting.localization.Localization(__file__, 169, 36), getitem___38530, int_38528)
        
        
        # Obtaining the type of the subscript
        int_38532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 57), 'int')
        # Getting the type of 't_exact' (line 169)
        t_exact_38533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 49), 't_exact', False)
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___38534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 49), t_exact_38533, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_38535 = invoke(stypy.reporting.localization.Localization(__file__, 169, 49), getitem___38534, int_38532)
        
        # Applying the binary operator '-' (line 169)
        result_sub_38536 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 36), '-', subscript_call_result_38531, subscript_call_result_38535)
        
        keyword_38537 = result_sub_38536
        # Getting the type of 'solver' (line 170)
        solver_38538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 40), 'solver', False)
        keyword_38539 = solver_38538
        # Getting the type of 'meth' (line 171)
        meth_38540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 40), 'meth', False)
        keyword_38541 = meth_38540
        # Getting the type of 'use_jac' (line 172)
        use_jac_38542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 41), 'use_jac', False)
        keyword_38543 = use_jac_38542
        # Getting the type of 'with_jac' (line 173)
        with_jac_38544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 47), 'with_jac', False)
        keyword_38545 = with_jac_38544
        # Getting the type of 'banded' (line 174)
        banded_38546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 40), 'banded', False)
        keyword_38547 = banded_38546
        kwargs_38548 = {'solver': keyword_38539, 'banded': keyword_38547, 'tend': keyword_38527, 'with_jacobian': keyword_38545, 'use_jac': keyword_38543, 'dt': keyword_38537, 'method': keyword_38541}
        # Getting the type of '_solve_linear_sys' (line 167)
        _solve_linear_sys_38520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), '_solve_linear_sys', False)
        # Calling _solve_linear_sys(args, kwargs) (line 167)
        _solve_linear_sys_call_result_38549 = invoke(stypy.reporting.localization.Localization(__file__, 167, 15), _solve_linear_sys_38520, *[a_38521, y0_38522], **kwargs_38548)
        
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___38550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), _solve_linear_sys_call_result_38549, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_38551 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), getitem___38550, int_38519)
        
        # Assigning a type to the variable 'tuple_var_assignment_38011' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_38011', subscript_call_result_38551)
        
        # Assigning a Name to a Name (line 167):
        # Getting the type of 'tuple_var_assignment_38010' (line 167)
        tuple_var_assignment_38010_38552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_38010')
        # Assigning a type to the variable 't' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 't', tuple_var_assignment_38010_38552)
        
        # Assigning a Name to a Name (line 167):
        # Getting the type of 'tuple_var_assignment_38011' (line 167)
        tuple_var_assignment_38011_38553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_38011')
        # Assigning a type to the variable 'y' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'y', tuple_var_assignment_38011_38553)
        
        # Call to assert_allclose(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 't' (line 175)
        t_38555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 24), 't', False)
        # Getting the type of 't_exact' (line 175)
        t_exact_38556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 27), 't_exact', False)
        # Processing the call keyword arguments (line 175)
        kwargs_38557 = {}
        # Getting the type of 'assert_allclose' (line 175)
        assert_allclose_38554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 175)
        assert_allclose_call_result_38558 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), assert_allclose_38554, *[t_38555, t_exact_38556], **kwargs_38557)
        
        
        # Call to assert_allclose(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'y' (line 176)
        y_38560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'y', False)
        # Getting the type of 'y_exact' (line 176)
        y_exact_38561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 27), 'y_exact', False)
        # Processing the call keyword arguments (line 176)
        kwargs_38562 = {}
        # Getting the type of 'assert_allclose' (line 176)
        assert_allclose_38559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 176)
        assert_allclose_call_result_38563 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), assert_allclose_38559, *[y_38560, y_exact_38561], **kwargs_38562)
        
        
        # ################# End of 'check_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_real' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_38564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38564)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_real'
        return stypy_return_type_38564

    # Assigning a type to the variable 'check_real' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'check_real', check_real)
    
    
    # Call to range(...): (line 178)
    # Processing the call arguments (line 178)
    
    # Call to len(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'real_matrices' (line 178)
    real_matrices_38567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 25), 'real_matrices', False)
    # Processing the call keyword arguments (line 178)
    kwargs_38568 = {}
    # Getting the type of 'len' (line 178)
    len_38566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'len', False)
    # Calling len(args, kwargs) (line 178)
    len_call_result_38569 = invoke(stypy.reporting.localization.Localization(__file__, 178, 21), len_38566, *[real_matrices_38567], **kwargs_38568)
    
    # Processing the call keyword arguments (line 178)
    kwargs_38570 = {}
    # Getting the type of 'range' (line 178)
    range_38565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'range', False)
    # Calling range(args, kwargs) (line 178)
    range_call_result_38571 = invoke(stypy.reporting.localization.Localization(__file__, 178, 15), range_38565, *[len_call_result_38569], **kwargs_38570)
    
    # Testing the type of a for loop iterable (line 178)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 178, 4), range_call_result_38571)
    # Getting the type of the for loop variable (line 178)
    for_loop_var_38572 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 178, 4), range_call_result_38571)
    # Assigning a type to the variable 'idx' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'idx', for_loop_var_38572)
    # SSA begins for a for statement (line 178)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a List to a Name (line 179):
    
    # Assigning a List to a Name (line 179):
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_38573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    # Adding element type (line 179)
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_38574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    # Adding element type (line 179)
    str_38575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 14), 'str', 'vode')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 13), list_38574, str_38575)
    # Adding element type (line 179)
    str_38576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 22), 'str', 'lsoda')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 13), list_38574, str_38576)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 12), list_38573, list_38574)
    # Adding element type (line 179)
    
    # Obtaining an instance of the builtin type 'list' (line 180)
    list_38577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 180)
    # Adding element type (line 180)
    str_38578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 14), 'str', 'bdf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 13), list_38577, str_38578)
    # Adding element type (line 180)
    str_38579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 21), 'str', 'adams')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 13), list_38577, str_38579)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 12), list_38573, list_38577)
    # Adding element type (line 179)
    
    # Obtaining an instance of the builtin type 'list' (line 181)
    list_38580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 181)
    # Adding element type (line 181)
    # Getting the type of 'False' (line 181)
    False_38581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 14), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 13), list_38580, False_38581)
    # Adding element type (line 181)
    # Getting the type of 'True' (line 181)
    True_38582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 21), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 13), list_38580, True_38582)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 12), list_38573, list_38580)
    # Adding element type (line 179)
    
    # Obtaining an instance of the builtin type 'list' (line 182)
    list_38583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 182)
    # Adding element type (line 182)
    # Getting the type of 'False' (line 182)
    False_38584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 14), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 13), list_38583, False_38584)
    # Adding element type (line 182)
    # Getting the type of 'True' (line 182)
    True_38585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 21), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 13), list_38583, True_38585)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 12), list_38573, list_38583)
    # Adding element type (line 179)
    
    # Obtaining an instance of the builtin type 'list' (line 183)
    list_38586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 183)
    # Adding element type (line 183)
    # Getting the type of 'False' (line 183)
    False_38587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 14), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 13), list_38586, False_38587)
    # Adding element type (line 183)
    # Getting the type of 'True' (line 183)
    True_38588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 13), list_38586, True_38588)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 12), list_38573, list_38586)
    
    # Assigning a type to the variable 'p' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'p', list_38573)
    
    
    # Call to product(...): (line 184)
    # Getting the type of 'p' (line 184)
    p_38591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 74), 'p', False)
    # Processing the call keyword arguments (line 184)
    kwargs_38592 = {}
    # Getting the type of 'itertools' (line 184)
    itertools_38589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 55), 'itertools', False)
    # Obtaining the member 'product' of a type (line 184)
    product_38590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 55), itertools_38589, 'product')
    # Calling product(args, kwargs) (line 184)
    product_call_result_38593 = invoke(stypy.reporting.localization.Localization(__file__, 184, 55), product_38590, *[p_38591], **kwargs_38592)
    
    # Testing the type of a for loop iterable (line 184)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 184, 8), product_call_result_38593)
    # Getting the type of the for loop variable (line 184)
    for_loop_var_38594 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 184, 8), product_call_result_38593)
    # Assigning a type to the variable 'solver' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'solver', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 8), for_loop_var_38594))
    # Assigning a type to the variable 'meth' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'meth', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 8), for_loop_var_38594))
    # Assigning a type to the variable 'use_jac' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'use_jac', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 8), for_loop_var_38594))
    # Assigning a type to the variable 'with_jac' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'with_jac', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 8), for_loop_var_38594))
    # Assigning a type to the variable 'banded' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'banded', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 8), for_loop_var_38594))
    # SSA begins for a for statement (line 184)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check_real(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'idx' (line 185)
    idx_38596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'idx', False)
    # Getting the type of 'solver' (line 185)
    solver_38597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 28), 'solver', False)
    # Getting the type of 'meth' (line 185)
    meth_38598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 36), 'meth', False)
    # Getting the type of 'use_jac' (line 185)
    use_jac_38599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 42), 'use_jac', False)
    # Getting the type of 'with_jac' (line 185)
    with_jac_38600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 51), 'with_jac', False)
    # Getting the type of 'banded' (line 185)
    banded_38601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 61), 'banded', False)
    # Processing the call keyword arguments (line 185)
    kwargs_38602 = {}
    # Getting the type of 'check_real' (line 185)
    check_real_38595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'check_real', False)
    # Calling check_real(args, kwargs) (line 185)
    check_real_call_result_38603 = invoke(stypy.reporting.localization.Localization(__file__, 185, 12), check_real_38595, *[idx_38596, solver_38597, meth_38598, use_jac_38599, with_jac_38600, banded_38601], **kwargs_38602)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 190):
    
    # Assigning a BinOp to a Name (line 190):
    # Getting the type of 'a_real' (line 190)
    a_real_38604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'a_real')
    complex_38605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 25), 'complex')
    # Getting the type of 'a_real' (line 190)
    a_real_38606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 32), 'a_real')
    # Applying the binary operator '*' (line 190)
    result_mul_38607 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 25), '*', complex_38605, a_real_38606)
    
    # Applying the binary operator '-' (line 190)
    result_sub_38608 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 16), '-', a_real_38604, result_mul_38607)
    
    # Assigning a type to the variable 'a_complex' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'a_complex', result_sub_38608)
    
    # Assigning a Call to a Name (line 193):
    
    # Assigning a Call to a Name (line 193):
    
    # Call to diag(...): (line 193)
    # Processing the call arguments (line 193)
    
    # Call to diag(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'a_complex' (line 193)
    a_complex_38613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 37), 'a_complex', False)
    # Processing the call keyword arguments (line 193)
    kwargs_38614 = {}
    # Getting the type of 'np' (line 193)
    np_38611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 29), 'np', False)
    # Obtaining the member 'diag' of a type (line 193)
    diag_38612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 29), np_38611, 'diag')
    # Calling diag(args, kwargs) (line 193)
    diag_call_result_38615 = invoke(stypy.reporting.localization.Localization(__file__, 193, 29), diag_38612, *[a_complex_38613], **kwargs_38614)
    
    # Processing the call keyword arguments (line 193)
    kwargs_38616 = {}
    # Getting the type of 'np' (line 193)
    np_38609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 21), 'np', False)
    # Obtaining the member 'diag' of a type (line 193)
    diag_38610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 21), np_38609, 'diag')
    # Calling diag(args, kwargs) (line 193)
    diag_call_result_38617 = invoke(stypy.reporting.localization.Localization(__file__, 193, 21), diag_38610, *[diag_call_result_38615], **kwargs_38616)
    
    # Assigning a type to the variable 'a_complex_diag' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'a_complex_diag', diag_call_result_38617)
    
    # Assigning a List to a Name (line 195):
    
    # Assigning a List to a Name (line 195):
    
    # Obtaining an instance of the builtin type 'list' (line 195)
    list_38618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 195)
    # Adding element type (line 195)
    # Getting the type of 'a_complex' (line 195)
    a_complex_38619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 24), 'a_complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 23), list_38618, a_complex_38619)
    # Adding element type (line 195)
    # Getting the type of 'a_complex_diag' (line 195)
    a_complex_diag_38620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 35), 'a_complex_diag')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 23), list_38618, a_complex_diag_38620)
    
    # Assigning a type to the variable 'complex_matrices' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'complex_matrices', list_38618)
    
    # Assigning a List to a Name (line 196):
    
    # Assigning a List to a Name (line 196):
    
    # Obtaining an instance of the builtin type 'list' (line 196)
    list_38621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 196)
    
    # Assigning a type to the variable 'complex_solutions' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'complex_solutions', list_38621)
    
    # Getting the type of 'complex_matrices' (line 198)
    complex_matrices_38622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 13), 'complex_matrices')
    # Testing the type of a for loop iterable (line 198)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 198, 4), complex_matrices_38622)
    # Getting the type of the for loop variable (line 198)
    for_loop_var_38623 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 198, 4), complex_matrices_38622)
    # Assigning a type to the variable 'a' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'a', for_loop_var_38623)
    # SSA begins for a for statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 199):
    
    # Assigning a BinOp to a Name (line 199):
    
    # Call to arange(...): (line 199)
    # Processing the call arguments (line 199)
    int_38626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 23), 'int')
    
    # Obtaining the type of the subscript
    int_38627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 34), 'int')
    # Getting the type of 'a' (line 199)
    a_38628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 26), 'a', False)
    # Obtaining the member 'shape' of a type (line 199)
    shape_38629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 26), a_38628, 'shape')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___38630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 26), shape_38629, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_38631 = invoke(stypy.reporting.localization.Localization(__file__, 199, 26), getitem___38630, int_38627)
    
    int_38632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 39), 'int')
    # Applying the binary operator '+' (line 199)
    result_add_38633 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 26), '+', subscript_call_result_38631, int_38632)
    
    # Processing the call keyword arguments (line 199)
    kwargs_38634 = {}
    # Getting the type of 'np' (line 199)
    np_38624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 13), 'np', False)
    # Obtaining the member 'arange' of a type (line 199)
    arange_38625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 13), np_38624, 'arange')
    # Calling arange(args, kwargs) (line 199)
    arange_call_result_38635 = invoke(stypy.reporting.localization.Localization(__file__, 199, 13), arange_38625, *[int_38626, result_add_38633], **kwargs_38634)
    
    complex_38636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 44), 'complex')
    # Applying the binary operator '+' (line 199)
    result_add_38637 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 13), '+', arange_call_result_38635, complex_38636)
    
    # Assigning a type to the variable 'y0' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'y0', result_add_38637)
    
    # Assigning a Call to a Name (line 200):
    
    # Assigning a Call to a Name (line 200):
    
    # Call to _analytical_solution(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'a' (line 200)
    a_38639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 39), 'a', False)
    # Getting the type of 'y0' (line 200)
    y0_38640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 42), 'y0', False)
    # Getting the type of 't_exact' (line 200)
    t_exact_38641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 46), 't_exact', False)
    # Processing the call keyword arguments (line 200)
    kwargs_38642 = {}
    # Getting the type of '_analytical_solution' (line 200)
    _analytical_solution_38638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), '_analytical_solution', False)
    # Calling _analytical_solution(args, kwargs) (line 200)
    _analytical_solution_call_result_38643 = invoke(stypy.reporting.localization.Localization(__file__, 200, 18), _analytical_solution_38638, *[a_38639, y0_38640, t_exact_38641], **kwargs_38642)
    
    # Assigning a type to the variable 'y_exact' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'y_exact', _analytical_solution_call_result_38643)
    
    # Call to append(...): (line 201)
    # Processing the call arguments (line 201)
    
    # Obtaining an instance of the builtin type 'tuple' (line 201)
    tuple_38646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 201)
    # Adding element type (line 201)
    # Getting the type of 'y0' (line 201)
    y0_38647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 34), 'y0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 34), tuple_38646, y0_38647)
    # Adding element type (line 201)
    # Getting the type of 't_exact' (line 201)
    t_exact_38648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 38), 't_exact', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 34), tuple_38646, t_exact_38648)
    # Adding element type (line 201)
    # Getting the type of 'y_exact' (line 201)
    y_exact_38649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 47), 'y_exact', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 34), tuple_38646, y_exact_38649)
    
    # Processing the call keyword arguments (line 201)
    kwargs_38650 = {}
    # Getting the type of 'complex_solutions' (line 201)
    complex_solutions_38644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'complex_solutions', False)
    # Obtaining the member 'append' of a type (line 201)
    append_38645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), complex_solutions_38644, 'append')
    # Calling append(args, kwargs) (line 201)
    append_call_result_38651 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), append_38645, *[tuple_38646], **kwargs_38650)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def check_complex(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_complex'
        module_type_store = module_type_store.open_function_context('check_complex', 203, 4, False)
        
        # Passed parameters checking function
        check_complex.stypy_localization = localization
        check_complex.stypy_type_of_self = None
        check_complex.stypy_type_store = module_type_store
        check_complex.stypy_function_name = 'check_complex'
        check_complex.stypy_param_names_list = ['idx', 'solver', 'meth', 'use_jac', 'with_jac', 'banded']
        check_complex.stypy_varargs_param_name = None
        check_complex.stypy_kwargs_param_name = None
        check_complex.stypy_call_defaults = defaults
        check_complex.stypy_call_varargs = varargs
        check_complex.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check_complex', ['idx', 'solver', 'meth', 'use_jac', 'with_jac', 'banded'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_complex', localization, ['idx', 'solver', 'meth', 'use_jac', 'with_jac', 'banded'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_complex(...)' code ##################

        
        # Assigning a Subscript to a Name (line 204):
        
        # Assigning a Subscript to a Name (line 204):
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 204)
        idx_38652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 29), 'idx')
        # Getting the type of 'complex_matrices' (line 204)
        complex_matrices_38653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'complex_matrices')
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___38654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), complex_matrices_38653, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_38655 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), getitem___38654, idx_38652)
        
        # Assigning a type to the variable 'a' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'a', subscript_call_result_38655)
        
        # Assigning a Subscript to a Tuple (line 205):
        
        # Assigning a Subscript to a Name (line 205):
        
        # Obtaining the type of the subscript
        int_38656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 205)
        idx_38657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 49), 'idx')
        # Getting the type of 'complex_solutions' (line 205)
        complex_solutions_38658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 31), 'complex_solutions')
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___38659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 31), complex_solutions_38658, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
        subscript_call_result_38660 = invoke(stypy.reporting.localization.Localization(__file__, 205, 31), getitem___38659, idx_38657)
        
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___38661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), subscript_call_result_38660, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
        subscript_call_result_38662 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), getitem___38661, int_38656)
        
        # Assigning a type to the variable 'tuple_var_assignment_38012' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_var_assignment_38012', subscript_call_result_38662)
        
        # Assigning a Subscript to a Name (line 205):
        
        # Obtaining the type of the subscript
        int_38663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 205)
        idx_38664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 49), 'idx')
        # Getting the type of 'complex_solutions' (line 205)
        complex_solutions_38665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 31), 'complex_solutions')
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___38666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 31), complex_solutions_38665, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
        subscript_call_result_38667 = invoke(stypy.reporting.localization.Localization(__file__, 205, 31), getitem___38666, idx_38664)
        
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___38668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), subscript_call_result_38667, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
        subscript_call_result_38669 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), getitem___38668, int_38663)
        
        # Assigning a type to the variable 'tuple_var_assignment_38013' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_var_assignment_38013', subscript_call_result_38669)
        
        # Assigning a Subscript to a Name (line 205):
        
        # Obtaining the type of the subscript
        int_38670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 205)
        idx_38671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 49), 'idx')
        # Getting the type of 'complex_solutions' (line 205)
        complex_solutions_38672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 31), 'complex_solutions')
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___38673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 31), complex_solutions_38672, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
        subscript_call_result_38674 = invoke(stypy.reporting.localization.Localization(__file__, 205, 31), getitem___38673, idx_38671)
        
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___38675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), subscript_call_result_38674, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
        subscript_call_result_38676 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), getitem___38675, int_38670)
        
        # Assigning a type to the variable 'tuple_var_assignment_38014' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_var_assignment_38014', subscript_call_result_38676)
        
        # Assigning a Name to a Name (line 205):
        # Getting the type of 'tuple_var_assignment_38012' (line 205)
        tuple_var_assignment_38012_38677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_var_assignment_38012')
        # Assigning a type to the variable 'y0' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'y0', tuple_var_assignment_38012_38677)
        
        # Assigning a Name to a Name (line 205):
        # Getting the type of 'tuple_var_assignment_38013' (line 205)
        tuple_var_assignment_38013_38678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_var_assignment_38013')
        # Assigning a type to the variable 't_exact' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 't_exact', tuple_var_assignment_38013_38678)
        
        # Assigning a Name to a Name (line 205):
        # Getting the type of 'tuple_var_assignment_38014' (line 205)
        tuple_var_assignment_38014_38679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_var_assignment_38014')
        # Assigning a type to the variable 'y_exact' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 21), 'y_exact', tuple_var_assignment_38014_38679)
        
        # Assigning a Call to a Tuple (line 206):
        
        # Assigning a Subscript to a Name (line 206):
        
        # Obtaining the type of the subscript
        int_38680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 8), 'int')
        
        # Call to _solve_linear_sys(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'a' (line 206)
        a_38682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 33), 'a', False)
        # Getting the type of 'y0' (line 206)
        y0_38683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 36), 'y0', False)
        # Processing the call keyword arguments (line 206)
        
        # Obtaining the type of the subscript
        int_38684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 46), 'int')
        # Getting the type of 't_exact' (line 207)
        t_exact_38685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 38), 't_exact', False)
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___38686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 38), t_exact_38685, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_38687 = invoke(stypy.reporting.localization.Localization(__file__, 207, 38), getitem___38686, int_38684)
        
        keyword_38688 = subscript_call_result_38687
        
        # Obtaining the type of the subscript
        int_38689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 44), 'int')
        # Getting the type of 't_exact' (line 208)
        t_exact_38690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 36), 't_exact', False)
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___38691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 36), t_exact_38690, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_38692 = invoke(stypy.reporting.localization.Localization(__file__, 208, 36), getitem___38691, int_38689)
        
        
        # Obtaining the type of the subscript
        int_38693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 57), 'int')
        # Getting the type of 't_exact' (line 208)
        t_exact_38694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 49), 't_exact', False)
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___38695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 49), t_exact_38694, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_38696 = invoke(stypy.reporting.localization.Localization(__file__, 208, 49), getitem___38695, int_38693)
        
        # Applying the binary operator '-' (line 208)
        result_sub_38697 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 36), '-', subscript_call_result_38692, subscript_call_result_38696)
        
        keyword_38698 = result_sub_38697
        # Getting the type of 'solver' (line 209)
        solver_38699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 40), 'solver', False)
        keyword_38700 = solver_38699
        # Getting the type of 'meth' (line 210)
        meth_38701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 40), 'meth', False)
        keyword_38702 = meth_38701
        # Getting the type of 'use_jac' (line 211)
        use_jac_38703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 41), 'use_jac', False)
        keyword_38704 = use_jac_38703
        # Getting the type of 'with_jac' (line 212)
        with_jac_38705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 47), 'with_jac', False)
        keyword_38706 = with_jac_38705
        # Getting the type of 'banded' (line 213)
        banded_38707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 40), 'banded', False)
        keyword_38708 = banded_38707
        kwargs_38709 = {'solver': keyword_38700, 'banded': keyword_38708, 'tend': keyword_38688, 'with_jacobian': keyword_38706, 'use_jac': keyword_38704, 'dt': keyword_38698, 'method': keyword_38702}
        # Getting the type of '_solve_linear_sys' (line 206)
        _solve_linear_sys_38681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), '_solve_linear_sys', False)
        # Calling _solve_linear_sys(args, kwargs) (line 206)
        _solve_linear_sys_call_result_38710 = invoke(stypy.reporting.localization.Localization(__file__, 206, 15), _solve_linear_sys_38681, *[a_38682, y0_38683], **kwargs_38709)
        
        # Obtaining the member '__getitem__' of a type (line 206)
        getitem___38711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), _solve_linear_sys_call_result_38710, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
        subscript_call_result_38712 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), getitem___38711, int_38680)
        
        # Assigning a type to the variable 'tuple_var_assignment_38015' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_38015', subscript_call_result_38712)
        
        # Assigning a Subscript to a Name (line 206):
        
        # Obtaining the type of the subscript
        int_38713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 8), 'int')
        
        # Call to _solve_linear_sys(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'a' (line 206)
        a_38715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 33), 'a', False)
        # Getting the type of 'y0' (line 206)
        y0_38716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 36), 'y0', False)
        # Processing the call keyword arguments (line 206)
        
        # Obtaining the type of the subscript
        int_38717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 46), 'int')
        # Getting the type of 't_exact' (line 207)
        t_exact_38718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 38), 't_exact', False)
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___38719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 38), t_exact_38718, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_38720 = invoke(stypy.reporting.localization.Localization(__file__, 207, 38), getitem___38719, int_38717)
        
        keyword_38721 = subscript_call_result_38720
        
        # Obtaining the type of the subscript
        int_38722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 44), 'int')
        # Getting the type of 't_exact' (line 208)
        t_exact_38723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 36), 't_exact', False)
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___38724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 36), t_exact_38723, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_38725 = invoke(stypy.reporting.localization.Localization(__file__, 208, 36), getitem___38724, int_38722)
        
        
        # Obtaining the type of the subscript
        int_38726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 57), 'int')
        # Getting the type of 't_exact' (line 208)
        t_exact_38727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 49), 't_exact', False)
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___38728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 49), t_exact_38727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_38729 = invoke(stypy.reporting.localization.Localization(__file__, 208, 49), getitem___38728, int_38726)
        
        # Applying the binary operator '-' (line 208)
        result_sub_38730 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 36), '-', subscript_call_result_38725, subscript_call_result_38729)
        
        keyword_38731 = result_sub_38730
        # Getting the type of 'solver' (line 209)
        solver_38732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 40), 'solver', False)
        keyword_38733 = solver_38732
        # Getting the type of 'meth' (line 210)
        meth_38734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 40), 'meth', False)
        keyword_38735 = meth_38734
        # Getting the type of 'use_jac' (line 211)
        use_jac_38736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 41), 'use_jac', False)
        keyword_38737 = use_jac_38736
        # Getting the type of 'with_jac' (line 212)
        with_jac_38738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 47), 'with_jac', False)
        keyword_38739 = with_jac_38738
        # Getting the type of 'banded' (line 213)
        banded_38740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 40), 'banded', False)
        keyword_38741 = banded_38740
        kwargs_38742 = {'solver': keyword_38733, 'banded': keyword_38741, 'tend': keyword_38721, 'with_jacobian': keyword_38739, 'use_jac': keyword_38737, 'dt': keyword_38731, 'method': keyword_38735}
        # Getting the type of '_solve_linear_sys' (line 206)
        _solve_linear_sys_38714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), '_solve_linear_sys', False)
        # Calling _solve_linear_sys(args, kwargs) (line 206)
        _solve_linear_sys_call_result_38743 = invoke(stypy.reporting.localization.Localization(__file__, 206, 15), _solve_linear_sys_38714, *[a_38715, y0_38716], **kwargs_38742)
        
        # Obtaining the member '__getitem__' of a type (line 206)
        getitem___38744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), _solve_linear_sys_call_result_38743, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
        subscript_call_result_38745 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), getitem___38744, int_38713)
        
        # Assigning a type to the variable 'tuple_var_assignment_38016' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_38016', subscript_call_result_38745)
        
        # Assigning a Name to a Name (line 206):
        # Getting the type of 'tuple_var_assignment_38015' (line 206)
        tuple_var_assignment_38015_38746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_38015')
        # Assigning a type to the variable 't' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 't', tuple_var_assignment_38015_38746)
        
        # Assigning a Name to a Name (line 206):
        # Getting the type of 'tuple_var_assignment_38016' (line 206)
        tuple_var_assignment_38016_38747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_38016')
        # Assigning a type to the variable 'y' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'y', tuple_var_assignment_38016_38747)
        
        # Call to assert_allclose(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 't' (line 214)
        t_38749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 't', False)
        # Getting the type of 't_exact' (line 214)
        t_exact_38750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 27), 't_exact', False)
        # Processing the call keyword arguments (line 214)
        kwargs_38751 = {}
        # Getting the type of 'assert_allclose' (line 214)
        assert_allclose_38748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 214)
        assert_allclose_call_result_38752 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), assert_allclose_38748, *[t_38749, t_exact_38750], **kwargs_38751)
        
        
        # Call to assert_allclose(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'y' (line 215)
        y_38754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 24), 'y', False)
        # Getting the type of 'y_exact' (line 215)
        y_exact_38755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 27), 'y_exact', False)
        # Processing the call keyword arguments (line 215)
        kwargs_38756 = {}
        # Getting the type of 'assert_allclose' (line 215)
        assert_allclose_38753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 215)
        assert_allclose_call_result_38757 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), assert_allclose_38753, *[y_38754, y_exact_38755], **kwargs_38756)
        
        
        # ################# End of 'check_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 203)
        stypy_return_type_38758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38758)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_complex'
        return stypy_return_type_38758

    # Assigning a type to the variable 'check_complex' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'check_complex', check_complex)
    
    
    # Call to range(...): (line 217)
    # Processing the call arguments (line 217)
    
    # Call to len(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'complex_matrices' (line 217)
    complex_matrices_38761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 25), 'complex_matrices', False)
    # Processing the call keyword arguments (line 217)
    kwargs_38762 = {}
    # Getting the type of 'len' (line 217)
    len_38760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'len', False)
    # Calling len(args, kwargs) (line 217)
    len_call_result_38763 = invoke(stypy.reporting.localization.Localization(__file__, 217, 21), len_38760, *[complex_matrices_38761], **kwargs_38762)
    
    # Processing the call keyword arguments (line 217)
    kwargs_38764 = {}
    # Getting the type of 'range' (line 217)
    range_38759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'range', False)
    # Calling range(args, kwargs) (line 217)
    range_call_result_38765 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), range_38759, *[len_call_result_38763], **kwargs_38764)
    
    # Testing the type of a for loop iterable (line 217)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 217, 4), range_call_result_38765)
    # Getting the type of the for loop variable (line 217)
    for_loop_var_38766 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 217, 4), range_call_result_38765)
    # Assigning a type to the variable 'idx' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'idx', for_loop_var_38766)
    # SSA begins for a for statement (line 217)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a List to a Name (line 218):
    
    # Assigning a List to a Name (line 218):
    
    # Obtaining an instance of the builtin type 'list' (line 218)
    list_38767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 218)
    # Adding element type (line 218)
    
    # Obtaining an instance of the builtin type 'list' (line 218)
    list_38768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 218)
    # Adding element type (line 218)
    str_38769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 14), 'str', 'bdf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 13), list_38768, str_38769)
    # Adding element type (line 218)
    str_38770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 21), 'str', 'adams')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 13), list_38768, str_38770)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 12), list_38767, list_38768)
    # Adding element type (line 218)
    
    # Obtaining an instance of the builtin type 'list' (line 219)
    list_38771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 219)
    # Adding element type (line 219)
    # Getting the type of 'False' (line 219)
    False_38772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 14), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 13), list_38771, False_38772)
    # Adding element type (line 219)
    # Getting the type of 'True' (line 219)
    True_38773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 13), list_38771, True_38773)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 12), list_38767, list_38771)
    # Adding element type (line 218)
    
    # Obtaining an instance of the builtin type 'list' (line 220)
    list_38774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 220)
    # Adding element type (line 220)
    # Getting the type of 'False' (line 220)
    False_38775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 14), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 13), list_38774, False_38775)
    # Adding element type (line 220)
    # Getting the type of 'True' (line 220)
    True_38776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 13), list_38774, True_38776)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 12), list_38767, list_38774)
    # Adding element type (line 218)
    
    # Obtaining an instance of the builtin type 'list' (line 221)
    list_38777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 221)
    # Adding element type (line 221)
    # Getting the type of 'False' (line 221)
    False_38778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 14), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 13), list_38777, False_38778)
    # Adding element type (line 221)
    # Getting the type of 'True' (line 221)
    True_38779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 21), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 13), list_38777, True_38779)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 12), list_38767, list_38777)
    
    # Assigning a type to the variable 'p' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'p', list_38767)
    
    
    # Call to product(...): (line 222)
    # Getting the type of 'p' (line 222)
    p_38782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 66), 'p', False)
    # Processing the call keyword arguments (line 222)
    kwargs_38783 = {}
    # Getting the type of 'itertools' (line 222)
    itertools_38780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 47), 'itertools', False)
    # Obtaining the member 'product' of a type (line 222)
    product_38781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 47), itertools_38780, 'product')
    # Calling product(args, kwargs) (line 222)
    product_call_result_38784 = invoke(stypy.reporting.localization.Localization(__file__, 222, 47), product_38781, *[p_38782], **kwargs_38783)
    
    # Testing the type of a for loop iterable (line 222)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 222, 8), product_call_result_38784)
    # Getting the type of the for loop variable (line 222)
    for_loop_var_38785 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 222, 8), product_call_result_38784)
    # Assigning a type to the variable 'meth' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'meth', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 8), for_loop_var_38785))
    # Assigning a type to the variable 'use_jac' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'use_jac', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 8), for_loop_var_38785))
    # Assigning a type to the variable 'with_jac' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'with_jac', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 8), for_loop_var_38785))
    # Assigning a type to the variable 'banded' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'banded', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 8), for_loop_var_38785))
    # SSA begins for a for statement (line 222)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check_complex(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'idx' (line 223)
    idx_38787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 26), 'idx', False)
    str_38788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 31), 'str', 'zvode')
    # Getting the type of 'meth' (line 223)
    meth_38789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 40), 'meth', False)
    # Getting the type of 'use_jac' (line 223)
    use_jac_38790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 46), 'use_jac', False)
    # Getting the type of 'with_jac' (line 223)
    with_jac_38791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 55), 'with_jac', False)
    # Getting the type of 'banded' (line 223)
    banded_38792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 65), 'banded', False)
    # Processing the call keyword arguments (line 223)
    kwargs_38793 = {}
    # Getting the type of 'check_complex' (line 223)
    check_complex_38786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'check_complex', False)
    # Calling check_complex(args, kwargs) (line 223)
    check_complex_call_result_38794 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), check_complex_38786, *[idx_38787, str_38788, meth_38789, use_jac_38790, with_jac_38791, banded_38792], **kwargs_38793)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_banded_ode_solvers(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_banded_ode_solvers' in the type store
    # Getting the type of 'stypy_return_type' (line 132)
    stypy_return_type_38795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38795)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_banded_ode_solvers'
    return stypy_return_type_38795

# Assigning a type to the variable 'test_banded_ode_solvers' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'test_banded_ode_solvers', test_banded_ode_solvers)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
