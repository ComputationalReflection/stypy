
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Unit test for Linear Programming via Simplex Algorithm.
3: '''
4: 
5: # TODO: add tests for:
6: # https://github.com/scipy/scipy/issues/5400
7: # https://github.com/scipy/scipy/issues/6690
8: 
9: from __future__ import division, print_function, absolute_import
10: 
11: import numpy as np
12: from numpy.testing import (
13:     assert_,
14:     assert_allclose,
15:     assert_equal)
16: 
17: from .test_linprog import magic_square
18: from scipy.optimize._remove_redundancy import _remove_redundancy
19: 
20: 
21: def setup_module():
22:     np.random.seed(2017)
23: 
24: 
25: def _assert_success(
26:         res,
27:         desired_fun=None,
28:         desired_x=None,
29:         rtol=1e-7,
30:         atol=1e-7):
31:     # res: linprog result object
32:     # desired_fun: desired objective function value or None
33:     # desired_x: desired solution or None
34:     assert_(res.success)
35:     assert_equal(res.status, 0)
36:     if desired_fun is not None:
37:         assert_allclose(
38:             res.fun,
39:             desired_fun,
40:             err_msg="converged to an unexpected objective value",
41:             rtol=rtol,
42:             atol=atol)
43:     if desired_x is not None:
44:         assert_allclose(
45:             res.x,
46:             desired_x,
47:             err_msg="converged to an unexpected solution",
48:             rtol=rtol,
49:             atol=atol)
50: 
51: 
52: def test_no_redundancy():
53:     m, n = 10, 10
54:     A0 = np.random.rand(m, n)
55:     b0 = np.random.rand(m)
56:     A1, b1, status, message = _remove_redundancy(A0, b0)
57:     assert_allclose(A0, A1)
58:     assert_allclose(b0, b1)
59:     assert_equal(status, 0)
60: 
61: 
62: def test_infeasible_zero_row():
63:     A = np.eye(3)
64:     A[1, :] = 0
65:     b = np.random.rand(3)
66:     A1, b1, status, message = _remove_redundancy(A, b)
67:     assert_equal(status, 2)
68: 
69: 
70: def test_remove_zero_row():
71:     A = np.eye(3)
72:     A[1, :] = 0
73:     b = np.random.rand(3)
74:     b[1] = 0
75:     A1, b1, status, message = _remove_redundancy(A, b)
76:     assert_equal(status, 0)
77:     assert_allclose(A1, A[[0, 2], :])
78:     assert_allclose(b1, b[[0, 2]])
79: 
80: 
81: def test_infeasible_m_gt_n():
82:     m, n = 20, 10
83:     A0 = np.random.rand(m, n)
84:     b0 = np.random.rand(m)
85:     A1, b1, status, message = _remove_redundancy(A0, b0)
86:     assert_equal(status, 2)
87: 
88: 
89: def test_infeasible_m_eq_n():
90:     m, n = 10, 10
91:     A0 = np.random.rand(m, n)
92:     b0 = np.random.rand(m)
93:     A0[-1, :] = 2 * A0[-2, :]
94:     A1, b1, status, message = _remove_redundancy(A0, b0)
95:     assert_equal(status, 2)
96: 
97: 
98: def test_infeasible_m_lt_n():
99:     m, n = 9, 10
100:     A0 = np.random.rand(m, n)
101:     b0 = np.random.rand(m)
102:     A0[-1, :] = np.arange(m - 1).dot(A0[:-1])
103:     A1, b1, status, message = _remove_redundancy(A0, b0)
104:     assert_equal(status, 2)
105: 
106: 
107: def test_m_gt_n():
108:     m, n = 20, 10
109:     A0 = np.random.rand(m, n)
110:     b0 = np.random.rand(m)
111:     x = np.linalg.solve(A0[:n, :], b0[:n])
112:     b0[n:] = A0[n:, :].dot(x)
113:     A1, b1, status, message = _remove_redundancy(A0, b0)
114:     assert_equal(status, 0)
115:     assert_equal(A1.shape[0], n)
116:     assert_equal(np.linalg.matrix_rank(A1), n)
117: 
118: 
119: def test_m_gt_n_rank_deficient():
120:     m, n = 20, 10
121:     A0 = np.zeros((m, n))
122:     A0[:, 0] = 1
123:     b0 = np.ones(m)
124:     A1, b1, status, message = _remove_redundancy(A0, b0)
125:     assert_equal(status, 0)
126:     assert_allclose(A1, A0[0:1, :])
127:     assert_allclose(b1, b0[0])
128: 
129: 
130: def test_m_lt_n_rank_deficient():
131:     m, n = 9, 10
132:     A0 = np.random.rand(m, n)
133:     b0 = np.random.rand(m)
134:     A0[-1, :] = np.arange(m - 1).dot(A0[:-1])
135:     b0[-1] = np.arange(m - 1).dot(b0[:-1])
136:     A1, b1, status, message = _remove_redundancy(A0, b0)
137:     assert_equal(status, 0)
138:     assert_equal(A1.shape[0], 8)
139:     assert_equal(np.linalg.matrix_rank(A1), 8)
140: 
141: 
142: def test_dense1():
143:     A = np.ones((6, 6))
144:     A[0, :3] = 0
145:     A[1, 3:] = 0
146:     A[3:, ::2] = -1
147:     A[3, :2] = 0
148:     A[4, 2:] = 0
149:     b = np.zeros(A.shape[0])
150: 
151:     A2 = A[[0, 1, 3, 4], :]
152:     b2 = np.zeros(4)
153: 
154:     A1, b1, status, message = _remove_redundancy(A, b)
155:     assert_allclose(A1, A2)
156:     assert_allclose(b1, b2)
157:     assert_equal(status, 0)
158: 
159: 
160: def test_dense2():
161:     A = np.eye(6)
162:     A[-2, -1] = 1
163:     A[-1, :] = 1
164:     b = np.zeros(A.shape[0])
165:     A1, b1, status, message = _remove_redundancy(A, b)
166:     assert_allclose(A1, A[:-1, :])
167:     assert_allclose(b1, b[:-1])
168:     assert_equal(status, 0)
169: 
170: 
171: def test_dense3():
172:     A = np.eye(6)
173:     A[-2, -1] = 1
174:     A[-1, :] = 1
175:     b = np.random.rand(A.shape[0])
176:     b[-1] = np.sum(b[:-1])
177:     A1, b1, status, message = _remove_redundancy(A, b)
178:     assert_allclose(A1, A[:-1, :])
179:     assert_allclose(b1, b[:-1])
180:     assert_equal(status, 0)
181: 
182: 
183: def test_m_gt_n_sparse():
184:     np.random.seed(2013)
185:     m, n = 20, 5
186:     p = 0.1
187:     A = np.random.rand(m, n)
188:     A[np.random.rand(m, n) > p] = 0
189:     rank = np.linalg.matrix_rank(A)
190:     b = np.zeros(A.shape[0])
191:     A1, b1, status, message = _remove_redundancy(A, b)
192:     assert_equal(status, 0)
193:     assert_equal(A1.shape[0], rank)
194:     assert_equal(np.linalg.matrix_rank(A1), rank)
195: 
196: 
197: def test_m_lt_n_sparse():
198:     np.random.seed(2017)
199:     m, n = 20, 50
200:     p = 0.05
201:     A = np.random.rand(m, n)
202:     A[np.random.rand(m, n) > p] = 0
203:     rank = np.linalg.matrix_rank(A)
204:     b = np.zeros(A.shape[0])
205:     A1, b1, status, message = _remove_redundancy(A, b)
206:     assert_equal(status, 0)
207:     assert_equal(A1.shape[0], rank)
208:     assert_equal(np.linalg.matrix_rank(A1), rank)
209: 
210: 
211: def test_m_eq_n_sparse():
212:     np.random.seed(2017)
213:     m, n = 100, 100
214:     p = 0.01
215:     A = np.random.rand(m, n)
216:     A[np.random.rand(m, n) > p] = 0
217:     rank = np.linalg.matrix_rank(A)
218:     b = np.zeros(A.shape[0])
219:     A1, b1, status, message = _remove_redundancy(A, b)
220:     assert_equal(status, 0)
221:     assert_equal(A1.shape[0], rank)
222:     assert_equal(np.linalg.matrix_rank(A1), rank)
223: 
224: 
225: def test_magic_square():
226:     A, b, c, numbers = magic_square(3)
227:     A1, b1, status, message = _remove_redundancy(A, b)
228:     assert_equal(status, 0)
229:     assert_equal(A1.shape[0], 23)
230:     assert_equal(np.linalg.matrix_rank(A1), 23)
231: 
232: 
233: def test_magic_square2():
234:     A, b, c, numbers = magic_square(4)
235:     A1, b1, status, message = _remove_redundancy(A, b)
236:     assert_equal(status, 0)
237:     assert_equal(A1.shape[0], 39)
238:     assert_equal(np.linalg.matrix_rank(A1), 39)
239: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_244107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nUnit test for Linear Programming via Simplex Algorithm.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_244108 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_244108) is not StypyTypeError):

    if (import_244108 != 'pyd_module'):
        __import__(import_244108)
        sys_modules_244109 = sys.modules[import_244108]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_244109.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_244108)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.testing import assert_, assert_allclose, assert_equal' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_244110 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing')

if (type(import_244110) is not StypyTypeError):

    if (import_244110 != 'pyd_module'):
        __import__(import_244110)
        sys_modules_244111 = sys.modules[import_244110]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing', sys_modules_244111.module_type_store, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_244111, sys_modules_244111.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'], [assert_, assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing', import_244110)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.optimize.tests.test_linprog import magic_square' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_244112 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize.tests.test_linprog')

if (type(import_244112) is not StypyTypeError):

    if (import_244112 != 'pyd_module'):
        __import__(import_244112)
        sys_modules_244113 = sys.modules[import_244112]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize.tests.test_linprog', sys_modules_244113.module_type_store, module_type_store, ['magic_square'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_244113, sys_modules_244113.module_type_store, module_type_store)
    else:
        from scipy.optimize.tests.test_linprog import magic_square

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize.tests.test_linprog', None, module_type_store, ['magic_square'], [magic_square])

else:
    # Assigning a type to the variable 'scipy.optimize.tests.test_linprog' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.optimize.tests.test_linprog', import_244112)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.optimize._remove_redundancy import _remove_redundancy' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_244114 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize._remove_redundancy')

if (type(import_244114) is not StypyTypeError):

    if (import_244114 != 'pyd_module'):
        __import__(import_244114)
        sys_modules_244115 = sys.modules[import_244114]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize._remove_redundancy', sys_modules_244115.module_type_store, module_type_store, ['_remove_redundancy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_244115, sys_modules_244115.module_type_store, module_type_store)
    else:
        from scipy.optimize._remove_redundancy import _remove_redundancy

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize._remove_redundancy', None, module_type_store, ['_remove_redundancy'], [_remove_redundancy])

else:
    # Assigning a type to the variable 'scipy.optimize._remove_redundancy' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize._remove_redundancy', import_244114)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')


@norecursion
def setup_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'setup_module'
    module_type_store = module_type_store.open_function_context('setup_module', 21, 0, False)
    
    # Passed parameters checking function
    setup_module.stypy_localization = localization
    setup_module.stypy_type_of_self = None
    setup_module.stypy_type_store = module_type_store
    setup_module.stypy_function_name = 'setup_module'
    setup_module.stypy_param_names_list = []
    setup_module.stypy_varargs_param_name = None
    setup_module.stypy_kwargs_param_name = None
    setup_module.stypy_call_defaults = defaults
    setup_module.stypy_call_varargs = varargs
    setup_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setup_module', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setup_module', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setup_module(...)' code ##################

    
    # Call to seed(...): (line 22)
    # Processing the call arguments (line 22)
    int_244119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'int')
    # Processing the call keyword arguments (line 22)
    kwargs_244120 = {}
    # Getting the type of 'np' (line 22)
    np_244116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 22)
    random_244117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 4), np_244116, 'random')
    # Obtaining the member 'seed' of a type (line 22)
    seed_244118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 4), random_244117, 'seed')
    # Calling seed(args, kwargs) (line 22)
    seed_call_result_244121 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), seed_244118, *[int_244119], **kwargs_244120)
    
    
    # ################# End of 'setup_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setup_module' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_244122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_244122)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setup_module'
    return stypy_return_type_244122

# Assigning a type to the variable 'setup_module' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'setup_module', setup_module)

@norecursion
def _assert_success(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 27)
    None_244123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'None')
    # Getting the type of 'None' (line 28)
    None_244124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'None')
    float_244125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 13), 'float')
    float_244126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 13), 'float')
    defaults = [None_244123, None_244124, float_244125, float_244126]
    # Create a new context for function '_assert_success'
    module_type_store = module_type_store.open_function_context('_assert_success', 25, 0, False)
    
    # Passed parameters checking function
    _assert_success.stypy_localization = localization
    _assert_success.stypy_type_of_self = None
    _assert_success.stypy_type_store = module_type_store
    _assert_success.stypy_function_name = '_assert_success'
    _assert_success.stypy_param_names_list = ['res', 'desired_fun', 'desired_x', 'rtol', 'atol']
    _assert_success.stypy_varargs_param_name = None
    _assert_success.stypy_kwargs_param_name = None
    _assert_success.stypy_call_defaults = defaults
    _assert_success.stypy_call_varargs = varargs
    _assert_success.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_assert_success', ['res', 'desired_fun', 'desired_x', 'rtol', 'atol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_assert_success', localization, ['res', 'desired_fun', 'desired_x', 'rtol', 'atol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_assert_success(...)' code ##################

    
    # Call to assert_(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'res' (line 34)
    res_244128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'res', False)
    # Obtaining the member 'success' of a type (line 34)
    success_244129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), res_244128, 'success')
    # Processing the call keyword arguments (line 34)
    kwargs_244130 = {}
    # Getting the type of 'assert_' (line 34)
    assert__244127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 34)
    assert__call_result_244131 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), assert__244127, *[success_244129], **kwargs_244130)
    
    
    # Call to assert_equal(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'res' (line 35)
    res_244133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'res', False)
    # Obtaining the member 'status' of a type (line 35)
    status_244134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 17), res_244133, 'status')
    int_244135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 29), 'int')
    # Processing the call keyword arguments (line 35)
    kwargs_244136 = {}
    # Getting the type of 'assert_equal' (line 35)
    assert_equal_244132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 35)
    assert_equal_call_result_244137 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), assert_equal_244132, *[status_244134, int_244135], **kwargs_244136)
    
    
    # Type idiom detected: calculating its left and rigth part (line 36)
    # Getting the type of 'desired_fun' (line 36)
    desired_fun_244138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'desired_fun')
    # Getting the type of 'None' (line 36)
    None_244139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 26), 'None')
    
    (may_be_244140, more_types_in_union_244141) = may_not_be_none(desired_fun_244138, None_244139)

    if may_be_244140:

        if more_types_in_union_244141:
            # Runtime conditional SSA (line 36)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to assert_allclose(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'res' (line 38)
        res_244143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'res', False)
        # Obtaining the member 'fun' of a type (line 38)
        fun_244144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), res_244143, 'fun')
        # Getting the type of 'desired_fun' (line 39)
        desired_fun_244145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'desired_fun', False)
        # Processing the call keyword arguments (line 37)
        str_244146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'str', 'converged to an unexpected objective value')
        keyword_244147 = str_244146
        # Getting the type of 'rtol' (line 41)
        rtol_244148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'rtol', False)
        keyword_244149 = rtol_244148
        # Getting the type of 'atol' (line 42)
        atol_244150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'atol', False)
        keyword_244151 = atol_244150
        kwargs_244152 = {'rtol': keyword_244149, 'err_msg': keyword_244147, 'atol': keyword_244151}
        # Getting the type of 'assert_allclose' (line 37)
        assert_allclose_244142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 37)
        assert_allclose_call_result_244153 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), assert_allclose_244142, *[fun_244144, desired_fun_244145], **kwargs_244152)
        

        if more_types_in_union_244141:
            # SSA join for if statement (line 36)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 43)
    # Getting the type of 'desired_x' (line 43)
    desired_x_244154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'desired_x')
    # Getting the type of 'None' (line 43)
    None_244155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'None')
    
    (may_be_244156, more_types_in_union_244157) = may_not_be_none(desired_x_244154, None_244155)

    if may_be_244156:

        if more_types_in_union_244157:
            # Runtime conditional SSA (line 43)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to assert_allclose(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'res' (line 45)
        res_244159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'res', False)
        # Obtaining the member 'x' of a type (line 45)
        x_244160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), res_244159, 'x')
        # Getting the type of 'desired_x' (line 46)
        desired_x_244161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'desired_x', False)
        # Processing the call keyword arguments (line 44)
        str_244162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 20), 'str', 'converged to an unexpected solution')
        keyword_244163 = str_244162
        # Getting the type of 'rtol' (line 48)
        rtol_244164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'rtol', False)
        keyword_244165 = rtol_244164
        # Getting the type of 'atol' (line 49)
        atol_244166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'atol', False)
        keyword_244167 = atol_244166
        kwargs_244168 = {'rtol': keyword_244165, 'err_msg': keyword_244163, 'atol': keyword_244167}
        # Getting the type of 'assert_allclose' (line 44)
        assert_allclose_244158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 44)
        assert_allclose_call_result_244169 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), assert_allclose_244158, *[x_244160, desired_x_244161], **kwargs_244168)
        

        if more_types_in_union_244157:
            # SSA join for if statement (line 43)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_assert_success(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_assert_success' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_244170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_244170)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_assert_success'
    return stypy_return_type_244170

# Assigning a type to the variable '_assert_success' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), '_assert_success', _assert_success)

@norecursion
def test_no_redundancy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_no_redundancy'
    module_type_store = module_type_store.open_function_context('test_no_redundancy', 52, 0, False)
    
    # Passed parameters checking function
    test_no_redundancy.stypy_localization = localization
    test_no_redundancy.stypy_type_of_self = None
    test_no_redundancy.stypy_type_store = module_type_store
    test_no_redundancy.stypy_function_name = 'test_no_redundancy'
    test_no_redundancy.stypy_param_names_list = []
    test_no_redundancy.stypy_varargs_param_name = None
    test_no_redundancy.stypy_kwargs_param_name = None
    test_no_redundancy.stypy_call_defaults = defaults
    test_no_redundancy.stypy_call_varargs = varargs
    test_no_redundancy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_no_redundancy', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_no_redundancy', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_no_redundancy(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 53):
    
    # Assigning a Num to a Name (line 53):
    int_244171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_244011' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'tuple_assignment_244011', int_244171)
    
    # Assigning a Num to a Name (line 53):
    int_244172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 15), 'int')
    # Assigning a type to the variable 'tuple_assignment_244012' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'tuple_assignment_244012', int_244172)
    
    # Assigning a Name to a Name (line 53):
    # Getting the type of 'tuple_assignment_244011' (line 53)
    tuple_assignment_244011_244173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'tuple_assignment_244011')
    # Assigning a type to the variable 'm' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'm', tuple_assignment_244011_244173)
    
    # Assigning a Name to a Name (line 53):
    # Getting the type of 'tuple_assignment_244012' (line 53)
    tuple_assignment_244012_244174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'tuple_assignment_244012')
    # Assigning a type to the variable 'n' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 7), 'n', tuple_assignment_244012_244174)
    
    # Assigning a Call to a Name (line 54):
    
    # Assigning a Call to a Name (line 54):
    
    # Call to rand(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'm' (line 54)
    m_244178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'm', False)
    # Getting the type of 'n' (line 54)
    n_244179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 27), 'n', False)
    # Processing the call keyword arguments (line 54)
    kwargs_244180 = {}
    # Getting the type of 'np' (line 54)
    np_244175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 54)
    random_244176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 9), np_244175, 'random')
    # Obtaining the member 'rand' of a type (line 54)
    rand_244177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 9), random_244176, 'rand')
    # Calling rand(args, kwargs) (line 54)
    rand_call_result_244181 = invoke(stypy.reporting.localization.Localization(__file__, 54, 9), rand_244177, *[m_244178, n_244179], **kwargs_244180)
    
    # Assigning a type to the variable 'A0' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'A0', rand_call_result_244181)
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to rand(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'm' (line 55)
    m_244185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'm', False)
    # Processing the call keyword arguments (line 55)
    kwargs_244186 = {}
    # Getting the type of 'np' (line 55)
    np_244182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 55)
    random_244183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 9), np_244182, 'random')
    # Obtaining the member 'rand' of a type (line 55)
    rand_244184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 9), random_244183, 'rand')
    # Calling rand(args, kwargs) (line 55)
    rand_call_result_244187 = invoke(stypy.reporting.localization.Localization(__file__, 55, 9), rand_244184, *[m_244185], **kwargs_244186)
    
    # Assigning a type to the variable 'b0' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'b0', rand_call_result_244187)
    
    # Assigning a Call to a Tuple (line 56):
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    int_244188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'A0' (line 56)
    A0_244190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 49), 'A0', False)
    # Getting the type of 'b0' (line 56)
    b0_244191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 53), 'b0', False)
    # Processing the call keyword arguments (line 56)
    kwargs_244192 = {}
    # Getting the type of '_remove_redundancy' (line 56)
    _remove_redundancy_244189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 56)
    _remove_redundancy_call_result_244193 = invoke(stypy.reporting.localization.Localization(__file__, 56, 30), _remove_redundancy_244189, *[A0_244190, b0_244191], **kwargs_244192)
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___244194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), _remove_redundancy_call_result_244193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_244195 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), getitem___244194, int_244188)
    
    # Assigning a type to the variable 'tuple_var_assignment_244013' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_244013', subscript_call_result_244195)
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    int_244196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'A0' (line 56)
    A0_244198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 49), 'A0', False)
    # Getting the type of 'b0' (line 56)
    b0_244199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 53), 'b0', False)
    # Processing the call keyword arguments (line 56)
    kwargs_244200 = {}
    # Getting the type of '_remove_redundancy' (line 56)
    _remove_redundancy_244197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 56)
    _remove_redundancy_call_result_244201 = invoke(stypy.reporting.localization.Localization(__file__, 56, 30), _remove_redundancy_244197, *[A0_244198, b0_244199], **kwargs_244200)
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___244202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), _remove_redundancy_call_result_244201, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_244203 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), getitem___244202, int_244196)
    
    # Assigning a type to the variable 'tuple_var_assignment_244014' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_244014', subscript_call_result_244203)
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    int_244204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'A0' (line 56)
    A0_244206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 49), 'A0', False)
    # Getting the type of 'b0' (line 56)
    b0_244207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 53), 'b0', False)
    # Processing the call keyword arguments (line 56)
    kwargs_244208 = {}
    # Getting the type of '_remove_redundancy' (line 56)
    _remove_redundancy_244205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 56)
    _remove_redundancy_call_result_244209 = invoke(stypy.reporting.localization.Localization(__file__, 56, 30), _remove_redundancy_244205, *[A0_244206, b0_244207], **kwargs_244208)
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___244210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), _remove_redundancy_call_result_244209, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_244211 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), getitem___244210, int_244204)
    
    # Assigning a type to the variable 'tuple_var_assignment_244015' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_244015', subscript_call_result_244211)
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    int_244212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'A0' (line 56)
    A0_244214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 49), 'A0', False)
    # Getting the type of 'b0' (line 56)
    b0_244215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 53), 'b0', False)
    # Processing the call keyword arguments (line 56)
    kwargs_244216 = {}
    # Getting the type of '_remove_redundancy' (line 56)
    _remove_redundancy_244213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 56)
    _remove_redundancy_call_result_244217 = invoke(stypy.reporting.localization.Localization(__file__, 56, 30), _remove_redundancy_244213, *[A0_244214, b0_244215], **kwargs_244216)
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___244218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), _remove_redundancy_call_result_244217, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_244219 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), getitem___244218, int_244212)
    
    # Assigning a type to the variable 'tuple_var_assignment_244016' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_244016', subscript_call_result_244219)
    
    # Assigning a Name to a Name (line 56):
    # Getting the type of 'tuple_var_assignment_244013' (line 56)
    tuple_var_assignment_244013_244220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_244013')
    # Assigning a type to the variable 'A1' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'A1', tuple_var_assignment_244013_244220)
    
    # Assigning a Name to a Name (line 56):
    # Getting the type of 'tuple_var_assignment_244014' (line 56)
    tuple_var_assignment_244014_244221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_244014')
    # Assigning a type to the variable 'b1' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'b1', tuple_var_assignment_244014_244221)
    
    # Assigning a Name to a Name (line 56):
    # Getting the type of 'tuple_var_assignment_244015' (line 56)
    tuple_var_assignment_244015_244222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_244015')
    # Assigning a type to the variable 'status' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'status', tuple_var_assignment_244015_244222)
    
    # Assigning a Name to a Name (line 56):
    # Getting the type of 'tuple_var_assignment_244016' (line 56)
    tuple_var_assignment_244016_244223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_244016')
    # Assigning a type to the variable 'message' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'message', tuple_var_assignment_244016_244223)
    
    # Call to assert_allclose(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'A0' (line 57)
    A0_244225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'A0', False)
    # Getting the type of 'A1' (line 57)
    A1_244226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'A1', False)
    # Processing the call keyword arguments (line 57)
    kwargs_244227 = {}
    # Getting the type of 'assert_allclose' (line 57)
    assert_allclose_244224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 57)
    assert_allclose_call_result_244228 = invoke(stypy.reporting.localization.Localization(__file__, 57, 4), assert_allclose_244224, *[A0_244225, A1_244226], **kwargs_244227)
    
    
    # Call to assert_allclose(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'b0' (line 58)
    b0_244230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'b0', False)
    # Getting the type of 'b1' (line 58)
    b1_244231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 24), 'b1', False)
    # Processing the call keyword arguments (line 58)
    kwargs_244232 = {}
    # Getting the type of 'assert_allclose' (line 58)
    assert_allclose_244229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 58)
    assert_allclose_call_result_244233 = invoke(stypy.reporting.localization.Localization(__file__, 58, 4), assert_allclose_244229, *[b0_244230, b1_244231], **kwargs_244232)
    
    
    # Call to assert_equal(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'status' (line 59)
    status_244235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 17), 'status', False)
    int_244236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 25), 'int')
    # Processing the call keyword arguments (line 59)
    kwargs_244237 = {}
    # Getting the type of 'assert_equal' (line 59)
    assert_equal_244234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 59)
    assert_equal_call_result_244238 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), assert_equal_244234, *[status_244235, int_244236], **kwargs_244237)
    
    
    # ################# End of 'test_no_redundancy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_no_redundancy' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_244239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_244239)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_no_redundancy'
    return stypy_return_type_244239

# Assigning a type to the variable 'test_no_redundancy' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'test_no_redundancy', test_no_redundancy)

@norecursion
def test_infeasible_zero_row(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_infeasible_zero_row'
    module_type_store = module_type_store.open_function_context('test_infeasible_zero_row', 62, 0, False)
    
    # Passed parameters checking function
    test_infeasible_zero_row.stypy_localization = localization
    test_infeasible_zero_row.stypy_type_of_self = None
    test_infeasible_zero_row.stypy_type_store = module_type_store
    test_infeasible_zero_row.stypy_function_name = 'test_infeasible_zero_row'
    test_infeasible_zero_row.stypy_param_names_list = []
    test_infeasible_zero_row.stypy_varargs_param_name = None
    test_infeasible_zero_row.stypy_kwargs_param_name = None
    test_infeasible_zero_row.stypy_call_defaults = defaults
    test_infeasible_zero_row.stypy_call_varargs = varargs
    test_infeasible_zero_row.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_infeasible_zero_row', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_infeasible_zero_row', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_infeasible_zero_row(...)' code ##################

    
    # Assigning a Call to a Name (line 63):
    
    # Assigning a Call to a Name (line 63):
    
    # Call to eye(...): (line 63)
    # Processing the call arguments (line 63)
    int_244242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 15), 'int')
    # Processing the call keyword arguments (line 63)
    kwargs_244243 = {}
    # Getting the type of 'np' (line 63)
    np_244240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'np', False)
    # Obtaining the member 'eye' of a type (line 63)
    eye_244241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), np_244240, 'eye')
    # Calling eye(args, kwargs) (line 63)
    eye_call_result_244244 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), eye_244241, *[int_244242], **kwargs_244243)
    
    # Assigning a type to the variable 'A' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'A', eye_call_result_244244)
    
    # Assigning a Num to a Subscript (line 64):
    
    # Assigning a Num to a Subscript (line 64):
    int_244245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 14), 'int')
    # Getting the type of 'A' (line 64)
    A_244246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'A')
    int_244247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 6), 'int')
    slice_244248 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 64, 4), None, None, None)
    # Storing an element on a container (line 64)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 4), A_244246, ((int_244247, slice_244248), int_244245))
    
    # Assigning a Call to a Name (line 65):
    
    # Assigning a Call to a Name (line 65):
    
    # Call to rand(...): (line 65)
    # Processing the call arguments (line 65)
    int_244252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 23), 'int')
    # Processing the call keyword arguments (line 65)
    kwargs_244253 = {}
    # Getting the type of 'np' (line 65)
    np_244249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 65)
    random_244250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), np_244249, 'random')
    # Obtaining the member 'rand' of a type (line 65)
    rand_244251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), random_244250, 'rand')
    # Calling rand(args, kwargs) (line 65)
    rand_call_result_244254 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), rand_244251, *[int_244252], **kwargs_244253)
    
    # Assigning a type to the variable 'b' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'b', rand_call_result_244254)
    
    # Assigning a Call to a Tuple (line 66):
    
    # Assigning a Subscript to a Name (line 66):
    
    # Obtaining the type of the subscript
    int_244255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'A' (line 66)
    A_244257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 49), 'A', False)
    # Getting the type of 'b' (line 66)
    b_244258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 52), 'b', False)
    # Processing the call keyword arguments (line 66)
    kwargs_244259 = {}
    # Getting the type of '_remove_redundancy' (line 66)
    _remove_redundancy_244256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 66)
    _remove_redundancy_call_result_244260 = invoke(stypy.reporting.localization.Localization(__file__, 66, 30), _remove_redundancy_244256, *[A_244257, b_244258], **kwargs_244259)
    
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___244261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 4), _remove_redundancy_call_result_244260, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_244262 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), getitem___244261, int_244255)
    
    # Assigning a type to the variable 'tuple_var_assignment_244017' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_var_assignment_244017', subscript_call_result_244262)
    
    # Assigning a Subscript to a Name (line 66):
    
    # Obtaining the type of the subscript
    int_244263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'A' (line 66)
    A_244265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 49), 'A', False)
    # Getting the type of 'b' (line 66)
    b_244266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 52), 'b', False)
    # Processing the call keyword arguments (line 66)
    kwargs_244267 = {}
    # Getting the type of '_remove_redundancy' (line 66)
    _remove_redundancy_244264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 66)
    _remove_redundancy_call_result_244268 = invoke(stypy.reporting.localization.Localization(__file__, 66, 30), _remove_redundancy_244264, *[A_244265, b_244266], **kwargs_244267)
    
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___244269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 4), _remove_redundancy_call_result_244268, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_244270 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), getitem___244269, int_244263)
    
    # Assigning a type to the variable 'tuple_var_assignment_244018' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_var_assignment_244018', subscript_call_result_244270)
    
    # Assigning a Subscript to a Name (line 66):
    
    # Obtaining the type of the subscript
    int_244271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'A' (line 66)
    A_244273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 49), 'A', False)
    # Getting the type of 'b' (line 66)
    b_244274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 52), 'b', False)
    # Processing the call keyword arguments (line 66)
    kwargs_244275 = {}
    # Getting the type of '_remove_redundancy' (line 66)
    _remove_redundancy_244272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 66)
    _remove_redundancy_call_result_244276 = invoke(stypy.reporting.localization.Localization(__file__, 66, 30), _remove_redundancy_244272, *[A_244273, b_244274], **kwargs_244275)
    
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___244277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 4), _remove_redundancy_call_result_244276, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_244278 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), getitem___244277, int_244271)
    
    # Assigning a type to the variable 'tuple_var_assignment_244019' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_var_assignment_244019', subscript_call_result_244278)
    
    # Assigning a Subscript to a Name (line 66):
    
    # Obtaining the type of the subscript
    int_244279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'A' (line 66)
    A_244281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 49), 'A', False)
    # Getting the type of 'b' (line 66)
    b_244282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 52), 'b', False)
    # Processing the call keyword arguments (line 66)
    kwargs_244283 = {}
    # Getting the type of '_remove_redundancy' (line 66)
    _remove_redundancy_244280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 66)
    _remove_redundancy_call_result_244284 = invoke(stypy.reporting.localization.Localization(__file__, 66, 30), _remove_redundancy_244280, *[A_244281, b_244282], **kwargs_244283)
    
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___244285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 4), _remove_redundancy_call_result_244284, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_244286 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), getitem___244285, int_244279)
    
    # Assigning a type to the variable 'tuple_var_assignment_244020' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_var_assignment_244020', subscript_call_result_244286)
    
    # Assigning a Name to a Name (line 66):
    # Getting the type of 'tuple_var_assignment_244017' (line 66)
    tuple_var_assignment_244017_244287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_var_assignment_244017')
    # Assigning a type to the variable 'A1' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'A1', tuple_var_assignment_244017_244287)
    
    # Assigning a Name to a Name (line 66):
    # Getting the type of 'tuple_var_assignment_244018' (line 66)
    tuple_var_assignment_244018_244288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_var_assignment_244018')
    # Assigning a type to the variable 'b1' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'b1', tuple_var_assignment_244018_244288)
    
    # Assigning a Name to a Name (line 66):
    # Getting the type of 'tuple_var_assignment_244019' (line 66)
    tuple_var_assignment_244019_244289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_var_assignment_244019')
    # Assigning a type to the variable 'status' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'status', tuple_var_assignment_244019_244289)
    
    # Assigning a Name to a Name (line 66):
    # Getting the type of 'tuple_var_assignment_244020' (line 66)
    tuple_var_assignment_244020_244290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_var_assignment_244020')
    # Assigning a type to the variable 'message' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'message', tuple_var_assignment_244020_244290)
    
    # Call to assert_equal(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'status' (line 67)
    status_244292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'status', False)
    int_244293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'int')
    # Processing the call keyword arguments (line 67)
    kwargs_244294 = {}
    # Getting the type of 'assert_equal' (line 67)
    assert_equal_244291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 67)
    assert_equal_call_result_244295 = invoke(stypy.reporting.localization.Localization(__file__, 67, 4), assert_equal_244291, *[status_244292, int_244293], **kwargs_244294)
    
    
    # ################# End of 'test_infeasible_zero_row(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_infeasible_zero_row' in the type store
    # Getting the type of 'stypy_return_type' (line 62)
    stypy_return_type_244296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_244296)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_infeasible_zero_row'
    return stypy_return_type_244296

# Assigning a type to the variable 'test_infeasible_zero_row' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'test_infeasible_zero_row', test_infeasible_zero_row)

@norecursion
def test_remove_zero_row(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_remove_zero_row'
    module_type_store = module_type_store.open_function_context('test_remove_zero_row', 70, 0, False)
    
    # Passed parameters checking function
    test_remove_zero_row.stypy_localization = localization
    test_remove_zero_row.stypy_type_of_self = None
    test_remove_zero_row.stypy_type_store = module_type_store
    test_remove_zero_row.stypy_function_name = 'test_remove_zero_row'
    test_remove_zero_row.stypy_param_names_list = []
    test_remove_zero_row.stypy_varargs_param_name = None
    test_remove_zero_row.stypy_kwargs_param_name = None
    test_remove_zero_row.stypy_call_defaults = defaults
    test_remove_zero_row.stypy_call_varargs = varargs
    test_remove_zero_row.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_remove_zero_row', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_remove_zero_row', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_remove_zero_row(...)' code ##################

    
    # Assigning a Call to a Name (line 71):
    
    # Assigning a Call to a Name (line 71):
    
    # Call to eye(...): (line 71)
    # Processing the call arguments (line 71)
    int_244299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 15), 'int')
    # Processing the call keyword arguments (line 71)
    kwargs_244300 = {}
    # Getting the type of 'np' (line 71)
    np_244297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'np', False)
    # Obtaining the member 'eye' of a type (line 71)
    eye_244298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), np_244297, 'eye')
    # Calling eye(args, kwargs) (line 71)
    eye_call_result_244301 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), eye_244298, *[int_244299], **kwargs_244300)
    
    # Assigning a type to the variable 'A' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'A', eye_call_result_244301)
    
    # Assigning a Num to a Subscript (line 72):
    
    # Assigning a Num to a Subscript (line 72):
    int_244302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 14), 'int')
    # Getting the type of 'A' (line 72)
    A_244303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'A')
    int_244304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 6), 'int')
    slice_244305 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 72, 4), None, None, None)
    # Storing an element on a container (line 72)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 4), A_244303, ((int_244304, slice_244305), int_244302))
    
    # Assigning a Call to a Name (line 73):
    
    # Assigning a Call to a Name (line 73):
    
    # Call to rand(...): (line 73)
    # Processing the call arguments (line 73)
    int_244309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 23), 'int')
    # Processing the call keyword arguments (line 73)
    kwargs_244310 = {}
    # Getting the type of 'np' (line 73)
    np_244306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 73)
    random_244307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), np_244306, 'random')
    # Obtaining the member 'rand' of a type (line 73)
    rand_244308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), random_244307, 'rand')
    # Calling rand(args, kwargs) (line 73)
    rand_call_result_244311 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), rand_244308, *[int_244309], **kwargs_244310)
    
    # Assigning a type to the variable 'b' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'b', rand_call_result_244311)
    
    # Assigning a Num to a Subscript (line 74):
    
    # Assigning a Num to a Subscript (line 74):
    int_244312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 11), 'int')
    # Getting the type of 'b' (line 74)
    b_244313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'b')
    int_244314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 6), 'int')
    # Storing an element on a container (line 74)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 4), b_244313, (int_244314, int_244312))
    
    # Assigning a Call to a Tuple (line 75):
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    int_244315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'A' (line 75)
    A_244317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 49), 'A', False)
    # Getting the type of 'b' (line 75)
    b_244318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 52), 'b', False)
    # Processing the call keyword arguments (line 75)
    kwargs_244319 = {}
    # Getting the type of '_remove_redundancy' (line 75)
    _remove_redundancy_244316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 75)
    _remove_redundancy_call_result_244320 = invoke(stypy.reporting.localization.Localization(__file__, 75, 30), _remove_redundancy_244316, *[A_244317, b_244318], **kwargs_244319)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___244321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), _remove_redundancy_call_result_244320, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_244322 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), getitem___244321, int_244315)
    
    # Assigning a type to the variable 'tuple_var_assignment_244021' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_244021', subscript_call_result_244322)
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    int_244323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'A' (line 75)
    A_244325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 49), 'A', False)
    # Getting the type of 'b' (line 75)
    b_244326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 52), 'b', False)
    # Processing the call keyword arguments (line 75)
    kwargs_244327 = {}
    # Getting the type of '_remove_redundancy' (line 75)
    _remove_redundancy_244324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 75)
    _remove_redundancy_call_result_244328 = invoke(stypy.reporting.localization.Localization(__file__, 75, 30), _remove_redundancy_244324, *[A_244325, b_244326], **kwargs_244327)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___244329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), _remove_redundancy_call_result_244328, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_244330 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), getitem___244329, int_244323)
    
    # Assigning a type to the variable 'tuple_var_assignment_244022' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_244022', subscript_call_result_244330)
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    int_244331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'A' (line 75)
    A_244333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 49), 'A', False)
    # Getting the type of 'b' (line 75)
    b_244334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 52), 'b', False)
    # Processing the call keyword arguments (line 75)
    kwargs_244335 = {}
    # Getting the type of '_remove_redundancy' (line 75)
    _remove_redundancy_244332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 75)
    _remove_redundancy_call_result_244336 = invoke(stypy.reporting.localization.Localization(__file__, 75, 30), _remove_redundancy_244332, *[A_244333, b_244334], **kwargs_244335)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___244337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), _remove_redundancy_call_result_244336, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_244338 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), getitem___244337, int_244331)
    
    # Assigning a type to the variable 'tuple_var_assignment_244023' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_244023', subscript_call_result_244338)
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    int_244339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'A' (line 75)
    A_244341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 49), 'A', False)
    # Getting the type of 'b' (line 75)
    b_244342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 52), 'b', False)
    # Processing the call keyword arguments (line 75)
    kwargs_244343 = {}
    # Getting the type of '_remove_redundancy' (line 75)
    _remove_redundancy_244340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 75)
    _remove_redundancy_call_result_244344 = invoke(stypy.reporting.localization.Localization(__file__, 75, 30), _remove_redundancy_244340, *[A_244341, b_244342], **kwargs_244343)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___244345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), _remove_redundancy_call_result_244344, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_244346 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), getitem___244345, int_244339)
    
    # Assigning a type to the variable 'tuple_var_assignment_244024' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_244024', subscript_call_result_244346)
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'tuple_var_assignment_244021' (line 75)
    tuple_var_assignment_244021_244347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_244021')
    # Assigning a type to the variable 'A1' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'A1', tuple_var_assignment_244021_244347)
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'tuple_var_assignment_244022' (line 75)
    tuple_var_assignment_244022_244348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_244022')
    # Assigning a type to the variable 'b1' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'b1', tuple_var_assignment_244022_244348)
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'tuple_var_assignment_244023' (line 75)
    tuple_var_assignment_244023_244349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_244023')
    # Assigning a type to the variable 'status' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'status', tuple_var_assignment_244023_244349)
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'tuple_var_assignment_244024' (line 75)
    tuple_var_assignment_244024_244350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_244024')
    # Assigning a type to the variable 'message' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'message', tuple_var_assignment_244024_244350)
    
    # Call to assert_equal(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'status' (line 76)
    status_244352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'status', False)
    int_244353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 25), 'int')
    # Processing the call keyword arguments (line 76)
    kwargs_244354 = {}
    # Getting the type of 'assert_equal' (line 76)
    assert_equal_244351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 76)
    assert_equal_call_result_244355 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), assert_equal_244351, *[status_244352, int_244353], **kwargs_244354)
    
    
    # Call to assert_allclose(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'A1' (line 77)
    A1_244357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'A1', False)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'list' (line 77)
    list_244358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 77)
    # Adding element type (line 77)
    int_244359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 26), list_244358, int_244359)
    # Adding element type (line 77)
    int_244360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 26), list_244358, int_244360)
    
    slice_244361 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 24), None, None, None)
    # Getting the type of 'A' (line 77)
    A_244362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'A', False)
    # Obtaining the member '__getitem__' of a type (line 77)
    getitem___244363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 24), A_244362, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 77)
    subscript_call_result_244364 = invoke(stypy.reporting.localization.Localization(__file__, 77, 24), getitem___244363, (list_244358, slice_244361))
    
    # Processing the call keyword arguments (line 77)
    kwargs_244365 = {}
    # Getting the type of 'assert_allclose' (line 77)
    assert_allclose_244356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 77)
    assert_allclose_call_result_244366 = invoke(stypy.reporting.localization.Localization(__file__, 77, 4), assert_allclose_244356, *[A1_244357, subscript_call_result_244364], **kwargs_244365)
    
    
    # Call to assert_allclose(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'b1' (line 78)
    b1_244368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'b1', False)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'list' (line 78)
    list_244369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 78)
    # Adding element type (line 78)
    int_244370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), list_244369, int_244370)
    # Adding element type (line 78)
    int_244371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), list_244369, int_244371)
    
    # Getting the type of 'b' (line 78)
    b_244372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___244373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 24), b_244372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_244374 = invoke(stypy.reporting.localization.Localization(__file__, 78, 24), getitem___244373, list_244369)
    
    # Processing the call keyword arguments (line 78)
    kwargs_244375 = {}
    # Getting the type of 'assert_allclose' (line 78)
    assert_allclose_244367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 78)
    assert_allclose_call_result_244376 = invoke(stypy.reporting.localization.Localization(__file__, 78, 4), assert_allclose_244367, *[b1_244368, subscript_call_result_244374], **kwargs_244375)
    
    
    # ################# End of 'test_remove_zero_row(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_remove_zero_row' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_244377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_244377)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_remove_zero_row'
    return stypy_return_type_244377

# Assigning a type to the variable 'test_remove_zero_row' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'test_remove_zero_row', test_remove_zero_row)

@norecursion
def test_infeasible_m_gt_n(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_infeasible_m_gt_n'
    module_type_store = module_type_store.open_function_context('test_infeasible_m_gt_n', 81, 0, False)
    
    # Passed parameters checking function
    test_infeasible_m_gt_n.stypy_localization = localization
    test_infeasible_m_gt_n.stypy_type_of_self = None
    test_infeasible_m_gt_n.stypy_type_store = module_type_store
    test_infeasible_m_gt_n.stypy_function_name = 'test_infeasible_m_gt_n'
    test_infeasible_m_gt_n.stypy_param_names_list = []
    test_infeasible_m_gt_n.stypy_varargs_param_name = None
    test_infeasible_m_gt_n.stypy_kwargs_param_name = None
    test_infeasible_m_gt_n.stypy_call_defaults = defaults
    test_infeasible_m_gt_n.stypy_call_varargs = varargs
    test_infeasible_m_gt_n.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_infeasible_m_gt_n', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_infeasible_m_gt_n', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_infeasible_m_gt_n(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 82):
    
    # Assigning a Num to a Name (line 82):
    int_244378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_244025' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'tuple_assignment_244025', int_244378)
    
    # Assigning a Num to a Name (line 82):
    int_244379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 15), 'int')
    # Assigning a type to the variable 'tuple_assignment_244026' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'tuple_assignment_244026', int_244379)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'tuple_assignment_244025' (line 82)
    tuple_assignment_244025_244380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'tuple_assignment_244025')
    # Assigning a type to the variable 'm' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'm', tuple_assignment_244025_244380)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'tuple_assignment_244026' (line 82)
    tuple_assignment_244026_244381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'tuple_assignment_244026')
    # Assigning a type to the variable 'n' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 7), 'n', tuple_assignment_244026_244381)
    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to rand(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'm' (line 83)
    m_244385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'm', False)
    # Getting the type of 'n' (line 83)
    n_244386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'n', False)
    # Processing the call keyword arguments (line 83)
    kwargs_244387 = {}
    # Getting the type of 'np' (line 83)
    np_244382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 83)
    random_244383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 9), np_244382, 'random')
    # Obtaining the member 'rand' of a type (line 83)
    rand_244384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 9), random_244383, 'rand')
    # Calling rand(args, kwargs) (line 83)
    rand_call_result_244388 = invoke(stypy.reporting.localization.Localization(__file__, 83, 9), rand_244384, *[m_244385, n_244386], **kwargs_244387)
    
    # Assigning a type to the variable 'A0' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'A0', rand_call_result_244388)
    
    # Assigning a Call to a Name (line 84):
    
    # Assigning a Call to a Name (line 84):
    
    # Call to rand(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'm' (line 84)
    m_244392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 24), 'm', False)
    # Processing the call keyword arguments (line 84)
    kwargs_244393 = {}
    # Getting the type of 'np' (line 84)
    np_244389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 84)
    random_244390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 9), np_244389, 'random')
    # Obtaining the member 'rand' of a type (line 84)
    rand_244391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 9), random_244390, 'rand')
    # Calling rand(args, kwargs) (line 84)
    rand_call_result_244394 = invoke(stypy.reporting.localization.Localization(__file__, 84, 9), rand_244391, *[m_244392], **kwargs_244393)
    
    # Assigning a type to the variable 'b0' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'b0', rand_call_result_244394)
    
    # Assigning a Call to a Tuple (line 85):
    
    # Assigning a Subscript to a Name (line 85):
    
    # Obtaining the type of the subscript
    int_244395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'A0' (line 85)
    A0_244397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 49), 'A0', False)
    # Getting the type of 'b0' (line 85)
    b0_244398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 53), 'b0', False)
    # Processing the call keyword arguments (line 85)
    kwargs_244399 = {}
    # Getting the type of '_remove_redundancy' (line 85)
    _remove_redundancy_244396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 85)
    _remove_redundancy_call_result_244400 = invoke(stypy.reporting.localization.Localization(__file__, 85, 30), _remove_redundancy_244396, *[A0_244397, b0_244398], **kwargs_244399)
    
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___244401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), _remove_redundancy_call_result_244400, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_244402 = invoke(stypy.reporting.localization.Localization(__file__, 85, 4), getitem___244401, int_244395)
    
    # Assigning a type to the variable 'tuple_var_assignment_244027' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tuple_var_assignment_244027', subscript_call_result_244402)
    
    # Assigning a Subscript to a Name (line 85):
    
    # Obtaining the type of the subscript
    int_244403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'A0' (line 85)
    A0_244405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 49), 'A0', False)
    # Getting the type of 'b0' (line 85)
    b0_244406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 53), 'b0', False)
    # Processing the call keyword arguments (line 85)
    kwargs_244407 = {}
    # Getting the type of '_remove_redundancy' (line 85)
    _remove_redundancy_244404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 85)
    _remove_redundancy_call_result_244408 = invoke(stypy.reporting.localization.Localization(__file__, 85, 30), _remove_redundancy_244404, *[A0_244405, b0_244406], **kwargs_244407)
    
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___244409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), _remove_redundancy_call_result_244408, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_244410 = invoke(stypy.reporting.localization.Localization(__file__, 85, 4), getitem___244409, int_244403)
    
    # Assigning a type to the variable 'tuple_var_assignment_244028' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tuple_var_assignment_244028', subscript_call_result_244410)
    
    # Assigning a Subscript to a Name (line 85):
    
    # Obtaining the type of the subscript
    int_244411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'A0' (line 85)
    A0_244413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 49), 'A0', False)
    # Getting the type of 'b0' (line 85)
    b0_244414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 53), 'b0', False)
    # Processing the call keyword arguments (line 85)
    kwargs_244415 = {}
    # Getting the type of '_remove_redundancy' (line 85)
    _remove_redundancy_244412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 85)
    _remove_redundancy_call_result_244416 = invoke(stypy.reporting.localization.Localization(__file__, 85, 30), _remove_redundancy_244412, *[A0_244413, b0_244414], **kwargs_244415)
    
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___244417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), _remove_redundancy_call_result_244416, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_244418 = invoke(stypy.reporting.localization.Localization(__file__, 85, 4), getitem___244417, int_244411)
    
    # Assigning a type to the variable 'tuple_var_assignment_244029' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tuple_var_assignment_244029', subscript_call_result_244418)
    
    # Assigning a Subscript to a Name (line 85):
    
    # Obtaining the type of the subscript
    int_244419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'A0' (line 85)
    A0_244421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 49), 'A0', False)
    # Getting the type of 'b0' (line 85)
    b0_244422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 53), 'b0', False)
    # Processing the call keyword arguments (line 85)
    kwargs_244423 = {}
    # Getting the type of '_remove_redundancy' (line 85)
    _remove_redundancy_244420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 85)
    _remove_redundancy_call_result_244424 = invoke(stypy.reporting.localization.Localization(__file__, 85, 30), _remove_redundancy_244420, *[A0_244421, b0_244422], **kwargs_244423)
    
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___244425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), _remove_redundancy_call_result_244424, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_244426 = invoke(stypy.reporting.localization.Localization(__file__, 85, 4), getitem___244425, int_244419)
    
    # Assigning a type to the variable 'tuple_var_assignment_244030' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tuple_var_assignment_244030', subscript_call_result_244426)
    
    # Assigning a Name to a Name (line 85):
    # Getting the type of 'tuple_var_assignment_244027' (line 85)
    tuple_var_assignment_244027_244427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tuple_var_assignment_244027')
    # Assigning a type to the variable 'A1' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'A1', tuple_var_assignment_244027_244427)
    
    # Assigning a Name to a Name (line 85):
    # Getting the type of 'tuple_var_assignment_244028' (line 85)
    tuple_var_assignment_244028_244428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tuple_var_assignment_244028')
    # Assigning a type to the variable 'b1' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'b1', tuple_var_assignment_244028_244428)
    
    # Assigning a Name to a Name (line 85):
    # Getting the type of 'tuple_var_assignment_244029' (line 85)
    tuple_var_assignment_244029_244429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tuple_var_assignment_244029')
    # Assigning a type to the variable 'status' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'status', tuple_var_assignment_244029_244429)
    
    # Assigning a Name to a Name (line 85):
    # Getting the type of 'tuple_var_assignment_244030' (line 85)
    tuple_var_assignment_244030_244430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tuple_var_assignment_244030')
    # Assigning a type to the variable 'message' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'message', tuple_var_assignment_244030_244430)
    
    # Call to assert_equal(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'status' (line 86)
    status_244432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 17), 'status', False)
    int_244433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'int')
    # Processing the call keyword arguments (line 86)
    kwargs_244434 = {}
    # Getting the type of 'assert_equal' (line 86)
    assert_equal_244431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 86)
    assert_equal_call_result_244435 = invoke(stypy.reporting.localization.Localization(__file__, 86, 4), assert_equal_244431, *[status_244432, int_244433], **kwargs_244434)
    
    
    # ################# End of 'test_infeasible_m_gt_n(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_infeasible_m_gt_n' in the type store
    # Getting the type of 'stypy_return_type' (line 81)
    stypy_return_type_244436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_244436)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_infeasible_m_gt_n'
    return stypy_return_type_244436

# Assigning a type to the variable 'test_infeasible_m_gt_n' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'test_infeasible_m_gt_n', test_infeasible_m_gt_n)

@norecursion
def test_infeasible_m_eq_n(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_infeasible_m_eq_n'
    module_type_store = module_type_store.open_function_context('test_infeasible_m_eq_n', 89, 0, False)
    
    # Passed parameters checking function
    test_infeasible_m_eq_n.stypy_localization = localization
    test_infeasible_m_eq_n.stypy_type_of_self = None
    test_infeasible_m_eq_n.stypy_type_store = module_type_store
    test_infeasible_m_eq_n.stypy_function_name = 'test_infeasible_m_eq_n'
    test_infeasible_m_eq_n.stypy_param_names_list = []
    test_infeasible_m_eq_n.stypy_varargs_param_name = None
    test_infeasible_m_eq_n.stypy_kwargs_param_name = None
    test_infeasible_m_eq_n.stypy_call_defaults = defaults
    test_infeasible_m_eq_n.stypy_call_varargs = varargs
    test_infeasible_m_eq_n.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_infeasible_m_eq_n', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_infeasible_m_eq_n', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_infeasible_m_eq_n(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 90):
    
    # Assigning a Num to a Name (line 90):
    int_244437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_244031' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_assignment_244031', int_244437)
    
    # Assigning a Num to a Name (line 90):
    int_244438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 15), 'int')
    # Assigning a type to the variable 'tuple_assignment_244032' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_assignment_244032', int_244438)
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'tuple_assignment_244031' (line 90)
    tuple_assignment_244031_244439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_assignment_244031')
    # Assigning a type to the variable 'm' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'm', tuple_assignment_244031_244439)
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'tuple_assignment_244032' (line 90)
    tuple_assignment_244032_244440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_assignment_244032')
    # Assigning a type to the variable 'n' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 7), 'n', tuple_assignment_244032_244440)
    
    # Assigning a Call to a Name (line 91):
    
    # Assigning a Call to a Name (line 91):
    
    # Call to rand(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'm' (line 91)
    m_244444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'm', False)
    # Getting the type of 'n' (line 91)
    n_244445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'n', False)
    # Processing the call keyword arguments (line 91)
    kwargs_244446 = {}
    # Getting the type of 'np' (line 91)
    np_244441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 91)
    random_244442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 9), np_244441, 'random')
    # Obtaining the member 'rand' of a type (line 91)
    rand_244443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 9), random_244442, 'rand')
    # Calling rand(args, kwargs) (line 91)
    rand_call_result_244447 = invoke(stypy.reporting.localization.Localization(__file__, 91, 9), rand_244443, *[m_244444, n_244445], **kwargs_244446)
    
    # Assigning a type to the variable 'A0' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'A0', rand_call_result_244447)
    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to rand(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'm' (line 92)
    m_244451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'm', False)
    # Processing the call keyword arguments (line 92)
    kwargs_244452 = {}
    # Getting the type of 'np' (line 92)
    np_244448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 92)
    random_244449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 9), np_244448, 'random')
    # Obtaining the member 'rand' of a type (line 92)
    rand_244450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 9), random_244449, 'rand')
    # Calling rand(args, kwargs) (line 92)
    rand_call_result_244453 = invoke(stypy.reporting.localization.Localization(__file__, 92, 9), rand_244450, *[m_244451], **kwargs_244452)
    
    # Assigning a type to the variable 'b0' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'b0', rand_call_result_244453)
    
    # Assigning a BinOp to a Subscript (line 93):
    
    # Assigning a BinOp to a Subscript (line 93):
    int_244454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 16), 'int')
    
    # Obtaining the type of the subscript
    int_244455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 23), 'int')
    slice_244456 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 93, 20), None, None, None)
    # Getting the type of 'A0' (line 93)
    A0_244457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'A0')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___244458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 20), A0_244457, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_244459 = invoke(stypy.reporting.localization.Localization(__file__, 93, 20), getitem___244458, (int_244455, slice_244456))
    
    # Applying the binary operator '*' (line 93)
    result_mul_244460 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 16), '*', int_244454, subscript_call_result_244459)
    
    # Getting the type of 'A0' (line 93)
    A0_244461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'A0')
    int_244462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 7), 'int')
    slice_244463 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 93, 4), None, None, None)
    # Storing an element on a container (line 93)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 4), A0_244461, ((int_244462, slice_244463), result_mul_244460))
    
    # Assigning a Call to a Tuple (line 94):
    
    # Assigning a Subscript to a Name (line 94):
    
    # Obtaining the type of the subscript
    int_244464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'A0' (line 94)
    A0_244466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 49), 'A0', False)
    # Getting the type of 'b0' (line 94)
    b0_244467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 53), 'b0', False)
    # Processing the call keyword arguments (line 94)
    kwargs_244468 = {}
    # Getting the type of '_remove_redundancy' (line 94)
    _remove_redundancy_244465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 94)
    _remove_redundancy_call_result_244469 = invoke(stypy.reporting.localization.Localization(__file__, 94, 30), _remove_redundancy_244465, *[A0_244466, b0_244467], **kwargs_244468)
    
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___244470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 4), _remove_redundancy_call_result_244469, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_244471 = invoke(stypy.reporting.localization.Localization(__file__, 94, 4), getitem___244470, int_244464)
    
    # Assigning a type to the variable 'tuple_var_assignment_244033' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_var_assignment_244033', subscript_call_result_244471)
    
    # Assigning a Subscript to a Name (line 94):
    
    # Obtaining the type of the subscript
    int_244472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'A0' (line 94)
    A0_244474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 49), 'A0', False)
    # Getting the type of 'b0' (line 94)
    b0_244475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 53), 'b0', False)
    # Processing the call keyword arguments (line 94)
    kwargs_244476 = {}
    # Getting the type of '_remove_redundancy' (line 94)
    _remove_redundancy_244473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 94)
    _remove_redundancy_call_result_244477 = invoke(stypy.reporting.localization.Localization(__file__, 94, 30), _remove_redundancy_244473, *[A0_244474, b0_244475], **kwargs_244476)
    
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___244478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 4), _remove_redundancy_call_result_244477, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_244479 = invoke(stypy.reporting.localization.Localization(__file__, 94, 4), getitem___244478, int_244472)
    
    # Assigning a type to the variable 'tuple_var_assignment_244034' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_var_assignment_244034', subscript_call_result_244479)
    
    # Assigning a Subscript to a Name (line 94):
    
    # Obtaining the type of the subscript
    int_244480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'A0' (line 94)
    A0_244482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 49), 'A0', False)
    # Getting the type of 'b0' (line 94)
    b0_244483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 53), 'b0', False)
    # Processing the call keyword arguments (line 94)
    kwargs_244484 = {}
    # Getting the type of '_remove_redundancy' (line 94)
    _remove_redundancy_244481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 94)
    _remove_redundancy_call_result_244485 = invoke(stypy.reporting.localization.Localization(__file__, 94, 30), _remove_redundancy_244481, *[A0_244482, b0_244483], **kwargs_244484)
    
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___244486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 4), _remove_redundancy_call_result_244485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_244487 = invoke(stypy.reporting.localization.Localization(__file__, 94, 4), getitem___244486, int_244480)
    
    # Assigning a type to the variable 'tuple_var_assignment_244035' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_var_assignment_244035', subscript_call_result_244487)
    
    # Assigning a Subscript to a Name (line 94):
    
    # Obtaining the type of the subscript
    int_244488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'A0' (line 94)
    A0_244490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 49), 'A0', False)
    # Getting the type of 'b0' (line 94)
    b0_244491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 53), 'b0', False)
    # Processing the call keyword arguments (line 94)
    kwargs_244492 = {}
    # Getting the type of '_remove_redundancy' (line 94)
    _remove_redundancy_244489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 94)
    _remove_redundancy_call_result_244493 = invoke(stypy.reporting.localization.Localization(__file__, 94, 30), _remove_redundancy_244489, *[A0_244490, b0_244491], **kwargs_244492)
    
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___244494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 4), _remove_redundancy_call_result_244493, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_244495 = invoke(stypy.reporting.localization.Localization(__file__, 94, 4), getitem___244494, int_244488)
    
    # Assigning a type to the variable 'tuple_var_assignment_244036' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_var_assignment_244036', subscript_call_result_244495)
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'tuple_var_assignment_244033' (line 94)
    tuple_var_assignment_244033_244496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_var_assignment_244033')
    # Assigning a type to the variable 'A1' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'A1', tuple_var_assignment_244033_244496)
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'tuple_var_assignment_244034' (line 94)
    tuple_var_assignment_244034_244497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_var_assignment_244034')
    # Assigning a type to the variable 'b1' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'b1', tuple_var_assignment_244034_244497)
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'tuple_var_assignment_244035' (line 94)
    tuple_var_assignment_244035_244498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_var_assignment_244035')
    # Assigning a type to the variable 'status' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'status', tuple_var_assignment_244035_244498)
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'tuple_var_assignment_244036' (line 94)
    tuple_var_assignment_244036_244499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_var_assignment_244036')
    # Assigning a type to the variable 'message' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'message', tuple_var_assignment_244036_244499)
    
    # Call to assert_equal(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'status' (line 95)
    status_244501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 17), 'status', False)
    int_244502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 25), 'int')
    # Processing the call keyword arguments (line 95)
    kwargs_244503 = {}
    # Getting the type of 'assert_equal' (line 95)
    assert_equal_244500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 95)
    assert_equal_call_result_244504 = invoke(stypy.reporting.localization.Localization(__file__, 95, 4), assert_equal_244500, *[status_244501, int_244502], **kwargs_244503)
    
    
    # ################# End of 'test_infeasible_m_eq_n(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_infeasible_m_eq_n' in the type store
    # Getting the type of 'stypy_return_type' (line 89)
    stypy_return_type_244505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_244505)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_infeasible_m_eq_n'
    return stypy_return_type_244505

# Assigning a type to the variable 'test_infeasible_m_eq_n' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'test_infeasible_m_eq_n', test_infeasible_m_eq_n)

@norecursion
def test_infeasible_m_lt_n(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_infeasible_m_lt_n'
    module_type_store = module_type_store.open_function_context('test_infeasible_m_lt_n', 98, 0, False)
    
    # Passed parameters checking function
    test_infeasible_m_lt_n.stypy_localization = localization
    test_infeasible_m_lt_n.stypy_type_of_self = None
    test_infeasible_m_lt_n.stypy_type_store = module_type_store
    test_infeasible_m_lt_n.stypy_function_name = 'test_infeasible_m_lt_n'
    test_infeasible_m_lt_n.stypy_param_names_list = []
    test_infeasible_m_lt_n.stypy_varargs_param_name = None
    test_infeasible_m_lt_n.stypy_kwargs_param_name = None
    test_infeasible_m_lt_n.stypy_call_defaults = defaults
    test_infeasible_m_lt_n.stypy_call_varargs = varargs
    test_infeasible_m_lt_n.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_infeasible_m_lt_n', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_infeasible_m_lt_n', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_infeasible_m_lt_n(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 99):
    
    # Assigning a Num to a Name (line 99):
    int_244506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_244037' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_assignment_244037', int_244506)
    
    # Assigning a Num to a Name (line 99):
    int_244507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 14), 'int')
    # Assigning a type to the variable 'tuple_assignment_244038' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_assignment_244038', int_244507)
    
    # Assigning a Name to a Name (line 99):
    # Getting the type of 'tuple_assignment_244037' (line 99)
    tuple_assignment_244037_244508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_assignment_244037')
    # Assigning a type to the variable 'm' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'm', tuple_assignment_244037_244508)
    
    # Assigning a Name to a Name (line 99):
    # Getting the type of 'tuple_assignment_244038' (line 99)
    tuple_assignment_244038_244509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_assignment_244038')
    # Assigning a type to the variable 'n' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 7), 'n', tuple_assignment_244038_244509)
    
    # Assigning a Call to a Name (line 100):
    
    # Assigning a Call to a Name (line 100):
    
    # Call to rand(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'm' (line 100)
    m_244513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'm', False)
    # Getting the type of 'n' (line 100)
    n_244514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'n', False)
    # Processing the call keyword arguments (line 100)
    kwargs_244515 = {}
    # Getting the type of 'np' (line 100)
    np_244510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 100)
    random_244511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 9), np_244510, 'random')
    # Obtaining the member 'rand' of a type (line 100)
    rand_244512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 9), random_244511, 'rand')
    # Calling rand(args, kwargs) (line 100)
    rand_call_result_244516 = invoke(stypy.reporting.localization.Localization(__file__, 100, 9), rand_244512, *[m_244513, n_244514], **kwargs_244515)
    
    # Assigning a type to the variable 'A0' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'A0', rand_call_result_244516)
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to rand(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'm' (line 101)
    m_244520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'm', False)
    # Processing the call keyword arguments (line 101)
    kwargs_244521 = {}
    # Getting the type of 'np' (line 101)
    np_244517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 101)
    random_244518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 9), np_244517, 'random')
    # Obtaining the member 'rand' of a type (line 101)
    rand_244519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 9), random_244518, 'rand')
    # Calling rand(args, kwargs) (line 101)
    rand_call_result_244522 = invoke(stypy.reporting.localization.Localization(__file__, 101, 9), rand_244519, *[m_244520], **kwargs_244521)
    
    # Assigning a type to the variable 'b0' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'b0', rand_call_result_244522)
    
    # Assigning a Call to a Subscript (line 102):
    
    # Assigning a Call to a Subscript (line 102):
    
    # Call to dot(...): (line 102)
    # Processing the call arguments (line 102)
    
    # Obtaining the type of the subscript
    int_244531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 41), 'int')
    slice_244532 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 102, 37), None, int_244531, None)
    # Getting the type of 'A0' (line 102)
    A0_244533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 37), 'A0', False)
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___244534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 37), A0_244533, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_244535 = invoke(stypy.reporting.localization.Localization(__file__, 102, 37), getitem___244534, slice_244532)
    
    # Processing the call keyword arguments (line 102)
    kwargs_244536 = {}
    
    # Call to arange(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'm' (line 102)
    m_244525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 26), 'm', False)
    int_244526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 30), 'int')
    # Applying the binary operator '-' (line 102)
    result_sub_244527 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 26), '-', m_244525, int_244526)
    
    # Processing the call keyword arguments (line 102)
    kwargs_244528 = {}
    # Getting the type of 'np' (line 102)
    np_244523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'np', False)
    # Obtaining the member 'arange' of a type (line 102)
    arange_244524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 16), np_244523, 'arange')
    # Calling arange(args, kwargs) (line 102)
    arange_call_result_244529 = invoke(stypy.reporting.localization.Localization(__file__, 102, 16), arange_244524, *[result_sub_244527], **kwargs_244528)
    
    # Obtaining the member 'dot' of a type (line 102)
    dot_244530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 16), arange_call_result_244529, 'dot')
    # Calling dot(args, kwargs) (line 102)
    dot_call_result_244537 = invoke(stypy.reporting.localization.Localization(__file__, 102, 16), dot_244530, *[subscript_call_result_244535], **kwargs_244536)
    
    # Getting the type of 'A0' (line 102)
    A0_244538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'A0')
    int_244539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 7), 'int')
    slice_244540 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 102, 4), None, None, None)
    # Storing an element on a container (line 102)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 4), A0_244538, ((int_244539, slice_244540), dot_call_result_244537))
    
    # Assigning a Call to a Tuple (line 103):
    
    # Assigning a Subscript to a Name (line 103):
    
    # Obtaining the type of the subscript
    int_244541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'A0' (line 103)
    A0_244543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 49), 'A0', False)
    # Getting the type of 'b0' (line 103)
    b0_244544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 53), 'b0', False)
    # Processing the call keyword arguments (line 103)
    kwargs_244545 = {}
    # Getting the type of '_remove_redundancy' (line 103)
    _remove_redundancy_244542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 103)
    _remove_redundancy_call_result_244546 = invoke(stypy.reporting.localization.Localization(__file__, 103, 30), _remove_redundancy_244542, *[A0_244543, b0_244544], **kwargs_244545)
    
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___244547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 4), _remove_redundancy_call_result_244546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_244548 = invoke(stypy.reporting.localization.Localization(__file__, 103, 4), getitem___244547, int_244541)
    
    # Assigning a type to the variable 'tuple_var_assignment_244039' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'tuple_var_assignment_244039', subscript_call_result_244548)
    
    # Assigning a Subscript to a Name (line 103):
    
    # Obtaining the type of the subscript
    int_244549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'A0' (line 103)
    A0_244551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 49), 'A0', False)
    # Getting the type of 'b0' (line 103)
    b0_244552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 53), 'b0', False)
    # Processing the call keyword arguments (line 103)
    kwargs_244553 = {}
    # Getting the type of '_remove_redundancy' (line 103)
    _remove_redundancy_244550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 103)
    _remove_redundancy_call_result_244554 = invoke(stypy.reporting.localization.Localization(__file__, 103, 30), _remove_redundancy_244550, *[A0_244551, b0_244552], **kwargs_244553)
    
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___244555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 4), _remove_redundancy_call_result_244554, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_244556 = invoke(stypy.reporting.localization.Localization(__file__, 103, 4), getitem___244555, int_244549)
    
    # Assigning a type to the variable 'tuple_var_assignment_244040' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'tuple_var_assignment_244040', subscript_call_result_244556)
    
    # Assigning a Subscript to a Name (line 103):
    
    # Obtaining the type of the subscript
    int_244557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'A0' (line 103)
    A0_244559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 49), 'A0', False)
    # Getting the type of 'b0' (line 103)
    b0_244560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 53), 'b0', False)
    # Processing the call keyword arguments (line 103)
    kwargs_244561 = {}
    # Getting the type of '_remove_redundancy' (line 103)
    _remove_redundancy_244558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 103)
    _remove_redundancy_call_result_244562 = invoke(stypy.reporting.localization.Localization(__file__, 103, 30), _remove_redundancy_244558, *[A0_244559, b0_244560], **kwargs_244561)
    
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___244563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 4), _remove_redundancy_call_result_244562, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_244564 = invoke(stypy.reporting.localization.Localization(__file__, 103, 4), getitem___244563, int_244557)
    
    # Assigning a type to the variable 'tuple_var_assignment_244041' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'tuple_var_assignment_244041', subscript_call_result_244564)
    
    # Assigning a Subscript to a Name (line 103):
    
    # Obtaining the type of the subscript
    int_244565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'A0' (line 103)
    A0_244567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 49), 'A0', False)
    # Getting the type of 'b0' (line 103)
    b0_244568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 53), 'b0', False)
    # Processing the call keyword arguments (line 103)
    kwargs_244569 = {}
    # Getting the type of '_remove_redundancy' (line 103)
    _remove_redundancy_244566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 103)
    _remove_redundancy_call_result_244570 = invoke(stypy.reporting.localization.Localization(__file__, 103, 30), _remove_redundancy_244566, *[A0_244567, b0_244568], **kwargs_244569)
    
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___244571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 4), _remove_redundancy_call_result_244570, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_244572 = invoke(stypy.reporting.localization.Localization(__file__, 103, 4), getitem___244571, int_244565)
    
    # Assigning a type to the variable 'tuple_var_assignment_244042' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'tuple_var_assignment_244042', subscript_call_result_244572)
    
    # Assigning a Name to a Name (line 103):
    # Getting the type of 'tuple_var_assignment_244039' (line 103)
    tuple_var_assignment_244039_244573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'tuple_var_assignment_244039')
    # Assigning a type to the variable 'A1' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'A1', tuple_var_assignment_244039_244573)
    
    # Assigning a Name to a Name (line 103):
    # Getting the type of 'tuple_var_assignment_244040' (line 103)
    tuple_var_assignment_244040_244574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'tuple_var_assignment_244040')
    # Assigning a type to the variable 'b1' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'b1', tuple_var_assignment_244040_244574)
    
    # Assigning a Name to a Name (line 103):
    # Getting the type of 'tuple_var_assignment_244041' (line 103)
    tuple_var_assignment_244041_244575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'tuple_var_assignment_244041')
    # Assigning a type to the variable 'status' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'status', tuple_var_assignment_244041_244575)
    
    # Assigning a Name to a Name (line 103):
    # Getting the type of 'tuple_var_assignment_244042' (line 103)
    tuple_var_assignment_244042_244576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'tuple_var_assignment_244042')
    # Assigning a type to the variable 'message' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'message', tuple_var_assignment_244042_244576)
    
    # Call to assert_equal(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'status' (line 104)
    status_244578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'status', False)
    int_244579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 25), 'int')
    # Processing the call keyword arguments (line 104)
    kwargs_244580 = {}
    # Getting the type of 'assert_equal' (line 104)
    assert_equal_244577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 104)
    assert_equal_call_result_244581 = invoke(stypy.reporting.localization.Localization(__file__, 104, 4), assert_equal_244577, *[status_244578, int_244579], **kwargs_244580)
    
    
    # ################# End of 'test_infeasible_m_lt_n(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_infeasible_m_lt_n' in the type store
    # Getting the type of 'stypy_return_type' (line 98)
    stypy_return_type_244582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_244582)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_infeasible_m_lt_n'
    return stypy_return_type_244582

# Assigning a type to the variable 'test_infeasible_m_lt_n' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'test_infeasible_m_lt_n', test_infeasible_m_lt_n)

@norecursion
def test_m_gt_n(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_m_gt_n'
    module_type_store = module_type_store.open_function_context('test_m_gt_n', 107, 0, False)
    
    # Passed parameters checking function
    test_m_gt_n.stypy_localization = localization
    test_m_gt_n.stypy_type_of_self = None
    test_m_gt_n.stypy_type_store = module_type_store
    test_m_gt_n.stypy_function_name = 'test_m_gt_n'
    test_m_gt_n.stypy_param_names_list = []
    test_m_gt_n.stypy_varargs_param_name = None
    test_m_gt_n.stypy_kwargs_param_name = None
    test_m_gt_n.stypy_call_defaults = defaults
    test_m_gt_n.stypy_call_varargs = varargs
    test_m_gt_n.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_m_gt_n', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_m_gt_n', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_m_gt_n(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 108):
    
    # Assigning a Num to a Name (line 108):
    int_244583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_244043' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'tuple_assignment_244043', int_244583)
    
    # Assigning a Num to a Name (line 108):
    int_244584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 15), 'int')
    # Assigning a type to the variable 'tuple_assignment_244044' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'tuple_assignment_244044', int_244584)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_assignment_244043' (line 108)
    tuple_assignment_244043_244585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'tuple_assignment_244043')
    # Assigning a type to the variable 'm' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'm', tuple_assignment_244043_244585)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_assignment_244044' (line 108)
    tuple_assignment_244044_244586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'tuple_assignment_244044')
    # Assigning a type to the variable 'n' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 7), 'n', tuple_assignment_244044_244586)
    
    # Assigning a Call to a Name (line 109):
    
    # Assigning a Call to a Name (line 109):
    
    # Call to rand(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'm' (line 109)
    m_244590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 24), 'm', False)
    # Getting the type of 'n' (line 109)
    n_244591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'n', False)
    # Processing the call keyword arguments (line 109)
    kwargs_244592 = {}
    # Getting the type of 'np' (line 109)
    np_244587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 109)
    random_244588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 9), np_244587, 'random')
    # Obtaining the member 'rand' of a type (line 109)
    rand_244589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 9), random_244588, 'rand')
    # Calling rand(args, kwargs) (line 109)
    rand_call_result_244593 = invoke(stypy.reporting.localization.Localization(__file__, 109, 9), rand_244589, *[m_244590, n_244591], **kwargs_244592)
    
    # Assigning a type to the variable 'A0' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'A0', rand_call_result_244593)
    
    # Assigning a Call to a Name (line 110):
    
    # Assigning a Call to a Name (line 110):
    
    # Call to rand(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'm' (line 110)
    m_244597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 24), 'm', False)
    # Processing the call keyword arguments (line 110)
    kwargs_244598 = {}
    # Getting the type of 'np' (line 110)
    np_244594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 110)
    random_244595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 9), np_244594, 'random')
    # Obtaining the member 'rand' of a type (line 110)
    rand_244596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 9), random_244595, 'rand')
    # Calling rand(args, kwargs) (line 110)
    rand_call_result_244599 = invoke(stypy.reporting.localization.Localization(__file__, 110, 9), rand_244596, *[m_244597], **kwargs_244598)
    
    # Assigning a type to the variable 'b0' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'b0', rand_call_result_244599)
    
    # Assigning a Call to a Name (line 111):
    
    # Assigning a Call to a Name (line 111):
    
    # Call to solve(...): (line 111)
    # Processing the call arguments (line 111)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 111)
    n_244603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 28), 'n', False)
    slice_244604 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 111, 24), None, n_244603, None)
    slice_244605 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 111, 24), None, None, None)
    # Getting the type of 'A0' (line 111)
    A0_244606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'A0', False)
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___244607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), A0_244606, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_244608 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), getitem___244607, (slice_244604, slice_244605))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 111)
    n_244609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 39), 'n', False)
    slice_244610 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 111, 35), None, n_244609, None)
    # Getting the type of 'b0' (line 111)
    b0_244611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 35), 'b0', False)
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___244612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 35), b0_244611, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_244613 = invoke(stypy.reporting.localization.Localization(__file__, 111, 35), getitem___244612, slice_244610)
    
    # Processing the call keyword arguments (line 111)
    kwargs_244614 = {}
    # Getting the type of 'np' (line 111)
    np_244600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'np', False)
    # Obtaining the member 'linalg' of a type (line 111)
    linalg_244601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), np_244600, 'linalg')
    # Obtaining the member 'solve' of a type (line 111)
    solve_244602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), linalg_244601, 'solve')
    # Calling solve(args, kwargs) (line 111)
    solve_call_result_244615 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), solve_244602, *[subscript_call_result_244608, subscript_call_result_244613], **kwargs_244614)
    
    # Assigning a type to the variable 'x' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'x', solve_call_result_244615)
    
    # Assigning a Call to a Subscript (line 112):
    
    # Assigning a Call to a Subscript (line 112):
    
    # Call to dot(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'x' (line 112)
    x_244623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), 'x', False)
    # Processing the call keyword arguments (line 112)
    kwargs_244624 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 112)
    n_244616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'n', False)
    slice_244617 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 112, 13), n_244616, None, None)
    slice_244618 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 112, 13), None, None, None)
    # Getting the type of 'A0' (line 112)
    A0_244619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 13), 'A0', False)
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___244620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 13), A0_244619, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_244621 = invoke(stypy.reporting.localization.Localization(__file__, 112, 13), getitem___244620, (slice_244617, slice_244618))
    
    # Obtaining the member 'dot' of a type (line 112)
    dot_244622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 13), subscript_call_result_244621, 'dot')
    # Calling dot(args, kwargs) (line 112)
    dot_call_result_244625 = invoke(stypy.reporting.localization.Localization(__file__, 112, 13), dot_244622, *[x_244623], **kwargs_244624)
    
    # Getting the type of 'b0' (line 112)
    b0_244626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'b0')
    # Getting the type of 'n' (line 112)
    n_244627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 'n')
    slice_244628 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 112, 4), n_244627, None, None)
    # Storing an element on a container (line 112)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 4), b0_244626, (slice_244628, dot_call_result_244625))
    
    # Assigning a Call to a Tuple (line 113):
    
    # Assigning a Subscript to a Name (line 113):
    
    # Obtaining the type of the subscript
    int_244629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'A0' (line 113)
    A0_244631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 49), 'A0', False)
    # Getting the type of 'b0' (line 113)
    b0_244632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 53), 'b0', False)
    # Processing the call keyword arguments (line 113)
    kwargs_244633 = {}
    # Getting the type of '_remove_redundancy' (line 113)
    _remove_redundancy_244630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 113)
    _remove_redundancy_call_result_244634 = invoke(stypy.reporting.localization.Localization(__file__, 113, 30), _remove_redundancy_244630, *[A0_244631, b0_244632], **kwargs_244633)
    
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___244635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 4), _remove_redundancy_call_result_244634, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_244636 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), getitem___244635, int_244629)
    
    # Assigning a type to the variable 'tuple_var_assignment_244045' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'tuple_var_assignment_244045', subscript_call_result_244636)
    
    # Assigning a Subscript to a Name (line 113):
    
    # Obtaining the type of the subscript
    int_244637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'A0' (line 113)
    A0_244639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 49), 'A0', False)
    # Getting the type of 'b0' (line 113)
    b0_244640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 53), 'b0', False)
    # Processing the call keyword arguments (line 113)
    kwargs_244641 = {}
    # Getting the type of '_remove_redundancy' (line 113)
    _remove_redundancy_244638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 113)
    _remove_redundancy_call_result_244642 = invoke(stypy.reporting.localization.Localization(__file__, 113, 30), _remove_redundancy_244638, *[A0_244639, b0_244640], **kwargs_244641)
    
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___244643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 4), _remove_redundancy_call_result_244642, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_244644 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), getitem___244643, int_244637)
    
    # Assigning a type to the variable 'tuple_var_assignment_244046' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'tuple_var_assignment_244046', subscript_call_result_244644)
    
    # Assigning a Subscript to a Name (line 113):
    
    # Obtaining the type of the subscript
    int_244645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'A0' (line 113)
    A0_244647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 49), 'A0', False)
    # Getting the type of 'b0' (line 113)
    b0_244648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 53), 'b0', False)
    # Processing the call keyword arguments (line 113)
    kwargs_244649 = {}
    # Getting the type of '_remove_redundancy' (line 113)
    _remove_redundancy_244646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 113)
    _remove_redundancy_call_result_244650 = invoke(stypy.reporting.localization.Localization(__file__, 113, 30), _remove_redundancy_244646, *[A0_244647, b0_244648], **kwargs_244649)
    
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___244651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 4), _remove_redundancy_call_result_244650, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_244652 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), getitem___244651, int_244645)
    
    # Assigning a type to the variable 'tuple_var_assignment_244047' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'tuple_var_assignment_244047', subscript_call_result_244652)
    
    # Assigning a Subscript to a Name (line 113):
    
    # Obtaining the type of the subscript
    int_244653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'A0' (line 113)
    A0_244655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 49), 'A0', False)
    # Getting the type of 'b0' (line 113)
    b0_244656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 53), 'b0', False)
    # Processing the call keyword arguments (line 113)
    kwargs_244657 = {}
    # Getting the type of '_remove_redundancy' (line 113)
    _remove_redundancy_244654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 113)
    _remove_redundancy_call_result_244658 = invoke(stypy.reporting.localization.Localization(__file__, 113, 30), _remove_redundancy_244654, *[A0_244655, b0_244656], **kwargs_244657)
    
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___244659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 4), _remove_redundancy_call_result_244658, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_244660 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), getitem___244659, int_244653)
    
    # Assigning a type to the variable 'tuple_var_assignment_244048' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'tuple_var_assignment_244048', subscript_call_result_244660)
    
    # Assigning a Name to a Name (line 113):
    # Getting the type of 'tuple_var_assignment_244045' (line 113)
    tuple_var_assignment_244045_244661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'tuple_var_assignment_244045')
    # Assigning a type to the variable 'A1' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'A1', tuple_var_assignment_244045_244661)
    
    # Assigning a Name to a Name (line 113):
    # Getting the type of 'tuple_var_assignment_244046' (line 113)
    tuple_var_assignment_244046_244662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'tuple_var_assignment_244046')
    # Assigning a type to the variable 'b1' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'b1', tuple_var_assignment_244046_244662)
    
    # Assigning a Name to a Name (line 113):
    # Getting the type of 'tuple_var_assignment_244047' (line 113)
    tuple_var_assignment_244047_244663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'tuple_var_assignment_244047')
    # Assigning a type to the variable 'status' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'status', tuple_var_assignment_244047_244663)
    
    # Assigning a Name to a Name (line 113):
    # Getting the type of 'tuple_var_assignment_244048' (line 113)
    tuple_var_assignment_244048_244664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'tuple_var_assignment_244048')
    # Assigning a type to the variable 'message' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'message', tuple_var_assignment_244048_244664)
    
    # Call to assert_equal(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'status' (line 114)
    status_244666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'status', False)
    int_244667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 25), 'int')
    # Processing the call keyword arguments (line 114)
    kwargs_244668 = {}
    # Getting the type of 'assert_equal' (line 114)
    assert_equal_244665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 114)
    assert_equal_call_result_244669 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), assert_equal_244665, *[status_244666, int_244667], **kwargs_244668)
    
    
    # Call to assert_equal(...): (line 115)
    # Processing the call arguments (line 115)
    
    # Obtaining the type of the subscript
    int_244671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 26), 'int')
    # Getting the type of 'A1' (line 115)
    A1_244672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'A1', False)
    # Obtaining the member 'shape' of a type (line 115)
    shape_244673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 17), A1_244672, 'shape')
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___244674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 17), shape_244673, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_244675 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), getitem___244674, int_244671)
    
    # Getting the type of 'n' (line 115)
    n_244676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 30), 'n', False)
    # Processing the call keyword arguments (line 115)
    kwargs_244677 = {}
    # Getting the type of 'assert_equal' (line 115)
    assert_equal_244670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 115)
    assert_equal_call_result_244678 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), assert_equal_244670, *[subscript_call_result_244675, n_244676], **kwargs_244677)
    
    
    # Call to assert_equal(...): (line 116)
    # Processing the call arguments (line 116)
    
    # Call to matrix_rank(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'A1' (line 116)
    A1_244683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 39), 'A1', False)
    # Processing the call keyword arguments (line 116)
    kwargs_244684 = {}
    # Getting the type of 'np' (line 116)
    np_244680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'np', False)
    # Obtaining the member 'linalg' of a type (line 116)
    linalg_244681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 17), np_244680, 'linalg')
    # Obtaining the member 'matrix_rank' of a type (line 116)
    matrix_rank_244682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 17), linalg_244681, 'matrix_rank')
    # Calling matrix_rank(args, kwargs) (line 116)
    matrix_rank_call_result_244685 = invoke(stypy.reporting.localization.Localization(__file__, 116, 17), matrix_rank_244682, *[A1_244683], **kwargs_244684)
    
    # Getting the type of 'n' (line 116)
    n_244686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 44), 'n', False)
    # Processing the call keyword arguments (line 116)
    kwargs_244687 = {}
    # Getting the type of 'assert_equal' (line 116)
    assert_equal_244679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 116)
    assert_equal_call_result_244688 = invoke(stypy.reporting.localization.Localization(__file__, 116, 4), assert_equal_244679, *[matrix_rank_call_result_244685, n_244686], **kwargs_244687)
    
    
    # ################# End of 'test_m_gt_n(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_m_gt_n' in the type store
    # Getting the type of 'stypy_return_type' (line 107)
    stypy_return_type_244689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_244689)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_m_gt_n'
    return stypy_return_type_244689

# Assigning a type to the variable 'test_m_gt_n' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'test_m_gt_n', test_m_gt_n)

@norecursion
def test_m_gt_n_rank_deficient(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_m_gt_n_rank_deficient'
    module_type_store = module_type_store.open_function_context('test_m_gt_n_rank_deficient', 119, 0, False)
    
    # Passed parameters checking function
    test_m_gt_n_rank_deficient.stypy_localization = localization
    test_m_gt_n_rank_deficient.stypy_type_of_self = None
    test_m_gt_n_rank_deficient.stypy_type_store = module_type_store
    test_m_gt_n_rank_deficient.stypy_function_name = 'test_m_gt_n_rank_deficient'
    test_m_gt_n_rank_deficient.stypy_param_names_list = []
    test_m_gt_n_rank_deficient.stypy_varargs_param_name = None
    test_m_gt_n_rank_deficient.stypy_kwargs_param_name = None
    test_m_gt_n_rank_deficient.stypy_call_defaults = defaults
    test_m_gt_n_rank_deficient.stypy_call_varargs = varargs
    test_m_gt_n_rank_deficient.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_m_gt_n_rank_deficient', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_m_gt_n_rank_deficient', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_m_gt_n_rank_deficient(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 120):
    
    # Assigning a Num to a Name (line 120):
    int_244690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_244049' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'tuple_assignment_244049', int_244690)
    
    # Assigning a Num to a Name (line 120):
    int_244691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'int')
    # Assigning a type to the variable 'tuple_assignment_244050' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'tuple_assignment_244050', int_244691)
    
    # Assigning a Name to a Name (line 120):
    # Getting the type of 'tuple_assignment_244049' (line 120)
    tuple_assignment_244049_244692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'tuple_assignment_244049')
    # Assigning a type to the variable 'm' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'm', tuple_assignment_244049_244692)
    
    # Assigning a Name to a Name (line 120):
    # Getting the type of 'tuple_assignment_244050' (line 120)
    tuple_assignment_244050_244693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'tuple_assignment_244050')
    # Assigning a type to the variable 'n' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 7), 'n', tuple_assignment_244050_244693)
    
    # Assigning a Call to a Name (line 121):
    
    # Assigning a Call to a Name (line 121):
    
    # Call to zeros(...): (line 121)
    # Processing the call arguments (line 121)
    
    # Obtaining an instance of the builtin type 'tuple' (line 121)
    tuple_244696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 121)
    # Adding element type (line 121)
    # Getting the type of 'm' (line 121)
    m_244697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), tuple_244696, m_244697)
    # Adding element type (line 121)
    # Getting the type of 'n' (line 121)
    n_244698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 22), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), tuple_244696, n_244698)
    
    # Processing the call keyword arguments (line 121)
    kwargs_244699 = {}
    # Getting the type of 'np' (line 121)
    np_244694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 9), 'np', False)
    # Obtaining the member 'zeros' of a type (line 121)
    zeros_244695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 9), np_244694, 'zeros')
    # Calling zeros(args, kwargs) (line 121)
    zeros_call_result_244700 = invoke(stypy.reporting.localization.Localization(__file__, 121, 9), zeros_244695, *[tuple_244696], **kwargs_244699)
    
    # Assigning a type to the variable 'A0' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'A0', zeros_call_result_244700)
    
    # Assigning a Num to a Subscript (line 122):
    
    # Assigning a Num to a Subscript (line 122):
    int_244701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 15), 'int')
    # Getting the type of 'A0' (line 122)
    A0_244702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'A0')
    slice_244703 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 122, 4), None, None, None)
    int_244704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 10), 'int')
    # Storing an element on a container (line 122)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 4), A0_244702, ((slice_244703, int_244704), int_244701))
    
    # Assigning a Call to a Name (line 123):
    
    # Assigning a Call to a Name (line 123):
    
    # Call to ones(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'm' (line 123)
    m_244707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'm', False)
    # Processing the call keyword arguments (line 123)
    kwargs_244708 = {}
    # Getting the type of 'np' (line 123)
    np_244705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 9), 'np', False)
    # Obtaining the member 'ones' of a type (line 123)
    ones_244706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 9), np_244705, 'ones')
    # Calling ones(args, kwargs) (line 123)
    ones_call_result_244709 = invoke(stypy.reporting.localization.Localization(__file__, 123, 9), ones_244706, *[m_244707], **kwargs_244708)
    
    # Assigning a type to the variable 'b0' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'b0', ones_call_result_244709)
    
    # Assigning a Call to a Tuple (line 124):
    
    # Assigning a Subscript to a Name (line 124):
    
    # Obtaining the type of the subscript
    int_244710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'A0' (line 124)
    A0_244712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 49), 'A0', False)
    # Getting the type of 'b0' (line 124)
    b0_244713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 53), 'b0', False)
    # Processing the call keyword arguments (line 124)
    kwargs_244714 = {}
    # Getting the type of '_remove_redundancy' (line 124)
    _remove_redundancy_244711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 124)
    _remove_redundancy_call_result_244715 = invoke(stypy.reporting.localization.Localization(__file__, 124, 30), _remove_redundancy_244711, *[A0_244712, b0_244713], **kwargs_244714)
    
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___244716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), _remove_redundancy_call_result_244715, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_244717 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), getitem___244716, int_244710)
    
    # Assigning a type to the variable 'tuple_var_assignment_244051' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'tuple_var_assignment_244051', subscript_call_result_244717)
    
    # Assigning a Subscript to a Name (line 124):
    
    # Obtaining the type of the subscript
    int_244718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'A0' (line 124)
    A0_244720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 49), 'A0', False)
    # Getting the type of 'b0' (line 124)
    b0_244721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 53), 'b0', False)
    # Processing the call keyword arguments (line 124)
    kwargs_244722 = {}
    # Getting the type of '_remove_redundancy' (line 124)
    _remove_redundancy_244719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 124)
    _remove_redundancy_call_result_244723 = invoke(stypy.reporting.localization.Localization(__file__, 124, 30), _remove_redundancy_244719, *[A0_244720, b0_244721], **kwargs_244722)
    
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___244724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), _remove_redundancy_call_result_244723, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_244725 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), getitem___244724, int_244718)
    
    # Assigning a type to the variable 'tuple_var_assignment_244052' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'tuple_var_assignment_244052', subscript_call_result_244725)
    
    # Assigning a Subscript to a Name (line 124):
    
    # Obtaining the type of the subscript
    int_244726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'A0' (line 124)
    A0_244728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 49), 'A0', False)
    # Getting the type of 'b0' (line 124)
    b0_244729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 53), 'b0', False)
    # Processing the call keyword arguments (line 124)
    kwargs_244730 = {}
    # Getting the type of '_remove_redundancy' (line 124)
    _remove_redundancy_244727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 124)
    _remove_redundancy_call_result_244731 = invoke(stypy.reporting.localization.Localization(__file__, 124, 30), _remove_redundancy_244727, *[A0_244728, b0_244729], **kwargs_244730)
    
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___244732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), _remove_redundancy_call_result_244731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_244733 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), getitem___244732, int_244726)
    
    # Assigning a type to the variable 'tuple_var_assignment_244053' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'tuple_var_assignment_244053', subscript_call_result_244733)
    
    # Assigning a Subscript to a Name (line 124):
    
    # Obtaining the type of the subscript
    int_244734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'A0' (line 124)
    A0_244736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 49), 'A0', False)
    # Getting the type of 'b0' (line 124)
    b0_244737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 53), 'b0', False)
    # Processing the call keyword arguments (line 124)
    kwargs_244738 = {}
    # Getting the type of '_remove_redundancy' (line 124)
    _remove_redundancy_244735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 124)
    _remove_redundancy_call_result_244739 = invoke(stypy.reporting.localization.Localization(__file__, 124, 30), _remove_redundancy_244735, *[A0_244736, b0_244737], **kwargs_244738)
    
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___244740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), _remove_redundancy_call_result_244739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_244741 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), getitem___244740, int_244734)
    
    # Assigning a type to the variable 'tuple_var_assignment_244054' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'tuple_var_assignment_244054', subscript_call_result_244741)
    
    # Assigning a Name to a Name (line 124):
    # Getting the type of 'tuple_var_assignment_244051' (line 124)
    tuple_var_assignment_244051_244742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'tuple_var_assignment_244051')
    # Assigning a type to the variable 'A1' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'A1', tuple_var_assignment_244051_244742)
    
    # Assigning a Name to a Name (line 124):
    # Getting the type of 'tuple_var_assignment_244052' (line 124)
    tuple_var_assignment_244052_244743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'tuple_var_assignment_244052')
    # Assigning a type to the variable 'b1' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'b1', tuple_var_assignment_244052_244743)
    
    # Assigning a Name to a Name (line 124):
    # Getting the type of 'tuple_var_assignment_244053' (line 124)
    tuple_var_assignment_244053_244744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'tuple_var_assignment_244053')
    # Assigning a type to the variable 'status' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'status', tuple_var_assignment_244053_244744)
    
    # Assigning a Name to a Name (line 124):
    # Getting the type of 'tuple_var_assignment_244054' (line 124)
    tuple_var_assignment_244054_244745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'tuple_var_assignment_244054')
    # Assigning a type to the variable 'message' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'message', tuple_var_assignment_244054_244745)
    
    # Call to assert_equal(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'status' (line 125)
    status_244747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'status', False)
    int_244748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 25), 'int')
    # Processing the call keyword arguments (line 125)
    kwargs_244749 = {}
    # Getting the type of 'assert_equal' (line 125)
    assert_equal_244746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 125)
    assert_equal_call_result_244750 = invoke(stypy.reporting.localization.Localization(__file__, 125, 4), assert_equal_244746, *[status_244747, int_244748], **kwargs_244749)
    
    
    # Call to assert_allclose(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'A1' (line 126)
    A1_244752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'A1', False)
    
    # Obtaining the type of the subscript
    int_244753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 27), 'int')
    int_244754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 29), 'int')
    slice_244755 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 126, 24), int_244753, int_244754, None)
    slice_244756 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 126, 24), None, None, None)
    # Getting the type of 'A0' (line 126)
    A0_244757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'A0', False)
    # Obtaining the member '__getitem__' of a type (line 126)
    getitem___244758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 24), A0_244757, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 126)
    subscript_call_result_244759 = invoke(stypy.reporting.localization.Localization(__file__, 126, 24), getitem___244758, (slice_244755, slice_244756))
    
    # Processing the call keyword arguments (line 126)
    kwargs_244760 = {}
    # Getting the type of 'assert_allclose' (line 126)
    assert_allclose_244751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 126)
    assert_allclose_call_result_244761 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), assert_allclose_244751, *[A1_244752, subscript_call_result_244759], **kwargs_244760)
    
    
    # Call to assert_allclose(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'b1' (line 127)
    b1_244763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'b1', False)
    
    # Obtaining the type of the subscript
    int_244764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 27), 'int')
    # Getting the type of 'b0' (line 127)
    b0_244765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'b0', False)
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___244766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 24), b0_244765, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_244767 = invoke(stypy.reporting.localization.Localization(__file__, 127, 24), getitem___244766, int_244764)
    
    # Processing the call keyword arguments (line 127)
    kwargs_244768 = {}
    # Getting the type of 'assert_allclose' (line 127)
    assert_allclose_244762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 127)
    assert_allclose_call_result_244769 = invoke(stypy.reporting.localization.Localization(__file__, 127, 4), assert_allclose_244762, *[b1_244763, subscript_call_result_244767], **kwargs_244768)
    
    
    # ################# End of 'test_m_gt_n_rank_deficient(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_m_gt_n_rank_deficient' in the type store
    # Getting the type of 'stypy_return_type' (line 119)
    stypy_return_type_244770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_244770)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_m_gt_n_rank_deficient'
    return stypy_return_type_244770

# Assigning a type to the variable 'test_m_gt_n_rank_deficient' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'test_m_gt_n_rank_deficient', test_m_gt_n_rank_deficient)

@norecursion
def test_m_lt_n_rank_deficient(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_m_lt_n_rank_deficient'
    module_type_store = module_type_store.open_function_context('test_m_lt_n_rank_deficient', 130, 0, False)
    
    # Passed parameters checking function
    test_m_lt_n_rank_deficient.stypy_localization = localization
    test_m_lt_n_rank_deficient.stypy_type_of_self = None
    test_m_lt_n_rank_deficient.stypy_type_store = module_type_store
    test_m_lt_n_rank_deficient.stypy_function_name = 'test_m_lt_n_rank_deficient'
    test_m_lt_n_rank_deficient.stypy_param_names_list = []
    test_m_lt_n_rank_deficient.stypy_varargs_param_name = None
    test_m_lt_n_rank_deficient.stypy_kwargs_param_name = None
    test_m_lt_n_rank_deficient.stypy_call_defaults = defaults
    test_m_lt_n_rank_deficient.stypy_call_varargs = varargs
    test_m_lt_n_rank_deficient.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_m_lt_n_rank_deficient', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_m_lt_n_rank_deficient', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_m_lt_n_rank_deficient(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 131):
    
    # Assigning a Num to a Name (line 131):
    int_244771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_244055' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_assignment_244055', int_244771)
    
    # Assigning a Num to a Name (line 131):
    int_244772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 14), 'int')
    # Assigning a type to the variable 'tuple_assignment_244056' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_assignment_244056', int_244772)
    
    # Assigning a Name to a Name (line 131):
    # Getting the type of 'tuple_assignment_244055' (line 131)
    tuple_assignment_244055_244773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_assignment_244055')
    # Assigning a type to the variable 'm' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'm', tuple_assignment_244055_244773)
    
    # Assigning a Name to a Name (line 131):
    # Getting the type of 'tuple_assignment_244056' (line 131)
    tuple_assignment_244056_244774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_assignment_244056')
    # Assigning a type to the variable 'n' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 7), 'n', tuple_assignment_244056_244774)
    
    # Assigning a Call to a Name (line 132):
    
    # Assigning a Call to a Name (line 132):
    
    # Call to rand(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'm' (line 132)
    m_244778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'm', False)
    # Getting the type of 'n' (line 132)
    n_244779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'n', False)
    # Processing the call keyword arguments (line 132)
    kwargs_244780 = {}
    # Getting the type of 'np' (line 132)
    np_244775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 132)
    random_244776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 9), np_244775, 'random')
    # Obtaining the member 'rand' of a type (line 132)
    rand_244777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 9), random_244776, 'rand')
    # Calling rand(args, kwargs) (line 132)
    rand_call_result_244781 = invoke(stypy.reporting.localization.Localization(__file__, 132, 9), rand_244777, *[m_244778, n_244779], **kwargs_244780)
    
    # Assigning a type to the variable 'A0' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'A0', rand_call_result_244781)
    
    # Assigning a Call to a Name (line 133):
    
    # Assigning a Call to a Name (line 133):
    
    # Call to rand(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'm' (line 133)
    m_244785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'm', False)
    # Processing the call keyword arguments (line 133)
    kwargs_244786 = {}
    # Getting the type of 'np' (line 133)
    np_244782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 133)
    random_244783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 9), np_244782, 'random')
    # Obtaining the member 'rand' of a type (line 133)
    rand_244784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 9), random_244783, 'rand')
    # Calling rand(args, kwargs) (line 133)
    rand_call_result_244787 = invoke(stypy.reporting.localization.Localization(__file__, 133, 9), rand_244784, *[m_244785], **kwargs_244786)
    
    # Assigning a type to the variable 'b0' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'b0', rand_call_result_244787)
    
    # Assigning a Call to a Subscript (line 134):
    
    # Assigning a Call to a Subscript (line 134):
    
    # Call to dot(...): (line 134)
    # Processing the call arguments (line 134)
    
    # Obtaining the type of the subscript
    int_244796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 41), 'int')
    slice_244797 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 134, 37), None, int_244796, None)
    # Getting the type of 'A0' (line 134)
    A0_244798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 37), 'A0', False)
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___244799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 37), A0_244798, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_244800 = invoke(stypy.reporting.localization.Localization(__file__, 134, 37), getitem___244799, slice_244797)
    
    # Processing the call keyword arguments (line 134)
    kwargs_244801 = {}
    
    # Call to arange(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'm' (line 134)
    m_244790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 26), 'm', False)
    int_244791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 30), 'int')
    # Applying the binary operator '-' (line 134)
    result_sub_244792 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 26), '-', m_244790, int_244791)
    
    # Processing the call keyword arguments (line 134)
    kwargs_244793 = {}
    # Getting the type of 'np' (line 134)
    np_244788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'np', False)
    # Obtaining the member 'arange' of a type (line 134)
    arange_244789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 16), np_244788, 'arange')
    # Calling arange(args, kwargs) (line 134)
    arange_call_result_244794 = invoke(stypy.reporting.localization.Localization(__file__, 134, 16), arange_244789, *[result_sub_244792], **kwargs_244793)
    
    # Obtaining the member 'dot' of a type (line 134)
    dot_244795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 16), arange_call_result_244794, 'dot')
    # Calling dot(args, kwargs) (line 134)
    dot_call_result_244802 = invoke(stypy.reporting.localization.Localization(__file__, 134, 16), dot_244795, *[subscript_call_result_244800], **kwargs_244801)
    
    # Getting the type of 'A0' (line 134)
    A0_244803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'A0')
    int_244804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 7), 'int')
    slice_244805 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 134, 4), None, None, None)
    # Storing an element on a container (line 134)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 4), A0_244803, ((int_244804, slice_244805), dot_call_result_244802))
    
    # Assigning a Call to a Subscript (line 135):
    
    # Assigning a Call to a Subscript (line 135):
    
    # Call to dot(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Obtaining the type of the subscript
    int_244814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 38), 'int')
    slice_244815 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 135, 34), None, int_244814, None)
    # Getting the type of 'b0' (line 135)
    b0_244816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 34), 'b0', False)
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___244817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 34), b0_244816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_244818 = invoke(stypy.reporting.localization.Localization(__file__, 135, 34), getitem___244817, slice_244815)
    
    # Processing the call keyword arguments (line 135)
    kwargs_244819 = {}
    
    # Call to arange(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'm' (line 135)
    m_244808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'm', False)
    int_244809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 27), 'int')
    # Applying the binary operator '-' (line 135)
    result_sub_244810 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 23), '-', m_244808, int_244809)
    
    # Processing the call keyword arguments (line 135)
    kwargs_244811 = {}
    # Getting the type of 'np' (line 135)
    np_244806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 13), 'np', False)
    # Obtaining the member 'arange' of a type (line 135)
    arange_244807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 13), np_244806, 'arange')
    # Calling arange(args, kwargs) (line 135)
    arange_call_result_244812 = invoke(stypy.reporting.localization.Localization(__file__, 135, 13), arange_244807, *[result_sub_244810], **kwargs_244811)
    
    # Obtaining the member 'dot' of a type (line 135)
    dot_244813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 13), arange_call_result_244812, 'dot')
    # Calling dot(args, kwargs) (line 135)
    dot_call_result_244820 = invoke(stypy.reporting.localization.Localization(__file__, 135, 13), dot_244813, *[subscript_call_result_244818], **kwargs_244819)
    
    # Getting the type of 'b0' (line 135)
    b0_244821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'b0')
    int_244822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 7), 'int')
    # Storing an element on a container (line 135)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 4), b0_244821, (int_244822, dot_call_result_244820))
    
    # Assigning a Call to a Tuple (line 136):
    
    # Assigning a Subscript to a Name (line 136):
    
    # Obtaining the type of the subscript
    int_244823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'A0' (line 136)
    A0_244825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 49), 'A0', False)
    # Getting the type of 'b0' (line 136)
    b0_244826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 53), 'b0', False)
    # Processing the call keyword arguments (line 136)
    kwargs_244827 = {}
    # Getting the type of '_remove_redundancy' (line 136)
    _remove_redundancy_244824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 136)
    _remove_redundancy_call_result_244828 = invoke(stypy.reporting.localization.Localization(__file__, 136, 30), _remove_redundancy_244824, *[A0_244825, b0_244826], **kwargs_244827)
    
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___244829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 4), _remove_redundancy_call_result_244828, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_244830 = invoke(stypy.reporting.localization.Localization(__file__, 136, 4), getitem___244829, int_244823)
    
    # Assigning a type to the variable 'tuple_var_assignment_244057' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_var_assignment_244057', subscript_call_result_244830)
    
    # Assigning a Subscript to a Name (line 136):
    
    # Obtaining the type of the subscript
    int_244831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'A0' (line 136)
    A0_244833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 49), 'A0', False)
    # Getting the type of 'b0' (line 136)
    b0_244834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 53), 'b0', False)
    # Processing the call keyword arguments (line 136)
    kwargs_244835 = {}
    # Getting the type of '_remove_redundancy' (line 136)
    _remove_redundancy_244832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 136)
    _remove_redundancy_call_result_244836 = invoke(stypy.reporting.localization.Localization(__file__, 136, 30), _remove_redundancy_244832, *[A0_244833, b0_244834], **kwargs_244835)
    
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___244837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 4), _remove_redundancy_call_result_244836, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_244838 = invoke(stypy.reporting.localization.Localization(__file__, 136, 4), getitem___244837, int_244831)
    
    # Assigning a type to the variable 'tuple_var_assignment_244058' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_var_assignment_244058', subscript_call_result_244838)
    
    # Assigning a Subscript to a Name (line 136):
    
    # Obtaining the type of the subscript
    int_244839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'A0' (line 136)
    A0_244841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 49), 'A0', False)
    # Getting the type of 'b0' (line 136)
    b0_244842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 53), 'b0', False)
    # Processing the call keyword arguments (line 136)
    kwargs_244843 = {}
    # Getting the type of '_remove_redundancy' (line 136)
    _remove_redundancy_244840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 136)
    _remove_redundancy_call_result_244844 = invoke(stypy.reporting.localization.Localization(__file__, 136, 30), _remove_redundancy_244840, *[A0_244841, b0_244842], **kwargs_244843)
    
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___244845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 4), _remove_redundancy_call_result_244844, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_244846 = invoke(stypy.reporting.localization.Localization(__file__, 136, 4), getitem___244845, int_244839)
    
    # Assigning a type to the variable 'tuple_var_assignment_244059' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_var_assignment_244059', subscript_call_result_244846)
    
    # Assigning a Subscript to a Name (line 136):
    
    # Obtaining the type of the subscript
    int_244847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'A0' (line 136)
    A0_244849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 49), 'A0', False)
    # Getting the type of 'b0' (line 136)
    b0_244850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 53), 'b0', False)
    # Processing the call keyword arguments (line 136)
    kwargs_244851 = {}
    # Getting the type of '_remove_redundancy' (line 136)
    _remove_redundancy_244848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 136)
    _remove_redundancy_call_result_244852 = invoke(stypy.reporting.localization.Localization(__file__, 136, 30), _remove_redundancy_244848, *[A0_244849, b0_244850], **kwargs_244851)
    
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___244853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 4), _remove_redundancy_call_result_244852, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_244854 = invoke(stypy.reporting.localization.Localization(__file__, 136, 4), getitem___244853, int_244847)
    
    # Assigning a type to the variable 'tuple_var_assignment_244060' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_var_assignment_244060', subscript_call_result_244854)
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'tuple_var_assignment_244057' (line 136)
    tuple_var_assignment_244057_244855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_var_assignment_244057')
    # Assigning a type to the variable 'A1' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'A1', tuple_var_assignment_244057_244855)
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'tuple_var_assignment_244058' (line 136)
    tuple_var_assignment_244058_244856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_var_assignment_244058')
    # Assigning a type to the variable 'b1' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'b1', tuple_var_assignment_244058_244856)
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'tuple_var_assignment_244059' (line 136)
    tuple_var_assignment_244059_244857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_var_assignment_244059')
    # Assigning a type to the variable 'status' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'status', tuple_var_assignment_244059_244857)
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'tuple_var_assignment_244060' (line 136)
    tuple_var_assignment_244060_244858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_var_assignment_244060')
    # Assigning a type to the variable 'message' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'message', tuple_var_assignment_244060_244858)
    
    # Call to assert_equal(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'status' (line 137)
    status_244860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 17), 'status', False)
    int_244861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 25), 'int')
    # Processing the call keyword arguments (line 137)
    kwargs_244862 = {}
    # Getting the type of 'assert_equal' (line 137)
    assert_equal_244859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 137)
    assert_equal_call_result_244863 = invoke(stypy.reporting.localization.Localization(__file__, 137, 4), assert_equal_244859, *[status_244860, int_244861], **kwargs_244862)
    
    
    # Call to assert_equal(...): (line 138)
    # Processing the call arguments (line 138)
    
    # Obtaining the type of the subscript
    int_244865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 26), 'int')
    # Getting the type of 'A1' (line 138)
    A1_244866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 17), 'A1', False)
    # Obtaining the member 'shape' of a type (line 138)
    shape_244867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 17), A1_244866, 'shape')
    # Obtaining the member '__getitem__' of a type (line 138)
    getitem___244868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 17), shape_244867, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 138)
    subscript_call_result_244869 = invoke(stypy.reporting.localization.Localization(__file__, 138, 17), getitem___244868, int_244865)
    
    int_244870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 30), 'int')
    # Processing the call keyword arguments (line 138)
    kwargs_244871 = {}
    # Getting the type of 'assert_equal' (line 138)
    assert_equal_244864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 138)
    assert_equal_call_result_244872 = invoke(stypy.reporting.localization.Localization(__file__, 138, 4), assert_equal_244864, *[subscript_call_result_244869, int_244870], **kwargs_244871)
    
    
    # Call to assert_equal(...): (line 139)
    # Processing the call arguments (line 139)
    
    # Call to matrix_rank(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'A1' (line 139)
    A1_244877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 39), 'A1', False)
    # Processing the call keyword arguments (line 139)
    kwargs_244878 = {}
    # Getting the type of 'np' (line 139)
    np_244874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 17), 'np', False)
    # Obtaining the member 'linalg' of a type (line 139)
    linalg_244875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 17), np_244874, 'linalg')
    # Obtaining the member 'matrix_rank' of a type (line 139)
    matrix_rank_244876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 17), linalg_244875, 'matrix_rank')
    # Calling matrix_rank(args, kwargs) (line 139)
    matrix_rank_call_result_244879 = invoke(stypy.reporting.localization.Localization(__file__, 139, 17), matrix_rank_244876, *[A1_244877], **kwargs_244878)
    
    int_244880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 44), 'int')
    # Processing the call keyword arguments (line 139)
    kwargs_244881 = {}
    # Getting the type of 'assert_equal' (line 139)
    assert_equal_244873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 139)
    assert_equal_call_result_244882 = invoke(stypy.reporting.localization.Localization(__file__, 139, 4), assert_equal_244873, *[matrix_rank_call_result_244879, int_244880], **kwargs_244881)
    
    
    # ################# End of 'test_m_lt_n_rank_deficient(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_m_lt_n_rank_deficient' in the type store
    # Getting the type of 'stypy_return_type' (line 130)
    stypy_return_type_244883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_244883)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_m_lt_n_rank_deficient'
    return stypy_return_type_244883

# Assigning a type to the variable 'test_m_lt_n_rank_deficient' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'test_m_lt_n_rank_deficient', test_m_lt_n_rank_deficient)

@norecursion
def test_dense1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_dense1'
    module_type_store = module_type_store.open_function_context('test_dense1', 142, 0, False)
    
    # Passed parameters checking function
    test_dense1.stypy_localization = localization
    test_dense1.stypy_type_of_self = None
    test_dense1.stypy_type_store = module_type_store
    test_dense1.stypy_function_name = 'test_dense1'
    test_dense1.stypy_param_names_list = []
    test_dense1.stypy_varargs_param_name = None
    test_dense1.stypy_kwargs_param_name = None
    test_dense1.stypy_call_defaults = defaults
    test_dense1.stypy_call_varargs = varargs
    test_dense1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_dense1', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_dense1', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_dense1(...)' code ##################

    
    # Assigning a Call to a Name (line 143):
    
    # Assigning a Call to a Name (line 143):
    
    # Call to ones(...): (line 143)
    # Processing the call arguments (line 143)
    
    # Obtaining an instance of the builtin type 'tuple' (line 143)
    tuple_244886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 143)
    # Adding element type (line 143)
    int_244887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 17), tuple_244886, int_244887)
    # Adding element type (line 143)
    int_244888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 17), tuple_244886, int_244888)
    
    # Processing the call keyword arguments (line 143)
    kwargs_244889 = {}
    # Getting the type of 'np' (line 143)
    np_244884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 143)
    ones_244885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), np_244884, 'ones')
    # Calling ones(args, kwargs) (line 143)
    ones_call_result_244890 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), ones_244885, *[tuple_244886], **kwargs_244889)
    
    # Assigning a type to the variable 'A' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'A', ones_call_result_244890)
    
    # Assigning a Num to a Subscript (line 144):
    
    # Assigning a Num to a Subscript (line 144):
    int_244891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 15), 'int')
    # Getting the type of 'A' (line 144)
    A_244892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'A')
    int_244893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 6), 'int')
    int_244894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 10), 'int')
    slice_244895 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 144, 4), None, int_244894, None)
    # Storing an element on a container (line 144)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 4), A_244892, ((int_244893, slice_244895), int_244891))
    
    # Assigning a Num to a Subscript (line 145):
    
    # Assigning a Num to a Subscript (line 145):
    int_244896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 15), 'int')
    # Getting the type of 'A' (line 145)
    A_244897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'A')
    int_244898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 6), 'int')
    int_244899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 9), 'int')
    slice_244900 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 145, 4), int_244899, None, None)
    # Storing an element on a container (line 145)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 4), A_244897, ((int_244898, slice_244900), int_244896))
    
    # Assigning a Num to a Subscript (line 146):
    
    # Assigning a Num to a Subscript (line 146):
    int_244901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 17), 'int')
    # Getting the type of 'A' (line 146)
    A_244902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'A')
    int_244903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 6), 'int')
    slice_244904 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 146, 4), int_244903, None, None)
    int_244905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 12), 'int')
    slice_244906 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 146, 4), None, None, int_244905)
    # Storing an element on a container (line 146)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 4), A_244902, ((slice_244904, slice_244906), int_244901))
    
    # Assigning a Num to a Subscript (line 147):
    
    # Assigning a Num to a Subscript (line 147):
    int_244907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 15), 'int')
    # Getting the type of 'A' (line 147)
    A_244908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'A')
    int_244909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 6), 'int')
    int_244910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 10), 'int')
    slice_244911 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 147, 4), None, int_244910, None)
    # Storing an element on a container (line 147)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 4), A_244908, ((int_244909, slice_244911), int_244907))
    
    # Assigning a Num to a Subscript (line 148):
    
    # Assigning a Num to a Subscript (line 148):
    int_244912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 15), 'int')
    # Getting the type of 'A' (line 148)
    A_244913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'A')
    int_244914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 6), 'int')
    int_244915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 9), 'int')
    slice_244916 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 148, 4), int_244915, None, None)
    # Storing an element on a container (line 148)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 4), A_244913, ((int_244914, slice_244916), int_244912))
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to zeros(...): (line 149)
    # Processing the call arguments (line 149)
    
    # Obtaining the type of the subscript
    int_244919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 25), 'int')
    # Getting the type of 'A' (line 149)
    A_244920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'A', False)
    # Obtaining the member 'shape' of a type (line 149)
    shape_244921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 17), A_244920, 'shape')
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___244922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 17), shape_244921, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_244923 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), getitem___244922, int_244919)
    
    # Processing the call keyword arguments (line 149)
    kwargs_244924 = {}
    # Getting the type of 'np' (line 149)
    np_244917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 149)
    zeros_244918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), np_244917, 'zeros')
    # Calling zeros(args, kwargs) (line 149)
    zeros_call_result_244925 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), zeros_244918, *[subscript_call_result_244923], **kwargs_244924)
    
    # Assigning a type to the variable 'b' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'b', zeros_call_result_244925)
    
    # Assigning a Subscript to a Name (line 151):
    
    # Assigning a Subscript to a Name (line 151):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'list' (line 151)
    list_244926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 151)
    # Adding element type (line 151)
    int_244927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 11), list_244926, int_244927)
    # Adding element type (line 151)
    int_244928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 11), list_244926, int_244928)
    # Adding element type (line 151)
    int_244929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 11), list_244926, int_244929)
    # Adding element type (line 151)
    int_244930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 11), list_244926, int_244930)
    
    slice_244931 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 151, 9), None, None, None)
    # Getting the type of 'A' (line 151)
    A_244932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 9), 'A')
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___244933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 9), A_244932, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_244934 = invoke(stypy.reporting.localization.Localization(__file__, 151, 9), getitem___244933, (list_244926, slice_244931))
    
    # Assigning a type to the variable 'A2' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'A2', subscript_call_result_244934)
    
    # Assigning a Call to a Name (line 152):
    
    # Assigning a Call to a Name (line 152):
    
    # Call to zeros(...): (line 152)
    # Processing the call arguments (line 152)
    int_244937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 18), 'int')
    # Processing the call keyword arguments (line 152)
    kwargs_244938 = {}
    # Getting the type of 'np' (line 152)
    np_244935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 9), 'np', False)
    # Obtaining the member 'zeros' of a type (line 152)
    zeros_244936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 9), np_244935, 'zeros')
    # Calling zeros(args, kwargs) (line 152)
    zeros_call_result_244939 = invoke(stypy.reporting.localization.Localization(__file__, 152, 9), zeros_244936, *[int_244937], **kwargs_244938)
    
    # Assigning a type to the variable 'b2' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'b2', zeros_call_result_244939)
    
    # Assigning a Call to a Tuple (line 154):
    
    # Assigning a Subscript to a Name (line 154):
    
    # Obtaining the type of the subscript
    int_244940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'A' (line 154)
    A_244942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 49), 'A', False)
    # Getting the type of 'b' (line 154)
    b_244943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 52), 'b', False)
    # Processing the call keyword arguments (line 154)
    kwargs_244944 = {}
    # Getting the type of '_remove_redundancy' (line 154)
    _remove_redundancy_244941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 154)
    _remove_redundancy_call_result_244945 = invoke(stypy.reporting.localization.Localization(__file__, 154, 30), _remove_redundancy_244941, *[A_244942, b_244943], **kwargs_244944)
    
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___244946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 4), _remove_redundancy_call_result_244945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_244947 = invoke(stypy.reporting.localization.Localization(__file__, 154, 4), getitem___244946, int_244940)
    
    # Assigning a type to the variable 'tuple_var_assignment_244061' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'tuple_var_assignment_244061', subscript_call_result_244947)
    
    # Assigning a Subscript to a Name (line 154):
    
    # Obtaining the type of the subscript
    int_244948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'A' (line 154)
    A_244950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 49), 'A', False)
    # Getting the type of 'b' (line 154)
    b_244951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 52), 'b', False)
    # Processing the call keyword arguments (line 154)
    kwargs_244952 = {}
    # Getting the type of '_remove_redundancy' (line 154)
    _remove_redundancy_244949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 154)
    _remove_redundancy_call_result_244953 = invoke(stypy.reporting.localization.Localization(__file__, 154, 30), _remove_redundancy_244949, *[A_244950, b_244951], **kwargs_244952)
    
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___244954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 4), _remove_redundancy_call_result_244953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_244955 = invoke(stypy.reporting.localization.Localization(__file__, 154, 4), getitem___244954, int_244948)
    
    # Assigning a type to the variable 'tuple_var_assignment_244062' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'tuple_var_assignment_244062', subscript_call_result_244955)
    
    # Assigning a Subscript to a Name (line 154):
    
    # Obtaining the type of the subscript
    int_244956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'A' (line 154)
    A_244958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 49), 'A', False)
    # Getting the type of 'b' (line 154)
    b_244959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 52), 'b', False)
    # Processing the call keyword arguments (line 154)
    kwargs_244960 = {}
    # Getting the type of '_remove_redundancy' (line 154)
    _remove_redundancy_244957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 154)
    _remove_redundancy_call_result_244961 = invoke(stypy.reporting.localization.Localization(__file__, 154, 30), _remove_redundancy_244957, *[A_244958, b_244959], **kwargs_244960)
    
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___244962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 4), _remove_redundancy_call_result_244961, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_244963 = invoke(stypy.reporting.localization.Localization(__file__, 154, 4), getitem___244962, int_244956)
    
    # Assigning a type to the variable 'tuple_var_assignment_244063' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'tuple_var_assignment_244063', subscript_call_result_244963)
    
    # Assigning a Subscript to a Name (line 154):
    
    # Obtaining the type of the subscript
    int_244964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'A' (line 154)
    A_244966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 49), 'A', False)
    # Getting the type of 'b' (line 154)
    b_244967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 52), 'b', False)
    # Processing the call keyword arguments (line 154)
    kwargs_244968 = {}
    # Getting the type of '_remove_redundancy' (line 154)
    _remove_redundancy_244965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 154)
    _remove_redundancy_call_result_244969 = invoke(stypy.reporting.localization.Localization(__file__, 154, 30), _remove_redundancy_244965, *[A_244966, b_244967], **kwargs_244968)
    
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___244970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 4), _remove_redundancy_call_result_244969, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_244971 = invoke(stypy.reporting.localization.Localization(__file__, 154, 4), getitem___244970, int_244964)
    
    # Assigning a type to the variable 'tuple_var_assignment_244064' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'tuple_var_assignment_244064', subscript_call_result_244971)
    
    # Assigning a Name to a Name (line 154):
    # Getting the type of 'tuple_var_assignment_244061' (line 154)
    tuple_var_assignment_244061_244972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'tuple_var_assignment_244061')
    # Assigning a type to the variable 'A1' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'A1', tuple_var_assignment_244061_244972)
    
    # Assigning a Name to a Name (line 154):
    # Getting the type of 'tuple_var_assignment_244062' (line 154)
    tuple_var_assignment_244062_244973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'tuple_var_assignment_244062')
    # Assigning a type to the variable 'b1' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'b1', tuple_var_assignment_244062_244973)
    
    # Assigning a Name to a Name (line 154):
    # Getting the type of 'tuple_var_assignment_244063' (line 154)
    tuple_var_assignment_244063_244974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'tuple_var_assignment_244063')
    # Assigning a type to the variable 'status' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'status', tuple_var_assignment_244063_244974)
    
    # Assigning a Name to a Name (line 154):
    # Getting the type of 'tuple_var_assignment_244064' (line 154)
    tuple_var_assignment_244064_244975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'tuple_var_assignment_244064')
    # Assigning a type to the variable 'message' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'message', tuple_var_assignment_244064_244975)
    
    # Call to assert_allclose(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'A1' (line 155)
    A1_244977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'A1', False)
    # Getting the type of 'A2' (line 155)
    A2_244978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'A2', False)
    # Processing the call keyword arguments (line 155)
    kwargs_244979 = {}
    # Getting the type of 'assert_allclose' (line 155)
    assert_allclose_244976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 155)
    assert_allclose_call_result_244980 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), assert_allclose_244976, *[A1_244977, A2_244978], **kwargs_244979)
    
    
    # Call to assert_allclose(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'b1' (line 156)
    b1_244982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'b1', False)
    # Getting the type of 'b2' (line 156)
    b2_244983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 24), 'b2', False)
    # Processing the call keyword arguments (line 156)
    kwargs_244984 = {}
    # Getting the type of 'assert_allclose' (line 156)
    assert_allclose_244981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 156)
    assert_allclose_call_result_244985 = invoke(stypy.reporting.localization.Localization(__file__, 156, 4), assert_allclose_244981, *[b1_244982, b2_244983], **kwargs_244984)
    
    
    # Call to assert_equal(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'status' (line 157)
    status_244987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 17), 'status', False)
    int_244988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 25), 'int')
    # Processing the call keyword arguments (line 157)
    kwargs_244989 = {}
    # Getting the type of 'assert_equal' (line 157)
    assert_equal_244986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 157)
    assert_equal_call_result_244990 = invoke(stypy.reporting.localization.Localization(__file__, 157, 4), assert_equal_244986, *[status_244987, int_244988], **kwargs_244989)
    
    
    # ################# End of 'test_dense1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_dense1' in the type store
    # Getting the type of 'stypy_return_type' (line 142)
    stypy_return_type_244991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_244991)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_dense1'
    return stypy_return_type_244991

# Assigning a type to the variable 'test_dense1' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'test_dense1', test_dense1)

@norecursion
def test_dense2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_dense2'
    module_type_store = module_type_store.open_function_context('test_dense2', 160, 0, False)
    
    # Passed parameters checking function
    test_dense2.stypy_localization = localization
    test_dense2.stypy_type_of_self = None
    test_dense2.stypy_type_store = module_type_store
    test_dense2.stypy_function_name = 'test_dense2'
    test_dense2.stypy_param_names_list = []
    test_dense2.stypy_varargs_param_name = None
    test_dense2.stypy_kwargs_param_name = None
    test_dense2.stypy_call_defaults = defaults
    test_dense2.stypy_call_varargs = varargs
    test_dense2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_dense2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_dense2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_dense2(...)' code ##################

    
    # Assigning a Call to a Name (line 161):
    
    # Assigning a Call to a Name (line 161):
    
    # Call to eye(...): (line 161)
    # Processing the call arguments (line 161)
    int_244994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 15), 'int')
    # Processing the call keyword arguments (line 161)
    kwargs_244995 = {}
    # Getting the type of 'np' (line 161)
    np_244992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'np', False)
    # Obtaining the member 'eye' of a type (line 161)
    eye_244993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), np_244992, 'eye')
    # Calling eye(args, kwargs) (line 161)
    eye_call_result_244996 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), eye_244993, *[int_244994], **kwargs_244995)
    
    # Assigning a type to the variable 'A' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'A', eye_call_result_244996)
    
    # Assigning a Num to a Subscript (line 162):
    
    # Assigning a Num to a Subscript (line 162):
    int_244997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 16), 'int')
    # Getting the type of 'A' (line 162)
    A_244998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'A')
    
    # Obtaining an instance of the builtin type 'tuple' (line 162)
    tuple_244999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 6), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 162)
    # Adding element type (line 162)
    int_245000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 6), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 6), tuple_244999, int_245000)
    # Adding element type (line 162)
    int_245001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 6), tuple_244999, int_245001)
    
    # Storing an element on a container (line 162)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 4), A_244998, (tuple_244999, int_244997))
    
    # Assigning a Num to a Subscript (line 163):
    
    # Assigning a Num to a Subscript (line 163):
    int_245002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 15), 'int')
    # Getting the type of 'A' (line 163)
    A_245003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'A')
    int_245004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 6), 'int')
    slice_245005 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 163, 4), None, None, None)
    # Storing an element on a container (line 163)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 4), A_245003, ((int_245004, slice_245005), int_245002))
    
    # Assigning a Call to a Name (line 164):
    
    # Assigning a Call to a Name (line 164):
    
    # Call to zeros(...): (line 164)
    # Processing the call arguments (line 164)
    
    # Obtaining the type of the subscript
    int_245008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 25), 'int')
    # Getting the type of 'A' (line 164)
    A_245009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 17), 'A', False)
    # Obtaining the member 'shape' of a type (line 164)
    shape_245010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 17), A_245009, 'shape')
    # Obtaining the member '__getitem__' of a type (line 164)
    getitem___245011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 17), shape_245010, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 164)
    subscript_call_result_245012 = invoke(stypy.reporting.localization.Localization(__file__, 164, 17), getitem___245011, int_245008)
    
    # Processing the call keyword arguments (line 164)
    kwargs_245013 = {}
    # Getting the type of 'np' (line 164)
    np_245006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 164)
    zeros_245007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), np_245006, 'zeros')
    # Calling zeros(args, kwargs) (line 164)
    zeros_call_result_245014 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), zeros_245007, *[subscript_call_result_245012], **kwargs_245013)
    
    # Assigning a type to the variable 'b' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'b', zeros_call_result_245014)
    
    # Assigning a Call to a Tuple (line 165):
    
    # Assigning a Subscript to a Name (line 165):
    
    # Obtaining the type of the subscript
    int_245015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'A' (line 165)
    A_245017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 49), 'A', False)
    # Getting the type of 'b' (line 165)
    b_245018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 52), 'b', False)
    # Processing the call keyword arguments (line 165)
    kwargs_245019 = {}
    # Getting the type of '_remove_redundancy' (line 165)
    _remove_redundancy_245016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 165)
    _remove_redundancy_call_result_245020 = invoke(stypy.reporting.localization.Localization(__file__, 165, 30), _remove_redundancy_245016, *[A_245017, b_245018], **kwargs_245019)
    
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___245021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 4), _remove_redundancy_call_result_245020, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_245022 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), getitem___245021, int_245015)
    
    # Assigning a type to the variable 'tuple_var_assignment_244065' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_244065', subscript_call_result_245022)
    
    # Assigning a Subscript to a Name (line 165):
    
    # Obtaining the type of the subscript
    int_245023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'A' (line 165)
    A_245025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 49), 'A', False)
    # Getting the type of 'b' (line 165)
    b_245026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 52), 'b', False)
    # Processing the call keyword arguments (line 165)
    kwargs_245027 = {}
    # Getting the type of '_remove_redundancy' (line 165)
    _remove_redundancy_245024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 165)
    _remove_redundancy_call_result_245028 = invoke(stypy.reporting.localization.Localization(__file__, 165, 30), _remove_redundancy_245024, *[A_245025, b_245026], **kwargs_245027)
    
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___245029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 4), _remove_redundancy_call_result_245028, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_245030 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), getitem___245029, int_245023)
    
    # Assigning a type to the variable 'tuple_var_assignment_244066' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_244066', subscript_call_result_245030)
    
    # Assigning a Subscript to a Name (line 165):
    
    # Obtaining the type of the subscript
    int_245031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'A' (line 165)
    A_245033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 49), 'A', False)
    # Getting the type of 'b' (line 165)
    b_245034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 52), 'b', False)
    # Processing the call keyword arguments (line 165)
    kwargs_245035 = {}
    # Getting the type of '_remove_redundancy' (line 165)
    _remove_redundancy_245032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 165)
    _remove_redundancy_call_result_245036 = invoke(stypy.reporting.localization.Localization(__file__, 165, 30), _remove_redundancy_245032, *[A_245033, b_245034], **kwargs_245035)
    
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___245037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 4), _remove_redundancy_call_result_245036, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_245038 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), getitem___245037, int_245031)
    
    # Assigning a type to the variable 'tuple_var_assignment_244067' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_244067', subscript_call_result_245038)
    
    # Assigning a Subscript to a Name (line 165):
    
    # Obtaining the type of the subscript
    int_245039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'A' (line 165)
    A_245041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 49), 'A', False)
    # Getting the type of 'b' (line 165)
    b_245042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 52), 'b', False)
    # Processing the call keyword arguments (line 165)
    kwargs_245043 = {}
    # Getting the type of '_remove_redundancy' (line 165)
    _remove_redundancy_245040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 165)
    _remove_redundancy_call_result_245044 = invoke(stypy.reporting.localization.Localization(__file__, 165, 30), _remove_redundancy_245040, *[A_245041, b_245042], **kwargs_245043)
    
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___245045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 4), _remove_redundancy_call_result_245044, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_245046 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), getitem___245045, int_245039)
    
    # Assigning a type to the variable 'tuple_var_assignment_244068' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_244068', subscript_call_result_245046)
    
    # Assigning a Name to a Name (line 165):
    # Getting the type of 'tuple_var_assignment_244065' (line 165)
    tuple_var_assignment_244065_245047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_244065')
    # Assigning a type to the variable 'A1' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'A1', tuple_var_assignment_244065_245047)
    
    # Assigning a Name to a Name (line 165):
    # Getting the type of 'tuple_var_assignment_244066' (line 165)
    tuple_var_assignment_244066_245048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_244066')
    # Assigning a type to the variable 'b1' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'b1', tuple_var_assignment_244066_245048)
    
    # Assigning a Name to a Name (line 165):
    # Getting the type of 'tuple_var_assignment_244067' (line 165)
    tuple_var_assignment_244067_245049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_244067')
    # Assigning a type to the variable 'status' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'status', tuple_var_assignment_244067_245049)
    
    # Assigning a Name to a Name (line 165):
    # Getting the type of 'tuple_var_assignment_244068' (line 165)
    tuple_var_assignment_244068_245050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple_var_assignment_244068')
    # Assigning a type to the variable 'message' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 20), 'message', tuple_var_assignment_244068_245050)
    
    # Call to assert_allclose(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'A1' (line 166)
    A1_245052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'A1', False)
    
    # Obtaining the type of the subscript
    int_245053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 27), 'int')
    slice_245054 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 166, 24), None, int_245053, None)
    slice_245055 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 166, 24), None, None, None)
    # Getting the type of 'A' (line 166)
    A_245056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'A', False)
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___245057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 24), A_245056, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_245058 = invoke(stypy.reporting.localization.Localization(__file__, 166, 24), getitem___245057, (slice_245054, slice_245055))
    
    # Processing the call keyword arguments (line 166)
    kwargs_245059 = {}
    # Getting the type of 'assert_allclose' (line 166)
    assert_allclose_245051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 166)
    assert_allclose_call_result_245060 = invoke(stypy.reporting.localization.Localization(__file__, 166, 4), assert_allclose_245051, *[A1_245052, subscript_call_result_245058], **kwargs_245059)
    
    
    # Call to assert_allclose(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'b1' (line 167)
    b1_245062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'b1', False)
    
    # Obtaining the type of the subscript
    int_245063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 27), 'int')
    slice_245064 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 167, 24), None, int_245063, None)
    # Getting the type of 'b' (line 167)
    b_245065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___245066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 24), b_245065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_245067 = invoke(stypy.reporting.localization.Localization(__file__, 167, 24), getitem___245066, slice_245064)
    
    # Processing the call keyword arguments (line 167)
    kwargs_245068 = {}
    # Getting the type of 'assert_allclose' (line 167)
    assert_allclose_245061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 167)
    assert_allclose_call_result_245069 = invoke(stypy.reporting.localization.Localization(__file__, 167, 4), assert_allclose_245061, *[b1_245062, subscript_call_result_245067], **kwargs_245068)
    
    
    # Call to assert_equal(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'status' (line 168)
    status_245071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 17), 'status', False)
    int_245072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 25), 'int')
    # Processing the call keyword arguments (line 168)
    kwargs_245073 = {}
    # Getting the type of 'assert_equal' (line 168)
    assert_equal_245070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 168)
    assert_equal_call_result_245074 = invoke(stypy.reporting.localization.Localization(__file__, 168, 4), assert_equal_245070, *[status_245071, int_245072], **kwargs_245073)
    
    
    # ################# End of 'test_dense2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_dense2' in the type store
    # Getting the type of 'stypy_return_type' (line 160)
    stypy_return_type_245075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_245075)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_dense2'
    return stypy_return_type_245075

# Assigning a type to the variable 'test_dense2' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'test_dense2', test_dense2)

@norecursion
def test_dense3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_dense3'
    module_type_store = module_type_store.open_function_context('test_dense3', 171, 0, False)
    
    # Passed parameters checking function
    test_dense3.stypy_localization = localization
    test_dense3.stypy_type_of_self = None
    test_dense3.stypy_type_store = module_type_store
    test_dense3.stypy_function_name = 'test_dense3'
    test_dense3.stypy_param_names_list = []
    test_dense3.stypy_varargs_param_name = None
    test_dense3.stypy_kwargs_param_name = None
    test_dense3.stypy_call_defaults = defaults
    test_dense3.stypy_call_varargs = varargs
    test_dense3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_dense3', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_dense3', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_dense3(...)' code ##################

    
    # Assigning a Call to a Name (line 172):
    
    # Assigning a Call to a Name (line 172):
    
    # Call to eye(...): (line 172)
    # Processing the call arguments (line 172)
    int_245078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 15), 'int')
    # Processing the call keyword arguments (line 172)
    kwargs_245079 = {}
    # Getting the type of 'np' (line 172)
    np_245076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'np', False)
    # Obtaining the member 'eye' of a type (line 172)
    eye_245077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), np_245076, 'eye')
    # Calling eye(args, kwargs) (line 172)
    eye_call_result_245080 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), eye_245077, *[int_245078], **kwargs_245079)
    
    # Assigning a type to the variable 'A' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'A', eye_call_result_245080)
    
    # Assigning a Num to a Subscript (line 173):
    
    # Assigning a Num to a Subscript (line 173):
    int_245081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 16), 'int')
    # Getting the type of 'A' (line 173)
    A_245082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'A')
    
    # Obtaining an instance of the builtin type 'tuple' (line 173)
    tuple_245083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 6), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 173)
    # Adding element type (line 173)
    int_245084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 6), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 6), tuple_245083, int_245084)
    # Adding element type (line 173)
    int_245085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 6), tuple_245083, int_245085)
    
    # Storing an element on a container (line 173)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 4), A_245082, (tuple_245083, int_245081))
    
    # Assigning a Num to a Subscript (line 174):
    
    # Assigning a Num to a Subscript (line 174):
    int_245086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 15), 'int')
    # Getting the type of 'A' (line 174)
    A_245087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'A')
    int_245088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 6), 'int')
    slice_245089 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 4), None, None, None)
    # Storing an element on a container (line 174)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 4), A_245087, ((int_245088, slice_245089), int_245086))
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to rand(...): (line 175)
    # Processing the call arguments (line 175)
    
    # Obtaining the type of the subscript
    int_245093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 31), 'int')
    # Getting the type of 'A' (line 175)
    A_245094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'A', False)
    # Obtaining the member 'shape' of a type (line 175)
    shape_245095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 23), A_245094, 'shape')
    # Obtaining the member '__getitem__' of a type (line 175)
    getitem___245096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 23), shape_245095, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 175)
    subscript_call_result_245097 = invoke(stypy.reporting.localization.Localization(__file__, 175, 23), getitem___245096, int_245093)
    
    # Processing the call keyword arguments (line 175)
    kwargs_245098 = {}
    # Getting the type of 'np' (line 175)
    np_245090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 175)
    random_245091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), np_245090, 'random')
    # Obtaining the member 'rand' of a type (line 175)
    rand_245092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), random_245091, 'rand')
    # Calling rand(args, kwargs) (line 175)
    rand_call_result_245099 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), rand_245092, *[subscript_call_result_245097], **kwargs_245098)
    
    # Assigning a type to the variable 'b' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'b', rand_call_result_245099)
    
    # Assigning a Call to a Subscript (line 176):
    
    # Assigning a Call to a Subscript (line 176):
    
    # Call to sum(...): (line 176)
    # Processing the call arguments (line 176)
    
    # Obtaining the type of the subscript
    int_245102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 22), 'int')
    slice_245103 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 176, 19), None, int_245102, None)
    # Getting the type of 'b' (line 176)
    b_245104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___245105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 19), b_245104, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_245106 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), getitem___245105, slice_245103)
    
    # Processing the call keyword arguments (line 176)
    kwargs_245107 = {}
    # Getting the type of 'np' (line 176)
    np_245100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'np', False)
    # Obtaining the member 'sum' of a type (line 176)
    sum_245101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), np_245100, 'sum')
    # Calling sum(args, kwargs) (line 176)
    sum_call_result_245108 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), sum_245101, *[subscript_call_result_245106], **kwargs_245107)
    
    # Getting the type of 'b' (line 176)
    b_245109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'b')
    int_245110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 6), 'int')
    # Storing an element on a container (line 176)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 4), b_245109, (int_245110, sum_call_result_245108))
    
    # Assigning a Call to a Tuple (line 177):
    
    # Assigning a Subscript to a Name (line 177):
    
    # Obtaining the type of the subscript
    int_245111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'A' (line 177)
    A_245113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 49), 'A', False)
    # Getting the type of 'b' (line 177)
    b_245114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 52), 'b', False)
    # Processing the call keyword arguments (line 177)
    kwargs_245115 = {}
    # Getting the type of '_remove_redundancy' (line 177)
    _remove_redundancy_245112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 177)
    _remove_redundancy_call_result_245116 = invoke(stypy.reporting.localization.Localization(__file__, 177, 30), _remove_redundancy_245112, *[A_245113, b_245114], **kwargs_245115)
    
    # Obtaining the member '__getitem__' of a type (line 177)
    getitem___245117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 4), _remove_redundancy_call_result_245116, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 177)
    subscript_call_result_245118 = invoke(stypy.reporting.localization.Localization(__file__, 177, 4), getitem___245117, int_245111)
    
    # Assigning a type to the variable 'tuple_var_assignment_244069' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_var_assignment_244069', subscript_call_result_245118)
    
    # Assigning a Subscript to a Name (line 177):
    
    # Obtaining the type of the subscript
    int_245119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'A' (line 177)
    A_245121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 49), 'A', False)
    # Getting the type of 'b' (line 177)
    b_245122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 52), 'b', False)
    # Processing the call keyword arguments (line 177)
    kwargs_245123 = {}
    # Getting the type of '_remove_redundancy' (line 177)
    _remove_redundancy_245120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 177)
    _remove_redundancy_call_result_245124 = invoke(stypy.reporting.localization.Localization(__file__, 177, 30), _remove_redundancy_245120, *[A_245121, b_245122], **kwargs_245123)
    
    # Obtaining the member '__getitem__' of a type (line 177)
    getitem___245125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 4), _remove_redundancy_call_result_245124, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 177)
    subscript_call_result_245126 = invoke(stypy.reporting.localization.Localization(__file__, 177, 4), getitem___245125, int_245119)
    
    # Assigning a type to the variable 'tuple_var_assignment_244070' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_var_assignment_244070', subscript_call_result_245126)
    
    # Assigning a Subscript to a Name (line 177):
    
    # Obtaining the type of the subscript
    int_245127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'A' (line 177)
    A_245129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 49), 'A', False)
    # Getting the type of 'b' (line 177)
    b_245130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 52), 'b', False)
    # Processing the call keyword arguments (line 177)
    kwargs_245131 = {}
    # Getting the type of '_remove_redundancy' (line 177)
    _remove_redundancy_245128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 177)
    _remove_redundancy_call_result_245132 = invoke(stypy.reporting.localization.Localization(__file__, 177, 30), _remove_redundancy_245128, *[A_245129, b_245130], **kwargs_245131)
    
    # Obtaining the member '__getitem__' of a type (line 177)
    getitem___245133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 4), _remove_redundancy_call_result_245132, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 177)
    subscript_call_result_245134 = invoke(stypy.reporting.localization.Localization(__file__, 177, 4), getitem___245133, int_245127)
    
    # Assigning a type to the variable 'tuple_var_assignment_244071' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_var_assignment_244071', subscript_call_result_245134)
    
    # Assigning a Subscript to a Name (line 177):
    
    # Obtaining the type of the subscript
    int_245135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'A' (line 177)
    A_245137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 49), 'A', False)
    # Getting the type of 'b' (line 177)
    b_245138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 52), 'b', False)
    # Processing the call keyword arguments (line 177)
    kwargs_245139 = {}
    # Getting the type of '_remove_redundancy' (line 177)
    _remove_redundancy_245136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 177)
    _remove_redundancy_call_result_245140 = invoke(stypy.reporting.localization.Localization(__file__, 177, 30), _remove_redundancy_245136, *[A_245137, b_245138], **kwargs_245139)
    
    # Obtaining the member '__getitem__' of a type (line 177)
    getitem___245141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 4), _remove_redundancy_call_result_245140, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 177)
    subscript_call_result_245142 = invoke(stypy.reporting.localization.Localization(__file__, 177, 4), getitem___245141, int_245135)
    
    # Assigning a type to the variable 'tuple_var_assignment_244072' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_var_assignment_244072', subscript_call_result_245142)
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'tuple_var_assignment_244069' (line 177)
    tuple_var_assignment_244069_245143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_var_assignment_244069')
    # Assigning a type to the variable 'A1' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'A1', tuple_var_assignment_244069_245143)
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'tuple_var_assignment_244070' (line 177)
    tuple_var_assignment_244070_245144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_var_assignment_244070')
    # Assigning a type to the variable 'b1' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'b1', tuple_var_assignment_244070_245144)
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'tuple_var_assignment_244071' (line 177)
    tuple_var_assignment_244071_245145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_var_assignment_244071')
    # Assigning a type to the variable 'status' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'status', tuple_var_assignment_244071_245145)
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'tuple_var_assignment_244072' (line 177)
    tuple_var_assignment_244072_245146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'tuple_var_assignment_244072')
    # Assigning a type to the variable 'message' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'message', tuple_var_assignment_244072_245146)
    
    # Call to assert_allclose(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'A1' (line 178)
    A1_245148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'A1', False)
    
    # Obtaining the type of the subscript
    int_245149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 27), 'int')
    slice_245150 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 178, 24), None, int_245149, None)
    slice_245151 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 178, 24), None, None, None)
    # Getting the type of 'A' (line 178)
    A_245152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'A', False)
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___245153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 24), A_245152, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_245154 = invoke(stypy.reporting.localization.Localization(__file__, 178, 24), getitem___245153, (slice_245150, slice_245151))
    
    # Processing the call keyword arguments (line 178)
    kwargs_245155 = {}
    # Getting the type of 'assert_allclose' (line 178)
    assert_allclose_245147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 178)
    assert_allclose_call_result_245156 = invoke(stypy.reporting.localization.Localization(__file__, 178, 4), assert_allclose_245147, *[A1_245148, subscript_call_result_245154], **kwargs_245155)
    
    
    # Call to assert_allclose(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'b1' (line 179)
    b1_245158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'b1', False)
    
    # Obtaining the type of the subscript
    int_245159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 27), 'int')
    slice_245160 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 179, 24), None, int_245159, None)
    # Getting the type of 'b' (line 179)
    b_245161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 24), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 179)
    getitem___245162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 24), b_245161, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 179)
    subscript_call_result_245163 = invoke(stypy.reporting.localization.Localization(__file__, 179, 24), getitem___245162, slice_245160)
    
    # Processing the call keyword arguments (line 179)
    kwargs_245164 = {}
    # Getting the type of 'assert_allclose' (line 179)
    assert_allclose_245157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 179)
    assert_allclose_call_result_245165 = invoke(stypy.reporting.localization.Localization(__file__, 179, 4), assert_allclose_245157, *[b1_245158, subscript_call_result_245163], **kwargs_245164)
    
    
    # Call to assert_equal(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'status' (line 180)
    status_245167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'status', False)
    int_245168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 25), 'int')
    # Processing the call keyword arguments (line 180)
    kwargs_245169 = {}
    # Getting the type of 'assert_equal' (line 180)
    assert_equal_245166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 180)
    assert_equal_call_result_245170 = invoke(stypy.reporting.localization.Localization(__file__, 180, 4), assert_equal_245166, *[status_245167, int_245168], **kwargs_245169)
    
    
    # ################# End of 'test_dense3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_dense3' in the type store
    # Getting the type of 'stypy_return_type' (line 171)
    stypy_return_type_245171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_245171)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_dense3'
    return stypy_return_type_245171

# Assigning a type to the variable 'test_dense3' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'test_dense3', test_dense3)

@norecursion
def test_m_gt_n_sparse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_m_gt_n_sparse'
    module_type_store = module_type_store.open_function_context('test_m_gt_n_sparse', 183, 0, False)
    
    # Passed parameters checking function
    test_m_gt_n_sparse.stypy_localization = localization
    test_m_gt_n_sparse.stypy_type_of_self = None
    test_m_gt_n_sparse.stypy_type_store = module_type_store
    test_m_gt_n_sparse.stypy_function_name = 'test_m_gt_n_sparse'
    test_m_gt_n_sparse.stypy_param_names_list = []
    test_m_gt_n_sparse.stypy_varargs_param_name = None
    test_m_gt_n_sparse.stypy_kwargs_param_name = None
    test_m_gt_n_sparse.stypy_call_defaults = defaults
    test_m_gt_n_sparse.stypy_call_varargs = varargs
    test_m_gt_n_sparse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_m_gt_n_sparse', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_m_gt_n_sparse', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_m_gt_n_sparse(...)' code ##################

    
    # Call to seed(...): (line 184)
    # Processing the call arguments (line 184)
    int_245175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 19), 'int')
    # Processing the call keyword arguments (line 184)
    kwargs_245176 = {}
    # Getting the type of 'np' (line 184)
    np_245172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 184)
    random_245173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 4), np_245172, 'random')
    # Obtaining the member 'seed' of a type (line 184)
    seed_245174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 4), random_245173, 'seed')
    # Calling seed(args, kwargs) (line 184)
    seed_call_result_245177 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), seed_245174, *[int_245175], **kwargs_245176)
    
    
    # Assigning a Tuple to a Tuple (line 185):
    
    # Assigning a Num to a Name (line 185):
    int_245178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_244073' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_assignment_244073', int_245178)
    
    # Assigning a Num to a Name (line 185):
    int_245179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 15), 'int')
    # Assigning a type to the variable 'tuple_assignment_244074' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_assignment_244074', int_245179)
    
    # Assigning a Name to a Name (line 185):
    # Getting the type of 'tuple_assignment_244073' (line 185)
    tuple_assignment_244073_245180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_assignment_244073')
    # Assigning a type to the variable 'm' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'm', tuple_assignment_244073_245180)
    
    # Assigning a Name to a Name (line 185):
    # Getting the type of 'tuple_assignment_244074' (line 185)
    tuple_assignment_244074_245181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_assignment_244074')
    # Assigning a type to the variable 'n' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 7), 'n', tuple_assignment_244074_245181)
    
    # Assigning a Num to a Name (line 186):
    
    # Assigning a Num to a Name (line 186):
    float_245182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 8), 'float')
    # Assigning a type to the variable 'p' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'p', float_245182)
    
    # Assigning a Call to a Name (line 187):
    
    # Assigning a Call to a Name (line 187):
    
    # Call to rand(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'm' (line 187)
    m_245186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'm', False)
    # Getting the type of 'n' (line 187)
    n_245187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 26), 'n', False)
    # Processing the call keyword arguments (line 187)
    kwargs_245188 = {}
    # Getting the type of 'np' (line 187)
    np_245183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 187)
    random_245184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), np_245183, 'random')
    # Obtaining the member 'rand' of a type (line 187)
    rand_245185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), random_245184, 'rand')
    # Calling rand(args, kwargs) (line 187)
    rand_call_result_245189 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), rand_245185, *[m_245186, n_245187], **kwargs_245188)
    
    # Assigning a type to the variable 'A' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'A', rand_call_result_245189)
    
    # Assigning a Num to a Subscript (line 188):
    
    # Assigning a Num to a Subscript (line 188):
    int_245190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 34), 'int')
    # Getting the type of 'A' (line 188)
    A_245191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'A')
    
    
    # Call to rand(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'm' (line 188)
    m_245195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 21), 'm', False)
    # Getting the type of 'n' (line 188)
    n_245196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'n', False)
    # Processing the call keyword arguments (line 188)
    kwargs_245197 = {}
    # Getting the type of 'np' (line 188)
    np_245192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 6), 'np', False)
    # Obtaining the member 'random' of a type (line 188)
    random_245193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 6), np_245192, 'random')
    # Obtaining the member 'rand' of a type (line 188)
    rand_245194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 6), random_245193, 'rand')
    # Calling rand(args, kwargs) (line 188)
    rand_call_result_245198 = invoke(stypy.reporting.localization.Localization(__file__, 188, 6), rand_245194, *[m_245195, n_245196], **kwargs_245197)
    
    # Getting the type of 'p' (line 188)
    p_245199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 29), 'p')
    # Applying the binary operator '>' (line 188)
    result_gt_245200 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 6), '>', rand_call_result_245198, p_245199)
    
    # Storing an element on a container (line 188)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 4), A_245191, (result_gt_245200, int_245190))
    
    # Assigning a Call to a Name (line 189):
    
    # Assigning a Call to a Name (line 189):
    
    # Call to matrix_rank(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'A' (line 189)
    A_245204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 33), 'A', False)
    # Processing the call keyword arguments (line 189)
    kwargs_245205 = {}
    # Getting the type of 'np' (line 189)
    np_245201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'np', False)
    # Obtaining the member 'linalg' of a type (line 189)
    linalg_245202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 11), np_245201, 'linalg')
    # Obtaining the member 'matrix_rank' of a type (line 189)
    matrix_rank_245203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 11), linalg_245202, 'matrix_rank')
    # Calling matrix_rank(args, kwargs) (line 189)
    matrix_rank_call_result_245206 = invoke(stypy.reporting.localization.Localization(__file__, 189, 11), matrix_rank_245203, *[A_245204], **kwargs_245205)
    
    # Assigning a type to the variable 'rank' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'rank', matrix_rank_call_result_245206)
    
    # Assigning a Call to a Name (line 190):
    
    # Assigning a Call to a Name (line 190):
    
    # Call to zeros(...): (line 190)
    # Processing the call arguments (line 190)
    
    # Obtaining the type of the subscript
    int_245209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 25), 'int')
    # Getting the type of 'A' (line 190)
    A_245210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'A', False)
    # Obtaining the member 'shape' of a type (line 190)
    shape_245211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 17), A_245210, 'shape')
    # Obtaining the member '__getitem__' of a type (line 190)
    getitem___245212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 17), shape_245211, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 190)
    subscript_call_result_245213 = invoke(stypy.reporting.localization.Localization(__file__, 190, 17), getitem___245212, int_245209)
    
    # Processing the call keyword arguments (line 190)
    kwargs_245214 = {}
    # Getting the type of 'np' (line 190)
    np_245207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 190)
    zeros_245208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), np_245207, 'zeros')
    # Calling zeros(args, kwargs) (line 190)
    zeros_call_result_245215 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), zeros_245208, *[subscript_call_result_245213], **kwargs_245214)
    
    # Assigning a type to the variable 'b' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'b', zeros_call_result_245215)
    
    # Assigning a Call to a Tuple (line 191):
    
    # Assigning a Subscript to a Name (line 191):
    
    # Obtaining the type of the subscript
    int_245216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'A' (line 191)
    A_245218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 49), 'A', False)
    # Getting the type of 'b' (line 191)
    b_245219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 52), 'b', False)
    # Processing the call keyword arguments (line 191)
    kwargs_245220 = {}
    # Getting the type of '_remove_redundancy' (line 191)
    _remove_redundancy_245217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 191)
    _remove_redundancy_call_result_245221 = invoke(stypy.reporting.localization.Localization(__file__, 191, 30), _remove_redundancy_245217, *[A_245218, b_245219], **kwargs_245220)
    
    # Obtaining the member '__getitem__' of a type (line 191)
    getitem___245222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 4), _remove_redundancy_call_result_245221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
    subscript_call_result_245223 = invoke(stypy.reporting.localization.Localization(__file__, 191, 4), getitem___245222, int_245216)
    
    # Assigning a type to the variable 'tuple_var_assignment_244075' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'tuple_var_assignment_244075', subscript_call_result_245223)
    
    # Assigning a Subscript to a Name (line 191):
    
    # Obtaining the type of the subscript
    int_245224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'A' (line 191)
    A_245226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 49), 'A', False)
    # Getting the type of 'b' (line 191)
    b_245227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 52), 'b', False)
    # Processing the call keyword arguments (line 191)
    kwargs_245228 = {}
    # Getting the type of '_remove_redundancy' (line 191)
    _remove_redundancy_245225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 191)
    _remove_redundancy_call_result_245229 = invoke(stypy.reporting.localization.Localization(__file__, 191, 30), _remove_redundancy_245225, *[A_245226, b_245227], **kwargs_245228)
    
    # Obtaining the member '__getitem__' of a type (line 191)
    getitem___245230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 4), _remove_redundancy_call_result_245229, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
    subscript_call_result_245231 = invoke(stypy.reporting.localization.Localization(__file__, 191, 4), getitem___245230, int_245224)
    
    # Assigning a type to the variable 'tuple_var_assignment_244076' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'tuple_var_assignment_244076', subscript_call_result_245231)
    
    # Assigning a Subscript to a Name (line 191):
    
    # Obtaining the type of the subscript
    int_245232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'A' (line 191)
    A_245234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 49), 'A', False)
    # Getting the type of 'b' (line 191)
    b_245235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 52), 'b', False)
    # Processing the call keyword arguments (line 191)
    kwargs_245236 = {}
    # Getting the type of '_remove_redundancy' (line 191)
    _remove_redundancy_245233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 191)
    _remove_redundancy_call_result_245237 = invoke(stypy.reporting.localization.Localization(__file__, 191, 30), _remove_redundancy_245233, *[A_245234, b_245235], **kwargs_245236)
    
    # Obtaining the member '__getitem__' of a type (line 191)
    getitem___245238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 4), _remove_redundancy_call_result_245237, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
    subscript_call_result_245239 = invoke(stypy.reporting.localization.Localization(__file__, 191, 4), getitem___245238, int_245232)
    
    # Assigning a type to the variable 'tuple_var_assignment_244077' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'tuple_var_assignment_244077', subscript_call_result_245239)
    
    # Assigning a Subscript to a Name (line 191):
    
    # Obtaining the type of the subscript
    int_245240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'A' (line 191)
    A_245242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 49), 'A', False)
    # Getting the type of 'b' (line 191)
    b_245243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 52), 'b', False)
    # Processing the call keyword arguments (line 191)
    kwargs_245244 = {}
    # Getting the type of '_remove_redundancy' (line 191)
    _remove_redundancy_245241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 191)
    _remove_redundancy_call_result_245245 = invoke(stypy.reporting.localization.Localization(__file__, 191, 30), _remove_redundancy_245241, *[A_245242, b_245243], **kwargs_245244)
    
    # Obtaining the member '__getitem__' of a type (line 191)
    getitem___245246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 4), _remove_redundancy_call_result_245245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
    subscript_call_result_245247 = invoke(stypy.reporting.localization.Localization(__file__, 191, 4), getitem___245246, int_245240)
    
    # Assigning a type to the variable 'tuple_var_assignment_244078' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'tuple_var_assignment_244078', subscript_call_result_245247)
    
    # Assigning a Name to a Name (line 191):
    # Getting the type of 'tuple_var_assignment_244075' (line 191)
    tuple_var_assignment_244075_245248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'tuple_var_assignment_244075')
    # Assigning a type to the variable 'A1' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'A1', tuple_var_assignment_244075_245248)
    
    # Assigning a Name to a Name (line 191):
    # Getting the type of 'tuple_var_assignment_244076' (line 191)
    tuple_var_assignment_244076_245249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'tuple_var_assignment_244076')
    # Assigning a type to the variable 'b1' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'b1', tuple_var_assignment_244076_245249)
    
    # Assigning a Name to a Name (line 191):
    # Getting the type of 'tuple_var_assignment_244077' (line 191)
    tuple_var_assignment_244077_245250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'tuple_var_assignment_244077')
    # Assigning a type to the variable 'status' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'status', tuple_var_assignment_244077_245250)
    
    # Assigning a Name to a Name (line 191):
    # Getting the type of 'tuple_var_assignment_244078' (line 191)
    tuple_var_assignment_244078_245251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'tuple_var_assignment_244078')
    # Assigning a type to the variable 'message' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), 'message', tuple_var_assignment_244078_245251)
    
    # Call to assert_equal(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'status' (line 192)
    status_245253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 17), 'status', False)
    int_245254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 25), 'int')
    # Processing the call keyword arguments (line 192)
    kwargs_245255 = {}
    # Getting the type of 'assert_equal' (line 192)
    assert_equal_245252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 192)
    assert_equal_call_result_245256 = invoke(stypy.reporting.localization.Localization(__file__, 192, 4), assert_equal_245252, *[status_245253, int_245254], **kwargs_245255)
    
    
    # Call to assert_equal(...): (line 193)
    # Processing the call arguments (line 193)
    
    # Obtaining the type of the subscript
    int_245258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 26), 'int')
    # Getting the type of 'A1' (line 193)
    A1_245259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 17), 'A1', False)
    # Obtaining the member 'shape' of a type (line 193)
    shape_245260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 17), A1_245259, 'shape')
    # Obtaining the member '__getitem__' of a type (line 193)
    getitem___245261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 17), shape_245260, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 193)
    subscript_call_result_245262 = invoke(stypy.reporting.localization.Localization(__file__, 193, 17), getitem___245261, int_245258)
    
    # Getting the type of 'rank' (line 193)
    rank_245263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 30), 'rank', False)
    # Processing the call keyword arguments (line 193)
    kwargs_245264 = {}
    # Getting the type of 'assert_equal' (line 193)
    assert_equal_245257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 193)
    assert_equal_call_result_245265 = invoke(stypy.reporting.localization.Localization(__file__, 193, 4), assert_equal_245257, *[subscript_call_result_245262, rank_245263], **kwargs_245264)
    
    
    # Call to assert_equal(...): (line 194)
    # Processing the call arguments (line 194)
    
    # Call to matrix_rank(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'A1' (line 194)
    A1_245270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 39), 'A1', False)
    # Processing the call keyword arguments (line 194)
    kwargs_245271 = {}
    # Getting the type of 'np' (line 194)
    np_245267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'np', False)
    # Obtaining the member 'linalg' of a type (line 194)
    linalg_245268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), np_245267, 'linalg')
    # Obtaining the member 'matrix_rank' of a type (line 194)
    matrix_rank_245269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), linalg_245268, 'matrix_rank')
    # Calling matrix_rank(args, kwargs) (line 194)
    matrix_rank_call_result_245272 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), matrix_rank_245269, *[A1_245270], **kwargs_245271)
    
    # Getting the type of 'rank' (line 194)
    rank_245273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 44), 'rank', False)
    # Processing the call keyword arguments (line 194)
    kwargs_245274 = {}
    # Getting the type of 'assert_equal' (line 194)
    assert_equal_245266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 194)
    assert_equal_call_result_245275 = invoke(stypy.reporting.localization.Localization(__file__, 194, 4), assert_equal_245266, *[matrix_rank_call_result_245272, rank_245273], **kwargs_245274)
    
    
    # ################# End of 'test_m_gt_n_sparse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_m_gt_n_sparse' in the type store
    # Getting the type of 'stypy_return_type' (line 183)
    stypy_return_type_245276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_245276)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_m_gt_n_sparse'
    return stypy_return_type_245276

# Assigning a type to the variable 'test_m_gt_n_sparse' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'test_m_gt_n_sparse', test_m_gt_n_sparse)

@norecursion
def test_m_lt_n_sparse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_m_lt_n_sparse'
    module_type_store = module_type_store.open_function_context('test_m_lt_n_sparse', 197, 0, False)
    
    # Passed parameters checking function
    test_m_lt_n_sparse.stypy_localization = localization
    test_m_lt_n_sparse.stypy_type_of_self = None
    test_m_lt_n_sparse.stypy_type_store = module_type_store
    test_m_lt_n_sparse.stypy_function_name = 'test_m_lt_n_sparse'
    test_m_lt_n_sparse.stypy_param_names_list = []
    test_m_lt_n_sparse.stypy_varargs_param_name = None
    test_m_lt_n_sparse.stypy_kwargs_param_name = None
    test_m_lt_n_sparse.stypy_call_defaults = defaults
    test_m_lt_n_sparse.stypy_call_varargs = varargs
    test_m_lt_n_sparse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_m_lt_n_sparse', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_m_lt_n_sparse', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_m_lt_n_sparse(...)' code ##################

    
    # Call to seed(...): (line 198)
    # Processing the call arguments (line 198)
    int_245280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 19), 'int')
    # Processing the call keyword arguments (line 198)
    kwargs_245281 = {}
    # Getting the type of 'np' (line 198)
    np_245277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 198)
    random_245278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 4), np_245277, 'random')
    # Obtaining the member 'seed' of a type (line 198)
    seed_245279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 4), random_245278, 'seed')
    # Calling seed(args, kwargs) (line 198)
    seed_call_result_245282 = invoke(stypy.reporting.localization.Localization(__file__, 198, 4), seed_245279, *[int_245280], **kwargs_245281)
    
    
    # Assigning a Tuple to a Tuple (line 199):
    
    # Assigning a Num to a Name (line 199):
    int_245283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_244079' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'tuple_assignment_244079', int_245283)
    
    # Assigning a Num to a Name (line 199):
    int_245284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 15), 'int')
    # Assigning a type to the variable 'tuple_assignment_244080' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'tuple_assignment_244080', int_245284)
    
    # Assigning a Name to a Name (line 199):
    # Getting the type of 'tuple_assignment_244079' (line 199)
    tuple_assignment_244079_245285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'tuple_assignment_244079')
    # Assigning a type to the variable 'm' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'm', tuple_assignment_244079_245285)
    
    # Assigning a Name to a Name (line 199):
    # Getting the type of 'tuple_assignment_244080' (line 199)
    tuple_assignment_244080_245286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'tuple_assignment_244080')
    # Assigning a type to the variable 'n' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 7), 'n', tuple_assignment_244080_245286)
    
    # Assigning a Num to a Name (line 200):
    
    # Assigning a Num to a Name (line 200):
    float_245287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'float')
    # Assigning a type to the variable 'p' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'p', float_245287)
    
    # Assigning a Call to a Name (line 201):
    
    # Assigning a Call to a Name (line 201):
    
    # Call to rand(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'm' (line 201)
    m_245291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), 'm', False)
    # Getting the type of 'n' (line 201)
    n_245292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 26), 'n', False)
    # Processing the call keyword arguments (line 201)
    kwargs_245293 = {}
    # Getting the type of 'np' (line 201)
    np_245288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 201)
    random_245289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), np_245288, 'random')
    # Obtaining the member 'rand' of a type (line 201)
    rand_245290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), random_245289, 'rand')
    # Calling rand(args, kwargs) (line 201)
    rand_call_result_245294 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), rand_245290, *[m_245291, n_245292], **kwargs_245293)
    
    # Assigning a type to the variable 'A' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'A', rand_call_result_245294)
    
    # Assigning a Num to a Subscript (line 202):
    
    # Assigning a Num to a Subscript (line 202):
    int_245295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 34), 'int')
    # Getting the type of 'A' (line 202)
    A_245296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'A')
    
    
    # Call to rand(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'm' (line 202)
    m_245300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 21), 'm', False)
    # Getting the type of 'n' (line 202)
    n_245301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'n', False)
    # Processing the call keyword arguments (line 202)
    kwargs_245302 = {}
    # Getting the type of 'np' (line 202)
    np_245297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 6), 'np', False)
    # Obtaining the member 'random' of a type (line 202)
    random_245298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 6), np_245297, 'random')
    # Obtaining the member 'rand' of a type (line 202)
    rand_245299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 6), random_245298, 'rand')
    # Calling rand(args, kwargs) (line 202)
    rand_call_result_245303 = invoke(stypy.reporting.localization.Localization(__file__, 202, 6), rand_245299, *[m_245300, n_245301], **kwargs_245302)
    
    # Getting the type of 'p' (line 202)
    p_245304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), 'p')
    # Applying the binary operator '>' (line 202)
    result_gt_245305 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 6), '>', rand_call_result_245303, p_245304)
    
    # Storing an element on a container (line 202)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 4), A_245296, (result_gt_245305, int_245295))
    
    # Assigning a Call to a Name (line 203):
    
    # Assigning a Call to a Name (line 203):
    
    # Call to matrix_rank(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'A' (line 203)
    A_245309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 33), 'A', False)
    # Processing the call keyword arguments (line 203)
    kwargs_245310 = {}
    # Getting the type of 'np' (line 203)
    np_245306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'np', False)
    # Obtaining the member 'linalg' of a type (line 203)
    linalg_245307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 11), np_245306, 'linalg')
    # Obtaining the member 'matrix_rank' of a type (line 203)
    matrix_rank_245308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 11), linalg_245307, 'matrix_rank')
    # Calling matrix_rank(args, kwargs) (line 203)
    matrix_rank_call_result_245311 = invoke(stypy.reporting.localization.Localization(__file__, 203, 11), matrix_rank_245308, *[A_245309], **kwargs_245310)
    
    # Assigning a type to the variable 'rank' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'rank', matrix_rank_call_result_245311)
    
    # Assigning a Call to a Name (line 204):
    
    # Assigning a Call to a Name (line 204):
    
    # Call to zeros(...): (line 204)
    # Processing the call arguments (line 204)
    
    # Obtaining the type of the subscript
    int_245314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 25), 'int')
    # Getting the type of 'A' (line 204)
    A_245315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 17), 'A', False)
    # Obtaining the member 'shape' of a type (line 204)
    shape_245316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 17), A_245315, 'shape')
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___245317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 17), shape_245316, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_245318 = invoke(stypy.reporting.localization.Localization(__file__, 204, 17), getitem___245317, int_245314)
    
    # Processing the call keyword arguments (line 204)
    kwargs_245319 = {}
    # Getting the type of 'np' (line 204)
    np_245312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 204)
    zeros_245313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), np_245312, 'zeros')
    # Calling zeros(args, kwargs) (line 204)
    zeros_call_result_245320 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), zeros_245313, *[subscript_call_result_245318], **kwargs_245319)
    
    # Assigning a type to the variable 'b' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'b', zeros_call_result_245320)
    
    # Assigning a Call to a Tuple (line 205):
    
    # Assigning a Subscript to a Name (line 205):
    
    # Obtaining the type of the subscript
    int_245321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'A' (line 205)
    A_245323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 49), 'A', False)
    # Getting the type of 'b' (line 205)
    b_245324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 52), 'b', False)
    # Processing the call keyword arguments (line 205)
    kwargs_245325 = {}
    # Getting the type of '_remove_redundancy' (line 205)
    _remove_redundancy_245322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 205)
    _remove_redundancy_call_result_245326 = invoke(stypy.reporting.localization.Localization(__file__, 205, 30), _remove_redundancy_245322, *[A_245323, b_245324], **kwargs_245325)
    
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___245327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 4), _remove_redundancy_call_result_245326, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_245328 = invoke(stypy.reporting.localization.Localization(__file__, 205, 4), getitem___245327, int_245321)
    
    # Assigning a type to the variable 'tuple_var_assignment_244081' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'tuple_var_assignment_244081', subscript_call_result_245328)
    
    # Assigning a Subscript to a Name (line 205):
    
    # Obtaining the type of the subscript
    int_245329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'A' (line 205)
    A_245331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 49), 'A', False)
    # Getting the type of 'b' (line 205)
    b_245332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 52), 'b', False)
    # Processing the call keyword arguments (line 205)
    kwargs_245333 = {}
    # Getting the type of '_remove_redundancy' (line 205)
    _remove_redundancy_245330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 205)
    _remove_redundancy_call_result_245334 = invoke(stypy.reporting.localization.Localization(__file__, 205, 30), _remove_redundancy_245330, *[A_245331, b_245332], **kwargs_245333)
    
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___245335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 4), _remove_redundancy_call_result_245334, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_245336 = invoke(stypy.reporting.localization.Localization(__file__, 205, 4), getitem___245335, int_245329)
    
    # Assigning a type to the variable 'tuple_var_assignment_244082' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'tuple_var_assignment_244082', subscript_call_result_245336)
    
    # Assigning a Subscript to a Name (line 205):
    
    # Obtaining the type of the subscript
    int_245337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'A' (line 205)
    A_245339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 49), 'A', False)
    # Getting the type of 'b' (line 205)
    b_245340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 52), 'b', False)
    # Processing the call keyword arguments (line 205)
    kwargs_245341 = {}
    # Getting the type of '_remove_redundancy' (line 205)
    _remove_redundancy_245338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 205)
    _remove_redundancy_call_result_245342 = invoke(stypy.reporting.localization.Localization(__file__, 205, 30), _remove_redundancy_245338, *[A_245339, b_245340], **kwargs_245341)
    
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___245343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 4), _remove_redundancy_call_result_245342, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_245344 = invoke(stypy.reporting.localization.Localization(__file__, 205, 4), getitem___245343, int_245337)
    
    # Assigning a type to the variable 'tuple_var_assignment_244083' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'tuple_var_assignment_244083', subscript_call_result_245344)
    
    # Assigning a Subscript to a Name (line 205):
    
    # Obtaining the type of the subscript
    int_245345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'A' (line 205)
    A_245347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 49), 'A', False)
    # Getting the type of 'b' (line 205)
    b_245348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 52), 'b', False)
    # Processing the call keyword arguments (line 205)
    kwargs_245349 = {}
    # Getting the type of '_remove_redundancy' (line 205)
    _remove_redundancy_245346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 205)
    _remove_redundancy_call_result_245350 = invoke(stypy.reporting.localization.Localization(__file__, 205, 30), _remove_redundancy_245346, *[A_245347, b_245348], **kwargs_245349)
    
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___245351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 4), _remove_redundancy_call_result_245350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_245352 = invoke(stypy.reporting.localization.Localization(__file__, 205, 4), getitem___245351, int_245345)
    
    # Assigning a type to the variable 'tuple_var_assignment_244084' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'tuple_var_assignment_244084', subscript_call_result_245352)
    
    # Assigning a Name to a Name (line 205):
    # Getting the type of 'tuple_var_assignment_244081' (line 205)
    tuple_var_assignment_244081_245353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'tuple_var_assignment_244081')
    # Assigning a type to the variable 'A1' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'A1', tuple_var_assignment_244081_245353)
    
    # Assigning a Name to a Name (line 205):
    # Getting the type of 'tuple_var_assignment_244082' (line 205)
    tuple_var_assignment_244082_245354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'tuple_var_assignment_244082')
    # Assigning a type to the variable 'b1' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'b1', tuple_var_assignment_244082_245354)
    
    # Assigning a Name to a Name (line 205):
    # Getting the type of 'tuple_var_assignment_244083' (line 205)
    tuple_var_assignment_244083_245355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'tuple_var_assignment_244083')
    # Assigning a type to the variable 'status' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'status', tuple_var_assignment_244083_245355)
    
    # Assigning a Name to a Name (line 205):
    # Getting the type of 'tuple_var_assignment_244084' (line 205)
    tuple_var_assignment_244084_245356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'tuple_var_assignment_244084')
    # Assigning a type to the variable 'message' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'message', tuple_var_assignment_244084_245356)
    
    # Call to assert_equal(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'status' (line 206)
    status_245358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'status', False)
    int_245359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 25), 'int')
    # Processing the call keyword arguments (line 206)
    kwargs_245360 = {}
    # Getting the type of 'assert_equal' (line 206)
    assert_equal_245357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 206)
    assert_equal_call_result_245361 = invoke(stypy.reporting.localization.Localization(__file__, 206, 4), assert_equal_245357, *[status_245358, int_245359], **kwargs_245360)
    
    
    # Call to assert_equal(...): (line 207)
    # Processing the call arguments (line 207)
    
    # Obtaining the type of the subscript
    int_245363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 26), 'int')
    # Getting the type of 'A1' (line 207)
    A1_245364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 17), 'A1', False)
    # Obtaining the member 'shape' of a type (line 207)
    shape_245365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 17), A1_245364, 'shape')
    # Obtaining the member '__getitem__' of a type (line 207)
    getitem___245366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 17), shape_245365, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 207)
    subscript_call_result_245367 = invoke(stypy.reporting.localization.Localization(__file__, 207, 17), getitem___245366, int_245363)
    
    # Getting the type of 'rank' (line 207)
    rank_245368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 30), 'rank', False)
    # Processing the call keyword arguments (line 207)
    kwargs_245369 = {}
    # Getting the type of 'assert_equal' (line 207)
    assert_equal_245362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 207)
    assert_equal_call_result_245370 = invoke(stypy.reporting.localization.Localization(__file__, 207, 4), assert_equal_245362, *[subscript_call_result_245367, rank_245368], **kwargs_245369)
    
    
    # Call to assert_equal(...): (line 208)
    # Processing the call arguments (line 208)
    
    # Call to matrix_rank(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'A1' (line 208)
    A1_245375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 39), 'A1', False)
    # Processing the call keyword arguments (line 208)
    kwargs_245376 = {}
    # Getting the type of 'np' (line 208)
    np_245372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 17), 'np', False)
    # Obtaining the member 'linalg' of a type (line 208)
    linalg_245373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 17), np_245372, 'linalg')
    # Obtaining the member 'matrix_rank' of a type (line 208)
    matrix_rank_245374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 17), linalg_245373, 'matrix_rank')
    # Calling matrix_rank(args, kwargs) (line 208)
    matrix_rank_call_result_245377 = invoke(stypy.reporting.localization.Localization(__file__, 208, 17), matrix_rank_245374, *[A1_245375], **kwargs_245376)
    
    # Getting the type of 'rank' (line 208)
    rank_245378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 44), 'rank', False)
    # Processing the call keyword arguments (line 208)
    kwargs_245379 = {}
    # Getting the type of 'assert_equal' (line 208)
    assert_equal_245371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 208)
    assert_equal_call_result_245380 = invoke(stypy.reporting.localization.Localization(__file__, 208, 4), assert_equal_245371, *[matrix_rank_call_result_245377, rank_245378], **kwargs_245379)
    
    
    # ################# End of 'test_m_lt_n_sparse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_m_lt_n_sparse' in the type store
    # Getting the type of 'stypy_return_type' (line 197)
    stypy_return_type_245381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_245381)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_m_lt_n_sparse'
    return stypy_return_type_245381

# Assigning a type to the variable 'test_m_lt_n_sparse' (line 197)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 0), 'test_m_lt_n_sparse', test_m_lt_n_sparse)

@norecursion
def test_m_eq_n_sparse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_m_eq_n_sparse'
    module_type_store = module_type_store.open_function_context('test_m_eq_n_sparse', 211, 0, False)
    
    # Passed parameters checking function
    test_m_eq_n_sparse.stypy_localization = localization
    test_m_eq_n_sparse.stypy_type_of_self = None
    test_m_eq_n_sparse.stypy_type_store = module_type_store
    test_m_eq_n_sparse.stypy_function_name = 'test_m_eq_n_sparse'
    test_m_eq_n_sparse.stypy_param_names_list = []
    test_m_eq_n_sparse.stypy_varargs_param_name = None
    test_m_eq_n_sparse.stypy_kwargs_param_name = None
    test_m_eq_n_sparse.stypy_call_defaults = defaults
    test_m_eq_n_sparse.stypy_call_varargs = varargs
    test_m_eq_n_sparse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_m_eq_n_sparse', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_m_eq_n_sparse', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_m_eq_n_sparse(...)' code ##################

    
    # Call to seed(...): (line 212)
    # Processing the call arguments (line 212)
    int_245385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 19), 'int')
    # Processing the call keyword arguments (line 212)
    kwargs_245386 = {}
    # Getting the type of 'np' (line 212)
    np_245382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 212)
    random_245383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 4), np_245382, 'random')
    # Obtaining the member 'seed' of a type (line 212)
    seed_245384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 4), random_245383, 'seed')
    # Calling seed(args, kwargs) (line 212)
    seed_call_result_245387 = invoke(stypy.reporting.localization.Localization(__file__, 212, 4), seed_245384, *[int_245385], **kwargs_245386)
    
    
    # Assigning a Tuple to a Tuple (line 213):
    
    # Assigning a Num to a Name (line 213):
    int_245388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_244085' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'tuple_assignment_244085', int_245388)
    
    # Assigning a Num to a Name (line 213):
    int_245389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 16), 'int')
    # Assigning a type to the variable 'tuple_assignment_244086' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'tuple_assignment_244086', int_245389)
    
    # Assigning a Name to a Name (line 213):
    # Getting the type of 'tuple_assignment_244085' (line 213)
    tuple_assignment_244085_245390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'tuple_assignment_244085')
    # Assigning a type to the variable 'm' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'm', tuple_assignment_244085_245390)
    
    # Assigning a Name to a Name (line 213):
    # Getting the type of 'tuple_assignment_244086' (line 213)
    tuple_assignment_244086_245391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'tuple_assignment_244086')
    # Assigning a type to the variable 'n' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 7), 'n', tuple_assignment_244086_245391)
    
    # Assigning a Num to a Name (line 214):
    
    # Assigning a Num to a Name (line 214):
    float_245392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 8), 'float')
    # Assigning a type to the variable 'p' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'p', float_245392)
    
    # Assigning a Call to a Name (line 215):
    
    # Assigning a Call to a Name (line 215):
    
    # Call to rand(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'm' (line 215)
    m_245396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 'm', False)
    # Getting the type of 'n' (line 215)
    n_245397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 26), 'n', False)
    # Processing the call keyword arguments (line 215)
    kwargs_245398 = {}
    # Getting the type of 'np' (line 215)
    np_245393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 215)
    random_245394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), np_245393, 'random')
    # Obtaining the member 'rand' of a type (line 215)
    rand_245395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), random_245394, 'rand')
    # Calling rand(args, kwargs) (line 215)
    rand_call_result_245399 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), rand_245395, *[m_245396, n_245397], **kwargs_245398)
    
    # Assigning a type to the variable 'A' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'A', rand_call_result_245399)
    
    # Assigning a Num to a Subscript (line 216):
    
    # Assigning a Num to a Subscript (line 216):
    int_245400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 34), 'int')
    # Getting the type of 'A' (line 216)
    A_245401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'A')
    
    
    # Call to rand(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'm' (line 216)
    m_245405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'm', False)
    # Getting the type of 'n' (line 216)
    n_245406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 24), 'n', False)
    # Processing the call keyword arguments (line 216)
    kwargs_245407 = {}
    # Getting the type of 'np' (line 216)
    np_245402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 6), 'np', False)
    # Obtaining the member 'random' of a type (line 216)
    random_245403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 6), np_245402, 'random')
    # Obtaining the member 'rand' of a type (line 216)
    rand_245404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 6), random_245403, 'rand')
    # Calling rand(args, kwargs) (line 216)
    rand_call_result_245408 = invoke(stypy.reporting.localization.Localization(__file__, 216, 6), rand_245404, *[m_245405, n_245406], **kwargs_245407)
    
    # Getting the type of 'p' (line 216)
    p_245409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 29), 'p')
    # Applying the binary operator '>' (line 216)
    result_gt_245410 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 6), '>', rand_call_result_245408, p_245409)
    
    # Storing an element on a container (line 216)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 4), A_245401, (result_gt_245410, int_245400))
    
    # Assigning a Call to a Name (line 217):
    
    # Assigning a Call to a Name (line 217):
    
    # Call to matrix_rank(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'A' (line 217)
    A_245414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 33), 'A', False)
    # Processing the call keyword arguments (line 217)
    kwargs_245415 = {}
    # Getting the type of 'np' (line 217)
    np_245411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'np', False)
    # Obtaining the member 'linalg' of a type (line 217)
    linalg_245412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 11), np_245411, 'linalg')
    # Obtaining the member 'matrix_rank' of a type (line 217)
    matrix_rank_245413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 11), linalg_245412, 'matrix_rank')
    # Calling matrix_rank(args, kwargs) (line 217)
    matrix_rank_call_result_245416 = invoke(stypy.reporting.localization.Localization(__file__, 217, 11), matrix_rank_245413, *[A_245414], **kwargs_245415)
    
    # Assigning a type to the variable 'rank' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'rank', matrix_rank_call_result_245416)
    
    # Assigning a Call to a Name (line 218):
    
    # Assigning a Call to a Name (line 218):
    
    # Call to zeros(...): (line 218)
    # Processing the call arguments (line 218)
    
    # Obtaining the type of the subscript
    int_245419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 25), 'int')
    # Getting the type of 'A' (line 218)
    A_245420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 17), 'A', False)
    # Obtaining the member 'shape' of a type (line 218)
    shape_245421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 17), A_245420, 'shape')
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___245422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 17), shape_245421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_245423 = invoke(stypy.reporting.localization.Localization(__file__, 218, 17), getitem___245422, int_245419)
    
    # Processing the call keyword arguments (line 218)
    kwargs_245424 = {}
    # Getting the type of 'np' (line 218)
    np_245417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 218)
    zeros_245418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), np_245417, 'zeros')
    # Calling zeros(args, kwargs) (line 218)
    zeros_call_result_245425 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), zeros_245418, *[subscript_call_result_245423], **kwargs_245424)
    
    # Assigning a type to the variable 'b' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'b', zeros_call_result_245425)
    
    # Assigning a Call to a Tuple (line 219):
    
    # Assigning a Subscript to a Name (line 219):
    
    # Obtaining the type of the subscript
    int_245426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'A' (line 219)
    A_245428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 49), 'A', False)
    # Getting the type of 'b' (line 219)
    b_245429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 52), 'b', False)
    # Processing the call keyword arguments (line 219)
    kwargs_245430 = {}
    # Getting the type of '_remove_redundancy' (line 219)
    _remove_redundancy_245427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 219)
    _remove_redundancy_call_result_245431 = invoke(stypy.reporting.localization.Localization(__file__, 219, 30), _remove_redundancy_245427, *[A_245428, b_245429], **kwargs_245430)
    
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___245432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 4), _remove_redundancy_call_result_245431, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_245433 = invoke(stypy.reporting.localization.Localization(__file__, 219, 4), getitem___245432, int_245426)
    
    # Assigning a type to the variable 'tuple_var_assignment_244087' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'tuple_var_assignment_244087', subscript_call_result_245433)
    
    # Assigning a Subscript to a Name (line 219):
    
    # Obtaining the type of the subscript
    int_245434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'A' (line 219)
    A_245436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 49), 'A', False)
    # Getting the type of 'b' (line 219)
    b_245437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 52), 'b', False)
    # Processing the call keyword arguments (line 219)
    kwargs_245438 = {}
    # Getting the type of '_remove_redundancy' (line 219)
    _remove_redundancy_245435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 219)
    _remove_redundancy_call_result_245439 = invoke(stypy.reporting.localization.Localization(__file__, 219, 30), _remove_redundancy_245435, *[A_245436, b_245437], **kwargs_245438)
    
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___245440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 4), _remove_redundancy_call_result_245439, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_245441 = invoke(stypy.reporting.localization.Localization(__file__, 219, 4), getitem___245440, int_245434)
    
    # Assigning a type to the variable 'tuple_var_assignment_244088' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'tuple_var_assignment_244088', subscript_call_result_245441)
    
    # Assigning a Subscript to a Name (line 219):
    
    # Obtaining the type of the subscript
    int_245442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'A' (line 219)
    A_245444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 49), 'A', False)
    # Getting the type of 'b' (line 219)
    b_245445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 52), 'b', False)
    # Processing the call keyword arguments (line 219)
    kwargs_245446 = {}
    # Getting the type of '_remove_redundancy' (line 219)
    _remove_redundancy_245443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 219)
    _remove_redundancy_call_result_245447 = invoke(stypy.reporting.localization.Localization(__file__, 219, 30), _remove_redundancy_245443, *[A_245444, b_245445], **kwargs_245446)
    
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___245448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 4), _remove_redundancy_call_result_245447, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_245449 = invoke(stypy.reporting.localization.Localization(__file__, 219, 4), getitem___245448, int_245442)
    
    # Assigning a type to the variable 'tuple_var_assignment_244089' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'tuple_var_assignment_244089', subscript_call_result_245449)
    
    # Assigning a Subscript to a Name (line 219):
    
    # Obtaining the type of the subscript
    int_245450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'A' (line 219)
    A_245452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 49), 'A', False)
    # Getting the type of 'b' (line 219)
    b_245453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 52), 'b', False)
    # Processing the call keyword arguments (line 219)
    kwargs_245454 = {}
    # Getting the type of '_remove_redundancy' (line 219)
    _remove_redundancy_245451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 219)
    _remove_redundancy_call_result_245455 = invoke(stypy.reporting.localization.Localization(__file__, 219, 30), _remove_redundancy_245451, *[A_245452, b_245453], **kwargs_245454)
    
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___245456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 4), _remove_redundancy_call_result_245455, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_245457 = invoke(stypy.reporting.localization.Localization(__file__, 219, 4), getitem___245456, int_245450)
    
    # Assigning a type to the variable 'tuple_var_assignment_244090' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'tuple_var_assignment_244090', subscript_call_result_245457)
    
    # Assigning a Name to a Name (line 219):
    # Getting the type of 'tuple_var_assignment_244087' (line 219)
    tuple_var_assignment_244087_245458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'tuple_var_assignment_244087')
    # Assigning a type to the variable 'A1' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'A1', tuple_var_assignment_244087_245458)
    
    # Assigning a Name to a Name (line 219):
    # Getting the type of 'tuple_var_assignment_244088' (line 219)
    tuple_var_assignment_244088_245459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'tuple_var_assignment_244088')
    # Assigning a type to the variable 'b1' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'b1', tuple_var_assignment_244088_245459)
    
    # Assigning a Name to a Name (line 219):
    # Getting the type of 'tuple_var_assignment_244089' (line 219)
    tuple_var_assignment_244089_245460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'tuple_var_assignment_244089')
    # Assigning a type to the variable 'status' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'status', tuple_var_assignment_244089_245460)
    
    # Assigning a Name to a Name (line 219):
    # Getting the type of 'tuple_var_assignment_244090' (line 219)
    tuple_var_assignment_244090_245461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'tuple_var_assignment_244090')
    # Assigning a type to the variable 'message' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 20), 'message', tuple_var_assignment_244090_245461)
    
    # Call to assert_equal(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'status' (line 220)
    status_245463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 17), 'status', False)
    int_245464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 25), 'int')
    # Processing the call keyword arguments (line 220)
    kwargs_245465 = {}
    # Getting the type of 'assert_equal' (line 220)
    assert_equal_245462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 220)
    assert_equal_call_result_245466 = invoke(stypy.reporting.localization.Localization(__file__, 220, 4), assert_equal_245462, *[status_245463, int_245464], **kwargs_245465)
    
    
    # Call to assert_equal(...): (line 221)
    # Processing the call arguments (line 221)
    
    # Obtaining the type of the subscript
    int_245468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 26), 'int')
    # Getting the type of 'A1' (line 221)
    A1_245469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 17), 'A1', False)
    # Obtaining the member 'shape' of a type (line 221)
    shape_245470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 17), A1_245469, 'shape')
    # Obtaining the member '__getitem__' of a type (line 221)
    getitem___245471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 17), shape_245470, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 221)
    subscript_call_result_245472 = invoke(stypy.reporting.localization.Localization(__file__, 221, 17), getitem___245471, int_245468)
    
    # Getting the type of 'rank' (line 221)
    rank_245473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 30), 'rank', False)
    # Processing the call keyword arguments (line 221)
    kwargs_245474 = {}
    # Getting the type of 'assert_equal' (line 221)
    assert_equal_245467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 221)
    assert_equal_call_result_245475 = invoke(stypy.reporting.localization.Localization(__file__, 221, 4), assert_equal_245467, *[subscript_call_result_245472, rank_245473], **kwargs_245474)
    
    
    # Call to assert_equal(...): (line 222)
    # Processing the call arguments (line 222)
    
    # Call to matrix_rank(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'A1' (line 222)
    A1_245480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 39), 'A1', False)
    # Processing the call keyword arguments (line 222)
    kwargs_245481 = {}
    # Getting the type of 'np' (line 222)
    np_245477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 17), 'np', False)
    # Obtaining the member 'linalg' of a type (line 222)
    linalg_245478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 17), np_245477, 'linalg')
    # Obtaining the member 'matrix_rank' of a type (line 222)
    matrix_rank_245479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 17), linalg_245478, 'matrix_rank')
    # Calling matrix_rank(args, kwargs) (line 222)
    matrix_rank_call_result_245482 = invoke(stypy.reporting.localization.Localization(__file__, 222, 17), matrix_rank_245479, *[A1_245480], **kwargs_245481)
    
    # Getting the type of 'rank' (line 222)
    rank_245483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 44), 'rank', False)
    # Processing the call keyword arguments (line 222)
    kwargs_245484 = {}
    # Getting the type of 'assert_equal' (line 222)
    assert_equal_245476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 222)
    assert_equal_call_result_245485 = invoke(stypy.reporting.localization.Localization(__file__, 222, 4), assert_equal_245476, *[matrix_rank_call_result_245482, rank_245483], **kwargs_245484)
    
    
    # ################# End of 'test_m_eq_n_sparse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_m_eq_n_sparse' in the type store
    # Getting the type of 'stypy_return_type' (line 211)
    stypy_return_type_245486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_245486)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_m_eq_n_sparse'
    return stypy_return_type_245486

# Assigning a type to the variable 'test_m_eq_n_sparse' (line 211)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), 'test_m_eq_n_sparse', test_m_eq_n_sparse)

@norecursion
def test_magic_square(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_magic_square'
    module_type_store = module_type_store.open_function_context('test_magic_square', 225, 0, False)
    
    # Passed parameters checking function
    test_magic_square.stypy_localization = localization
    test_magic_square.stypy_type_of_self = None
    test_magic_square.stypy_type_store = module_type_store
    test_magic_square.stypy_function_name = 'test_magic_square'
    test_magic_square.stypy_param_names_list = []
    test_magic_square.stypy_varargs_param_name = None
    test_magic_square.stypy_kwargs_param_name = None
    test_magic_square.stypy_call_defaults = defaults
    test_magic_square.stypy_call_varargs = varargs
    test_magic_square.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_magic_square', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_magic_square', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_magic_square(...)' code ##################

    
    # Assigning a Call to a Tuple (line 226):
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_245487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'int')
    
    # Call to magic_square(...): (line 226)
    # Processing the call arguments (line 226)
    int_245489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 36), 'int')
    # Processing the call keyword arguments (line 226)
    kwargs_245490 = {}
    # Getting the type of 'magic_square' (line 226)
    magic_square_245488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'magic_square', False)
    # Calling magic_square(args, kwargs) (line 226)
    magic_square_call_result_245491 = invoke(stypy.reporting.localization.Localization(__file__, 226, 23), magic_square_245488, *[int_245489], **kwargs_245490)
    
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___245492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 4), magic_square_call_result_245491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_245493 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), getitem___245492, int_245487)
    
    # Assigning a type to the variable 'tuple_var_assignment_244091' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_244091', subscript_call_result_245493)
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_245494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'int')
    
    # Call to magic_square(...): (line 226)
    # Processing the call arguments (line 226)
    int_245496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 36), 'int')
    # Processing the call keyword arguments (line 226)
    kwargs_245497 = {}
    # Getting the type of 'magic_square' (line 226)
    magic_square_245495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'magic_square', False)
    # Calling magic_square(args, kwargs) (line 226)
    magic_square_call_result_245498 = invoke(stypy.reporting.localization.Localization(__file__, 226, 23), magic_square_245495, *[int_245496], **kwargs_245497)
    
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___245499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 4), magic_square_call_result_245498, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_245500 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), getitem___245499, int_245494)
    
    # Assigning a type to the variable 'tuple_var_assignment_244092' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_244092', subscript_call_result_245500)
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_245501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'int')
    
    # Call to magic_square(...): (line 226)
    # Processing the call arguments (line 226)
    int_245503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 36), 'int')
    # Processing the call keyword arguments (line 226)
    kwargs_245504 = {}
    # Getting the type of 'magic_square' (line 226)
    magic_square_245502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'magic_square', False)
    # Calling magic_square(args, kwargs) (line 226)
    magic_square_call_result_245505 = invoke(stypy.reporting.localization.Localization(__file__, 226, 23), magic_square_245502, *[int_245503], **kwargs_245504)
    
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___245506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 4), magic_square_call_result_245505, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_245507 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), getitem___245506, int_245501)
    
    # Assigning a type to the variable 'tuple_var_assignment_244093' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_244093', subscript_call_result_245507)
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_245508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'int')
    
    # Call to magic_square(...): (line 226)
    # Processing the call arguments (line 226)
    int_245510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 36), 'int')
    # Processing the call keyword arguments (line 226)
    kwargs_245511 = {}
    # Getting the type of 'magic_square' (line 226)
    magic_square_245509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'magic_square', False)
    # Calling magic_square(args, kwargs) (line 226)
    magic_square_call_result_245512 = invoke(stypy.reporting.localization.Localization(__file__, 226, 23), magic_square_245509, *[int_245510], **kwargs_245511)
    
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___245513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 4), magic_square_call_result_245512, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_245514 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), getitem___245513, int_245508)
    
    # Assigning a type to the variable 'tuple_var_assignment_244094' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_244094', subscript_call_result_245514)
    
    # Assigning a Name to a Name (line 226):
    # Getting the type of 'tuple_var_assignment_244091' (line 226)
    tuple_var_assignment_244091_245515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_244091')
    # Assigning a type to the variable 'A' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'A', tuple_var_assignment_244091_245515)
    
    # Assigning a Name to a Name (line 226):
    # Getting the type of 'tuple_var_assignment_244092' (line 226)
    tuple_var_assignment_244092_245516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_244092')
    # Assigning a type to the variable 'b' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 7), 'b', tuple_var_assignment_244092_245516)
    
    # Assigning a Name to a Name (line 226):
    # Getting the type of 'tuple_var_assignment_244093' (line 226)
    tuple_var_assignment_244093_245517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_244093')
    # Assigning a type to the variable 'c' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 10), 'c', tuple_var_assignment_244093_245517)
    
    # Assigning a Name to a Name (line 226):
    # Getting the type of 'tuple_var_assignment_244094' (line 226)
    tuple_var_assignment_244094_245518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_244094')
    # Assigning a type to the variable 'numbers' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 13), 'numbers', tuple_var_assignment_244094_245518)
    
    # Assigning a Call to a Tuple (line 227):
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_245519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'A' (line 227)
    A_245521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 49), 'A', False)
    # Getting the type of 'b' (line 227)
    b_245522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 52), 'b', False)
    # Processing the call keyword arguments (line 227)
    kwargs_245523 = {}
    # Getting the type of '_remove_redundancy' (line 227)
    _remove_redundancy_245520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 227)
    _remove_redundancy_call_result_245524 = invoke(stypy.reporting.localization.Localization(__file__, 227, 30), _remove_redundancy_245520, *[A_245521, b_245522], **kwargs_245523)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___245525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 4), _remove_redundancy_call_result_245524, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_245526 = invoke(stypy.reporting.localization.Localization(__file__, 227, 4), getitem___245525, int_245519)
    
    # Assigning a type to the variable 'tuple_var_assignment_244095' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'tuple_var_assignment_244095', subscript_call_result_245526)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_245527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'A' (line 227)
    A_245529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 49), 'A', False)
    # Getting the type of 'b' (line 227)
    b_245530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 52), 'b', False)
    # Processing the call keyword arguments (line 227)
    kwargs_245531 = {}
    # Getting the type of '_remove_redundancy' (line 227)
    _remove_redundancy_245528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 227)
    _remove_redundancy_call_result_245532 = invoke(stypy.reporting.localization.Localization(__file__, 227, 30), _remove_redundancy_245528, *[A_245529, b_245530], **kwargs_245531)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___245533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 4), _remove_redundancy_call_result_245532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_245534 = invoke(stypy.reporting.localization.Localization(__file__, 227, 4), getitem___245533, int_245527)
    
    # Assigning a type to the variable 'tuple_var_assignment_244096' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'tuple_var_assignment_244096', subscript_call_result_245534)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_245535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'A' (line 227)
    A_245537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 49), 'A', False)
    # Getting the type of 'b' (line 227)
    b_245538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 52), 'b', False)
    # Processing the call keyword arguments (line 227)
    kwargs_245539 = {}
    # Getting the type of '_remove_redundancy' (line 227)
    _remove_redundancy_245536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 227)
    _remove_redundancy_call_result_245540 = invoke(stypy.reporting.localization.Localization(__file__, 227, 30), _remove_redundancy_245536, *[A_245537, b_245538], **kwargs_245539)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___245541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 4), _remove_redundancy_call_result_245540, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_245542 = invoke(stypy.reporting.localization.Localization(__file__, 227, 4), getitem___245541, int_245535)
    
    # Assigning a type to the variable 'tuple_var_assignment_244097' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'tuple_var_assignment_244097', subscript_call_result_245542)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_245543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'A' (line 227)
    A_245545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 49), 'A', False)
    # Getting the type of 'b' (line 227)
    b_245546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 52), 'b', False)
    # Processing the call keyword arguments (line 227)
    kwargs_245547 = {}
    # Getting the type of '_remove_redundancy' (line 227)
    _remove_redundancy_245544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 227)
    _remove_redundancy_call_result_245548 = invoke(stypy.reporting.localization.Localization(__file__, 227, 30), _remove_redundancy_245544, *[A_245545, b_245546], **kwargs_245547)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___245549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 4), _remove_redundancy_call_result_245548, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_245550 = invoke(stypy.reporting.localization.Localization(__file__, 227, 4), getitem___245549, int_245543)
    
    # Assigning a type to the variable 'tuple_var_assignment_244098' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'tuple_var_assignment_244098', subscript_call_result_245550)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_244095' (line 227)
    tuple_var_assignment_244095_245551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'tuple_var_assignment_244095')
    # Assigning a type to the variable 'A1' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'A1', tuple_var_assignment_244095_245551)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_244096' (line 227)
    tuple_var_assignment_244096_245552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'tuple_var_assignment_244096')
    # Assigning a type to the variable 'b1' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'b1', tuple_var_assignment_244096_245552)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_244097' (line 227)
    tuple_var_assignment_244097_245553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'tuple_var_assignment_244097')
    # Assigning a type to the variable 'status' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'status', tuple_var_assignment_244097_245553)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_244098' (line 227)
    tuple_var_assignment_244098_245554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'tuple_var_assignment_244098')
    # Assigning a type to the variable 'message' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'message', tuple_var_assignment_244098_245554)
    
    # Call to assert_equal(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'status' (line 228)
    status_245556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), 'status', False)
    int_245557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 25), 'int')
    # Processing the call keyword arguments (line 228)
    kwargs_245558 = {}
    # Getting the type of 'assert_equal' (line 228)
    assert_equal_245555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 228)
    assert_equal_call_result_245559 = invoke(stypy.reporting.localization.Localization(__file__, 228, 4), assert_equal_245555, *[status_245556, int_245557], **kwargs_245558)
    
    
    # Call to assert_equal(...): (line 229)
    # Processing the call arguments (line 229)
    
    # Obtaining the type of the subscript
    int_245561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 26), 'int')
    # Getting the type of 'A1' (line 229)
    A1_245562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 17), 'A1', False)
    # Obtaining the member 'shape' of a type (line 229)
    shape_245563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 17), A1_245562, 'shape')
    # Obtaining the member '__getitem__' of a type (line 229)
    getitem___245564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 17), shape_245563, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 229)
    subscript_call_result_245565 = invoke(stypy.reporting.localization.Localization(__file__, 229, 17), getitem___245564, int_245561)
    
    int_245566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 30), 'int')
    # Processing the call keyword arguments (line 229)
    kwargs_245567 = {}
    # Getting the type of 'assert_equal' (line 229)
    assert_equal_245560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 229)
    assert_equal_call_result_245568 = invoke(stypy.reporting.localization.Localization(__file__, 229, 4), assert_equal_245560, *[subscript_call_result_245565, int_245566], **kwargs_245567)
    
    
    # Call to assert_equal(...): (line 230)
    # Processing the call arguments (line 230)
    
    # Call to matrix_rank(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'A1' (line 230)
    A1_245573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 39), 'A1', False)
    # Processing the call keyword arguments (line 230)
    kwargs_245574 = {}
    # Getting the type of 'np' (line 230)
    np_245570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 17), 'np', False)
    # Obtaining the member 'linalg' of a type (line 230)
    linalg_245571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 17), np_245570, 'linalg')
    # Obtaining the member 'matrix_rank' of a type (line 230)
    matrix_rank_245572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 17), linalg_245571, 'matrix_rank')
    # Calling matrix_rank(args, kwargs) (line 230)
    matrix_rank_call_result_245575 = invoke(stypy.reporting.localization.Localization(__file__, 230, 17), matrix_rank_245572, *[A1_245573], **kwargs_245574)
    
    int_245576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 44), 'int')
    # Processing the call keyword arguments (line 230)
    kwargs_245577 = {}
    # Getting the type of 'assert_equal' (line 230)
    assert_equal_245569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 230)
    assert_equal_call_result_245578 = invoke(stypy.reporting.localization.Localization(__file__, 230, 4), assert_equal_245569, *[matrix_rank_call_result_245575, int_245576], **kwargs_245577)
    
    
    # ################# End of 'test_magic_square(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_magic_square' in the type store
    # Getting the type of 'stypy_return_type' (line 225)
    stypy_return_type_245579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_245579)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_magic_square'
    return stypy_return_type_245579

# Assigning a type to the variable 'test_magic_square' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'test_magic_square', test_magic_square)

@norecursion
def test_magic_square2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_magic_square2'
    module_type_store = module_type_store.open_function_context('test_magic_square2', 233, 0, False)
    
    # Passed parameters checking function
    test_magic_square2.stypy_localization = localization
    test_magic_square2.stypy_type_of_self = None
    test_magic_square2.stypy_type_store = module_type_store
    test_magic_square2.stypy_function_name = 'test_magic_square2'
    test_magic_square2.stypy_param_names_list = []
    test_magic_square2.stypy_varargs_param_name = None
    test_magic_square2.stypy_kwargs_param_name = None
    test_magic_square2.stypy_call_defaults = defaults
    test_magic_square2.stypy_call_varargs = varargs
    test_magic_square2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_magic_square2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_magic_square2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_magic_square2(...)' code ##################

    
    # Assigning a Call to a Tuple (line 234):
    
    # Assigning a Subscript to a Name (line 234):
    
    # Obtaining the type of the subscript
    int_245580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 4), 'int')
    
    # Call to magic_square(...): (line 234)
    # Processing the call arguments (line 234)
    int_245582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 36), 'int')
    # Processing the call keyword arguments (line 234)
    kwargs_245583 = {}
    # Getting the type of 'magic_square' (line 234)
    magic_square_245581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 23), 'magic_square', False)
    # Calling magic_square(args, kwargs) (line 234)
    magic_square_call_result_245584 = invoke(stypy.reporting.localization.Localization(__file__, 234, 23), magic_square_245581, *[int_245582], **kwargs_245583)
    
    # Obtaining the member '__getitem__' of a type (line 234)
    getitem___245585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 4), magic_square_call_result_245584, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 234)
    subscript_call_result_245586 = invoke(stypy.reporting.localization.Localization(__file__, 234, 4), getitem___245585, int_245580)
    
    # Assigning a type to the variable 'tuple_var_assignment_244099' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'tuple_var_assignment_244099', subscript_call_result_245586)
    
    # Assigning a Subscript to a Name (line 234):
    
    # Obtaining the type of the subscript
    int_245587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 4), 'int')
    
    # Call to magic_square(...): (line 234)
    # Processing the call arguments (line 234)
    int_245589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 36), 'int')
    # Processing the call keyword arguments (line 234)
    kwargs_245590 = {}
    # Getting the type of 'magic_square' (line 234)
    magic_square_245588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 23), 'magic_square', False)
    # Calling magic_square(args, kwargs) (line 234)
    magic_square_call_result_245591 = invoke(stypy.reporting.localization.Localization(__file__, 234, 23), magic_square_245588, *[int_245589], **kwargs_245590)
    
    # Obtaining the member '__getitem__' of a type (line 234)
    getitem___245592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 4), magic_square_call_result_245591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 234)
    subscript_call_result_245593 = invoke(stypy.reporting.localization.Localization(__file__, 234, 4), getitem___245592, int_245587)
    
    # Assigning a type to the variable 'tuple_var_assignment_244100' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'tuple_var_assignment_244100', subscript_call_result_245593)
    
    # Assigning a Subscript to a Name (line 234):
    
    # Obtaining the type of the subscript
    int_245594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 4), 'int')
    
    # Call to magic_square(...): (line 234)
    # Processing the call arguments (line 234)
    int_245596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 36), 'int')
    # Processing the call keyword arguments (line 234)
    kwargs_245597 = {}
    # Getting the type of 'magic_square' (line 234)
    magic_square_245595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 23), 'magic_square', False)
    # Calling magic_square(args, kwargs) (line 234)
    magic_square_call_result_245598 = invoke(stypy.reporting.localization.Localization(__file__, 234, 23), magic_square_245595, *[int_245596], **kwargs_245597)
    
    # Obtaining the member '__getitem__' of a type (line 234)
    getitem___245599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 4), magic_square_call_result_245598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 234)
    subscript_call_result_245600 = invoke(stypy.reporting.localization.Localization(__file__, 234, 4), getitem___245599, int_245594)
    
    # Assigning a type to the variable 'tuple_var_assignment_244101' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'tuple_var_assignment_244101', subscript_call_result_245600)
    
    # Assigning a Subscript to a Name (line 234):
    
    # Obtaining the type of the subscript
    int_245601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 4), 'int')
    
    # Call to magic_square(...): (line 234)
    # Processing the call arguments (line 234)
    int_245603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 36), 'int')
    # Processing the call keyword arguments (line 234)
    kwargs_245604 = {}
    # Getting the type of 'magic_square' (line 234)
    magic_square_245602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 23), 'magic_square', False)
    # Calling magic_square(args, kwargs) (line 234)
    magic_square_call_result_245605 = invoke(stypy.reporting.localization.Localization(__file__, 234, 23), magic_square_245602, *[int_245603], **kwargs_245604)
    
    # Obtaining the member '__getitem__' of a type (line 234)
    getitem___245606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 4), magic_square_call_result_245605, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 234)
    subscript_call_result_245607 = invoke(stypy.reporting.localization.Localization(__file__, 234, 4), getitem___245606, int_245601)
    
    # Assigning a type to the variable 'tuple_var_assignment_244102' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'tuple_var_assignment_244102', subscript_call_result_245607)
    
    # Assigning a Name to a Name (line 234):
    # Getting the type of 'tuple_var_assignment_244099' (line 234)
    tuple_var_assignment_244099_245608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'tuple_var_assignment_244099')
    # Assigning a type to the variable 'A' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'A', tuple_var_assignment_244099_245608)
    
    # Assigning a Name to a Name (line 234):
    # Getting the type of 'tuple_var_assignment_244100' (line 234)
    tuple_var_assignment_244100_245609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'tuple_var_assignment_244100')
    # Assigning a type to the variable 'b' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 7), 'b', tuple_var_assignment_244100_245609)
    
    # Assigning a Name to a Name (line 234):
    # Getting the type of 'tuple_var_assignment_244101' (line 234)
    tuple_var_assignment_244101_245610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'tuple_var_assignment_244101')
    # Assigning a type to the variable 'c' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 10), 'c', tuple_var_assignment_244101_245610)
    
    # Assigning a Name to a Name (line 234):
    # Getting the type of 'tuple_var_assignment_244102' (line 234)
    tuple_var_assignment_244102_245611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'tuple_var_assignment_244102')
    # Assigning a type to the variable 'numbers' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 13), 'numbers', tuple_var_assignment_244102_245611)
    
    # Assigning a Call to a Tuple (line 235):
    
    # Assigning a Subscript to a Name (line 235):
    
    # Obtaining the type of the subscript
    int_245612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'A' (line 235)
    A_245614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 49), 'A', False)
    # Getting the type of 'b' (line 235)
    b_245615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 52), 'b', False)
    # Processing the call keyword arguments (line 235)
    kwargs_245616 = {}
    # Getting the type of '_remove_redundancy' (line 235)
    _remove_redundancy_245613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 235)
    _remove_redundancy_call_result_245617 = invoke(stypy.reporting.localization.Localization(__file__, 235, 30), _remove_redundancy_245613, *[A_245614, b_245615], **kwargs_245616)
    
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___245618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 4), _remove_redundancy_call_result_245617, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_245619 = invoke(stypy.reporting.localization.Localization(__file__, 235, 4), getitem___245618, int_245612)
    
    # Assigning a type to the variable 'tuple_var_assignment_244103' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'tuple_var_assignment_244103', subscript_call_result_245619)
    
    # Assigning a Subscript to a Name (line 235):
    
    # Obtaining the type of the subscript
    int_245620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'A' (line 235)
    A_245622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 49), 'A', False)
    # Getting the type of 'b' (line 235)
    b_245623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 52), 'b', False)
    # Processing the call keyword arguments (line 235)
    kwargs_245624 = {}
    # Getting the type of '_remove_redundancy' (line 235)
    _remove_redundancy_245621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 235)
    _remove_redundancy_call_result_245625 = invoke(stypy.reporting.localization.Localization(__file__, 235, 30), _remove_redundancy_245621, *[A_245622, b_245623], **kwargs_245624)
    
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___245626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 4), _remove_redundancy_call_result_245625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_245627 = invoke(stypy.reporting.localization.Localization(__file__, 235, 4), getitem___245626, int_245620)
    
    # Assigning a type to the variable 'tuple_var_assignment_244104' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'tuple_var_assignment_244104', subscript_call_result_245627)
    
    # Assigning a Subscript to a Name (line 235):
    
    # Obtaining the type of the subscript
    int_245628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'A' (line 235)
    A_245630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 49), 'A', False)
    # Getting the type of 'b' (line 235)
    b_245631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 52), 'b', False)
    # Processing the call keyword arguments (line 235)
    kwargs_245632 = {}
    # Getting the type of '_remove_redundancy' (line 235)
    _remove_redundancy_245629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 235)
    _remove_redundancy_call_result_245633 = invoke(stypy.reporting.localization.Localization(__file__, 235, 30), _remove_redundancy_245629, *[A_245630, b_245631], **kwargs_245632)
    
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___245634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 4), _remove_redundancy_call_result_245633, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_245635 = invoke(stypy.reporting.localization.Localization(__file__, 235, 4), getitem___245634, int_245628)
    
    # Assigning a type to the variable 'tuple_var_assignment_244105' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'tuple_var_assignment_244105', subscript_call_result_245635)
    
    # Assigning a Subscript to a Name (line 235):
    
    # Obtaining the type of the subscript
    int_245636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 4), 'int')
    
    # Call to _remove_redundancy(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'A' (line 235)
    A_245638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 49), 'A', False)
    # Getting the type of 'b' (line 235)
    b_245639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 52), 'b', False)
    # Processing the call keyword arguments (line 235)
    kwargs_245640 = {}
    # Getting the type of '_remove_redundancy' (line 235)
    _remove_redundancy_245637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), '_remove_redundancy', False)
    # Calling _remove_redundancy(args, kwargs) (line 235)
    _remove_redundancy_call_result_245641 = invoke(stypy.reporting.localization.Localization(__file__, 235, 30), _remove_redundancy_245637, *[A_245638, b_245639], **kwargs_245640)
    
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___245642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 4), _remove_redundancy_call_result_245641, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_245643 = invoke(stypy.reporting.localization.Localization(__file__, 235, 4), getitem___245642, int_245636)
    
    # Assigning a type to the variable 'tuple_var_assignment_244106' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'tuple_var_assignment_244106', subscript_call_result_245643)
    
    # Assigning a Name to a Name (line 235):
    # Getting the type of 'tuple_var_assignment_244103' (line 235)
    tuple_var_assignment_244103_245644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'tuple_var_assignment_244103')
    # Assigning a type to the variable 'A1' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'A1', tuple_var_assignment_244103_245644)
    
    # Assigning a Name to a Name (line 235):
    # Getting the type of 'tuple_var_assignment_244104' (line 235)
    tuple_var_assignment_244104_245645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'tuple_var_assignment_244104')
    # Assigning a type to the variable 'b1' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'b1', tuple_var_assignment_244104_245645)
    
    # Assigning a Name to a Name (line 235):
    # Getting the type of 'tuple_var_assignment_244105' (line 235)
    tuple_var_assignment_244105_245646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'tuple_var_assignment_244105')
    # Assigning a type to the variable 'status' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'status', tuple_var_assignment_244105_245646)
    
    # Assigning a Name to a Name (line 235):
    # Getting the type of 'tuple_var_assignment_244106' (line 235)
    tuple_var_assignment_244106_245647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'tuple_var_assignment_244106')
    # Assigning a type to the variable 'message' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'message', tuple_var_assignment_244106_245647)
    
    # Call to assert_equal(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'status' (line 236)
    status_245649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 17), 'status', False)
    int_245650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 25), 'int')
    # Processing the call keyword arguments (line 236)
    kwargs_245651 = {}
    # Getting the type of 'assert_equal' (line 236)
    assert_equal_245648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 236)
    assert_equal_call_result_245652 = invoke(stypy.reporting.localization.Localization(__file__, 236, 4), assert_equal_245648, *[status_245649, int_245650], **kwargs_245651)
    
    
    # Call to assert_equal(...): (line 237)
    # Processing the call arguments (line 237)
    
    # Obtaining the type of the subscript
    int_245654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 26), 'int')
    # Getting the type of 'A1' (line 237)
    A1_245655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 17), 'A1', False)
    # Obtaining the member 'shape' of a type (line 237)
    shape_245656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 17), A1_245655, 'shape')
    # Obtaining the member '__getitem__' of a type (line 237)
    getitem___245657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 17), shape_245656, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 237)
    subscript_call_result_245658 = invoke(stypy.reporting.localization.Localization(__file__, 237, 17), getitem___245657, int_245654)
    
    int_245659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 30), 'int')
    # Processing the call keyword arguments (line 237)
    kwargs_245660 = {}
    # Getting the type of 'assert_equal' (line 237)
    assert_equal_245653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 237)
    assert_equal_call_result_245661 = invoke(stypy.reporting.localization.Localization(__file__, 237, 4), assert_equal_245653, *[subscript_call_result_245658, int_245659], **kwargs_245660)
    
    
    # Call to assert_equal(...): (line 238)
    # Processing the call arguments (line 238)
    
    # Call to matrix_rank(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'A1' (line 238)
    A1_245666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 39), 'A1', False)
    # Processing the call keyword arguments (line 238)
    kwargs_245667 = {}
    # Getting the type of 'np' (line 238)
    np_245663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 17), 'np', False)
    # Obtaining the member 'linalg' of a type (line 238)
    linalg_245664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 17), np_245663, 'linalg')
    # Obtaining the member 'matrix_rank' of a type (line 238)
    matrix_rank_245665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 17), linalg_245664, 'matrix_rank')
    # Calling matrix_rank(args, kwargs) (line 238)
    matrix_rank_call_result_245668 = invoke(stypy.reporting.localization.Localization(__file__, 238, 17), matrix_rank_245665, *[A1_245666], **kwargs_245667)
    
    int_245669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 44), 'int')
    # Processing the call keyword arguments (line 238)
    kwargs_245670 = {}
    # Getting the type of 'assert_equal' (line 238)
    assert_equal_245662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 238)
    assert_equal_call_result_245671 = invoke(stypy.reporting.localization.Localization(__file__, 238, 4), assert_equal_245662, *[matrix_rank_call_result_245668, int_245669], **kwargs_245670)
    
    
    # ################# End of 'test_magic_square2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_magic_square2' in the type store
    # Getting the type of 'stypy_return_type' (line 233)
    stypy_return_type_245672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_245672)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_magic_square2'
    return stypy_return_type_245672

# Assigning a type to the variable 'test_magic_square2' (line 233)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 0), 'test_magic_square2', test_magic_square2)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
