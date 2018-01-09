
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import (assert_almost_equal, assert_equal,
5:                            assert_array_almost_equal, assert_)
6: 
7: from scipy.special import logsumexp
8: 
9: 
10: def test_logsumexp():
11:     # Test whether logsumexp() function correctly handles large inputs.
12:     a = np.arange(200)
13:     desired = np.log(np.sum(np.exp(a)))
14:     assert_almost_equal(logsumexp(a), desired)
15: 
16:     # Now test with large numbers
17:     b = [1000, 1000]
18:     desired = 1000.0 + np.log(2.0)
19:     assert_almost_equal(logsumexp(b), desired)
20: 
21:     n = 1000
22:     b = np.ones(n) * 10000
23:     desired = 10000.0 + np.log(n)
24:     assert_almost_equal(logsumexp(b), desired)
25: 
26:     x = np.array([1e-40] * 1000000)
27:     logx = np.log(x)
28: 
29:     X = np.vstack([x, x])
30:     logX = np.vstack([logx, logx])
31:     assert_array_almost_equal(np.exp(logsumexp(logX)), X.sum())
32:     assert_array_almost_equal(np.exp(logsumexp(logX, axis=0)), X.sum(axis=0))
33:     assert_array_almost_equal(np.exp(logsumexp(logX, axis=1)), X.sum(axis=1))
34: 
35:     # Handling special values properly
36:     assert_equal(logsumexp(np.inf), np.inf)
37:     assert_equal(logsumexp(-np.inf), -np.inf)
38:     assert_equal(logsumexp(np.nan), np.nan)
39:     assert_equal(logsumexp([-np.inf, -np.inf]), -np.inf)
40: 
41:     # Handling an array with different magnitudes on the axes
42:     assert_array_almost_equal(logsumexp([[1e10, 1e-10],
43:                                          [-1e10, -np.inf]], axis=-1),
44:                               [1e10, -1e10])
45: 
46:     # Test keeping dimensions
47:     assert_array_almost_equal(logsumexp([[1e10, 1e-10],
48:                                          [-1e10, -np.inf]],
49:                                         axis=-1,
50:                                         keepdims=True),
51:                               [[1e10], [-1e10]])
52: 
53:     # Test multiple axes
54:     assert_array_almost_equal(logsumexp([[1e10, 1e-10],
55:                                          [-1e10, -np.inf]],
56:                                         axis=(-1,-2)),
57:                               1e10)
58: 
59: 
60: def test_logsumexp_b():
61:     a = np.arange(200)
62:     b = np.arange(200, 0, -1)
63:     desired = np.log(np.sum(b*np.exp(a)))
64:     assert_almost_equal(logsumexp(a, b=b), desired)
65: 
66:     a = [1000, 1000]
67:     b = [1.2, 1.2]
68:     desired = 1000 + np.log(2 * 1.2)
69:     assert_almost_equal(logsumexp(a, b=b), desired)
70: 
71:     x = np.array([1e-40] * 100000)
72:     b = np.linspace(1, 1000, 100000)
73:     logx = np.log(x)
74: 
75:     X = np.vstack((x, x))
76:     logX = np.vstack((logx, logx))
77:     B = np.vstack((b, b))
78:     assert_array_almost_equal(np.exp(logsumexp(logX, b=B)), (B * X).sum())
79:     assert_array_almost_equal(np.exp(logsumexp(logX, b=B, axis=0)),
80:                                 (B * X).sum(axis=0))
81:     assert_array_almost_equal(np.exp(logsumexp(logX, b=B, axis=1)),
82:                                 (B * X).sum(axis=1))
83: 
84: 
85: def test_logsumexp_sign():
86:     a = [1,1,1]
87:     b = [1,-1,-1]
88: 
89:     r, s = logsumexp(a, b=b, return_sign=True)
90:     assert_almost_equal(r,1)
91:     assert_equal(s,-1)
92: 
93: 
94: def test_logsumexp_sign_zero():
95:     a = [1,1]
96:     b = [1,-1]
97: 
98:     r, s = logsumexp(a, b=b, return_sign=True)
99:     assert_(not np.isfinite(r))
100:     assert_(not np.isnan(r))
101:     assert_(r < 0)
102:     assert_equal(s,0)
103: 
104: 
105: def test_logsumexp_sign_shape():
106:     a = np.ones((1,2,3,4))
107:     b = np.ones_like(a)
108: 
109:     r, s = logsumexp(a, axis=2, b=b, return_sign=True)
110: 
111:     assert_equal(r.shape, s.shape)
112:     assert_equal(r.shape, (1,2,4))
113: 
114:     r, s = logsumexp(a, axis=(1,3), b=b, return_sign=True)
115: 
116:     assert_equal(r.shape, s.shape)
117:     assert_equal(r.shape, (1,3))
118: 
119: 
120: def test_logsumexp_shape():
121:     a = np.ones((1, 2, 3, 4))
122:     b = np.ones_like(a)
123: 
124:     r = logsumexp(a, axis=2, b=b)
125:     assert_equal(r.shape, (1, 2, 4))
126: 
127:     r = logsumexp(a, axis=(1, 3), b=b)
128:     assert_equal(r.shape, (1, 3))
129: 
130: 
131: def test_logsumexp_b_zero():
132:     a = [1,10000]
133:     b = [1,0]
134: 
135:     assert_almost_equal(logsumexp(a, b=b), 1)
136: 
137: 
138: def test_logsumexp_b_shape():
139:     a = np.zeros((4,1,2,1))
140:     b = np.ones((3,1,5))
141: 
142:     logsumexp(a, b=b)
143: 
144: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_542023 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_542023) is not StypyTypeError):

    if (import_542023 != 'pyd_module'):
        __import__(import_542023)
        sys_modules_542024 = sys.modules[import_542023]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_542024.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_542023)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_almost_equal, assert_equal, assert_array_almost_equal, assert_' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_542025 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_542025) is not StypyTypeError):

    if (import_542025 != 'pyd_module'):
        __import__(import_542025)
        sys_modules_542026 = sys.modules[import_542025]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_542026.module_type_store, module_type_store, ['assert_almost_equal', 'assert_equal', 'assert_array_almost_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_542026, sys_modules_542026.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_almost_equal, assert_equal, assert_array_almost_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_almost_equal', 'assert_equal', 'assert_array_almost_equal', 'assert_'], [assert_almost_equal, assert_equal, assert_array_almost_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_542025)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.special import logsumexp' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_542027 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special')

if (type(import_542027) is not StypyTypeError):

    if (import_542027 != 'pyd_module'):
        __import__(import_542027)
        sys_modules_542028 = sys.modules[import_542027]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special', sys_modules_542028.module_type_store, module_type_store, ['logsumexp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_542028, sys_modules_542028.module_type_store, module_type_store)
    else:
        from scipy.special import logsumexp

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special', None, module_type_store, ['logsumexp'], [logsumexp])

else:
    # Assigning a type to the variable 'scipy.special' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special', import_542027)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


@norecursion
def test_logsumexp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_logsumexp'
    module_type_store = module_type_store.open_function_context('test_logsumexp', 10, 0, False)
    
    # Passed parameters checking function
    test_logsumexp.stypy_localization = localization
    test_logsumexp.stypy_type_of_self = None
    test_logsumexp.stypy_type_store = module_type_store
    test_logsumexp.stypy_function_name = 'test_logsumexp'
    test_logsumexp.stypy_param_names_list = []
    test_logsumexp.stypy_varargs_param_name = None
    test_logsumexp.stypy_kwargs_param_name = None
    test_logsumexp.stypy_call_defaults = defaults
    test_logsumexp.stypy_call_varargs = varargs
    test_logsumexp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_logsumexp', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_logsumexp', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_logsumexp(...)' code ##################

    
    # Assigning a Call to a Name (line 12):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to arange(...): (line 12)
    # Processing the call arguments (line 12)
    int_542031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'int')
    # Processing the call keyword arguments (line 12)
    kwargs_542032 = {}
    # Getting the type of 'np' (line 12)
    np_542029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 12)
    arange_542030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), np_542029, 'arange')
    # Calling arange(args, kwargs) (line 12)
    arange_call_result_542033 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), arange_542030, *[int_542031], **kwargs_542032)
    
    # Assigning a type to the variable 'a' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'a', arange_call_result_542033)
    
    # Assigning a Call to a Name (line 13):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to log(...): (line 13)
    # Processing the call arguments (line 13)
    
    # Call to sum(...): (line 13)
    # Processing the call arguments (line 13)
    
    # Call to exp(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'a' (line 13)
    a_542040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 35), 'a', False)
    # Processing the call keyword arguments (line 13)
    kwargs_542041 = {}
    # Getting the type of 'np' (line 13)
    np_542038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'np', False)
    # Obtaining the member 'exp' of a type (line 13)
    exp_542039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 28), np_542038, 'exp')
    # Calling exp(args, kwargs) (line 13)
    exp_call_result_542042 = invoke(stypy.reporting.localization.Localization(__file__, 13, 28), exp_542039, *[a_542040], **kwargs_542041)
    
    # Processing the call keyword arguments (line 13)
    kwargs_542043 = {}
    # Getting the type of 'np' (line 13)
    np_542036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 21), 'np', False)
    # Obtaining the member 'sum' of a type (line 13)
    sum_542037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 21), np_542036, 'sum')
    # Calling sum(args, kwargs) (line 13)
    sum_call_result_542044 = invoke(stypy.reporting.localization.Localization(__file__, 13, 21), sum_542037, *[exp_call_result_542042], **kwargs_542043)
    
    # Processing the call keyword arguments (line 13)
    kwargs_542045 = {}
    # Getting the type of 'np' (line 13)
    np_542034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'np', False)
    # Obtaining the member 'log' of a type (line 13)
    log_542035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 14), np_542034, 'log')
    # Calling log(args, kwargs) (line 13)
    log_call_result_542046 = invoke(stypy.reporting.localization.Localization(__file__, 13, 14), log_542035, *[sum_call_result_542044], **kwargs_542045)
    
    # Assigning a type to the variable 'desired' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'desired', log_call_result_542046)
    
    # Call to assert_almost_equal(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Call to logsumexp(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'a' (line 14)
    a_542049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 34), 'a', False)
    # Processing the call keyword arguments (line 14)
    kwargs_542050 = {}
    # Getting the type of 'logsumexp' (line 14)
    logsumexp_542048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 24), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 14)
    logsumexp_call_result_542051 = invoke(stypy.reporting.localization.Localization(__file__, 14, 24), logsumexp_542048, *[a_542049], **kwargs_542050)
    
    # Getting the type of 'desired' (line 14)
    desired_542052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 38), 'desired', False)
    # Processing the call keyword arguments (line 14)
    kwargs_542053 = {}
    # Getting the type of 'assert_almost_equal' (line 14)
    assert_almost_equal_542047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 14)
    assert_almost_equal_call_result_542054 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), assert_almost_equal_542047, *[logsumexp_call_result_542051, desired_542052], **kwargs_542053)
    
    
    # Assigning a List to a Name (line 17):
    
    # Assigning a List to a Name (line 17):
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_542055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_542056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 8), list_542055, int_542056)
    # Adding element type (line 17)
    int_542057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 8), list_542055, int_542057)
    
    # Assigning a type to the variable 'b' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'b', list_542055)
    
    # Assigning a BinOp to a Name (line 18):
    
    # Assigning a BinOp to a Name (line 18):
    float_542058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'float')
    
    # Call to log(...): (line 18)
    # Processing the call arguments (line 18)
    float_542061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'float')
    # Processing the call keyword arguments (line 18)
    kwargs_542062 = {}
    # Getting the type of 'np' (line 18)
    np_542059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'np', False)
    # Obtaining the member 'log' of a type (line 18)
    log_542060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 23), np_542059, 'log')
    # Calling log(args, kwargs) (line 18)
    log_call_result_542063 = invoke(stypy.reporting.localization.Localization(__file__, 18, 23), log_542060, *[float_542061], **kwargs_542062)
    
    # Applying the binary operator '+' (line 18)
    result_add_542064 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 14), '+', float_542058, log_call_result_542063)
    
    # Assigning a type to the variable 'desired' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'desired', result_add_542064)
    
    # Call to assert_almost_equal(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Call to logsumexp(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'b' (line 19)
    b_542067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 34), 'b', False)
    # Processing the call keyword arguments (line 19)
    kwargs_542068 = {}
    # Getting the type of 'logsumexp' (line 19)
    logsumexp_542066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 19)
    logsumexp_call_result_542069 = invoke(stypy.reporting.localization.Localization(__file__, 19, 24), logsumexp_542066, *[b_542067], **kwargs_542068)
    
    # Getting the type of 'desired' (line 19)
    desired_542070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 38), 'desired', False)
    # Processing the call keyword arguments (line 19)
    kwargs_542071 = {}
    # Getting the type of 'assert_almost_equal' (line 19)
    assert_almost_equal_542065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 19)
    assert_almost_equal_call_result_542072 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), assert_almost_equal_542065, *[logsumexp_call_result_542069, desired_542070], **kwargs_542071)
    
    
    # Assigning a Num to a Name (line 21):
    
    # Assigning a Num to a Name (line 21):
    int_542073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'int')
    # Assigning a type to the variable 'n' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'n', int_542073)
    
    # Assigning a BinOp to a Name (line 22):
    
    # Assigning a BinOp to a Name (line 22):
    
    # Call to ones(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'n' (line 22)
    n_542076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'n', False)
    # Processing the call keyword arguments (line 22)
    kwargs_542077 = {}
    # Getting the type of 'np' (line 22)
    np_542074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 22)
    ones_542075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), np_542074, 'ones')
    # Calling ones(args, kwargs) (line 22)
    ones_call_result_542078 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), ones_542075, *[n_542076], **kwargs_542077)
    
    int_542079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
    # Applying the binary operator '*' (line 22)
    result_mul_542080 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 8), '*', ones_call_result_542078, int_542079)
    
    # Assigning a type to the variable 'b' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'b', result_mul_542080)
    
    # Assigning a BinOp to a Name (line 23):
    
    # Assigning a BinOp to a Name (line 23):
    float_542081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'float')
    
    # Call to log(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'n' (line 23)
    n_542084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 31), 'n', False)
    # Processing the call keyword arguments (line 23)
    kwargs_542085 = {}
    # Getting the type of 'np' (line 23)
    np_542082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'np', False)
    # Obtaining the member 'log' of a type (line 23)
    log_542083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 24), np_542082, 'log')
    # Calling log(args, kwargs) (line 23)
    log_call_result_542086 = invoke(stypy.reporting.localization.Localization(__file__, 23, 24), log_542083, *[n_542084], **kwargs_542085)
    
    # Applying the binary operator '+' (line 23)
    result_add_542087 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 14), '+', float_542081, log_call_result_542086)
    
    # Assigning a type to the variable 'desired' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'desired', result_add_542087)
    
    # Call to assert_almost_equal(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to logsumexp(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'b' (line 24)
    b_542090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 34), 'b', False)
    # Processing the call keyword arguments (line 24)
    kwargs_542091 = {}
    # Getting the type of 'logsumexp' (line 24)
    logsumexp_542089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 24)
    logsumexp_call_result_542092 = invoke(stypy.reporting.localization.Localization(__file__, 24, 24), logsumexp_542089, *[b_542090], **kwargs_542091)
    
    # Getting the type of 'desired' (line 24)
    desired_542093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 38), 'desired', False)
    # Processing the call keyword arguments (line 24)
    kwargs_542094 = {}
    # Getting the type of 'assert_almost_equal' (line 24)
    assert_almost_equal_542088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 24)
    assert_almost_equal_call_result_542095 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), assert_almost_equal_542088, *[logsumexp_call_result_542092, desired_542093], **kwargs_542094)
    
    
    # Assigning a Call to a Name (line 26):
    
    # Assigning a Call to a Name (line 26):
    
    # Call to array(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_542098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    float_542099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 17), list_542098, float_542099)
    
    int_542100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 27), 'int')
    # Applying the binary operator '*' (line 26)
    result_mul_542101 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 17), '*', list_542098, int_542100)
    
    # Processing the call keyword arguments (line 26)
    kwargs_542102 = {}
    # Getting the type of 'np' (line 26)
    np_542096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 26)
    array_542097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), np_542096, 'array')
    # Calling array(args, kwargs) (line 26)
    array_call_result_542103 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), array_542097, *[result_mul_542101], **kwargs_542102)
    
    # Assigning a type to the variable 'x' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'x', array_call_result_542103)
    
    # Assigning a Call to a Name (line 27):
    
    # Assigning a Call to a Name (line 27):
    
    # Call to log(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'x' (line 27)
    x_542106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'x', False)
    # Processing the call keyword arguments (line 27)
    kwargs_542107 = {}
    # Getting the type of 'np' (line 27)
    np_542104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'np', False)
    # Obtaining the member 'log' of a type (line 27)
    log_542105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 11), np_542104, 'log')
    # Calling log(args, kwargs) (line 27)
    log_call_result_542108 = invoke(stypy.reporting.localization.Localization(__file__, 27, 11), log_542105, *[x_542106], **kwargs_542107)
    
    # Assigning a type to the variable 'logx' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'logx', log_call_result_542108)
    
    # Assigning a Call to a Name (line 29):
    
    # Assigning a Call to a Name (line 29):
    
    # Call to vstack(...): (line 29)
    # Processing the call arguments (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_542111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    # Getting the type of 'x' (line 29)
    x_542112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 18), list_542111, x_542112)
    # Adding element type (line 29)
    # Getting the type of 'x' (line 29)
    x_542113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 18), list_542111, x_542113)
    
    # Processing the call keyword arguments (line 29)
    kwargs_542114 = {}
    # Getting the type of 'np' (line 29)
    np_542109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'np', False)
    # Obtaining the member 'vstack' of a type (line 29)
    vstack_542110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), np_542109, 'vstack')
    # Calling vstack(args, kwargs) (line 29)
    vstack_call_result_542115 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), vstack_542110, *[list_542111], **kwargs_542114)
    
    # Assigning a type to the variable 'X' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'X', vstack_call_result_542115)
    
    # Assigning a Call to a Name (line 30):
    
    # Assigning a Call to a Name (line 30):
    
    # Call to vstack(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_542118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    # Getting the type of 'logx' (line 30)
    logx_542119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'logx', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), list_542118, logx_542119)
    # Adding element type (line 30)
    # Getting the type of 'logx' (line 30)
    logx_542120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 28), 'logx', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), list_542118, logx_542120)
    
    # Processing the call keyword arguments (line 30)
    kwargs_542121 = {}
    # Getting the type of 'np' (line 30)
    np_542116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'np', False)
    # Obtaining the member 'vstack' of a type (line 30)
    vstack_542117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 11), np_542116, 'vstack')
    # Calling vstack(args, kwargs) (line 30)
    vstack_call_result_542122 = invoke(stypy.reporting.localization.Localization(__file__, 30, 11), vstack_542117, *[list_542118], **kwargs_542121)
    
    # Assigning a type to the variable 'logX' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'logX', vstack_call_result_542122)
    
    # Call to assert_array_almost_equal(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to exp(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to logsumexp(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'logX' (line 31)
    logX_542127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 47), 'logX', False)
    # Processing the call keyword arguments (line 31)
    kwargs_542128 = {}
    # Getting the type of 'logsumexp' (line 31)
    logsumexp_542126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 31)
    logsumexp_call_result_542129 = invoke(stypy.reporting.localization.Localization(__file__, 31, 37), logsumexp_542126, *[logX_542127], **kwargs_542128)
    
    # Processing the call keyword arguments (line 31)
    kwargs_542130 = {}
    # Getting the type of 'np' (line 31)
    np_542124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'np', False)
    # Obtaining the member 'exp' of a type (line 31)
    exp_542125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), np_542124, 'exp')
    # Calling exp(args, kwargs) (line 31)
    exp_call_result_542131 = invoke(stypy.reporting.localization.Localization(__file__, 31, 30), exp_542125, *[logsumexp_call_result_542129], **kwargs_542130)
    
    
    # Call to sum(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_542134 = {}
    # Getting the type of 'X' (line 31)
    X_542132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 55), 'X', False)
    # Obtaining the member 'sum' of a type (line 31)
    sum_542133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 55), X_542132, 'sum')
    # Calling sum(args, kwargs) (line 31)
    sum_call_result_542135 = invoke(stypy.reporting.localization.Localization(__file__, 31, 55), sum_542133, *[], **kwargs_542134)
    
    # Processing the call keyword arguments (line 31)
    kwargs_542136 = {}
    # Getting the type of 'assert_array_almost_equal' (line 31)
    assert_array_almost_equal_542123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 31)
    assert_array_almost_equal_call_result_542137 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), assert_array_almost_equal_542123, *[exp_call_result_542131, sum_call_result_542135], **kwargs_542136)
    
    
    # Call to assert_array_almost_equal(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to exp(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to logsumexp(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'logX' (line 32)
    logX_542142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 47), 'logX', False)
    # Processing the call keyword arguments (line 32)
    int_542143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 58), 'int')
    keyword_542144 = int_542143
    kwargs_542145 = {'axis': keyword_542144}
    # Getting the type of 'logsumexp' (line 32)
    logsumexp_542141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 37), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 32)
    logsumexp_call_result_542146 = invoke(stypy.reporting.localization.Localization(__file__, 32, 37), logsumexp_542141, *[logX_542142], **kwargs_542145)
    
    # Processing the call keyword arguments (line 32)
    kwargs_542147 = {}
    # Getting the type of 'np' (line 32)
    np_542139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 30), 'np', False)
    # Obtaining the member 'exp' of a type (line 32)
    exp_542140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 30), np_542139, 'exp')
    # Calling exp(args, kwargs) (line 32)
    exp_call_result_542148 = invoke(stypy.reporting.localization.Localization(__file__, 32, 30), exp_542140, *[logsumexp_call_result_542146], **kwargs_542147)
    
    
    # Call to sum(...): (line 32)
    # Processing the call keyword arguments (line 32)
    int_542151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 74), 'int')
    keyword_542152 = int_542151
    kwargs_542153 = {'axis': keyword_542152}
    # Getting the type of 'X' (line 32)
    X_542149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 63), 'X', False)
    # Obtaining the member 'sum' of a type (line 32)
    sum_542150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 63), X_542149, 'sum')
    # Calling sum(args, kwargs) (line 32)
    sum_call_result_542154 = invoke(stypy.reporting.localization.Localization(__file__, 32, 63), sum_542150, *[], **kwargs_542153)
    
    # Processing the call keyword arguments (line 32)
    kwargs_542155 = {}
    # Getting the type of 'assert_array_almost_equal' (line 32)
    assert_array_almost_equal_542138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 32)
    assert_array_almost_equal_call_result_542156 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), assert_array_almost_equal_542138, *[exp_call_result_542148, sum_call_result_542154], **kwargs_542155)
    
    
    # Call to assert_array_almost_equal(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Call to exp(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Call to logsumexp(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'logX' (line 33)
    logX_542161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 47), 'logX', False)
    # Processing the call keyword arguments (line 33)
    int_542162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 58), 'int')
    keyword_542163 = int_542162
    kwargs_542164 = {'axis': keyword_542163}
    # Getting the type of 'logsumexp' (line 33)
    logsumexp_542160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 37), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 33)
    logsumexp_call_result_542165 = invoke(stypy.reporting.localization.Localization(__file__, 33, 37), logsumexp_542160, *[logX_542161], **kwargs_542164)
    
    # Processing the call keyword arguments (line 33)
    kwargs_542166 = {}
    # Getting the type of 'np' (line 33)
    np_542158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 30), 'np', False)
    # Obtaining the member 'exp' of a type (line 33)
    exp_542159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 30), np_542158, 'exp')
    # Calling exp(args, kwargs) (line 33)
    exp_call_result_542167 = invoke(stypy.reporting.localization.Localization(__file__, 33, 30), exp_542159, *[logsumexp_call_result_542165], **kwargs_542166)
    
    
    # Call to sum(...): (line 33)
    # Processing the call keyword arguments (line 33)
    int_542170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 74), 'int')
    keyword_542171 = int_542170
    kwargs_542172 = {'axis': keyword_542171}
    # Getting the type of 'X' (line 33)
    X_542168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 63), 'X', False)
    # Obtaining the member 'sum' of a type (line 33)
    sum_542169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 63), X_542168, 'sum')
    # Calling sum(args, kwargs) (line 33)
    sum_call_result_542173 = invoke(stypy.reporting.localization.Localization(__file__, 33, 63), sum_542169, *[], **kwargs_542172)
    
    # Processing the call keyword arguments (line 33)
    kwargs_542174 = {}
    # Getting the type of 'assert_array_almost_equal' (line 33)
    assert_array_almost_equal_542157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 33)
    assert_array_almost_equal_call_result_542175 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), assert_array_almost_equal_542157, *[exp_call_result_542167, sum_call_result_542173], **kwargs_542174)
    
    
    # Call to assert_equal(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Call to logsumexp(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'np' (line 36)
    np_542178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'np', False)
    # Obtaining the member 'inf' of a type (line 36)
    inf_542179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 27), np_542178, 'inf')
    # Processing the call keyword arguments (line 36)
    kwargs_542180 = {}
    # Getting the type of 'logsumexp' (line 36)
    logsumexp_542177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 36)
    logsumexp_call_result_542181 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), logsumexp_542177, *[inf_542179], **kwargs_542180)
    
    # Getting the type of 'np' (line 36)
    np_542182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 36), 'np', False)
    # Obtaining the member 'inf' of a type (line 36)
    inf_542183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 36), np_542182, 'inf')
    # Processing the call keyword arguments (line 36)
    kwargs_542184 = {}
    # Getting the type of 'assert_equal' (line 36)
    assert_equal_542176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 36)
    assert_equal_call_result_542185 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), assert_equal_542176, *[logsumexp_call_result_542181, inf_542183], **kwargs_542184)
    
    
    # Call to assert_equal(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Call to logsumexp(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Getting the type of 'np' (line 37)
    np_542188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 28), 'np', False)
    # Obtaining the member 'inf' of a type (line 37)
    inf_542189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 28), np_542188, 'inf')
    # Applying the 'usub' unary operator (line 37)
    result___neg___542190 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 27), 'usub', inf_542189)
    
    # Processing the call keyword arguments (line 37)
    kwargs_542191 = {}
    # Getting the type of 'logsumexp' (line 37)
    logsumexp_542187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 37)
    logsumexp_call_result_542192 = invoke(stypy.reporting.localization.Localization(__file__, 37, 17), logsumexp_542187, *[result___neg___542190], **kwargs_542191)
    
    
    # Getting the type of 'np' (line 37)
    np_542193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 38), 'np', False)
    # Obtaining the member 'inf' of a type (line 37)
    inf_542194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 38), np_542193, 'inf')
    # Applying the 'usub' unary operator (line 37)
    result___neg___542195 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 37), 'usub', inf_542194)
    
    # Processing the call keyword arguments (line 37)
    kwargs_542196 = {}
    # Getting the type of 'assert_equal' (line 37)
    assert_equal_542186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 37)
    assert_equal_call_result_542197 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), assert_equal_542186, *[logsumexp_call_result_542192, result___neg___542195], **kwargs_542196)
    
    
    # Call to assert_equal(...): (line 38)
    # Processing the call arguments (line 38)
    
    # Call to logsumexp(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'np' (line 38)
    np_542200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'np', False)
    # Obtaining the member 'nan' of a type (line 38)
    nan_542201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 27), np_542200, 'nan')
    # Processing the call keyword arguments (line 38)
    kwargs_542202 = {}
    # Getting the type of 'logsumexp' (line 38)
    logsumexp_542199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 38)
    logsumexp_call_result_542203 = invoke(stypy.reporting.localization.Localization(__file__, 38, 17), logsumexp_542199, *[nan_542201], **kwargs_542202)
    
    # Getting the type of 'np' (line 38)
    np_542204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 36), 'np', False)
    # Obtaining the member 'nan' of a type (line 38)
    nan_542205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 36), np_542204, 'nan')
    # Processing the call keyword arguments (line 38)
    kwargs_542206 = {}
    # Getting the type of 'assert_equal' (line 38)
    assert_equal_542198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 38)
    assert_equal_call_result_542207 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), assert_equal_542198, *[logsumexp_call_result_542203, nan_542205], **kwargs_542206)
    
    
    # Call to assert_equal(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Call to logsumexp(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_542210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    
    # Getting the type of 'np' (line 39)
    np_542211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'np', False)
    # Obtaining the member 'inf' of a type (line 39)
    inf_542212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 29), np_542211, 'inf')
    # Applying the 'usub' unary operator (line 39)
    result___neg___542213 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 28), 'usub', inf_542212)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 27), list_542210, result___neg___542213)
    # Adding element type (line 39)
    
    # Getting the type of 'np' (line 39)
    np_542214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 38), 'np', False)
    # Obtaining the member 'inf' of a type (line 39)
    inf_542215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 38), np_542214, 'inf')
    # Applying the 'usub' unary operator (line 39)
    result___neg___542216 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 37), 'usub', inf_542215)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 27), list_542210, result___neg___542216)
    
    # Processing the call keyword arguments (line 39)
    kwargs_542217 = {}
    # Getting the type of 'logsumexp' (line 39)
    logsumexp_542209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 39)
    logsumexp_call_result_542218 = invoke(stypy.reporting.localization.Localization(__file__, 39, 17), logsumexp_542209, *[list_542210], **kwargs_542217)
    
    
    # Getting the type of 'np' (line 39)
    np_542219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 49), 'np', False)
    # Obtaining the member 'inf' of a type (line 39)
    inf_542220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 49), np_542219, 'inf')
    # Applying the 'usub' unary operator (line 39)
    result___neg___542221 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 48), 'usub', inf_542220)
    
    # Processing the call keyword arguments (line 39)
    kwargs_542222 = {}
    # Getting the type of 'assert_equal' (line 39)
    assert_equal_542208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 39)
    assert_equal_call_result_542223 = invoke(stypy.reporting.localization.Localization(__file__, 39, 4), assert_equal_542208, *[logsumexp_call_result_542218, result___neg___542221], **kwargs_542222)
    
    
    # Call to assert_array_almost_equal(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Call to logsumexp(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Obtaining an instance of the builtin type 'list' (line 42)
    list_542226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 42)
    # Adding element type (line 42)
    
    # Obtaining an instance of the builtin type 'list' (line 42)
    list_542227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 42)
    # Adding element type (line 42)
    float_542228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 42), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 41), list_542227, float_542228)
    # Adding element type (line 42)
    float_542229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 48), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 41), list_542227, float_542229)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 40), list_542226, list_542227)
    # Adding element type (line 42)
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_542230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    # Adding element type (line 43)
    float_542231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 42), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 41), list_542230, float_542231)
    # Adding element type (line 43)
    
    # Getting the type of 'np' (line 43)
    np_542232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 50), 'np', False)
    # Obtaining the member 'inf' of a type (line 43)
    inf_542233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 50), np_542232, 'inf')
    # Applying the 'usub' unary operator (line 43)
    result___neg___542234 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 49), 'usub', inf_542233)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 41), list_542230, result___neg___542234)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 40), list_542226, list_542230)
    
    # Processing the call keyword arguments (line 42)
    int_542235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 65), 'int')
    keyword_542236 = int_542235
    kwargs_542237 = {'axis': keyword_542236}
    # Getting the type of 'logsumexp' (line 42)
    logsumexp_542225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 30), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 42)
    logsumexp_call_result_542238 = invoke(stypy.reporting.localization.Localization(__file__, 42, 30), logsumexp_542225, *[list_542226], **kwargs_542237)
    
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_542239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    float_542240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 30), list_542239, float_542240)
    # Adding element type (line 44)
    float_542241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 37), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 30), list_542239, float_542241)
    
    # Processing the call keyword arguments (line 42)
    kwargs_542242 = {}
    # Getting the type of 'assert_array_almost_equal' (line 42)
    assert_array_almost_equal_542224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 42)
    assert_array_almost_equal_call_result_542243 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), assert_array_almost_equal_542224, *[logsumexp_call_result_542238, list_542239], **kwargs_542242)
    
    
    # Call to assert_array_almost_equal(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Call to logsumexp(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Obtaining an instance of the builtin type 'list' (line 47)
    list_542246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 47)
    # Adding element type (line 47)
    
    # Obtaining an instance of the builtin type 'list' (line 47)
    list_542247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 47)
    # Adding element type (line 47)
    float_542248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 42), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 41), list_542247, float_542248)
    # Adding element type (line 47)
    float_542249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 48), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 41), list_542247, float_542249)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 40), list_542246, list_542247)
    # Adding element type (line 47)
    
    # Obtaining an instance of the builtin type 'list' (line 48)
    list_542250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 48)
    # Adding element type (line 48)
    float_542251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 42), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 41), list_542250, float_542251)
    # Adding element type (line 48)
    
    # Getting the type of 'np' (line 48)
    np_542252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 50), 'np', False)
    # Obtaining the member 'inf' of a type (line 48)
    inf_542253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 50), np_542252, 'inf')
    # Applying the 'usub' unary operator (line 48)
    result___neg___542254 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 49), 'usub', inf_542253)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 41), list_542250, result___neg___542254)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 40), list_542246, list_542250)
    
    # Processing the call keyword arguments (line 47)
    int_542255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 45), 'int')
    keyword_542256 = int_542255
    # Getting the type of 'True' (line 50)
    True_542257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 49), 'True', False)
    keyword_542258 = True_542257
    kwargs_542259 = {'keepdims': keyword_542258, 'axis': keyword_542256}
    # Getting the type of 'logsumexp' (line 47)
    logsumexp_542245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 47)
    logsumexp_call_result_542260 = invoke(stypy.reporting.localization.Localization(__file__, 47, 30), logsumexp_542245, *[list_542246], **kwargs_542259)
    
    
    # Obtaining an instance of the builtin type 'list' (line 51)
    list_542261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 51)
    # Adding element type (line 51)
    
    # Obtaining an instance of the builtin type 'list' (line 51)
    list_542262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 51)
    # Adding element type (line 51)
    float_542263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 32), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 31), list_542262, float_542263)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 30), list_542261, list_542262)
    # Adding element type (line 51)
    
    # Obtaining an instance of the builtin type 'list' (line 51)
    list_542264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 51)
    # Adding element type (line 51)
    float_542265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 39), list_542264, float_542265)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 30), list_542261, list_542264)
    
    # Processing the call keyword arguments (line 47)
    kwargs_542266 = {}
    # Getting the type of 'assert_array_almost_equal' (line 47)
    assert_array_almost_equal_542244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 47)
    assert_array_almost_equal_call_result_542267 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), assert_array_almost_equal_542244, *[logsumexp_call_result_542260, list_542261], **kwargs_542266)
    
    
    # Call to assert_array_almost_equal(...): (line 54)
    # Processing the call arguments (line 54)
    
    # Call to logsumexp(...): (line 54)
    # Processing the call arguments (line 54)
    
    # Obtaining an instance of the builtin type 'list' (line 54)
    list_542270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 54)
    # Adding element type (line 54)
    
    # Obtaining an instance of the builtin type 'list' (line 54)
    list_542271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 54)
    # Adding element type (line 54)
    float_542272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 42), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 41), list_542271, float_542272)
    # Adding element type (line 54)
    float_542273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 48), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 41), list_542271, float_542273)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 40), list_542270, list_542271)
    # Adding element type (line 54)
    
    # Obtaining an instance of the builtin type 'list' (line 55)
    list_542274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 55)
    # Adding element type (line 55)
    float_542275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 42), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 41), list_542274, float_542275)
    # Adding element type (line 55)
    
    # Getting the type of 'np' (line 55)
    np_542276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 50), 'np', False)
    # Obtaining the member 'inf' of a type (line 55)
    inf_542277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 50), np_542276, 'inf')
    # Applying the 'usub' unary operator (line 55)
    result___neg___542278 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 49), 'usub', inf_542277)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 41), list_542274, result___neg___542278)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 40), list_542270, list_542274)
    
    # Processing the call keyword arguments (line 54)
    
    # Obtaining an instance of the builtin type 'tuple' (line 56)
    tuple_542279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 56)
    # Adding element type (line 56)
    int_542280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 46), tuple_542279, int_542280)
    # Adding element type (line 56)
    int_542281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 46), tuple_542279, int_542281)
    
    keyword_542282 = tuple_542279
    kwargs_542283 = {'axis': keyword_542282}
    # Getting the type of 'logsumexp' (line 54)
    logsumexp_542269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 30), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 54)
    logsumexp_call_result_542284 = invoke(stypy.reporting.localization.Localization(__file__, 54, 30), logsumexp_542269, *[list_542270], **kwargs_542283)
    
    float_542285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 30), 'float')
    # Processing the call keyword arguments (line 54)
    kwargs_542286 = {}
    # Getting the type of 'assert_array_almost_equal' (line 54)
    assert_array_almost_equal_542268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 54)
    assert_array_almost_equal_call_result_542287 = invoke(stypy.reporting.localization.Localization(__file__, 54, 4), assert_array_almost_equal_542268, *[logsumexp_call_result_542284, float_542285], **kwargs_542286)
    
    
    # ################# End of 'test_logsumexp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_logsumexp' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_542288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_542288)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_logsumexp'
    return stypy_return_type_542288

# Assigning a type to the variable 'test_logsumexp' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'test_logsumexp', test_logsumexp)

@norecursion
def test_logsumexp_b(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_logsumexp_b'
    module_type_store = module_type_store.open_function_context('test_logsumexp_b', 60, 0, False)
    
    # Passed parameters checking function
    test_logsumexp_b.stypy_localization = localization
    test_logsumexp_b.stypy_type_of_self = None
    test_logsumexp_b.stypy_type_store = module_type_store
    test_logsumexp_b.stypy_function_name = 'test_logsumexp_b'
    test_logsumexp_b.stypy_param_names_list = []
    test_logsumexp_b.stypy_varargs_param_name = None
    test_logsumexp_b.stypy_kwargs_param_name = None
    test_logsumexp_b.stypy_call_defaults = defaults
    test_logsumexp_b.stypy_call_varargs = varargs
    test_logsumexp_b.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_logsumexp_b', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_logsumexp_b', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_logsumexp_b(...)' code ##################

    
    # Assigning a Call to a Name (line 61):
    
    # Assigning a Call to a Name (line 61):
    
    # Call to arange(...): (line 61)
    # Processing the call arguments (line 61)
    int_542291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 18), 'int')
    # Processing the call keyword arguments (line 61)
    kwargs_542292 = {}
    # Getting the type of 'np' (line 61)
    np_542289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 61)
    arange_542290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), np_542289, 'arange')
    # Calling arange(args, kwargs) (line 61)
    arange_call_result_542293 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), arange_542290, *[int_542291], **kwargs_542292)
    
    # Assigning a type to the variable 'a' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'a', arange_call_result_542293)
    
    # Assigning a Call to a Name (line 62):
    
    # Assigning a Call to a Name (line 62):
    
    # Call to arange(...): (line 62)
    # Processing the call arguments (line 62)
    int_542296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 18), 'int')
    int_542297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'int')
    int_542298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 26), 'int')
    # Processing the call keyword arguments (line 62)
    kwargs_542299 = {}
    # Getting the type of 'np' (line 62)
    np_542294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 62)
    arange_542295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), np_542294, 'arange')
    # Calling arange(args, kwargs) (line 62)
    arange_call_result_542300 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), arange_542295, *[int_542296, int_542297, int_542298], **kwargs_542299)
    
    # Assigning a type to the variable 'b' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'b', arange_call_result_542300)
    
    # Assigning a Call to a Name (line 63):
    
    # Assigning a Call to a Name (line 63):
    
    # Call to log(...): (line 63)
    # Processing the call arguments (line 63)
    
    # Call to sum(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'b' (line 63)
    b_542305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'b', False)
    
    # Call to exp(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'a' (line 63)
    a_542308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'a', False)
    # Processing the call keyword arguments (line 63)
    kwargs_542309 = {}
    # Getting the type of 'np' (line 63)
    np_542306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'np', False)
    # Obtaining the member 'exp' of a type (line 63)
    exp_542307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 30), np_542306, 'exp')
    # Calling exp(args, kwargs) (line 63)
    exp_call_result_542310 = invoke(stypy.reporting.localization.Localization(__file__, 63, 30), exp_542307, *[a_542308], **kwargs_542309)
    
    # Applying the binary operator '*' (line 63)
    result_mul_542311 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 28), '*', b_542305, exp_call_result_542310)
    
    # Processing the call keyword arguments (line 63)
    kwargs_542312 = {}
    # Getting the type of 'np' (line 63)
    np_542303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 21), 'np', False)
    # Obtaining the member 'sum' of a type (line 63)
    sum_542304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 21), np_542303, 'sum')
    # Calling sum(args, kwargs) (line 63)
    sum_call_result_542313 = invoke(stypy.reporting.localization.Localization(__file__, 63, 21), sum_542304, *[result_mul_542311], **kwargs_542312)
    
    # Processing the call keyword arguments (line 63)
    kwargs_542314 = {}
    # Getting the type of 'np' (line 63)
    np_542301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 14), 'np', False)
    # Obtaining the member 'log' of a type (line 63)
    log_542302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 14), np_542301, 'log')
    # Calling log(args, kwargs) (line 63)
    log_call_result_542315 = invoke(stypy.reporting.localization.Localization(__file__, 63, 14), log_542302, *[sum_call_result_542313], **kwargs_542314)
    
    # Assigning a type to the variable 'desired' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'desired', log_call_result_542315)
    
    # Call to assert_almost_equal(...): (line 64)
    # Processing the call arguments (line 64)
    
    # Call to logsumexp(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'a' (line 64)
    a_542318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'a', False)
    # Processing the call keyword arguments (line 64)
    # Getting the type of 'b' (line 64)
    b_542319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 39), 'b', False)
    keyword_542320 = b_542319
    kwargs_542321 = {'b': keyword_542320}
    # Getting the type of 'logsumexp' (line 64)
    logsumexp_542317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 64)
    logsumexp_call_result_542322 = invoke(stypy.reporting.localization.Localization(__file__, 64, 24), logsumexp_542317, *[a_542318], **kwargs_542321)
    
    # Getting the type of 'desired' (line 64)
    desired_542323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 43), 'desired', False)
    # Processing the call keyword arguments (line 64)
    kwargs_542324 = {}
    # Getting the type of 'assert_almost_equal' (line 64)
    assert_almost_equal_542316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 64)
    assert_almost_equal_call_result_542325 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), assert_almost_equal_542316, *[logsumexp_call_result_542322, desired_542323], **kwargs_542324)
    
    
    # Assigning a List to a Name (line 66):
    
    # Assigning a List to a Name (line 66):
    
    # Obtaining an instance of the builtin type 'list' (line 66)
    list_542326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 66)
    # Adding element type (line 66)
    int_542327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 8), list_542326, int_542327)
    # Adding element type (line 66)
    int_542328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 8), list_542326, int_542328)
    
    # Assigning a type to the variable 'a' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'a', list_542326)
    
    # Assigning a List to a Name (line 67):
    
    # Assigning a List to a Name (line 67):
    
    # Obtaining an instance of the builtin type 'list' (line 67)
    list_542329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 67)
    # Adding element type (line 67)
    float_542330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), list_542329, float_542330)
    # Adding element type (line 67)
    float_542331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 14), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), list_542329, float_542331)
    
    # Assigning a type to the variable 'b' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'b', list_542329)
    
    # Assigning a BinOp to a Name (line 68):
    
    # Assigning a BinOp to a Name (line 68):
    int_542332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 14), 'int')
    
    # Call to log(...): (line 68)
    # Processing the call arguments (line 68)
    int_542335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'int')
    float_542336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 32), 'float')
    # Applying the binary operator '*' (line 68)
    result_mul_542337 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 28), '*', int_542335, float_542336)
    
    # Processing the call keyword arguments (line 68)
    kwargs_542338 = {}
    # Getting the type of 'np' (line 68)
    np_542333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'np', False)
    # Obtaining the member 'log' of a type (line 68)
    log_542334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 21), np_542333, 'log')
    # Calling log(args, kwargs) (line 68)
    log_call_result_542339 = invoke(stypy.reporting.localization.Localization(__file__, 68, 21), log_542334, *[result_mul_542337], **kwargs_542338)
    
    # Applying the binary operator '+' (line 68)
    result_add_542340 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 14), '+', int_542332, log_call_result_542339)
    
    # Assigning a type to the variable 'desired' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'desired', result_add_542340)
    
    # Call to assert_almost_equal(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Call to logsumexp(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'a' (line 69)
    a_542343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 34), 'a', False)
    # Processing the call keyword arguments (line 69)
    # Getting the type of 'b' (line 69)
    b_542344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 39), 'b', False)
    keyword_542345 = b_542344
    kwargs_542346 = {'b': keyword_542345}
    # Getting the type of 'logsumexp' (line 69)
    logsumexp_542342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 69)
    logsumexp_call_result_542347 = invoke(stypy.reporting.localization.Localization(__file__, 69, 24), logsumexp_542342, *[a_542343], **kwargs_542346)
    
    # Getting the type of 'desired' (line 69)
    desired_542348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), 'desired', False)
    # Processing the call keyword arguments (line 69)
    kwargs_542349 = {}
    # Getting the type of 'assert_almost_equal' (line 69)
    assert_almost_equal_542341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 69)
    assert_almost_equal_call_result_542350 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), assert_almost_equal_542341, *[logsumexp_call_result_542347, desired_542348], **kwargs_542349)
    
    
    # Assigning a Call to a Name (line 71):
    
    # Assigning a Call to a Name (line 71):
    
    # Call to array(...): (line 71)
    # Processing the call arguments (line 71)
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_542353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    # Adding element type (line 71)
    float_542354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 17), list_542353, float_542354)
    
    int_542355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 27), 'int')
    # Applying the binary operator '*' (line 71)
    result_mul_542356 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 17), '*', list_542353, int_542355)
    
    # Processing the call keyword arguments (line 71)
    kwargs_542357 = {}
    # Getting the type of 'np' (line 71)
    np_542351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 71)
    array_542352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), np_542351, 'array')
    # Calling array(args, kwargs) (line 71)
    array_call_result_542358 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), array_542352, *[result_mul_542356], **kwargs_542357)
    
    # Assigning a type to the variable 'x' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'x', array_call_result_542358)
    
    # Assigning a Call to a Name (line 72):
    
    # Assigning a Call to a Name (line 72):
    
    # Call to linspace(...): (line 72)
    # Processing the call arguments (line 72)
    int_542361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 20), 'int')
    int_542362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 23), 'int')
    int_542363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 29), 'int')
    # Processing the call keyword arguments (line 72)
    kwargs_542364 = {}
    # Getting the type of 'np' (line 72)
    np_542359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 72)
    linspace_542360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), np_542359, 'linspace')
    # Calling linspace(args, kwargs) (line 72)
    linspace_call_result_542365 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), linspace_542360, *[int_542361, int_542362, int_542363], **kwargs_542364)
    
    # Assigning a type to the variable 'b' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'b', linspace_call_result_542365)
    
    # Assigning a Call to a Name (line 73):
    
    # Assigning a Call to a Name (line 73):
    
    # Call to log(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'x' (line 73)
    x_542368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'x', False)
    # Processing the call keyword arguments (line 73)
    kwargs_542369 = {}
    # Getting the type of 'np' (line 73)
    np_542366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'np', False)
    # Obtaining the member 'log' of a type (line 73)
    log_542367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), np_542366, 'log')
    # Calling log(args, kwargs) (line 73)
    log_call_result_542370 = invoke(stypy.reporting.localization.Localization(__file__, 73, 11), log_542367, *[x_542368], **kwargs_542369)
    
    # Assigning a type to the variable 'logx' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'logx', log_call_result_542370)
    
    # Assigning a Call to a Name (line 75):
    
    # Assigning a Call to a Name (line 75):
    
    # Call to vstack(...): (line 75)
    # Processing the call arguments (line 75)
    
    # Obtaining an instance of the builtin type 'tuple' (line 75)
    tuple_542373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 75)
    # Adding element type (line 75)
    # Getting the type of 'x' (line 75)
    x_542374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 19), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 19), tuple_542373, x_542374)
    # Adding element type (line 75)
    # Getting the type of 'x' (line 75)
    x_542375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 19), tuple_542373, x_542375)
    
    # Processing the call keyword arguments (line 75)
    kwargs_542376 = {}
    # Getting the type of 'np' (line 75)
    np_542371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'np', False)
    # Obtaining the member 'vstack' of a type (line 75)
    vstack_542372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), np_542371, 'vstack')
    # Calling vstack(args, kwargs) (line 75)
    vstack_call_result_542377 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), vstack_542372, *[tuple_542373], **kwargs_542376)
    
    # Assigning a type to the variable 'X' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'X', vstack_call_result_542377)
    
    # Assigning a Call to a Name (line 76):
    
    # Assigning a Call to a Name (line 76):
    
    # Call to vstack(...): (line 76)
    # Processing the call arguments (line 76)
    
    # Obtaining an instance of the builtin type 'tuple' (line 76)
    tuple_542380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 76)
    # Adding element type (line 76)
    # Getting the type of 'logx' (line 76)
    logx_542381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 22), 'logx', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 22), tuple_542380, logx_542381)
    # Adding element type (line 76)
    # Getting the type of 'logx' (line 76)
    logx_542382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 28), 'logx', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 22), tuple_542380, logx_542382)
    
    # Processing the call keyword arguments (line 76)
    kwargs_542383 = {}
    # Getting the type of 'np' (line 76)
    np_542378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'np', False)
    # Obtaining the member 'vstack' of a type (line 76)
    vstack_542379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 11), np_542378, 'vstack')
    # Calling vstack(args, kwargs) (line 76)
    vstack_call_result_542384 = invoke(stypy.reporting.localization.Localization(__file__, 76, 11), vstack_542379, *[tuple_542380], **kwargs_542383)
    
    # Assigning a type to the variable 'logX' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'logX', vstack_call_result_542384)
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to vstack(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Obtaining an instance of the builtin type 'tuple' (line 77)
    tuple_542387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 77)
    # Adding element type (line 77)
    # Getting the type of 'b' (line 77)
    b_542388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 19), tuple_542387, b_542388)
    # Adding element type (line 77)
    # Getting the type of 'b' (line 77)
    b_542389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 19), tuple_542387, b_542389)
    
    # Processing the call keyword arguments (line 77)
    kwargs_542390 = {}
    # Getting the type of 'np' (line 77)
    np_542385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'np', False)
    # Obtaining the member 'vstack' of a type (line 77)
    vstack_542386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), np_542385, 'vstack')
    # Calling vstack(args, kwargs) (line 77)
    vstack_call_result_542391 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), vstack_542386, *[tuple_542387], **kwargs_542390)
    
    # Assigning a type to the variable 'B' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'B', vstack_call_result_542391)
    
    # Call to assert_array_almost_equal(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Call to exp(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Call to logsumexp(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'logX' (line 78)
    logX_542396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 47), 'logX', False)
    # Processing the call keyword arguments (line 78)
    # Getting the type of 'B' (line 78)
    B_542397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 55), 'B', False)
    keyword_542398 = B_542397
    kwargs_542399 = {'b': keyword_542398}
    # Getting the type of 'logsumexp' (line 78)
    logsumexp_542395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 37), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 78)
    logsumexp_call_result_542400 = invoke(stypy.reporting.localization.Localization(__file__, 78, 37), logsumexp_542395, *[logX_542396], **kwargs_542399)
    
    # Processing the call keyword arguments (line 78)
    kwargs_542401 = {}
    # Getting the type of 'np' (line 78)
    np_542393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'np', False)
    # Obtaining the member 'exp' of a type (line 78)
    exp_542394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 30), np_542393, 'exp')
    # Calling exp(args, kwargs) (line 78)
    exp_call_result_542402 = invoke(stypy.reporting.localization.Localization(__file__, 78, 30), exp_542394, *[logsumexp_call_result_542400], **kwargs_542401)
    
    
    # Call to sum(...): (line 78)
    # Processing the call keyword arguments (line 78)
    kwargs_542407 = {}
    # Getting the type of 'B' (line 78)
    B_542403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 61), 'B', False)
    # Getting the type of 'X' (line 78)
    X_542404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 65), 'X', False)
    # Applying the binary operator '*' (line 78)
    result_mul_542405 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 61), '*', B_542403, X_542404)
    
    # Obtaining the member 'sum' of a type (line 78)
    sum_542406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 61), result_mul_542405, 'sum')
    # Calling sum(args, kwargs) (line 78)
    sum_call_result_542408 = invoke(stypy.reporting.localization.Localization(__file__, 78, 61), sum_542406, *[], **kwargs_542407)
    
    # Processing the call keyword arguments (line 78)
    kwargs_542409 = {}
    # Getting the type of 'assert_array_almost_equal' (line 78)
    assert_array_almost_equal_542392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 78)
    assert_array_almost_equal_call_result_542410 = invoke(stypy.reporting.localization.Localization(__file__, 78, 4), assert_array_almost_equal_542392, *[exp_call_result_542402, sum_call_result_542408], **kwargs_542409)
    
    
    # Call to assert_array_almost_equal(...): (line 79)
    # Processing the call arguments (line 79)
    
    # Call to exp(...): (line 79)
    # Processing the call arguments (line 79)
    
    # Call to logsumexp(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'logX' (line 79)
    logX_542415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 47), 'logX', False)
    # Processing the call keyword arguments (line 79)
    # Getting the type of 'B' (line 79)
    B_542416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 55), 'B', False)
    keyword_542417 = B_542416
    int_542418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 63), 'int')
    keyword_542419 = int_542418
    kwargs_542420 = {'b': keyword_542417, 'axis': keyword_542419}
    # Getting the type of 'logsumexp' (line 79)
    logsumexp_542414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 37), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 79)
    logsumexp_call_result_542421 = invoke(stypy.reporting.localization.Localization(__file__, 79, 37), logsumexp_542414, *[logX_542415], **kwargs_542420)
    
    # Processing the call keyword arguments (line 79)
    kwargs_542422 = {}
    # Getting the type of 'np' (line 79)
    np_542412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'np', False)
    # Obtaining the member 'exp' of a type (line 79)
    exp_542413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 30), np_542412, 'exp')
    # Calling exp(args, kwargs) (line 79)
    exp_call_result_542423 = invoke(stypy.reporting.localization.Localization(__file__, 79, 30), exp_542413, *[logsumexp_call_result_542421], **kwargs_542422)
    
    
    # Call to sum(...): (line 80)
    # Processing the call keyword arguments (line 80)
    int_542428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 49), 'int')
    keyword_542429 = int_542428
    kwargs_542430 = {'axis': keyword_542429}
    # Getting the type of 'B' (line 80)
    B_542424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 33), 'B', False)
    # Getting the type of 'X' (line 80)
    X_542425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 37), 'X', False)
    # Applying the binary operator '*' (line 80)
    result_mul_542426 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 33), '*', B_542424, X_542425)
    
    # Obtaining the member 'sum' of a type (line 80)
    sum_542427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 33), result_mul_542426, 'sum')
    # Calling sum(args, kwargs) (line 80)
    sum_call_result_542431 = invoke(stypy.reporting.localization.Localization(__file__, 80, 33), sum_542427, *[], **kwargs_542430)
    
    # Processing the call keyword arguments (line 79)
    kwargs_542432 = {}
    # Getting the type of 'assert_array_almost_equal' (line 79)
    assert_array_almost_equal_542411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 79)
    assert_array_almost_equal_call_result_542433 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), assert_array_almost_equal_542411, *[exp_call_result_542423, sum_call_result_542431], **kwargs_542432)
    
    
    # Call to assert_array_almost_equal(...): (line 81)
    # Processing the call arguments (line 81)
    
    # Call to exp(...): (line 81)
    # Processing the call arguments (line 81)
    
    # Call to logsumexp(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'logX' (line 81)
    logX_542438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 47), 'logX', False)
    # Processing the call keyword arguments (line 81)
    # Getting the type of 'B' (line 81)
    B_542439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 55), 'B', False)
    keyword_542440 = B_542439
    int_542441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 63), 'int')
    keyword_542442 = int_542441
    kwargs_542443 = {'b': keyword_542440, 'axis': keyword_542442}
    # Getting the type of 'logsumexp' (line 81)
    logsumexp_542437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 37), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 81)
    logsumexp_call_result_542444 = invoke(stypy.reporting.localization.Localization(__file__, 81, 37), logsumexp_542437, *[logX_542438], **kwargs_542443)
    
    # Processing the call keyword arguments (line 81)
    kwargs_542445 = {}
    # Getting the type of 'np' (line 81)
    np_542435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'np', False)
    # Obtaining the member 'exp' of a type (line 81)
    exp_542436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 30), np_542435, 'exp')
    # Calling exp(args, kwargs) (line 81)
    exp_call_result_542446 = invoke(stypy.reporting.localization.Localization(__file__, 81, 30), exp_542436, *[logsumexp_call_result_542444], **kwargs_542445)
    
    
    # Call to sum(...): (line 82)
    # Processing the call keyword arguments (line 82)
    int_542451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 49), 'int')
    keyword_542452 = int_542451
    kwargs_542453 = {'axis': keyword_542452}
    # Getting the type of 'B' (line 82)
    B_542447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 33), 'B', False)
    # Getting the type of 'X' (line 82)
    X_542448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 37), 'X', False)
    # Applying the binary operator '*' (line 82)
    result_mul_542449 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 33), '*', B_542447, X_542448)
    
    # Obtaining the member 'sum' of a type (line 82)
    sum_542450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 33), result_mul_542449, 'sum')
    # Calling sum(args, kwargs) (line 82)
    sum_call_result_542454 = invoke(stypy.reporting.localization.Localization(__file__, 82, 33), sum_542450, *[], **kwargs_542453)
    
    # Processing the call keyword arguments (line 81)
    kwargs_542455 = {}
    # Getting the type of 'assert_array_almost_equal' (line 81)
    assert_array_almost_equal_542434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 81)
    assert_array_almost_equal_call_result_542456 = invoke(stypy.reporting.localization.Localization(__file__, 81, 4), assert_array_almost_equal_542434, *[exp_call_result_542446, sum_call_result_542454], **kwargs_542455)
    
    
    # ################# End of 'test_logsumexp_b(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_logsumexp_b' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_542457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_542457)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_logsumexp_b'
    return stypy_return_type_542457

# Assigning a type to the variable 'test_logsumexp_b' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'test_logsumexp_b', test_logsumexp_b)

@norecursion
def test_logsumexp_sign(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_logsumexp_sign'
    module_type_store = module_type_store.open_function_context('test_logsumexp_sign', 85, 0, False)
    
    # Passed parameters checking function
    test_logsumexp_sign.stypy_localization = localization
    test_logsumexp_sign.stypy_type_of_self = None
    test_logsumexp_sign.stypy_type_store = module_type_store
    test_logsumexp_sign.stypy_function_name = 'test_logsumexp_sign'
    test_logsumexp_sign.stypy_param_names_list = []
    test_logsumexp_sign.stypy_varargs_param_name = None
    test_logsumexp_sign.stypy_kwargs_param_name = None
    test_logsumexp_sign.stypy_call_defaults = defaults
    test_logsumexp_sign.stypy_call_varargs = varargs
    test_logsumexp_sign.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_logsumexp_sign', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_logsumexp_sign', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_logsumexp_sign(...)' code ##################

    
    # Assigning a List to a Name (line 86):
    
    # Assigning a List to a Name (line 86):
    
    # Obtaining an instance of the builtin type 'list' (line 86)
    list_542458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    # Adding element type (line 86)
    int_542459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_542458, int_542459)
    # Adding element type (line 86)
    int_542460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_542458, int_542460)
    # Adding element type (line 86)
    int_542461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_542458, int_542461)
    
    # Assigning a type to the variable 'a' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'a', list_542458)
    
    # Assigning a List to a Name (line 87):
    
    # Assigning a List to a Name (line 87):
    
    # Obtaining an instance of the builtin type 'list' (line 87)
    list_542462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 87)
    # Adding element type (line 87)
    int_542463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), list_542462, int_542463)
    # Adding element type (line 87)
    int_542464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), list_542462, int_542464)
    # Adding element type (line 87)
    int_542465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), list_542462, int_542465)
    
    # Assigning a type to the variable 'b' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'b', list_542462)
    
    # Assigning a Call to a Tuple (line 89):
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_542466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'int')
    
    # Call to logsumexp(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'a' (line 89)
    a_542468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 21), 'a', False)
    # Processing the call keyword arguments (line 89)
    # Getting the type of 'b' (line 89)
    b_542469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'b', False)
    keyword_542470 = b_542469
    # Getting the type of 'True' (line 89)
    True_542471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 41), 'True', False)
    keyword_542472 = True_542471
    kwargs_542473 = {'return_sign': keyword_542472, 'b': keyword_542470}
    # Getting the type of 'logsumexp' (line 89)
    logsumexp_542467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 89)
    logsumexp_call_result_542474 = invoke(stypy.reporting.localization.Localization(__file__, 89, 11), logsumexp_542467, *[a_542468], **kwargs_542473)
    
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___542475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), logsumexp_call_result_542474, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_542476 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), getitem___542475, int_542466)
    
    # Assigning a type to the variable 'tuple_var_assignment_542015' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_542015', subscript_call_result_542476)
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_542477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'int')
    
    # Call to logsumexp(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'a' (line 89)
    a_542479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 21), 'a', False)
    # Processing the call keyword arguments (line 89)
    # Getting the type of 'b' (line 89)
    b_542480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'b', False)
    keyword_542481 = b_542480
    # Getting the type of 'True' (line 89)
    True_542482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 41), 'True', False)
    keyword_542483 = True_542482
    kwargs_542484 = {'return_sign': keyword_542483, 'b': keyword_542481}
    # Getting the type of 'logsumexp' (line 89)
    logsumexp_542478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 89)
    logsumexp_call_result_542485 = invoke(stypy.reporting.localization.Localization(__file__, 89, 11), logsumexp_542478, *[a_542479], **kwargs_542484)
    
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___542486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), logsumexp_call_result_542485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_542487 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), getitem___542486, int_542477)
    
    # Assigning a type to the variable 'tuple_var_assignment_542016' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_542016', subscript_call_result_542487)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_542015' (line 89)
    tuple_var_assignment_542015_542488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_542015')
    # Assigning a type to the variable 'r' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'r', tuple_var_assignment_542015_542488)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_542016' (line 89)
    tuple_var_assignment_542016_542489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_542016')
    # Assigning a type to the variable 's' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 7), 's', tuple_var_assignment_542016_542489)
    
    # Call to assert_almost_equal(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'r' (line 90)
    r_542491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'r', False)
    int_542492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 26), 'int')
    # Processing the call keyword arguments (line 90)
    kwargs_542493 = {}
    # Getting the type of 'assert_almost_equal' (line 90)
    assert_almost_equal_542490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 90)
    assert_almost_equal_call_result_542494 = invoke(stypy.reporting.localization.Localization(__file__, 90, 4), assert_almost_equal_542490, *[r_542491, int_542492], **kwargs_542493)
    
    
    # Call to assert_equal(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 's' (line 91)
    s_542496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 17), 's', False)
    int_542497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'int')
    # Processing the call keyword arguments (line 91)
    kwargs_542498 = {}
    # Getting the type of 'assert_equal' (line 91)
    assert_equal_542495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 91)
    assert_equal_call_result_542499 = invoke(stypy.reporting.localization.Localization(__file__, 91, 4), assert_equal_542495, *[s_542496, int_542497], **kwargs_542498)
    
    
    # ################# End of 'test_logsumexp_sign(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_logsumexp_sign' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_542500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_542500)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_logsumexp_sign'
    return stypy_return_type_542500

# Assigning a type to the variable 'test_logsumexp_sign' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'test_logsumexp_sign', test_logsumexp_sign)

@norecursion
def test_logsumexp_sign_zero(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_logsumexp_sign_zero'
    module_type_store = module_type_store.open_function_context('test_logsumexp_sign_zero', 94, 0, False)
    
    # Passed parameters checking function
    test_logsumexp_sign_zero.stypy_localization = localization
    test_logsumexp_sign_zero.stypy_type_of_self = None
    test_logsumexp_sign_zero.stypy_type_store = module_type_store
    test_logsumexp_sign_zero.stypy_function_name = 'test_logsumexp_sign_zero'
    test_logsumexp_sign_zero.stypy_param_names_list = []
    test_logsumexp_sign_zero.stypy_varargs_param_name = None
    test_logsumexp_sign_zero.stypy_kwargs_param_name = None
    test_logsumexp_sign_zero.stypy_call_defaults = defaults
    test_logsumexp_sign_zero.stypy_call_varargs = varargs
    test_logsumexp_sign_zero.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_logsumexp_sign_zero', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_logsumexp_sign_zero', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_logsumexp_sign_zero(...)' code ##################

    
    # Assigning a List to a Name (line 95):
    
    # Assigning a List to a Name (line 95):
    
    # Obtaining an instance of the builtin type 'list' (line 95)
    list_542501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 95)
    # Adding element type (line 95)
    int_542502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 8), list_542501, int_542502)
    # Adding element type (line 95)
    int_542503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 8), list_542501, int_542503)
    
    # Assigning a type to the variable 'a' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'a', list_542501)
    
    # Assigning a List to a Name (line 96):
    
    # Assigning a List to a Name (line 96):
    
    # Obtaining an instance of the builtin type 'list' (line 96)
    list_542504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 96)
    # Adding element type (line 96)
    int_542505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 8), list_542504, int_542505)
    # Adding element type (line 96)
    int_542506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 8), list_542504, int_542506)
    
    # Assigning a type to the variable 'b' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'b', list_542504)
    
    # Assigning a Call to a Tuple (line 98):
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_542507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'int')
    
    # Call to logsumexp(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'a' (line 98)
    a_542509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'a', False)
    # Processing the call keyword arguments (line 98)
    # Getting the type of 'b' (line 98)
    b_542510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'b', False)
    keyword_542511 = b_542510
    # Getting the type of 'True' (line 98)
    True_542512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 41), 'True', False)
    keyword_542513 = True_542512
    kwargs_542514 = {'return_sign': keyword_542513, 'b': keyword_542511}
    # Getting the type of 'logsumexp' (line 98)
    logsumexp_542508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 98)
    logsumexp_call_result_542515 = invoke(stypy.reporting.localization.Localization(__file__, 98, 11), logsumexp_542508, *[a_542509], **kwargs_542514)
    
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___542516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), logsumexp_call_result_542515, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_542517 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), getitem___542516, int_542507)
    
    # Assigning a type to the variable 'tuple_var_assignment_542017' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_542017', subscript_call_result_542517)
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_542518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'int')
    
    # Call to logsumexp(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'a' (line 98)
    a_542520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'a', False)
    # Processing the call keyword arguments (line 98)
    # Getting the type of 'b' (line 98)
    b_542521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'b', False)
    keyword_542522 = b_542521
    # Getting the type of 'True' (line 98)
    True_542523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 41), 'True', False)
    keyword_542524 = True_542523
    kwargs_542525 = {'return_sign': keyword_542524, 'b': keyword_542522}
    # Getting the type of 'logsumexp' (line 98)
    logsumexp_542519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 98)
    logsumexp_call_result_542526 = invoke(stypy.reporting.localization.Localization(__file__, 98, 11), logsumexp_542519, *[a_542520], **kwargs_542525)
    
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___542527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), logsumexp_call_result_542526, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_542528 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), getitem___542527, int_542518)
    
    # Assigning a type to the variable 'tuple_var_assignment_542018' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_542018', subscript_call_result_542528)
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'tuple_var_assignment_542017' (line 98)
    tuple_var_assignment_542017_542529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_542017')
    # Assigning a type to the variable 'r' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'r', tuple_var_assignment_542017_542529)
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'tuple_var_assignment_542018' (line 98)
    tuple_var_assignment_542018_542530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_542018')
    # Assigning a type to the variable 's' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 7), 's', tuple_var_assignment_542018_542530)
    
    # Call to assert_(...): (line 99)
    # Processing the call arguments (line 99)
    
    
    # Call to isfinite(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'r' (line 99)
    r_542534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 28), 'r', False)
    # Processing the call keyword arguments (line 99)
    kwargs_542535 = {}
    # Getting the type of 'np' (line 99)
    np_542532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 99)
    isfinite_542533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), np_542532, 'isfinite')
    # Calling isfinite(args, kwargs) (line 99)
    isfinite_call_result_542536 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), isfinite_542533, *[r_542534], **kwargs_542535)
    
    # Applying the 'not' unary operator (line 99)
    result_not__542537 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 12), 'not', isfinite_call_result_542536)
    
    # Processing the call keyword arguments (line 99)
    kwargs_542538 = {}
    # Getting the type of 'assert_' (line 99)
    assert__542531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 99)
    assert__call_result_542539 = invoke(stypy.reporting.localization.Localization(__file__, 99, 4), assert__542531, *[result_not__542537], **kwargs_542538)
    
    
    # Call to assert_(...): (line 100)
    # Processing the call arguments (line 100)
    
    
    # Call to isnan(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'r' (line 100)
    r_542543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'r', False)
    # Processing the call keyword arguments (line 100)
    kwargs_542544 = {}
    # Getting the type of 'np' (line 100)
    np_542541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'np', False)
    # Obtaining the member 'isnan' of a type (line 100)
    isnan_542542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 16), np_542541, 'isnan')
    # Calling isnan(args, kwargs) (line 100)
    isnan_call_result_542545 = invoke(stypy.reporting.localization.Localization(__file__, 100, 16), isnan_542542, *[r_542543], **kwargs_542544)
    
    # Applying the 'not' unary operator (line 100)
    result_not__542546 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 12), 'not', isnan_call_result_542545)
    
    # Processing the call keyword arguments (line 100)
    kwargs_542547 = {}
    # Getting the type of 'assert_' (line 100)
    assert__542540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 100)
    assert__call_result_542548 = invoke(stypy.reporting.localization.Localization(__file__, 100, 4), assert__542540, *[result_not__542546], **kwargs_542547)
    
    
    # Call to assert_(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Getting the type of 'r' (line 101)
    r_542550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'r', False)
    int_542551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 16), 'int')
    # Applying the binary operator '<' (line 101)
    result_lt_542552 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 12), '<', r_542550, int_542551)
    
    # Processing the call keyword arguments (line 101)
    kwargs_542553 = {}
    # Getting the type of 'assert_' (line 101)
    assert__542549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 101)
    assert__call_result_542554 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), assert__542549, *[result_lt_542552], **kwargs_542553)
    
    
    # Call to assert_equal(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 's' (line 102)
    s_542556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 's', False)
    int_542557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 19), 'int')
    # Processing the call keyword arguments (line 102)
    kwargs_542558 = {}
    # Getting the type of 'assert_equal' (line 102)
    assert_equal_542555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 102)
    assert_equal_call_result_542559 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), assert_equal_542555, *[s_542556, int_542557], **kwargs_542558)
    
    
    # ################# End of 'test_logsumexp_sign_zero(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_logsumexp_sign_zero' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_542560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_542560)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_logsumexp_sign_zero'
    return stypy_return_type_542560

# Assigning a type to the variable 'test_logsumexp_sign_zero' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'test_logsumexp_sign_zero', test_logsumexp_sign_zero)

@norecursion
def test_logsumexp_sign_shape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_logsumexp_sign_shape'
    module_type_store = module_type_store.open_function_context('test_logsumexp_sign_shape', 105, 0, False)
    
    # Passed parameters checking function
    test_logsumexp_sign_shape.stypy_localization = localization
    test_logsumexp_sign_shape.stypy_type_of_self = None
    test_logsumexp_sign_shape.stypy_type_store = module_type_store
    test_logsumexp_sign_shape.stypy_function_name = 'test_logsumexp_sign_shape'
    test_logsumexp_sign_shape.stypy_param_names_list = []
    test_logsumexp_sign_shape.stypy_varargs_param_name = None
    test_logsumexp_sign_shape.stypy_kwargs_param_name = None
    test_logsumexp_sign_shape.stypy_call_defaults = defaults
    test_logsumexp_sign_shape.stypy_call_varargs = varargs
    test_logsumexp_sign_shape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_logsumexp_sign_shape', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_logsumexp_sign_shape', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_logsumexp_sign_shape(...)' code ##################

    
    # Assigning a Call to a Name (line 106):
    
    # Assigning a Call to a Name (line 106):
    
    # Call to ones(...): (line 106)
    # Processing the call arguments (line 106)
    
    # Obtaining an instance of the builtin type 'tuple' (line 106)
    tuple_542563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 106)
    # Adding element type (line 106)
    int_542564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 17), tuple_542563, int_542564)
    # Adding element type (line 106)
    int_542565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 17), tuple_542563, int_542565)
    # Adding element type (line 106)
    int_542566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 17), tuple_542563, int_542566)
    # Adding element type (line 106)
    int_542567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 17), tuple_542563, int_542567)
    
    # Processing the call keyword arguments (line 106)
    kwargs_542568 = {}
    # Getting the type of 'np' (line 106)
    np_542561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 106)
    ones_542562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), np_542561, 'ones')
    # Calling ones(args, kwargs) (line 106)
    ones_call_result_542569 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), ones_542562, *[tuple_542563], **kwargs_542568)
    
    # Assigning a type to the variable 'a' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'a', ones_call_result_542569)
    
    # Assigning a Call to a Name (line 107):
    
    # Assigning a Call to a Name (line 107):
    
    # Call to ones_like(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'a' (line 107)
    a_542572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 21), 'a', False)
    # Processing the call keyword arguments (line 107)
    kwargs_542573 = {}
    # Getting the type of 'np' (line 107)
    np_542570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 107)
    ones_like_542571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), np_542570, 'ones_like')
    # Calling ones_like(args, kwargs) (line 107)
    ones_like_call_result_542574 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), ones_like_542571, *[a_542572], **kwargs_542573)
    
    # Assigning a type to the variable 'b' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'b', ones_like_call_result_542574)
    
    # Assigning a Call to a Tuple (line 109):
    
    # Assigning a Subscript to a Name (line 109):
    
    # Obtaining the type of the subscript
    int_542575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 4), 'int')
    
    # Call to logsumexp(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'a' (line 109)
    a_542577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'a', False)
    # Processing the call keyword arguments (line 109)
    int_542578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 29), 'int')
    keyword_542579 = int_542578
    # Getting the type of 'b' (line 109)
    b_542580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 34), 'b', False)
    keyword_542581 = b_542580
    # Getting the type of 'True' (line 109)
    True_542582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 49), 'True', False)
    keyword_542583 = True_542582
    kwargs_542584 = {'return_sign': keyword_542583, 'b': keyword_542581, 'axis': keyword_542579}
    # Getting the type of 'logsumexp' (line 109)
    logsumexp_542576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 109)
    logsumexp_call_result_542585 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), logsumexp_542576, *[a_542577], **kwargs_542584)
    
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___542586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 4), logsumexp_call_result_542585, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_542587 = invoke(stypy.reporting.localization.Localization(__file__, 109, 4), getitem___542586, int_542575)
    
    # Assigning a type to the variable 'tuple_var_assignment_542019' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'tuple_var_assignment_542019', subscript_call_result_542587)
    
    # Assigning a Subscript to a Name (line 109):
    
    # Obtaining the type of the subscript
    int_542588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 4), 'int')
    
    # Call to logsumexp(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'a' (line 109)
    a_542590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'a', False)
    # Processing the call keyword arguments (line 109)
    int_542591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 29), 'int')
    keyword_542592 = int_542591
    # Getting the type of 'b' (line 109)
    b_542593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 34), 'b', False)
    keyword_542594 = b_542593
    # Getting the type of 'True' (line 109)
    True_542595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 49), 'True', False)
    keyword_542596 = True_542595
    kwargs_542597 = {'return_sign': keyword_542596, 'b': keyword_542594, 'axis': keyword_542592}
    # Getting the type of 'logsumexp' (line 109)
    logsumexp_542589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 109)
    logsumexp_call_result_542598 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), logsumexp_542589, *[a_542590], **kwargs_542597)
    
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___542599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 4), logsumexp_call_result_542598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_542600 = invoke(stypy.reporting.localization.Localization(__file__, 109, 4), getitem___542599, int_542588)
    
    # Assigning a type to the variable 'tuple_var_assignment_542020' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'tuple_var_assignment_542020', subscript_call_result_542600)
    
    # Assigning a Name to a Name (line 109):
    # Getting the type of 'tuple_var_assignment_542019' (line 109)
    tuple_var_assignment_542019_542601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'tuple_var_assignment_542019')
    # Assigning a type to the variable 'r' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'r', tuple_var_assignment_542019_542601)
    
    # Assigning a Name to a Name (line 109):
    # Getting the type of 'tuple_var_assignment_542020' (line 109)
    tuple_var_assignment_542020_542602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'tuple_var_assignment_542020')
    # Assigning a type to the variable 's' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 7), 's', tuple_var_assignment_542020_542602)
    
    # Call to assert_equal(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'r' (line 111)
    r_542604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 17), 'r', False)
    # Obtaining the member 'shape' of a type (line 111)
    shape_542605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 17), r_542604, 'shape')
    # Getting the type of 's' (line 111)
    s_542606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 's', False)
    # Obtaining the member 'shape' of a type (line 111)
    shape_542607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 26), s_542606, 'shape')
    # Processing the call keyword arguments (line 111)
    kwargs_542608 = {}
    # Getting the type of 'assert_equal' (line 111)
    assert_equal_542603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 111)
    assert_equal_call_result_542609 = invoke(stypy.reporting.localization.Localization(__file__, 111, 4), assert_equal_542603, *[shape_542605, shape_542607], **kwargs_542608)
    
    
    # Call to assert_equal(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'r' (line 112)
    r_542611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'r', False)
    # Obtaining the member 'shape' of a type (line 112)
    shape_542612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 17), r_542611, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 112)
    tuple_542613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 112)
    # Adding element type (line 112)
    int_542614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 27), tuple_542613, int_542614)
    # Adding element type (line 112)
    int_542615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 27), tuple_542613, int_542615)
    # Adding element type (line 112)
    int_542616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 27), tuple_542613, int_542616)
    
    # Processing the call keyword arguments (line 112)
    kwargs_542617 = {}
    # Getting the type of 'assert_equal' (line 112)
    assert_equal_542610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 112)
    assert_equal_call_result_542618 = invoke(stypy.reporting.localization.Localization(__file__, 112, 4), assert_equal_542610, *[shape_542612, tuple_542613], **kwargs_542617)
    
    
    # Assigning a Call to a Tuple (line 114):
    
    # Assigning a Subscript to a Name (line 114):
    
    # Obtaining the type of the subscript
    int_542619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 4), 'int')
    
    # Call to logsumexp(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'a' (line 114)
    a_542621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'a', False)
    # Processing the call keyword arguments (line 114)
    
    # Obtaining an instance of the builtin type 'tuple' (line 114)
    tuple_542622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 114)
    # Adding element type (line 114)
    int_542623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 30), tuple_542622, int_542623)
    # Adding element type (line 114)
    int_542624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 30), tuple_542622, int_542624)
    
    keyword_542625 = tuple_542622
    # Getting the type of 'b' (line 114)
    b_542626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 38), 'b', False)
    keyword_542627 = b_542626
    # Getting the type of 'True' (line 114)
    True_542628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 53), 'True', False)
    keyword_542629 = True_542628
    kwargs_542630 = {'return_sign': keyword_542629, 'b': keyword_542627, 'axis': keyword_542625}
    # Getting the type of 'logsumexp' (line 114)
    logsumexp_542620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 114)
    logsumexp_call_result_542631 = invoke(stypy.reporting.localization.Localization(__file__, 114, 11), logsumexp_542620, *[a_542621], **kwargs_542630)
    
    # Obtaining the member '__getitem__' of a type (line 114)
    getitem___542632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 4), logsumexp_call_result_542631, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 114)
    subscript_call_result_542633 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), getitem___542632, int_542619)
    
    # Assigning a type to the variable 'tuple_var_assignment_542021' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'tuple_var_assignment_542021', subscript_call_result_542633)
    
    # Assigning a Subscript to a Name (line 114):
    
    # Obtaining the type of the subscript
    int_542634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 4), 'int')
    
    # Call to logsumexp(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'a' (line 114)
    a_542636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'a', False)
    # Processing the call keyword arguments (line 114)
    
    # Obtaining an instance of the builtin type 'tuple' (line 114)
    tuple_542637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 114)
    # Adding element type (line 114)
    int_542638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 30), tuple_542637, int_542638)
    # Adding element type (line 114)
    int_542639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 30), tuple_542637, int_542639)
    
    keyword_542640 = tuple_542637
    # Getting the type of 'b' (line 114)
    b_542641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 38), 'b', False)
    keyword_542642 = b_542641
    # Getting the type of 'True' (line 114)
    True_542643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 53), 'True', False)
    keyword_542644 = True_542643
    kwargs_542645 = {'return_sign': keyword_542644, 'b': keyword_542642, 'axis': keyword_542640}
    # Getting the type of 'logsumexp' (line 114)
    logsumexp_542635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 114)
    logsumexp_call_result_542646 = invoke(stypy.reporting.localization.Localization(__file__, 114, 11), logsumexp_542635, *[a_542636], **kwargs_542645)
    
    # Obtaining the member '__getitem__' of a type (line 114)
    getitem___542647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 4), logsumexp_call_result_542646, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 114)
    subscript_call_result_542648 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), getitem___542647, int_542634)
    
    # Assigning a type to the variable 'tuple_var_assignment_542022' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'tuple_var_assignment_542022', subscript_call_result_542648)
    
    # Assigning a Name to a Name (line 114):
    # Getting the type of 'tuple_var_assignment_542021' (line 114)
    tuple_var_assignment_542021_542649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'tuple_var_assignment_542021')
    # Assigning a type to the variable 'r' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'r', tuple_var_assignment_542021_542649)
    
    # Assigning a Name to a Name (line 114):
    # Getting the type of 'tuple_var_assignment_542022' (line 114)
    tuple_var_assignment_542022_542650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'tuple_var_assignment_542022')
    # Assigning a type to the variable 's' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 7), 's', tuple_var_assignment_542022_542650)
    
    # Call to assert_equal(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'r' (line 116)
    r_542652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'r', False)
    # Obtaining the member 'shape' of a type (line 116)
    shape_542653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 17), r_542652, 'shape')
    # Getting the type of 's' (line 116)
    s_542654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 's', False)
    # Obtaining the member 'shape' of a type (line 116)
    shape_542655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 26), s_542654, 'shape')
    # Processing the call keyword arguments (line 116)
    kwargs_542656 = {}
    # Getting the type of 'assert_equal' (line 116)
    assert_equal_542651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 116)
    assert_equal_call_result_542657 = invoke(stypy.reporting.localization.Localization(__file__, 116, 4), assert_equal_542651, *[shape_542653, shape_542655], **kwargs_542656)
    
    
    # Call to assert_equal(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'r' (line 117)
    r_542659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'r', False)
    # Obtaining the member 'shape' of a type (line 117)
    shape_542660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 17), r_542659, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 117)
    tuple_542661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 117)
    # Adding element type (line 117)
    int_542662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 27), tuple_542661, int_542662)
    # Adding element type (line 117)
    int_542663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 27), tuple_542661, int_542663)
    
    # Processing the call keyword arguments (line 117)
    kwargs_542664 = {}
    # Getting the type of 'assert_equal' (line 117)
    assert_equal_542658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 117)
    assert_equal_call_result_542665 = invoke(stypy.reporting.localization.Localization(__file__, 117, 4), assert_equal_542658, *[shape_542660, tuple_542661], **kwargs_542664)
    
    
    # ################# End of 'test_logsumexp_sign_shape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_logsumexp_sign_shape' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_542666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_542666)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_logsumexp_sign_shape'
    return stypy_return_type_542666

# Assigning a type to the variable 'test_logsumexp_sign_shape' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'test_logsumexp_sign_shape', test_logsumexp_sign_shape)

@norecursion
def test_logsumexp_shape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_logsumexp_shape'
    module_type_store = module_type_store.open_function_context('test_logsumexp_shape', 120, 0, False)
    
    # Passed parameters checking function
    test_logsumexp_shape.stypy_localization = localization
    test_logsumexp_shape.stypy_type_of_self = None
    test_logsumexp_shape.stypy_type_store = module_type_store
    test_logsumexp_shape.stypy_function_name = 'test_logsumexp_shape'
    test_logsumexp_shape.stypy_param_names_list = []
    test_logsumexp_shape.stypy_varargs_param_name = None
    test_logsumexp_shape.stypy_kwargs_param_name = None
    test_logsumexp_shape.stypy_call_defaults = defaults
    test_logsumexp_shape.stypy_call_varargs = varargs
    test_logsumexp_shape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_logsumexp_shape', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_logsumexp_shape', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_logsumexp_shape(...)' code ##################

    
    # Assigning a Call to a Name (line 121):
    
    # Assigning a Call to a Name (line 121):
    
    # Call to ones(...): (line 121)
    # Processing the call arguments (line 121)
    
    # Obtaining an instance of the builtin type 'tuple' (line 121)
    tuple_542669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 121)
    # Adding element type (line 121)
    int_542670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 17), tuple_542669, int_542670)
    # Adding element type (line 121)
    int_542671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 17), tuple_542669, int_542671)
    # Adding element type (line 121)
    int_542672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 17), tuple_542669, int_542672)
    # Adding element type (line 121)
    int_542673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 17), tuple_542669, int_542673)
    
    # Processing the call keyword arguments (line 121)
    kwargs_542674 = {}
    # Getting the type of 'np' (line 121)
    np_542667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 121)
    ones_542668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), np_542667, 'ones')
    # Calling ones(args, kwargs) (line 121)
    ones_call_result_542675 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), ones_542668, *[tuple_542669], **kwargs_542674)
    
    # Assigning a type to the variable 'a' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'a', ones_call_result_542675)
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to ones_like(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'a' (line 122)
    a_542678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'a', False)
    # Processing the call keyword arguments (line 122)
    kwargs_542679 = {}
    # Getting the type of 'np' (line 122)
    np_542676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 122)
    ones_like_542677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), np_542676, 'ones_like')
    # Calling ones_like(args, kwargs) (line 122)
    ones_like_call_result_542680 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), ones_like_542677, *[a_542678], **kwargs_542679)
    
    # Assigning a type to the variable 'b' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'b', ones_like_call_result_542680)
    
    # Assigning a Call to a Name (line 124):
    
    # Assigning a Call to a Name (line 124):
    
    # Call to logsumexp(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'a' (line 124)
    a_542682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'a', False)
    # Processing the call keyword arguments (line 124)
    int_542683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 26), 'int')
    keyword_542684 = int_542683
    # Getting the type of 'b' (line 124)
    b_542685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 31), 'b', False)
    keyword_542686 = b_542685
    kwargs_542687 = {'b': keyword_542686, 'axis': keyword_542684}
    # Getting the type of 'logsumexp' (line 124)
    logsumexp_542681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 124)
    logsumexp_call_result_542688 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), logsumexp_542681, *[a_542682], **kwargs_542687)
    
    # Assigning a type to the variable 'r' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'r', logsumexp_call_result_542688)
    
    # Call to assert_equal(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'r' (line 125)
    r_542690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'r', False)
    # Obtaining the member 'shape' of a type (line 125)
    shape_542691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 17), r_542690, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 125)
    tuple_542692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 125)
    # Adding element type (line 125)
    int_542693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 27), tuple_542692, int_542693)
    # Adding element type (line 125)
    int_542694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 27), tuple_542692, int_542694)
    # Adding element type (line 125)
    int_542695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 27), tuple_542692, int_542695)
    
    # Processing the call keyword arguments (line 125)
    kwargs_542696 = {}
    # Getting the type of 'assert_equal' (line 125)
    assert_equal_542689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 125)
    assert_equal_call_result_542697 = invoke(stypy.reporting.localization.Localization(__file__, 125, 4), assert_equal_542689, *[shape_542691, tuple_542692], **kwargs_542696)
    
    
    # Assigning a Call to a Name (line 127):
    
    # Assigning a Call to a Name (line 127):
    
    # Call to logsumexp(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'a' (line 127)
    a_542699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 18), 'a', False)
    # Processing the call keyword arguments (line 127)
    
    # Obtaining an instance of the builtin type 'tuple' (line 127)
    tuple_542700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 127)
    # Adding element type (line 127)
    int_542701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 27), tuple_542700, int_542701)
    # Adding element type (line 127)
    int_542702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 27), tuple_542700, int_542702)
    
    keyword_542703 = tuple_542700
    # Getting the type of 'b' (line 127)
    b_542704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 36), 'b', False)
    keyword_542705 = b_542704
    kwargs_542706 = {'b': keyword_542705, 'axis': keyword_542703}
    # Getting the type of 'logsumexp' (line 127)
    logsumexp_542698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 127)
    logsumexp_call_result_542707 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), logsumexp_542698, *[a_542699], **kwargs_542706)
    
    # Assigning a type to the variable 'r' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'r', logsumexp_call_result_542707)
    
    # Call to assert_equal(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'r' (line 128)
    r_542709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'r', False)
    # Obtaining the member 'shape' of a type (line 128)
    shape_542710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 17), r_542709, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 128)
    tuple_542711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 128)
    # Adding element type (line 128)
    int_542712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 27), tuple_542711, int_542712)
    # Adding element type (line 128)
    int_542713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 27), tuple_542711, int_542713)
    
    # Processing the call keyword arguments (line 128)
    kwargs_542714 = {}
    # Getting the type of 'assert_equal' (line 128)
    assert_equal_542708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 128)
    assert_equal_call_result_542715 = invoke(stypy.reporting.localization.Localization(__file__, 128, 4), assert_equal_542708, *[shape_542710, tuple_542711], **kwargs_542714)
    
    
    # ################# End of 'test_logsumexp_shape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_logsumexp_shape' in the type store
    # Getting the type of 'stypy_return_type' (line 120)
    stypy_return_type_542716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_542716)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_logsumexp_shape'
    return stypy_return_type_542716

# Assigning a type to the variable 'test_logsumexp_shape' (line 120)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'test_logsumexp_shape', test_logsumexp_shape)

@norecursion
def test_logsumexp_b_zero(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_logsumexp_b_zero'
    module_type_store = module_type_store.open_function_context('test_logsumexp_b_zero', 131, 0, False)
    
    # Passed parameters checking function
    test_logsumexp_b_zero.stypy_localization = localization
    test_logsumexp_b_zero.stypy_type_of_self = None
    test_logsumexp_b_zero.stypy_type_store = module_type_store
    test_logsumexp_b_zero.stypy_function_name = 'test_logsumexp_b_zero'
    test_logsumexp_b_zero.stypy_param_names_list = []
    test_logsumexp_b_zero.stypy_varargs_param_name = None
    test_logsumexp_b_zero.stypy_kwargs_param_name = None
    test_logsumexp_b_zero.stypy_call_defaults = defaults
    test_logsumexp_b_zero.stypy_call_varargs = varargs
    test_logsumexp_b_zero.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_logsumexp_b_zero', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_logsumexp_b_zero', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_logsumexp_b_zero(...)' code ##################

    
    # Assigning a List to a Name (line 132):
    
    # Assigning a List to a Name (line 132):
    
    # Obtaining an instance of the builtin type 'list' (line 132)
    list_542717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 132)
    # Adding element type (line 132)
    int_542718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 8), list_542717, int_542718)
    # Adding element type (line 132)
    int_542719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 8), list_542717, int_542719)
    
    # Assigning a type to the variable 'a' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'a', list_542717)
    
    # Assigning a List to a Name (line 133):
    
    # Assigning a List to a Name (line 133):
    
    # Obtaining an instance of the builtin type 'list' (line 133)
    list_542720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 133)
    # Adding element type (line 133)
    int_542721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 8), list_542720, int_542721)
    # Adding element type (line 133)
    int_542722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 8), list_542720, int_542722)
    
    # Assigning a type to the variable 'b' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'b', list_542720)
    
    # Call to assert_almost_equal(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Call to logsumexp(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'a' (line 135)
    a_542725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 34), 'a', False)
    # Processing the call keyword arguments (line 135)
    # Getting the type of 'b' (line 135)
    b_542726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 39), 'b', False)
    keyword_542727 = b_542726
    kwargs_542728 = {'b': keyword_542727}
    # Getting the type of 'logsumexp' (line 135)
    logsumexp_542724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 135)
    logsumexp_call_result_542729 = invoke(stypy.reporting.localization.Localization(__file__, 135, 24), logsumexp_542724, *[a_542725], **kwargs_542728)
    
    int_542730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 43), 'int')
    # Processing the call keyword arguments (line 135)
    kwargs_542731 = {}
    # Getting the type of 'assert_almost_equal' (line 135)
    assert_almost_equal_542723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 135)
    assert_almost_equal_call_result_542732 = invoke(stypy.reporting.localization.Localization(__file__, 135, 4), assert_almost_equal_542723, *[logsumexp_call_result_542729, int_542730], **kwargs_542731)
    
    
    # ################# End of 'test_logsumexp_b_zero(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_logsumexp_b_zero' in the type store
    # Getting the type of 'stypy_return_type' (line 131)
    stypy_return_type_542733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_542733)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_logsumexp_b_zero'
    return stypy_return_type_542733

# Assigning a type to the variable 'test_logsumexp_b_zero' (line 131)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'test_logsumexp_b_zero', test_logsumexp_b_zero)

@norecursion
def test_logsumexp_b_shape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_logsumexp_b_shape'
    module_type_store = module_type_store.open_function_context('test_logsumexp_b_shape', 138, 0, False)
    
    # Passed parameters checking function
    test_logsumexp_b_shape.stypy_localization = localization
    test_logsumexp_b_shape.stypy_type_of_self = None
    test_logsumexp_b_shape.stypy_type_store = module_type_store
    test_logsumexp_b_shape.stypy_function_name = 'test_logsumexp_b_shape'
    test_logsumexp_b_shape.stypy_param_names_list = []
    test_logsumexp_b_shape.stypy_varargs_param_name = None
    test_logsumexp_b_shape.stypy_kwargs_param_name = None
    test_logsumexp_b_shape.stypy_call_defaults = defaults
    test_logsumexp_b_shape.stypy_call_varargs = varargs
    test_logsumexp_b_shape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_logsumexp_b_shape', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_logsumexp_b_shape', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_logsumexp_b_shape(...)' code ##################

    
    # Assigning a Call to a Name (line 139):
    
    # Assigning a Call to a Name (line 139):
    
    # Call to zeros(...): (line 139)
    # Processing the call arguments (line 139)
    
    # Obtaining an instance of the builtin type 'tuple' (line 139)
    tuple_542736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 139)
    # Adding element type (line 139)
    int_542737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 18), tuple_542736, int_542737)
    # Adding element type (line 139)
    int_542738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 18), tuple_542736, int_542738)
    # Adding element type (line 139)
    int_542739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 18), tuple_542736, int_542739)
    # Adding element type (line 139)
    int_542740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 18), tuple_542736, int_542740)
    
    # Processing the call keyword arguments (line 139)
    kwargs_542741 = {}
    # Getting the type of 'np' (line 139)
    np_542734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 139)
    zeros_542735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), np_542734, 'zeros')
    # Calling zeros(args, kwargs) (line 139)
    zeros_call_result_542742 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), zeros_542735, *[tuple_542736], **kwargs_542741)
    
    # Assigning a type to the variable 'a' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'a', zeros_call_result_542742)
    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to ones(...): (line 140)
    # Processing the call arguments (line 140)
    
    # Obtaining an instance of the builtin type 'tuple' (line 140)
    tuple_542745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 140)
    # Adding element type (line 140)
    int_542746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 17), tuple_542745, int_542746)
    # Adding element type (line 140)
    int_542747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 17), tuple_542745, int_542747)
    # Adding element type (line 140)
    int_542748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 17), tuple_542745, int_542748)
    
    # Processing the call keyword arguments (line 140)
    kwargs_542749 = {}
    # Getting the type of 'np' (line 140)
    np_542743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 140)
    ones_542744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), np_542743, 'ones')
    # Calling ones(args, kwargs) (line 140)
    ones_call_result_542750 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), ones_542744, *[tuple_542745], **kwargs_542749)
    
    # Assigning a type to the variable 'b' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'b', ones_call_result_542750)
    
    # Call to logsumexp(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'a' (line 142)
    a_542752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 14), 'a', False)
    # Processing the call keyword arguments (line 142)
    # Getting the type of 'b' (line 142)
    b_542753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'b', False)
    keyword_542754 = b_542753
    kwargs_542755 = {'b': keyword_542754}
    # Getting the type of 'logsumexp' (line 142)
    logsumexp_542751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'logsumexp', False)
    # Calling logsumexp(args, kwargs) (line 142)
    logsumexp_call_result_542756 = invoke(stypy.reporting.localization.Localization(__file__, 142, 4), logsumexp_542751, *[a_542752], **kwargs_542755)
    
    
    # ################# End of 'test_logsumexp_b_shape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_logsumexp_b_shape' in the type store
    # Getting the type of 'stypy_return_type' (line 138)
    stypy_return_type_542757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_542757)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_logsumexp_b_shape'
    return stypy_return_type_542757

# Assigning a type to the variable 'test_logsumexp_b_shape' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'test_logsumexp_b_shape', test_logsumexp_b_shape)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
