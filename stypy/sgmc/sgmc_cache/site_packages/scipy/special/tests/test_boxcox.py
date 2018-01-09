
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
5: from scipy.special import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p
6: 
7: 
8: # There are more tests of boxcox and boxcox1p in test_mpmath.py.
9: 
10: def test_boxcox_basic():
11:     x = np.array([0.5, 1, 2, 4])
12: 
13:     # lambda = 0  =>  y = log(x)
14:     y = boxcox(x, 0)
15:     assert_almost_equal(y, np.log(x))
16: 
17:     # lambda = 1  =>  y = x - 1
18:     y = boxcox(x, 1)
19:     assert_almost_equal(y, x - 1)
20: 
21:     # lambda = 2  =>  y = 0.5*(x**2 - 1)
22:     y = boxcox(x, 2)
23:     assert_almost_equal(y, 0.5*(x**2 - 1))
24: 
25:     # x = 0 and lambda > 0  =>  y = -1 / lambda
26:     lam = np.array([0.5, 1, 2])
27:     y = boxcox(0, lam)
28:     assert_almost_equal(y, -1.0 / lam)
29: 
30: def test_boxcox_underflow():
31:     x = 1 + 1e-15
32:     lmbda = 1e-306
33:     y = boxcox(x, lmbda)
34:     assert_allclose(y, np.log(x), rtol=1e-14)
35: 
36: 
37: def test_boxcox_nonfinite():
38:     # x < 0  =>  y = nan
39:     x = np.array([-1, -1, -0.5])
40:     y = boxcox(x, [0.5, 2.0, -1.5])
41:     assert_equal(y, np.array([np.nan, np.nan, np.nan]))
42: 
43:     # x = 0 and lambda <= 0  =>  y = -inf
44:     x = 0
45:     y = boxcox(x, [-2.5, 0])
46:     assert_equal(y, np.array([-np.inf, -np.inf]))
47: 
48: 
49: def test_boxcox1p_basic():
50:     x = np.array([-0.25, -1e-20, 0, 1e-20, 0.25, 1, 3])
51: 
52:     # lambda = 0  =>  y = log(1+x)
53:     y = boxcox1p(x, 0)
54:     assert_almost_equal(y, np.log1p(x))
55: 
56:     # lambda = 1  =>  y = x
57:     y = boxcox1p(x, 1)
58:     assert_almost_equal(y, x)
59: 
60:     # lambda = 2  =>  y = 0.5*((1+x)**2 - 1) = 0.5*x*(2 + x)
61:     y = boxcox1p(x, 2)
62:     assert_almost_equal(y, 0.5*x*(2 + x))
63: 
64:     # x = -1 and lambda > 0  =>  y = -1 / lambda
65:     lam = np.array([0.5, 1, 2])
66:     y = boxcox1p(-1, lam)
67:     assert_almost_equal(y, -1.0 / lam)
68: 
69: 
70: def test_boxcox1p_underflow():
71:     x = np.array([1e-15, 1e-306])
72:     lmbda = np.array([1e-306, 1e-18])
73:     y = boxcox1p(x, lmbda)
74:     assert_allclose(y, np.log1p(x), rtol=1e-14)
75: 
76: 
77: def test_boxcox1p_nonfinite():
78:     # x < -1  =>  y = nan
79:     x = np.array([-2, -2, -1.5])
80:     y = boxcox1p(x, [0.5, 2.0, -1.5])
81:     assert_equal(y, np.array([np.nan, np.nan, np.nan]))
82: 
83:     # x = -1 and lambda <= 0  =>  y = -inf
84:     x = -1
85:     y = boxcox1p(x, [-2.5, 0])
86:     assert_equal(y, np.array([-np.inf, -np.inf]))
87: 
88: 
89: def test_inv_boxcox():
90:     x = np.array([0., 1., 2.])
91:     lam = np.array([0., 1., 2.])
92:     y = boxcox(x, lam)
93:     x2 = inv_boxcox(y, lam)
94:     assert_almost_equal(x, x2)
95: 
96:     x = np.array([0., 1., 2.])
97:     lam = np.array([0., 1., 2.])
98:     y = boxcox1p(x, lam)
99:     x2 = inv_boxcox1p(y, lam)
100:     assert_almost_equal(x, x2)
101: 
102: 
103: def test_inv_boxcox1p_underflow():
104:     x = 1e-15
105:     lam = 1e-306
106:     y = inv_boxcox1p(x, lam)
107:     assert_allclose(y, x, rtol=1e-14)
108: 
109: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_530487 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_530487) is not StypyTypeError):

    if (import_530487 != 'pyd_module'):
        __import__(import_530487)
        sys_modules_530488 = sys.modules[import_530487]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_530488.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_530487)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_equal, assert_almost_equal, assert_allclose' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_530489 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_530489) is not StypyTypeError):

    if (import_530489 != 'pyd_module'):
        __import__(import_530489)
        sys_modules_530490 = sys.modules[import_530489]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_530490.module_type_store, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_530490, sys_modules_530490.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_almost_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_allclose'], [assert_equal, assert_almost_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_530489)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.special import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_530491 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special')

if (type(import_530491) is not StypyTypeError):

    if (import_530491 != 'pyd_module'):
        __import__(import_530491)
        sys_modules_530492 = sys.modules[import_530491]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', sys_modules_530492.module_type_store, module_type_store, ['boxcox', 'boxcox1p', 'inv_boxcox', 'inv_boxcox1p'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_530492, sys_modules_530492.module_type_store, module_type_store)
    else:
        from scipy.special import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', None, module_type_store, ['boxcox', 'boxcox1p', 'inv_boxcox', 'inv_boxcox1p'], [boxcox, boxcox1p, inv_boxcox, inv_boxcox1p])

else:
    # Assigning a type to the variable 'scipy.special' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', import_530491)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


@norecursion
def test_boxcox_basic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_boxcox_basic'
    module_type_store = module_type_store.open_function_context('test_boxcox_basic', 10, 0, False)
    
    # Passed parameters checking function
    test_boxcox_basic.stypy_localization = localization
    test_boxcox_basic.stypy_type_of_self = None
    test_boxcox_basic.stypy_type_store = module_type_store
    test_boxcox_basic.stypy_function_name = 'test_boxcox_basic'
    test_boxcox_basic.stypy_param_names_list = []
    test_boxcox_basic.stypy_varargs_param_name = None
    test_boxcox_basic.stypy_kwargs_param_name = None
    test_boxcox_basic.stypy_call_defaults = defaults
    test_boxcox_basic.stypy_call_varargs = varargs
    test_boxcox_basic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_boxcox_basic', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_boxcox_basic', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_boxcox_basic(...)' code ##################

    
    # Assigning a Call to a Name (line 11):
    
    # Call to array(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Obtaining an instance of the builtin type 'list' (line 11)
    list_530495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 11)
    # Adding element type (line 11)
    float_530496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 17), list_530495, float_530496)
    # Adding element type (line 11)
    int_530497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 17), list_530495, int_530497)
    # Adding element type (line 11)
    int_530498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 17), list_530495, int_530498)
    # Adding element type (line 11)
    int_530499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 17), list_530495, int_530499)
    
    # Processing the call keyword arguments (line 11)
    kwargs_530500 = {}
    # Getting the type of 'np' (line 11)
    np_530493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 11)
    array_530494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), np_530493, 'array')
    # Calling array(args, kwargs) (line 11)
    array_call_result_530501 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), array_530494, *[list_530495], **kwargs_530500)
    
    # Assigning a type to the variable 'x' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'x', array_call_result_530501)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to boxcox(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'x' (line 14)
    x_530503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'x', False)
    int_530504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_530505 = {}
    # Getting the type of 'boxcox' (line 14)
    boxcox_530502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'boxcox', False)
    # Calling boxcox(args, kwargs) (line 14)
    boxcox_call_result_530506 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), boxcox_530502, *[x_530503, int_530504], **kwargs_530505)
    
    # Assigning a type to the variable 'y' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'y', boxcox_call_result_530506)
    
    # Call to assert_almost_equal(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'y' (line 15)
    y_530508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'y', False)
    
    # Call to log(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'x' (line 15)
    x_530511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 34), 'x', False)
    # Processing the call keyword arguments (line 15)
    kwargs_530512 = {}
    # Getting the type of 'np' (line 15)
    np_530509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 27), 'np', False)
    # Obtaining the member 'log' of a type (line 15)
    log_530510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 27), np_530509, 'log')
    # Calling log(args, kwargs) (line 15)
    log_call_result_530513 = invoke(stypy.reporting.localization.Localization(__file__, 15, 27), log_530510, *[x_530511], **kwargs_530512)
    
    # Processing the call keyword arguments (line 15)
    kwargs_530514 = {}
    # Getting the type of 'assert_almost_equal' (line 15)
    assert_almost_equal_530507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 15)
    assert_almost_equal_call_result_530515 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), assert_almost_equal_530507, *[y_530508, log_call_result_530513], **kwargs_530514)
    
    
    # Assigning a Call to a Name (line 18):
    
    # Call to boxcox(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'x' (line 18)
    x_530517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'x', False)
    int_530518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 18), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_530519 = {}
    # Getting the type of 'boxcox' (line 18)
    boxcox_530516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'boxcox', False)
    # Calling boxcox(args, kwargs) (line 18)
    boxcox_call_result_530520 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), boxcox_530516, *[x_530517, int_530518], **kwargs_530519)
    
    # Assigning a type to the variable 'y' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'y', boxcox_call_result_530520)
    
    # Call to assert_almost_equal(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'y' (line 19)
    y_530522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'y', False)
    # Getting the type of 'x' (line 19)
    x_530523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'x', False)
    int_530524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'int')
    # Applying the binary operator '-' (line 19)
    result_sub_530525 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 27), '-', x_530523, int_530524)
    
    # Processing the call keyword arguments (line 19)
    kwargs_530526 = {}
    # Getting the type of 'assert_almost_equal' (line 19)
    assert_almost_equal_530521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 19)
    assert_almost_equal_call_result_530527 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), assert_almost_equal_530521, *[y_530522, result_sub_530525], **kwargs_530526)
    
    
    # Assigning a Call to a Name (line 22):
    
    # Call to boxcox(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'x' (line 22)
    x_530529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'x', False)
    int_530530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'int')
    # Processing the call keyword arguments (line 22)
    kwargs_530531 = {}
    # Getting the type of 'boxcox' (line 22)
    boxcox_530528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'boxcox', False)
    # Calling boxcox(args, kwargs) (line 22)
    boxcox_call_result_530532 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), boxcox_530528, *[x_530529, int_530530], **kwargs_530531)
    
    # Assigning a type to the variable 'y' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'y', boxcox_call_result_530532)
    
    # Call to assert_almost_equal(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'y' (line 23)
    y_530534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'y', False)
    float_530535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 27), 'float')
    # Getting the type of 'x' (line 23)
    x_530536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 32), 'x', False)
    int_530537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 35), 'int')
    # Applying the binary operator '**' (line 23)
    result_pow_530538 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 32), '**', x_530536, int_530537)
    
    int_530539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 39), 'int')
    # Applying the binary operator '-' (line 23)
    result_sub_530540 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 32), '-', result_pow_530538, int_530539)
    
    # Applying the binary operator '*' (line 23)
    result_mul_530541 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 27), '*', float_530535, result_sub_530540)
    
    # Processing the call keyword arguments (line 23)
    kwargs_530542 = {}
    # Getting the type of 'assert_almost_equal' (line 23)
    assert_almost_equal_530533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 23)
    assert_almost_equal_call_result_530543 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), assert_almost_equal_530533, *[y_530534, result_mul_530541], **kwargs_530542)
    
    
    # Assigning a Call to a Name (line 26):
    
    # Call to array(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_530546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    float_530547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_530546, float_530547)
    # Adding element type (line 26)
    int_530548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_530546, int_530548)
    # Adding element type (line 26)
    int_530549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_530546, int_530549)
    
    # Processing the call keyword arguments (line 26)
    kwargs_530550 = {}
    # Getting the type of 'np' (line 26)
    np_530544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 26)
    array_530545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 10), np_530544, 'array')
    # Calling array(args, kwargs) (line 26)
    array_call_result_530551 = invoke(stypy.reporting.localization.Localization(__file__, 26, 10), array_530545, *[list_530546], **kwargs_530550)
    
    # Assigning a type to the variable 'lam' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'lam', array_call_result_530551)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to boxcox(...): (line 27)
    # Processing the call arguments (line 27)
    int_530553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'int')
    # Getting the type of 'lam' (line 27)
    lam_530554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'lam', False)
    # Processing the call keyword arguments (line 27)
    kwargs_530555 = {}
    # Getting the type of 'boxcox' (line 27)
    boxcox_530552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'boxcox', False)
    # Calling boxcox(args, kwargs) (line 27)
    boxcox_call_result_530556 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), boxcox_530552, *[int_530553, lam_530554], **kwargs_530555)
    
    # Assigning a type to the variable 'y' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'y', boxcox_call_result_530556)
    
    # Call to assert_almost_equal(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'y' (line 28)
    y_530558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'y', False)
    float_530559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 27), 'float')
    # Getting the type of 'lam' (line 28)
    lam_530560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'lam', False)
    # Applying the binary operator 'div' (line 28)
    result_div_530561 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 27), 'div', float_530559, lam_530560)
    
    # Processing the call keyword arguments (line 28)
    kwargs_530562 = {}
    # Getting the type of 'assert_almost_equal' (line 28)
    assert_almost_equal_530557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 28)
    assert_almost_equal_call_result_530563 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), assert_almost_equal_530557, *[y_530558, result_div_530561], **kwargs_530562)
    
    
    # ################# End of 'test_boxcox_basic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_boxcox_basic' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_530564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_530564)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_boxcox_basic'
    return stypy_return_type_530564

# Assigning a type to the variable 'test_boxcox_basic' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'test_boxcox_basic', test_boxcox_basic)

@norecursion
def test_boxcox_underflow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_boxcox_underflow'
    module_type_store = module_type_store.open_function_context('test_boxcox_underflow', 30, 0, False)
    
    # Passed parameters checking function
    test_boxcox_underflow.stypy_localization = localization
    test_boxcox_underflow.stypy_type_of_self = None
    test_boxcox_underflow.stypy_type_store = module_type_store
    test_boxcox_underflow.stypy_function_name = 'test_boxcox_underflow'
    test_boxcox_underflow.stypy_param_names_list = []
    test_boxcox_underflow.stypy_varargs_param_name = None
    test_boxcox_underflow.stypy_kwargs_param_name = None
    test_boxcox_underflow.stypy_call_defaults = defaults
    test_boxcox_underflow.stypy_call_varargs = varargs
    test_boxcox_underflow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_boxcox_underflow', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_boxcox_underflow', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_boxcox_underflow(...)' code ##################

    
    # Assigning a BinOp to a Name (line 31):
    int_530565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 8), 'int')
    float_530566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'float')
    # Applying the binary operator '+' (line 31)
    result_add_530567 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 8), '+', int_530565, float_530566)
    
    # Assigning a type to the variable 'x' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'x', result_add_530567)
    
    # Assigning a Num to a Name (line 32):
    float_530568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 12), 'float')
    # Assigning a type to the variable 'lmbda' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'lmbda', float_530568)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to boxcox(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'x' (line 33)
    x_530570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'x', False)
    # Getting the type of 'lmbda' (line 33)
    lmbda_530571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 18), 'lmbda', False)
    # Processing the call keyword arguments (line 33)
    kwargs_530572 = {}
    # Getting the type of 'boxcox' (line 33)
    boxcox_530569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'boxcox', False)
    # Calling boxcox(args, kwargs) (line 33)
    boxcox_call_result_530573 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), boxcox_530569, *[x_530570, lmbda_530571], **kwargs_530572)
    
    # Assigning a type to the variable 'y' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'y', boxcox_call_result_530573)
    
    # Call to assert_allclose(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'y' (line 34)
    y_530575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'y', False)
    
    # Call to log(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'x' (line 34)
    x_530578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'x', False)
    # Processing the call keyword arguments (line 34)
    kwargs_530579 = {}
    # Getting the type of 'np' (line 34)
    np_530576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'np', False)
    # Obtaining the member 'log' of a type (line 34)
    log_530577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 23), np_530576, 'log')
    # Calling log(args, kwargs) (line 34)
    log_call_result_530580 = invoke(stypy.reporting.localization.Localization(__file__, 34, 23), log_530577, *[x_530578], **kwargs_530579)
    
    # Processing the call keyword arguments (line 34)
    float_530581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 39), 'float')
    keyword_530582 = float_530581
    kwargs_530583 = {'rtol': keyword_530582}
    # Getting the type of 'assert_allclose' (line 34)
    assert_allclose_530574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 34)
    assert_allclose_call_result_530584 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), assert_allclose_530574, *[y_530575, log_call_result_530580], **kwargs_530583)
    
    
    # ################# End of 'test_boxcox_underflow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_boxcox_underflow' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_530585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_530585)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_boxcox_underflow'
    return stypy_return_type_530585

# Assigning a type to the variable 'test_boxcox_underflow' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'test_boxcox_underflow', test_boxcox_underflow)

@norecursion
def test_boxcox_nonfinite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_boxcox_nonfinite'
    module_type_store = module_type_store.open_function_context('test_boxcox_nonfinite', 37, 0, False)
    
    # Passed parameters checking function
    test_boxcox_nonfinite.stypy_localization = localization
    test_boxcox_nonfinite.stypy_type_of_self = None
    test_boxcox_nonfinite.stypy_type_store = module_type_store
    test_boxcox_nonfinite.stypy_function_name = 'test_boxcox_nonfinite'
    test_boxcox_nonfinite.stypy_param_names_list = []
    test_boxcox_nonfinite.stypy_varargs_param_name = None
    test_boxcox_nonfinite.stypy_kwargs_param_name = None
    test_boxcox_nonfinite.stypy_call_defaults = defaults
    test_boxcox_nonfinite.stypy_call_varargs = varargs
    test_boxcox_nonfinite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_boxcox_nonfinite', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_boxcox_nonfinite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_boxcox_nonfinite(...)' code ##################

    
    # Assigning a Call to a Name (line 39):
    
    # Call to array(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_530588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    int_530589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), list_530588, int_530589)
    # Adding element type (line 39)
    int_530590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), list_530588, int_530590)
    # Adding element type (line 39)
    float_530591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), list_530588, float_530591)
    
    # Processing the call keyword arguments (line 39)
    kwargs_530592 = {}
    # Getting the type of 'np' (line 39)
    np_530586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 39)
    array_530587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), np_530586, 'array')
    # Calling array(args, kwargs) (line 39)
    array_call_result_530593 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), array_530587, *[list_530588], **kwargs_530592)
    
    # Assigning a type to the variable 'x' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'x', array_call_result_530593)
    
    # Assigning a Call to a Name (line 40):
    
    # Call to boxcox(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'x' (line 40)
    x_530595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_530596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    float_530597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), list_530596, float_530597)
    # Adding element type (line 40)
    float_530598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), list_530596, float_530598)
    # Adding element type (line 40)
    float_530599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), list_530596, float_530599)
    
    # Processing the call keyword arguments (line 40)
    kwargs_530600 = {}
    # Getting the type of 'boxcox' (line 40)
    boxcox_530594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'boxcox', False)
    # Calling boxcox(args, kwargs) (line 40)
    boxcox_call_result_530601 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), boxcox_530594, *[x_530595, list_530596], **kwargs_530600)
    
    # Assigning a type to the variable 'y' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'y', boxcox_call_result_530601)
    
    # Call to assert_equal(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'y' (line 41)
    y_530603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'y', False)
    
    # Call to array(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_530606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    # Getting the type of 'np' (line 41)
    np_530607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 30), 'np', False)
    # Obtaining the member 'nan' of a type (line 41)
    nan_530608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 30), np_530607, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 29), list_530606, nan_530608)
    # Adding element type (line 41)
    # Getting the type of 'np' (line 41)
    np_530609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 38), 'np', False)
    # Obtaining the member 'nan' of a type (line 41)
    nan_530610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 38), np_530609, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 29), list_530606, nan_530610)
    # Adding element type (line 41)
    # Getting the type of 'np' (line 41)
    np_530611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 46), 'np', False)
    # Obtaining the member 'nan' of a type (line 41)
    nan_530612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 46), np_530611, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 29), list_530606, nan_530612)
    
    # Processing the call keyword arguments (line 41)
    kwargs_530613 = {}
    # Getting the type of 'np' (line 41)
    np_530604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'np', False)
    # Obtaining the member 'array' of a type (line 41)
    array_530605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 20), np_530604, 'array')
    # Calling array(args, kwargs) (line 41)
    array_call_result_530614 = invoke(stypy.reporting.localization.Localization(__file__, 41, 20), array_530605, *[list_530606], **kwargs_530613)
    
    # Processing the call keyword arguments (line 41)
    kwargs_530615 = {}
    # Getting the type of 'assert_equal' (line 41)
    assert_equal_530602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 41)
    assert_equal_call_result_530616 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), assert_equal_530602, *[y_530603, array_call_result_530614], **kwargs_530615)
    
    
    # Assigning a Num to a Name (line 44):
    int_530617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 8), 'int')
    # Assigning a type to the variable 'x' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'x', int_530617)
    
    # Assigning a Call to a Name (line 45):
    
    # Call to boxcox(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'x' (line 45)
    x_530619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 45)
    list_530620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 45)
    # Adding element type (line 45)
    float_530621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), list_530620, float_530621)
    # Adding element type (line 45)
    int_530622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), list_530620, int_530622)
    
    # Processing the call keyword arguments (line 45)
    kwargs_530623 = {}
    # Getting the type of 'boxcox' (line 45)
    boxcox_530618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'boxcox', False)
    # Calling boxcox(args, kwargs) (line 45)
    boxcox_call_result_530624 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), boxcox_530618, *[x_530619, list_530620], **kwargs_530623)
    
    # Assigning a type to the variable 'y' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'y', boxcox_call_result_530624)
    
    # Call to assert_equal(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'y' (line 46)
    y_530626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 17), 'y', False)
    
    # Call to array(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_530629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    
    # Getting the type of 'np' (line 46)
    np_530630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 31), 'np', False)
    # Obtaining the member 'inf' of a type (line 46)
    inf_530631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 31), np_530630, 'inf')
    # Applying the 'usub' unary operator (line 46)
    result___neg___530632 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 30), 'usub', inf_530631)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 29), list_530629, result___neg___530632)
    # Adding element type (line 46)
    
    # Getting the type of 'np' (line 46)
    np_530633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 40), 'np', False)
    # Obtaining the member 'inf' of a type (line 46)
    inf_530634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 40), np_530633, 'inf')
    # Applying the 'usub' unary operator (line 46)
    result___neg___530635 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 39), 'usub', inf_530634)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 29), list_530629, result___neg___530635)
    
    # Processing the call keyword arguments (line 46)
    kwargs_530636 = {}
    # Getting the type of 'np' (line 46)
    np_530627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'np', False)
    # Obtaining the member 'array' of a type (line 46)
    array_530628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 20), np_530627, 'array')
    # Calling array(args, kwargs) (line 46)
    array_call_result_530637 = invoke(stypy.reporting.localization.Localization(__file__, 46, 20), array_530628, *[list_530629], **kwargs_530636)
    
    # Processing the call keyword arguments (line 46)
    kwargs_530638 = {}
    # Getting the type of 'assert_equal' (line 46)
    assert_equal_530625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 46)
    assert_equal_call_result_530639 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), assert_equal_530625, *[y_530626, array_call_result_530637], **kwargs_530638)
    
    
    # ################# End of 'test_boxcox_nonfinite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_boxcox_nonfinite' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_530640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_530640)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_boxcox_nonfinite'
    return stypy_return_type_530640

# Assigning a type to the variable 'test_boxcox_nonfinite' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'test_boxcox_nonfinite', test_boxcox_nonfinite)

@norecursion
def test_boxcox1p_basic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_boxcox1p_basic'
    module_type_store = module_type_store.open_function_context('test_boxcox1p_basic', 49, 0, False)
    
    # Passed parameters checking function
    test_boxcox1p_basic.stypy_localization = localization
    test_boxcox1p_basic.stypy_type_of_self = None
    test_boxcox1p_basic.stypy_type_store = module_type_store
    test_boxcox1p_basic.stypy_function_name = 'test_boxcox1p_basic'
    test_boxcox1p_basic.stypy_param_names_list = []
    test_boxcox1p_basic.stypy_varargs_param_name = None
    test_boxcox1p_basic.stypy_kwargs_param_name = None
    test_boxcox1p_basic.stypy_call_defaults = defaults
    test_boxcox1p_basic.stypy_call_varargs = varargs
    test_boxcox1p_basic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_boxcox1p_basic', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_boxcox1p_basic', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_boxcox1p_basic(...)' code ##################

    
    # Assigning a Call to a Name (line 50):
    
    # Call to array(...): (line 50)
    # Processing the call arguments (line 50)
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_530643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    # Adding element type (line 50)
    float_530644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), list_530643, float_530644)
    # Adding element type (line 50)
    float_530645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), list_530643, float_530645)
    # Adding element type (line 50)
    int_530646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), list_530643, int_530646)
    # Adding element type (line 50)
    float_530647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 36), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), list_530643, float_530647)
    # Adding element type (line 50)
    float_530648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), list_530643, float_530648)
    # Adding element type (line 50)
    int_530649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), list_530643, int_530649)
    # Adding element type (line 50)
    int_530650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), list_530643, int_530650)
    
    # Processing the call keyword arguments (line 50)
    kwargs_530651 = {}
    # Getting the type of 'np' (line 50)
    np_530641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 50)
    array_530642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), np_530641, 'array')
    # Calling array(args, kwargs) (line 50)
    array_call_result_530652 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), array_530642, *[list_530643], **kwargs_530651)
    
    # Assigning a type to the variable 'x' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'x', array_call_result_530652)
    
    # Assigning a Call to a Name (line 53):
    
    # Call to boxcox1p(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'x' (line 53)
    x_530654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'x', False)
    int_530655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 20), 'int')
    # Processing the call keyword arguments (line 53)
    kwargs_530656 = {}
    # Getting the type of 'boxcox1p' (line 53)
    boxcox1p_530653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'boxcox1p', False)
    # Calling boxcox1p(args, kwargs) (line 53)
    boxcox1p_call_result_530657 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), boxcox1p_530653, *[x_530654, int_530655], **kwargs_530656)
    
    # Assigning a type to the variable 'y' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'y', boxcox1p_call_result_530657)
    
    # Call to assert_almost_equal(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'y' (line 54)
    y_530659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'y', False)
    
    # Call to log1p(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'x' (line 54)
    x_530662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 36), 'x', False)
    # Processing the call keyword arguments (line 54)
    kwargs_530663 = {}
    # Getting the type of 'np' (line 54)
    np_530660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 27), 'np', False)
    # Obtaining the member 'log1p' of a type (line 54)
    log1p_530661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 27), np_530660, 'log1p')
    # Calling log1p(args, kwargs) (line 54)
    log1p_call_result_530664 = invoke(stypy.reporting.localization.Localization(__file__, 54, 27), log1p_530661, *[x_530662], **kwargs_530663)
    
    # Processing the call keyword arguments (line 54)
    kwargs_530665 = {}
    # Getting the type of 'assert_almost_equal' (line 54)
    assert_almost_equal_530658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 54)
    assert_almost_equal_call_result_530666 = invoke(stypy.reporting.localization.Localization(__file__, 54, 4), assert_almost_equal_530658, *[y_530659, log1p_call_result_530664], **kwargs_530665)
    
    
    # Assigning a Call to a Name (line 57):
    
    # Call to boxcox1p(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'x' (line 57)
    x_530668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'x', False)
    int_530669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 20), 'int')
    # Processing the call keyword arguments (line 57)
    kwargs_530670 = {}
    # Getting the type of 'boxcox1p' (line 57)
    boxcox1p_530667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'boxcox1p', False)
    # Calling boxcox1p(args, kwargs) (line 57)
    boxcox1p_call_result_530671 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), boxcox1p_530667, *[x_530668, int_530669], **kwargs_530670)
    
    # Assigning a type to the variable 'y' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'y', boxcox1p_call_result_530671)
    
    # Call to assert_almost_equal(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'y' (line 58)
    y_530673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 24), 'y', False)
    # Getting the type of 'x' (line 58)
    x_530674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'x', False)
    # Processing the call keyword arguments (line 58)
    kwargs_530675 = {}
    # Getting the type of 'assert_almost_equal' (line 58)
    assert_almost_equal_530672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 58)
    assert_almost_equal_call_result_530676 = invoke(stypy.reporting.localization.Localization(__file__, 58, 4), assert_almost_equal_530672, *[y_530673, x_530674], **kwargs_530675)
    
    
    # Assigning a Call to a Name (line 61):
    
    # Call to boxcox1p(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'x' (line 61)
    x_530678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'x', False)
    int_530679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 20), 'int')
    # Processing the call keyword arguments (line 61)
    kwargs_530680 = {}
    # Getting the type of 'boxcox1p' (line 61)
    boxcox1p_530677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'boxcox1p', False)
    # Calling boxcox1p(args, kwargs) (line 61)
    boxcox1p_call_result_530681 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), boxcox1p_530677, *[x_530678, int_530679], **kwargs_530680)
    
    # Assigning a type to the variable 'y' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'y', boxcox1p_call_result_530681)
    
    # Call to assert_almost_equal(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'y' (line 62)
    y_530683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'y', False)
    float_530684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 27), 'float')
    # Getting the type of 'x' (line 62)
    x_530685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'x', False)
    # Applying the binary operator '*' (line 62)
    result_mul_530686 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 27), '*', float_530684, x_530685)
    
    int_530687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 34), 'int')
    # Getting the type of 'x' (line 62)
    x_530688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 38), 'x', False)
    # Applying the binary operator '+' (line 62)
    result_add_530689 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 34), '+', int_530687, x_530688)
    
    # Applying the binary operator '*' (line 62)
    result_mul_530690 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 32), '*', result_mul_530686, result_add_530689)
    
    # Processing the call keyword arguments (line 62)
    kwargs_530691 = {}
    # Getting the type of 'assert_almost_equal' (line 62)
    assert_almost_equal_530682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 62)
    assert_almost_equal_call_result_530692 = invoke(stypy.reporting.localization.Localization(__file__, 62, 4), assert_almost_equal_530682, *[y_530683, result_mul_530690], **kwargs_530691)
    
    
    # Assigning a Call to a Name (line 65):
    
    # Call to array(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Obtaining an instance of the builtin type 'list' (line 65)
    list_530695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 65)
    # Adding element type (line 65)
    float_530696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 20), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), list_530695, float_530696)
    # Adding element type (line 65)
    int_530697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), list_530695, int_530697)
    # Adding element type (line 65)
    int_530698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), list_530695, int_530698)
    
    # Processing the call keyword arguments (line 65)
    kwargs_530699 = {}
    # Getting the type of 'np' (line 65)
    np_530693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 65)
    array_530694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 10), np_530693, 'array')
    # Calling array(args, kwargs) (line 65)
    array_call_result_530700 = invoke(stypy.reporting.localization.Localization(__file__, 65, 10), array_530694, *[list_530695], **kwargs_530699)
    
    # Assigning a type to the variable 'lam' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'lam', array_call_result_530700)
    
    # Assigning a Call to a Name (line 66):
    
    # Call to boxcox1p(...): (line 66)
    # Processing the call arguments (line 66)
    int_530702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 17), 'int')
    # Getting the type of 'lam' (line 66)
    lam_530703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'lam', False)
    # Processing the call keyword arguments (line 66)
    kwargs_530704 = {}
    # Getting the type of 'boxcox1p' (line 66)
    boxcox1p_530701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'boxcox1p', False)
    # Calling boxcox1p(args, kwargs) (line 66)
    boxcox1p_call_result_530705 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), boxcox1p_530701, *[int_530702, lam_530703], **kwargs_530704)
    
    # Assigning a type to the variable 'y' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'y', boxcox1p_call_result_530705)
    
    # Call to assert_almost_equal(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'y' (line 67)
    y_530707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'y', False)
    float_530708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'float')
    # Getting the type of 'lam' (line 67)
    lam_530709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 34), 'lam', False)
    # Applying the binary operator 'div' (line 67)
    result_div_530710 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 27), 'div', float_530708, lam_530709)
    
    # Processing the call keyword arguments (line 67)
    kwargs_530711 = {}
    # Getting the type of 'assert_almost_equal' (line 67)
    assert_almost_equal_530706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 67)
    assert_almost_equal_call_result_530712 = invoke(stypy.reporting.localization.Localization(__file__, 67, 4), assert_almost_equal_530706, *[y_530707, result_div_530710], **kwargs_530711)
    
    
    # ################# End of 'test_boxcox1p_basic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_boxcox1p_basic' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_530713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_530713)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_boxcox1p_basic'
    return stypy_return_type_530713

# Assigning a type to the variable 'test_boxcox1p_basic' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'test_boxcox1p_basic', test_boxcox1p_basic)

@norecursion
def test_boxcox1p_underflow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_boxcox1p_underflow'
    module_type_store = module_type_store.open_function_context('test_boxcox1p_underflow', 70, 0, False)
    
    # Passed parameters checking function
    test_boxcox1p_underflow.stypy_localization = localization
    test_boxcox1p_underflow.stypy_type_of_self = None
    test_boxcox1p_underflow.stypy_type_store = module_type_store
    test_boxcox1p_underflow.stypy_function_name = 'test_boxcox1p_underflow'
    test_boxcox1p_underflow.stypy_param_names_list = []
    test_boxcox1p_underflow.stypy_varargs_param_name = None
    test_boxcox1p_underflow.stypy_kwargs_param_name = None
    test_boxcox1p_underflow.stypy_call_defaults = defaults
    test_boxcox1p_underflow.stypy_call_varargs = varargs
    test_boxcox1p_underflow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_boxcox1p_underflow', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_boxcox1p_underflow', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_boxcox1p_underflow(...)' code ##################

    
    # Assigning a Call to a Name (line 71):
    
    # Call to array(...): (line 71)
    # Processing the call arguments (line 71)
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_530716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    # Adding element type (line 71)
    float_530717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 17), list_530716, float_530717)
    # Adding element type (line 71)
    float_530718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 17), list_530716, float_530718)
    
    # Processing the call keyword arguments (line 71)
    kwargs_530719 = {}
    # Getting the type of 'np' (line 71)
    np_530714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 71)
    array_530715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), np_530714, 'array')
    # Calling array(args, kwargs) (line 71)
    array_call_result_530720 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), array_530715, *[list_530716], **kwargs_530719)
    
    # Assigning a type to the variable 'x' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'x', array_call_result_530720)
    
    # Assigning a Call to a Name (line 72):
    
    # Call to array(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Obtaining an instance of the builtin type 'list' (line 72)
    list_530723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 72)
    # Adding element type (line 72)
    float_530724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 21), list_530723, float_530724)
    # Adding element type (line 72)
    float_530725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 21), list_530723, float_530725)
    
    # Processing the call keyword arguments (line 72)
    kwargs_530726 = {}
    # Getting the type of 'np' (line 72)
    np_530721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'np', False)
    # Obtaining the member 'array' of a type (line 72)
    array_530722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), np_530721, 'array')
    # Calling array(args, kwargs) (line 72)
    array_call_result_530727 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), array_530722, *[list_530723], **kwargs_530726)
    
    # Assigning a type to the variable 'lmbda' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'lmbda', array_call_result_530727)
    
    # Assigning a Call to a Name (line 73):
    
    # Call to boxcox1p(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'x' (line 73)
    x_530729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'x', False)
    # Getting the type of 'lmbda' (line 73)
    lmbda_530730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'lmbda', False)
    # Processing the call keyword arguments (line 73)
    kwargs_530731 = {}
    # Getting the type of 'boxcox1p' (line 73)
    boxcox1p_530728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'boxcox1p', False)
    # Calling boxcox1p(args, kwargs) (line 73)
    boxcox1p_call_result_530732 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), boxcox1p_530728, *[x_530729, lmbda_530730], **kwargs_530731)
    
    # Assigning a type to the variable 'y' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'y', boxcox1p_call_result_530732)
    
    # Call to assert_allclose(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'y' (line 74)
    y_530734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'y', False)
    
    # Call to log1p(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'x' (line 74)
    x_530737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 32), 'x', False)
    # Processing the call keyword arguments (line 74)
    kwargs_530738 = {}
    # Getting the type of 'np' (line 74)
    np_530735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 23), 'np', False)
    # Obtaining the member 'log1p' of a type (line 74)
    log1p_530736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 23), np_530735, 'log1p')
    # Calling log1p(args, kwargs) (line 74)
    log1p_call_result_530739 = invoke(stypy.reporting.localization.Localization(__file__, 74, 23), log1p_530736, *[x_530737], **kwargs_530738)
    
    # Processing the call keyword arguments (line 74)
    float_530740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 41), 'float')
    keyword_530741 = float_530740
    kwargs_530742 = {'rtol': keyword_530741}
    # Getting the type of 'assert_allclose' (line 74)
    assert_allclose_530733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 74)
    assert_allclose_call_result_530743 = invoke(stypy.reporting.localization.Localization(__file__, 74, 4), assert_allclose_530733, *[y_530734, log1p_call_result_530739], **kwargs_530742)
    
    
    # ################# End of 'test_boxcox1p_underflow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_boxcox1p_underflow' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_530744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_530744)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_boxcox1p_underflow'
    return stypy_return_type_530744

# Assigning a type to the variable 'test_boxcox1p_underflow' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'test_boxcox1p_underflow', test_boxcox1p_underflow)

@norecursion
def test_boxcox1p_nonfinite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_boxcox1p_nonfinite'
    module_type_store = module_type_store.open_function_context('test_boxcox1p_nonfinite', 77, 0, False)
    
    # Passed parameters checking function
    test_boxcox1p_nonfinite.stypy_localization = localization
    test_boxcox1p_nonfinite.stypy_type_of_self = None
    test_boxcox1p_nonfinite.stypy_type_store = module_type_store
    test_boxcox1p_nonfinite.stypy_function_name = 'test_boxcox1p_nonfinite'
    test_boxcox1p_nonfinite.stypy_param_names_list = []
    test_boxcox1p_nonfinite.stypy_varargs_param_name = None
    test_boxcox1p_nonfinite.stypy_kwargs_param_name = None
    test_boxcox1p_nonfinite.stypy_call_defaults = defaults
    test_boxcox1p_nonfinite.stypy_call_varargs = varargs
    test_boxcox1p_nonfinite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_boxcox1p_nonfinite', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_boxcox1p_nonfinite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_boxcox1p_nonfinite(...)' code ##################

    
    # Assigning a Call to a Name (line 79):
    
    # Call to array(...): (line 79)
    # Processing the call arguments (line 79)
    
    # Obtaining an instance of the builtin type 'list' (line 79)
    list_530747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 79)
    # Adding element type (line 79)
    int_530748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 17), list_530747, int_530748)
    # Adding element type (line 79)
    int_530749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 17), list_530747, int_530749)
    # Adding element type (line 79)
    float_530750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 17), list_530747, float_530750)
    
    # Processing the call keyword arguments (line 79)
    kwargs_530751 = {}
    # Getting the type of 'np' (line 79)
    np_530745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 79)
    array_530746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), np_530745, 'array')
    # Calling array(args, kwargs) (line 79)
    array_call_result_530752 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), array_530746, *[list_530747], **kwargs_530751)
    
    # Assigning a type to the variable 'x' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'x', array_call_result_530752)
    
    # Assigning a Call to a Name (line 80):
    
    # Call to boxcox1p(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'x' (line 80)
    x_530754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_530755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    # Adding element type (line 80)
    float_530756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 20), list_530755, float_530756)
    # Adding element type (line 80)
    float_530757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 20), list_530755, float_530757)
    # Adding element type (line 80)
    float_530758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 20), list_530755, float_530758)
    
    # Processing the call keyword arguments (line 80)
    kwargs_530759 = {}
    # Getting the type of 'boxcox1p' (line 80)
    boxcox1p_530753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'boxcox1p', False)
    # Calling boxcox1p(args, kwargs) (line 80)
    boxcox1p_call_result_530760 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), boxcox1p_530753, *[x_530754, list_530755], **kwargs_530759)
    
    # Assigning a type to the variable 'y' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'y', boxcox1p_call_result_530760)
    
    # Call to assert_equal(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'y' (line 81)
    y_530762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'y', False)
    
    # Call to array(...): (line 81)
    # Processing the call arguments (line 81)
    
    # Obtaining an instance of the builtin type 'list' (line 81)
    list_530765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 81)
    # Adding element type (line 81)
    # Getting the type of 'np' (line 81)
    np_530766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'np', False)
    # Obtaining the member 'nan' of a type (line 81)
    nan_530767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 30), np_530766, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 29), list_530765, nan_530767)
    # Adding element type (line 81)
    # Getting the type of 'np' (line 81)
    np_530768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 38), 'np', False)
    # Obtaining the member 'nan' of a type (line 81)
    nan_530769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 38), np_530768, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 29), list_530765, nan_530769)
    # Adding element type (line 81)
    # Getting the type of 'np' (line 81)
    np_530770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 46), 'np', False)
    # Obtaining the member 'nan' of a type (line 81)
    nan_530771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 46), np_530770, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 29), list_530765, nan_530771)
    
    # Processing the call keyword arguments (line 81)
    kwargs_530772 = {}
    # Getting the type of 'np' (line 81)
    np_530763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'np', False)
    # Obtaining the member 'array' of a type (line 81)
    array_530764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 20), np_530763, 'array')
    # Calling array(args, kwargs) (line 81)
    array_call_result_530773 = invoke(stypy.reporting.localization.Localization(__file__, 81, 20), array_530764, *[list_530765], **kwargs_530772)
    
    # Processing the call keyword arguments (line 81)
    kwargs_530774 = {}
    # Getting the type of 'assert_equal' (line 81)
    assert_equal_530761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 81)
    assert_equal_call_result_530775 = invoke(stypy.reporting.localization.Localization(__file__, 81, 4), assert_equal_530761, *[y_530762, array_call_result_530773], **kwargs_530774)
    
    
    # Assigning a Num to a Name (line 84):
    int_530776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'int')
    # Assigning a type to the variable 'x' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'x', int_530776)
    
    # Assigning a Call to a Name (line 85):
    
    # Call to boxcox1p(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'x' (line 85)
    x_530778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 85)
    list_530779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 85)
    # Adding element type (line 85)
    float_530780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 20), list_530779, float_530780)
    # Adding element type (line 85)
    int_530781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 20), list_530779, int_530781)
    
    # Processing the call keyword arguments (line 85)
    kwargs_530782 = {}
    # Getting the type of 'boxcox1p' (line 85)
    boxcox1p_530777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'boxcox1p', False)
    # Calling boxcox1p(args, kwargs) (line 85)
    boxcox1p_call_result_530783 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), boxcox1p_530777, *[x_530778, list_530779], **kwargs_530782)
    
    # Assigning a type to the variable 'y' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'y', boxcox1p_call_result_530783)
    
    # Call to assert_equal(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'y' (line 86)
    y_530785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 17), 'y', False)
    
    # Call to array(...): (line 86)
    # Processing the call arguments (line 86)
    
    # Obtaining an instance of the builtin type 'list' (line 86)
    list_530788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    # Adding element type (line 86)
    
    # Getting the type of 'np' (line 86)
    np_530789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 31), 'np', False)
    # Obtaining the member 'inf' of a type (line 86)
    inf_530790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 31), np_530789, 'inf')
    # Applying the 'usub' unary operator (line 86)
    result___neg___530791 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 30), 'usub', inf_530790)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 29), list_530788, result___neg___530791)
    # Adding element type (line 86)
    
    # Getting the type of 'np' (line 86)
    np_530792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 40), 'np', False)
    # Obtaining the member 'inf' of a type (line 86)
    inf_530793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 40), np_530792, 'inf')
    # Applying the 'usub' unary operator (line 86)
    result___neg___530794 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 39), 'usub', inf_530793)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 29), list_530788, result___neg___530794)
    
    # Processing the call keyword arguments (line 86)
    kwargs_530795 = {}
    # Getting the type of 'np' (line 86)
    np_530786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'np', False)
    # Obtaining the member 'array' of a type (line 86)
    array_530787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 20), np_530786, 'array')
    # Calling array(args, kwargs) (line 86)
    array_call_result_530796 = invoke(stypy.reporting.localization.Localization(__file__, 86, 20), array_530787, *[list_530788], **kwargs_530795)
    
    # Processing the call keyword arguments (line 86)
    kwargs_530797 = {}
    # Getting the type of 'assert_equal' (line 86)
    assert_equal_530784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 86)
    assert_equal_call_result_530798 = invoke(stypy.reporting.localization.Localization(__file__, 86, 4), assert_equal_530784, *[y_530785, array_call_result_530796], **kwargs_530797)
    
    
    # ################# End of 'test_boxcox1p_nonfinite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_boxcox1p_nonfinite' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_530799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_530799)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_boxcox1p_nonfinite'
    return stypy_return_type_530799

# Assigning a type to the variable 'test_boxcox1p_nonfinite' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'test_boxcox1p_nonfinite', test_boxcox1p_nonfinite)

@norecursion
def test_inv_boxcox(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_inv_boxcox'
    module_type_store = module_type_store.open_function_context('test_inv_boxcox', 89, 0, False)
    
    # Passed parameters checking function
    test_inv_boxcox.stypy_localization = localization
    test_inv_boxcox.stypy_type_of_self = None
    test_inv_boxcox.stypy_type_store = module_type_store
    test_inv_boxcox.stypy_function_name = 'test_inv_boxcox'
    test_inv_boxcox.stypy_param_names_list = []
    test_inv_boxcox.stypy_varargs_param_name = None
    test_inv_boxcox.stypy_kwargs_param_name = None
    test_inv_boxcox.stypy_call_defaults = defaults
    test_inv_boxcox.stypy_call_varargs = varargs
    test_inv_boxcox.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_inv_boxcox', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_inv_boxcox', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_inv_boxcox(...)' code ##################

    
    # Assigning a Call to a Name (line 90):
    
    # Call to array(...): (line 90)
    # Processing the call arguments (line 90)
    
    # Obtaining an instance of the builtin type 'list' (line 90)
    list_530802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 90)
    # Adding element type (line 90)
    float_530803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_530802, float_530803)
    # Adding element type (line 90)
    float_530804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_530802, float_530804)
    # Adding element type (line 90)
    float_530805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_530802, float_530805)
    
    # Processing the call keyword arguments (line 90)
    kwargs_530806 = {}
    # Getting the type of 'np' (line 90)
    np_530800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 90)
    array_530801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), np_530800, 'array')
    # Calling array(args, kwargs) (line 90)
    array_call_result_530807 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), array_530801, *[list_530802], **kwargs_530806)
    
    # Assigning a type to the variable 'x' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'x', array_call_result_530807)
    
    # Assigning a Call to a Name (line 91):
    
    # Call to array(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Obtaining an instance of the builtin type 'list' (line 91)
    list_530810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 91)
    # Adding element type (line 91)
    float_530811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 20), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 19), list_530810, float_530811)
    # Adding element type (line 91)
    float_530812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 19), list_530810, float_530812)
    # Adding element type (line 91)
    float_530813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 19), list_530810, float_530813)
    
    # Processing the call keyword arguments (line 91)
    kwargs_530814 = {}
    # Getting the type of 'np' (line 91)
    np_530808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 91)
    array_530809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 10), np_530808, 'array')
    # Calling array(args, kwargs) (line 91)
    array_call_result_530815 = invoke(stypy.reporting.localization.Localization(__file__, 91, 10), array_530809, *[list_530810], **kwargs_530814)
    
    # Assigning a type to the variable 'lam' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'lam', array_call_result_530815)
    
    # Assigning a Call to a Name (line 92):
    
    # Call to boxcox(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'x' (line 92)
    x_530817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'x', False)
    # Getting the type of 'lam' (line 92)
    lam_530818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 18), 'lam', False)
    # Processing the call keyword arguments (line 92)
    kwargs_530819 = {}
    # Getting the type of 'boxcox' (line 92)
    boxcox_530816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'boxcox', False)
    # Calling boxcox(args, kwargs) (line 92)
    boxcox_call_result_530820 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), boxcox_530816, *[x_530817, lam_530818], **kwargs_530819)
    
    # Assigning a type to the variable 'y' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'y', boxcox_call_result_530820)
    
    # Assigning a Call to a Name (line 93):
    
    # Call to inv_boxcox(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'y' (line 93)
    y_530822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'y', False)
    # Getting the type of 'lam' (line 93)
    lam_530823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'lam', False)
    # Processing the call keyword arguments (line 93)
    kwargs_530824 = {}
    # Getting the type of 'inv_boxcox' (line 93)
    inv_boxcox_530821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 9), 'inv_boxcox', False)
    # Calling inv_boxcox(args, kwargs) (line 93)
    inv_boxcox_call_result_530825 = invoke(stypy.reporting.localization.Localization(__file__, 93, 9), inv_boxcox_530821, *[y_530822, lam_530823], **kwargs_530824)
    
    # Assigning a type to the variable 'x2' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'x2', inv_boxcox_call_result_530825)
    
    # Call to assert_almost_equal(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'x' (line 94)
    x_530827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'x', False)
    # Getting the type of 'x2' (line 94)
    x2_530828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'x2', False)
    # Processing the call keyword arguments (line 94)
    kwargs_530829 = {}
    # Getting the type of 'assert_almost_equal' (line 94)
    assert_almost_equal_530826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 94)
    assert_almost_equal_call_result_530830 = invoke(stypy.reporting.localization.Localization(__file__, 94, 4), assert_almost_equal_530826, *[x_530827, x2_530828], **kwargs_530829)
    
    
    # Assigning a Call to a Name (line 96):
    
    # Call to array(...): (line 96)
    # Processing the call arguments (line 96)
    
    # Obtaining an instance of the builtin type 'list' (line 96)
    list_530833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 96)
    # Adding element type (line 96)
    float_530834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 17), list_530833, float_530834)
    # Adding element type (line 96)
    float_530835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 17), list_530833, float_530835)
    # Adding element type (line 96)
    float_530836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 17), list_530833, float_530836)
    
    # Processing the call keyword arguments (line 96)
    kwargs_530837 = {}
    # Getting the type of 'np' (line 96)
    np_530831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 96)
    array_530832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), np_530831, 'array')
    # Calling array(args, kwargs) (line 96)
    array_call_result_530838 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), array_530832, *[list_530833], **kwargs_530837)
    
    # Assigning a type to the variable 'x' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'x', array_call_result_530838)
    
    # Assigning a Call to a Name (line 97):
    
    # Call to array(...): (line 97)
    # Processing the call arguments (line 97)
    
    # Obtaining an instance of the builtin type 'list' (line 97)
    list_530841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 97)
    # Adding element type (line 97)
    float_530842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 20), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_530841, float_530842)
    # Adding element type (line 97)
    float_530843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_530841, float_530843)
    # Adding element type (line 97)
    float_530844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_530841, float_530844)
    
    # Processing the call keyword arguments (line 97)
    kwargs_530845 = {}
    # Getting the type of 'np' (line 97)
    np_530839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 97)
    array_530840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 10), np_530839, 'array')
    # Calling array(args, kwargs) (line 97)
    array_call_result_530846 = invoke(stypy.reporting.localization.Localization(__file__, 97, 10), array_530840, *[list_530841], **kwargs_530845)
    
    # Assigning a type to the variable 'lam' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'lam', array_call_result_530846)
    
    # Assigning a Call to a Name (line 98):
    
    # Call to boxcox1p(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'x' (line 98)
    x_530848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'x', False)
    # Getting the type of 'lam' (line 98)
    lam_530849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'lam', False)
    # Processing the call keyword arguments (line 98)
    kwargs_530850 = {}
    # Getting the type of 'boxcox1p' (line 98)
    boxcox1p_530847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'boxcox1p', False)
    # Calling boxcox1p(args, kwargs) (line 98)
    boxcox1p_call_result_530851 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), boxcox1p_530847, *[x_530848, lam_530849], **kwargs_530850)
    
    # Assigning a type to the variable 'y' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'y', boxcox1p_call_result_530851)
    
    # Assigning a Call to a Name (line 99):
    
    # Call to inv_boxcox1p(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'y' (line 99)
    y_530853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'y', False)
    # Getting the type of 'lam' (line 99)
    lam_530854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'lam', False)
    # Processing the call keyword arguments (line 99)
    kwargs_530855 = {}
    # Getting the type of 'inv_boxcox1p' (line 99)
    inv_boxcox1p_530852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 9), 'inv_boxcox1p', False)
    # Calling inv_boxcox1p(args, kwargs) (line 99)
    inv_boxcox1p_call_result_530856 = invoke(stypy.reporting.localization.Localization(__file__, 99, 9), inv_boxcox1p_530852, *[y_530853, lam_530854], **kwargs_530855)
    
    # Assigning a type to the variable 'x2' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'x2', inv_boxcox1p_call_result_530856)
    
    # Call to assert_almost_equal(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'x' (line 100)
    x_530858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'x', False)
    # Getting the type of 'x2' (line 100)
    x2_530859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'x2', False)
    # Processing the call keyword arguments (line 100)
    kwargs_530860 = {}
    # Getting the type of 'assert_almost_equal' (line 100)
    assert_almost_equal_530857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 100)
    assert_almost_equal_call_result_530861 = invoke(stypy.reporting.localization.Localization(__file__, 100, 4), assert_almost_equal_530857, *[x_530858, x2_530859], **kwargs_530860)
    
    
    # ################# End of 'test_inv_boxcox(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_inv_boxcox' in the type store
    # Getting the type of 'stypy_return_type' (line 89)
    stypy_return_type_530862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_530862)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_inv_boxcox'
    return stypy_return_type_530862

# Assigning a type to the variable 'test_inv_boxcox' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'test_inv_boxcox', test_inv_boxcox)

@norecursion
def test_inv_boxcox1p_underflow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_inv_boxcox1p_underflow'
    module_type_store = module_type_store.open_function_context('test_inv_boxcox1p_underflow', 103, 0, False)
    
    # Passed parameters checking function
    test_inv_boxcox1p_underflow.stypy_localization = localization
    test_inv_boxcox1p_underflow.stypy_type_of_self = None
    test_inv_boxcox1p_underflow.stypy_type_store = module_type_store
    test_inv_boxcox1p_underflow.stypy_function_name = 'test_inv_boxcox1p_underflow'
    test_inv_boxcox1p_underflow.stypy_param_names_list = []
    test_inv_boxcox1p_underflow.stypy_varargs_param_name = None
    test_inv_boxcox1p_underflow.stypy_kwargs_param_name = None
    test_inv_boxcox1p_underflow.stypy_call_defaults = defaults
    test_inv_boxcox1p_underflow.stypy_call_varargs = varargs
    test_inv_boxcox1p_underflow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_inv_boxcox1p_underflow', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_inv_boxcox1p_underflow', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_inv_boxcox1p_underflow(...)' code ##################

    
    # Assigning a Num to a Name (line 104):
    float_530863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 8), 'float')
    # Assigning a type to the variable 'x' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'x', float_530863)
    
    # Assigning a Num to a Name (line 105):
    float_530864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 10), 'float')
    # Assigning a type to the variable 'lam' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'lam', float_530864)
    
    # Assigning a Call to a Name (line 106):
    
    # Call to inv_boxcox1p(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'x' (line 106)
    x_530866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'x', False)
    # Getting the type of 'lam' (line 106)
    lam_530867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'lam', False)
    # Processing the call keyword arguments (line 106)
    kwargs_530868 = {}
    # Getting the type of 'inv_boxcox1p' (line 106)
    inv_boxcox1p_530865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'inv_boxcox1p', False)
    # Calling inv_boxcox1p(args, kwargs) (line 106)
    inv_boxcox1p_call_result_530869 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), inv_boxcox1p_530865, *[x_530866, lam_530867], **kwargs_530868)
    
    # Assigning a type to the variable 'y' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'y', inv_boxcox1p_call_result_530869)
    
    # Call to assert_allclose(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'y' (line 107)
    y_530871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 20), 'y', False)
    # Getting the type of 'x' (line 107)
    x_530872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 23), 'x', False)
    # Processing the call keyword arguments (line 107)
    float_530873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 31), 'float')
    keyword_530874 = float_530873
    kwargs_530875 = {'rtol': keyword_530874}
    # Getting the type of 'assert_allclose' (line 107)
    assert_allclose_530870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 107)
    assert_allclose_call_result_530876 = invoke(stypy.reporting.localization.Localization(__file__, 107, 4), assert_allclose_530870, *[y_530871, x_530872], **kwargs_530875)
    
    
    # ################# End of 'test_inv_boxcox1p_underflow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_inv_boxcox1p_underflow' in the type store
    # Getting the type of 'stypy_return_type' (line 103)
    stypy_return_type_530877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_530877)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_inv_boxcox1p_underflow'
    return stypy_return_type_530877

# Assigning a type to the variable 'test_inv_boxcox1p_underflow' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'test_inv_boxcox1p_underflow', test_inv_boxcox1p_underflow)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
