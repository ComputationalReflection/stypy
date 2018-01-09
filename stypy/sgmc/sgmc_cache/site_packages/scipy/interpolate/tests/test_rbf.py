
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Created by John Travers, Robert Hetland, 2007
2: ''' Test functions for rbf module '''
3: from __future__ import division, print_function, absolute_import
4: 
5: 
6: import numpy as np
7: from numpy.testing import (assert_, assert_array_almost_equal,
8:                            assert_almost_equal)
9: from numpy import linspace, sin, random, exp, allclose
10: from scipy.interpolate.rbf import Rbf
11: 
12: FUNCTIONS = ('multiquadric', 'inverse multiquadric', 'gaussian',
13:              'cubic', 'quintic', 'thin-plate', 'linear')
14: 
15: 
16: def check_rbf1d_interpolation(function):
17:     # Check that the Rbf function interpolates through the nodes (1D)
18:     x = linspace(0,10,9)
19:     y = sin(x)
20:     rbf = Rbf(x, y, function=function)
21:     yi = rbf(x)
22:     assert_array_almost_equal(y, yi)
23:     assert_almost_equal(rbf(float(x[0])), y[0])
24: 
25: 
26: def check_rbf2d_interpolation(function):
27:     # Check that the Rbf function interpolates through the nodes (2D).
28:     x = random.rand(50,1)*4-2
29:     y = random.rand(50,1)*4-2
30:     z = x*exp(-x**2-1j*y**2)
31:     rbf = Rbf(x, y, z, epsilon=2, function=function)
32:     zi = rbf(x, y)
33:     zi.shape = x.shape
34:     assert_array_almost_equal(z, zi)
35: 
36: 
37: def check_rbf3d_interpolation(function):
38:     # Check that the Rbf function interpolates through the nodes (3D).
39:     x = random.rand(50, 1)*4 - 2
40:     y = random.rand(50, 1)*4 - 2
41:     z = random.rand(50, 1)*4 - 2
42:     d = x*exp(-x**2 - y**2)
43:     rbf = Rbf(x, y, z, d, epsilon=2, function=function)
44:     di = rbf(x, y, z)
45:     di.shape = x.shape
46:     assert_array_almost_equal(di, d)
47: 
48: 
49: def test_rbf_interpolation():
50:     for function in FUNCTIONS:
51:         check_rbf1d_interpolation(function)
52:         check_rbf2d_interpolation(function)
53:         check_rbf3d_interpolation(function)
54: 
55: 
56: def check_rbf1d_regularity(function, atol):
57:     # Check that the Rbf function approximates a smooth function well away
58:     # from the nodes.
59:     x = linspace(0, 10, 9)
60:     y = sin(x)
61:     rbf = Rbf(x, y, function=function)
62:     xi = linspace(0, 10, 100)
63:     yi = rbf(xi)
64:     # import matplotlib.pyplot as plt
65:     # plt.figure()
66:     # plt.plot(x, y, 'o', xi, sin(xi), ':', xi, yi, '-')
67:     # plt.plot(x, y, 'o', xi, yi-sin(xi), ':')
68:     # plt.title(function)
69:     # plt.show()
70:     msg = "abs-diff: %f" % abs(yi - sin(xi)).max()
71:     assert_(allclose(yi, sin(xi), atol=atol), msg)
72: 
73: 
74: def test_rbf_regularity():
75:     tolerances = {
76:         'multiquadric': 0.1,
77:         'inverse multiquadric': 0.15,
78:         'gaussian': 0.15,
79:         'cubic': 0.15,
80:         'quintic': 0.1,
81:         'thin-plate': 0.1,
82:         'linear': 0.2
83:     }
84:     for function in FUNCTIONS:
85:         check_rbf1d_regularity(function, tolerances.get(function, 1e-2))
86: 
87: 
88: def check_rbf1d_stability(function):
89:     # Check that the Rbf function with default epsilon is not subject 
90:     # to overshoot.  Regression for issue #4523.
91:     #
92:     # Generate some data (fixed random seed hence deterministic) 
93:     np.random.seed(1234)
94:     x = np.linspace(0, 10, 50)
95:     z = x + 4.0 * np.random.randn(len(x))
96: 
97:     rbf = Rbf(x, z, function=function)
98:     xi = np.linspace(0, 10, 1000)
99:     yi = rbf(xi)
100: 
101:     # subtract the linear trend and make sure there no spikes
102:     assert_(np.abs(yi-xi).max() / np.abs(z-x).max() < 1.1)
103: 
104: def test_rbf_stability():
105:     for function in FUNCTIONS:
106:         check_rbf1d_stability(function)
107: 
108: 
109: def test_default_construction():
110:     # Check that the Rbf class can be constructed with the default
111:     # multiquadric basis function. Regression test for ticket #1228.
112:     x = linspace(0,10,9)
113:     y = sin(x)
114:     rbf = Rbf(x, y)
115:     yi = rbf(x)
116:     assert_array_almost_equal(y, yi)
117: 
118: 
119: def test_function_is_callable():
120:     # Check that the Rbf class can be constructed with function=callable.
121:     x = linspace(0,10,9)
122:     y = sin(x)
123:     linfunc = lambda x:x
124:     rbf = Rbf(x, y, function=linfunc)
125:     yi = rbf(x)
126:     assert_array_almost_equal(y, yi)
127: 
128: 
129: def test_two_arg_function_is_callable():
130:     # Check that the Rbf class can be constructed with a two argument
131:     # function=callable.
132:     def _func(self, r):
133:         return self.epsilon + r
134: 
135:     x = linspace(0,10,9)
136:     y = sin(x)
137:     rbf = Rbf(x, y, function=_func)
138:     yi = rbf(x)
139:     assert_array_almost_equal(y, yi)
140: 
141: 
142: def test_rbf_epsilon_none():
143:     x = linspace(0, 10, 9)
144:     y = sin(x)
145:     rbf = Rbf(x, y, epsilon=None)
146: 
147: 
148: def test_rbf_epsilon_none_collinear():
149:     # Check that collinear points in one dimension doesn't cause an error
150:     # due to epsilon = 0
151:     x = [1, 2, 3]
152:     y = [4, 4, 4]
153:     z = [5, 6, 7]
154:     rbf = Rbf(x, y, z, epsilon=None)
155:     assert_(rbf.epsilon > 0)
156: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_119051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 0), 'str', ' Test functions for rbf module ')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_119052 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_119052) is not StypyTypeError):

    if (import_119052 != 'pyd_module'):
        __import__(import_119052)
        sys_modules_119053 = sys.modules[import_119052]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_119053.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_119052)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_, assert_array_almost_equal, assert_almost_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_119054 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_119054) is not StypyTypeError):

    if (import_119054 != 'pyd_module'):
        __import__(import_119054)
        sys_modules_119055 = sys.modules[import_119054]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_119055.module_type_store, module_type_store, ['assert_', 'assert_array_almost_equal', 'assert_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_119055, sys_modules_119055.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_array_almost_equal, assert_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_array_almost_equal', 'assert_almost_equal'], [assert_, assert_array_almost_equal, assert_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_119054)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy import linspace, sin, random, exp, allclose' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_119056 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_119056) is not StypyTypeError):

    if (import_119056 != 'pyd_module'):
        __import__(import_119056)
        sys_modules_119057 = sys.modules[import_119056]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', sys_modules_119057.module_type_store, module_type_store, ['linspace', 'sin', 'random', 'exp', 'allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_119057, sys_modules_119057.module_type_store, module_type_store)
    else:
        from numpy import linspace, sin, random, exp, allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', None, module_type_store, ['linspace', 'sin', 'random', 'exp', 'allclose'], [linspace, sin, random, exp, allclose])

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_119056)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.interpolate.rbf import Rbf' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_119058 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.rbf')

if (type(import_119058) is not StypyTypeError):

    if (import_119058 != 'pyd_module'):
        __import__(import_119058)
        sys_modules_119059 = sys.modules[import_119058]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.rbf', sys_modules_119059.module_type_store, module_type_store, ['Rbf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_119059, sys_modules_119059.module_type_store, module_type_store)
    else:
        from scipy.interpolate.rbf import Rbf

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.rbf', None, module_type_store, ['Rbf'], [Rbf])

else:
    # Assigning a type to the variable 'scipy.interpolate.rbf' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.rbf', import_119058)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')


# Assigning a Tuple to a Name (line 12):

# Obtaining an instance of the builtin type 'tuple' (line 12)
tuple_119060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 12)
# Adding element type (line 12)
str_119061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 13), 'str', 'multiquadric')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 13), tuple_119060, str_119061)
# Adding element type (line 12)
str_119062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 29), 'str', 'inverse multiquadric')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 13), tuple_119060, str_119062)
# Adding element type (line 12)
str_119063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 53), 'str', 'gaussian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 13), tuple_119060, str_119063)
# Adding element type (line 12)
str_119064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 13), 'str', 'cubic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 13), tuple_119060, str_119064)
# Adding element type (line 12)
str_119065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'str', 'quintic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 13), tuple_119060, str_119065)
# Adding element type (line 12)
str_119066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 33), 'str', 'thin-plate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 13), tuple_119060, str_119066)
# Adding element type (line 12)
str_119067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 47), 'str', 'linear')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 13), tuple_119060, str_119067)

# Assigning a type to the variable 'FUNCTIONS' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'FUNCTIONS', tuple_119060)

@norecursion
def check_rbf1d_interpolation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_rbf1d_interpolation'
    module_type_store = module_type_store.open_function_context('check_rbf1d_interpolation', 16, 0, False)
    
    # Passed parameters checking function
    check_rbf1d_interpolation.stypy_localization = localization
    check_rbf1d_interpolation.stypy_type_of_self = None
    check_rbf1d_interpolation.stypy_type_store = module_type_store
    check_rbf1d_interpolation.stypy_function_name = 'check_rbf1d_interpolation'
    check_rbf1d_interpolation.stypy_param_names_list = ['function']
    check_rbf1d_interpolation.stypy_varargs_param_name = None
    check_rbf1d_interpolation.stypy_kwargs_param_name = None
    check_rbf1d_interpolation.stypy_call_defaults = defaults
    check_rbf1d_interpolation.stypy_call_varargs = varargs
    check_rbf1d_interpolation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_rbf1d_interpolation', ['function'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_rbf1d_interpolation', localization, ['function'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_rbf1d_interpolation(...)' code ##################

    
    # Assigning a Call to a Name (line 18):
    
    # Call to linspace(...): (line 18)
    # Processing the call arguments (line 18)
    int_119069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 17), 'int')
    int_119070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 19), 'int')
    int_119071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_119072 = {}
    # Getting the type of 'linspace' (line 18)
    linspace_119068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'linspace', False)
    # Calling linspace(args, kwargs) (line 18)
    linspace_call_result_119073 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), linspace_119068, *[int_119069, int_119070, int_119071], **kwargs_119072)
    
    # Assigning a type to the variable 'x' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'x', linspace_call_result_119073)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to sin(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'x' (line 19)
    x_119075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'x', False)
    # Processing the call keyword arguments (line 19)
    kwargs_119076 = {}
    # Getting the type of 'sin' (line 19)
    sin_119074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'sin', False)
    # Calling sin(args, kwargs) (line 19)
    sin_call_result_119077 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), sin_119074, *[x_119075], **kwargs_119076)
    
    # Assigning a type to the variable 'y' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'y', sin_call_result_119077)
    
    # Assigning a Call to a Name (line 20):
    
    # Call to Rbf(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'x' (line 20)
    x_119079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'x', False)
    # Getting the type of 'y' (line 20)
    y_119080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'y', False)
    # Processing the call keyword arguments (line 20)
    # Getting the type of 'function' (line 20)
    function_119081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 29), 'function', False)
    keyword_119082 = function_119081
    kwargs_119083 = {'function': keyword_119082}
    # Getting the type of 'Rbf' (line 20)
    Rbf_119078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'Rbf', False)
    # Calling Rbf(args, kwargs) (line 20)
    Rbf_call_result_119084 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), Rbf_119078, *[x_119079, y_119080], **kwargs_119083)
    
    # Assigning a type to the variable 'rbf' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'rbf', Rbf_call_result_119084)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to rbf(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'x' (line 21)
    x_119086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 13), 'x', False)
    # Processing the call keyword arguments (line 21)
    kwargs_119087 = {}
    # Getting the type of 'rbf' (line 21)
    rbf_119085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 9), 'rbf', False)
    # Calling rbf(args, kwargs) (line 21)
    rbf_call_result_119088 = invoke(stypy.reporting.localization.Localization(__file__, 21, 9), rbf_119085, *[x_119086], **kwargs_119087)
    
    # Assigning a type to the variable 'yi' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'yi', rbf_call_result_119088)
    
    # Call to assert_array_almost_equal(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'y' (line 22)
    y_119090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 30), 'y', False)
    # Getting the type of 'yi' (line 22)
    yi_119091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 33), 'yi', False)
    # Processing the call keyword arguments (line 22)
    kwargs_119092 = {}
    # Getting the type of 'assert_array_almost_equal' (line 22)
    assert_array_almost_equal_119089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 22)
    assert_array_almost_equal_call_result_119093 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), assert_array_almost_equal_119089, *[y_119090, yi_119091], **kwargs_119092)
    
    
    # Call to assert_almost_equal(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to rbf(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to float(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Obtaining the type of the subscript
    int_119097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 36), 'int')
    # Getting the type of 'x' (line 23)
    x_119098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 34), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 23)
    getitem___119099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 34), x_119098, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 23)
    subscript_call_result_119100 = invoke(stypy.reporting.localization.Localization(__file__, 23, 34), getitem___119099, int_119097)
    
    # Processing the call keyword arguments (line 23)
    kwargs_119101 = {}
    # Getting the type of 'float' (line 23)
    float_119096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 28), 'float', False)
    # Calling float(args, kwargs) (line 23)
    float_call_result_119102 = invoke(stypy.reporting.localization.Localization(__file__, 23, 28), float_119096, *[subscript_call_result_119100], **kwargs_119101)
    
    # Processing the call keyword arguments (line 23)
    kwargs_119103 = {}
    # Getting the type of 'rbf' (line 23)
    rbf_119095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'rbf', False)
    # Calling rbf(args, kwargs) (line 23)
    rbf_call_result_119104 = invoke(stypy.reporting.localization.Localization(__file__, 23, 24), rbf_119095, *[float_call_result_119102], **kwargs_119103)
    
    
    # Obtaining the type of the subscript
    int_119105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 44), 'int')
    # Getting the type of 'y' (line 23)
    y_119106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 42), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 23)
    getitem___119107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 42), y_119106, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 23)
    subscript_call_result_119108 = invoke(stypy.reporting.localization.Localization(__file__, 23, 42), getitem___119107, int_119105)
    
    # Processing the call keyword arguments (line 23)
    kwargs_119109 = {}
    # Getting the type of 'assert_almost_equal' (line 23)
    assert_almost_equal_119094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 23)
    assert_almost_equal_call_result_119110 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), assert_almost_equal_119094, *[rbf_call_result_119104, subscript_call_result_119108], **kwargs_119109)
    
    
    # ################# End of 'check_rbf1d_interpolation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_rbf1d_interpolation' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_119111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119111)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_rbf1d_interpolation'
    return stypy_return_type_119111

# Assigning a type to the variable 'check_rbf1d_interpolation' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'check_rbf1d_interpolation', check_rbf1d_interpolation)

@norecursion
def check_rbf2d_interpolation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_rbf2d_interpolation'
    module_type_store = module_type_store.open_function_context('check_rbf2d_interpolation', 26, 0, False)
    
    # Passed parameters checking function
    check_rbf2d_interpolation.stypy_localization = localization
    check_rbf2d_interpolation.stypy_type_of_self = None
    check_rbf2d_interpolation.stypy_type_store = module_type_store
    check_rbf2d_interpolation.stypy_function_name = 'check_rbf2d_interpolation'
    check_rbf2d_interpolation.stypy_param_names_list = ['function']
    check_rbf2d_interpolation.stypy_varargs_param_name = None
    check_rbf2d_interpolation.stypy_kwargs_param_name = None
    check_rbf2d_interpolation.stypy_call_defaults = defaults
    check_rbf2d_interpolation.stypy_call_varargs = varargs
    check_rbf2d_interpolation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_rbf2d_interpolation', ['function'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_rbf2d_interpolation', localization, ['function'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_rbf2d_interpolation(...)' code ##################

    
    # Assigning a BinOp to a Name (line 28):
    
    # Call to rand(...): (line 28)
    # Processing the call arguments (line 28)
    int_119114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 20), 'int')
    int_119115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 23), 'int')
    # Processing the call keyword arguments (line 28)
    kwargs_119116 = {}
    # Getting the type of 'random' (line 28)
    random_119112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'random', False)
    # Obtaining the member 'rand' of a type (line 28)
    rand_119113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), random_119112, 'rand')
    # Calling rand(args, kwargs) (line 28)
    rand_call_result_119117 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), rand_119113, *[int_119114, int_119115], **kwargs_119116)
    
    int_119118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 26), 'int')
    # Applying the binary operator '*' (line 28)
    result_mul_119119 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 8), '*', rand_call_result_119117, int_119118)
    
    int_119120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'int')
    # Applying the binary operator '-' (line 28)
    result_sub_119121 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 8), '-', result_mul_119119, int_119120)
    
    # Assigning a type to the variable 'x' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'x', result_sub_119121)
    
    # Assigning a BinOp to a Name (line 29):
    
    # Call to rand(...): (line 29)
    # Processing the call arguments (line 29)
    int_119124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 20), 'int')
    int_119125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 23), 'int')
    # Processing the call keyword arguments (line 29)
    kwargs_119126 = {}
    # Getting the type of 'random' (line 29)
    random_119122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'random', False)
    # Obtaining the member 'rand' of a type (line 29)
    rand_119123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), random_119122, 'rand')
    # Calling rand(args, kwargs) (line 29)
    rand_call_result_119127 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), rand_119123, *[int_119124, int_119125], **kwargs_119126)
    
    int_119128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'int')
    # Applying the binary operator '*' (line 29)
    result_mul_119129 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 8), '*', rand_call_result_119127, int_119128)
    
    int_119130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'int')
    # Applying the binary operator '-' (line 29)
    result_sub_119131 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 8), '-', result_mul_119129, int_119130)
    
    # Assigning a type to the variable 'y' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'y', result_sub_119131)
    
    # Assigning a BinOp to a Name (line 30):
    # Getting the type of 'x' (line 30)
    x_119132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'x')
    
    # Call to exp(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Getting the type of 'x' (line 30)
    x_119134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'x', False)
    int_119135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 18), 'int')
    # Applying the binary operator '**' (line 30)
    result_pow_119136 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 15), '**', x_119134, int_119135)
    
    # Applying the 'usub' unary operator (line 30)
    result___neg___119137 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 14), 'usub', result_pow_119136)
    
    complex_119138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 20), 'complex')
    # Getting the type of 'y' (line 30)
    y_119139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'y', False)
    int_119140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 26), 'int')
    # Applying the binary operator '**' (line 30)
    result_pow_119141 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 23), '**', y_119139, int_119140)
    
    # Applying the binary operator '*' (line 30)
    result_mul_119142 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 20), '*', complex_119138, result_pow_119141)
    
    # Applying the binary operator '-' (line 30)
    result_sub_119143 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 14), '-', result___neg___119137, result_mul_119142)
    
    # Processing the call keyword arguments (line 30)
    kwargs_119144 = {}
    # Getting the type of 'exp' (line 30)
    exp_119133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 10), 'exp', False)
    # Calling exp(args, kwargs) (line 30)
    exp_call_result_119145 = invoke(stypy.reporting.localization.Localization(__file__, 30, 10), exp_119133, *[result_sub_119143], **kwargs_119144)
    
    # Applying the binary operator '*' (line 30)
    result_mul_119146 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 8), '*', x_119132, exp_call_result_119145)
    
    # Assigning a type to the variable 'z' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'z', result_mul_119146)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to Rbf(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'x' (line 31)
    x_119148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'x', False)
    # Getting the type of 'y' (line 31)
    y_119149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'y', False)
    # Getting the type of 'z' (line 31)
    z_119150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'z', False)
    # Processing the call keyword arguments (line 31)
    int_119151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'int')
    keyword_119152 = int_119151
    # Getting the type of 'function' (line 31)
    function_119153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 43), 'function', False)
    keyword_119154 = function_119153
    kwargs_119155 = {'function': keyword_119154, 'epsilon': keyword_119152}
    # Getting the type of 'Rbf' (line 31)
    Rbf_119147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'Rbf', False)
    # Calling Rbf(args, kwargs) (line 31)
    Rbf_call_result_119156 = invoke(stypy.reporting.localization.Localization(__file__, 31, 10), Rbf_119147, *[x_119148, y_119149, z_119150], **kwargs_119155)
    
    # Assigning a type to the variable 'rbf' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'rbf', Rbf_call_result_119156)
    
    # Assigning a Call to a Name (line 32):
    
    # Call to rbf(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'x' (line 32)
    x_119158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 13), 'x', False)
    # Getting the type of 'y' (line 32)
    y_119159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'y', False)
    # Processing the call keyword arguments (line 32)
    kwargs_119160 = {}
    # Getting the type of 'rbf' (line 32)
    rbf_119157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 9), 'rbf', False)
    # Calling rbf(args, kwargs) (line 32)
    rbf_call_result_119161 = invoke(stypy.reporting.localization.Localization(__file__, 32, 9), rbf_119157, *[x_119158, y_119159], **kwargs_119160)
    
    # Assigning a type to the variable 'zi' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'zi', rbf_call_result_119161)
    
    # Assigning a Attribute to a Attribute (line 33):
    # Getting the type of 'x' (line 33)
    x_119162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'x')
    # Obtaining the member 'shape' of a type (line 33)
    shape_119163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 15), x_119162, 'shape')
    # Getting the type of 'zi' (line 33)
    zi_119164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'zi')
    # Setting the type of the member 'shape' of a type (line 33)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 4), zi_119164, 'shape', shape_119163)
    
    # Call to assert_array_almost_equal(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'z' (line 34)
    z_119166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'z', False)
    # Getting the type of 'zi' (line 34)
    zi_119167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 33), 'zi', False)
    # Processing the call keyword arguments (line 34)
    kwargs_119168 = {}
    # Getting the type of 'assert_array_almost_equal' (line 34)
    assert_array_almost_equal_119165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 34)
    assert_array_almost_equal_call_result_119169 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), assert_array_almost_equal_119165, *[z_119166, zi_119167], **kwargs_119168)
    
    
    # ################# End of 'check_rbf2d_interpolation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_rbf2d_interpolation' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_119170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119170)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_rbf2d_interpolation'
    return stypy_return_type_119170

# Assigning a type to the variable 'check_rbf2d_interpolation' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'check_rbf2d_interpolation', check_rbf2d_interpolation)

@norecursion
def check_rbf3d_interpolation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_rbf3d_interpolation'
    module_type_store = module_type_store.open_function_context('check_rbf3d_interpolation', 37, 0, False)
    
    # Passed parameters checking function
    check_rbf3d_interpolation.stypy_localization = localization
    check_rbf3d_interpolation.stypy_type_of_self = None
    check_rbf3d_interpolation.stypy_type_store = module_type_store
    check_rbf3d_interpolation.stypy_function_name = 'check_rbf3d_interpolation'
    check_rbf3d_interpolation.stypy_param_names_list = ['function']
    check_rbf3d_interpolation.stypy_varargs_param_name = None
    check_rbf3d_interpolation.stypy_kwargs_param_name = None
    check_rbf3d_interpolation.stypy_call_defaults = defaults
    check_rbf3d_interpolation.stypy_call_varargs = varargs
    check_rbf3d_interpolation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_rbf3d_interpolation', ['function'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_rbf3d_interpolation', localization, ['function'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_rbf3d_interpolation(...)' code ##################

    
    # Assigning a BinOp to a Name (line 39):
    
    # Call to rand(...): (line 39)
    # Processing the call arguments (line 39)
    int_119173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'int')
    int_119174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'int')
    # Processing the call keyword arguments (line 39)
    kwargs_119175 = {}
    # Getting the type of 'random' (line 39)
    random_119171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'random', False)
    # Obtaining the member 'rand' of a type (line 39)
    rand_119172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), random_119171, 'rand')
    # Calling rand(args, kwargs) (line 39)
    rand_call_result_119176 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), rand_119172, *[int_119173, int_119174], **kwargs_119175)
    
    int_119177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 27), 'int')
    # Applying the binary operator '*' (line 39)
    result_mul_119178 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 8), '*', rand_call_result_119176, int_119177)
    
    int_119179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 31), 'int')
    # Applying the binary operator '-' (line 39)
    result_sub_119180 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 8), '-', result_mul_119178, int_119179)
    
    # Assigning a type to the variable 'x' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'x', result_sub_119180)
    
    # Assigning a BinOp to a Name (line 40):
    
    # Call to rand(...): (line 40)
    # Processing the call arguments (line 40)
    int_119183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'int')
    int_119184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'int')
    # Processing the call keyword arguments (line 40)
    kwargs_119185 = {}
    # Getting the type of 'random' (line 40)
    random_119181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'random', False)
    # Obtaining the member 'rand' of a type (line 40)
    rand_119182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), random_119181, 'rand')
    # Calling rand(args, kwargs) (line 40)
    rand_call_result_119186 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), rand_119182, *[int_119183, int_119184], **kwargs_119185)
    
    int_119187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 27), 'int')
    # Applying the binary operator '*' (line 40)
    result_mul_119188 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 8), '*', rand_call_result_119186, int_119187)
    
    int_119189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 31), 'int')
    # Applying the binary operator '-' (line 40)
    result_sub_119190 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 8), '-', result_mul_119188, int_119189)
    
    # Assigning a type to the variable 'y' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'y', result_sub_119190)
    
    # Assigning a BinOp to a Name (line 41):
    
    # Call to rand(...): (line 41)
    # Processing the call arguments (line 41)
    int_119193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'int')
    int_119194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 24), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_119195 = {}
    # Getting the type of 'random' (line 41)
    random_119191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'random', False)
    # Obtaining the member 'rand' of a type (line 41)
    rand_119192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), random_119191, 'rand')
    # Calling rand(args, kwargs) (line 41)
    rand_call_result_119196 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), rand_119192, *[int_119193, int_119194], **kwargs_119195)
    
    int_119197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 27), 'int')
    # Applying the binary operator '*' (line 41)
    result_mul_119198 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 8), '*', rand_call_result_119196, int_119197)
    
    int_119199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 31), 'int')
    # Applying the binary operator '-' (line 41)
    result_sub_119200 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 8), '-', result_mul_119198, int_119199)
    
    # Assigning a type to the variable 'z' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'z', result_sub_119200)
    
    # Assigning a BinOp to a Name (line 42):
    # Getting the type of 'x' (line 42)
    x_119201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'x')
    
    # Call to exp(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Getting the type of 'x' (line 42)
    x_119203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'x', False)
    int_119204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 18), 'int')
    # Applying the binary operator '**' (line 42)
    result_pow_119205 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 15), '**', x_119203, int_119204)
    
    # Applying the 'usub' unary operator (line 42)
    result___neg___119206 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 14), 'usub', result_pow_119205)
    
    # Getting the type of 'y' (line 42)
    y_119207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'y', False)
    int_119208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'int')
    # Applying the binary operator '**' (line 42)
    result_pow_119209 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 22), '**', y_119207, int_119208)
    
    # Applying the binary operator '-' (line 42)
    result_sub_119210 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 14), '-', result___neg___119206, result_pow_119209)
    
    # Processing the call keyword arguments (line 42)
    kwargs_119211 = {}
    # Getting the type of 'exp' (line 42)
    exp_119202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 10), 'exp', False)
    # Calling exp(args, kwargs) (line 42)
    exp_call_result_119212 = invoke(stypy.reporting.localization.Localization(__file__, 42, 10), exp_119202, *[result_sub_119210], **kwargs_119211)
    
    # Applying the binary operator '*' (line 42)
    result_mul_119213 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 8), '*', x_119201, exp_call_result_119212)
    
    # Assigning a type to the variable 'd' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'd', result_mul_119213)
    
    # Assigning a Call to a Name (line 43):
    
    # Call to Rbf(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'x' (line 43)
    x_119215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'x', False)
    # Getting the type of 'y' (line 43)
    y_119216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'y', False)
    # Getting the type of 'z' (line 43)
    z_119217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'z', False)
    # Getting the type of 'd' (line 43)
    d_119218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'd', False)
    # Processing the call keyword arguments (line 43)
    int_119219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'int')
    keyword_119220 = int_119219
    # Getting the type of 'function' (line 43)
    function_119221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 46), 'function', False)
    keyword_119222 = function_119221
    kwargs_119223 = {'function': keyword_119222, 'epsilon': keyword_119220}
    # Getting the type of 'Rbf' (line 43)
    Rbf_119214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'Rbf', False)
    # Calling Rbf(args, kwargs) (line 43)
    Rbf_call_result_119224 = invoke(stypy.reporting.localization.Localization(__file__, 43, 10), Rbf_119214, *[x_119215, y_119216, z_119217, d_119218], **kwargs_119223)
    
    # Assigning a type to the variable 'rbf' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'rbf', Rbf_call_result_119224)
    
    # Assigning a Call to a Name (line 44):
    
    # Call to rbf(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'x' (line 44)
    x_119226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'x', False)
    # Getting the type of 'y' (line 44)
    y_119227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'y', False)
    # Getting the type of 'z' (line 44)
    z_119228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'z', False)
    # Processing the call keyword arguments (line 44)
    kwargs_119229 = {}
    # Getting the type of 'rbf' (line 44)
    rbf_119225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 9), 'rbf', False)
    # Calling rbf(args, kwargs) (line 44)
    rbf_call_result_119230 = invoke(stypy.reporting.localization.Localization(__file__, 44, 9), rbf_119225, *[x_119226, y_119227, z_119228], **kwargs_119229)
    
    # Assigning a type to the variable 'di' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'di', rbf_call_result_119230)
    
    # Assigning a Attribute to a Attribute (line 45):
    # Getting the type of 'x' (line 45)
    x_119231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'x')
    # Obtaining the member 'shape' of a type (line 45)
    shape_119232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 15), x_119231, 'shape')
    # Getting the type of 'di' (line 45)
    di_119233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'di')
    # Setting the type of the member 'shape' of a type (line 45)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 4), di_119233, 'shape', shape_119232)
    
    # Call to assert_array_almost_equal(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'di' (line 46)
    di_119235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 30), 'di', False)
    # Getting the type of 'd' (line 46)
    d_119236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'd', False)
    # Processing the call keyword arguments (line 46)
    kwargs_119237 = {}
    # Getting the type of 'assert_array_almost_equal' (line 46)
    assert_array_almost_equal_119234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 46)
    assert_array_almost_equal_call_result_119238 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), assert_array_almost_equal_119234, *[di_119235, d_119236], **kwargs_119237)
    
    
    # ################# End of 'check_rbf3d_interpolation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_rbf3d_interpolation' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_119239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119239)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_rbf3d_interpolation'
    return stypy_return_type_119239

# Assigning a type to the variable 'check_rbf3d_interpolation' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'check_rbf3d_interpolation', check_rbf3d_interpolation)

@norecursion
def test_rbf_interpolation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_rbf_interpolation'
    module_type_store = module_type_store.open_function_context('test_rbf_interpolation', 49, 0, False)
    
    # Passed parameters checking function
    test_rbf_interpolation.stypy_localization = localization
    test_rbf_interpolation.stypy_type_of_self = None
    test_rbf_interpolation.stypy_type_store = module_type_store
    test_rbf_interpolation.stypy_function_name = 'test_rbf_interpolation'
    test_rbf_interpolation.stypy_param_names_list = []
    test_rbf_interpolation.stypy_varargs_param_name = None
    test_rbf_interpolation.stypy_kwargs_param_name = None
    test_rbf_interpolation.stypy_call_defaults = defaults
    test_rbf_interpolation.stypy_call_varargs = varargs
    test_rbf_interpolation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_rbf_interpolation', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_rbf_interpolation', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_rbf_interpolation(...)' code ##################

    
    # Getting the type of 'FUNCTIONS' (line 50)
    FUNCTIONS_119240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'FUNCTIONS')
    # Testing the type of a for loop iterable (line 50)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 4), FUNCTIONS_119240)
    # Getting the type of the for loop variable (line 50)
    for_loop_var_119241 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 4), FUNCTIONS_119240)
    # Assigning a type to the variable 'function' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'function', for_loop_var_119241)
    # SSA begins for a for statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check_rbf1d_interpolation(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'function' (line 51)
    function_119243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 34), 'function', False)
    # Processing the call keyword arguments (line 51)
    kwargs_119244 = {}
    # Getting the type of 'check_rbf1d_interpolation' (line 51)
    check_rbf1d_interpolation_119242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'check_rbf1d_interpolation', False)
    # Calling check_rbf1d_interpolation(args, kwargs) (line 51)
    check_rbf1d_interpolation_call_result_119245 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), check_rbf1d_interpolation_119242, *[function_119243], **kwargs_119244)
    
    
    # Call to check_rbf2d_interpolation(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'function' (line 52)
    function_119247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 34), 'function', False)
    # Processing the call keyword arguments (line 52)
    kwargs_119248 = {}
    # Getting the type of 'check_rbf2d_interpolation' (line 52)
    check_rbf2d_interpolation_119246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'check_rbf2d_interpolation', False)
    # Calling check_rbf2d_interpolation(args, kwargs) (line 52)
    check_rbf2d_interpolation_call_result_119249 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), check_rbf2d_interpolation_119246, *[function_119247], **kwargs_119248)
    
    
    # Call to check_rbf3d_interpolation(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'function' (line 53)
    function_119251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'function', False)
    # Processing the call keyword arguments (line 53)
    kwargs_119252 = {}
    # Getting the type of 'check_rbf3d_interpolation' (line 53)
    check_rbf3d_interpolation_119250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'check_rbf3d_interpolation', False)
    # Calling check_rbf3d_interpolation(args, kwargs) (line 53)
    check_rbf3d_interpolation_call_result_119253 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), check_rbf3d_interpolation_119250, *[function_119251], **kwargs_119252)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_rbf_interpolation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_rbf_interpolation' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_119254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119254)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_rbf_interpolation'
    return stypy_return_type_119254

# Assigning a type to the variable 'test_rbf_interpolation' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'test_rbf_interpolation', test_rbf_interpolation)

@norecursion
def check_rbf1d_regularity(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_rbf1d_regularity'
    module_type_store = module_type_store.open_function_context('check_rbf1d_regularity', 56, 0, False)
    
    # Passed parameters checking function
    check_rbf1d_regularity.stypy_localization = localization
    check_rbf1d_regularity.stypy_type_of_self = None
    check_rbf1d_regularity.stypy_type_store = module_type_store
    check_rbf1d_regularity.stypy_function_name = 'check_rbf1d_regularity'
    check_rbf1d_regularity.stypy_param_names_list = ['function', 'atol']
    check_rbf1d_regularity.stypy_varargs_param_name = None
    check_rbf1d_regularity.stypy_kwargs_param_name = None
    check_rbf1d_regularity.stypy_call_defaults = defaults
    check_rbf1d_regularity.stypy_call_varargs = varargs
    check_rbf1d_regularity.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_rbf1d_regularity', ['function', 'atol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_rbf1d_regularity', localization, ['function', 'atol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_rbf1d_regularity(...)' code ##################

    
    # Assigning a Call to a Name (line 59):
    
    # Call to linspace(...): (line 59)
    # Processing the call arguments (line 59)
    int_119256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 17), 'int')
    int_119257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 20), 'int')
    int_119258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 24), 'int')
    # Processing the call keyword arguments (line 59)
    kwargs_119259 = {}
    # Getting the type of 'linspace' (line 59)
    linspace_119255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'linspace', False)
    # Calling linspace(args, kwargs) (line 59)
    linspace_call_result_119260 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), linspace_119255, *[int_119256, int_119257, int_119258], **kwargs_119259)
    
    # Assigning a type to the variable 'x' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'x', linspace_call_result_119260)
    
    # Assigning a Call to a Name (line 60):
    
    # Call to sin(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'x' (line 60)
    x_119262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'x', False)
    # Processing the call keyword arguments (line 60)
    kwargs_119263 = {}
    # Getting the type of 'sin' (line 60)
    sin_119261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'sin', False)
    # Calling sin(args, kwargs) (line 60)
    sin_call_result_119264 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), sin_119261, *[x_119262], **kwargs_119263)
    
    # Assigning a type to the variable 'y' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'y', sin_call_result_119264)
    
    # Assigning a Call to a Name (line 61):
    
    # Call to Rbf(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'x' (line 61)
    x_119266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'x', False)
    # Getting the type of 'y' (line 61)
    y_119267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'y', False)
    # Processing the call keyword arguments (line 61)
    # Getting the type of 'function' (line 61)
    function_119268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'function', False)
    keyword_119269 = function_119268
    kwargs_119270 = {'function': keyword_119269}
    # Getting the type of 'Rbf' (line 61)
    Rbf_119265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 10), 'Rbf', False)
    # Calling Rbf(args, kwargs) (line 61)
    Rbf_call_result_119271 = invoke(stypy.reporting.localization.Localization(__file__, 61, 10), Rbf_119265, *[x_119266, y_119267], **kwargs_119270)
    
    # Assigning a type to the variable 'rbf' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'rbf', Rbf_call_result_119271)
    
    # Assigning a Call to a Name (line 62):
    
    # Call to linspace(...): (line 62)
    # Processing the call arguments (line 62)
    int_119273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 18), 'int')
    int_119274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 21), 'int')
    int_119275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 25), 'int')
    # Processing the call keyword arguments (line 62)
    kwargs_119276 = {}
    # Getting the type of 'linspace' (line 62)
    linspace_119272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 9), 'linspace', False)
    # Calling linspace(args, kwargs) (line 62)
    linspace_call_result_119277 = invoke(stypy.reporting.localization.Localization(__file__, 62, 9), linspace_119272, *[int_119273, int_119274, int_119275], **kwargs_119276)
    
    # Assigning a type to the variable 'xi' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'xi', linspace_call_result_119277)
    
    # Assigning a Call to a Name (line 63):
    
    # Call to rbf(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'xi' (line 63)
    xi_119279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), 'xi', False)
    # Processing the call keyword arguments (line 63)
    kwargs_119280 = {}
    # Getting the type of 'rbf' (line 63)
    rbf_119278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 9), 'rbf', False)
    # Calling rbf(args, kwargs) (line 63)
    rbf_call_result_119281 = invoke(stypy.reporting.localization.Localization(__file__, 63, 9), rbf_119278, *[xi_119279], **kwargs_119280)
    
    # Assigning a type to the variable 'yi' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'yi', rbf_call_result_119281)
    
    # Assigning a BinOp to a Name (line 70):
    str_119282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 10), 'str', 'abs-diff: %f')
    
    # Call to max(...): (line 70)
    # Processing the call keyword arguments (line 70)
    kwargs_119293 = {}
    
    # Call to abs(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'yi' (line 70)
    yi_119284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 31), 'yi', False)
    
    # Call to sin(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'xi' (line 70)
    xi_119286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 40), 'xi', False)
    # Processing the call keyword arguments (line 70)
    kwargs_119287 = {}
    # Getting the type of 'sin' (line 70)
    sin_119285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 36), 'sin', False)
    # Calling sin(args, kwargs) (line 70)
    sin_call_result_119288 = invoke(stypy.reporting.localization.Localization(__file__, 70, 36), sin_119285, *[xi_119286], **kwargs_119287)
    
    # Applying the binary operator '-' (line 70)
    result_sub_119289 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 31), '-', yi_119284, sin_call_result_119288)
    
    # Processing the call keyword arguments (line 70)
    kwargs_119290 = {}
    # Getting the type of 'abs' (line 70)
    abs_119283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 27), 'abs', False)
    # Calling abs(args, kwargs) (line 70)
    abs_call_result_119291 = invoke(stypy.reporting.localization.Localization(__file__, 70, 27), abs_119283, *[result_sub_119289], **kwargs_119290)
    
    # Obtaining the member 'max' of a type (line 70)
    max_119292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 27), abs_call_result_119291, 'max')
    # Calling max(args, kwargs) (line 70)
    max_call_result_119294 = invoke(stypy.reporting.localization.Localization(__file__, 70, 27), max_119292, *[], **kwargs_119293)
    
    # Applying the binary operator '%' (line 70)
    result_mod_119295 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 10), '%', str_119282, max_call_result_119294)
    
    # Assigning a type to the variable 'msg' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'msg', result_mod_119295)
    
    # Call to assert_(...): (line 71)
    # Processing the call arguments (line 71)
    
    # Call to allclose(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'yi' (line 71)
    yi_119298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'yi', False)
    
    # Call to sin(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'xi' (line 71)
    xi_119300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 29), 'xi', False)
    # Processing the call keyword arguments (line 71)
    kwargs_119301 = {}
    # Getting the type of 'sin' (line 71)
    sin_119299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'sin', False)
    # Calling sin(args, kwargs) (line 71)
    sin_call_result_119302 = invoke(stypy.reporting.localization.Localization(__file__, 71, 25), sin_119299, *[xi_119300], **kwargs_119301)
    
    # Processing the call keyword arguments (line 71)
    # Getting the type of 'atol' (line 71)
    atol_119303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 39), 'atol', False)
    keyword_119304 = atol_119303
    kwargs_119305 = {'atol': keyword_119304}
    # Getting the type of 'allclose' (line 71)
    allclose_119297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'allclose', False)
    # Calling allclose(args, kwargs) (line 71)
    allclose_call_result_119306 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), allclose_119297, *[yi_119298, sin_call_result_119302], **kwargs_119305)
    
    # Getting the type of 'msg' (line 71)
    msg_119307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 46), 'msg', False)
    # Processing the call keyword arguments (line 71)
    kwargs_119308 = {}
    # Getting the type of 'assert_' (line 71)
    assert__119296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 71)
    assert__call_result_119309 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), assert__119296, *[allclose_call_result_119306, msg_119307], **kwargs_119308)
    
    
    # ################# End of 'check_rbf1d_regularity(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_rbf1d_regularity' in the type store
    # Getting the type of 'stypy_return_type' (line 56)
    stypy_return_type_119310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119310)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_rbf1d_regularity'
    return stypy_return_type_119310

# Assigning a type to the variable 'check_rbf1d_regularity' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'check_rbf1d_regularity', check_rbf1d_regularity)

@norecursion
def test_rbf_regularity(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_rbf_regularity'
    module_type_store = module_type_store.open_function_context('test_rbf_regularity', 74, 0, False)
    
    # Passed parameters checking function
    test_rbf_regularity.stypy_localization = localization
    test_rbf_regularity.stypy_type_of_self = None
    test_rbf_regularity.stypy_type_store = module_type_store
    test_rbf_regularity.stypy_function_name = 'test_rbf_regularity'
    test_rbf_regularity.stypy_param_names_list = []
    test_rbf_regularity.stypy_varargs_param_name = None
    test_rbf_regularity.stypy_kwargs_param_name = None
    test_rbf_regularity.stypy_call_defaults = defaults
    test_rbf_regularity.stypy_call_varargs = varargs
    test_rbf_regularity.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_rbf_regularity', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_rbf_regularity', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_rbf_regularity(...)' code ##################

    
    # Assigning a Dict to a Name (line 75):
    
    # Obtaining an instance of the builtin type 'dict' (line 75)
    dict_119311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 75)
    # Adding element type (key, value) (line 75)
    str_119312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'str', 'multiquadric')
    float_119313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 24), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), dict_119311, (str_119312, float_119313))
    # Adding element type (key, value) (line 75)
    str_119314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 8), 'str', 'inverse multiquadric')
    float_119315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 32), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), dict_119311, (str_119314, float_119315))
    # Adding element type (key, value) (line 75)
    str_119316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 8), 'str', 'gaussian')
    float_119317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 20), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), dict_119311, (str_119316, float_119317))
    # Adding element type (key, value) (line 75)
    str_119318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 8), 'str', 'cubic')
    float_119319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 17), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), dict_119311, (str_119318, float_119319))
    # Adding element type (key, value) (line 75)
    str_119320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 8), 'str', 'quintic')
    float_119321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 19), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), dict_119311, (str_119320, float_119321))
    # Adding element type (key, value) (line 75)
    str_119322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 8), 'str', 'thin-plate')
    float_119323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 22), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), dict_119311, (str_119322, float_119323))
    # Adding element type (key, value) (line 75)
    str_119324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'str', 'linear')
    float_119325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), dict_119311, (str_119324, float_119325))
    
    # Assigning a type to the variable 'tolerances' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tolerances', dict_119311)
    
    # Getting the type of 'FUNCTIONS' (line 84)
    FUNCTIONS_119326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'FUNCTIONS')
    # Testing the type of a for loop iterable (line 84)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 4), FUNCTIONS_119326)
    # Getting the type of the for loop variable (line 84)
    for_loop_var_119327 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 4), FUNCTIONS_119326)
    # Assigning a type to the variable 'function' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'function', for_loop_var_119327)
    # SSA begins for a for statement (line 84)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check_rbf1d_regularity(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'function' (line 85)
    function_119329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 31), 'function', False)
    
    # Call to get(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'function' (line 85)
    function_119332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 56), 'function', False)
    float_119333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 66), 'float')
    # Processing the call keyword arguments (line 85)
    kwargs_119334 = {}
    # Getting the type of 'tolerances' (line 85)
    tolerances_119330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 41), 'tolerances', False)
    # Obtaining the member 'get' of a type (line 85)
    get_119331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 41), tolerances_119330, 'get')
    # Calling get(args, kwargs) (line 85)
    get_call_result_119335 = invoke(stypy.reporting.localization.Localization(__file__, 85, 41), get_119331, *[function_119332, float_119333], **kwargs_119334)
    
    # Processing the call keyword arguments (line 85)
    kwargs_119336 = {}
    # Getting the type of 'check_rbf1d_regularity' (line 85)
    check_rbf1d_regularity_119328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'check_rbf1d_regularity', False)
    # Calling check_rbf1d_regularity(args, kwargs) (line 85)
    check_rbf1d_regularity_call_result_119337 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), check_rbf1d_regularity_119328, *[function_119329, get_call_result_119335], **kwargs_119336)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_rbf_regularity(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_rbf_regularity' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_119338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119338)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_rbf_regularity'
    return stypy_return_type_119338

# Assigning a type to the variable 'test_rbf_regularity' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'test_rbf_regularity', test_rbf_regularity)

@norecursion
def check_rbf1d_stability(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_rbf1d_stability'
    module_type_store = module_type_store.open_function_context('check_rbf1d_stability', 88, 0, False)
    
    # Passed parameters checking function
    check_rbf1d_stability.stypy_localization = localization
    check_rbf1d_stability.stypy_type_of_self = None
    check_rbf1d_stability.stypy_type_store = module_type_store
    check_rbf1d_stability.stypy_function_name = 'check_rbf1d_stability'
    check_rbf1d_stability.stypy_param_names_list = ['function']
    check_rbf1d_stability.stypy_varargs_param_name = None
    check_rbf1d_stability.stypy_kwargs_param_name = None
    check_rbf1d_stability.stypy_call_defaults = defaults
    check_rbf1d_stability.stypy_call_varargs = varargs
    check_rbf1d_stability.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_rbf1d_stability', ['function'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_rbf1d_stability', localization, ['function'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_rbf1d_stability(...)' code ##################

    
    # Call to seed(...): (line 93)
    # Processing the call arguments (line 93)
    int_119342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 19), 'int')
    # Processing the call keyword arguments (line 93)
    kwargs_119343 = {}
    # Getting the type of 'np' (line 93)
    np_119339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 93)
    random_119340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), np_119339, 'random')
    # Obtaining the member 'seed' of a type (line 93)
    seed_119341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), random_119340, 'seed')
    # Calling seed(args, kwargs) (line 93)
    seed_call_result_119344 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), seed_119341, *[int_119342], **kwargs_119343)
    
    
    # Assigning a Call to a Name (line 94):
    
    # Call to linspace(...): (line 94)
    # Processing the call arguments (line 94)
    int_119347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 20), 'int')
    int_119348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 23), 'int')
    int_119349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 27), 'int')
    # Processing the call keyword arguments (line 94)
    kwargs_119350 = {}
    # Getting the type of 'np' (line 94)
    np_119345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 94)
    linspace_119346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), np_119345, 'linspace')
    # Calling linspace(args, kwargs) (line 94)
    linspace_call_result_119351 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), linspace_119346, *[int_119347, int_119348, int_119349], **kwargs_119350)
    
    # Assigning a type to the variable 'x' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'x', linspace_call_result_119351)
    
    # Assigning a BinOp to a Name (line 95):
    # Getting the type of 'x' (line 95)
    x_119352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'x')
    float_119353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 12), 'float')
    
    # Call to randn(...): (line 95)
    # Processing the call arguments (line 95)
    
    # Call to len(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'x' (line 95)
    x_119358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 38), 'x', False)
    # Processing the call keyword arguments (line 95)
    kwargs_119359 = {}
    # Getting the type of 'len' (line 95)
    len_119357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 34), 'len', False)
    # Calling len(args, kwargs) (line 95)
    len_call_result_119360 = invoke(stypy.reporting.localization.Localization(__file__, 95, 34), len_119357, *[x_119358], **kwargs_119359)
    
    # Processing the call keyword arguments (line 95)
    kwargs_119361 = {}
    # Getting the type of 'np' (line 95)
    np_119354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'np', False)
    # Obtaining the member 'random' of a type (line 95)
    random_119355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 18), np_119354, 'random')
    # Obtaining the member 'randn' of a type (line 95)
    randn_119356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 18), random_119355, 'randn')
    # Calling randn(args, kwargs) (line 95)
    randn_call_result_119362 = invoke(stypy.reporting.localization.Localization(__file__, 95, 18), randn_119356, *[len_call_result_119360], **kwargs_119361)
    
    # Applying the binary operator '*' (line 95)
    result_mul_119363 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 12), '*', float_119353, randn_call_result_119362)
    
    # Applying the binary operator '+' (line 95)
    result_add_119364 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 8), '+', x_119352, result_mul_119363)
    
    # Assigning a type to the variable 'z' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'z', result_add_119364)
    
    # Assigning a Call to a Name (line 97):
    
    # Call to Rbf(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'x' (line 97)
    x_119366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'x', False)
    # Getting the type of 'z' (line 97)
    z_119367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 17), 'z', False)
    # Processing the call keyword arguments (line 97)
    # Getting the type of 'function' (line 97)
    function_119368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 29), 'function', False)
    keyword_119369 = function_119368
    kwargs_119370 = {'function': keyword_119369}
    # Getting the type of 'Rbf' (line 97)
    Rbf_119365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 10), 'Rbf', False)
    # Calling Rbf(args, kwargs) (line 97)
    Rbf_call_result_119371 = invoke(stypy.reporting.localization.Localization(__file__, 97, 10), Rbf_119365, *[x_119366, z_119367], **kwargs_119370)
    
    # Assigning a type to the variable 'rbf' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'rbf', Rbf_call_result_119371)
    
    # Assigning a Call to a Name (line 98):
    
    # Call to linspace(...): (line 98)
    # Processing the call arguments (line 98)
    int_119374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 21), 'int')
    int_119375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 24), 'int')
    int_119376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 28), 'int')
    # Processing the call keyword arguments (line 98)
    kwargs_119377 = {}
    # Getting the type of 'np' (line 98)
    np_119372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 9), 'np', False)
    # Obtaining the member 'linspace' of a type (line 98)
    linspace_119373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 9), np_119372, 'linspace')
    # Calling linspace(args, kwargs) (line 98)
    linspace_call_result_119378 = invoke(stypy.reporting.localization.Localization(__file__, 98, 9), linspace_119373, *[int_119374, int_119375, int_119376], **kwargs_119377)
    
    # Assigning a type to the variable 'xi' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'xi', linspace_call_result_119378)
    
    # Assigning a Call to a Name (line 99):
    
    # Call to rbf(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'xi' (line 99)
    xi_119380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'xi', False)
    # Processing the call keyword arguments (line 99)
    kwargs_119381 = {}
    # Getting the type of 'rbf' (line 99)
    rbf_119379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 9), 'rbf', False)
    # Calling rbf(args, kwargs) (line 99)
    rbf_call_result_119382 = invoke(stypy.reporting.localization.Localization(__file__, 99, 9), rbf_119379, *[xi_119380], **kwargs_119381)
    
    # Assigning a type to the variable 'yi' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'yi', rbf_call_result_119382)
    
    # Call to assert_(...): (line 102)
    # Processing the call arguments (line 102)
    
    
    # Call to max(...): (line 102)
    # Processing the call keyword arguments (line 102)
    kwargs_119392 = {}
    
    # Call to abs(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'yi' (line 102)
    yi_119386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'yi', False)
    # Getting the type of 'xi' (line 102)
    xi_119387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 22), 'xi', False)
    # Applying the binary operator '-' (line 102)
    result_sub_119388 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 19), '-', yi_119386, xi_119387)
    
    # Processing the call keyword arguments (line 102)
    kwargs_119389 = {}
    # Getting the type of 'np' (line 102)
    np_119384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'np', False)
    # Obtaining the member 'abs' of a type (line 102)
    abs_119385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), np_119384, 'abs')
    # Calling abs(args, kwargs) (line 102)
    abs_call_result_119390 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), abs_119385, *[result_sub_119388], **kwargs_119389)
    
    # Obtaining the member 'max' of a type (line 102)
    max_119391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), abs_call_result_119390, 'max')
    # Calling max(args, kwargs) (line 102)
    max_call_result_119393 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), max_119391, *[], **kwargs_119392)
    
    
    # Call to max(...): (line 102)
    # Processing the call keyword arguments (line 102)
    kwargs_119402 = {}
    
    # Call to abs(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'z' (line 102)
    z_119396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 41), 'z', False)
    # Getting the type of 'x' (line 102)
    x_119397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 43), 'x', False)
    # Applying the binary operator '-' (line 102)
    result_sub_119398 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 41), '-', z_119396, x_119397)
    
    # Processing the call keyword arguments (line 102)
    kwargs_119399 = {}
    # Getting the type of 'np' (line 102)
    np_119394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'np', False)
    # Obtaining the member 'abs' of a type (line 102)
    abs_119395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 34), np_119394, 'abs')
    # Calling abs(args, kwargs) (line 102)
    abs_call_result_119400 = invoke(stypy.reporting.localization.Localization(__file__, 102, 34), abs_119395, *[result_sub_119398], **kwargs_119399)
    
    # Obtaining the member 'max' of a type (line 102)
    max_119401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 34), abs_call_result_119400, 'max')
    # Calling max(args, kwargs) (line 102)
    max_call_result_119403 = invoke(stypy.reporting.localization.Localization(__file__, 102, 34), max_119401, *[], **kwargs_119402)
    
    # Applying the binary operator 'div' (line 102)
    result_div_119404 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 12), 'div', max_call_result_119393, max_call_result_119403)
    
    float_119405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 54), 'float')
    # Applying the binary operator '<' (line 102)
    result_lt_119406 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 12), '<', result_div_119404, float_119405)
    
    # Processing the call keyword arguments (line 102)
    kwargs_119407 = {}
    # Getting the type of 'assert_' (line 102)
    assert__119383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 102)
    assert__call_result_119408 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), assert__119383, *[result_lt_119406], **kwargs_119407)
    
    
    # ################# End of 'check_rbf1d_stability(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_rbf1d_stability' in the type store
    # Getting the type of 'stypy_return_type' (line 88)
    stypy_return_type_119409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119409)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_rbf1d_stability'
    return stypy_return_type_119409

# Assigning a type to the variable 'check_rbf1d_stability' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'check_rbf1d_stability', check_rbf1d_stability)

@norecursion
def test_rbf_stability(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_rbf_stability'
    module_type_store = module_type_store.open_function_context('test_rbf_stability', 104, 0, False)
    
    # Passed parameters checking function
    test_rbf_stability.stypy_localization = localization
    test_rbf_stability.stypy_type_of_self = None
    test_rbf_stability.stypy_type_store = module_type_store
    test_rbf_stability.stypy_function_name = 'test_rbf_stability'
    test_rbf_stability.stypy_param_names_list = []
    test_rbf_stability.stypy_varargs_param_name = None
    test_rbf_stability.stypy_kwargs_param_name = None
    test_rbf_stability.stypy_call_defaults = defaults
    test_rbf_stability.stypy_call_varargs = varargs
    test_rbf_stability.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_rbf_stability', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_rbf_stability', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_rbf_stability(...)' code ##################

    
    # Getting the type of 'FUNCTIONS' (line 105)
    FUNCTIONS_119410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'FUNCTIONS')
    # Testing the type of a for loop iterable (line 105)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 4), FUNCTIONS_119410)
    # Getting the type of the for loop variable (line 105)
    for_loop_var_119411 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 4), FUNCTIONS_119410)
    # Assigning a type to the variable 'function' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'function', for_loop_var_119411)
    # SSA begins for a for statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check_rbf1d_stability(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'function' (line 106)
    function_119413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 30), 'function', False)
    # Processing the call keyword arguments (line 106)
    kwargs_119414 = {}
    # Getting the type of 'check_rbf1d_stability' (line 106)
    check_rbf1d_stability_119412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'check_rbf1d_stability', False)
    # Calling check_rbf1d_stability(args, kwargs) (line 106)
    check_rbf1d_stability_call_result_119415 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), check_rbf1d_stability_119412, *[function_119413], **kwargs_119414)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_rbf_stability(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_rbf_stability' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_119416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119416)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_rbf_stability'
    return stypy_return_type_119416

# Assigning a type to the variable 'test_rbf_stability' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'test_rbf_stability', test_rbf_stability)

@norecursion
def test_default_construction(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_default_construction'
    module_type_store = module_type_store.open_function_context('test_default_construction', 109, 0, False)
    
    # Passed parameters checking function
    test_default_construction.stypy_localization = localization
    test_default_construction.stypy_type_of_self = None
    test_default_construction.stypy_type_store = module_type_store
    test_default_construction.stypy_function_name = 'test_default_construction'
    test_default_construction.stypy_param_names_list = []
    test_default_construction.stypy_varargs_param_name = None
    test_default_construction.stypy_kwargs_param_name = None
    test_default_construction.stypy_call_defaults = defaults
    test_default_construction.stypy_call_varargs = varargs
    test_default_construction.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_default_construction', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_default_construction', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_default_construction(...)' code ##################

    
    # Assigning a Call to a Name (line 112):
    
    # Call to linspace(...): (line 112)
    # Processing the call arguments (line 112)
    int_119418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 17), 'int')
    int_119419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 19), 'int')
    int_119420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 22), 'int')
    # Processing the call keyword arguments (line 112)
    kwargs_119421 = {}
    # Getting the type of 'linspace' (line 112)
    linspace_119417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'linspace', False)
    # Calling linspace(args, kwargs) (line 112)
    linspace_call_result_119422 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), linspace_119417, *[int_119418, int_119419, int_119420], **kwargs_119421)
    
    # Assigning a type to the variable 'x' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'x', linspace_call_result_119422)
    
    # Assigning a Call to a Name (line 113):
    
    # Call to sin(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'x' (line 113)
    x_119424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'x', False)
    # Processing the call keyword arguments (line 113)
    kwargs_119425 = {}
    # Getting the type of 'sin' (line 113)
    sin_119423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'sin', False)
    # Calling sin(args, kwargs) (line 113)
    sin_call_result_119426 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), sin_119423, *[x_119424], **kwargs_119425)
    
    # Assigning a type to the variable 'y' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'y', sin_call_result_119426)
    
    # Assigning a Call to a Name (line 114):
    
    # Call to Rbf(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'x' (line 114)
    x_119428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'x', False)
    # Getting the type of 'y' (line 114)
    y_119429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'y', False)
    # Processing the call keyword arguments (line 114)
    kwargs_119430 = {}
    # Getting the type of 'Rbf' (line 114)
    Rbf_119427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 10), 'Rbf', False)
    # Calling Rbf(args, kwargs) (line 114)
    Rbf_call_result_119431 = invoke(stypy.reporting.localization.Localization(__file__, 114, 10), Rbf_119427, *[x_119428, y_119429], **kwargs_119430)
    
    # Assigning a type to the variable 'rbf' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'rbf', Rbf_call_result_119431)
    
    # Assigning a Call to a Name (line 115):
    
    # Call to rbf(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'x' (line 115)
    x_119433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 13), 'x', False)
    # Processing the call keyword arguments (line 115)
    kwargs_119434 = {}
    # Getting the type of 'rbf' (line 115)
    rbf_119432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 9), 'rbf', False)
    # Calling rbf(args, kwargs) (line 115)
    rbf_call_result_119435 = invoke(stypy.reporting.localization.Localization(__file__, 115, 9), rbf_119432, *[x_119433], **kwargs_119434)
    
    # Assigning a type to the variable 'yi' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'yi', rbf_call_result_119435)
    
    # Call to assert_array_almost_equal(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'y' (line 116)
    y_119437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 30), 'y', False)
    # Getting the type of 'yi' (line 116)
    yi_119438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 33), 'yi', False)
    # Processing the call keyword arguments (line 116)
    kwargs_119439 = {}
    # Getting the type of 'assert_array_almost_equal' (line 116)
    assert_array_almost_equal_119436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 116)
    assert_array_almost_equal_call_result_119440 = invoke(stypy.reporting.localization.Localization(__file__, 116, 4), assert_array_almost_equal_119436, *[y_119437, yi_119438], **kwargs_119439)
    
    
    # ################# End of 'test_default_construction(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_default_construction' in the type store
    # Getting the type of 'stypy_return_type' (line 109)
    stypy_return_type_119441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119441)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_default_construction'
    return stypy_return_type_119441

# Assigning a type to the variable 'test_default_construction' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'test_default_construction', test_default_construction)

@norecursion
def test_function_is_callable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_function_is_callable'
    module_type_store = module_type_store.open_function_context('test_function_is_callable', 119, 0, False)
    
    # Passed parameters checking function
    test_function_is_callable.stypy_localization = localization
    test_function_is_callable.stypy_type_of_self = None
    test_function_is_callable.stypy_type_store = module_type_store
    test_function_is_callable.stypy_function_name = 'test_function_is_callable'
    test_function_is_callable.stypy_param_names_list = []
    test_function_is_callable.stypy_varargs_param_name = None
    test_function_is_callable.stypy_kwargs_param_name = None
    test_function_is_callable.stypy_call_defaults = defaults
    test_function_is_callable.stypy_call_varargs = varargs
    test_function_is_callable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_function_is_callable', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_function_is_callable', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_function_is_callable(...)' code ##################

    
    # Assigning a Call to a Name (line 121):
    
    # Call to linspace(...): (line 121)
    # Processing the call arguments (line 121)
    int_119443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 17), 'int')
    int_119444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 19), 'int')
    int_119445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 22), 'int')
    # Processing the call keyword arguments (line 121)
    kwargs_119446 = {}
    # Getting the type of 'linspace' (line 121)
    linspace_119442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'linspace', False)
    # Calling linspace(args, kwargs) (line 121)
    linspace_call_result_119447 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), linspace_119442, *[int_119443, int_119444, int_119445], **kwargs_119446)
    
    # Assigning a type to the variable 'x' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'x', linspace_call_result_119447)
    
    # Assigning a Call to a Name (line 122):
    
    # Call to sin(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'x' (line 122)
    x_119449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'x', False)
    # Processing the call keyword arguments (line 122)
    kwargs_119450 = {}
    # Getting the type of 'sin' (line 122)
    sin_119448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'sin', False)
    # Calling sin(args, kwargs) (line 122)
    sin_call_result_119451 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), sin_119448, *[x_119449], **kwargs_119450)
    
    # Assigning a type to the variable 'y' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'y', sin_call_result_119451)
    
    # Assigning a Lambda to a Name (line 123):

    @norecursion
    def _stypy_temp_lambda_85(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_85'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_85', 123, 14, True)
        # Passed parameters checking function
        _stypy_temp_lambda_85.stypy_localization = localization
        _stypy_temp_lambda_85.stypy_type_of_self = None
        _stypy_temp_lambda_85.stypy_type_store = module_type_store
        _stypy_temp_lambda_85.stypy_function_name = '_stypy_temp_lambda_85'
        _stypy_temp_lambda_85.stypy_param_names_list = ['x']
        _stypy_temp_lambda_85.stypy_varargs_param_name = None
        _stypy_temp_lambda_85.stypy_kwargs_param_name = None
        _stypy_temp_lambda_85.stypy_call_defaults = defaults
        _stypy_temp_lambda_85.stypy_call_varargs = varargs
        _stypy_temp_lambda_85.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_85', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_85', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'x' (line 123)
        x_119452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'x')
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), 'stypy_return_type', x_119452)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_85' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_119453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_119453)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_85'
        return stypy_return_type_119453

    # Assigning a type to the variable '_stypy_temp_lambda_85' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), '_stypy_temp_lambda_85', _stypy_temp_lambda_85)
    # Getting the type of '_stypy_temp_lambda_85' (line 123)
    _stypy_temp_lambda_85_119454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), '_stypy_temp_lambda_85')
    # Assigning a type to the variable 'linfunc' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'linfunc', _stypy_temp_lambda_85_119454)
    
    # Assigning a Call to a Name (line 124):
    
    # Call to Rbf(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'x' (line 124)
    x_119456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 14), 'x', False)
    # Getting the type of 'y' (line 124)
    y_119457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 17), 'y', False)
    # Processing the call keyword arguments (line 124)
    # Getting the type of 'linfunc' (line 124)
    linfunc_119458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 29), 'linfunc', False)
    keyword_119459 = linfunc_119458
    kwargs_119460 = {'function': keyword_119459}
    # Getting the type of 'Rbf' (line 124)
    Rbf_119455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 10), 'Rbf', False)
    # Calling Rbf(args, kwargs) (line 124)
    Rbf_call_result_119461 = invoke(stypy.reporting.localization.Localization(__file__, 124, 10), Rbf_119455, *[x_119456, y_119457], **kwargs_119460)
    
    # Assigning a type to the variable 'rbf' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'rbf', Rbf_call_result_119461)
    
    # Assigning a Call to a Name (line 125):
    
    # Call to rbf(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'x' (line 125)
    x_119463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'x', False)
    # Processing the call keyword arguments (line 125)
    kwargs_119464 = {}
    # Getting the type of 'rbf' (line 125)
    rbf_119462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 9), 'rbf', False)
    # Calling rbf(args, kwargs) (line 125)
    rbf_call_result_119465 = invoke(stypy.reporting.localization.Localization(__file__, 125, 9), rbf_119462, *[x_119463], **kwargs_119464)
    
    # Assigning a type to the variable 'yi' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'yi', rbf_call_result_119465)
    
    # Call to assert_array_almost_equal(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'y' (line 126)
    y_119467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 30), 'y', False)
    # Getting the type of 'yi' (line 126)
    yi_119468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 'yi', False)
    # Processing the call keyword arguments (line 126)
    kwargs_119469 = {}
    # Getting the type of 'assert_array_almost_equal' (line 126)
    assert_array_almost_equal_119466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 126)
    assert_array_almost_equal_call_result_119470 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), assert_array_almost_equal_119466, *[y_119467, yi_119468], **kwargs_119469)
    
    
    # ################# End of 'test_function_is_callable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_function_is_callable' in the type store
    # Getting the type of 'stypy_return_type' (line 119)
    stypy_return_type_119471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119471)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_function_is_callable'
    return stypy_return_type_119471

# Assigning a type to the variable 'test_function_is_callable' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'test_function_is_callable', test_function_is_callable)

@norecursion
def test_two_arg_function_is_callable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_two_arg_function_is_callable'
    module_type_store = module_type_store.open_function_context('test_two_arg_function_is_callable', 129, 0, False)
    
    # Passed parameters checking function
    test_two_arg_function_is_callable.stypy_localization = localization
    test_two_arg_function_is_callable.stypy_type_of_self = None
    test_two_arg_function_is_callable.stypy_type_store = module_type_store
    test_two_arg_function_is_callable.stypy_function_name = 'test_two_arg_function_is_callable'
    test_two_arg_function_is_callable.stypy_param_names_list = []
    test_two_arg_function_is_callable.stypy_varargs_param_name = None
    test_two_arg_function_is_callable.stypy_kwargs_param_name = None
    test_two_arg_function_is_callable.stypy_call_defaults = defaults
    test_two_arg_function_is_callable.stypy_call_varargs = varargs
    test_two_arg_function_is_callable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_two_arg_function_is_callable', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_two_arg_function_is_callable', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_two_arg_function_is_callable(...)' code ##################


    @norecursion
    def _func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_func'
        module_type_store = module_type_store.open_function_context('_func', 132, 4, False)
        
        # Passed parameters checking function
        _func.stypy_localization = localization
        _func.stypy_type_of_self = None
        _func.stypy_type_store = module_type_store
        _func.stypy_function_name = '_func'
        _func.stypy_param_names_list = ['self', 'r']
        _func.stypy_varargs_param_name = None
        _func.stypy_kwargs_param_name = None
        _func.stypy_call_defaults = defaults
        _func.stypy_call_varargs = varargs
        _func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_func', ['self', 'r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_func', localization, ['self', 'r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_func(...)' code ##################

        # Getting the type of 'self' (line 133)
        self_119472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'self')
        # Obtaining the member 'epsilon' of a type (line 133)
        epsilon_119473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 15), self_119472, 'epsilon')
        # Getting the type of 'r' (line 133)
        r_119474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 30), 'r')
        # Applying the binary operator '+' (line 133)
        result_add_119475 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 15), '+', epsilon_119473, r_119474)
        
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'stypy_return_type', result_add_119475)
        
        # ################# End of '_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_func' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_119476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_119476)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_func'
        return stypy_return_type_119476

    # Assigning a type to the variable '_func' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), '_func', _func)
    
    # Assigning a Call to a Name (line 135):
    
    # Call to linspace(...): (line 135)
    # Processing the call arguments (line 135)
    int_119478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 17), 'int')
    int_119479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 19), 'int')
    int_119480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 22), 'int')
    # Processing the call keyword arguments (line 135)
    kwargs_119481 = {}
    # Getting the type of 'linspace' (line 135)
    linspace_119477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'linspace', False)
    # Calling linspace(args, kwargs) (line 135)
    linspace_call_result_119482 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), linspace_119477, *[int_119478, int_119479, int_119480], **kwargs_119481)
    
    # Assigning a type to the variable 'x' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'x', linspace_call_result_119482)
    
    # Assigning a Call to a Name (line 136):
    
    # Call to sin(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'x' (line 136)
    x_119484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'x', False)
    # Processing the call keyword arguments (line 136)
    kwargs_119485 = {}
    # Getting the type of 'sin' (line 136)
    sin_119483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'sin', False)
    # Calling sin(args, kwargs) (line 136)
    sin_call_result_119486 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), sin_119483, *[x_119484], **kwargs_119485)
    
    # Assigning a type to the variable 'y' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'y', sin_call_result_119486)
    
    # Assigning a Call to a Name (line 137):
    
    # Call to Rbf(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'x' (line 137)
    x_119488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 14), 'x', False)
    # Getting the type of 'y' (line 137)
    y_119489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 17), 'y', False)
    # Processing the call keyword arguments (line 137)
    # Getting the type of '_func' (line 137)
    _func_119490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), '_func', False)
    keyword_119491 = _func_119490
    kwargs_119492 = {'function': keyword_119491}
    # Getting the type of 'Rbf' (line 137)
    Rbf_119487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 10), 'Rbf', False)
    # Calling Rbf(args, kwargs) (line 137)
    Rbf_call_result_119493 = invoke(stypy.reporting.localization.Localization(__file__, 137, 10), Rbf_119487, *[x_119488, y_119489], **kwargs_119492)
    
    # Assigning a type to the variable 'rbf' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'rbf', Rbf_call_result_119493)
    
    # Assigning a Call to a Name (line 138):
    
    # Call to rbf(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'x' (line 138)
    x_119495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 13), 'x', False)
    # Processing the call keyword arguments (line 138)
    kwargs_119496 = {}
    # Getting the type of 'rbf' (line 138)
    rbf_119494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 9), 'rbf', False)
    # Calling rbf(args, kwargs) (line 138)
    rbf_call_result_119497 = invoke(stypy.reporting.localization.Localization(__file__, 138, 9), rbf_119494, *[x_119495], **kwargs_119496)
    
    # Assigning a type to the variable 'yi' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'yi', rbf_call_result_119497)
    
    # Call to assert_array_almost_equal(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'y' (line 139)
    y_119499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 30), 'y', False)
    # Getting the type of 'yi' (line 139)
    yi_119500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 33), 'yi', False)
    # Processing the call keyword arguments (line 139)
    kwargs_119501 = {}
    # Getting the type of 'assert_array_almost_equal' (line 139)
    assert_array_almost_equal_119498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 139)
    assert_array_almost_equal_call_result_119502 = invoke(stypy.reporting.localization.Localization(__file__, 139, 4), assert_array_almost_equal_119498, *[y_119499, yi_119500], **kwargs_119501)
    
    
    # ################# End of 'test_two_arg_function_is_callable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_two_arg_function_is_callable' in the type store
    # Getting the type of 'stypy_return_type' (line 129)
    stypy_return_type_119503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119503)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_two_arg_function_is_callable'
    return stypy_return_type_119503

# Assigning a type to the variable 'test_two_arg_function_is_callable' (line 129)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'test_two_arg_function_is_callable', test_two_arg_function_is_callable)

@norecursion
def test_rbf_epsilon_none(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_rbf_epsilon_none'
    module_type_store = module_type_store.open_function_context('test_rbf_epsilon_none', 142, 0, False)
    
    # Passed parameters checking function
    test_rbf_epsilon_none.stypy_localization = localization
    test_rbf_epsilon_none.stypy_type_of_self = None
    test_rbf_epsilon_none.stypy_type_store = module_type_store
    test_rbf_epsilon_none.stypy_function_name = 'test_rbf_epsilon_none'
    test_rbf_epsilon_none.stypy_param_names_list = []
    test_rbf_epsilon_none.stypy_varargs_param_name = None
    test_rbf_epsilon_none.stypy_kwargs_param_name = None
    test_rbf_epsilon_none.stypy_call_defaults = defaults
    test_rbf_epsilon_none.stypy_call_varargs = varargs
    test_rbf_epsilon_none.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_rbf_epsilon_none', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_rbf_epsilon_none', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_rbf_epsilon_none(...)' code ##################

    
    # Assigning a Call to a Name (line 143):
    
    # Call to linspace(...): (line 143)
    # Processing the call arguments (line 143)
    int_119505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 17), 'int')
    int_119506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'int')
    int_119507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 24), 'int')
    # Processing the call keyword arguments (line 143)
    kwargs_119508 = {}
    # Getting the type of 'linspace' (line 143)
    linspace_119504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'linspace', False)
    # Calling linspace(args, kwargs) (line 143)
    linspace_call_result_119509 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), linspace_119504, *[int_119505, int_119506, int_119507], **kwargs_119508)
    
    # Assigning a type to the variable 'x' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'x', linspace_call_result_119509)
    
    # Assigning a Call to a Name (line 144):
    
    # Call to sin(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'x' (line 144)
    x_119511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'x', False)
    # Processing the call keyword arguments (line 144)
    kwargs_119512 = {}
    # Getting the type of 'sin' (line 144)
    sin_119510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'sin', False)
    # Calling sin(args, kwargs) (line 144)
    sin_call_result_119513 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), sin_119510, *[x_119511], **kwargs_119512)
    
    # Assigning a type to the variable 'y' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'y', sin_call_result_119513)
    
    # Assigning a Call to a Name (line 145):
    
    # Call to Rbf(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'x' (line 145)
    x_119515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 14), 'x', False)
    # Getting the type of 'y' (line 145)
    y_119516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 17), 'y', False)
    # Processing the call keyword arguments (line 145)
    # Getting the type of 'None' (line 145)
    None_119517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 28), 'None', False)
    keyword_119518 = None_119517
    kwargs_119519 = {'epsilon': keyword_119518}
    # Getting the type of 'Rbf' (line 145)
    Rbf_119514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 10), 'Rbf', False)
    # Calling Rbf(args, kwargs) (line 145)
    Rbf_call_result_119520 = invoke(stypy.reporting.localization.Localization(__file__, 145, 10), Rbf_119514, *[x_119515, y_119516], **kwargs_119519)
    
    # Assigning a type to the variable 'rbf' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'rbf', Rbf_call_result_119520)
    
    # ################# End of 'test_rbf_epsilon_none(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_rbf_epsilon_none' in the type store
    # Getting the type of 'stypy_return_type' (line 142)
    stypy_return_type_119521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119521)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_rbf_epsilon_none'
    return stypy_return_type_119521

# Assigning a type to the variable 'test_rbf_epsilon_none' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'test_rbf_epsilon_none', test_rbf_epsilon_none)

@norecursion
def test_rbf_epsilon_none_collinear(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_rbf_epsilon_none_collinear'
    module_type_store = module_type_store.open_function_context('test_rbf_epsilon_none_collinear', 148, 0, False)
    
    # Passed parameters checking function
    test_rbf_epsilon_none_collinear.stypy_localization = localization
    test_rbf_epsilon_none_collinear.stypy_type_of_self = None
    test_rbf_epsilon_none_collinear.stypy_type_store = module_type_store
    test_rbf_epsilon_none_collinear.stypy_function_name = 'test_rbf_epsilon_none_collinear'
    test_rbf_epsilon_none_collinear.stypy_param_names_list = []
    test_rbf_epsilon_none_collinear.stypy_varargs_param_name = None
    test_rbf_epsilon_none_collinear.stypy_kwargs_param_name = None
    test_rbf_epsilon_none_collinear.stypy_call_defaults = defaults
    test_rbf_epsilon_none_collinear.stypy_call_varargs = varargs
    test_rbf_epsilon_none_collinear.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_rbf_epsilon_none_collinear', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_rbf_epsilon_none_collinear', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_rbf_epsilon_none_collinear(...)' code ##################

    
    # Assigning a List to a Name (line 151):
    
    # Obtaining an instance of the builtin type 'list' (line 151)
    list_119522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 151)
    # Adding element type (line 151)
    int_119523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 8), list_119522, int_119523)
    # Adding element type (line 151)
    int_119524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 8), list_119522, int_119524)
    # Adding element type (line 151)
    int_119525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 8), list_119522, int_119525)
    
    # Assigning a type to the variable 'x' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'x', list_119522)
    
    # Assigning a List to a Name (line 152):
    
    # Obtaining an instance of the builtin type 'list' (line 152)
    list_119526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 152)
    # Adding element type (line 152)
    int_119527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 8), list_119526, int_119527)
    # Adding element type (line 152)
    int_119528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 8), list_119526, int_119528)
    # Adding element type (line 152)
    int_119529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 8), list_119526, int_119529)
    
    # Assigning a type to the variable 'y' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'y', list_119526)
    
    # Assigning a List to a Name (line 153):
    
    # Obtaining an instance of the builtin type 'list' (line 153)
    list_119530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 153)
    # Adding element type (line 153)
    int_119531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 8), list_119530, int_119531)
    # Adding element type (line 153)
    int_119532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 8), list_119530, int_119532)
    # Adding element type (line 153)
    int_119533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 8), list_119530, int_119533)
    
    # Assigning a type to the variable 'z' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'z', list_119530)
    
    # Assigning a Call to a Name (line 154):
    
    # Call to Rbf(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'x' (line 154)
    x_119535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 14), 'x', False)
    # Getting the type of 'y' (line 154)
    y_119536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 17), 'y', False)
    # Getting the type of 'z' (line 154)
    z_119537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'z', False)
    # Processing the call keyword arguments (line 154)
    # Getting the type of 'None' (line 154)
    None_119538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 31), 'None', False)
    keyword_119539 = None_119538
    kwargs_119540 = {'epsilon': keyword_119539}
    # Getting the type of 'Rbf' (line 154)
    Rbf_119534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 10), 'Rbf', False)
    # Calling Rbf(args, kwargs) (line 154)
    Rbf_call_result_119541 = invoke(stypy.reporting.localization.Localization(__file__, 154, 10), Rbf_119534, *[x_119535, y_119536, z_119537], **kwargs_119540)
    
    # Assigning a type to the variable 'rbf' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'rbf', Rbf_call_result_119541)
    
    # Call to assert_(...): (line 155)
    # Processing the call arguments (line 155)
    
    # Getting the type of 'rbf' (line 155)
    rbf_119543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'rbf', False)
    # Obtaining the member 'epsilon' of a type (line 155)
    epsilon_119544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), rbf_119543, 'epsilon')
    int_119545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 26), 'int')
    # Applying the binary operator '>' (line 155)
    result_gt_119546 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 12), '>', epsilon_119544, int_119545)
    
    # Processing the call keyword arguments (line 155)
    kwargs_119547 = {}
    # Getting the type of 'assert_' (line 155)
    assert__119542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 155)
    assert__call_result_119548 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), assert__119542, *[result_gt_119546], **kwargs_119547)
    
    
    # ################# End of 'test_rbf_epsilon_none_collinear(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_rbf_epsilon_none_collinear' in the type store
    # Getting the type of 'stypy_return_type' (line 148)
    stypy_return_type_119549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119549)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_rbf_epsilon_none_collinear'
    return stypy_return_type_119549

# Assigning a type to the variable 'test_rbf_epsilon_none_collinear' (line 148)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'test_rbf_epsilon_none_collinear', test_rbf_epsilon_none_collinear)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
