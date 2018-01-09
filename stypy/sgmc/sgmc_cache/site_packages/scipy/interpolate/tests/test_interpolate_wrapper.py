
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' module to test interpolate_wrapper.py
2: '''
3: from __future__ import division, print_function, absolute_import
4: 
5: from numpy import arange, allclose, ones, isnan
6: import numpy as np
7: from numpy.testing import (assert_, assert_allclose)
8: from scipy._lib._numpy_compat import suppress_warnings
9: 
10: # functionality to be tested
11: from scipy.interpolate.interpolate_wrapper import (linear, logarithmic,
12:     block_average_above, nearest)
13: 
14: 
15: class Test(object):
16: 
17:     def assertAllclose(self, x, y, rtol=1.0e-5):
18:         for i, xi in enumerate(x):
19:             assert_(allclose(xi, y[i], rtol) or (isnan(xi) and isnan(y[i])))
20: 
21:     def test_nearest(self):
22:         N = 5
23:         x = arange(N)
24:         y = arange(N)
25:         with suppress_warnings() as sup:
26:             sup.filter(DeprecationWarning, "`nearest` is deprecated")
27:             assert_allclose(y, nearest(x, y, x+.1))
28:             assert_allclose(y, nearest(x, y, x-.1))
29: 
30:     def test_linear(self):
31:         N = 3000.
32:         x = arange(N)
33:         y = arange(N)
34:         new_x = arange(N)+0.5
35:         with suppress_warnings() as sup:
36:             sup.filter(DeprecationWarning, "`linear` is deprecated")
37:             new_y = linear(x, y, new_x)
38: 
39:         assert_allclose(new_y[:5], [0.5, 1.5, 2.5, 3.5, 4.5])
40: 
41:     def test_block_average_above(self):
42:         N = 3000
43:         x = arange(N, dtype=float)
44:         y = arange(N, dtype=float)
45: 
46:         new_x = arange(N // 2) * 2
47:         with suppress_warnings() as sup:
48:             sup.filter(DeprecationWarning, "`block_average_above` is deprecated")
49:             new_y = block_average_above(x, y, new_x)
50:         assert_allclose(new_y[:5], [0.0, 0.5, 2.5, 4.5, 6.5])
51: 
52:     def test_linear2(self):
53:         N = 3000
54:         x = arange(N, dtype=float)
55:         y = ones((100,N)) * arange(N)
56:         new_x = arange(N) + 0.5
57:         with suppress_warnings() as sup:
58:             sup.filter(DeprecationWarning, "`linear` is deprecated")
59:             new_y = linear(x, y, new_x)
60:         assert_allclose(new_y[:5,:5],
61:                             [[0.5, 1.5, 2.5, 3.5, 4.5],
62:                              [0.5, 1.5, 2.5, 3.5, 4.5],
63:                              [0.5, 1.5, 2.5, 3.5, 4.5],
64:                              [0.5, 1.5, 2.5, 3.5, 4.5],
65:                              [0.5, 1.5, 2.5, 3.5, 4.5]])
66: 
67:     def test_logarithmic(self):
68:         N = 4000.
69:         x = arange(N)
70:         y = arange(N)
71:         new_x = arange(N)+0.5
72:         with suppress_warnings() as sup:
73:             sup.filter(DeprecationWarning, "`logarithmic` is deprecated")
74:             new_y = logarithmic(x, y, new_x)
75:         correct_y = [np.NaN, 1.41421356, 2.44948974, 3.46410162, 4.47213595]
76:         assert_allclose(new_y[:5], correct_y)
77: 
78:     def runTest(self):
79:         test_list = [name for name in dir(self) if name.find('test_') == 0]
80:         for test_name in test_list:
81:             exec("self.%s()" % test_name)
82: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_113552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' module to test interpolate_wrapper.py\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy import arange, allclose, ones, isnan' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_113553 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_113553) is not StypyTypeError):

    if (import_113553 != 'pyd_module'):
        __import__(import_113553)
        sys_modules_113554 = sys.modules[import_113553]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', sys_modules_113554.module_type_store, module_type_store, ['arange', 'allclose', 'ones', 'isnan'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_113554, sys_modules_113554.module_type_store, module_type_store)
    else:
        from numpy import arange, allclose, ones, isnan

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', None, module_type_store, ['arange', 'allclose', 'ones', 'isnan'], [arange, allclose, ones, isnan])

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_113553)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_113555 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_113555) is not StypyTypeError):

    if (import_113555 != 'pyd_module'):
        __import__(import_113555)
        sys_modules_113556 = sys.modules[import_113555]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_113556.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_113555)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_, assert_allclose' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_113557 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_113557) is not StypyTypeError):

    if (import_113557 != 'pyd_module'):
        __import__(import_113557)
        sys_modules_113558 = sys.modules[import_113557]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_113558.module_type_store, module_type_store, ['assert_', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_113558, sys_modules_113558.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_allclose'], [assert_, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_113557)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_113559 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat')

if (type(import_113559) is not StypyTypeError):

    if (import_113559 != 'pyd_module'):
        __import__(import_113559)
        sys_modules_113560 = sys.modules[import_113559]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', sys_modules_113560.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_113560, sys_modules_113560.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', import_113559)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.interpolate.interpolate_wrapper import linear, logarithmic, block_average_above, nearest' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_113561 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.interpolate.interpolate_wrapper')

if (type(import_113561) is not StypyTypeError):

    if (import_113561 != 'pyd_module'):
        __import__(import_113561)
        sys_modules_113562 = sys.modules[import_113561]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.interpolate.interpolate_wrapper', sys_modules_113562.module_type_store, module_type_store, ['linear', 'logarithmic', 'block_average_above', 'nearest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_113562, sys_modules_113562.module_type_store, module_type_store)
    else:
        from scipy.interpolate.interpolate_wrapper import linear, logarithmic, block_average_above, nearest

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.interpolate.interpolate_wrapper', None, module_type_store, ['linear', 'logarithmic', 'block_average_above', 'nearest'], [linear, logarithmic, block_average_above, nearest])

else:
    # Assigning a type to the variable 'scipy.interpolate.interpolate_wrapper' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.interpolate.interpolate_wrapper', import_113561)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

# Declaration of the 'Test' class

class Test(object, ):

    @norecursion
    def assertAllclose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_113563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 40), 'float')
        defaults = [float_113563]
        # Create a new context for function 'assertAllclose'
        module_type_store = module_type_store.open_function_context('assertAllclose', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test.assertAllclose.__dict__.__setitem__('stypy_localization', localization)
        Test.assertAllclose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test.assertAllclose.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test.assertAllclose.__dict__.__setitem__('stypy_function_name', 'Test.assertAllclose')
        Test.assertAllclose.__dict__.__setitem__('stypy_param_names_list', ['x', 'y', 'rtol'])
        Test.assertAllclose.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test.assertAllclose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test.assertAllclose.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test.assertAllclose.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test.assertAllclose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test.assertAllclose.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.assertAllclose', ['x', 'y', 'rtol'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertAllclose', localization, ['x', 'y', 'rtol'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertAllclose(...)' code ##################

        
        
        # Call to enumerate(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'x' (line 18)
        x_113565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 31), 'x', False)
        # Processing the call keyword arguments (line 18)
        kwargs_113566 = {}
        # Getting the type of 'enumerate' (line 18)
        enumerate_113564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 18)
        enumerate_call_result_113567 = invoke(stypy.reporting.localization.Localization(__file__, 18, 21), enumerate_113564, *[x_113565], **kwargs_113566)
        
        # Testing the type of a for loop iterable (line 18)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 18, 8), enumerate_call_result_113567)
        # Getting the type of the for loop variable (line 18)
        for_loop_var_113568 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 18, 8), enumerate_call_result_113567)
        # Assigning a type to the variable 'i' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 8), for_loop_var_113568))
        # Assigning a type to the variable 'xi' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'xi', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 8), for_loop_var_113568))
        # SSA begins for a for statement (line 18)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_(...): (line 19)
        # Processing the call arguments (line 19)
        
        # Evaluating a boolean operation
        
        # Call to allclose(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'xi' (line 19)
        xi_113571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 29), 'xi', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 19)
        i_113572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 35), 'i', False)
        # Getting the type of 'y' (line 19)
        y_113573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 33), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 19)
        getitem___113574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 33), y_113573, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 19)
        subscript_call_result_113575 = invoke(stypy.reporting.localization.Localization(__file__, 19, 33), getitem___113574, i_113572)
        
        # Getting the type of 'rtol' (line 19)
        rtol_113576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 39), 'rtol', False)
        # Processing the call keyword arguments (line 19)
        kwargs_113577 = {}
        # Getting the type of 'allclose' (line 19)
        allclose_113570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'allclose', False)
        # Calling allclose(args, kwargs) (line 19)
        allclose_call_result_113578 = invoke(stypy.reporting.localization.Localization(__file__, 19, 20), allclose_113570, *[xi_113571, subscript_call_result_113575, rtol_113576], **kwargs_113577)
        
        
        # Evaluating a boolean operation
        
        # Call to isnan(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'xi' (line 19)
        xi_113580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 55), 'xi', False)
        # Processing the call keyword arguments (line 19)
        kwargs_113581 = {}
        # Getting the type of 'isnan' (line 19)
        isnan_113579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 49), 'isnan', False)
        # Calling isnan(args, kwargs) (line 19)
        isnan_call_result_113582 = invoke(stypy.reporting.localization.Localization(__file__, 19, 49), isnan_113579, *[xi_113580], **kwargs_113581)
        
        
        # Call to isnan(...): (line 19)
        # Processing the call arguments (line 19)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 19)
        i_113584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 71), 'i', False)
        # Getting the type of 'y' (line 19)
        y_113585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 69), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 19)
        getitem___113586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 69), y_113585, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 19)
        subscript_call_result_113587 = invoke(stypy.reporting.localization.Localization(__file__, 19, 69), getitem___113586, i_113584)
        
        # Processing the call keyword arguments (line 19)
        kwargs_113588 = {}
        # Getting the type of 'isnan' (line 19)
        isnan_113583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 63), 'isnan', False)
        # Calling isnan(args, kwargs) (line 19)
        isnan_call_result_113589 = invoke(stypy.reporting.localization.Localization(__file__, 19, 63), isnan_113583, *[subscript_call_result_113587], **kwargs_113588)
        
        # Applying the binary operator 'and' (line 19)
        result_and_keyword_113590 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 49), 'and', isnan_call_result_113582, isnan_call_result_113589)
        
        # Applying the binary operator 'or' (line 19)
        result_or_keyword_113591 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 20), 'or', allclose_call_result_113578, result_and_keyword_113590)
        
        # Processing the call keyword arguments (line 19)
        kwargs_113592 = {}
        # Getting the type of 'assert_' (line 19)
        assert__113569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 19)
        assert__call_result_113593 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), assert__113569, *[result_or_keyword_113591], **kwargs_113592)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertAllclose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertAllclose' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_113594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113594)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertAllclose'
        return stypy_return_type_113594


    @norecursion
    def test_nearest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_nearest'
        module_type_store = module_type_store.open_function_context('test_nearest', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test.test_nearest.__dict__.__setitem__('stypy_localization', localization)
        Test.test_nearest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test.test_nearest.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test.test_nearest.__dict__.__setitem__('stypy_function_name', 'Test.test_nearest')
        Test.test_nearest.__dict__.__setitem__('stypy_param_names_list', [])
        Test.test_nearest.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test.test_nearest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test.test_nearest.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test.test_nearest.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test.test_nearest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test.test_nearest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_nearest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_nearest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_nearest(...)' code ##################

        
        # Assigning a Num to a Name (line 22):
        int_113595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 12), 'int')
        # Assigning a type to the variable 'N' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'N', int_113595)
        
        # Assigning a Call to a Name (line 23):
        
        # Call to arange(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'N' (line 23)
        N_113597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'N', False)
        # Processing the call keyword arguments (line 23)
        kwargs_113598 = {}
        # Getting the type of 'arange' (line 23)
        arange_113596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 23)
        arange_call_result_113599 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), arange_113596, *[N_113597], **kwargs_113598)
        
        # Assigning a type to the variable 'x' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'x', arange_call_result_113599)
        
        # Assigning a Call to a Name (line 24):
        
        # Call to arange(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'N' (line 24)
        N_113601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'N', False)
        # Processing the call keyword arguments (line 24)
        kwargs_113602 = {}
        # Getting the type of 'arange' (line 24)
        arange_113600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 24)
        arange_call_result_113603 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), arange_113600, *[N_113601], **kwargs_113602)
        
        # Assigning a type to the variable 'y' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'y', arange_call_result_113603)
        
        # Call to suppress_warnings(...): (line 25)
        # Processing the call keyword arguments (line 25)
        kwargs_113605 = {}
        # Getting the type of 'suppress_warnings' (line 25)
        suppress_warnings_113604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 25)
        suppress_warnings_call_result_113606 = invoke(stypy.reporting.localization.Localization(__file__, 25, 13), suppress_warnings_113604, *[], **kwargs_113605)
        
        with_113607 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 25, 13), suppress_warnings_call_result_113606, 'with parameter', '__enter__', '__exit__')

        if with_113607:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 25)
            enter___113608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 13), suppress_warnings_call_result_113606, '__enter__')
            with_enter_113609 = invoke(stypy.reporting.localization.Localization(__file__, 25, 13), enter___113608)
            # Assigning a type to the variable 'sup' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 13), 'sup', with_enter_113609)
            
            # Call to filter(...): (line 26)
            # Processing the call arguments (line 26)
            # Getting the type of 'DeprecationWarning' (line 26)
            DeprecationWarning_113612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'DeprecationWarning', False)
            str_113613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 43), 'str', '`nearest` is deprecated')
            # Processing the call keyword arguments (line 26)
            kwargs_113614 = {}
            # Getting the type of 'sup' (line 26)
            sup_113610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 26)
            filter_113611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), sup_113610, 'filter')
            # Calling filter(args, kwargs) (line 26)
            filter_call_result_113615 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), filter_113611, *[DeprecationWarning_113612, str_113613], **kwargs_113614)
            
            
            # Call to assert_allclose(...): (line 27)
            # Processing the call arguments (line 27)
            # Getting the type of 'y' (line 27)
            y_113617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 28), 'y', False)
            
            # Call to nearest(...): (line 27)
            # Processing the call arguments (line 27)
            # Getting the type of 'x' (line 27)
            x_113619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 39), 'x', False)
            # Getting the type of 'y' (line 27)
            y_113620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 42), 'y', False)
            # Getting the type of 'x' (line 27)
            x_113621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 45), 'x', False)
            float_113622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 47), 'float')
            # Applying the binary operator '+' (line 27)
            result_add_113623 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 45), '+', x_113621, float_113622)
            
            # Processing the call keyword arguments (line 27)
            kwargs_113624 = {}
            # Getting the type of 'nearest' (line 27)
            nearest_113618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 31), 'nearest', False)
            # Calling nearest(args, kwargs) (line 27)
            nearest_call_result_113625 = invoke(stypy.reporting.localization.Localization(__file__, 27, 31), nearest_113618, *[x_113619, y_113620, result_add_113623], **kwargs_113624)
            
            # Processing the call keyword arguments (line 27)
            kwargs_113626 = {}
            # Getting the type of 'assert_allclose' (line 27)
            assert_allclose_113616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 27)
            assert_allclose_call_result_113627 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), assert_allclose_113616, *[y_113617, nearest_call_result_113625], **kwargs_113626)
            
            
            # Call to assert_allclose(...): (line 28)
            # Processing the call arguments (line 28)
            # Getting the type of 'y' (line 28)
            y_113629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 28), 'y', False)
            
            # Call to nearest(...): (line 28)
            # Processing the call arguments (line 28)
            # Getting the type of 'x' (line 28)
            x_113631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 39), 'x', False)
            # Getting the type of 'y' (line 28)
            y_113632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 42), 'y', False)
            # Getting the type of 'x' (line 28)
            x_113633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 45), 'x', False)
            float_113634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 47), 'float')
            # Applying the binary operator '-' (line 28)
            result_sub_113635 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 45), '-', x_113633, float_113634)
            
            # Processing the call keyword arguments (line 28)
            kwargs_113636 = {}
            # Getting the type of 'nearest' (line 28)
            nearest_113630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 31), 'nearest', False)
            # Calling nearest(args, kwargs) (line 28)
            nearest_call_result_113637 = invoke(stypy.reporting.localization.Localization(__file__, 28, 31), nearest_113630, *[x_113631, y_113632, result_sub_113635], **kwargs_113636)
            
            # Processing the call keyword arguments (line 28)
            kwargs_113638 = {}
            # Getting the type of 'assert_allclose' (line 28)
            assert_allclose_113628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 28)
            assert_allclose_call_result_113639 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), assert_allclose_113628, *[y_113629, nearest_call_result_113637], **kwargs_113638)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 25)
            exit___113640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 13), suppress_warnings_call_result_113606, '__exit__')
            with_exit_113641 = invoke(stypy.reporting.localization.Localization(__file__, 25, 13), exit___113640, None, None, None)

        
        # ################# End of 'test_nearest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_nearest' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_113642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113642)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_nearest'
        return stypy_return_type_113642


    @norecursion
    def test_linear(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_linear'
        module_type_store = module_type_store.open_function_context('test_linear', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test.test_linear.__dict__.__setitem__('stypy_localization', localization)
        Test.test_linear.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test.test_linear.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test.test_linear.__dict__.__setitem__('stypy_function_name', 'Test.test_linear')
        Test.test_linear.__dict__.__setitem__('stypy_param_names_list', [])
        Test.test_linear.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test.test_linear.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test.test_linear.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test.test_linear.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test.test_linear.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test.test_linear.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_linear', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_linear', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_linear(...)' code ##################

        
        # Assigning a Num to a Name (line 31):
        float_113643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'float')
        # Assigning a type to the variable 'N' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'N', float_113643)
        
        # Assigning a Call to a Name (line 32):
        
        # Call to arange(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'N' (line 32)
        N_113645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'N', False)
        # Processing the call keyword arguments (line 32)
        kwargs_113646 = {}
        # Getting the type of 'arange' (line 32)
        arange_113644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 32)
        arange_call_result_113647 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), arange_113644, *[N_113645], **kwargs_113646)
        
        # Assigning a type to the variable 'x' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'x', arange_call_result_113647)
        
        # Assigning a Call to a Name (line 33):
        
        # Call to arange(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'N' (line 33)
        N_113649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'N', False)
        # Processing the call keyword arguments (line 33)
        kwargs_113650 = {}
        # Getting the type of 'arange' (line 33)
        arange_113648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 33)
        arange_call_result_113651 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), arange_113648, *[N_113649], **kwargs_113650)
        
        # Assigning a type to the variable 'y' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'y', arange_call_result_113651)
        
        # Assigning a BinOp to a Name (line 34):
        
        # Call to arange(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'N' (line 34)
        N_113653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'N', False)
        # Processing the call keyword arguments (line 34)
        kwargs_113654 = {}
        # Getting the type of 'arange' (line 34)
        arange_113652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 34)
        arange_call_result_113655 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), arange_113652, *[N_113653], **kwargs_113654)
        
        float_113656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 26), 'float')
        # Applying the binary operator '+' (line 34)
        result_add_113657 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 16), '+', arange_call_result_113655, float_113656)
        
        # Assigning a type to the variable 'new_x' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'new_x', result_add_113657)
        
        # Call to suppress_warnings(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_113659 = {}
        # Getting the type of 'suppress_warnings' (line 35)
        suppress_warnings_113658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 35)
        suppress_warnings_call_result_113660 = invoke(stypy.reporting.localization.Localization(__file__, 35, 13), suppress_warnings_113658, *[], **kwargs_113659)
        
        with_113661 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 35, 13), suppress_warnings_call_result_113660, 'with parameter', '__enter__', '__exit__')

        if with_113661:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 35)
            enter___113662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 13), suppress_warnings_call_result_113660, '__enter__')
            with_enter_113663 = invoke(stypy.reporting.localization.Localization(__file__, 35, 13), enter___113662)
            # Assigning a type to the variable 'sup' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'sup', with_enter_113663)
            
            # Call to filter(...): (line 36)
            # Processing the call arguments (line 36)
            # Getting the type of 'DeprecationWarning' (line 36)
            DeprecationWarning_113666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'DeprecationWarning', False)
            str_113667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 43), 'str', '`linear` is deprecated')
            # Processing the call keyword arguments (line 36)
            kwargs_113668 = {}
            # Getting the type of 'sup' (line 36)
            sup_113664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 36)
            filter_113665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), sup_113664, 'filter')
            # Calling filter(args, kwargs) (line 36)
            filter_call_result_113669 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), filter_113665, *[DeprecationWarning_113666, str_113667], **kwargs_113668)
            
            
            # Assigning a Call to a Name (line 37):
            
            # Call to linear(...): (line 37)
            # Processing the call arguments (line 37)
            # Getting the type of 'x' (line 37)
            x_113671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'x', False)
            # Getting the type of 'y' (line 37)
            y_113672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 30), 'y', False)
            # Getting the type of 'new_x' (line 37)
            new_x_113673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 33), 'new_x', False)
            # Processing the call keyword arguments (line 37)
            kwargs_113674 = {}
            # Getting the type of 'linear' (line 37)
            linear_113670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'linear', False)
            # Calling linear(args, kwargs) (line 37)
            linear_call_result_113675 = invoke(stypy.reporting.localization.Localization(__file__, 37, 20), linear_113670, *[x_113671, y_113672, new_x_113673], **kwargs_113674)
            
            # Assigning a type to the variable 'new_y' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'new_y', linear_call_result_113675)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 35)
            exit___113676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 13), suppress_warnings_call_result_113660, '__exit__')
            with_exit_113677 = invoke(stypy.reporting.localization.Localization(__file__, 35, 13), exit___113676, None, None, None)

        
        # Call to assert_allclose(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Obtaining the type of the subscript
        int_113679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 31), 'int')
        slice_113680 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 39, 24), None, int_113679, None)
        # Getting the type of 'new_y' (line 39)
        new_y_113681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 24), 'new_y', False)
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___113682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 24), new_y_113681, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_113683 = invoke(stypy.reporting.localization.Localization(__file__, 39, 24), getitem___113682, slice_113680)
        
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_113684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        # Adding element type (line 39)
        float_113685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 35), list_113684, float_113685)
        # Adding element type (line 39)
        float_113686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 35), list_113684, float_113686)
        # Adding element type (line 39)
        float_113687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 35), list_113684, float_113687)
        # Adding element type (line 39)
        float_113688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 35), list_113684, float_113688)
        # Adding element type (line 39)
        float_113689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 35), list_113684, float_113689)
        
        # Processing the call keyword arguments (line 39)
        kwargs_113690 = {}
        # Getting the type of 'assert_allclose' (line 39)
        assert_allclose_113678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 39)
        assert_allclose_call_result_113691 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), assert_allclose_113678, *[subscript_call_result_113683, list_113684], **kwargs_113690)
        
        
        # ################# End of 'test_linear(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_linear' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_113692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113692)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_linear'
        return stypy_return_type_113692


    @norecursion
    def test_block_average_above(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_block_average_above'
        module_type_store = module_type_store.open_function_context('test_block_average_above', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test.test_block_average_above.__dict__.__setitem__('stypy_localization', localization)
        Test.test_block_average_above.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test.test_block_average_above.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test.test_block_average_above.__dict__.__setitem__('stypy_function_name', 'Test.test_block_average_above')
        Test.test_block_average_above.__dict__.__setitem__('stypy_param_names_list', [])
        Test.test_block_average_above.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test.test_block_average_above.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test.test_block_average_above.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test.test_block_average_above.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test.test_block_average_above.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test.test_block_average_above.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_block_average_above', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_block_average_above', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_block_average_above(...)' code ##################

        
        # Assigning a Num to a Name (line 42):
        int_113693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 12), 'int')
        # Assigning a type to the variable 'N' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'N', int_113693)
        
        # Assigning a Call to a Name (line 43):
        
        # Call to arange(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'N' (line 43)
        N_113695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'N', False)
        # Processing the call keyword arguments (line 43)
        # Getting the type of 'float' (line 43)
        float_113696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 28), 'float', False)
        keyword_113697 = float_113696
        kwargs_113698 = {'dtype': keyword_113697}
        # Getting the type of 'arange' (line 43)
        arange_113694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 43)
        arange_call_result_113699 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), arange_113694, *[N_113695], **kwargs_113698)
        
        # Assigning a type to the variable 'x' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'x', arange_call_result_113699)
        
        # Assigning a Call to a Name (line 44):
        
        # Call to arange(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'N' (line 44)
        N_113701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'N', False)
        # Processing the call keyword arguments (line 44)
        # Getting the type of 'float' (line 44)
        float_113702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 28), 'float', False)
        keyword_113703 = float_113702
        kwargs_113704 = {'dtype': keyword_113703}
        # Getting the type of 'arange' (line 44)
        arange_113700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 44)
        arange_call_result_113705 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), arange_113700, *[N_113701], **kwargs_113704)
        
        # Assigning a type to the variable 'y' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'y', arange_call_result_113705)
        
        # Assigning a BinOp to a Name (line 46):
        
        # Call to arange(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'N' (line 46)
        N_113707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 23), 'N', False)
        int_113708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 28), 'int')
        # Applying the binary operator '//' (line 46)
        result_floordiv_113709 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 23), '//', N_113707, int_113708)
        
        # Processing the call keyword arguments (line 46)
        kwargs_113710 = {}
        # Getting the type of 'arange' (line 46)
        arange_113706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 46)
        arange_call_result_113711 = invoke(stypy.reporting.localization.Localization(__file__, 46, 16), arange_113706, *[result_floordiv_113709], **kwargs_113710)
        
        int_113712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 33), 'int')
        # Applying the binary operator '*' (line 46)
        result_mul_113713 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 16), '*', arange_call_result_113711, int_113712)
        
        # Assigning a type to the variable 'new_x' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'new_x', result_mul_113713)
        
        # Call to suppress_warnings(...): (line 47)
        # Processing the call keyword arguments (line 47)
        kwargs_113715 = {}
        # Getting the type of 'suppress_warnings' (line 47)
        suppress_warnings_113714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 47)
        suppress_warnings_call_result_113716 = invoke(stypy.reporting.localization.Localization(__file__, 47, 13), suppress_warnings_113714, *[], **kwargs_113715)
        
        with_113717 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 47, 13), suppress_warnings_call_result_113716, 'with parameter', '__enter__', '__exit__')

        if with_113717:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 47)
            enter___113718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 13), suppress_warnings_call_result_113716, '__enter__')
            with_enter_113719 = invoke(stypy.reporting.localization.Localization(__file__, 47, 13), enter___113718)
            # Assigning a type to the variable 'sup' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'sup', with_enter_113719)
            
            # Call to filter(...): (line 48)
            # Processing the call arguments (line 48)
            # Getting the type of 'DeprecationWarning' (line 48)
            DeprecationWarning_113722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'DeprecationWarning', False)
            str_113723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 43), 'str', '`block_average_above` is deprecated')
            # Processing the call keyword arguments (line 48)
            kwargs_113724 = {}
            # Getting the type of 'sup' (line 48)
            sup_113720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 48)
            filter_113721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), sup_113720, 'filter')
            # Calling filter(args, kwargs) (line 48)
            filter_call_result_113725 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), filter_113721, *[DeprecationWarning_113722, str_113723], **kwargs_113724)
            
            
            # Assigning a Call to a Name (line 49):
            
            # Call to block_average_above(...): (line 49)
            # Processing the call arguments (line 49)
            # Getting the type of 'x' (line 49)
            x_113727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 40), 'x', False)
            # Getting the type of 'y' (line 49)
            y_113728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 43), 'y', False)
            # Getting the type of 'new_x' (line 49)
            new_x_113729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 46), 'new_x', False)
            # Processing the call keyword arguments (line 49)
            kwargs_113730 = {}
            # Getting the type of 'block_average_above' (line 49)
            block_average_above_113726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 20), 'block_average_above', False)
            # Calling block_average_above(args, kwargs) (line 49)
            block_average_above_call_result_113731 = invoke(stypy.reporting.localization.Localization(__file__, 49, 20), block_average_above_113726, *[x_113727, y_113728, new_x_113729], **kwargs_113730)
            
            # Assigning a type to the variable 'new_y' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'new_y', block_average_above_call_result_113731)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 47)
            exit___113732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 13), suppress_warnings_call_result_113716, '__exit__')
            with_exit_113733 = invoke(stypy.reporting.localization.Localization(__file__, 47, 13), exit___113732, None, None, None)

        
        # Call to assert_allclose(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Obtaining the type of the subscript
        int_113735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 31), 'int')
        slice_113736 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 50, 24), None, int_113735, None)
        # Getting the type of 'new_y' (line 50)
        new_y_113737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'new_y', False)
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___113738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 24), new_y_113737, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_113739 = invoke(stypy.reporting.localization.Localization(__file__, 50, 24), getitem___113738, slice_113736)
        
        
        # Obtaining an instance of the builtin type 'list' (line 50)
        list_113740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 50)
        # Adding element type (line 50)
        float_113741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 35), list_113740, float_113741)
        # Adding element type (line 50)
        float_113742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 35), list_113740, float_113742)
        # Adding element type (line 50)
        float_113743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 35), list_113740, float_113743)
        # Adding element type (line 50)
        float_113744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 35), list_113740, float_113744)
        # Adding element type (line 50)
        float_113745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 35), list_113740, float_113745)
        
        # Processing the call keyword arguments (line 50)
        kwargs_113746 = {}
        # Getting the type of 'assert_allclose' (line 50)
        assert_allclose_113734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 50)
        assert_allclose_call_result_113747 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), assert_allclose_113734, *[subscript_call_result_113739, list_113740], **kwargs_113746)
        
        
        # ################# End of 'test_block_average_above(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_block_average_above' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_113748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113748)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_block_average_above'
        return stypy_return_type_113748


    @norecursion
    def test_linear2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_linear2'
        module_type_store = module_type_store.open_function_context('test_linear2', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test.test_linear2.__dict__.__setitem__('stypy_localization', localization)
        Test.test_linear2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test.test_linear2.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test.test_linear2.__dict__.__setitem__('stypy_function_name', 'Test.test_linear2')
        Test.test_linear2.__dict__.__setitem__('stypy_param_names_list', [])
        Test.test_linear2.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test.test_linear2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test.test_linear2.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test.test_linear2.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test.test_linear2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test.test_linear2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_linear2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_linear2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_linear2(...)' code ##################

        
        # Assigning a Num to a Name (line 53):
        int_113749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 12), 'int')
        # Assigning a type to the variable 'N' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'N', int_113749)
        
        # Assigning a Call to a Name (line 54):
        
        # Call to arange(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'N' (line 54)
        N_113751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'N', False)
        # Processing the call keyword arguments (line 54)
        # Getting the type of 'float' (line 54)
        float_113752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'float', False)
        keyword_113753 = float_113752
        kwargs_113754 = {'dtype': keyword_113753}
        # Getting the type of 'arange' (line 54)
        arange_113750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 54)
        arange_call_result_113755 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), arange_113750, *[N_113751], **kwargs_113754)
        
        # Assigning a type to the variable 'x' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'x', arange_call_result_113755)
        
        # Assigning a BinOp to a Name (line 55):
        
        # Call to ones(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Obtaining an instance of the builtin type 'tuple' (line 55)
        tuple_113757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 55)
        # Adding element type (line 55)
        int_113758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 18), tuple_113757, int_113758)
        # Adding element type (line 55)
        # Getting the type of 'N' (line 55)
        N_113759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 22), 'N', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 18), tuple_113757, N_113759)
        
        # Processing the call keyword arguments (line 55)
        kwargs_113760 = {}
        # Getting the type of 'ones' (line 55)
        ones_113756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 55)
        ones_call_result_113761 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), ones_113756, *[tuple_113757], **kwargs_113760)
        
        
        # Call to arange(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'N' (line 55)
        N_113763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'N', False)
        # Processing the call keyword arguments (line 55)
        kwargs_113764 = {}
        # Getting the type of 'arange' (line 55)
        arange_113762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 28), 'arange', False)
        # Calling arange(args, kwargs) (line 55)
        arange_call_result_113765 = invoke(stypy.reporting.localization.Localization(__file__, 55, 28), arange_113762, *[N_113763], **kwargs_113764)
        
        # Applying the binary operator '*' (line 55)
        result_mul_113766 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 12), '*', ones_call_result_113761, arange_call_result_113765)
        
        # Assigning a type to the variable 'y' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'y', result_mul_113766)
        
        # Assigning a BinOp to a Name (line 56):
        
        # Call to arange(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'N' (line 56)
        N_113768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 23), 'N', False)
        # Processing the call keyword arguments (line 56)
        kwargs_113769 = {}
        # Getting the type of 'arange' (line 56)
        arange_113767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 56)
        arange_call_result_113770 = invoke(stypy.reporting.localization.Localization(__file__, 56, 16), arange_113767, *[N_113768], **kwargs_113769)
        
        float_113771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 28), 'float')
        # Applying the binary operator '+' (line 56)
        result_add_113772 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 16), '+', arange_call_result_113770, float_113771)
        
        # Assigning a type to the variable 'new_x' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'new_x', result_add_113772)
        
        # Call to suppress_warnings(...): (line 57)
        # Processing the call keyword arguments (line 57)
        kwargs_113774 = {}
        # Getting the type of 'suppress_warnings' (line 57)
        suppress_warnings_113773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 57)
        suppress_warnings_call_result_113775 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), suppress_warnings_113773, *[], **kwargs_113774)
        
        with_113776 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 57, 13), suppress_warnings_call_result_113775, 'with parameter', '__enter__', '__exit__')

        if with_113776:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 57)
            enter___113777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 13), suppress_warnings_call_result_113775, '__enter__')
            with_enter_113778 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), enter___113777)
            # Assigning a type to the variable 'sup' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'sup', with_enter_113778)
            
            # Call to filter(...): (line 58)
            # Processing the call arguments (line 58)
            # Getting the type of 'DeprecationWarning' (line 58)
            DeprecationWarning_113781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'DeprecationWarning', False)
            str_113782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 43), 'str', '`linear` is deprecated')
            # Processing the call keyword arguments (line 58)
            kwargs_113783 = {}
            # Getting the type of 'sup' (line 58)
            sup_113779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 58)
            filter_113780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), sup_113779, 'filter')
            # Calling filter(args, kwargs) (line 58)
            filter_call_result_113784 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), filter_113780, *[DeprecationWarning_113781, str_113782], **kwargs_113783)
            
            
            # Assigning a Call to a Name (line 59):
            
            # Call to linear(...): (line 59)
            # Processing the call arguments (line 59)
            # Getting the type of 'x' (line 59)
            x_113786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 27), 'x', False)
            # Getting the type of 'y' (line 59)
            y_113787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 30), 'y', False)
            # Getting the type of 'new_x' (line 59)
            new_x_113788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'new_x', False)
            # Processing the call keyword arguments (line 59)
            kwargs_113789 = {}
            # Getting the type of 'linear' (line 59)
            linear_113785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'linear', False)
            # Calling linear(args, kwargs) (line 59)
            linear_call_result_113790 = invoke(stypy.reporting.localization.Localization(__file__, 59, 20), linear_113785, *[x_113786, y_113787, new_x_113788], **kwargs_113789)
            
            # Assigning a type to the variable 'new_y' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'new_y', linear_call_result_113790)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 57)
            exit___113791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 13), suppress_warnings_call_result_113775, '__exit__')
            with_exit_113792 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), exit___113791, None, None, None)

        
        # Call to assert_allclose(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Obtaining the type of the subscript
        int_113794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 31), 'int')
        slice_113795 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 60, 24), None, int_113794, None)
        int_113796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 34), 'int')
        slice_113797 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 60, 24), None, int_113796, None)
        # Getting the type of 'new_y' (line 60)
        new_y_113798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 'new_y', False)
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___113799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 24), new_y_113798, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_113800 = invoke(stypy.reporting.localization.Localization(__file__, 60, 24), getitem___113799, (slice_113795, slice_113797))
        
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_113801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_113802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        float_113803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 29), list_113802, float_113803)
        # Adding element type (line 61)
        float_113804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 29), list_113802, float_113804)
        # Adding element type (line 61)
        float_113805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 29), list_113802, float_113805)
        # Adding element type (line 61)
        float_113806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 29), list_113802, float_113806)
        # Adding element type (line 61)
        float_113807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 29), list_113802, float_113807)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 28), list_113801, list_113802)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_113808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        float_113809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 29), list_113808, float_113809)
        # Adding element type (line 62)
        float_113810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 29), list_113808, float_113810)
        # Adding element type (line 62)
        float_113811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 29), list_113808, float_113811)
        # Adding element type (line 62)
        float_113812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 29), list_113808, float_113812)
        # Adding element type (line 62)
        float_113813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 29), list_113808, float_113813)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 28), list_113801, list_113808)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_113814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        float_113815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 29), list_113814, float_113815)
        # Adding element type (line 63)
        float_113816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 29), list_113814, float_113816)
        # Adding element type (line 63)
        float_113817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 29), list_113814, float_113817)
        # Adding element type (line 63)
        float_113818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 29), list_113814, float_113818)
        # Adding element type (line 63)
        float_113819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 29), list_113814, float_113819)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 28), list_113801, list_113814)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_113820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        float_113821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 29), list_113820, float_113821)
        # Adding element type (line 64)
        float_113822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 29), list_113820, float_113822)
        # Adding element type (line 64)
        float_113823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 29), list_113820, float_113823)
        # Adding element type (line 64)
        float_113824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 29), list_113820, float_113824)
        # Adding element type (line 64)
        float_113825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 29), list_113820, float_113825)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 28), list_113801, list_113820)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_113826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        float_113827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 29), list_113826, float_113827)
        # Adding element type (line 65)
        float_113828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 29), list_113826, float_113828)
        # Adding element type (line 65)
        float_113829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 29), list_113826, float_113829)
        # Adding element type (line 65)
        float_113830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 29), list_113826, float_113830)
        # Adding element type (line 65)
        float_113831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 29), list_113826, float_113831)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 28), list_113801, list_113826)
        
        # Processing the call keyword arguments (line 60)
        kwargs_113832 = {}
        # Getting the type of 'assert_allclose' (line 60)
        assert_allclose_113793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 60)
        assert_allclose_call_result_113833 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assert_allclose_113793, *[subscript_call_result_113800, list_113801], **kwargs_113832)
        
        
        # ################# End of 'test_linear2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_linear2' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_113834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113834)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_linear2'
        return stypy_return_type_113834


    @norecursion
    def test_logarithmic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_logarithmic'
        module_type_store = module_type_store.open_function_context('test_logarithmic', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test.test_logarithmic.__dict__.__setitem__('stypy_localization', localization)
        Test.test_logarithmic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test.test_logarithmic.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test.test_logarithmic.__dict__.__setitem__('stypy_function_name', 'Test.test_logarithmic')
        Test.test_logarithmic.__dict__.__setitem__('stypy_param_names_list', [])
        Test.test_logarithmic.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test.test_logarithmic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test.test_logarithmic.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test.test_logarithmic.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test.test_logarithmic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test.test_logarithmic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_logarithmic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_logarithmic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_logarithmic(...)' code ##################

        
        # Assigning a Num to a Name (line 68):
        float_113835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 12), 'float')
        # Assigning a type to the variable 'N' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'N', float_113835)
        
        # Assigning a Call to a Name (line 69):
        
        # Call to arange(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'N' (line 69)
        N_113837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'N', False)
        # Processing the call keyword arguments (line 69)
        kwargs_113838 = {}
        # Getting the type of 'arange' (line 69)
        arange_113836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 69)
        arange_call_result_113839 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), arange_113836, *[N_113837], **kwargs_113838)
        
        # Assigning a type to the variable 'x' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'x', arange_call_result_113839)
        
        # Assigning a Call to a Name (line 70):
        
        # Call to arange(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'N' (line 70)
        N_113841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'N', False)
        # Processing the call keyword arguments (line 70)
        kwargs_113842 = {}
        # Getting the type of 'arange' (line 70)
        arange_113840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 70)
        arange_call_result_113843 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), arange_113840, *[N_113841], **kwargs_113842)
        
        # Assigning a type to the variable 'y' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'y', arange_call_result_113843)
        
        # Assigning a BinOp to a Name (line 71):
        
        # Call to arange(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'N' (line 71)
        N_113845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'N', False)
        # Processing the call keyword arguments (line 71)
        kwargs_113846 = {}
        # Getting the type of 'arange' (line 71)
        arange_113844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 71)
        arange_call_result_113847 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), arange_113844, *[N_113845], **kwargs_113846)
        
        float_113848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 26), 'float')
        # Applying the binary operator '+' (line 71)
        result_add_113849 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 16), '+', arange_call_result_113847, float_113848)
        
        # Assigning a type to the variable 'new_x' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'new_x', result_add_113849)
        
        # Call to suppress_warnings(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_113851 = {}
        # Getting the type of 'suppress_warnings' (line 72)
        suppress_warnings_113850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 72)
        suppress_warnings_call_result_113852 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), suppress_warnings_113850, *[], **kwargs_113851)
        
        with_113853 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 72, 13), suppress_warnings_call_result_113852, 'with parameter', '__enter__', '__exit__')

        if with_113853:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 72)
            enter___113854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), suppress_warnings_call_result_113852, '__enter__')
            with_enter_113855 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), enter___113854)
            # Assigning a type to the variable 'sup' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'sup', with_enter_113855)
            
            # Call to filter(...): (line 73)
            # Processing the call arguments (line 73)
            # Getting the type of 'DeprecationWarning' (line 73)
            DeprecationWarning_113858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'DeprecationWarning', False)
            str_113859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 43), 'str', '`logarithmic` is deprecated')
            # Processing the call keyword arguments (line 73)
            kwargs_113860 = {}
            # Getting the type of 'sup' (line 73)
            sup_113856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 73)
            filter_113857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 12), sup_113856, 'filter')
            # Calling filter(args, kwargs) (line 73)
            filter_call_result_113861 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), filter_113857, *[DeprecationWarning_113858, str_113859], **kwargs_113860)
            
            
            # Assigning a Call to a Name (line 74):
            
            # Call to logarithmic(...): (line 74)
            # Processing the call arguments (line 74)
            # Getting the type of 'x' (line 74)
            x_113863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 32), 'x', False)
            # Getting the type of 'y' (line 74)
            y_113864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 35), 'y', False)
            # Getting the type of 'new_x' (line 74)
            new_x_113865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 38), 'new_x', False)
            # Processing the call keyword arguments (line 74)
            kwargs_113866 = {}
            # Getting the type of 'logarithmic' (line 74)
            logarithmic_113862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'logarithmic', False)
            # Calling logarithmic(args, kwargs) (line 74)
            logarithmic_call_result_113867 = invoke(stypy.reporting.localization.Localization(__file__, 74, 20), logarithmic_113862, *[x_113863, y_113864, new_x_113865], **kwargs_113866)
            
            # Assigning a type to the variable 'new_y' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'new_y', logarithmic_call_result_113867)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 72)
            exit___113868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), suppress_warnings_call_result_113852, '__exit__')
            with_exit_113869 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), exit___113868, None, None, None)

        
        # Assigning a List to a Name (line 75):
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_113870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        # Getting the type of 'np' (line 75)
        np_113871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 21), 'np')
        # Obtaining the member 'NaN' of a type (line 75)
        NaN_113872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 21), np_113871, 'NaN')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 20), list_113870, NaN_113872)
        # Adding element type (line 75)
        float_113873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 20), list_113870, float_113873)
        # Adding element type (line 75)
        float_113874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 20), list_113870, float_113874)
        # Adding element type (line 75)
        float_113875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 20), list_113870, float_113875)
        # Adding element type (line 75)
        float_113876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 20), list_113870, float_113876)
        
        # Assigning a type to the variable 'correct_y' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'correct_y', list_113870)
        
        # Call to assert_allclose(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining the type of the subscript
        int_113878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 31), 'int')
        slice_113879 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 76, 24), None, int_113878, None)
        # Getting the type of 'new_y' (line 76)
        new_y_113880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'new_y', False)
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___113881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 24), new_y_113880, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_113882 = invoke(stypy.reporting.localization.Localization(__file__, 76, 24), getitem___113881, slice_113879)
        
        # Getting the type of 'correct_y' (line 76)
        correct_y_113883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 35), 'correct_y', False)
        # Processing the call keyword arguments (line 76)
        kwargs_113884 = {}
        # Getting the type of 'assert_allclose' (line 76)
        assert_allclose_113877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 76)
        assert_allclose_call_result_113885 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), assert_allclose_113877, *[subscript_call_result_113882, correct_y_113883], **kwargs_113884)
        
        
        # ################# End of 'test_logarithmic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_logarithmic' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_113886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113886)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_logarithmic'
        return stypy_return_type_113886


    @norecursion
    def runTest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runTest'
        module_type_store = module_type_store.open_function_context('runTest', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test.runTest.__dict__.__setitem__('stypy_localization', localization)
        Test.runTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test.runTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test.runTest.__dict__.__setitem__('stypy_function_name', 'Test.runTest')
        Test.runTest.__dict__.__setitem__('stypy_param_names_list', [])
        Test.runTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test.runTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test.runTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test.runTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test.runTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test.runTest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.runTest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'runTest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'runTest(...)' code ##################

        
        # Assigning a ListComp to a Name (line 79):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to dir(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'self' (line 79)
        self_113896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 42), 'self', False)
        # Processing the call keyword arguments (line 79)
        kwargs_113897 = {}
        # Getting the type of 'dir' (line 79)
        dir_113895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 38), 'dir', False)
        # Calling dir(args, kwargs) (line 79)
        dir_call_result_113898 = invoke(stypy.reporting.localization.Localization(__file__, 79, 38), dir_113895, *[self_113896], **kwargs_113897)
        
        comprehension_113899 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 21), dir_call_result_113898)
        # Assigning a type to the variable 'name' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'name', comprehension_113899)
        
        
        # Call to find(...): (line 79)
        # Processing the call arguments (line 79)
        str_113890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 61), 'str', 'test_')
        # Processing the call keyword arguments (line 79)
        kwargs_113891 = {}
        # Getting the type of 'name' (line 79)
        name_113888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 51), 'name', False)
        # Obtaining the member 'find' of a type (line 79)
        find_113889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 51), name_113888, 'find')
        # Calling find(args, kwargs) (line 79)
        find_call_result_113892 = invoke(stypy.reporting.localization.Localization(__file__, 79, 51), find_113889, *[str_113890], **kwargs_113891)
        
        int_113893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 73), 'int')
        # Applying the binary operator '==' (line 79)
        result_eq_113894 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 51), '==', find_call_result_113892, int_113893)
        
        # Getting the type of 'name' (line 79)
        name_113887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'name')
        list_113900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 21), list_113900, name_113887)
        # Assigning a type to the variable 'test_list' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'test_list', list_113900)
        
        # Getting the type of 'test_list' (line 80)
        test_list_113901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'test_list')
        # Testing the type of a for loop iterable (line 80)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 80, 8), test_list_113901)
        # Getting the type of the for loop variable (line 80)
        for_loop_var_113902 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 80, 8), test_list_113901)
        # Assigning a type to the variable 'test_name' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'test_name', for_loop_var_113902)
        # SSA begins for a for statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Dynamic code evaluation using an exec statement
        str_113903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 17), 'str', 'self.%s()')
        # Getting the type of 'test_name' (line 81)
        test_name_113904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 31), 'test_name')
        # Applying the binary operator '%' (line 81)
        result_mod_113905 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 17), '%', str_113903, test_name_113904)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 81, 12), result_mod_113905, 'exec parameter', 'StringType', 'FileType', 'CodeType')
        enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 81, 12))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'runTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runTest' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_113906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113906)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runTest'
        return stypy_return_type_113906


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 0, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'Test', Test)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
