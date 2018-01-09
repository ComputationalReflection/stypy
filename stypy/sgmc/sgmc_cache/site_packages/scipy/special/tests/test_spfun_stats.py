
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import (assert_array_equal,
5:         assert_array_almost_equal_nulp, assert_almost_equal)
6: from pytest import raises as assert_raises
7: 
8: from scipy.special import gammaln, multigammaln
9: 
10: 
11: class TestMultiGammaLn(object):
12: 
13:     def test1(self):
14:         # A test of the identity
15:         #     Gamma_1(a) = Gamma(a)
16:         np.random.seed(1234)
17:         a = np.abs(np.random.randn())
18:         assert_array_equal(multigammaln(a, 1), gammaln(a))
19: 
20:     def test2(self):
21:         # A test of the identity
22:         #     Gamma_2(a) = sqrt(pi) * Gamma(a) * Gamma(a - 0.5)
23:         a = np.array([2.5, 10.0])
24:         result = multigammaln(a, 2)
25:         expected = np.log(np.sqrt(np.pi)) + gammaln(a) + gammaln(a - 0.5)
26:         assert_almost_equal(result, expected)
27: 
28:     def test_bararg(self):
29:         assert_raises(ValueError, multigammaln, 0.5, 1.2)
30: 
31: 
32: def _check_multigammaln_array_result(a, d):
33:     # Test that the shape of the array returned by multigammaln
34:     # matches the input shape, and that all the values match
35:     # the value computed when multigammaln is called with a scalar.
36:     result = multigammaln(a, d)
37:     assert_array_equal(a.shape, result.shape)
38:     a1 = a.ravel()
39:     result1 = result.ravel()
40:     for i in range(a.size):
41:         assert_array_almost_equal_nulp(result1[i], multigammaln(a1[i], d))
42: 
43: 
44: def test_multigammaln_array_arg():
45:     # Check that the array returned by multigammaln has the correct
46:     # shape and contains the correct values.  The cases have arrays
47:     # with several differnent shapes.
48:     # The cases include a regression test for ticket #1849
49:     # (a = np.array([2.0]), an array with a single element).
50:     np.random.seed(1234)
51: 
52:     cases = [
53:         # a, d
54:         (np.abs(np.random.randn(3, 2)) + 5, 5),
55:         (np.abs(np.random.randn(1, 2)) + 5, 5),
56:         (np.arange(10.0, 18.0).reshape(2, 2, 2), 3),
57:         (np.array([2.0]), 3),
58:         (np.float64(2.0), 3),
59:     ]
60: 
61:     for a, d in cases:
62:         _check_multigammaln_array_result(a, d)
63: 
64: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560688 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_560688) is not StypyTypeError):

    if (import_560688 != 'pyd_module'):
        __import__(import_560688)
        sys_modules_560689 = sys.modules[import_560688]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_560689.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_560688)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp, assert_almost_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560690 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_560690) is not StypyTypeError):

    if (import_560690 != 'pyd_module'):
        __import__(import_560690)
        sys_modules_560691 = sys.modules[import_560690]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_560691.module_type_store, module_type_store, ['assert_array_equal', 'assert_array_almost_equal_nulp', 'assert_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_560691, sys_modules_560691.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp, assert_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_array_equal', 'assert_array_almost_equal_nulp', 'assert_almost_equal'], [assert_array_equal, assert_array_almost_equal_nulp, assert_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_560690)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from pytest import assert_raises' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560692 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest')

if (type(import_560692) is not StypyTypeError):

    if (import_560692 != 'pyd_module'):
        __import__(import_560692)
        sys_modules_560693 = sys.modules[import_560692]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', sys_modules_560693.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_560693, sys_modules_560693.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', import_560692)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.special import gammaln, multigammaln' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560694 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.special')

if (type(import_560694) is not StypyTypeError):

    if (import_560694 != 'pyd_module'):
        __import__(import_560694)
        sys_modules_560695 = sys.modules[import_560694]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.special', sys_modules_560695.module_type_store, module_type_store, ['gammaln', 'multigammaln'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_560695, sys_modules_560695.module_type_store, module_type_store)
    else:
        from scipy.special import gammaln, multigammaln

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.special', None, module_type_store, ['gammaln', 'multigammaln'], [gammaln, multigammaln])

else:
    # Assigning a type to the variable 'scipy.special' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.special', import_560694)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

# Declaration of the 'TestMultiGammaLn' class

class TestMultiGammaLn(object, ):

    @norecursion
    def test1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test1'
        module_type_store = module_type_store.open_function_context('test1', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMultiGammaLn.test1.__dict__.__setitem__('stypy_localization', localization)
        TestMultiGammaLn.test1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMultiGammaLn.test1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMultiGammaLn.test1.__dict__.__setitem__('stypy_function_name', 'TestMultiGammaLn.test1')
        TestMultiGammaLn.test1.__dict__.__setitem__('stypy_param_names_list', [])
        TestMultiGammaLn.test1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMultiGammaLn.test1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMultiGammaLn.test1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMultiGammaLn.test1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMultiGammaLn.test1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMultiGammaLn.test1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMultiGammaLn.test1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test1(...)' code ##################

        
        # Call to seed(...): (line 16)
        # Processing the call arguments (line 16)
        int_560699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'int')
        # Processing the call keyword arguments (line 16)
        kwargs_560700 = {}
        # Getting the type of 'np' (line 16)
        np_560696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 16)
        random_560697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), np_560696, 'random')
        # Obtaining the member 'seed' of a type (line 16)
        seed_560698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), random_560697, 'seed')
        # Calling seed(args, kwargs) (line 16)
        seed_call_result_560701 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), seed_560698, *[int_560699], **kwargs_560700)
        
        
        # Assigning a Call to a Name (line 17):
        
        # Call to abs(...): (line 17)
        # Processing the call arguments (line 17)
        
        # Call to randn(...): (line 17)
        # Processing the call keyword arguments (line 17)
        kwargs_560707 = {}
        # Getting the type of 'np' (line 17)
        np_560704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'np', False)
        # Obtaining the member 'random' of a type (line 17)
        random_560705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 19), np_560704, 'random')
        # Obtaining the member 'randn' of a type (line 17)
        randn_560706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 19), random_560705, 'randn')
        # Calling randn(args, kwargs) (line 17)
        randn_call_result_560708 = invoke(stypy.reporting.localization.Localization(__file__, 17, 19), randn_560706, *[], **kwargs_560707)
        
        # Processing the call keyword arguments (line 17)
        kwargs_560709 = {}
        # Getting the type of 'np' (line 17)
        np_560702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'np', False)
        # Obtaining the member 'abs' of a type (line 17)
        abs_560703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 12), np_560702, 'abs')
        # Calling abs(args, kwargs) (line 17)
        abs_call_result_560710 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), abs_560703, *[randn_call_result_560708], **kwargs_560709)
        
        # Assigning a type to the variable 'a' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'a', abs_call_result_560710)
        
        # Call to assert_array_equal(...): (line 18)
        # Processing the call arguments (line 18)
        
        # Call to multigammaln(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'a' (line 18)
        a_560713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 40), 'a', False)
        int_560714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 43), 'int')
        # Processing the call keyword arguments (line 18)
        kwargs_560715 = {}
        # Getting the type of 'multigammaln' (line 18)
        multigammaln_560712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 27), 'multigammaln', False)
        # Calling multigammaln(args, kwargs) (line 18)
        multigammaln_call_result_560716 = invoke(stypy.reporting.localization.Localization(__file__, 18, 27), multigammaln_560712, *[a_560713, int_560714], **kwargs_560715)
        
        
        # Call to gammaln(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'a' (line 18)
        a_560718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 55), 'a', False)
        # Processing the call keyword arguments (line 18)
        kwargs_560719 = {}
        # Getting the type of 'gammaln' (line 18)
        gammaln_560717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 47), 'gammaln', False)
        # Calling gammaln(args, kwargs) (line 18)
        gammaln_call_result_560720 = invoke(stypy.reporting.localization.Localization(__file__, 18, 47), gammaln_560717, *[a_560718], **kwargs_560719)
        
        # Processing the call keyword arguments (line 18)
        kwargs_560721 = {}
        # Getting the type of 'assert_array_equal' (line 18)
        assert_array_equal_560711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 18)
        assert_array_equal_call_result_560722 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), assert_array_equal_560711, *[multigammaln_call_result_560716, gammaln_call_result_560720], **kwargs_560721)
        
        
        # ################# End of 'test1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test1' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_560723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_560723)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test1'
        return stypy_return_type_560723


    @norecursion
    def test2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test2'
        module_type_store = module_type_store.open_function_context('test2', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMultiGammaLn.test2.__dict__.__setitem__('stypy_localization', localization)
        TestMultiGammaLn.test2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMultiGammaLn.test2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMultiGammaLn.test2.__dict__.__setitem__('stypy_function_name', 'TestMultiGammaLn.test2')
        TestMultiGammaLn.test2.__dict__.__setitem__('stypy_param_names_list', [])
        TestMultiGammaLn.test2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMultiGammaLn.test2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMultiGammaLn.test2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMultiGammaLn.test2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMultiGammaLn.test2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMultiGammaLn.test2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMultiGammaLn.test2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test2(...)' code ##################

        
        # Assigning a Call to a Name (line 23):
        
        # Call to array(...): (line 23)
        # Processing the call arguments (line 23)
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_560726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        # Adding element type (line 23)
        float_560727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 21), list_560726, float_560727)
        # Adding element type (line 23)
        float_560728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 21), list_560726, float_560728)
        
        # Processing the call keyword arguments (line 23)
        kwargs_560729 = {}
        # Getting the type of 'np' (line 23)
        np_560724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 23)
        array_560725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), np_560724, 'array')
        # Calling array(args, kwargs) (line 23)
        array_call_result_560730 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), array_560725, *[list_560726], **kwargs_560729)
        
        # Assigning a type to the variable 'a' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'a', array_call_result_560730)
        
        # Assigning a Call to a Name (line 24):
        
        # Call to multigammaln(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'a' (line 24)
        a_560732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 30), 'a', False)
        int_560733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 33), 'int')
        # Processing the call keyword arguments (line 24)
        kwargs_560734 = {}
        # Getting the type of 'multigammaln' (line 24)
        multigammaln_560731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'multigammaln', False)
        # Calling multigammaln(args, kwargs) (line 24)
        multigammaln_call_result_560735 = invoke(stypy.reporting.localization.Localization(__file__, 24, 17), multigammaln_560731, *[a_560732, int_560733], **kwargs_560734)
        
        # Assigning a type to the variable 'result' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'result', multigammaln_call_result_560735)
        
        # Assigning a BinOp to a Name (line 25):
        
        # Call to log(...): (line 25)
        # Processing the call arguments (line 25)
        
        # Call to sqrt(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'np' (line 25)
        np_560740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 34), 'np', False)
        # Obtaining the member 'pi' of a type (line 25)
        pi_560741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 34), np_560740, 'pi')
        # Processing the call keyword arguments (line 25)
        kwargs_560742 = {}
        # Getting the type of 'np' (line 25)
        np_560738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 26), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 25)
        sqrt_560739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 26), np_560738, 'sqrt')
        # Calling sqrt(args, kwargs) (line 25)
        sqrt_call_result_560743 = invoke(stypy.reporting.localization.Localization(__file__, 25, 26), sqrt_560739, *[pi_560741], **kwargs_560742)
        
        # Processing the call keyword arguments (line 25)
        kwargs_560744 = {}
        # Getting the type of 'np' (line 25)
        np_560736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'np', False)
        # Obtaining the member 'log' of a type (line 25)
        log_560737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 19), np_560736, 'log')
        # Calling log(args, kwargs) (line 25)
        log_call_result_560745 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), log_560737, *[sqrt_call_result_560743], **kwargs_560744)
        
        
        # Call to gammaln(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'a' (line 25)
        a_560747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 52), 'a', False)
        # Processing the call keyword arguments (line 25)
        kwargs_560748 = {}
        # Getting the type of 'gammaln' (line 25)
        gammaln_560746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 44), 'gammaln', False)
        # Calling gammaln(args, kwargs) (line 25)
        gammaln_call_result_560749 = invoke(stypy.reporting.localization.Localization(__file__, 25, 44), gammaln_560746, *[a_560747], **kwargs_560748)
        
        # Applying the binary operator '+' (line 25)
        result_add_560750 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 19), '+', log_call_result_560745, gammaln_call_result_560749)
        
        
        # Call to gammaln(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'a' (line 25)
        a_560752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 65), 'a', False)
        float_560753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 69), 'float')
        # Applying the binary operator '-' (line 25)
        result_sub_560754 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 65), '-', a_560752, float_560753)
        
        # Processing the call keyword arguments (line 25)
        kwargs_560755 = {}
        # Getting the type of 'gammaln' (line 25)
        gammaln_560751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 57), 'gammaln', False)
        # Calling gammaln(args, kwargs) (line 25)
        gammaln_call_result_560756 = invoke(stypy.reporting.localization.Localization(__file__, 25, 57), gammaln_560751, *[result_sub_560754], **kwargs_560755)
        
        # Applying the binary operator '+' (line 25)
        result_add_560757 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 55), '+', result_add_560750, gammaln_call_result_560756)
        
        # Assigning a type to the variable 'expected' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'expected', result_add_560757)
        
        # Call to assert_almost_equal(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'result' (line 26)
        result_560759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 28), 'result', False)
        # Getting the type of 'expected' (line 26)
        expected_560760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 36), 'expected', False)
        # Processing the call keyword arguments (line 26)
        kwargs_560761 = {}
        # Getting the type of 'assert_almost_equal' (line 26)
        assert_almost_equal_560758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 26)
        assert_almost_equal_call_result_560762 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), assert_almost_equal_560758, *[result_560759, expected_560760], **kwargs_560761)
        
        
        # ################# End of 'test2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test2' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_560763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_560763)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test2'
        return stypy_return_type_560763


    @norecursion
    def test_bararg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bararg'
        module_type_store = module_type_store.open_function_context('test_bararg', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMultiGammaLn.test_bararg.__dict__.__setitem__('stypy_localization', localization)
        TestMultiGammaLn.test_bararg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMultiGammaLn.test_bararg.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMultiGammaLn.test_bararg.__dict__.__setitem__('stypy_function_name', 'TestMultiGammaLn.test_bararg')
        TestMultiGammaLn.test_bararg.__dict__.__setitem__('stypy_param_names_list', [])
        TestMultiGammaLn.test_bararg.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMultiGammaLn.test_bararg.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMultiGammaLn.test_bararg.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMultiGammaLn.test_bararg.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMultiGammaLn.test_bararg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMultiGammaLn.test_bararg.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMultiGammaLn.test_bararg', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bararg', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bararg(...)' code ##################

        
        # Call to assert_raises(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'ValueError' (line 29)
        ValueError_560765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'ValueError', False)
        # Getting the type of 'multigammaln' (line 29)
        multigammaln_560766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 34), 'multigammaln', False)
        float_560767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 48), 'float')
        float_560768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 53), 'float')
        # Processing the call keyword arguments (line 29)
        kwargs_560769 = {}
        # Getting the type of 'assert_raises' (line 29)
        assert_raises_560764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 29)
        assert_raises_call_result_560770 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), assert_raises_560764, *[ValueError_560765, multigammaln_560766, float_560767, float_560768], **kwargs_560769)
        
        
        # ################# End of 'test_bararg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bararg' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_560771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_560771)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bararg'
        return stypy_return_type_560771


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 11, 0, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMultiGammaLn.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestMultiGammaLn' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'TestMultiGammaLn', TestMultiGammaLn)

@norecursion
def _check_multigammaln_array_result(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_multigammaln_array_result'
    module_type_store = module_type_store.open_function_context('_check_multigammaln_array_result', 32, 0, False)
    
    # Passed parameters checking function
    _check_multigammaln_array_result.stypy_localization = localization
    _check_multigammaln_array_result.stypy_type_of_self = None
    _check_multigammaln_array_result.stypy_type_store = module_type_store
    _check_multigammaln_array_result.stypy_function_name = '_check_multigammaln_array_result'
    _check_multigammaln_array_result.stypy_param_names_list = ['a', 'd']
    _check_multigammaln_array_result.stypy_varargs_param_name = None
    _check_multigammaln_array_result.stypy_kwargs_param_name = None
    _check_multigammaln_array_result.stypy_call_defaults = defaults
    _check_multigammaln_array_result.stypy_call_varargs = varargs
    _check_multigammaln_array_result.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_multigammaln_array_result', ['a', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_multigammaln_array_result', localization, ['a', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_multigammaln_array_result(...)' code ##################

    
    # Assigning a Call to a Name (line 36):
    
    # Call to multigammaln(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'a' (line 36)
    a_560773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 26), 'a', False)
    # Getting the type of 'd' (line 36)
    d_560774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 29), 'd', False)
    # Processing the call keyword arguments (line 36)
    kwargs_560775 = {}
    # Getting the type of 'multigammaln' (line 36)
    multigammaln_560772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'multigammaln', False)
    # Calling multigammaln(args, kwargs) (line 36)
    multigammaln_call_result_560776 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), multigammaln_560772, *[a_560773, d_560774], **kwargs_560775)
    
    # Assigning a type to the variable 'result' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'result', multigammaln_call_result_560776)
    
    # Call to assert_array_equal(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'a' (line 37)
    a_560778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'a', False)
    # Obtaining the member 'shape' of a type (line 37)
    shape_560779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 23), a_560778, 'shape')
    # Getting the type of 'result' (line 37)
    result_560780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 32), 'result', False)
    # Obtaining the member 'shape' of a type (line 37)
    shape_560781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 32), result_560780, 'shape')
    # Processing the call keyword arguments (line 37)
    kwargs_560782 = {}
    # Getting the type of 'assert_array_equal' (line 37)
    assert_array_equal_560777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 37)
    assert_array_equal_call_result_560783 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), assert_array_equal_560777, *[shape_560779, shape_560781], **kwargs_560782)
    
    
    # Assigning a Call to a Name (line 38):
    
    # Call to ravel(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_560786 = {}
    # Getting the type of 'a' (line 38)
    a_560784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'a', False)
    # Obtaining the member 'ravel' of a type (line 38)
    ravel_560785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 9), a_560784, 'ravel')
    # Calling ravel(args, kwargs) (line 38)
    ravel_call_result_560787 = invoke(stypy.reporting.localization.Localization(__file__, 38, 9), ravel_560785, *[], **kwargs_560786)
    
    # Assigning a type to the variable 'a1' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'a1', ravel_call_result_560787)
    
    # Assigning a Call to a Name (line 39):
    
    # Call to ravel(...): (line 39)
    # Processing the call keyword arguments (line 39)
    kwargs_560790 = {}
    # Getting the type of 'result' (line 39)
    result_560788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'result', False)
    # Obtaining the member 'ravel' of a type (line 39)
    ravel_560789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 14), result_560788, 'ravel')
    # Calling ravel(args, kwargs) (line 39)
    ravel_call_result_560791 = invoke(stypy.reporting.localization.Localization(__file__, 39, 14), ravel_560789, *[], **kwargs_560790)
    
    # Assigning a type to the variable 'result1' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'result1', ravel_call_result_560791)
    
    
    # Call to range(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'a' (line 40)
    a_560793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'a', False)
    # Obtaining the member 'size' of a type (line 40)
    size_560794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), a_560793, 'size')
    # Processing the call keyword arguments (line 40)
    kwargs_560795 = {}
    # Getting the type of 'range' (line 40)
    range_560792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'range', False)
    # Calling range(args, kwargs) (line 40)
    range_call_result_560796 = invoke(stypy.reporting.localization.Localization(__file__, 40, 13), range_560792, *[size_560794], **kwargs_560795)
    
    # Testing the type of a for loop iterable (line 40)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 4), range_call_result_560796)
    # Getting the type of the for loop variable (line 40)
    for_loop_var_560797 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 4), range_call_result_560796)
    # Assigning a type to the variable 'i' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'i', for_loop_var_560797)
    # SSA begins for a for statement (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_array_almost_equal_nulp(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 41)
    i_560799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 47), 'i', False)
    # Getting the type of 'result1' (line 41)
    result1_560800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 39), 'result1', False)
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___560801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 39), result1_560800, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_560802 = invoke(stypy.reporting.localization.Localization(__file__, 41, 39), getitem___560801, i_560799)
    
    
    # Call to multigammaln(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 41)
    i_560804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 67), 'i', False)
    # Getting the type of 'a1' (line 41)
    a1_560805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 64), 'a1', False)
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___560806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 64), a1_560805, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_560807 = invoke(stypy.reporting.localization.Localization(__file__, 41, 64), getitem___560806, i_560804)
    
    # Getting the type of 'd' (line 41)
    d_560808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 71), 'd', False)
    # Processing the call keyword arguments (line 41)
    kwargs_560809 = {}
    # Getting the type of 'multigammaln' (line 41)
    multigammaln_560803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 51), 'multigammaln', False)
    # Calling multigammaln(args, kwargs) (line 41)
    multigammaln_call_result_560810 = invoke(stypy.reporting.localization.Localization(__file__, 41, 51), multigammaln_560803, *[subscript_call_result_560807, d_560808], **kwargs_560809)
    
    # Processing the call keyword arguments (line 41)
    kwargs_560811 = {}
    # Getting the type of 'assert_array_almost_equal_nulp' (line 41)
    assert_array_almost_equal_nulp_560798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'assert_array_almost_equal_nulp', False)
    # Calling assert_array_almost_equal_nulp(args, kwargs) (line 41)
    assert_array_almost_equal_nulp_call_result_560812 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), assert_array_almost_equal_nulp_560798, *[subscript_call_result_560802, multigammaln_call_result_560810], **kwargs_560811)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_check_multigammaln_array_result(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_multigammaln_array_result' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_560813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_560813)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_multigammaln_array_result'
    return stypy_return_type_560813

# Assigning a type to the variable '_check_multigammaln_array_result' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), '_check_multigammaln_array_result', _check_multigammaln_array_result)

@norecursion
def test_multigammaln_array_arg(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_multigammaln_array_arg'
    module_type_store = module_type_store.open_function_context('test_multigammaln_array_arg', 44, 0, False)
    
    # Passed parameters checking function
    test_multigammaln_array_arg.stypy_localization = localization
    test_multigammaln_array_arg.stypy_type_of_self = None
    test_multigammaln_array_arg.stypy_type_store = module_type_store
    test_multigammaln_array_arg.stypy_function_name = 'test_multigammaln_array_arg'
    test_multigammaln_array_arg.stypy_param_names_list = []
    test_multigammaln_array_arg.stypy_varargs_param_name = None
    test_multigammaln_array_arg.stypy_kwargs_param_name = None
    test_multigammaln_array_arg.stypy_call_defaults = defaults
    test_multigammaln_array_arg.stypy_call_varargs = varargs
    test_multigammaln_array_arg.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_multigammaln_array_arg', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_multigammaln_array_arg', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_multigammaln_array_arg(...)' code ##################

    
    # Call to seed(...): (line 50)
    # Processing the call arguments (line 50)
    int_560817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 19), 'int')
    # Processing the call keyword arguments (line 50)
    kwargs_560818 = {}
    # Getting the type of 'np' (line 50)
    np_560814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 50)
    random_560815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 4), np_560814, 'random')
    # Obtaining the member 'seed' of a type (line 50)
    seed_560816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 4), random_560815, 'seed')
    # Calling seed(args, kwargs) (line 50)
    seed_call_result_560819 = invoke(stypy.reporting.localization.Localization(__file__, 50, 4), seed_560816, *[int_560817], **kwargs_560818)
    
    
    # Assigning a List to a Name (line 52):
    
    # Obtaining an instance of the builtin type 'list' (line 52)
    list_560820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 52)
    # Adding element type (line 52)
    
    # Obtaining an instance of the builtin type 'tuple' (line 54)
    tuple_560821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 54)
    # Adding element type (line 54)
    
    # Call to abs(...): (line 54)
    # Processing the call arguments (line 54)
    
    # Call to randn(...): (line 54)
    # Processing the call arguments (line 54)
    int_560827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 32), 'int')
    int_560828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 35), 'int')
    # Processing the call keyword arguments (line 54)
    kwargs_560829 = {}
    # Getting the type of 'np' (line 54)
    np_560824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'np', False)
    # Obtaining the member 'random' of a type (line 54)
    random_560825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), np_560824, 'random')
    # Obtaining the member 'randn' of a type (line 54)
    randn_560826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), random_560825, 'randn')
    # Calling randn(args, kwargs) (line 54)
    randn_call_result_560830 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), randn_560826, *[int_560827, int_560828], **kwargs_560829)
    
    # Processing the call keyword arguments (line 54)
    kwargs_560831 = {}
    # Getting the type of 'np' (line 54)
    np_560822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 9), 'np', False)
    # Obtaining the member 'abs' of a type (line 54)
    abs_560823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 9), np_560822, 'abs')
    # Calling abs(args, kwargs) (line 54)
    abs_call_result_560832 = invoke(stypy.reporting.localization.Localization(__file__, 54, 9), abs_560823, *[randn_call_result_560830], **kwargs_560831)
    
    int_560833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 41), 'int')
    # Applying the binary operator '+' (line 54)
    result_add_560834 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 9), '+', abs_call_result_560832, int_560833)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_560821, result_add_560834)
    # Adding element type (line 54)
    int_560835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_560821, int_560835)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), list_560820, tuple_560821)
    # Adding element type (line 52)
    
    # Obtaining an instance of the builtin type 'tuple' (line 55)
    tuple_560836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 55)
    # Adding element type (line 55)
    
    # Call to abs(...): (line 55)
    # Processing the call arguments (line 55)
    
    # Call to randn(...): (line 55)
    # Processing the call arguments (line 55)
    int_560842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 32), 'int')
    int_560843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 35), 'int')
    # Processing the call keyword arguments (line 55)
    kwargs_560844 = {}
    # Getting the type of 'np' (line 55)
    np_560839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'np', False)
    # Obtaining the member 'random' of a type (line 55)
    random_560840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), np_560839, 'random')
    # Obtaining the member 'randn' of a type (line 55)
    randn_560841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), random_560840, 'randn')
    # Calling randn(args, kwargs) (line 55)
    randn_call_result_560845 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), randn_560841, *[int_560842, int_560843], **kwargs_560844)
    
    # Processing the call keyword arguments (line 55)
    kwargs_560846 = {}
    # Getting the type of 'np' (line 55)
    np_560837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 9), 'np', False)
    # Obtaining the member 'abs' of a type (line 55)
    abs_560838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 9), np_560837, 'abs')
    # Calling abs(args, kwargs) (line 55)
    abs_call_result_560847 = invoke(stypy.reporting.localization.Localization(__file__, 55, 9), abs_560838, *[randn_call_result_560845], **kwargs_560846)
    
    int_560848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 41), 'int')
    # Applying the binary operator '+' (line 55)
    result_add_560849 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 9), '+', abs_call_result_560847, int_560848)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_560836, result_add_560849)
    # Adding element type (line 55)
    int_560850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_560836, int_560850)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), list_560820, tuple_560836)
    # Adding element type (line 52)
    
    # Obtaining an instance of the builtin type 'tuple' (line 56)
    tuple_560851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 56)
    # Adding element type (line 56)
    
    # Call to reshape(...): (line 56)
    # Processing the call arguments (line 56)
    int_560859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 39), 'int')
    int_560860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 42), 'int')
    int_560861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 45), 'int')
    # Processing the call keyword arguments (line 56)
    kwargs_560862 = {}
    
    # Call to arange(...): (line 56)
    # Processing the call arguments (line 56)
    float_560854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 19), 'float')
    float_560855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'float')
    # Processing the call keyword arguments (line 56)
    kwargs_560856 = {}
    # Getting the type of 'np' (line 56)
    np_560852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 9), 'np', False)
    # Obtaining the member 'arange' of a type (line 56)
    arange_560853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 9), np_560852, 'arange')
    # Calling arange(args, kwargs) (line 56)
    arange_call_result_560857 = invoke(stypy.reporting.localization.Localization(__file__, 56, 9), arange_560853, *[float_560854, float_560855], **kwargs_560856)
    
    # Obtaining the member 'reshape' of a type (line 56)
    reshape_560858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 9), arange_call_result_560857, 'reshape')
    # Calling reshape(args, kwargs) (line 56)
    reshape_call_result_560863 = invoke(stypy.reporting.localization.Localization(__file__, 56, 9), reshape_560858, *[int_560859, int_560860, int_560861], **kwargs_560862)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_560851, reshape_call_result_560863)
    # Adding element type (line 56)
    int_560864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_560851, int_560864)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), list_560820, tuple_560851)
    # Adding element type (line 52)
    
    # Obtaining an instance of the builtin type 'tuple' (line 57)
    tuple_560865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 57)
    # Adding element type (line 57)
    
    # Call to array(...): (line 57)
    # Processing the call arguments (line 57)
    
    # Obtaining an instance of the builtin type 'list' (line 57)
    list_560868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 57)
    # Adding element type (line 57)
    float_560869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 18), list_560868, float_560869)
    
    # Processing the call keyword arguments (line 57)
    kwargs_560870 = {}
    # Getting the type of 'np' (line 57)
    np_560866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 57)
    array_560867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 9), np_560866, 'array')
    # Calling array(args, kwargs) (line 57)
    array_call_result_560871 = invoke(stypy.reporting.localization.Localization(__file__, 57, 9), array_560867, *[list_560868], **kwargs_560870)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 9), tuple_560865, array_call_result_560871)
    # Adding element type (line 57)
    int_560872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 9), tuple_560865, int_560872)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), list_560820, tuple_560865)
    # Adding element type (line 52)
    
    # Obtaining an instance of the builtin type 'tuple' (line 58)
    tuple_560873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 58)
    # Adding element type (line 58)
    
    # Call to float64(...): (line 58)
    # Processing the call arguments (line 58)
    float_560876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 20), 'float')
    # Processing the call keyword arguments (line 58)
    kwargs_560877 = {}
    # Getting the type of 'np' (line 58)
    np_560874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 9), 'np', False)
    # Obtaining the member 'float64' of a type (line 58)
    float64_560875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 9), np_560874, 'float64')
    # Calling float64(args, kwargs) (line 58)
    float64_call_result_560878 = invoke(stypy.reporting.localization.Localization(__file__, 58, 9), float64_560875, *[float_560876], **kwargs_560877)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), tuple_560873, float64_call_result_560878)
    # Adding element type (line 58)
    int_560879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), tuple_560873, int_560879)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), list_560820, tuple_560873)
    
    # Assigning a type to the variable 'cases' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'cases', list_560820)
    
    # Getting the type of 'cases' (line 61)
    cases_560880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'cases')
    # Testing the type of a for loop iterable (line 61)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 61, 4), cases_560880)
    # Getting the type of the for loop variable (line 61)
    for_loop_var_560881 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 61, 4), cases_560880)
    # Assigning a type to the variable 'a' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 4), for_loop_var_560881))
    # Assigning a type to the variable 'd' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 4), for_loop_var_560881))
    # SSA begins for a for statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to _check_multigammaln_array_result(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'a' (line 62)
    a_560883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 41), 'a', False)
    # Getting the type of 'd' (line 62)
    d_560884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 44), 'd', False)
    # Processing the call keyword arguments (line 62)
    kwargs_560885 = {}
    # Getting the type of '_check_multigammaln_array_result' (line 62)
    _check_multigammaln_array_result_560882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), '_check_multigammaln_array_result', False)
    # Calling _check_multigammaln_array_result(args, kwargs) (line 62)
    _check_multigammaln_array_result_call_result_560886 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), _check_multigammaln_array_result_560882, *[a_560883, d_560884], **kwargs_560885)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_multigammaln_array_arg(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_multigammaln_array_arg' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_560887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_560887)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_multigammaln_array_arg'
    return stypy_return_type_560887

# Assigning a type to the variable 'test_multigammaln_array_arg' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'test_multigammaln_array_arg', test_multigammaln_array_arg)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
