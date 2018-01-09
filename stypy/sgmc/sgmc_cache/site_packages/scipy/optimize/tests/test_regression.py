
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Regression tests for optimize.
2: 
3: '''
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from numpy.testing import assert_almost_equal
8: from pytest import raises as assert_raises
9: 
10: import scipy.optimize
11: 
12: 
13: class TestRegression(object):
14: 
15:     def test_newton_x0_is_0(self):
16:         # Regression test for gh-1601
17:         tgt = 1
18:         res = scipy.optimize.newton(lambda x: x - 1, 0)
19:         assert_almost_equal(res, tgt)
20: 
21:     def test_newton_integers(self):
22:         # Regression test for gh-1741
23:         root = scipy.optimize.newton(lambda x: x**2 - 1, x0=2,
24:                                     fprime=lambda x: 2*x)
25:         assert_almost_equal(root, 1.0)
26: 
27:     def test_lmdif_errmsg(self):
28:         # This shouldn't cause a crash on Python 3
29:         class SomeError(Exception):
30:             pass
31:         counter = [0]
32: 
33:         def func(x):
34:             counter[0] += 1
35:             if counter[0] < 3:
36:                 return x**2 - np.array([9, 10, 11])
37:             else:
38:                 raise SomeError()
39:         assert_raises(SomeError,
40:                       scipy.optimize.leastsq,
41:                       func, [1, 2, 3])
42: 
43: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_229485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', 'Regression tests for optimize.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_229486 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_229486) is not StypyTypeError):

    if (import_229486 != 'pyd_module'):
        __import__(import_229486)
        sys_modules_229487 = sys.modules[import_229486]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_229487.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_229486)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_almost_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_229488 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_229488) is not StypyTypeError):

    if (import_229488 != 'pyd_module'):
        __import__(import_229488)
        sys_modules_229489 = sys.modules[import_229488]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_229489.module_type_store, module_type_store, ['assert_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_229489, sys_modules_229489.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_almost_equal'], [assert_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_229488)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from pytest import assert_raises' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_229490 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_229490) is not StypyTypeError):

    if (import_229490 != 'pyd_module'):
        __import__(import_229490)
        sys_modules_229491 = sys.modules[import_229490]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_229491.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_229491, sys_modules_229491.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_229490)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import scipy.optimize' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_229492 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize')

if (type(import_229492) is not StypyTypeError):

    if (import_229492 != 'pyd_module'):
        __import__(import_229492)
        sys_modules_229493 = sys.modules[import_229492]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize', sys_modules_229493.module_type_store, module_type_store)
    else:
        import scipy.optimize

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize', scipy.optimize, module_type_store)

else:
    # Assigning a type to the variable 'scipy.optimize' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize', import_229492)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

# Declaration of the 'TestRegression' class

class TestRegression(object, ):

    @norecursion
    def test_newton_x0_is_0(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_newton_x0_is_0'
        module_type_store = module_type_store.open_function_context('test_newton_x0_is_0', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRegression.test_newton_x0_is_0.__dict__.__setitem__('stypy_localization', localization)
        TestRegression.test_newton_x0_is_0.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRegression.test_newton_x0_is_0.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRegression.test_newton_x0_is_0.__dict__.__setitem__('stypy_function_name', 'TestRegression.test_newton_x0_is_0')
        TestRegression.test_newton_x0_is_0.__dict__.__setitem__('stypy_param_names_list', [])
        TestRegression.test_newton_x0_is_0.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRegression.test_newton_x0_is_0.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRegression.test_newton_x0_is_0.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRegression.test_newton_x0_is_0.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRegression.test_newton_x0_is_0.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRegression.test_newton_x0_is_0.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRegression.test_newton_x0_is_0', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_newton_x0_is_0', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_newton_x0_is_0(...)' code ##################

        
        # Assigning a Num to a Name (line 17):
        int_229494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
        # Assigning a type to the variable 'tgt' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tgt', int_229494)
        
        # Assigning a Call to a Name (line 18):
        
        # Call to newton(...): (line 18)
        # Processing the call arguments (line 18)

        @norecursion
        def _stypy_temp_lambda_96(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_96'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_96', 18, 36, True)
            # Passed parameters checking function
            _stypy_temp_lambda_96.stypy_localization = localization
            _stypy_temp_lambda_96.stypy_type_of_self = None
            _stypy_temp_lambda_96.stypy_type_store = module_type_store
            _stypy_temp_lambda_96.stypy_function_name = '_stypy_temp_lambda_96'
            _stypy_temp_lambda_96.stypy_param_names_list = ['x']
            _stypy_temp_lambda_96.stypy_varargs_param_name = None
            _stypy_temp_lambda_96.stypy_kwargs_param_name = None
            _stypy_temp_lambda_96.stypy_call_defaults = defaults
            _stypy_temp_lambda_96.stypy_call_varargs = varargs
            _stypy_temp_lambda_96.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_96', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_96', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 18)
            x_229498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 46), 'x', False)
            int_229499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 50), 'int')
            # Applying the binary operator '-' (line 18)
            result_sub_229500 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 46), '-', x_229498, int_229499)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 36), 'stypy_return_type', result_sub_229500)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_96' in the type store
            # Getting the type of 'stypy_return_type' (line 18)
            stypy_return_type_229501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 36), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_229501)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_96'
            return stypy_return_type_229501

        # Assigning a type to the variable '_stypy_temp_lambda_96' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 36), '_stypy_temp_lambda_96', _stypy_temp_lambda_96)
        # Getting the type of '_stypy_temp_lambda_96' (line 18)
        _stypy_temp_lambda_96_229502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 36), '_stypy_temp_lambda_96')
        int_229503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 53), 'int')
        # Processing the call keyword arguments (line 18)
        kwargs_229504 = {}
        # Getting the type of 'scipy' (line 18)
        scipy_229495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'scipy', False)
        # Obtaining the member 'optimize' of a type (line 18)
        optimize_229496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 14), scipy_229495, 'optimize')
        # Obtaining the member 'newton' of a type (line 18)
        newton_229497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 14), optimize_229496, 'newton')
        # Calling newton(args, kwargs) (line 18)
        newton_call_result_229505 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), newton_229497, *[_stypy_temp_lambda_96_229502, int_229503], **kwargs_229504)
        
        # Assigning a type to the variable 'res' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'res', newton_call_result_229505)
        
        # Call to assert_almost_equal(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'res' (line 19)
        res_229507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 28), 'res', False)
        # Getting the type of 'tgt' (line 19)
        tgt_229508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 33), 'tgt', False)
        # Processing the call keyword arguments (line 19)
        kwargs_229509 = {}
        # Getting the type of 'assert_almost_equal' (line 19)
        assert_almost_equal_229506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 19)
        assert_almost_equal_call_result_229510 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), assert_almost_equal_229506, *[res_229507, tgt_229508], **kwargs_229509)
        
        
        # ################# End of 'test_newton_x0_is_0(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_newton_x0_is_0' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_229511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229511)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_newton_x0_is_0'
        return stypy_return_type_229511


    @norecursion
    def test_newton_integers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_newton_integers'
        module_type_store = module_type_store.open_function_context('test_newton_integers', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRegression.test_newton_integers.__dict__.__setitem__('stypy_localization', localization)
        TestRegression.test_newton_integers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRegression.test_newton_integers.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRegression.test_newton_integers.__dict__.__setitem__('stypy_function_name', 'TestRegression.test_newton_integers')
        TestRegression.test_newton_integers.__dict__.__setitem__('stypy_param_names_list', [])
        TestRegression.test_newton_integers.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRegression.test_newton_integers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRegression.test_newton_integers.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRegression.test_newton_integers.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRegression.test_newton_integers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRegression.test_newton_integers.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRegression.test_newton_integers', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_newton_integers', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_newton_integers(...)' code ##################

        
        # Assigning a Call to a Name (line 23):
        
        # Call to newton(...): (line 23)
        # Processing the call arguments (line 23)

        @norecursion
        def _stypy_temp_lambda_97(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_97'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_97', 23, 37, True)
            # Passed parameters checking function
            _stypy_temp_lambda_97.stypy_localization = localization
            _stypy_temp_lambda_97.stypy_type_of_self = None
            _stypy_temp_lambda_97.stypy_type_store = module_type_store
            _stypy_temp_lambda_97.stypy_function_name = '_stypy_temp_lambda_97'
            _stypy_temp_lambda_97.stypy_param_names_list = ['x']
            _stypy_temp_lambda_97.stypy_varargs_param_name = None
            _stypy_temp_lambda_97.stypy_kwargs_param_name = None
            _stypy_temp_lambda_97.stypy_call_defaults = defaults
            _stypy_temp_lambda_97.stypy_call_varargs = varargs
            _stypy_temp_lambda_97.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_97', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_97', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 23)
            x_229515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 47), 'x', False)
            int_229516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 50), 'int')
            # Applying the binary operator '**' (line 23)
            result_pow_229517 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 47), '**', x_229515, int_229516)
            
            int_229518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 54), 'int')
            # Applying the binary operator '-' (line 23)
            result_sub_229519 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 47), '-', result_pow_229517, int_229518)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 37), 'stypy_return_type', result_sub_229519)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_97' in the type store
            # Getting the type of 'stypy_return_type' (line 23)
            stypy_return_type_229520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 37), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_229520)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_97'
            return stypy_return_type_229520

        # Assigning a type to the variable '_stypy_temp_lambda_97' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 37), '_stypy_temp_lambda_97', _stypy_temp_lambda_97)
        # Getting the type of '_stypy_temp_lambda_97' (line 23)
        _stypy_temp_lambda_97_229521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 37), '_stypy_temp_lambda_97')
        # Processing the call keyword arguments (line 23)
        int_229522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 60), 'int')
        keyword_229523 = int_229522

        @norecursion
        def _stypy_temp_lambda_98(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_98'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_98', 24, 43, True)
            # Passed parameters checking function
            _stypy_temp_lambda_98.stypy_localization = localization
            _stypy_temp_lambda_98.stypy_type_of_self = None
            _stypy_temp_lambda_98.stypy_type_store = module_type_store
            _stypy_temp_lambda_98.stypy_function_name = '_stypy_temp_lambda_98'
            _stypy_temp_lambda_98.stypy_param_names_list = ['x']
            _stypy_temp_lambda_98.stypy_varargs_param_name = None
            _stypy_temp_lambda_98.stypy_kwargs_param_name = None
            _stypy_temp_lambda_98.stypy_call_defaults = defaults
            _stypy_temp_lambda_98.stypy_call_varargs = varargs
            _stypy_temp_lambda_98.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_98', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_98', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_229524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 53), 'int')
            # Getting the type of 'x' (line 24)
            x_229525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 55), 'x', False)
            # Applying the binary operator '*' (line 24)
            result_mul_229526 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 53), '*', int_229524, x_229525)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 43), 'stypy_return_type', result_mul_229526)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_98' in the type store
            # Getting the type of 'stypy_return_type' (line 24)
            stypy_return_type_229527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 43), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_229527)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_98'
            return stypy_return_type_229527

        # Assigning a type to the variable '_stypy_temp_lambda_98' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 43), '_stypy_temp_lambda_98', _stypy_temp_lambda_98)
        # Getting the type of '_stypy_temp_lambda_98' (line 24)
        _stypy_temp_lambda_98_229528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 43), '_stypy_temp_lambda_98')
        keyword_229529 = _stypy_temp_lambda_98_229528
        kwargs_229530 = {'x0': keyword_229523, 'fprime': keyword_229529}
        # Getting the type of 'scipy' (line 23)
        scipy_229512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'scipy', False)
        # Obtaining the member 'optimize' of a type (line 23)
        optimize_229513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 15), scipy_229512, 'optimize')
        # Obtaining the member 'newton' of a type (line 23)
        newton_229514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 15), optimize_229513, 'newton')
        # Calling newton(args, kwargs) (line 23)
        newton_call_result_229531 = invoke(stypy.reporting.localization.Localization(__file__, 23, 15), newton_229514, *[_stypy_temp_lambda_97_229521], **kwargs_229530)
        
        # Assigning a type to the variable 'root' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'root', newton_call_result_229531)
        
        # Call to assert_almost_equal(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'root' (line 25)
        root_229533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 28), 'root', False)
        float_229534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 34), 'float')
        # Processing the call keyword arguments (line 25)
        kwargs_229535 = {}
        # Getting the type of 'assert_almost_equal' (line 25)
        assert_almost_equal_229532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 25)
        assert_almost_equal_call_result_229536 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), assert_almost_equal_229532, *[root_229533, float_229534], **kwargs_229535)
        
        
        # ################# End of 'test_newton_integers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_newton_integers' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_229537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229537)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_newton_integers'
        return stypy_return_type_229537


    @norecursion
    def test_lmdif_errmsg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_lmdif_errmsg'
        module_type_store = module_type_store.open_function_context('test_lmdif_errmsg', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRegression.test_lmdif_errmsg.__dict__.__setitem__('stypy_localization', localization)
        TestRegression.test_lmdif_errmsg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRegression.test_lmdif_errmsg.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRegression.test_lmdif_errmsg.__dict__.__setitem__('stypy_function_name', 'TestRegression.test_lmdif_errmsg')
        TestRegression.test_lmdif_errmsg.__dict__.__setitem__('stypy_param_names_list', [])
        TestRegression.test_lmdif_errmsg.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRegression.test_lmdif_errmsg.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRegression.test_lmdif_errmsg.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRegression.test_lmdif_errmsg.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRegression.test_lmdif_errmsg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRegression.test_lmdif_errmsg.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRegression.test_lmdif_errmsg', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_lmdif_errmsg', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_lmdif_errmsg(...)' code ##################

        # Declaration of the 'SomeError' class
        # Getting the type of 'Exception' (line 29)
        Exception_229538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'Exception')

        class SomeError(Exception_229538, ):
            pass
        
        # Assigning a type to the variable 'SomeError' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'SomeError', SomeError)
        
        # Assigning a List to a Name (line 31):
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_229539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        int_229540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 18), list_229539, int_229540)
        
        # Assigning a type to the variable 'counter' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'counter', list_229539)

        @norecursion
        def func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func'
            module_type_store = module_type_store.open_function_context('func', 33, 8, False)
            
            # Passed parameters checking function
            func.stypy_localization = localization
            func.stypy_type_of_self = None
            func.stypy_type_store = module_type_store
            func.stypy_function_name = 'func'
            func.stypy_param_names_list = ['x']
            func.stypy_varargs_param_name = None
            func.stypy_kwargs_param_name = None
            func.stypy_call_defaults = defaults
            func.stypy_call_varargs = varargs
            func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func(...)' code ##################

            
            # Getting the type of 'counter' (line 34)
            counter_229541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'counter')
            
            # Obtaining the type of the subscript
            int_229542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 20), 'int')
            # Getting the type of 'counter' (line 34)
            counter_229543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'counter')
            # Obtaining the member '__getitem__' of a type (line 34)
            getitem___229544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), counter_229543, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 34)
            subscript_call_result_229545 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), getitem___229544, int_229542)
            
            int_229546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 26), 'int')
            # Applying the binary operator '+=' (line 34)
            result_iadd_229547 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 12), '+=', subscript_call_result_229545, int_229546)
            # Getting the type of 'counter' (line 34)
            counter_229548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'counter')
            int_229549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 20), 'int')
            # Storing an element on a container (line 34)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 12), counter_229548, (int_229549, result_iadd_229547))
            
            
            
            
            # Obtaining the type of the subscript
            int_229550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'int')
            # Getting the type of 'counter' (line 35)
            counter_229551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'counter')
            # Obtaining the member '__getitem__' of a type (line 35)
            getitem___229552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 15), counter_229551, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 35)
            subscript_call_result_229553 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), getitem___229552, int_229550)
            
            int_229554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'int')
            # Applying the binary operator '<' (line 35)
            result_lt_229555 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 15), '<', subscript_call_result_229553, int_229554)
            
            # Testing the type of an if condition (line 35)
            if_condition_229556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 12), result_lt_229555)
            # Assigning a type to the variable 'if_condition_229556' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'if_condition_229556', if_condition_229556)
            # SSA begins for if statement (line 35)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'x' (line 36)
            x_229557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'x')
            int_229558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 26), 'int')
            # Applying the binary operator '**' (line 36)
            result_pow_229559 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 23), '**', x_229557, int_229558)
            
            
            # Call to array(...): (line 36)
            # Processing the call arguments (line 36)
            
            # Obtaining an instance of the builtin type 'list' (line 36)
            list_229562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 39), 'list')
            # Adding type elements to the builtin type 'list' instance (line 36)
            # Adding element type (line 36)
            int_229563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 40), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 39), list_229562, int_229563)
            # Adding element type (line 36)
            int_229564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 43), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 39), list_229562, int_229564)
            # Adding element type (line 36)
            int_229565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 47), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 39), list_229562, int_229565)
            
            # Processing the call keyword arguments (line 36)
            kwargs_229566 = {}
            # Getting the type of 'np' (line 36)
            np_229560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), 'np', False)
            # Obtaining the member 'array' of a type (line 36)
            array_229561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 30), np_229560, 'array')
            # Calling array(args, kwargs) (line 36)
            array_call_result_229567 = invoke(stypy.reporting.localization.Localization(__file__, 36, 30), array_229561, *[list_229562], **kwargs_229566)
            
            # Applying the binary operator '-' (line 36)
            result_sub_229568 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 23), '-', result_pow_229559, array_call_result_229567)
            
            # Assigning a type to the variable 'stypy_return_type' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'stypy_return_type', result_sub_229568)
            # SSA branch for the else part of an if statement (line 35)
            module_type_store.open_ssa_branch('else')
            
            # Call to SomeError(...): (line 38)
            # Processing the call keyword arguments (line 38)
            kwargs_229570 = {}
            # Getting the type of 'SomeError' (line 38)
            SomeError_229569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'SomeError', False)
            # Calling SomeError(args, kwargs) (line 38)
            SomeError_call_result_229571 = invoke(stypy.reporting.localization.Localization(__file__, 38, 22), SomeError_229569, *[], **kwargs_229570)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 38, 16), SomeError_call_result_229571, 'raise parameter', BaseException)
            # SSA join for if statement (line 35)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func' in the type store
            # Getting the type of 'stypy_return_type' (line 33)
            stypy_return_type_229572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_229572)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func'
            return stypy_return_type_229572

        # Assigning a type to the variable 'func' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'func', func)
        
        # Call to assert_raises(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'SomeError' (line 39)
        SomeError_229574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 22), 'SomeError', False)
        # Getting the type of 'scipy' (line 40)
        scipy_229575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'scipy', False)
        # Obtaining the member 'optimize' of a type (line 40)
        optimize_229576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 22), scipy_229575, 'optimize')
        # Obtaining the member 'leastsq' of a type (line 40)
        leastsq_229577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 22), optimize_229576, 'leastsq')
        # Getting the type of 'func' (line 41)
        func_229578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'func', False)
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_229579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        int_229580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 28), list_229579, int_229580)
        # Adding element type (line 41)
        int_229581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 28), list_229579, int_229581)
        # Adding element type (line 41)
        int_229582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 28), list_229579, int_229582)
        
        # Processing the call keyword arguments (line 39)
        kwargs_229583 = {}
        # Getting the type of 'assert_raises' (line 39)
        assert_raises_229573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 39)
        assert_raises_call_result_229584 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), assert_raises_229573, *[SomeError_229574, leastsq_229577, func_229578, list_229579], **kwargs_229583)
        
        
        # ################# End of 'test_lmdif_errmsg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_lmdif_errmsg' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_229585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229585)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_lmdif_errmsg'
        return stypy_return_type_229585


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRegression.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestRegression' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'TestRegression', TestRegression)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
