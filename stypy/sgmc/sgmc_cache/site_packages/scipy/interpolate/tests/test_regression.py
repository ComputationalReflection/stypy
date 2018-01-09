
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: import scipy.interpolate as interp
5: from numpy.testing import assert_almost_equal
6: 
7: 
8: class TestRegression(object):
9:     def test_spalde_scalar_input(self):
10:         '''Ticket #629'''
11:         x = np.linspace(0,10)
12:         y = x**3
13:         tck = interp.splrep(x, y, k=3, t=[5])
14:         res = interp.spalde(np.float64(1), tck)
15:         des = np.array([1., 3., 6., 6.])
16:         assert_almost_equal(res, des)
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_119550 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_119550) is not StypyTypeError):

    if (import_119550 != 'pyd_module'):
        __import__(import_119550)
        sys_modules_119551 = sys.modules[import_119550]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_119551.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_119550)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import scipy.interpolate' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_119552 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.interpolate')

if (type(import_119552) is not StypyTypeError):

    if (import_119552 != 'pyd_module'):
        __import__(import_119552)
        sys_modules_119553 = sys.modules[import_119552]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'interp', sys_modules_119553.module_type_store, module_type_store)
    else:
        import scipy.interpolate as interp

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'interp', scipy.interpolate, module_type_store)

else:
    # Assigning a type to the variable 'scipy.interpolate' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.interpolate', import_119552)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.testing import assert_almost_equal' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_119554 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing')

if (type(import_119554) is not StypyTypeError):

    if (import_119554 != 'pyd_module'):
        __import__(import_119554)
        sys_modules_119555 = sys.modules[import_119554]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', sys_modules_119555.module_type_store, module_type_store, ['assert_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_119555, sys_modules_119555.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', None, module_type_store, ['assert_almost_equal'], [assert_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', import_119554)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

# Declaration of the 'TestRegression' class

class TestRegression(object, ):

    @norecursion
    def test_spalde_scalar_input(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spalde_scalar_input'
        module_type_store = module_type_store.open_function_context('test_spalde_scalar_input', 9, 4, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRegression.test_spalde_scalar_input.__dict__.__setitem__('stypy_localization', localization)
        TestRegression.test_spalde_scalar_input.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRegression.test_spalde_scalar_input.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRegression.test_spalde_scalar_input.__dict__.__setitem__('stypy_function_name', 'TestRegression.test_spalde_scalar_input')
        TestRegression.test_spalde_scalar_input.__dict__.__setitem__('stypy_param_names_list', [])
        TestRegression.test_spalde_scalar_input.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRegression.test_spalde_scalar_input.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRegression.test_spalde_scalar_input.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRegression.test_spalde_scalar_input.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRegression.test_spalde_scalar_input.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRegression.test_spalde_scalar_input.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRegression.test_spalde_scalar_input', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spalde_scalar_input', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spalde_scalar_input(...)' code ##################

        str_119556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', 'Ticket #629')
        
        # Assigning a Call to a Name (line 11):
        
        # Call to linspace(...): (line 11)
        # Processing the call arguments (line 11)
        int_119559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 24), 'int')
        int_119560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'int')
        # Processing the call keyword arguments (line 11)
        kwargs_119561 = {}
        # Getting the type of 'np' (line 11)
        np_119557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 11)
        linspace_119558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 12), np_119557, 'linspace')
        # Calling linspace(args, kwargs) (line 11)
        linspace_call_result_119562 = invoke(stypy.reporting.localization.Localization(__file__, 11, 12), linspace_119558, *[int_119559, int_119560], **kwargs_119561)
        
        # Assigning a type to the variable 'x' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'x', linspace_call_result_119562)
        
        # Assigning a BinOp to a Name (line 12):
        # Getting the type of 'x' (line 12)
        x_119563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'x')
        int_119564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'int')
        # Applying the binary operator '**' (line 12)
        result_pow_119565 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), '**', x_119563, int_119564)
        
        # Assigning a type to the variable 'y' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'y', result_pow_119565)
        
        # Assigning a Call to a Name (line 13):
        
        # Call to splrep(...): (line 13)
        # Processing the call arguments (line 13)
        # Getting the type of 'x' (line 13)
        x_119568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'x', False)
        # Getting the type of 'y' (line 13)
        y_119569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 31), 'y', False)
        # Processing the call keyword arguments (line 13)
        int_119570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 36), 'int')
        keyword_119571 = int_119570
        
        # Obtaining an instance of the builtin type 'list' (line 13)
        list_119572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 13)
        # Adding element type (line 13)
        int_119573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 41), list_119572, int_119573)
        
        keyword_119574 = list_119572
        kwargs_119575 = {'k': keyword_119571, 't': keyword_119574}
        # Getting the type of 'interp' (line 13)
        interp_119566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'interp', False)
        # Obtaining the member 'splrep' of a type (line 13)
        splrep_119567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 14), interp_119566, 'splrep')
        # Calling splrep(args, kwargs) (line 13)
        splrep_call_result_119576 = invoke(stypy.reporting.localization.Localization(__file__, 13, 14), splrep_119567, *[x_119568, y_119569], **kwargs_119575)
        
        # Assigning a type to the variable 'tck' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'tck', splrep_call_result_119576)
        
        # Assigning a Call to a Name (line 14):
        
        # Call to spalde(...): (line 14)
        # Processing the call arguments (line 14)
        
        # Call to float64(...): (line 14)
        # Processing the call arguments (line 14)
        int_119581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 39), 'int')
        # Processing the call keyword arguments (line 14)
        kwargs_119582 = {}
        # Getting the type of 'np' (line 14)
        np_119579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 28), 'np', False)
        # Obtaining the member 'float64' of a type (line 14)
        float64_119580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 28), np_119579, 'float64')
        # Calling float64(args, kwargs) (line 14)
        float64_call_result_119583 = invoke(stypy.reporting.localization.Localization(__file__, 14, 28), float64_119580, *[int_119581], **kwargs_119582)
        
        # Getting the type of 'tck' (line 14)
        tck_119584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 43), 'tck', False)
        # Processing the call keyword arguments (line 14)
        kwargs_119585 = {}
        # Getting the type of 'interp' (line 14)
        interp_119577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'interp', False)
        # Obtaining the member 'spalde' of a type (line 14)
        spalde_119578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 14), interp_119577, 'spalde')
        # Calling spalde(args, kwargs) (line 14)
        spalde_call_result_119586 = invoke(stypy.reporting.localization.Localization(__file__, 14, 14), spalde_119578, *[float64_call_result_119583, tck_119584], **kwargs_119585)
        
        # Assigning a type to the variable 'res' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'res', spalde_call_result_119586)
        
        # Assigning a Call to a Name (line 15):
        
        # Call to array(...): (line 15)
        # Processing the call arguments (line 15)
        
        # Obtaining an instance of the builtin type 'list' (line 15)
        list_119589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 15)
        # Adding element type (line 15)
        float_119590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 23), list_119589, float_119590)
        # Adding element type (line 15)
        float_119591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 23), list_119589, float_119591)
        # Adding element type (line 15)
        float_119592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 23), list_119589, float_119592)
        # Adding element type (line 15)
        float_119593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 23), list_119589, float_119593)
        
        # Processing the call keyword arguments (line 15)
        kwargs_119594 = {}
        # Getting the type of 'np' (line 15)
        np_119587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 14), 'np', False)
        # Obtaining the member 'array' of a type (line 15)
        array_119588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 14), np_119587, 'array')
        # Calling array(args, kwargs) (line 15)
        array_call_result_119595 = invoke(stypy.reporting.localization.Localization(__file__, 15, 14), array_119588, *[list_119589], **kwargs_119594)
        
        # Assigning a type to the variable 'des' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'des', array_call_result_119595)
        
        # Call to assert_almost_equal(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'res' (line 16)
        res_119597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 28), 'res', False)
        # Getting the type of 'des' (line 16)
        des_119598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 33), 'des', False)
        # Processing the call keyword arguments (line 16)
        kwargs_119599 = {}
        # Getting the type of 'assert_almost_equal' (line 16)
        assert_almost_equal_119596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 16)
        assert_almost_equal_call_result_119600 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), assert_almost_equal_119596, *[res_119597, des_119598], **kwargs_119599)
        
        
        # ################# End of 'test_spalde_scalar_input(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spalde_scalar_input' in the type store
        # Getting the type of 'stypy_return_type' (line 9)
        stypy_return_type_119601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_119601)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spalde_scalar_input'
        return stypy_return_type_119601


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 8, 0, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'self', type_of_self)
        
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


# Assigning a type to the variable 'TestRegression' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'TestRegression', TestRegression)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
