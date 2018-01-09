
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Unit tests for nonnegative least squares
2: Author: Uwe Schmitt
3: Sep 2008
4: '''
5: from __future__ import division, print_function, absolute_import
6: 
7: from numpy.testing import assert_
8: 
9: from scipy.optimize import nnls
10: from numpy import arange, dot
11: from numpy.linalg import norm
12: 
13: 
14: class TestNNLS(object):
15: 
16:     def test_nnls(self):
17:         a = arange(25.0).reshape(-1,5)
18:         x = arange(5.0)
19:         y = dot(a,x)
20:         x, res = nnls(a,y)
21:         assert_(res < 1e-7)
22:         assert_(norm(dot(a,x)-y) < 1e-7)
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_221486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', ' Unit tests for nonnegative least squares\nAuthor: Uwe Schmitt\nSep 2008\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_221487 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_221487) is not StypyTypeError):

    if (import_221487 != 'pyd_module'):
        __import__(import_221487)
        sys_modules_221488 = sys.modules[import_221487]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_221488.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_221488, sys_modules_221488.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_221487)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.optimize import nnls' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_221489 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize')

if (type(import_221489) is not StypyTypeError):

    if (import_221489 != 'pyd_module'):
        __import__(import_221489)
        sys_modules_221490 = sys.modules[import_221489]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', sys_modules_221490.module_type_store, module_type_store, ['nnls'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_221490, sys_modules_221490.module_type_store, module_type_store)
    else:
        from scipy.optimize import nnls

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', None, module_type_store, ['nnls'], [nnls])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', import_221489)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy import arange, dot' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_221491 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_221491) is not StypyTypeError):

    if (import_221491 != 'pyd_module'):
        __import__(import_221491)
        sys_modules_221492 = sys.modules[import_221491]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', sys_modules_221492.module_type_store, module_type_store, ['arange', 'dot'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_221492, sys_modules_221492.module_type_store, module_type_store)
    else:
        from numpy import arange, dot

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', None, module_type_store, ['arange', 'dot'], [arange, dot])

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_221491)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.linalg import norm' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_221493 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.linalg')

if (type(import_221493) is not StypyTypeError):

    if (import_221493 != 'pyd_module'):
        __import__(import_221493)
        sys_modules_221494 = sys.modules[import_221493]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.linalg', sys_modules_221494.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_221494, sys_modules_221494.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.linalg', import_221493)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

# Declaration of the 'TestNNLS' class

class TestNNLS(object, ):

    @norecursion
    def test_nnls(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_nnls'
        module_type_store = module_type_store.open_function_context('test_nnls', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNNLS.test_nnls.__dict__.__setitem__('stypy_localization', localization)
        TestNNLS.test_nnls.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNNLS.test_nnls.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNNLS.test_nnls.__dict__.__setitem__('stypy_function_name', 'TestNNLS.test_nnls')
        TestNNLS.test_nnls.__dict__.__setitem__('stypy_param_names_list', [])
        TestNNLS.test_nnls.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNNLS.test_nnls.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNNLS.test_nnls.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNNLS.test_nnls.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNNLS.test_nnls.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNNLS.test_nnls.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNNLS.test_nnls', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_nnls', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_nnls(...)' code ##################

        
        # Assigning a Call to a Name (line 17):
        
        # Assigning a Call to a Name (line 17):
        
        # Call to reshape(...): (line 17)
        # Processing the call arguments (line 17)
        int_221500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 33), 'int')
        int_221501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 36), 'int')
        # Processing the call keyword arguments (line 17)
        kwargs_221502 = {}
        
        # Call to arange(...): (line 17)
        # Processing the call arguments (line 17)
        float_221496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'float')
        # Processing the call keyword arguments (line 17)
        kwargs_221497 = {}
        # Getting the type of 'arange' (line 17)
        arange_221495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 17)
        arange_call_result_221498 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), arange_221495, *[float_221496], **kwargs_221497)
        
        # Obtaining the member 'reshape' of a type (line 17)
        reshape_221499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 12), arange_call_result_221498, 'reshape')
        # Calling reshape(args, kwargs) (line 17)
        reshape_call_result_221503 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), reshape_221499, *[int_221500, int_221501], **kwargs_221502)
        
        # Assigning a type to the variable 'a' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'a', reshape_call_result_221503)
        
        # Assigning a Call to a Name (line 18):
        
        # Assigning a Call to a Name (line 18):
        
        # Call to arange(...): (line 18)
        # Processing the call arguments (line 18)
        float_221505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 19), 'float')
        # Processing the call keyword arguments (line 18)
        kwargs_221506 = {}
        # Getting the type of 'arange' (line 18)
        arange_221504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 18)
        arange_call_result_221507 = invoke(stypy.reporting.localization.Localization(__file__, 18, 12), arange_221504, *[float_221505], **kwargs_221506)
        
        # Assigning a type to the variable 'x' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'x', arange_call_result_221507)
        
        # Assigning a Call to a Name (line 19):
        
        # Assigning a Call to a Name (line 19):
        
        # Call to dot(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'a' (line 19)
        a_221509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'a', False)
        # Getting the type of 'x' (line 19)
        x_221510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 18), 'x', False)
        # Processing the call keyword arguments (line 19)
        kwargs_221511 = {}
        # Getting the type of 'dot' (line 19)
        dot_221508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'dot', False)
        # Calling dot(args, kwargs) (line 19)
        dot_call_result_221512 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), dot_221508, *[a_221509, x_221510], **kwargs_221511)
        
        # Assigning a type to the variable 'y' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'y', dot_call_result_221512)
        
        # Assigning a Call to a Tuple (line 20):
        
        # Assigning a Subscript to a Name (line 20):
        
        # Obtaining the type of the subscript
        int_221513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'int')
        
        # Call to nnls(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'a' (line 20)
        a_221515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'a', False)
        # Getting the type of 'y' (line 20)
        y_221516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 24), 'y', False)
        # Processing the call keyword arguments (line 20)
        kwargs_221517 = {}
        # Getting the type of 'nnls' (line 20)
        nnls_221514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'nnls', False)
        # Calling nnls(args, kwargs) (line 20)
        nnls_call_result_221518 = invoke(stypy.reporting.localization.Localization(__file__, 20, 17), nnls_221514, *[a_221515, y_221516], **kwargs_221517)
        
        # Obtaining the member '__getitem__' of a type (line 20)
        getitem___221519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), nnls_call_result_221518, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 20)
        subscript_call_result_221520 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), getitem___221519, int_221513)
        
        # Assigning a type to the variable 'tuple_var_assignment_221484' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'tuple_var_assignment_221484', subscript_call_result_221520)
        
        # Assigning a Subscript to a Name (line 20):
        
        # Obtaining the type of the subscript
        int_221521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'int')
        
        # Call to nnls(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'a' (line 20)
        a_221523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'a', False)
        # Getting the type of 'y' (line 20)
        y_221524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 24), 'y', False)
        # Processing the call keyword arguments (line 20)
        kwargs_221525 = {}
        # Getting the type of 'nnls' (line 20)
        nnls_221522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'nnls', False)
        # Calling nnls(args, kwargs) (line 20)
        nnls_call_result_221526 = invoke(stypy.reporting.localization.Localization(__file__, 20, 17), nnls_221522, *[a_221523, y_221524], **kwargs_221525)
        
        # Obtaining the member '__getitem__' of a type (line 20)
        getitem___221527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), nnls_call_result_221526, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 20)
        subscript_call_result_221528 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), getitem___221527, int_221521)
        
        # Assigning a type to the variable 'tuple_var_assignment_221485' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'tuple_var_assignment_221485', subscript_call_result_221528)
        
        # Assigning a Name to a Name (line 20):
        # Getting the type of 'tuple_var_assignment_221484' (line 20)
        tuple_var_assignment_221484_221529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'tuple_var_assignment_221484')
        # Assigning a type to the variable 'x' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'x', tuple_var_assignment_221484_221529)
        
        # Assigning a Name to a Name (line 20):
        # Getting the type of 'tuple_var_assignment_221485' (line 20)
        tuple_var_assignment_221485_221530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'tuple_var_assignment_221485')
        # Assigning a type to the variable 'res' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'res', tuple_var_assignment_221485_221530)
        
        # Call to assert_(...): (line 21)
        # Processing the call arguments (line 21)
        
        # Getting the type of 'res' (line 21)
        res_221532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'res', False)
        float_221533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 22), 'float')
        # Applying the binary operator '<' (line 21)
        result_lt_221534 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 16), '<', res_221532, float_221533)
        
        # Processing the call keyword arguments (line 21)
        kwargs_221535 = {}
        # Getting the type of 'assert_' (line 21)
        assert__221531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 21)
        assert__call_result_221536 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assert__221531, *[result_lt_221534], **kwargs_221535)
        
        
        # Call to assert_(...): (line 22)
        # Processing the call arguments (line 22)
        
        
        # Call to norm(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Call to dot(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'a' (line 22)
        a_221540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'a', False)
        # Getting the type of 'x' (line 22)
        x_221541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 27), 'x', False)
        # Processing the call keyword arguments (line 22)
        kwargs_221542 = {}
        # Getting the type of 'dot' (line 22)
        dot_221539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 21), 'dot', False)
        # Calling dot(args, kwargs) (line 22)
        dot_call_result_221543 = invoke(stypy.reporting.localization.Localization(__file__, 22, 21), dot_221539, *[a_221540, x_221541], **kwargs_221542)
        
        # Getting the type of 'y' (line 22)
        y_221544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 30), 'y', False)
        # Applying the binary operator '-' (line 22)
        result_sub_221545 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 21), '-', dot_call_result_221543, y_221544)
        
        # Processing the call keyword arguments (line 22)
        kwargs_221546 = {}
        # Getting the type of 'norm' (line 22)
        norm_221538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'norm', False)
        # Calling norm(args, kwargs) (line 22)
        norm_call_result_221547 = invoke(stypy.reporting.localization.Localization(__file__, 22, 16), norm_221538, *[result_sub_221545], **kwargs_221546)
        
        float_221548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 35), 'float')
        # Applying the binary operator '<' (line 22)
        result_lt_221549 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 16), '<', norm_call_result_221547, float_221548)
        
        # Processing the call keyword arguments (line 22)
        kwargs_221550 = {}
        # Getting the type of 'assert_' (line 22)
        assert__221537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 22)
        assert__call_result_221551 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), assert__221537, *[result_lt_221549], **kwargs_221550)
        
        
        # ################# End of 'test_nnls(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_nnls' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_221552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_nnls'
        return stypy_return_type_221552


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 0, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNNLS.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestNNLS' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'TestNNLS', TestNNLS)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
