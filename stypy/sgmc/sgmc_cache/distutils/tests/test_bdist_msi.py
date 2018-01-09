
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.bdist_msi.'''
2: import sys
3: import unittest
4: from test.test_support import run_unittest
5: from distutils.tests import support
6: 
7: 
8: @unittest.skipUnless(sys.platform == 'win32', 'these tests require Windows')
9: class BDistMSITestCase(support.TempdirManager,
10:                        support.LoggingSilencer,
11:                        unittest.TestCase):
12: 
13:     def test_minimal(self):
14:         # minimal test XXX need more tests
15:         from distutils.command.bdist_msi import bdist_msi
16:         project_dir, dist = self.create_dist()
17:         cmd = bdist_msi(dist)
18:         cmd.ensure_finalized()
19: 
20: 
21: def test_suite():
22:     return unittest.makeSuite(BDistMSITestCase)
23: 
24: if __name__ == '__main__':
25:     run_unittest(test_suite())
26: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_30407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.bdist_msi.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import unittest' statement (line 3)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from test.test_support import run_unittest' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30408 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support')

if (type(import_30408) is not StypyTypeError):

    if (import_30408 != 'pyd_module'):
        __import__(import_30408)
        sys_modules_30409 = sys.modules[import_30408]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', sys_modules_30409.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_30409, sys_modules_30409.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', import_30408)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from distutils.tests import support' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30410 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.tests')

if (type(import_30410) is not StypyTypeError):

    if (import_30410 != 'pyd_module'):
        __import__(import_30410)
        sys_modules_30411 = sys.modules[import_30410]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.tests', sys_modules_30411.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_30411, sys_modules_30411.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.tests', import_30410)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'BDistMSITestCase' class
# Getting the type of 'support' (line 9)
support_30412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 23), 'support')
# Obtaining the member 'TempdirManager' of a type (line 9)
TempdirManager_30413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 23), support_30412, 'TempdirManager')
# Getting the type of 'support' (line 10)
support_30414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 23), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 10)
LoggingSilencer_30415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 23), support_30414, 'LoggingSilencer')
# Getting the type of 'unittest' (line 11)
unittest_30416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 23), 'unittest')
# Obtaining the member 'TestCase' of a type (line 11)
TestCase_30417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 23), unittest_30416, 'TestCase')

class BDistMSITestCase(TempdirManager_30413, LoggingSilencer_30415, TestCase_30417, ):

    @norecursion
    def test_minimal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimal'
        module_type_store = module_type_store.open_function_context('test_minimal', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BDistMSITestCase.test_minimal.__dict__.__setitem__('stypy_localization', localization)
        BDistMSITestCase.test_minimal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BDistMSITestCase.test_minimal.__dict__.__setitem__('stypy_type_store', module_type_store)
        BDistMSITestCase.test_minimal.__dict__.__setitem__('stypy_function_name', 'BDistMSITestCase.test_minimal')
        BDistMSITestCase.test_minimal.__dict__.__setitem__('stypy_param_names_list', [])
        BDistMSITestCase.test_minimal.__dict__.__setitem__('stypy_varargs_param_name', None)
        BDistMSITestCase.test_minimal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BDistMSITestCase.test_minimal.__dict__.__setitem__('stypy_call_defaults', defaults)
        BDistMSITestCase.test_minimal.__dict__.__setitem__('stypy_call_varargs', varargs)
        BDistMSITestCase.test_minimal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BDistMSITestCase.test_minimal.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BDistMSITestCase.test_minimal', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimal', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimal(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 8))
        
        # 'from distutils.command.bdist_msi import bdist_msi' statement (line 15)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_30418 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 8), 'distutils.command.bdist_msi')

        if (type(import_30418) is not StypyTypeError):

            if (import_30418 != 'pyd_module'):
                __import__(import_30418)
                sys_modules_30419 = sys.modules[import_30418]
                import_from_module(stypy.reporting.localization.Localization(__file__, 15, 8), 'distutils.command.bdist_msi', sys_modules_30419.module_type_store, module_type_store, ['bdist_msi'])
                nest_module(stypy.reporting.localization.Localization(__file__, 15, 8), __file__, sys_modules_30419, sys_modules_30419.module_type_store, module_type_store)
            else:
                from distutils.command.bdist_msi import bdist_msi

                import_from_module(stypy.reporting.localization.Localization(__file__, 15, 8), 'distutils.command.bdist_msi', None, module_type_store, ['bdist_msi'], [bdist_msi])

        else:
            # Assigning a type to the variable 'distutils.command.bdist_msi' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'distutils.command.bdist_msi', import_30418)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        
        # Assigning a Call to a Tuple (line 16):
        
        # Assigning a Subscript to a Name (line 16):
        
        # Obtaining the type of the subscript
        int_30420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'int')
        
        # Call to create_dist(...): (line 16)
        # Processing the call keyword arguments (line 16)
        kwargs_30423 = {}
        # Getting the type of 'self' (line 16)
        self_30421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 28), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 16)
        create_dist_30422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 28), self_30421, 'create_dist')
        # Calling create_dist(args, kwargs) (line 16)
        create_dist_call_result_30424 = invoke(stypy.reporting.localization.Localization(__file__, 16, 28), create_dist_30422, *[], **kwargs_30423)
        
        # Obtaining the member '__getitem__' of a type (line 16)
        getitem___30425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), create_dist_call_result_30424, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 16)
        subscript_call_result_30426 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), getitem___30425, int_30420)
        
        # Assigning a type to the variable 'tuple_var_assignment_30405' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_30405', subscript_call_result_30426)
        
        # Assigning a Subscript to a Name (line 16):
        
        # Obtaining the type of the subscript
        int_30427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'int')
        
        # Call to create_dist(...): (line 16)
        # Processing the call keyword arguments (line 16)
        kwargs_30430 = {}
        # Getting the type of 'self' (line 16)
        self_30428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 28), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 16)
        create_dist_30429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 28), self_30428, 'create_dist')
        # Calling create_dist(args, kwargs) (line 16)
        create_dist_call_result_30431 = invoke(stypy.reporting.localization.Localization(__file__, 16, 28), create_dist_30429, *[], **kwargs_30430)
        
        # Obtaining the member '__getitem__' of a type (line 16)
        getitem___30432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), create_dist_call_result_30431, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 16)
        subscript_call_result_30433 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), getitem___30432, int_30427)
        
        # Assigning a type to the variable 'tuple_var_assignment_30406' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_30406', subscript_call_result_30433)
        
        # Assigning a Name to a Name (line 16):
        # Getting the type of 'tuple_var_assignment_30405' (line 16)
        tuple_var_assignment_30405_30434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_30405')
        # Assigning a type to the variable 'project_dir' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'project_dir', tuple_var_assignment_30405_30434)
        
        # Assigning a Name to a Name (line 16):
        # Getting the type of 'tuple_var_assignment_30406' (line 16)
        tuple_var_assignment_30406_30435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_30406')
        # Assigning a type to the variable 'dist' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 21), 'dist', tuple_var_assignment_30406_30435)
        
        # Assigning a Call to a Name (line 17):
        
        # Assigning a Call to a Name (line 17):
        
        # Call to bdist_msi(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'dist' (line 17)
        dist_30437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'dist', False)
        # Processing the call keyword arguments (line 17)
        kwargs_30438 = {}
        # Getting the type of 'bdist_msi' (line 17)
        bdist_msi_30436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'bdist_msi', False)
        # Calling bdist_msi(args, kwargs) (line 17)
        bdist_msi_call_result_30439 = invoke(stypy.reporting.localization.Localization(__file__, 17, 14), bdist_msi_30436, *[dist_30437], **kwargs_30438)
        
        # Assigning a type to the variable 'cmd' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'cmd', bdist_msi_call_result_30439)
        
        # Call to ensure_finalized(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_30442 = {}
        # Getting the type of 'cmd' (line 18)
        cmd_30440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 18)
        ensure_finalized_30441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), cmd_30440, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 18)
        ensure_finalized_call_result_30443 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), ensure_finalized_30441, *[], **kwargs_30442)
        
        
        # ################# End of 'test_minimal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimal' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_30444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30444)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimal'
        return stypy_return_type_30444


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BDistMSITestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BDistMSITestCase' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'BDistMSITestCase', BDistMSITestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 21, 0, False)
    
    # Passed parameters checking function
    test_suite.stypy_localization = localization
    test_suite.stypy_type_of_self = None
    test_suite.stypy_type_store = module_type_store
    test_suite.stypy_function_name = 'test_suite'
    test_suite.stypy_param_names_list = []
    test_suite.stypy_varargs_param_name = None
    test_suite.stypy_kwargs_param_name = None
    test_suite.stypy_call_defaults = defaults
    test_suite.stypy_call_varargs = varargs
    test_suite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_suite', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_suite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_suite(...)' code ##################

    
    # Call to makeSuite(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'BDistMSITestCase' (line 22)
    BDistMSITestCase_30447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 30), 'BDistMSITestCase', False)
    # Processing the call keyword arguments (line 22)
    kwargs_30448 = {}
    # Getting the type of 'unittest' (line 22)
    unittest_30445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 22)
    makeSuite_30446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 11), unittest_30445, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 22)
    makeSuite_call_result_30449 = invoke(stypy.reporting.localization.Localization(__file__, 22, 11), makeSuite_30446, *[BDistMSITestCase_30447], **kwargs_30448)
    
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type', makeSuite_call_result_30449)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_30450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30450)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_30450

# Assigning a type to the variable 'test_suite' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Call to test_suite(...): (line 25)
    # Processing the call keyword arguments (line 25)
    kwargs_30453 = {}
    # Getting the type of 'test_suite' (line 25)
    test_suite_30452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 25)
    test_suite_call_result_30454 = invoke(stypy.reporting.localization.Localization(__file__, 25, 17), test_suite_30452, *[], **kwargs_30453)
    
    # Processing the call keyword arguments (line 25)
    kwargs_30455 = {}
    # Getting the type of 'run_unittest' (line 25)
    run_unittest_30451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 25)
    run_unittest_call_result_30456 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), run_unittest_30451, *[test_suite_call_result_30454], **kwargs_30455)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
