
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.bdist_wininst.'''
2: import unittest
3: 
4: from test.test_support import run_unittest
5: 
6: from distutils.command.bdist_wininst import bdist_wininst
7: from distutils.tests import support
8: 
9: class BuildWinInstTestCase(support.TempdirManager,
10:                            support.LoggingSilencer,
11:                            unittest.TestCase):
12: 
13:     def test_get_exe_bytes(self):
14: 
15:         # issue5731: command was broken on non-windows platforms
16:         # this test makes sure it works now for every platform
17:         # let's create a command
18:         pkg_pth, dist = self.create_dist()
19:         cmd = bdist_wininst(dist)
20:         cmd.ensure_finalized()
21: 
22:         # let's run the code that finds the right wininst*.exe file
23:         # and make sure it finds it and returns its content
24:         # no matter what platform we have
25:         exe_file = cmd.get_exe_bytes()
26:         self.assertGreater(len(exe_file), 10)
27: 
28: def test_suite():
29:     return unittest.makeSuite(BuildWinInstTestCase)
30: 
31: if __name__ == '__main__':
32:     run_unittest(test_suite())
33: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_30824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.bdist_wininst.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import unittest' statement (line 2)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from test.test_support import run_unittest' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30825 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support')

if (type(import_30825) is not StypyTypeError):

    if (import_30825 != 'pyd_module'):
        __import__(import_30825)
        sys_modules_30826 = sys.modules[import_30825]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', sys_modules_30826.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_30826, sys_modules_30826.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', import_30825)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.command.bdist_wininst import bdist_wininst' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30827 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.bdist_wininst')

if (type(import_30827) is not StypyTypeError):

    if (import_30827 != 'pyd_module'):
        __import__(import_30827)
        sys_modules_30828 = sys.modules[import_30827]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.bdist_wininst', sys_modules_30828.module_type_store, module_type_store, ['bdist_wininst'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_30828, sys_modules_30828.module_type_store, module_type_store)
    else:
        from distutils.command.bdist_wininst import bdist_wininst

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.bdist_wininst', None, module_type_store, ['bdist_wininst'], [bdist_wininst])

else:
    # Assigning a type to the variable 'distutils.command.bdist_wininst' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.bdist_wininst', import_30827)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.tests import support' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30829 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.tests')

if (type(import_30829) is not StypyTypeError):

    if (import_30829 != 'pyd_module'):
        __import__(import_30829)
        sys_modules_30830 = sys.modules[import_30829]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.tests', sys_modules_30830.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_30830, sys_modules_30830.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.tests', import_30829)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'BuildWinInstTestCase' class
# Getting the type of 'support' (line 9)
support_30831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 27), 'support')
# Obtaining the member 'TempdirManager' of a type (line 9)
TempdirManager_30832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 27), support_30831, 'TempdirManager')
# Getting the type of 'support' (line 10)
support_30833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 27), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 10)
LoggingSilencer_30834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 27), support_30833, 'LoggingSilencer')
# Getting the type of 'unittest' (line 11)
unittest_30835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 27), 'unittest')
# Obtaining the member 'TestCase' of a type (line 11)
TestCase_30836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 27), unittest_30835, 'TestCase')

class BuildWinInstTestCase(TempdirManager_30832, LoggingSilencer_30834, TestCase_30836, ):

    @norecursion
    def test_get_exe_bytes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_get_exe_bytes'
        module_type_store = module_type_store.open_function_context('test_get_exe_bytes', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildWinInstTestCase.test_get_exe_bytes.__dict__.__setitem__('stypy_localization', localization)
        BuildWinInstTestCase.test_get_exe_bytes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildWinInstTestCase.test_get_exe_bytes.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildWinInstTestCase.test_get_exe_bytes.__dict__.__setitem__('stypy_function_name', 'BuildWinInstTestCase.test_get_exe_bytes')
        BuildWinInstTestCase.test_get_exe_bytes.__dict__.__setitem__('stypy_param_names_list', [])
        BuildWinInstTestCase.test_get_exe_bytes.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildWinInstTestCase.test_get_exe_bytes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildWinInstTestCase.test_get_exe_bytes.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildWinInstTestCase.test_get_exe_bytes.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildWinInstTestCase.test_get_exe_bytes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildWinInstTestCase.test_get_exe_bytes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildWinInstTestCase.test_get_exe_bytes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_get_exe_bytes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_get_exe_bytes(...)' code ##################

        
        # Assigning a Call to a Tuple (line 18):
        
        # Assigning a Subscript to a Name (line 18):
        
        # Obtaining the type of the subscript
        int_30837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'int')
        
        # Call to create_dist(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_30840 = {}
        # Getting the type of 'self' (line 18)
        self_30838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 18)
        create_dist_30839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 24), self_30838, 'create_dist')
        # Calling create_dist(args, kwargs) (line 18)
        create_dist_call_result_30841 = invoke(stypy.reporting.localization.Localization(__file__, 18, 24), create_dist_30839, *[], **kwargs_30840)
        
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___30842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), create_dist_call_result_30841, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_30843 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), getitem___30842, int_30837)
        
        # Assigning a type to the variable 'tuple_var_assignment_30822' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_30822', subscript_call_result_30843)
        
        # Assigning a Subscript to a Name (line 18):
        
        # Obtaining the type of the subscript
        int_30844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'int')
        
        # Call to create_dist(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_30847 = {}
        # Getting the type of 'self' (line 18)
        self_30845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 18)
        create_dist_30846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 24), self_30845, 'create_dist')
        # Calling create_dist(args, kwargs) (line 18)
        create_dist_call_result_30848 = invoke(stypy.reporting.localization.Localization(__file__, 18, 24), create_dist_30846, *[], **kwargs_30847)
        
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___30849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), create_dist_call_result_30848, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_30850 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), getitem___30849, int_30844)
        
        # Assigning a type to the variable 'tuple_var_assignment_30823' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_30823', subscript_call_result_30850)
        
        # Assigning a Name to a Name (line 18):
        # Getting the type of 'tuple_var_assignment_30822' (line 18)
        tuple_var_assignment_30822_30851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_30822')
        # Assigning a type to the variable 'pkg_pth' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'pkg_pth', tuple_var_assignment_30822_30851)
        
        # Assigning a Name to a Name (line 18):
        # Getting the type of 'tuple_var_assignment_30823' (line 18)
        tuple_var_assignment_30823_30852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_30823')
        # Assigning a type to the variable 'dist' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'dist', tuple_var_assignment_30823_30852)
        
        # Assigning a Call to a Name (line 19):
        
        # Assigning a Call to a Name (line 19):
        
        # Call to bdist_wininst(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'dist' (line 19)
        dist_30854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 28), 'dist', False)
        # Processing the call keyword arguments (line 19)
        kwargs_30855 = {}
        # Getting the type of 'bdist_wininst' (line 19)
        bdist_wininst_30853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'bdist_wininst', False)
        # Calling bdist_wininst(args, kwargs) (line 19)
        bdist_wininst_call_result_30856 = invoke(stypy.reporting.localization.Localization(__file__, 19, 14), bdist_wininst_30853, *[dist_30854], **kwargs_30855)
        
        # Assigning a type to the variable 'cmd' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'cmd', bdist_wininst_call_result_30856)
        
        # Call to ensure_finalized(...): (line 20)
        # Processing the call keyword arguments (line 20)
        kwargs_30859 = {}
        # Getting the type of 'cmd' (line 20)
        cmd_30857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 20)
        ensure_finalized_30858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), cmd_30857, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 20)
        ensure_finalized_call_result_30860 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), ensure_finalized_30858, *[], **kwargs_30859)
        
        
        # Assigning a Call to a Name (line 25):
        
        # Assigning a Call to a Name (line 25):
        
        # Call to get_exe_bytes(...): (line 25)
        # Processing the call keyword arguments (line 25)
        kwargs_30863 = {}
        # Getting the type of 'cmd' (line 25)
        cmd_30861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'cmd', False)
        # Obtaining the member 'get_exe_bytes' of a type (line 25)
        get_exe_bytes_30862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 19), cmd_30861, 'get_exe_bytes')
        # Calling get_exe_bytes(args, kwargs) (line 25)
        get_exe_bytes_call_result_30864 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), get_exe_bytes_30862, *[], **kwargs_30863)
        
        # Assigning a type to the variable 'exe_file' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'exe_file', get_exe_bytes_call_result_30864)
        
        # Call to assertGreater(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Call to len(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'exe_file' (line 26)
        exe_file_30868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'exe_file', False)
        # Processing the call keyword arguments (line 26)
        kwargs_30869 = {}
        # Getting the type of 'len' (line 26)
        len_30867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 27), 'len', False)
        # Calling len(args, kwargs) (line 26)
        len_call_result_30870 = invoke(stypy.reporting.localization.Localization(__file__, 26, 27), len_30867, *[exe_file_30868], **kwargs_30869)
        
        int_30871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 42), 'int')
        # Processing the call keyword arguments (line 26)
        kwargs_30872 = {}
        # Getting the type of 'self' (line 26)
        self_30865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self', False)
        # Obtaining the member 'assertGreater' of a type (line 26)
        assertGreater_30866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_30865, 'assertGreater')
        # Calling assertGreater(args, kwargs) (line 26)
        assertGreater_call_result_30873 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), assertGreater_30866, *[len_call_result_30870, int_30871], **kwargs_30872)
        
        
        # ################# End of 'test_get_exe_bytes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_get_exe_bytes' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_30874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30874)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_get_exe_bytes'
        return stypy_return_type_30874


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 0, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildWinInstTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BuildWinInstTestCase' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'BuildWinInstTestCase', BuildWinInstTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 28, 0, False)
    
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

    
    # Call to makeSuite(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'BuildWinInstTestCase' (line 29)
    BuildWinInstTestCase_30877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 30), 'BuildWinInstTestCase', False)
    # Processing the call keyword arguments (line 29)
    kwargs_30878 = {}
    # Getting the type of 'unittest' (line 29)
    unittest_30875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 29)
    makeSuite_30876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 11), unittest_30875, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 29)
    makeSuite_call_result_30879 = invoke(stypy.reporting.localization.Localization(__file__, 29, 11), makeSuite_30876, *[BuildWinInstTestCase_30877], **kwargs_30878)
    
    # Assigning a type to the variable 'stypy_return_type' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type', makeSuite_call_result_30879)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_30880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30880)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_30880

# Assigning a type to the variable 'test_suite' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to test_suite(...): (line 32)
    # Processing the call keyword arguments (line 32)
    kwargs_30883 = {}
    # Getting the type of 'test_suite' (line 32)
    test_suite_30882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 32)
    test_suite_call_result_30884 = invoke(stypy.reporting.localization.Localization(__file__, 32, 17), test_suite_30882, *[], **kwargs_30883)
    
    # Processing the call keyword arguments (line 32)
    kwargs_30885 = {}
    # Getting the type of 'run_unittest' (line 32)
    run_unittest_30881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 32)
    run_unittest_call_result_30886 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), run_unittest_30881, *[test_suite_call_result_30884], **kwargs_30885)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
