
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.install_headers.'''
2: import sys
3: import os
4: import unittest
5: import getpass
6: 
7: from distutils.command.install_headers import install_headers
8: from distutils.tests import support
9: from test.test_support import run_unittest
10: 
11: class InstallHeadersTestCase(support.TempdirManager,
12:                              support.LoggingSilencer,
13:                              support.EnvironGuard,
14:                              unittest.TestCase):
15: 
16:     def test_simple_run(self):
17:         # we have two headers
18:         header_list = self.mkdtemp()
19:         header1 = os.path.join(header_list, 'header1')
20:         header2 = os.path.join(header_list, 'header2')
21:         self.write_file(header1)
22:         self.write_file(header2)
23:         headers = [header1, header2]
24: 
25:         pkg_dir, dist = self.create_dist(headers=headers)
26:         cmd = install_headers(dist)
27:         self.assertEqual(cmd.get_inputs(), headers)
28: 
29:         # let's run the command
30:         cmd.install_dir = os.path.join(pkg_dir, 'inst')
31:         cmd.ensure_finalized()
32:         cmd.run()
33: 
34:         # let's check the results
35:         self.assertEqual(len(cmd.get_outputs()), 2)
36: 
37: def test_suite():
38:     return unittest.makeSuite(InstallHeadersTestCase)
39: 
40: if __name__ == "__main__":
41:     run_unittest(test_suite())
42: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_40821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.install_headers.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import unittest' statement (line 4)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import getpass' statement (line 5)
import getpass

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'getpass', getpass, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.command.install_headers import install_headers' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_40822 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.install_headers')

if (type(import_40822) is not StypyTypeError):

    if (import_40822 != 'pyd_module'):
        __import__(import_40822)
        sys_modules_40823 = sys.modules[import_40822]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.install_headers', sys_modules_40823.module_type_store, module_type_store, ['install_headers'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_40823, sys_modules_40823.module_type_store, module_type_store)
    else:
        from distutils.command.install_headers import install_headers

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.install_headers', None, module_type_store, ['install_headers'], [install_headers])

else:
    # Assigning a type to the variable 'distutils.command.install_headers' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.install_headers', import_40822)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.tests import support' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_40824 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests')

if (type(import_40824) is not StypyTypeError):

    if (import_40824 != 'pyd_module'):
        __import__(import_40824)
        sys_modules_40825 = sys.modules[import_40824]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', sys_modules_40825.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_40825, sys_modules_40825.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', import_40824)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from test.test_support import run_unittest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_40826 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support')

if (type(import_40826) is not StypyTypeError):

    if (import_40826 != 'pyd_module'):
        __import__(import_40826)
        sys_modules_40827 = sys.modules[import_40826]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', sys_modules_40827.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_40827, sys_modules_40827.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', import_40826)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'InstallHeadersTestCase' class
# Getting the type of 'support' (line 11)
support_40828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 29), 'support')
# Obtaining the member 'TempdirManager' of a type (line 11)
TempdirManager_40829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 29), support_40828, 'TempdirManager')
# Getting the type of 'support' (line 12)
support_40830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 29), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 12)
LoggingSilencer_40831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 29), support_40830, 'LoggingSilencer')
# Getting the type of 'support' (line 13)
support_40832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), 'support')
# Obtaining the member 'EnvironGuard' of a type (line 13)
EnvironGuard_40833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 29), support_40832, 'EnvironGuard')
# Getting the type of 'unittest' (line 14)
unittest_40834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 29), 'unittest')
# Obtaining the member 'TestCase' of a type (line 14)
TestCase_40835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 29), unittest_40834, 'TestCase')

class InstallHeadersTestCase(TempdirManager_40829, LoggingSilencer_40831, EnvironGuard_40833, TestCase_40835, ):

    @norecursion
    def test_simple_run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_run'
        module_type_store = module_type_store.open_function_context('test_simple_run', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallHeadersTestCase.test_simple_run.__dict__.__setitem__('stypy_localization', localization)
        InstallHeadersTestCase.test_simple_run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallHeadersTestCase.test_simple_run.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallHeadersTestCase.test_simple_run.__dict__.__setitem__('stypy_function_name', 'InstallHeadersTestCase.test_simple_run')
        InstallHeadersTestCase.test_simple_run.__dict__.__setitem__('stypy_param_names_list', [])
        InstallHeadersTestCase.test_simple_run.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallHeadersTestCase.test_simple_run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallHeadersTestCase.test_simple_run.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallHeadersTestCase.test_simple_run.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallHeadersTestCase.test_simple_run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallHeadersTestCase.test_simple_run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallHeadersTestCase.test_simple_run', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_run', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_run(...)' code ##################

        
        # Assigning a Call to a Name (line 18):
        
        # Assigning a Call to a Name (line 18):
        
        # Call to mkdtemp(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_40838 = {}
        # Getting the type of 'self' (line 18)
        self_40836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 18)
        mkdtemp_40837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 22), self_40836, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 18)
        mkdtemp_call_result_40839 = invoke(stypy.reporting.localization.Localization(__file__, 18, 22), mkdtemp_40837, *[], **kwargs_40838)
        
        # Assigning a type to the variable 'header_list' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'header_list', mkdtemp_call_result_40839)
        
        # Assigning a Call to a Name (line 19):
        
        # Assigning a Call to a Name (line 19):
        
        # Call to join(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'header_list' (line 19)
        header_list_40843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), 'header_list', False)
        str_40844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 44), 'str', 'header1')
        # Processing the call keyword arguments (line 19)
        kwargs_40845 = {}
        # Getting the type of 'os' (line 19)
        os_40840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 19)
        path_40841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 18), os_40840, 'path')
        # Obtaining the member 'join' of a type (line 19)
        join_40842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 18), path_40841, 'join')
        # Calling join(args, kwargs) (line 19)
        join_call_result_40846 = invoke(stypy.reporting.localization.Localization(__file__, 19, 18), join_40842, *[header_list_40843, str_40844], **kwargs_40845)
        
        # Assigning a type to the variable 'header1' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'header1', join_call_result_40846)
        
        # Assigning a Call to a Name (line 20):
        
        # Assigning a Call to a Name (line 20):
        
        # Call to join(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'header_list' (line 20)
        header_list_40850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 31), 'header_list', False)
        str_40851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 44), 'str', 'header2')
        # Processing the call keyword arguments (line 20)
        kwargs_40852 = {}
        # Getting the type of 'os' (line 20)
        os_40847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 20)
        path_40848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 18), os_40847, 'path')
        # Obtaining the member 'join' of a type (line 20)
        join_40849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 18), path_40848, 'join')
        # Calling join(args, kwargs) (line 20)
        join_call_result_40853 = invoke(stypy.reporting.localization.Localization(__file__, 20, 18), join_40849, *[header_list_40850, str_40851], **kwargs_40852)
        
        # Assigning a type to the variable 'header2' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'header2', join_call_result_40853)
        
        # Call to write_file(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'header1' (line 21)
        header1_40856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'header1', False)
        # Processing the call keyword arguments (line 21)
        kwargs_40857 = {}
        # Getting the type of 'self' (line 21)
        self_40854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 21)
        write_file_40855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_40854, 'write_file')
        # Calling write_file(args, kwargs) (line 21)
        write_file_call_result_40858 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), write_file_40855, *[header1_40856], **kwargs_40857)
        
        
        # Call to write_file(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'header2' (line 22)
        header2_40861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 24), 'header2', False)
        # Processing the call keyword arguments (line 22)
        kwargs_40862 = {}
        # Getting the type of 'self' (line 22)
        self_40859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 22)
        write_file_40860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_40859, 'write_file')
        # Calling write_file(args, kwargs) (line 22)
        write_file_call_result_40863 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), write_file_40860, *[header2_40861], **kwargs_40862)
        
        
        # Assigning a List to a Name (line 23):
        
        # Assigning a List to a Name (line 23):
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_40864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        # Adding element type (line 23)
        # Getting the type of 'header1' (line 23)
        header1_40865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'header1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 18), list_40864, header1_40865)
        # Adding element type (line 23)
        # Getting the type of 'header2' (line 23)
        header2_40866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 28), 'header2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 18), list_40864, header2_40866)
        
        # Assigning a type to the variable 'headers' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'headers', list_40864)
        
        # Assigning a Call to a Tuple (line 25):
        
        # Assigning a Subscript to a Name (line 25):
        
        # Obtaining the type of the subscript
        int_40867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'int')
        
        # Call to create_dist(...): (line 25)
        # Processing the call keyword arguments (line 25)
        # Getting the type of 'headers' (line 25)
        headers_40870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 49), 'headers', False)
        keyword_40871 = headers_40870
        kwargs_40872 = {'headers': keyword_40871}
        # Getting the type of 'self' (line 25)
        self_40868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 25)
        create_dist_40869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), self_40868, 'create_dist')
        # Calling create_dist(args, kwargs) (line 25)
        create_dist_call_result_40873 = invoke(stypy.reporting.localization.Localization(__file__, 25, 24), create_dist_40869, *[], **kwargs_40872)
        
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___40874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), create_dist_call_result_40873, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_40875 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), getitem___40874, int_40867)
        
        # Assigning a type to the variable 'tuple_var_assignment_40819' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'tuple_var_assignment_40819', subscript_call_result_40875)
        
        # Assigning a Subscript to a Name (line 25):
        
        # Obtaining the type of the subscript
        int_40876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'int')
        
        # Call to create_dist(...): (line 25)
        # Processing the call keyword arguments (line 25)
        # Getting the type of 'headers' (line 25)
        headers_40879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 49), 'headers', False)
        keyword_40880 = headers_40879
        kwargs_40881 = {'headers': keyword_40880}
        # Getting the type of 'self' (line 25)
        self_40877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 25)
        create_dist_40878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), self_40877, 'create_dist')
        # Calling create_dist(args, kwargs) (line 25)
        create_dist_call_result_40882 = invoke(stypy.reporting.localization.Localization(__file__, 25, 24), create_dist_40878, *[], **kwargs_40881)
        
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___40883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), create_dist_call_result_40882, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_40884 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), getitem___40883, int_40876)
        
        # Assigning a type to the variable 'tuple_var_assignment_40820' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'tuple_var_assignment_40820', subscript_call_result_40884)
        
        # Assigning a Name to a Name (line 25):
        # Getting the type of 'tuple_var_assignment_40819' (line 25)
        tuple_var_assignment_40819_40885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'tuple_var_assignment_40819')
        # Assigning a type to the variable 'pkg_dir' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'pkg_dir', tuple_var_assignment_40819_40885)
        
        # Assigning a Name to a Name (line 25):
        # Getting the type of 'tuple_var_assignment_40820' (line 25)
        tuple_var_assignment_40820_40886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'tuple_var_assignment_40820')
        # Assigning a type to the variable 'dist' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'dist', tuple_var_assignment_40820_40886)
        
        # Assigning a Call to a Name (line 26):
        
        # Assigning a Call to a Name (line 26):
        
        # Call to install_headers(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'dist' (line 26)
        dist_40888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 30), 'dist', False)
        # Processing the call keyword arguments (line 26)
        kwargs_40889 = {}
        # Getting the type of 'install_headers' (line 26)
        install_headers_40887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'install_headers', False)
        # Calling install_headers(args, kwargs) (line 26)
        install_headers_call_result_40890 = invoke(stypy.reporting.localization.Localization(__file__, 26, 14), install_headers_40887, *[dist_40888], **kwargs_40889)
        
        # Assigning a type to the variable 'cmd' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'cmd', install_headers_call_result_40890)
        
        # Call to assertEqual(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Call to get_inputs(...): (line 27)
        # Processing the call keyword arguments (line 27)
        kwargs_40895 = {}
        # Getting the type of 'cmd' (line 27)
        cmd_40893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'cmd', False)
        # Obtaining the member 'get_inputs' of a type (line 27)
        get_inputs_40894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 25), cmd_40893, 'get_inputs')
        # Calling get_inputs(args, kwargs) (line 27)
        get_inputs_call_result_40896 = invoke(stypy.reporting.localization.Localization(__file__, 27, 25), get_inputs_40894, *[], **kwargs_40895)
        
        # Getting the type of 'headers' (line 27)
        headers_40897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 43), 'headers', False)
        # Processing the call keyword arguments (line 27)
        kwargs_40898 = {}
        # Getting the type of 'self' (line 27)
        self_40891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 27)
        assertEqual_40892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_40891, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 27)
        assertEqual_call_result_40899 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assertEqual_40892, *[get_inputs_call_result_40896, headers_40897], **kwargs_40898)
        
        
        # Assigning a Call to a Attribute (line 30):
        
        # Assigning a Call to a Attribute (line 30):
        
        # Call to join(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'pkg_dir' (line 30)
        pkg_dir_40903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 39), 'pkg_dir', False)
        str_40904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 48), 'str', 'inst')
        # Processing the call keyword arguments (line 30)
        kwargs_40905 = {}
        # Getting the type of 'os' (line 30)
        os_40900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 30)
        path_40901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 26), os_40900, 'path')
        # Obtaining the member 'join' of a type (line 30)
        join_40902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 26), path_40901, 'join')
        # Calling join(args, kwargs) (line 30)
        join_call_result_40906 = invoke(stypy.reporting.localization.Localization(__file__, 30, 26), join_40902, *[pkg_dir_40903, str_40904], **kwargs_40905)
        
        # Getting the type of 'cmd' (line 30)
        cmd_40907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'cmd')
        # Setting the type of the member 'install_dir' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), cmd_40907, 'install_dir', join_call_result_40906)
        
        # Call to ensure_finalized(...): (line 31)
        # Processing the call keyword arguments (line 31)
        kwargs_40910 = {}
        # Getting the type of 'cmd' (line 31)
        cmd_40908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 31)
        ensure_finalized_40909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), cmd_40908, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 31)
        ensure_finalized_call_result_40911 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), ensure_finalized_40909, *[], **kwargs_40910)
        
        
        # Call to run(...): (line 32)
        # Processing the call keyword arguments (line 32)
        kwargs_40914 = {}
        # Getting the type of 'cmd' (line 32)
        cmd_40912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 32)
        run_40913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), cmd_40912, 'run')
        # Calling run(args, kwargs) (line 32)
        run_call_result_40915 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), run_40913, *[], **kwargs_40914)
        
        
        # Call to assertEqual(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Call to len(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Call to get_outputs(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_40921 = {}
        # Getting the type of 'cmd' (line 35)
        cmd_40919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 29), 'cmd', False)
        # Obtaining the member 'get_outputs' of a type (line 35)
        get_outputs_40920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 29), cmd_40919, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 35)
        get_outputs_call_result_40922 = invoke(stypy.reporting.localization.Localization(__file__, 35, 29), get_outputs_40920, *[], **kwargs_40921)
        
        # Processing the call keyword arguments (line 35)
        kwargs_40923 = {}
        # Getting the type of 'len' (line 35)
        len_40918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 'len', False)
        # Calling len(args, kwargs) (line 35)
        len_call_result_40924 = invoke(stypy.reporting.localization.Localization(__file__, 35, 25), len_40918, *[get_outputs_call_result_40922], **kwargs_40923)
        
        int_40925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 49), 'int')
        # Processing the call keyword arguments (line 35)
        kwargs_40926 = {}
        # Getting the type of 'self' (line 35)
        self_40916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 35)
        assertEqual_40917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_40916, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 35)
        assertEqual_call_result_40927 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), assertEqual_40917, *[len_call_result_40924, int_40925], **kwargs_40926)
        
        
        # ################# End of 'test_simple_run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_run' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_40928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40928)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_run'
        return stypy_return_type_40928


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallHeadersTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'InstallHeadersTestCase' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'InstallHeadersTestCase', InstallHeadersTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 37, 0, False)
    
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

    
    # Call to makeSuite(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'InstallHeadersTestCase' (line 38)
    InstallHeadersTestCase_40931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 30), 'InstallHeadersTestCase', False)
    # Processing the call keyword arguments (line 38)
    kwargs_40932 = {}
    # Getting the type of 'unittest' (line 38)
    unittest_40929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 38)
    makeSuite_40930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 11), unittest_40929, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 38)
    makeSuite_call_result_40933 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), makeSuite_40930, *[InstallHeadersTestCase_40931], **kwargs_40932)
    
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type', makeSuite_call_result_40933)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_40934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_40934)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_40934

# Assigning a type to the variable 'test_suite' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Call to test_suite(...): (line 41)
    # Processing the call keyword arguments (line 41)
    kwargs_40937 = {}
    # Getting the type of 'test_suite' (line 41)
    test_suite_40936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 41)
    test_suite_call_result_40938 = invoke(stypy.reporting.localization.Localization(__file__, 41, 17), test_suite_40936, *[], **kwargs_40937)
    
    # Processing the call keyword arguments (line 41)
    kwargs_40939 = {}
    # Getting the type of 'run_unittest' (line 41)
    run_unittest_40935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 41)
    run_unittest_call_result_40940 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), run_unittest_40935, *[test_suite_call_result_40938], **kwargs_40939)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
