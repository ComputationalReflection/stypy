
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.bdist.'''
2: import os
3: import unittest
4: 
5: from test.test_support import run_unittest
6: 
7: from distutils.command.bdist import bdist
8: from distutils.tests import support
9: 
10: 
11: class BuildTestCase(support.TempdirManager,
12:                     unittest.TestCase):
13: 
14:     def test_formats(self):
15:         # let's create a command and make sure
16:         # we can set the format
17:         dist = self.create_dist()[1]
18:         cmd = bdist(dist)
19:         cmd.formats = ['msi']
20:         cmd.ensure_finalized()
21:         self.assertEqual(cmd.formats, ['msi'])
22: 
23:         # what formats does bdist offer?
24:         formats = ['bztar', 'gztar', 'msi', 'rpm', 'tar',
25:                    'wininst', 'zip', 'ztar']
26:         found = sorted(cmd.format_command)
27:         self.assertEqual(found, formats)
28: 
29:     def test_skip_build(self):
30:         # bug #10946: bdist --skip-build should trickle down to subcommands
31:         dist = self.create_dist()[1]
32:         cmd = bdist(dist)
33:         cmd.skip_build = 1
34:         cmd.ensure_finalized()
35:         dist.command_obj['bdist'] = cmd
36: 
37:         names = ['bdist_dumb', 'bdist_wininst']
38:         # bdist_rpm does not support --skip-build
39:         if os.name == 'nt':
40:             names.append('bdist_msi')
41: 
42:         for name in names:
43:             subcmd = cmd.get_finalized_command(name)
44:             self.assertTrue(subcmd.skip_build,
45:                             '%s should take --skip-build from bdist' % name)
46: 
47: 
48: def test_suite():
49:     return unittest.makeSuite(BuildTestCase)
50: 
51: if __name__ == '__main__':
52:     run_unittest(test_suite())
53: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_29943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.bdist.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import os' statement (line 2)
import os

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import unittest' statement (line 3)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from test.test_support import run_unittest' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_29944 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support')

if (type(import_29944) is not StypyTypeError):

    if (import_29944 != 'pyd_module'):
        __import__(import_29944)
        sys_modules_29945 = sys.modules[import_29944]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', sys_modules_29945.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_29945, sys_modules_29945.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', import_29944)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.command.bdist import bdist' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_29946 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.bdist')

if (type(import_29946) is not StypyTypeError):

    if (import_29946 != 'pyd_module'):
        __import__(import_29946)
        sys_modules_29947 = sys.modules[import_29946]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.bdist', sys_modules_29947.module_type_store, module_type_store, ['bdist'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_29947, sys_modules_29947.module_type_store, module_type_store)
    else:
        from distutils.command.bdist import bdist

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.bdist', None, module_type_store, ['bdist'], [bdist])

else:
    # Assigning a type to the variable 'distutils.command.bdist' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.bdist', import_29946)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.tests import support' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_29948 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests')

if (type(import_29948) is not StypyTypeError):

    if (import_29948 != 'pyd_module'):
        __import__(import_29948)
        sys_modules_29949 = sys.modules[import_29948]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', sys_modules_29949.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_29949, sys_modules_29949.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', import_29948)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'BuildTestCase' class
# Getting the type of 'support' (line 11)
support_29950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 20), 'support')
# Obtaining the member 'TempdirManager' of a type (line 11)
TempdirManager_29951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 20), support_29950, 'TempdirManager')
# Getting the type of 'unittest' (line 12)
unittest_29952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'unittest')
# Obtaining the member 'TestCase' of a type (line 12)
TestCase_29953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 20), unittest_29952, 'TestCase')

class BuildTestCase(TempdirManager_29951, TestCase_29953, ):

    @norecursion
    def test_formats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_formats'
        module_type_store = module_type_store.open_function_context('test_formats', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildTestCase.test_formats.__dict__.__setitem__('stypy_localization', localization)
        BuildTestCase.test_formats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildTestCase.test_formats.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildTestCase.test_formats.__dict__.__setitem__('stypy_function_name', 'BuildTestCase.test_formats')
        BuildTestCase.test_formats.__dict__.__setitem__('stypy_param_names_list', [])
        BuildTestCase.test_formats.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildTestCase.test_formats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildTestCase.test_formats.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildTestCase.test_formats.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildTestCase.test_formats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildTestCase.test_formats.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildTestCase.test_formats', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_formats', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_formats(...)' code ##################

        
        # Assigning a Subscript to a Name (line 17):
        
        # Obtaining the type of the subscript
        int_29954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 34), 'int')
        
        # Call to create_dist(...): (line 17)
        # Processing the call keyword arguments (line 17)
        kwargs_29957 = {}
        # Getting the type of 'self' (line 17)
        self_29955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 17)
        create_dist_29956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 15), self_29955, 'create_dist')
        # Calling create_dist(args, kwargs) (line 17)
        create_dist_call_result_29958 = invoke(stypy.reporting.localization.Localization(__file__, 17, 15), create_dist_29956, *[], **kwargs_29957)
        
        # Obtaining the member '__getitem__' of a type (line 17)
        getitem___29959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 15), create_dist_call_result_29958, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 17)
        subscript_call_result_29960 = invoke(stypy.reporting.localization.Localization(__file__, 17, 15), getitem___29959, int_29954)
        
        # Assigning a type to the variable 'dist' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'dist', subscript_call_result_29960)
        
        # Assigning a Call to a Name (line 18):
        
        # Call to bdist(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'dist' (line 18)
        dist_29962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'dist', False)
        # Processing the call keyword arguments (line 18)
        kwargs_29963 = {}
        # Getting the type of 'bdist' (line 18)
        bdist_29961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'bdist', False)
        # Calling bdist(args, kwargs) (line 18)
        bdist_call_result_29964 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), bdist_29961, *[dist_29962], **kwargs_29963)
        
        # Assigning a type to the variable 'cmd' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'cmd', bdist_call_result_29964)
        
        # Assigning a List to a Attribute (line 19):
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_29965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        # Adding element type (line 19)
        str_29966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'str', 'msi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 22), list_29965, str_29966)
        
        # Getting the type of 'cmd' (line 19)
        cmd_29967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'cmd')
        # Setting the type of the member 'formats' of a type (line 19)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), cmd_29967, 'formats', list_29965)
        
        # Call to ensure_finalized(...): (line 20)
        # Processing the call keyword arguments (line 20)
        kwargs_29970 = {}
        # Getting the type of 'cmd' (line 20)
        cmd_29968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 20)
        ensure_finalized_29969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), cmd_29968, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 20)
        ensure_finalized_call_result_29971 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), ensure_finalized_29969, *[], **kwargs_29970)
        
        
        # Call to assertEqual(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'cmd' (line 21)
        cmd_29974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 25), 'cmd', False)
        # Obtaining the member 'formats' of a type (line 21)
        formats_29975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 25), cmd_29974, 'formats')
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_29976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        str_29977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 39), 'str', 'msi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 38), list_29976, str_29977)
        
        # Processing the call keyword arguments (line 21)
        kwargs_29978 = {}
        # Getting the type of 'self' (line 21)
        self_29972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 21)
        assertEqual_29973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_29972, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 21)
        assertEqual_call_result_29979 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assertEqual_29973, *[formats_29975, list_29976], **kwargs_29978)
        
        
        # Assigning a List to a Name (line 24):
        
        # Obtaining an instance of the builtin type 'list' (line 24)
        list_29980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 24)
        # Adding element type (line 24)
        str_29981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'str', 'bztar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 18), list_29980, str_29981)
        # Adding element type (line 24)
        str_29982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 28), 'str', 'gztar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 18), list_29980, str_29982)
        # Adding element type (line 24)
        str_29983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 37), 'str', 'msi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 18), list_29980, str_29983)
        # Adding element type (line 24)
        str_29984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 44), 'str', 'rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 18), list_29980, str_29984)
        # Adding element type (line 24)
        str_29985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 51), 'str', 'tar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 18), list_29980, str_29985)
        # Adding element type (line 24)
        str_29986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'str', 'wininst')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 18), list_29980, str_29986)
        # Adding element type (line 24)
        str_29987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 30), 'str', 'zip')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 18), list_29980, str_29987)
        # Adding element type (line 24)
        str_29988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 37), 'str', 'ztar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 18), list_29980, str_29988)
        
        # Assigning a type to the variable 'formats' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'formats', list_29980)
        
        # Assigning a Call to a Name (line 26):
        
        # Call to sorted(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'cmd' (line 26)
        cmd_29990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'cmd', False)
        # Obtaining the member 'format_command' of a type (line 26)
        format_command_29991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 23), cmd_29990, 'format_command')
        # Processing the call keyword arguments (line 26)
        kwargs_29992 = {}
        # Getting the type of 'sorted' (line 26)
        sorted_29989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'sorted', False)
        # Calling sorted(args, kwargs) (line 26)
        sorted_call_result_29993 = invoke(stypy.reporting.localization.Localization(__file__, 26, 16), sorted_29989, *[format_command_29991], **kwargs_29992)
        
        # Assigning a type to the variable 'found' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'found', sorted_call_result_29993)
        
        # Call to assertEqual(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'found' (line 27)
        found_29996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'found', False)
        # Getting the type of 'formats' (line 27)
        formats_29997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 32), 'formats', False)
        # Processing the call keyword arguments (line 27)
        kwargs_29998 = {}
        # Getting the type of 'self' (line 27)
        self_29994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 27)
        assertEqual_29995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_29994, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 27)
        assertEqual_call_result_29999 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assertEqual_29995, *[found_29996, formats_29997], **kwargs_29998)
        
        
        # ################# End of 'test_formats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_formats' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_30000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30000)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_formats'
        return stypy_return_type_30000


    @norecursion
    def test_skip_build(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_skip_build'
        module_type_store = module_type_store.open_function_context('test_skip_build', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildTestCase.test_skip_build.__dict__.__setitem__('stypy_localization', localization)
        BuildTestCase.test_skip_build.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildTestCase.test_skip_build.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildTestCase.test_skip_build.__dict__.__setitem__('stypy_function_name', 'BuildTestCase.test_skip_build')
        BuildTestCase.test_skip_build.__dict__.__setitem__('stypy_param_names_list', [])
        BuildTestCase.test_skip_build.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildTestCase.test_skip_build.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildTestCase.test_skip_build.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildTestCase.test_skip_build.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildTestCase.test_skip_build.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildTestCase.test_skip_build.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildTestCase.test_skip_build', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_skip_build', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_skip_build(...)' code ##################

        
        # Assigning a Subscript to a Name (line 31):
        
        # Obtaining the type of the subscript
        int_30001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 34), 'int')
        
        # Call to create_dist(...): (line 31)
        # Processing the call keyword arguments (line 31)
        kwargs_30004 = {}
        # Getting the type of 'self' (line 31)
        self_30002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 31)
        create_dist_30003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), self_30002, 'create_dist')
        # Calling create_dist(args, kwargs) (line 31)
        create_dist_call_result_30005 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), create_dist_30003, *[], **kwargs_30004)
        
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___30006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), create_dist_call_result_30005, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_30007 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), getitem___30006, int_30001)
        
        # Assigning a type to the variable 'dist' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'dist', subscript_call_result_30007)
        
        # Assigning a Call to a Name (line 32):
        
        # Call to bdist(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'dist' (line 32)
        dist_30009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'dist', False)
        # Processing the call keyword arguments (line 32)
        kwargs_30010 = {}
        # Getting the type of 'bdist' (line 32)
        bdist_30008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'bdist', False)
        # Calling bdist(args, kwargs) (line 32)
        bdist_call_result_30011 = invoke(stypy.reporting.localization.Localization(__file__, 32, 14), bdist_30008, *[dist_30009], **kwargs_30010)
        
        # Assigning a type to the variable 'cmd' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'cmd', bdist_call_result_30011)
        
        # Assigning a Num to a Attribute (line 33):
        int_30012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'int')
        # Getting the type of 'cmd' (line 33)
        cmd_30013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'cmd')
        # Setting the type of the member 'skip_build' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), cmd_30013, 'skip_build', int_30012)
        
        # Call to ensure_finalized(...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_30016 = {}
        # Getting the type of 'cmd' (line 34)
        cmd_30014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 34)
        ensure_finalized_30015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), cmd_30014, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 34)
        ensure_finalized_call_result_30017 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), ensure_finalized_30015, *[], **kwargs_30016)
        
        
        # Assigning a Name to a Subscript (line 35):
        # Getting the type of 'cmd' (line 35)
        cmd_30018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 36), 'cmd')
        # Getting the type of 'dist' (line 35)
        dist_30019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'dist')
        # Obtaining the member 'command_obj' of a type (line 35)
        command_obj_30020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), dist_30019, 'command_obj')
        str_30021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 25), 'str', 'bdist')
        # Storing an element on a container (line 35)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 8), command_obj_30020, (str_30021, cmd_30018))
        
        # Assigning a List to a Name (line 37):
        
        # Obtaining an instance of the builtin type 'list' (line 37)
        list_30022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 37)
        # Adding element type (line 37)
        str_30023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 17), 'str', 'bdist_dumb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_30022, str_30023)
        # Adding element type (line 37)
        str_30024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 31), 'str', 'bdist_wininst')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_30022, str_30024)
        
        # Assigning a type to the variable 'names' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'names', list_30022)
        
        
        # Getting the type of 'os' (line 39)
        os_30025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'os')
        # Obtaining the member 'name' of a type (line 39)
        name_30026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 11), os_30025, 'name')
        str_30027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 22), 'str', 'nt')
        # Applying the binary operator '==' (line 39)
        result_eq_30028 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 11), '==', name_30026, str_30027)
        
        # Testing the type of an if condition (line 39)
        if_condition_30029 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 8), result_eq_30028)
        # Assigning a type to the variable 'if_condition_30029' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'if_condition_30029', if_condition_30029)
        # SSA begins for if statement (line 39)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 40)
        # Processing the call arguments (line 40)
        str_30032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'str', 'bdist_msi')
        # Processing the call keyword arguments (line 40)
        kwargs_30033 = {}
        # Getting the type of 'names' (line 40)
        names_30030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'names', False)
        # Obtaining the member 'append' of a type (line 40)
        append_30031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), names_30030, 'append')
        # Calling append(args, kwargs) (line 40)
        append_call_result_30034 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), append_30031, *[str_30032], **kwargs_30033)
        
        # SSA join for if statement (line 39)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'names' (line 42)
        names_30035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'names')
        # Testing the type of a for loop iterable (line 42)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 42, 8), names_30035)
        # Getting the type of the for loop variable (line 42)
        for_loop_var_30036 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 42, 8), names_30035)
        # Assigning a type to the variable 'name' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'name', for_loop_var_30036)
        # SSA begins for a for statement (line 42)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 43):
        
        # Call to get_finalized_command(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'name' (line 43)
        name_30039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 47), 'name', False)
        # Processing the call keyword arguments (line 43)
        kwargs_30040 = {}
        # Getting the type of 'cmd' (line 43)
        cmd_30037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'cmd', False)
        # Obtaining the member 'get_finalized_command' of a type (line 43)
        get_finalized_command_30038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 21), cmd_30037, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 43)
        get_finalized_command_call_result_30041 = invoke(stypy.reporting.localization.Localization(__file__, 43, 21), get_finalized_command_30038, *[name_30039], **kwargs_30040)
        
        # Assigning a type to the variable 'subcmd' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'subcmd', get_finalized_command_call_result_30041)
        
        # Call to assertTrue(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'subcmd' (line 44)
        subcmd_30044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 28), 'subcmd', False)
        # Obtaining the member 'skip_build' of a type (line 44)
        skip_build_30045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 28), subcmd_30044, 'skip_build')
        str_30046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 28), 'str', '%s should take --skip-build from bdist')
        # Getting the type of 'name' (line 45)
        name_30047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 71), 'name', False)
        # Applying the binary operator '%' (line 45)
        result_mod_30048 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 28), '%', str_30046, name_30047)
        
        # Processing the call keyword arguments (line 44)
        kwargs_30049 = {}
        # Getting the type of 'self' (line 44)
        self_30042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 44)
        assertTrue_30043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), self_30042, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 44)
        assertTrue_call_result_30050 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), assertTrue_30043, *[skip_build_30045, result_mod_30048], **kwargs_30049)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_skip_build(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_skip_build' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_30051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30051)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_skip_build'
        return stypy_return_type_30051


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BuildTestCase' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'BuildTestCase', BuildTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 48, 0, False)
    
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

    
    # Call to makeSuite(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'BuildTestCase' (line 49)
    BuildTestCase_30054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'BuildTestCase', False)
    # Processing the call keyword arguments (line 49)
    kwargs_30055 = {}
    # Getting the type of 'unittest' (line 49)
    unittest_30052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 49)
    makeSuite_30053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 11), unittest_30052, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 49)
    makeSuite_call_result_30056 = invoke(stypy.reporting.localization.Localization(__file__, 49, 11), makeSuite_30053, *[BuildTestCase_30054], **kwargs_30055)
    
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type', makeSuite_call_result_30056)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_30057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30057)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_30057

# Assigning a type to the variable 'test_suite' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Call to test_suite(...): (line 52)
    # Processing the call keyword arguments (line 52)
    kwargs_30060 = {}
    # Getting the type of 'test_suite' (line 52)
    test_suite_30059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 52)
    test_suite_call_result_30061 = invoke(stypy.reporting.localization.Localization(__file__, 52, 17), test_suite_30059, *[], **kwargs_30060)
    
    # Processing the call keyword arguments (line 52)
    kwargs_30062 = {}
    # Getting the type of 'run_unittest' (line 52)
    run_unittest_30058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 52)
    run_unittest_call_result_30063 = invoke(stypy.reporting.localization.Localization(__file__, 52, 4), run_unittest_30058, *[test_suite_call_result_30061], **kwargs_30062)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
