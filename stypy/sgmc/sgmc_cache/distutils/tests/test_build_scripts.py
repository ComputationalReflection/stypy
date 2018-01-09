
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.build_scripts.'''
2: 
3: import os
4: import unittest
5: 
6: from distutils.command.build_scripts import build_scripts
7: from distutils.core import Distribution
8: import sysconfig
9: 
10: from distutils.tests import support
11: from test.test_support import run_unittest
12: 
13: 
14: class BuildScriptsTestCase(support.TempdirManager,
15:                            support.LoggingSilencer,
16:                            unittest.TestCase):
17: 
18:     def test_default_settings(self):
19:         cmd = self.get_build_scripts_cmd("/foo/bar", [])
20:         self.assertFalse(cmd.force)
21:         self.assertIsNone(cmd.build_dir)
22: 
23:         cmd.finalize_options()
24: 
25:         self.assertTrue(cmd.force)
26:         self.assertEqual(cmd.build_dir, "/foo/bar")
27: 
28:     def test_build(self):
29:         source = self.mkdtemp()
30:         target = self.mkdtemp()
31:         expected = self.write_sample_scripts(source)
32: 
33:         cmd = self.get_build_scripts_cmd(target,
34:                                          [os.path.join(source, fn)
35:                                           for fn in expected])
36:         cmd.finalize_options()
37:         cmd.run()
38: 
39:         built = os.listdir(target)
40:         for name in expected:
41:             self.assertIn(name, built)
42: 
43:     def get_build_scripts_cmd(self, target, scripts):
44:         import sys
45:         dist = Distribution()
46:         dist.scripts = scripts
47:         dist.command_obj["build"] = support.DummyCommand(
48:             build_scripts=target,
49:             force=1,
50:             executable=sys.executable
51:             )
52:         return build_scripts(dist)
53: 
54:     def write_sample_scripts(self, dir):
55:         expected = []
56:         expected.append("script1.py")
57:         self.write_script(dir, "script1.py",
58:                           ("#! /usr/bin/env python2.3\n"
59:                            "# bogus script w/ Python sh-bang\n"
60:                            "pass\n"))
61:         expected.append("script2.py")
62:         self.write_script(dir, "script2.py",
63:                           ("#!/usr/bin/python\n"
64:                            "# bogus script w/ Python sh-bang\n"
65:                            "pass\n"))
66:         expected.append("shell.sh")
67:         self.write_script(dir, "shell.sh",
68:                           ("#!/bin/sh\n"
69:                            "# bogus shell script w/ sh-bang\n"
70:                            "exit 0\n"))
71:         return expected
72: 
73:     def write_script(self, dir, name, text):
74:         f = open(os.path.join(dir, name), "w")
75:         try:
76:             f.write(text)
77:         finally:
78:             f.close()
79: 
80:     def test_version_int(self):
81:         source = self.mkdtemp()
82:         target = self.mkdtemp()
83:         expected = self.write_sample_scripts(source)
84: 
85: 
86:         cmd = self.get_build_scripts_cmd(target,
87:                                          [os.path.join(source, fn)
88:                                           for fn in expected])
89:         cmd.finalize_options()
90: 
91:         # http://bugs.python.org/issue4524
92:         #
93:         # On linux-g++-32 with command line `./configure --enable-ipv6
94:         # --with-suffix=3`, python is compiled okay but the build scripts
95:         # failed when writing the name of the executable
96:         old = sysconfig.get_config_vars().get('VERSION')
97:         sysconfig._CONFIG_VARS['VERSION'] = 4
98:         try:
99:             cmd.run()
100:         finally:
101:             if old is not None:
102:                 sysconfig._CONFIG_VARS['VERSION'] = old
103: 
104:         built = os.listdir(target)
105:         for name in expected:
106:             self.assertIn(name, built)
107: 
108: def test_suite():
109:     return unittest.makeSuite(BuildScriptsTestCase)
110: 
111: if __name__ == "__main__":
112:     run_unittest(test_suite())
113: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_33771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.build_scripts.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import unittest' statement (line 4)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.command.build_scripts import build_scripts' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_33772 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.build_scripts')

if (type(import_33772) is not StypyTypeError):

    if (import_33772 != 'pyd_module'):
        __import__(import_33772)
        sys_modules_33773 = sys.modules[import_33772]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.build_scripts', sys_modules_33773.module_type_store, module_type_store, ['build_scripts'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_33773, sys_modules_33773.module_type_store, module_type_store)
    else:
        from distutils.command.build_scripts import build_scripts

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.build_scripts', None, module_type_store, ['build_scripts'], [build_scripts])

else:
    # Assigning a type to the variable 'distutils.command.build_scripts' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.build_scripts', import_33772)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.core import Distribution' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_33774 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.core')

if (type(import_33774) is not StypyTypeError):

    if (import_33774 != 'pyd_module'):
        __import__(import_33774)
        sys_modules_33775 = sys.modules[import_33774]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.core', sys_modules_33775.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_33775, sys_modules_33775.module_type_store, module_type_store)
    else:
        from distutils.core import Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.core', None, module_type_store, ['Distribution'], [Distribution])

else:
    # Assigning a type to the variable 'distutils.core' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.core', import_33774)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import sysconfig' statement (line 8)
import sysconfig

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sysconfig', sysconfig, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.tests import support' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_33776 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests')

if (type(import_33776) is not StypyTypeError):

    if (import_33776 != 'pyd_module'):
        __import__(import_33776)
        sys_modules_33777 = sys.modules[import_33776]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests', sys_modules_33777.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_33777, sys_modules_33777.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests', import_33776)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from test.test_support import run_unittest' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_33778 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'test.test_support')

if (type(import_33778) is not StypyTypeError):

    if (import_33778 != 'pyd_module'):
        __import__(import_33778)
        sys_modules_33779 = sys.modules[import_33778]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'test.test_support', sys_modules_33779.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_33779, sys_modules_33779.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'test.test_support', import_33778)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'BuildScriptsTestCase' class
# Getting the type of 'support' (line 14)
support_33780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 27), 'support')
# Obtaining the member 'TempdirManager' of a type (line 14)
TempdirManager_33781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 27), support_33780, 'TempdirManager')
# Getting the type of 'support' (line 15)
support_33782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 27), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 15)
LoggingSilencer_33783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 27), support_33782, 'LoggingSilencer')
# Getting the type of 'unittest' (line 16)
unittest_33784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 27), 'unittest')
# Obtaining the member 'TestCase' of a type (line 16)
TestCase_33785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 27), unittest_33784, 'TestCase')

class BuildScriptsTestCase(TempdirManager_33781, LoggingSilencer_33783, TestCase_33785, ):

    @norecursion
    def test_default_settings(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_default_settings'
        module_type_store = module_type_store.open_function_context('test_default_settings', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_localization', localization)
        BuildScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_function_name', 'BuildScriptsTestCase.test_default_settings')
        BuildScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_param_names_list', [])
        BuildScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildScriptsTestCase.test_default_settings', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_default_settings', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_default_settings(...)' code ##################

        
        # Assigning a Call to a Name (line 19):
        
        # Call to get_build_scripts_cmd(...): (line 19)
        # Processing the call arguments (line 19)
        str_33788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 41), 'str', '/foo/bar')
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_33789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        
        # Processing the call keyword arguments (line 19)
        kwargs_33790 = {}
        # Getting the type of 'self' (line 19)
        self_33786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'self', False)
        # Obtaining the member 'get_build_scripts_cmd' of a type (line 19)
        get_build_scripts_cmd_33787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 14), self_33786, 'get_build_scripts_cmd')
        # Calling get_build_scripts_cmd(args, kwargs) (line 19)
        get_build_scripts_cmd_call_result_33791 = invoke(stypy.reporting.localization.Localization(__file__, 19, 14), get_build_scripts_cmd_33787, *[str_33788, list_33789], **kwargs_33790)
        
        # Assigning a type to the variable 'cmd' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'cmd', get_build_scripts_cmd_call_result_33791)
        
        # Call to assertFalse(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'cmd' (line 20)
        cmd_33794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 25), 'cmd', False)
        # Obtaining the member 'force' of a type (line 20)
        force_33795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 25), cmd_33794, 'force')
        # Processing the call keyword arguments (line 20)
        kwargs_33796 = {}
        # Getting the type of 'self' (line 20)
        self_33792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 20)
        assertFalse_33793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_33792, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 20)
        assertFalse_call_result_33797 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), assertFalse_33793, *[force_33795], **kwargs_33796)
        
        
        # Call to assertIsNone(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'cmd' (line 21)
        cmd_33800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 26), 'cmd', False)
        # Obtaining the member 'build_dir' of a type (line 21)
        build_dir_33801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 26), cmd_33800, 'build_dir')
        # Processing the call keyword arguments (line 21)
        kwargs_33802 = {}
        # Getting the type of 'self' (line 21)
        self_33798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self', False)
        # Obtaining the member 'assertIsNone' of a type (line 21)
        assertIsNone_33799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_33798, 'assertIsNone')
        # Calling assertIsNone(args, kwargs) (line 21)
        assertIsNone_call_result_33803 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assertIsNone_33799, *[build_dir_33801], **kwargs_33802)
        
        
        # Call to finalize_options(...): (line 23)
        # Processing the call keyword arguments (line 23)
        kwargs_33806 = {}
        # Getting the type of 'cmd' (line 23)
        cmd_33804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 23)
        finalize_options_33805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), cmd_33804, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 23)
        finalize_options_call_result_33807 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), finalize_options_33805, *[], **kwargs_33806)
        
        
        # Call to assertTrue(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'cmd' (line 25)
        cmd_33810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'cmd', False)
        # Obtaining the member 'force' of a type (line 25)
        force_33811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), cmd_33810, 'force')
        # Processing the call keyword arguments (line 25)
        kwargs_33812 = {}
        # Getting the type of 'self' (line 25)
        self_33808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 25)
        assertTrue_33809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_33808, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 25)
        assertTrue_call_result_33813 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), assertTrue_33809, *[force_33811], **kwargs_33812)
        
        
        # Call to assertEqual(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'cmd' (line 26)
        cmd_33816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'cmd', False)
        # Obtaining the member 'build_dir' of a type (line 26)
        build_dir_33817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 25), cmd_33816, 'build_dir')
        str_33818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 40), 'str', '/foo/bar')
        # Processing the call keyword arguments (line 26)
        kwargs_33819 = {}
        # Getting the type of 'self' (line 26)
        self_33814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 26)
        assertEqual_33815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_33814, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 26)
        assertEqual_call_result_33820 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), assertEqual_33815, *[build_dir_33817, str_33818], **kwargs_33819)
        
        
        # ################# End of 'test_default_settings(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_default_settings' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_33821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33821)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_default_settings'
        return stypy_return_type_33821


    @norecursion
    def test_build(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_build'
        module_type_store = module_type_store.open_function_context('test_build', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildScriptsTestCase.test_build.__dict__.__setitem__('stypy_localization', localization)
        BuildScriptsTestCase.test_build.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildScriptsTestCase.test_build.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildScriptsTestCase.test_build.__dict__.__setitem__('stypy_function_name', 'BuildScriptsTestCase.test_build')
        BuildScriptsTestCase.test_build.__dict__.__setitem__('stypy_param_names_list', [])
        BuildScriptsTestCase.test_build.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildScriptsTestCase.test_build.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildScriptsTestCase.test_build.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildScriptsTestCase.test_build.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildScriptsTestCase.test_build.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildScriptsTestCase.test_build.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildScriptsTestCase.test_build', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_build', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_build(...)' code ##################

        
        # Assigning a Call to a Name (line 29):
        
        # Call to mkdtemp(...): (line 29)
        # Processing the call keyword arguments (line 29)
        kwargs_33824 = {}
        # Getting the type of 'self' (line 29)
        self_33822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 29)
        mkdtemp_33823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 17), self_33822, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 29)
        mkdtemp_call_result_33825 = invoke(stypy.reporting.localization.Localization(__file__, 29, 17), mkdtemp_33823, *[], **kwargs_33824)
        
        # Assigning a type to the variable 'source' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'source', mkdtemp_call_result_33825)
        
        # Assigning a Call to a Name (line 30):
        
        # Call to mkdtemp(...): (line 30)
        # Processing the call keyword arguments (line 30)
        kwargs_33828 = {}
        # Getting the type of 'self' (line 30)
        self_33826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 30)
        mkdtemp_33827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 17), self_33826, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 30)
        mkdtemp_call_result_33829 = invoke(stypy.reporting.localization.Localization(__file__, 30, 17), mkdtemp_33827, *[], **kwargs_33828)
        
        # Assigning a type to the variable 'target' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'target', mkdtemp_call_result_33829)
        
        # Assigning a Call to a Name (line 31):
        
        # Call to write_sample_scripts(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'source' (line 31)
        source_33832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 45), 'source', False)
        # Processing the call keyword arguments (line 31)
        kwargs_33833 = {}
        # Getting the type of 'self' (line 31)
        self_33830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'self', False)
        # Obtaining the member 'write_sample_scripts' of a type (line 31)
        write_sample_scripts_33831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 19), self_33830, 'write_sample_scripts')
        # Calling write_sample_scripts(args, kwargs) (line 31)
        write_sample_scripts_call_result_33834 = invoke(stypy.reporting.localization.Localization(__file__, 31, 19), write_sample_scripts_33831, *[source_33832], **kwargs_33833)
        
        # Assigning a type to the variable 'expected' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'expected', write_sample_scripts_call_result_33834)
        
        # Assigning a Call to a Name (line 33):
        
        # Call to get_build_scripts_cmd(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'target' (line 33)
        target_33837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 41), 'target', False)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'expected' (line 35)
        expected_33845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 52), 'expected', False)
        comprehension_33846 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 42), expected_33845)
        # Assigning a type to the variable 'fn' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 42), 'fn', comprehension_33846)
        
        # Call to join(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'source' (line 34)
        source_33841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 55), 'source', False)
        # Getting the type of 'fn' (line 34)
        fn_33842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 63), 'fn', False)
        # Processing the call keyword arguments (line 34)
        kwargs_33843 = {}
        # Getting the type of 'os' (line 34)
        os_33838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 42), 'os', False)
        # Obtaining the member 'path' of a type (line 34)
        path_33839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 42), os_33838, 'path')
        # Obtaining the member 'join' of a type (line 34)
        join_33840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 42), path_33839, 'join')
        # Calling join(args, kwargs) (line 34)
        join_call_result_33844 = invoke(stypy.reporting.localization.Localization(__file__, 34, 42), join_33840, *[source_33841, fn_33842], **kwargs_33843)
        
        list_33847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 42), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 42), list_33847, join_call_result_33844)
        # Processing the call keyword arguments (line 33)
        kwargs_33848 = {}
        # Getting the type of 'self' (line 33)
        self_33835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'self', False)
        # Obtaining the member 'get_build_scripts_cmd' of a type (line 33)
        get_build_scripts_cmd_33836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 14), self_33835, 'get_build_scripts_cmd')
        # Calling get_build_scripts_cmd(args, kwargs) (line 33)
        get_build_scripts_cmd_call_result_33849 = invoke(stypy.reporting.localization.Localization(__file__, 33, 14), get_build_scripts_cmd_33836, *[target_33837, list_33847], **kwargs_33848)
        
        # Assigning a type to the variable 'cmd' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'cmd', get_build_scripts_cmd_call_result_33849)
        
        # Call to finalize_options(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_33852 = {}
        # Getting the type of 'cmd' (line 36)
        cmd_33850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 36)
        finalize_options_33851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), cmd_33850, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 36)
        finalize_options_call_result_33853 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), finalize_options_33851, *[], **kwargs_33852)
        
        
        # Call to run(...): (line 37)
        # Processing the call keyword arguments (line 37)
        kwargs_33856 = {}
        # Getting the type of 'cmd' (line 37)
        cmd_33854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 37)
        run_33855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), cmd_33854, 'run')
        # Calling run(args, kwargs) (line 37)
        run_call_result_33857 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), run_33855, *[], **kwargs_33856)
        
        
        # Assigning a Call to a Name (line 39):
        
        # Call to listdir(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'target' (line 39)
        target_33860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'target', False)
        # Processing the call keyword arguments (line 39)
        kwargs_33861 = {}
        # Getting the type of 'os' (line 39)
        os_33858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'os', False)
        # Obtaining the member 'listdir' of a type (line 39)
        listdir_33859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 16), os_33858, 'listdir')
        # Calling listdir(args, kwargs) (line 39)
        listdir_call_result_33862 = invoke(stypy.reporting.localization.Localization(__file__, 39, 16), listdir_33859, *[target_33860], **kwargs_33861)
        
        # Assigning a type to the variable 'built' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'built', listdir_call_result_33862)
        
        # Getting the type of 'expected' (line 40)
        expected_33863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'expected')
        # Testing the type of a for loop iterable (line 40)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 8), expected_33863)
        # Getting the type of the for loop variable (line 40)
        for_loop_var_33864 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 8), expected_33863)
        # Assigning a type to the variable 'name' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'name', for_loop_var_33864)
        # SSA begins for a for statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertIn(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'name' (line 41)
        name_33867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 26), 'name', False)
        # Getting the type of 'built' (line 41)
        built_33868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 32), 'built', False)
        # Processing the call keyword arguments (line 41)
        kwargs_33869 = {}
        # Getting the type of 'self' (line 41)
        self_33865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 41)
        assertIn_33866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), self_33865, 'assertIn')
        # Calling assertIn(args, kwargs) (line 41)
        assertIn_call_result_33870 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), assertIn_33866, *[name_33867, built_33868], **kwargs_33869)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_build(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_build' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_33871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33871)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_build'
        return stypy_return_type_33871


    @norecursion
    def get_build_scripts_cmd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_build_scripts_cmd'
        module_type_store = module_type_store.open_function_context('get_build_scripts_cmd', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildScriptsTestCase.get_build_scripts_cmd.__dict__.__setitem__('stypy_localization', localization)
        BuildScriptsTestCase.get_build_scripts_cmd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildScriptsTestCase.get_build_scripts_cmd.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildScriptsTestCase.get_build_scripts_cmd.__dict__.__setitem__('stypy_function_name', 'BuildScriptsTestCase.get_build_scripts_cmd')
        BuildScriptsTestCase.get_build_scripts_cmd.__dict__.__setitem__('stypy_param_names_list', ['target', 'scripts'])
        BuildScriptsTestCase.get_build_scripts_cmd.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildScriptsTestCase.get_build_scripts_cmd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildScriptsTestCase.get_build_scripts_cmd.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildScriptsTestCase.get_build_scripts_cmd.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildScriptsTestCase.get_build_scripts_cmd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildScriptsTestCase.get_build_scripts_cmd.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildScriptsTestCase.get_build_scripts_cmd', ['target', 'scripts'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_build_scripts_cmd', localization, ['target', 'scripts'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_build_scripts_cmd(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 8))
        
        # 'import sys' statement (line 44)
        import sys

        import_module(stypy.reporting.localization.Localization(__file__, 44, 8), 'sys', sys, module_type_store)
        
        
        # Assigning a Call to a Name (line 45):
        
        # Call to Distribution(...): (line 45)
        # Processing the call keyword arguments (line 45)
        kwargs_33873 = {}
        # Getting the type of 'Distribution' (line 45)
        Distribution_33872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 45)
        Distribution_call_result_33874 = invoke(stypy.reporting.localization.Localization(__file__, 45, 15), Distribution_33872, *[], **kwargs_33873)
        
        # Assigning a type to the variable 'dist' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'dist', Distribution_call_result_33874)
        
        # Assigning a Name to a Attribute (line 46):
        # Getting the type of 'scripts' (line 46)
        scripts_33875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 23), 'scripts')
        # Getting the type of 'dist' (line 46)
        dist_33876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'dist')
        # Setting the type of the member 'scripts' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), dist_33876, 'scripts', scripts_33875)
        
        # Assigning a Call to a Subscript (line 47):
        
        # Call to DummyCommand(...): (line 47)
        # Processing the call keyword arguments (line 47)
        # Getting the type of 'target' (line 48)
        target_33879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'target', False)
        keyword_33880 = target_33879
        int_33881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 18), 'int')
        keyword_33882 = int_33881
        # Getting the type of 'sys' (line 50)
        sys_33883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 23), 'sys', False)
        # Obtaining the member 'executable' of a type (line 50)
        executable_33884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 23), sys_33883, 'executable')
        keyword_33885 = executable_33884
        kwargs_33886 = {'force': keyword_33882, 'executable': keyword_33885, 'build_scripts': keyword_33880}
        # Getting the type of 'support' (line 47)
        support_33877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 36), 'support', False)
        # Obtaining the member 'DummyCommand' of a type (line 47)
        DummyCommand_33878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 36), support_33877, 'DummyCommand')
        # Calling DummyCommand(args, kwargs) (line 47)
        DummyCommand_call_result_33887 = invoke(stypy.reporting.localization.Localization(__file__, 47, 36), DummyCommand_33878, *[], **kwargs_33886)
        
        # Getting the type of 'dist' (line 47)
        dist_33888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'dist')
        # Obtaining the member 'command_obj' of a type (line 47)
        command_obj_33889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), dist_33888, 'command_obj')
        str_33890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'str', 'build')
        # Storing an element on a container (line 47)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 8), command_obj_33889, (str_33890, DummyCommand_call_result_33887))
        
        # Call to build_scripts(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'dist' (line 52)
        dist_33892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 29), 'dist', False)
        # Processing the call keyword arguments (line 52)
        kwargs_33893 = {}
        # Getting the type of 'build_scripts' (line 52)
        build_scripts_33891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'build_scripts', False)
        # Calling build_scripts(args, kwargs) (line 52)
        build_scripts_call_result_33894 = invoke(stypy.reporting.localization.Localization(__file__, 52, 15), build_scripts_33891, *[dist_33892], **kwargs_33893)
        
        # Assigning a type to the variable 'stypy_return_type' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type', build_scripts_call_result_33894)
        
        # ################# End of 'get_build_scripts_cmd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_build_scripts_cmd' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_33895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33895)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_build_scripts_cmd'
        return stypy_return_type_33895


    @norecursion
    def write_sample_scripts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_sample_scripts'
        module_type_store = module_type_store.open_function_context('write_sample_scripts', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildScriptsTestCase.write_sample_scripts.__dict__.__setitem__('stypy_localization', localization)
        BuildScriptsTestCase.write_sample_scripts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildScriptsTestCase.write_sample_scripts.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildScriptsTestCase.write_sample_scripts.__dict__.__setitem__('stypy_function_name', 'BuildScriptsTestCase.write_sample_scripts')
        BuildScriptsTestCase.write_sample_scripts.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        BuildScriptsTestCase.write_sample_scripts.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildScriptsTestCase.write_sample_scripts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildScriptsTestCase.write_sample_scripts.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildScriptsTestCase.write_sample_scripts.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildScriptsTestCase.write_sample_scripts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildScriptsTestCase.write_sample_scripts.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildScriptsTestCase.write_sample_scripts', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_sample_scripts', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_sample_scripts(...)' code ##################

        
        # Assigning a List to a Name (line 55):
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_33896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        
        # Assigning a type to the variable 'expected' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'expected', list_33896)
        
        # Call to append(...): (line 56)
        # Processing the call arguments (line 56)
        str_33899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 24), 'str', 'script1.py')
        # Processing the call keyword arguments (line 56)
        kwargs_33900 = {}
        # Getting the type of 'expected' (line 56)
        expected_33897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'expected', False)
        # Obtaining the member 'append' of a type (line 56)
        append_33898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), expected_33897, 'append')
        # Calling append(args, kwargs) (line 56)
        append_call_result_33901 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), append_33898, *[str_33899], **kwargs_33900)
        
        
        # Call to write_script(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'dir' (line 57)
        dir_33904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'dir', False)
        str_33905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 31), 'str', 'script1.py')
        str_33906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'str', '#! /usr/bin/env python2.3\n# bogus script w/ Python sh-bang\npass\n')
        # Processing the call keyword arguments (line 57)
        kwargs_33907 = {}
        # Getting the type of 'self' (line 57)
        self_33902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self', False)
        # Obtaining the member 'write_script' of a type (line 57)
        write_script_33903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_33902, 'write_script')
        # Calling write_script(args, kwargs) (line 57)
        write_script_call_result_33908 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), write_script_33903, *[dir_33904, str_33905, str_33906], **kwargs_33907)
        
        
        # Call to append(...): (line 61)
        # Processing the call arguments (line 61)
        str_33911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 24), 'str', 'script2.py')
        # Processing the call keyword arguments (line 61)
        kwargs_33912 = {}
        # Getting the type of 'expected' (line 61)
        expected_33909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'expected', False)
        # Obtaining the member 'append' of a type (line 61)
        append_33910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), expected_33909, 'append')
        # Calling append(args, kwargs) (line 61)
        append_call_result_33913 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), append_33910, *[str_33911], **kwargs_33912)
        
        
        # Call to write_script(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'dir' (line 62)
        dir_33916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'dir', False)
        str_33917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 31), 'str', 'script2.py')
        str_33918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'str', '#!/usr/bin/python\n# bogus script w/ Python sh-bang\npass\n')
        # Processing the call keyword arguments (line 62)
        kwargs_33919 = {}
        # Getting the type of 'self' (line 62)
        self_33914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self', False)
        # Obtaining the member 'write_script' of a type (line 62)
        write_script_33915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_33914, 'write_script')
        # Calling write_script(args, kwargs) (line 62)
        write_script_call_result_33920 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), write_script_33915, *[dir_33916, str_33917, str_33918], **kwargs_33919)
        
        
        # Call to append(...): (line 66)
        # Processing the call arguments (line 66)
        str_33923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 24), 'str', 'shell.sh')
        # Processing the call keyword arguments (line 66)
        kwargs_33924 = {}
        # Getting the type of 'expected' (line 66)
        expected_33921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'expected', False)
        # Obtaining the member 'append' of a type (line 66)
        append_33922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), expected_33921, 'append')
        # Calling append(args, kwargs) (line 66)
        append_call_result_33925 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), append_33922, *[str_33923], **kwargs_33924)
        
        
        # Call to write_script(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'dir' (line 67)
        dir_33928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'dir', False)
        str_33929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 31), 'str', 'shell.sh')
        str_33930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 27), 'str', '#!/bin/sh\n# bogus shell script w/ sh-bang\nexit 0\n')
        # Processing the call keyword arguments (line 67)
        kwargs_33931 = {}
        # Getting the type of 'self' (line 67)
        self_33926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self', False)
        # Obtaining the member 'write_script' of a type (line 67)
        write_script_33927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_33926, 'write_script')
        # Calling write_script(args, kwargs) (line 67)
        write_script_call_result_33932 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), write_script_33927, *[dir_33928, str_33929, str_33930], **kwargs_33931)
        
        # Getting the type of 'expected' (line 71)
        expected_33933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'expected')
        # Assigning a type to the variable 'stypy_return_type' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'stypy_return_type', expected_33933)
        
        # ################# End of 'write_sample_scripts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_sample_scripts' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_33934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33934)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_sample_scripts'
        return stypy_return_type_33934


    @norecursion
    def write_script(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_script'
        module_type_store = module_type_store.open_function_context('write_script', 73, 4, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildScriptsTestCase.write_script.__dict__.__setitem__('stypy_localization', localization)
        BuildScriptsTestCase.write_script.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildScriptsTestCase.write_script.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildScriptsTestCase.write_script.__dict__.__setitem__('stypy_function_name', 'BuildScriptsTestCase.write_script')
        BuildScriptsTestCase.write_script.__dict__.__setitem__('stypy_param_names_list', ['dir', 'name', 'text'])
        BuildScriptsTestCase.write_script.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildScriptsTestCase.write_script.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildScriptsTestCase.write_script.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildScriptsTestCase.write_script.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildScriptsTestCase.write_script.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildScriptsTestCase.write_script.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildScriptsTestCase.write_script', ['dir', 'name', 'text'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_script', localization, ['dir', 'name', 'text'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_script(...)' code ##################

        
        # Assigning a Call to a Name (line 74):
        
        # Call to open(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Call to join(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'dir' (line 74)
        dir_33939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 30), 'dir', False)
        # Getting the type of 'name' (line 74)
        name_33940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 35), 'name', False)
        # Processing the call keyword arguments (line 74)
        kwargs_33941 = {}
        # Getting the type of 'os' (line 74)
        os_33936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 74)
        path_33937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 17), os_33936, 'path')
        # Obtaining the member 'join' of a type (line 74)
        join_33938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 17), path_33937, 'join')
        # Calling join(args, kwargs) (line 74)
        join_call_result_33942 = invoke(stypy.reporting.localization.Localization(__file__, 74, 17), join_33938, *[dir_33939, name_33940], **kwargs_33941)
        
        str_33943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 42), 'str', 'w')
        # Processing the call keyword arguments (line 74)
        kwargs_33944 = {}
        # Getting the type of 'open' (line 74)
        open_33935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'open', False)
        # Calling open(args, kwargs) (line 74)
        open_call_result_33945 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), open_33935, *[join_call_result_33942, str_33943], **kwargs_33944)
        
        # Assigning a type to the variable 'f' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'f', open_call_result_33945)
        
        # Try-finally block (line 75)
        
        # Call to write(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'text' (line 76)
        text_33948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'text', False)
        # Processing the call keyword arguments (line 76)
        kwargs_33949 = {}
        # Getting the type of 'f' (line 76)
        f_33946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 76)
        write_33947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), f_33946, 'write')
        # Calling write(args, kwargs) (line 76)
        write_call_result_33950 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), write_33947, *[text_33948], **kwargs_33949)
        
        
        # finally branch of the try-finally block (line 75)
        
        # Call to close(...): (line 78)
        # Processing the call keyword arguments (line 78)
        kwargs_33953 = {}
        # Getting the type of 'f' (line 78)
        f_33951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 78)
        close_33952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), f_33951, 'close')
        # Calling close(args, kwargs) (line 78)
        close_call_result_33954 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), close_33952, *[], **kwargs_33953)
        
        
        
        # ################# End of 'write_script(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_script' in the type store
        # Getting the type of 'stypy_return_type' (line 73)
        stypy_return_type_33955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33955)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_script'
        return stypy_return_type_33955


    @norecursion
    def test_version_int(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_version_int'
        module_type_store = module_type_store.open_function_context('test_version_int', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildScriptsTestCase.test_version_int.__dict__.__setitem__('stypy_localization', localization)
        BuildScriptsTestCase.test_version_int.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildScriptsTestCase.test_version_int.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildScriptsTestCase.test_version_int.__dict__.__setitem__('stypy_function_name', 'BuildScriptsTestCase.test_version_int')
        BuildScriptsTestCase.test_version_int.__dict__.__setitem__('stypy_param_names_list', [])
        BuildScriptsTestCase.test_version_int.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildScriptsTestCase.test_version_int.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildScriptsTestCase.test_version_int.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildScriptsTestCase.test_version_int.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildScriptsTestCase.test_version_int.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildScriptsTestCase.test_version_int.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildScriptsTestCase.test_version_int', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_version_int', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_version_int(...)' code ##################

        
        # Assigning a Call to a Name (line 81):
        
        # Call to mkdtemp(...): (line 81)
        # Processing the call keyword arguments (line 81)
        kwargs_33958 = {}
        # Getting the type of 'self' (line 81)
        self_33956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 81)
        mkdtemp_33957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 17), self_33956, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 81)
        mkdtemp_call_result_33959 = invoke(stypy.reporting.localization.Localization(__file__, 81, 17), mkdtemp_33957, *[], **kwargs_33958)
        
        # Assigning a type to the variable 'source' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'source', mkdtemp_call_result_33959)
        
        # Assigning a Call to a Name (line 82):
        
        # Call to mkdtemp(...): (line 82)
        # Processing the call keyword arguments (line 82)
        kwargs_33962 = {}
        # Getting the type of 'self' (line 82)
        self_33960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 82)
        mkdtemp_33961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 17), self_33960, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 82)
        mkdtemp_call_result_33963 = invoke(stypy.reporting.localization.Localization(__file__, 82, 17), mkdtemp_33961, *[], **kwargs_33962)
        
        # Assigning a type to the variable 'target' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'target', mkdtemp_call_result_33963)
        
        # Assigning a Call to a Name (line 83):
        
        # Call to write_sample_scripts(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'source' (line 83)
        source_33966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 45), 'source', False)
        # Processing the call keyword arguments (line 83)
        kwargs_33967 = {}
        # Getting the type of 'self' (line 83)
        self_33964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'self', False)
        # Obtaining the member 'write_sample_scripts' of a type (line 83)
        write_sample_scripts_33965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 19), self_33964, 'write_sample_scripts')
        # Calling write_sample_scripts(args, kwargs) (line 83)
        write_sample_scripts_call_result_33968 = invoke(stypy.reporting.localization.Localization(__file__, 83, 19), write_sample_scripts_33965, *[source_33966], **kwargs_33967)
        
        # Assigning a type to the variable 'expected' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'expected', write_sample_scripts_call_result_33968)
        
        # Assigning a Call to a Name (line 86):
        
        # Call to get_build_scripts_cmd(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'target' (line 86)
        target_33971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 41), 'target', False)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'expected' (line 88)
        expected_33979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 52), 'expected', False)
        comprehension_33980 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 42), expected_33979)
        # Assigning a type to the variable 'fn' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 42), 'fn', comprehension_33980)
        
        # Call to join(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'source' (line 87)
        source_33975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 55), 'source', False)
        # Getting the type of 'fn' (line 87)
        fn_33976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 63), 'fn', False)
        # Processing the call keyword arguments (line 87)
        kwargs_33977 = {}
        # Getting the type of 'os' (line 87)
        os_33972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 42), 'os', False)
        # Obtaining the member 'path' of a type (line 87)
        path_33973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 42), os_33972, 'path')
        # Obtaining the member 'join' of a type (line 87)
        join_33974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 42), path_33973, 'join')
        # Calling join(args, kwargs) (line 87)
        join_call_result_33978 = invoke(stypy.reporting.localization.Localization(__file__, 87, 42), join_33974, *[source_33975, fn_33976], **kwargs_33977)
        
        list_33981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 42), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 42), list_33981, join_call_result_33978)
        # Processing the call keyword arguments (line 86)
        kwargs_33982 = {}
        # Getting the type of 'self' (line 86)
        self_33969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'self', False)
        # Obtaining the member 'get_build_scripts_cmd' of a type (line 86)
        get_build_scripts_cmd_33970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 14), self_33969, 'get_build_scripts_cmd')
        # Calling get_build_scripts_cmd(args, kwargs) (line 86)
        get_build_scripts_cmd_call_result_33983 = invoke(stypy.reporting.localization.Localization(__file__, 86, 14), get_build_scripts_cmd_33970, *[target_33971, list_33981], **kwargs_33982)
        
        # Assigning a type to the variable 'cmd' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'cmd', get_build_scripts_cmd_call_result_33983)
        
        # Call to finalize_options(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_33986 = {}
        # Getting the type of 'cmd' (line 89)
        cmd_33984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 89)
        finalize_options_33985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), cmd_33984, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 89)
        finalize_options_call_result_33987 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), finalize_options_33985, *[], **kwargs_33986)
        
        
        # Assigning a Call to a Name (line 96):
        
        # Call to get(...): (line 96)
        # Processing the call arguments (line 96)
        str_33993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 46), 'str', 'VERSION')
        # Processing the call keyword arguments (line 96)
        kwargs_33994 = {}
        
        # Call to get_config_vars(...): (line 96)
        # Processing the call keyword arguments (line 96)
        kwargs_33990 = {}
        # Getting the type of 'sysconfig' (line 96)
        sysconfig_33988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 14), 'sysconfig', False)
        # Obtaining the member 'get_config_vars' of a type (line 96)
        get_config_vars_33989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 14), sysconfig_33988, 'get_config_vars')
        # Calling get_config_vars(args, kwargs) (line 96)
        get_config_vars_call_result_33991 = invoke(stypy.reporting.localization.Localization(__file__, 96, 14), get_config_vars_33989, *[], **kwargs_33990)
        
        # Obtaining the member 'get' of a type (line 96)
        get_33992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 14), get_config_vars_call_result_33991, 'get')
        # Calling get(args, kwargs) (line 96)
        get_call_result_33995 = invoke(stypy.reporting.localization.Localization(__file__, 96, 14), get_33992, *[str_33993], **kwargs_33994)
        
        # Assigning a type to the variable 'old' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'old', get_call_result_33995)
        
        # Assigning a Num to a Subscript (line 97):
        int_33996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 44), 'int')
        # Getting the type of 'sysconfig' (line 97)
        sysconfig_33997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'sysconfig')
        # Obtaining the member '_CONFIG_VARS' of a type (line 97)
        _CONFIG_VARS_33998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), sysconfig_33997, '_CONFIG_VARS')
        str_33999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 31), 'str', 'VERSION')
        # Storing an element on a container (line 97)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 8), _CONFIG_VARS_33998, (str_33999, int_33996))
        
        # Try-finally block (line 98)
        
        # Call to run(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_34002 = {}
        # Getting the type of 'cmd' (line 99)
        cmd_34000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'cmd', False)
        # Obtaining the member 'run' of a type (line 99)
        run_34001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), cmd_34000, 'run')
        # Calling run(args, kwargs) (line 99)
        run_call_result_34003 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), run_34001, *[], **kwargs_34002)
        
        
        # finally branch of the try-finally block (line 98)
        
        # Type idiom detected: calculating its left and rigth part (line 101)
        # Getting the type of 'old' (line 101)
        old_34004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'old')
        # Getting the type of 'None' (line 101)
        None_34005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'None')
        
        (may_be_34006, more_types_in_union_34007) = may_not_be_none(old_34004, None_34005)

        if may_be_34006:

            if more_types_in_union_34007:
                # Runtime conditional SSA (line 101)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 102):
            # Getting the type of 'old' (line 102)
            old_34008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 52), 'old')
            # Getting the type of 'sysconfig' (line 102)
            sysconfig_34009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'sysconfig')
            # Obtaining the member '_CONFIG_VARS' of a type (line 102)
            _CONFIG_VARS_34010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 16), sysconfig_34009, '_CONFIG_VARS')
            str_34011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 39), 'str', 'VERSION')
            # Storing an element on a container (line 102)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 16), _CONFIG_VARS_34010, (str_34011, old_34008))

            if more_types_in_union_34007:
                # SSA join for if statement (line 101)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Assigning a Call to a Name (line 104):
        
        # Call to listdir(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'target' (line 104)
        target_34014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 'target', False)
        # Processing the call keyword arguments (line 104)
        kwargs_34015 = {}
        # Getting the type of 'os' (line 104)
        os_34012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'os', False)
        # Obtaining the member 'listdir' of a type (line 104)
        listdir_34013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 16), os_34012, 'listdir')
        # Calling listdir(args, kwargs) (line 104)
        listdir_call_result_34016 = invoke(stypy.reporting.localization.Localization(__file__, 104, 16), listdir_34013, *[target_34014], **kwargs_34015)
        
        # Assigning a type to the variable 'built' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'built', listdir_call_result_34016)
        
        # Getting the type of 'expected' (line 105)
        expected_34017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'expected')
        # Testing the type of a for loop iterable (line 105)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 8), expected_34017)
        # Getting the type of the for loop variable (line 105)
        for_loop_var_34018 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 8), expected_34017)
        # Assigning a type to the variable 'name' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'name', for_loop_var_34018)
        # SSA begins for a for statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertIn(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'name' (line 106)
        name_34021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'name', False)
        # Getting the type of 'built' (line 106)
        built_34022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 32), 'built', False)
        # Processing the call keyword arguments (line 106)
        kwargs_34023 = {}
        # Getting the type of 'self' (line 106)
        self_34019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 106)
        assertIn_34020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), self_34019, 'assertIn')
        # Calling assertIn(args, kwargs) (line 106)
        assertIn_call_result_34024 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), assertIn_34020, *[name_34021, built_34022], **kwargs_34023)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_version_int(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_version_int' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_34025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34025)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_version_int'
        return stypy_return_type_34025


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildScriptsTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BuildScriptsTestCase' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'BuildScriptsTestCase', BuildScriptsTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 108, 0, False)
    
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

    
    # Call to makeSuite(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'BuildScriptsTestCase' (line 109)
    BuildScriptsTestCase_34028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'BuildScriptsTestCase', False)
    # Processing the call keyword arguments (line 109)
    kwargs_34029 = {}
    # Getting the type of 'unittest' (line 109)
    unittest_34026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 109)
    makeSuite_34027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), unittest_34026, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 109)
    makeSuite_call_result_34030 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), makeSuite_34027, *[BuildScriptsTestCase_34028], **kwargs_34029)
    
    # Assigning a type to the variable 'stypy_return_type' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type', makeSuite_call_result_34030)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 108)
    stypy_return_type_34031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34031)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_34031

# Assigning a type to the variable 'test_suite' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Call to test_suite(...): (line 112)
    # Processing the call keyword arguments (line 112)
    kwargs_34034 = {}
    # Getting the type of 'test_suite' (line 112)
    test_suite_34033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 112)
    test_suite_call_result_34035 = invoke(stypy.reporting.localization.Localization(__file__, 112, 17), test_suite_34033, *[], **kwargs_34034)
    
    # Processing the call keyword arguments (line 112)
    kwargs_34036 = {}
    # Getting the type of 'run_unittest' (line 112)
    run_unittest_34032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 112)
    run_unittest_call_result_34037 = invoke(stypy.reporting.localization.Localization(__file__, 112, 4), run_unittest_34032, *[test_suite_call_result_34035], **kwargs_34036)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
