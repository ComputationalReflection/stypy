
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.install_scripts.'''
2: 
3: import os
4: import unittest
5: 
6: from distutils.command.install_scripts import install_scripts
7: from distutils.core import Distribution
8: 
9: from distutils.tests import support
10: from test.test_support import run_unittest
11: 
12: 
13: class InstallScriptsTestCase(support.TempdirManager,
14:                              support.LoggingSilencer,
15:                              unittest.TestCase):
16: 
17:     def test_default_settings(self):
18:         dist = Distribution()
19:         dist.command_obj["build"] = support.DummyCommand(
20:             build_scripts="/foo/bar")
21:         dist.command_obj["install"] = support.DummyCommand(
22:             install_scripts="/splat/funk",
23:             force=1,
24:             skip_build=1,
25:             )
26:         cmd = install_scripts(dist)
27:         self.assertFalse(cmd.force)
28:         self.assertFalse(cmd.skip_build)
29:         self.assertIsNone(cmd.build_dir)
30:         self.assertIsNone(cmd.install_dir)
31: 
32:         cmd.finalize_options()
33: 
34:         self.assertTrue(cmd.force)
35:         self.assertTrue(cmd.skip_build)
36:         self.assertEqual(cmd.build_dir, "/foo/bar")
37:         self.assertEqual(cmd.install_dir, "/splat/funk")
38: 
39:     def test_installation(self):
40:         source = self.mkdtemp()
41:         expected = []
42: 
43:         def write_script(name, text):
44:             expected.append(name)
45:             f = open(os.path.join(source, name), "w")
46:             try:
47:                 f.write(text)
48:             finally:
49:                 f.close()
50: 
51:         write_script("script1.py", ("#! /usr/bin/env python2.3\n"
52:                                     "# bogus script w/ Python sh-bang\n"
53:                                     "pass\n"))
54:         write_script("script2.py", ("#!/usr/bin/python\n"
55:                                     "# bogus script w/ Python sh-bang\n"
56:                                     "pass\n"))
57:         write_script("shell.sh", ("#!/bin/sh\n"
58:                                   "# bogus shell script w/ sh-bang\n"
59:                                   "exit 0\n"))
60: 
61:         target = self.mkdtemp()
62:         dist = Distribution()
63:         dist.command_obj["build"] = support.DummyCommand(build_scripts=source)
64:         dist.command_obj["install"] = support.DummyCommand(
65:             install_scripts=target,
66:             force=1,
67:             skip_build=1,
68:             )
69:         cmd = install_scripts(dist)
70:         cmd.finalize_options()
71:         cmd.run()
72: 
73:         installed = os.listdir(target)
74:         for name in expected:
75:             self.assertIn(name, installed)
76: 
77: 
78: def test_suite():
79:     return unittest.makeSuite(InstallScriptsTestCase)
80: 
81: if __name__ == "__main__":
82:     run_unittest(test_suite())
83: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_41336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.install_scripts.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import unittest' statement (line 4)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.command.install_scripts import install_scripts' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41337 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.install_scripts')

if (type(import_41337) is not StypyTypeError):

    if (import_41337 != 'pyd_module'):
        __import__(import_41337)
        sys_modules_41338 = sys.modules[import_41337]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.install_scripts', sys_modules_41338.module_type_store, module_type_store, ['install_scripts'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_41338, sys_modules_41338.module_type_store, module_type_store)
    else:
        from distutils.command.install_scripts import install_scripts

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.install_scripts', None, module_type_store, ['install_scripts'], [install_scripts])

else:
    # Assigning a type to the variable 'distutils.command.install_scripts' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.install_scripts', import_41337)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.core import Distribution' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41339 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.core')

if (type(import_41339) is not StypyTypeError):

    if (import_41339 != 'pyd_module'):
        __import__(import_41339)
        sys_modules_41340 = sys.modules[import_41339]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.core', sys_modules_41340.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_41340, sys_modules_41340.module_type_store, module_type_store)
    else:
        from distutils.core import Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.core', None, module_type_store, ['Distribution'], [Distribution])

else:
    # Assigning a type to the variable 'distutils.core' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.core', import_41339)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.tests import support' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41341 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.tests')

if (type(import_41341) is not StypyTypeError):

    if (import_41341 != 'pyd_module'):
        __import__(import_41341)
        sys_modules_41342 = sys.modules[import_41341]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.tests', sys_modules_41342.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_41342, sys_modules_41342.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.tests', import_41341)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from test.test_support import run_unittest' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41343 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'test.test_support')

if (type(import_41343) is not StypyTypeError):

    if (import_41343 != 'pyd_module'):
        __import__(import_41343)
        sys_modules_41344 = sys.modules[import_41343]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'test.test_support', sys_modules_41344.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_41344, sys_modules_41344.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'test.test_support', import_41343)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'InstallScriptsTestCase' class
# Getting the type of 'support' (line 13)
support_41345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), 'support')
# Obtaining the member 'TempdirManager' of a type (line 13)
TempdirManager_41346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 29), support_41345, 'TempdirManager')
# Getting the type of 'support' (line 14)
support_41347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 29), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 14)
LoggingSilencer_41348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 29), support_41347, 'LoggingSilencer')
# Getting the type of 'unittest' (line 15)
unittest_41349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 29), 'unittest')
# Obtaining the member 'TestCase' of a type (line 15)
TestCase_41350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 29), unittest_41349, 'TestCase')

class InstallScriptsTestCase(TempdirManager_41346, LoggingSilencer_41348, TestCase_41350, ):

    @norecursion
    def test_default_settings(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_default_settings'
        module_type_store = module_type_store.open_function_context('test_default_settings', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_localization', localization)
        InstallScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_function_name', 'InstallScriptsTestCase.test_default_settings')
        InstallScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_param_names_list', [])
        InstallScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallScriptsTestCase.test_default_settings.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallScriptsTestCase.test_default_settings', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 18):
        
        # Call to Distribution(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_41352 = {}
        # Getting the type of 'Distribution' (line 18)
        Distribution_41351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 18)
        Distribution_call_result_41353 = invoke(stypy.reporting.localization.Localization(__file__, 18, 15), Distribution_41351, *[], **kwargs_41352)
        
        # Assigning a type to the variable 'dist' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'dist', Distribution_call_result_41353)
        
        # Assigning a Call to a Subscript (line 19):
        
        # Call to DummyCommand(...): (line 19)
        # Processing the call keyword arguments (line 19)
        str_41356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'str', '/foo/bar')
        keyword_41357 = str_41356
        kwargs_41358 = {'build_scripts': keyword_41357}
        # Getting the type of 'support' (line 19)
        support_41354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 36), 'support', False)
        # Obtaining the member 'DummyCommand' of a type (line 19)
        DummyCommand_41355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 36), support_41354, 'DummyCommand')
        # Calling DummyCommand(args, kwargs) (line 19)
        DummyCommand_call_result_41359 = invoke(stypy.reporting.localization.Localization(__file__, 19, 36), DummyCommand_41355, *[], **kwargs_41358)
        
        # Getting the type of 'dist' (line 19)
        dist_41360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'dist')
        # Obtaining the member 'command_obj' of a type (line 19)
        command_obj_41361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), dist_41360, 'command_obj')
        str_41362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'str', 'build')
        # Storing an element on a container (line 19)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 8), command_obj_41361, (str_41362, DummyCommand_call_result_41359))
        
        # Assigning a Call to a Subscript (line 21):
        
        # Call to DummyCommand(...): (line 21)
        # Processing the call keyword arguments (line 21)
        str_41365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 28), 'str', '/splat/funk')
        keyword_41366 = str_41365
        int_41367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 18), 'int')
        keyword_41368 = int_41367
        int_41369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'int')
        keyword_41370 = int_41369
        kwargs_41371 = {'skip_build': keyword_41370, 'install_scripts': keyword_41366, 'force': keyword_41368}
        # Getting the type of 'support' (line 21)
        support_41363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 38), 'support', False)
        # Obtaining the member 'DummyCommand' of a type (line 21)
        DummyCommand_41364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 38), support_41363, 'DummyCommand')
        # Calling DummyCommand(args, kwargs) (line 21)
        DummyCommand_call_result_41372 = invoke(stypy.reporting.localization.Localization(__file__, 21, 38), DummyCommand_41364, *[], **kwargs_41371)
        
        # Getting the type of 'dist' (line 21)
        dist_41373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'dist')
        # Obtaining the member 'command_obj' of a type (line 21)
        command_obj_41374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), dist_41373, 'command_obj')
        str_41375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'str', 'install')
        # Storing an element on a container (line 21)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 8), command_obj_41374, (str_41375, DummyCommand_call_result_41372))
        
        # Assigning a Call to a Name (line 26):
        
        # Call to install_scripts(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'dist' (line 26)
        dist_41377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 30), 'dist', False)
        # Processing the call keyword arguments (line 26)
        kwargs_41378 = {}
        # Getting the type of 'install_scripts' (line 26)
        install_scripts_41376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'install_scripts', False)
        # Calling install_scripts(args, kwargs) (line 26)
        install_scripts_call_result_41379 = invoke(stypy.reporting.localization.Localization(__file__, 26, 14), install_scripts_41376, *[dist_41377], **kwargs_41378)
        
        # Assigning a type to the variable 'cmd' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'cmd', install_scripts_call_result_41379)
        
        # Call to assertFalse(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'cmd' (line 27)
        cmd_41382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'cmd', False)
        # Obtaining the member 'force' of a type (line 27)
        force_41383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 25), cmd_41382, 'force')
        # Processing the call keyword arguments (line 27)
        kwargs_41384 = {}
        # Getting the type of 'self' (line 27)
        self_41380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 27)
        assertFalse_41381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_41380, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 27)
        assertFalse_call_result_41385 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assertFalse_41381, *[force_41383], **kwargs_41384)
        
        
        # Call to assertFalse(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'cmd' (line 28)
        cmd_41388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 25), 'cmd', False)
        # Obtaining the member 'skip_build' of a type (line 28)
        skip_build_41389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 25), cmd_41388, 'skip_build')
        # Processing the call keyword arguments (line 28)
        kwargs_41390 = {}
        # Getting the type of 'self' (line 28)
        self_41386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 28)
        assertFalse_41387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_41386, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 28)
        assertFalse_call_result_41391 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), assertFalse_41387, *[skip_build_41389], **kwargs_41390)
        
        
        # Call to assertIsNone(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'cmd' (line 29)
        cmd_41394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 26), 'cmd', False)
        # Obtaining the member 'build_dir' of a type (line 29)
        build_dir_41395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 26), cmd_41394, 'build_dir')
        # Processing the call keyword arguments (line 29)
        kwargs_41396 = {}
        # Getting the type of 'self' (line 29)
        self_41392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self', False)
        # Obtaining the member 'assertIsNone' of a type (line 29)
        assertIsNone_41393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_41392, 'assertIsNone')
        # Calling assertIsNone(args, kwargs) (line 29)
        assertIsNone_call_result_41397 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), assertIsNone_41393, *[build_dir_41395], **kwargs_41396)
        
        
        # Call to assertIsNone(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'cmd' (line 30)
        cmd_41400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'cmd', False)
        # Obtaining the member 'install_dir' of a type (line 30)
        install_dir_41401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 26), cmd_41400, 'install_dir')
        # Processing the call keyword arguments (line 30)
        kwargs_41402 = {}
        # Getting the type of 'self' (line 30)
        self_41398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self', False)
        # Obtaining the member 'assertIsNone' of a type (line 30)
        assertIsNone_41399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_41398, 'assertIsNone')
        # Calling assertIsNone(args, kwargs) (line 30)
        assertIsNone_call_result_41403 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assertIsNone_41399, *[install_dir_41401], **kwargs_41402)
        
        
        # Call to finalize_options(...): (line 32)
        # Processing the call keyword arguments (line 32)
        kwargs_41406 = {}
        # Getting the type of 'cmd' (line 32)
        cmd_41404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 32)
        finalize_options_41405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), cmd_41404, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 32)
        finalize_options_call_result_41407 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), finalize_options_41405, *[], **kwargs_41406)
        
        
        # Call to assertTrue(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'cmd' (line 34)
        cmd_41410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 24), 'cmd', False)
        # Obtaining the member 'force' of a type (line 34)
        force_41411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 24), cmd_41410, 'force')
        # Processing the call keyword arguments (line 34)
        kwargs_41412 = {}
        # Getting the type of 'self' (line 34)
        self_41408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 34)
        assertTrue_41409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_41408, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 34)
        assertTrue_call_result_41413 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assertTrue_41409, *[force_41411], **kwargs_41412)
        
        
        # Call to assertTrue(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'cmd' (line 35)
        cmd_41416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'cmd', False)
        # Obtaining the member 'skip_build' of a type (line 35)
        skip_build_41417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 24), cmd_41416, 'skip_build')
        # Processing the call keyword arguments (line 35)
        kwargs_41418 = {}
        # Getting the type of 'self' (line 35)
        self_41414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 35)
        assertTrue_41415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_41414, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 35)
        assertTrue_call_result_41419 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), assertTrue_41415, *[skip_build_41417], **kwargs_41418)
        
        
        # Call to assertEqual(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'cmd' (line 36)
        cmd_41422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'cmd', False)
        # Obtaining the member 'build_dir' of a type (line 36)
        build_dir_41423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 25), cmd_41422, 'build_dir')
        str_41424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 40), 'str', '/foo/bar')
        # Processing the call keyword arguments (line 36)
        kwargs_41425 = {}
        # Getting the type of 'self' (line 36)
        self_41420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 36)
        assertEqual_41421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_41420, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 36)
        assertEqual_call_result_41426 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assertEqual_41421, *[build_dir_41423, str_41424], **kwargs_41425)
        
        
        # Call to assertEqual(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'cmd' (line 37)
        cmd_41429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 25), 'cmd', False)
        # Obtaining the member 'install_dir' of a type (line 37)
        install_dir_41430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 25), cmd_41429, 'install_dir')
        str_41431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 42), 'str', '/splat/funk')
        # Processing the call keyword arguments (line 37)
        kwargs_41432 = {}
        # Getting the type of 'self' (line 37)
        self_41427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 37)
        assertEqual_41428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_41427, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 37)
        assertEqual_call_result_41433 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), assertEqual_41428, *[install_dir_41430, str_41431], **kwargs_41432)
        
        
        # ################# End of 'test_default_settings(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_default_settings' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_41434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41434)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_default_settings'
        return stypy_return_type_41434


    @norecursion
    def test_installation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_installation'
        module_type_store = module_type_store.open_function_context('test_installation', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallScriptsTestCase.test_installation.__dict__.__setitem__('stypy_localization', localization)
        InstallScriptsTestCase.test_installation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallScriptsTestCase.test_installation.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallScriptsTestCase.test_installation.__dict__.__setitem__('stypy_function_name', 'InstallScriptsTestCase.test_installation')
        InstallScriptsTestCase.test_installation.__dict__.__setitem__('stypy_param_names_list', [])
        InstallScriptsTestCase.test_installation.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallScriptsTestCase.test_installation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallScriptsTestCase.test_installation.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallScriptsTestCase.test_installation.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallScriptsTestCase.test_installation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallScriptsTestCase.test_installation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallScriptsTestCase.test_installation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_installation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_installation(...)' code ##################

        
        # Assigning a Call to a Name (line 40):
        
        # Call to mkdtemp(...): (line 40)
        # Processing the call keyword arguments (line 40)
        kwargs_41437 = {}
        # Getting the type of 'self' (line 40)
        self_41435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 40)
        mkdtemp_41436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 17), self_41435, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 40)
        mkdtemp_call_result_41438 = invoke(stypy.reporting.localization.Localization(__file__, 40, 17), mkdtemp_41436, *[], **kwargs_41437)
        
        # Assigning a type to the variable 'source' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'source', mkdtemp_call_result_41438)
        
        # Assigning a List to a Name (line 41):
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_41439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        
        # Assigning a type to the variable 'expected' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'expected', list_41439)

        @norecursion
        def write_script(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'write_script'
            module_type_store = module_type_store.open_function_context('write_script', 43, 8, False)
            
            # Passed parameters checking function
            write_script.stypy_localization = localization
            write_script.stypy_type_of_self = None
            write_script.stypy_type_store = module_type_store
            write_script.stypy_function_name = 'write_script'
            write_script.stypy_param_names_list = ['name', 'text']
            write_script.stypy_varargs_param_name = None
            write_script.stypy_kwargs_param_name = None
            write_script.stypy_call_defaults = defaults
            write_script.stypy_call_varargs = varargs
            write_script.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'write_script', ['name', 'text'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'write_script', localization, ['name', 'text'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'write_script(...)' code ##################

            
            # Call to append(...): (line 44)
            # Processing the call arguments (line 44)
            # Getting the type of 'name' (line 44)
            name_41442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 28), 'name', False)
            # Processing the call keyword arguments (line 44)
            kwargs_41443 = {}
            # Getting the type of 'expected' (line 44)
            expected_41440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'expected', False)
            # Obtaining the member 'append' of a type (line 44)
            append_41441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), expected_41440, 'append')
            # Calling append(args, kwargs) (line 44)
            append_call_result_41444 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), append_41441, *[name_41442], **kwargs_41443)
            
            
            # Assigning a Call to a Name (line 45):
            
            # Call to open(...): (line 45)
            # Processing the call arguments (line 45)
            
            # Call to join(...): (line 45)
            # Processing the call arguments (line 45)
            # Getting the type of 'source' (line 45)
            source_41449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), 'source', False)
            # Getting the type of 'name' (line 45)
            name_41450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 42), 'name', False)
            # Processing the call keyword arguments (line 45)
            kwargs_41451 = {}
            # Getting the type of 'os' (line 45)
            os_41446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 21), 'os', False)
            # Obtaining the member 'path' of a type (line 45)
            path_41447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 21), os_41446, 'path')
            # Obtaining the member 'join' of a type (line 45)
            join_41448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 21), path_41447, 'join')
            # Calling join(args, kwargs) (line 45)
            join_call_result_41452 = invoke(stypy.reporting.localization.Localization(__file__, 45, 21), join_41448, *[source_41449, name_41450], **kwargs_41451)
            
            str_41453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 49), 'str', 'w')
            # Processing the call keyword arguments (line 45)
            kwargs_41454 = {}
            # Getting the type of 'open' (line 45)
            open_41445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'open', False)
            # Calling open(args, kwargs) (line 45)
            open_call_result_41455 = invoke(stypy.reporting.localization.Localization(__file__, 45, 16), open_41445, *[join_call_result_41452, str_41453], **kwargs_41454)
            
            # Assigning a type to the variable 'f' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'f', open_call_result_41455)
            
            # Try-finally block (line 46)
            
            # Call to write(...): (line 47)
            # Processing the call arguments (line 47)
            # Getting the type of 'text' (line 47)
            text_41458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'text', False)
            # Processing the call keyword arguments (line 47)
            kwargs_41459 = {}
            # Getting the type of 'f' (line 47)
            f_41456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'f', False)
            # Obtaining the member 'write' of a type (line 47)
            write_41457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), f_41456, 'write')
            # Calling write(args, kwargs) (line 47)
            write_call_result_41460 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), write_41457, *[text_41458], **kwargs_41459)
            
            
            # finally branch of the try-finally block (line 46)
            
            # Call to close(...): (line 49)
            # Processing the call keyword arguments (line 49)
            kwargs_41463 = {}
            # Getting the type of 'f' (line 49)
            f_41461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'f', False)
            # Obtaining the member 'close' of a type (line 49)
            close_41462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), f_41461, 'close')
            # Calling close(args, kwargs) (line 49)
            close_call_result_41464 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), close_41462, *[], **kwargs_41463)
            
            
            
            # ################# End of 'write_script(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'write_script' in the type store
            # Getting the type of 'stypy_return_type' (line 43)
            stypy_return_type_41465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_41465)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'write_script'
            return stypy_return_type_41465

        # Assigning a type to the variable 'write_script' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'write_script', write_script)
        
        # Call to write_script(...): (line 51)
        # Processing the call arguments (line 51)
        str_41467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 21), 'str', 'script1.py')
        str_41468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 36), 'str', '#! /usr/bin/env python2.3\n# bogus script w/ Python sh-bang\npass\n')
        # Processing the call keyword arguments (line 51)
        kwargs_41469 = {}
        # Getting the type of 'write_script' (line 51)
        write_script_41466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'write_script', False)
        # Calling write_script(args, kwargs) (line 51)
        write_script_call_result_41470 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), write_script_41466, *[str_41467, str_41468], **kwargs_41469)
        
        
        # Call to write_script(...): (line 54)
        # Processing the call arguments (line 54)
        str_41472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 21), 'str', 'script2.py')
        str_41473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 36), 'str', '#!/usr/bin/python\n# bogus script w/ Python sh-bang\npass\n')
        # Processing the call keyword arguments (line 54)
        kwargs_41474 = {}
        # Getting the type of 'write_script' (line 54)
        write_script_41471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'write_script', False)
        # Calling write_script(args, kwargs) (line 54)
        write_script_call_result_41475 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), write_script_41471, *[str_41472, str_41473], **kwargs_41474)
        
        
        # Call to write_script(...): (line 57)
        # Processing the call arguments (line 57)
        str_41477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 21), 'str', 'shell.sh')
        str_41478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 34), 'str', '#!/bin/sh\n# bogus shell script w/ sh-bang\nexit 0\n')
        # Processing the call keyword arguments (line 57)
        kwargs_41479 = {}
        # Getting the type of 'write_script' (line 57)
        write_script_41476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'write_script', False)
        # Calling write_script(args, kwargs) (line 57)
        write_script_call_result_41480 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), write_script_41476, *[str_41477, str_41478], **kwargs_41479)
        
        
        # Assigning a Call to a Name (line 61):
        
        # Call to mkdtemp(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_41483 = {}
        # Getting the type of 'self' (line 61)
        self_41481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 61)
        mkdtemp_41482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 17), self_41481, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 61)
        mkdtemp_call_result_41484 = invoke(stypy.reporting.localization.Localization(__file__, 61, 17), mkdtemp_41482, *[], **kwargs_41483)
        
        # Assigning a type to the variable 'target' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'target', mkdtemp_call_result_41484)
        
        # Assigning a Call to a Name (line 62):
        
        # Call to Distribution(...): (line 62)
        # Processing the call keyword arguments (line 62)
        kwargs_41486 = {}
        # Getting the type of 'Distribution' (line 62)
        Distribution_41485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 62)
        Distribution_call_result_41487 = invoke(stypy.reporting.localization.Localization(__file__, 62, 15), Distribution_41485, *[], **kwargs_41486)
        
        # Assigning a type to the variable 'dist' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'dist', Distribution_call_result_41487)
        
        # Assigning a Call to a Subscript (line 63):
        
        # Call to DummyCommand(...): (line 63)
        # Processing the call keyword arguments (line 63)
        # Getting the type of 'source' (line 63)
        source_41490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 71), 'source', False)
        keyword_41491 = source_41490
        kwargs_41492 = {'build_scripts': keyword_41491}
        # Getting the type of 'support' (line 63)
        support_41488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 36), 'support', False)
        # Obtaining the member 'DummyCommand' of a type (line 63)
        DummyCommand_41489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 36), support_41488, 'DummyCommand')
        # Calling DummyCommand(args, kwargs) (line 63)
        DummyCommand_call_result_41493 = invoke(stypy.reporting.localization.Localization(__file__, 63, 36), DummyCommand_41489, *[], **kwargs_41492)
        
        # Getting the type of 'dist' (line 63)
        dist_41494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'dist')
        # Obtaining the member 'command_obj' of a type (line 63)
        command_obj_41495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), dist_41494, 'command_obj')
        str_41496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 25), 'str', 'build')
        # Storing an element on a container (line 63)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 8), command_obj_41495, (str_41496, DummyCommand_call_result_41493))
        
        # Assigning a Call to a Subscript (line 64):
        
        # Call to DummyCommand(...): (line 64)
        # Processing the call keyword arguments (line 64)
        # Getting the type of 'target' (line 65)
        target_41499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'target', False)
        keyword_41500 = target_41499
        int_41501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 18), 'int')
        keyword_41502 = int_41501
        int_41503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 23), 'int')
        keyword_41504 = int_41503
        kwargs_41505 = {'skip_build': keyword_41504, 'install_scripts': keyword_41500, 'force': keyword_41502}
        # Getting the type of 'support' (line 64)
        support_41497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 38), 'support', False)
        # Obtaining the member 'DummyCommand' of a type (line 64)
        DummyCommand_41498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 38), support_41497, 'DummyCommand')
        # Calling DummyCommand(args, kwargs) (line 64)
        DummyCommand_call_result_41506 = invoke(stypy.reporting.localization.Localization(__file__, 64, 38), DummyCommand_41498, *[], **kwargs_41505)
        
        # Getting the type of 'dist' (line 64)
        dist_41507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'dist')
        # Obtaining the member 'command_obj' of a type (line 64)
        command_obj_41508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), dist_41507, 'command_obj')
        str_41509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'str', 'install')
        # Storing an element on a container (line 64)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 8), command_obj_41508, (str_41509, DummyCommand_call_result_41506))
        
        # Assigning a Call to a Name (line 69):
        
        # Call to install_scripts(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'dist' (line 69)
        dist_41511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 30), 'dist', False)
        # Processing the call keyword arguments (line 69)
        kwargs_41512 = {}
        # Getting the type of 'install_scripts' (line 69)
        install_scripts_41510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'install_scripts', False)
        # Calling install_scripts(args, kwargs) (line 69)
        install_scripts_call_result_41513 = invoke(stypy.reporting.localization.Localization(__file__, 69, 14), install_scripts_41510, *[dist_41511], **kwargs_41512)
        
        # Assigning a type to the variable 'cmd' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'cmd', install_scripts_call_result_41513)
        
        # Call to finalize_options(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_41516 = {}
        # Getting the type of 'cmd' (line 70)
        cmd_41514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 70)
        finalize_options_41515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), cmd_41514, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 70)
        finalize_options_call_result_41517 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), finalize_options_41515, *[], **kwargs_41516)
        
        
        # Call to run(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_41520 = {}
        # Getting the type of 'cmd' (line 71)
        cmd_41518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 71)
        run_41519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), cmd_41518, 'run')
        # Calling run(args, kwargs) (line 71)
        run_call_result_41521 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), run_41519, *[], **kwargs_41520)
        
        
        # Assigning a Call to a Name (line 73):
        
        # Call to listdir(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'target' (line 73)
        target_41524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 31), 'target', False)
        # Processing the call keyword arguments (line 73)
        kwargs_41525 = {}
        # Getting the type of 'os' (line 73)
        os_41522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'os', False)
        # Obtaining the member 'listdir' of a type (line 73)
        listdir_41523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 20), os_41522, 'listdir')
        # Calling listdir(args, kwargs) (line 73)
        listdir_call_result_41526 = invoke(stypy.reporting.localization.Localization(__file__, 73, 20), listdir_41523, *[target_41524], **kwargs_41525)
        
        # Assigning a type to the variable 'installed' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'installed', listdir_call_result_41526)
        
        # Getting the type of 'expected' (line 74)
        expected_41527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'expected')
        # Testing the type of a for loop iterable (line 74)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 8), expected_41527)
        # Getting the type of the for loop variable (line 74)
        for_loop_var_41528 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 8), expected_41527)
        # Assigning a type to the variable 'name' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'name', for_loop_var_41528)
        # SSA begins for a for statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertIn(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'name' (line 75)
        name_41531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'name', False)
        # Getting the type of 'installed' (line 75)
        installed_41532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'installed', False)
        # Processing the call keyword arguments (line 75)
        kwargs_41533 = {}
        # Getting the type of 'self' (line 75)
        self_41529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 75)
        assertIn_41530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), self_41529, 'assertIn')
        # Calling assertIn(args, kwargs) (line 75)
        assertIn_call_result_41534 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), assertIn_41530, *[name_41531, installed_41532], **kwargs_41533)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_installation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_installation' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_41535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41535)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_installation'
        return stypy_return_type_41535


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallScriptsTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'InstallScriptsTestCase' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'InstallScriptsTestCase', InstallScriptsTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 78, 0, False)
    
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

    
    # Call to makeSuite(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'InstallScriptsTestCase' (line 79)
    InstallScriptsTestCase_41538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'InstallScriptsTestCase', False)
    # Processing the call keyword arguments (line 79)
    kwargs_41539 = {}
    # Getting the type of 'unittest' (line 79)
    unittest_41536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 79)
    makeSuite_41537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), unittest_41536, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 79)
    makeSuite_call_result_41540 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), makeSuite_41537, *[InstallScriptsTestCase_41538], **kwargs_41539)
    
    # Assigning a type to the variable 'stypy_return_type' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type', makeSuite_call_result_41540)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_41541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_41541)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_41541

# Assigning a type to the variable 'test_suite' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 82)
    # Processing the call arguments (line 82)
    
    # Call to test_suite(...): (line 82)
    # Processing the call keyword arguments (line 82)
    kwargs_41544 = {}
    # Getting the type of 'test_suite' (line 82)
    test_suite_41543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 82)
    test_suite_call_result_41545 = invoke(stypy.reporting.localization.Localization(__file__, 82, 17), test_suite_41543, *[], **kwargs_41544)
    
    # Processing the call keyword arguments (line 82)
    kwargs_41546 = {}
    # Getting the type of 'run_unittest' (line 82)
    run_unittest_41542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 82)
    run_unittest_call_result_41547 = invoke(stypy.reporting.localization.Localization(__file__, 82, 4), run_unittest_41542, *[test_suite_call_result_41545], **kwargs_41546)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
