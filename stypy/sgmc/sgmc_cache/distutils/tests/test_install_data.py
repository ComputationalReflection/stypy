
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.install_data.'''
2: import sys
3: import os
4: import unittest
5: import getpass
6: 
7: from distutils.command.install_data import install_data
8: from distutils.tests import support
9: from test.test_support import run_unittest
10: 
11: class InstallDataTestCase(support.TempdirManager,
12:                           support.LoggingSilencer,
13:                           support.EnvironGuard,
14:                           unittest.TestCase):
15: 
16:     def test_simple_run(self):
17:         pkg_dir, dist = self.create_dist()
18:         cmd = install_data(dist)
19:         cmd.install_dir = inst = os.path.join(pkg_dir, 'inst')
20: 
21:         # data_files can contain
22:         #  - simple files
23:         #  - a tuple with a path, and a list of file
24:         one = os.path.join(pkg_dir, 'one')
25:         self.write_file(one, 'xxx')
26:         inst2 = os.path.join(pkg_dir, 'inst2')
27:         two = os.path.join(pkg_dir, 'two')
28:         self.write_file(two, 'xxx')
29: 
30:         cmd.data_files = [one, (inst2, [two])]
31:         self.assertEqual(cmd.get_inputs(), [one, (inst2, [two])])
32: 
33:         # let's run the command
34:         cmd.ensure_finalized()
35:         cmd.run()
36: 
37:         # let's check the result
38:         self.assertEqual(len(cmd.get_outputs()), 2)
39:         rtwo = os.path.split(two)[-1]
40:         self.assertTrue(os.path.exists(os.path.join(inst2, rtwo)))
41:         rone = os.path.split(one)[-1]
42:         self.assertTrue(os.path.exists(os.path.join(inst, rone)))
43:         cmd.outfiles = []
44: 
45:         # let's try with warn_dir one
46:         cmd.warn_dir = 1
47:         cmd.ensure_finalized()
48:         cmd.run()
49: 
50:         # let's check the result
51:         self.assertEqual(len(cmd.get_outputs()), 2)
52:         self.assertTrue(os.path.exists(os.path.join(inst2, rtwo)))
53:         self.assertTrue(os.path.exists(os.path.join(inst, rone)))
54:         cmd.outfiles = []
55: 
56:         # now using root and empty dir
57:         cmd.root = os.path.join(pkg_dir, 'root')
58:         inst3 = os.path.join(cmd.install_dir, 'inst3')
59:         inst4 = os.path.join(pkg_dir, 'inst4')
60:         three = os.path.join(cmd.install_dir, 'three')
61:         self.write_file(three, 'xx')
62:         cmd.data_files = [one, (inst2, [two]),
63:                           ('inst3', [three]),
64:                           (inst4, [])]
65:         cmd.ensure_finalized()
66:         cmd.run()
67: 
68:         # let's check the result
69:         self.assertEqual(len(cmd.get_outputs()), 4)
70:         self.assertTrue(os.path.exists(os.path.join(inst2, rtwo)))
71:         self.assertTrue(os.path.exists(os.path.join(inst, rone)))
72: 
73: def test_suite():
74:     return unittest.makeSuite(InstallDataTestCase)
75: 
76: if __name__ == "__main__":
77:     run_unittest(test_suite())
78: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_40477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.install_data.')
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

# 'from distutils.command.install_data import install_data' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_40478 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.install_data')

if (type(import_40478) is not StypyTypeError):

    if (import_40478 != 'pyd_module'):
        __import__(import_40478)
        sys_modules_40479 = sys.modules[import_40478]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.install_data', sys_modules_40479.module_type_store, module_type_store, ['install_data'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_40479, sys_modules_40479.module_type_store, module_type_store)
    else:
        from distutils.command.install_data import install_data

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.install_data', None, module_type_store, ['install_data'], [install_data])

else:
    # Assigning a type to the variable 'distutils.command.install_data' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.install_data', import_40478)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.tests import support' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_40480 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests')

if (type(import_40480) is not StypyTypeError):

    if (import_40480 != 'pyd_module'):
        __import__(import_40480)
        sys_modules_40481 = sys.modules[import_40480]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', sys_modules_40481.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_40481, sys_modules_40481.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', import_40480)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from test.test_support import run_unittest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_40482 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support')

if (type(import_40482) is not StypyTypeError):

    if (import_40482 != 'pyd_module'):
        __import__(import_40482)
        sys_modules_40483 = sys.modules[import_40482]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', sys_modules_40483.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_40483, sys_modules_40483.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', import_40482)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'InstallDataTestCase' class
# Getting the type of 'support' (line 11)
support_40484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 26), 'support')
# Obtaining the member 'TempdirManager' of a type (line 11)
TempdirManager_40485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 26), support_40484, 'TempdirManager')
# Getting the type of 'support' (line 12)
support_40486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 26), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 12)
LoggingSilencer_40487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 26), support_40486, 'LoggingSilencer')
# Getting the type of 'support' (line 13)
support_40488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 26), 'support')
# Obtaining the member 'EnvironGuard' of a type (line 13)
EnvironGuard_40489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 26), support_40488, 'EnvironGuard')
# Getting the type of 'unittest' (line 14)
unittest_40490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 26), 'unittest')
# Obtaining the member 'TestCase' of a type (line 14)
TestCase_40491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 26), unittest_40490, 'TestCase')

class InstallDataTestCase(TempdirManager_40485, LoggingSilencer_40487, EnvironGuard_40489, TestCase_40491, ):

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
        InstallDataTestCase.test_simple_run.__dict__.__setitem__('stypy_localization', localization)
        InstallDataTestCase.test_simple_run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallDataTestCase.test_simple_run.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallDataTestCase.test_simple_run.__dict__.__setitem__('stypy_function_name', 'InstallDataTestCase.test_simple_run')
        InstallDataTestCase.test_simple_run.__dict__.__setitem__('stypy_param_names_list', [])
        InstallDataTestCase.test_simple_run.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallDataTestCase.test_simple_run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallDataTestCase.test_simple_run.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallDataTestCase.test_simple_run.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallDataTestCase.test_simple_run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallDataTestCase.test_simple_run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallDataTestCase.test_simple_run', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 17):
        
        # Assigning a Subscript to a Name (line 17):
        
        # Obtaining the type of the subscript
        int_40492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'int')
        
        # Call to create_dist(...): (line 17)
        # Processing the call keyword arguments (line 17)
        kwargs_40495 = {}
        # Getting the type of 'self' (line 17)
        self_40493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 17)
        create_dist_40494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 24), self_40493, 'create_dist')
        # Calling create_dist(args, kwargs) (line 17)
        create_dist_call_result_40496 = invoke(stypy.reporting.localization.Localization(__file__, 17, 24), create_dist_40494, *[], **kwargs_40495)
        
        # Obtaining the member '__getitem__' of a type (line 17)
        getitem___40497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), create_dist_call_result_40496, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 17)
        subscript_call_result_40498 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), getitem___40497, int_40492)
        
        # Assigning a type to the variable 'tuple_var_assignment_40475' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_var_assignment_40475', subscript_call_result_40498)
        
        # Assigning a Subscript to a Name (line 17):
        
        # Obtaining the type of the subscript
        int_40499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'int')
        
        # Call to create_dist(...): (line 17)
        # Processing the call keyword arguments (line 17)
        kwargs_40502 = {}
        # Getting the type of 'self' (line 17)
        self_40500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 17)
        create_dist_40501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 24), self_40500, 'create_dist')
        # Calling create_dist(args, kwargs) (line 17)
        create_dist_call_result_40503 = invoke(stypy.reporting.localization.Localization(__file__, 17, 24), create_dist_40501, *[], **kwargs_40502)
        
        # Obtaining the member '__getitem__' of a type (line 17)
        getitem___40504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), create_dist_call_result_40503, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 17)
        subscript_call_result_40505 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), getitem___40504, int_40499)
        
        # Assigning a type to the variable 'tuple_var_assignment_40476' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_var_assignment_40476', subscript_call_result_40505)
        
        # Assigning a Name to a Name (line 17):
        # Getting the type of 'tuple_var_assignment_40475' (line 17)
        tuple_var_assignment_40475_40506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_var_assignment_40475')
        # Assigning a type to the variable 'pkg_dir' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'pkg_dir', tuple_var_assignment_40475_40506)
        
        # Assigning a Name to a Name (line 17):
        # Getting the type of 'tuple_var_assignment_40476' (line 17)
        tuple_var_assignment_40476_40507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_var_assignment_40476')
        # Assigning a type to the variable 'dist' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'dist', tuple_var_assignment_40476_40507)
        
        # Assigning a Call to a Name (line 18):
        
        # Assigning a Call to a Name (line 18):
        
        # Call to install_data(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'dist' (line 18)
        dist_40509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 27), 'dist', False)
        # Processing the call keyword arguments (line 18)
        kwargs_40510 = {}
        # Getting the type of 'install_data' (line 18)
        install_data_40508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'install_data', False)
        # Calling install_data(args, kwargs) (line 18)
        install_data_call_result_40511 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), install_data_40508, *[dist_40509], **kwargs_40510)
        
        # Assigning a type to the variable 'cmd' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'cmd', install_data_call_result_40511)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Name (line 19):
        
        # Call to join(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'pkg_dir' (line 19)
        pkg_dir_40515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 46), 'pkg_dir', False)
        str_40516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 55), 'str', 'inst')
        # Processing the call keyword arguments (line 19)
        kwargs_40517 = {}
        # Getting the type of 'os' (line 19)
        os_40512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 33), 'os', False)
        # Obtaining the member 'path' of a type (line 19)
        path_40513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 33), os_40512, 'path')
        # Obtaining the member 'join' of a type (line 19)
        join_40514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 33), path_40513, 'join')
        # Calling join(args, kwargs) (line 19)
        join_call_result_40518 = invoke(stypy.reporting.localization.Localization(__file__, 19, 33), join_40514, *[pkg_dir_40515, str_40516], **kwargs_40517)
        
        # Assigning a type to the variable 'inst' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 26), 'inst', join_call_result_40518)
        
        # Assigning a Name to a Attribute (line 19):
        # Getting the type of 'inst' (line 19)
        inst_40519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 26), 'inst')
        # Getting the type of 'cmd' (line 19)
        cmd_40520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'cmd')
        # Setting the type of the member 'install_dir' of a type (line 19)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), cmd_40520, 'install_dir', inst_40519)
        
        # Assigning a Call to a Name (line 24):
        
        # Assigning a Call to a Name (line 24):
        
        # Call to join(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'pkg_dir' (line 24)
        pkg_dir_40524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 27), 'pkg_dir', False)
        str_40525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 36), 'str', 'one')
        # Processing the call keyword arguments (line 24)
        kwargs_40526 = {}
        # Getting the type of 'os' (line 24)
        os_40521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 14), 'os', False)
        # Obtaining the member 'path' of a type (line 24)
        path_40522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 14), os_40521, 'path')
        # Obtaining the member 'join' of a type (line 24)
        join_40523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 14), path_40522, 'join')
        # Calling join(args, kwargs) (line 24)
        join_call_result_40527 = invoke(stypy.reporting.localization.Localization(__file__, 24, 14), join_40523, *[pkg_dir_40524, str_40525], **kwargs_40526)
        
        # Assigning a type to the variable 'one' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'one', join_call_result_40527)
        
        # Call to write_file(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'one' (line 25)
        one_40530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'one', False)
        str_40531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'str', 'xxx')
        # Processing the call keyword arguments (line 25)
        kwargs_40532 = {}
        # Getting the type of 'self' (line 25)
        self_40528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 25)
        write_file_40529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_40528, 'write_file')
        # Calling write_file(args, kwargs) (line 25)
        write_file_call_result_40533 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), write_file_40529, *[one_40530, str_40531], **kwargs_40532)
        
        
        # Assigning a Call to a Name (line 26):
        
        # Assigning a Call to a Name (line 26):
        
        # Call to join(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'pkg_dir' (line 26)
        pkg_dir_40537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'pkg_dir', False)
        str_40538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 38), 'str', 'inst2')
        # Processing the call keyword arguments (line 26)
        kwargs_40539 = {}
        # Getting the type of 'os' (line 26)
        os_40534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 26)
        path_40535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), os_40534, 'path')
        # Obtaining the member 'join' of a type (line 26)
        join_40536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), path_40535, 'join')
        # Calling join(args, kwargs) (line 26)
        join_call_result_40540 = invoke(stypy.reporting.localization.Localization(__file__, 26, 16), join_40536, *[pkg_dir_40537, str_40538], **kwargs_40539)
        
        # Assigning a type to the variable 'inst2' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'inst2', join_call_result_40540)
        
        # Assigning a Call to a Name (line 27):
        
        # Assigning a Call to a Name (line 27):
        
        # Call to join(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'pkg_dir' (line 27)
        pkg_dir_40544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 27), 'pkg_dir', False)
        str_40545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 36), 'str', 'two')
        # Processing the call keyword arguments (line 27)
        kwargs_40546 = {}
        # Getting the type of 'os' (line 27)
        os_40541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 'os', False)
        # Obtaining the member 'path' of a type (line 27)
        path_40542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 14), os_40541, 'path')
        # Obtaining the member 'join' of a type (line 27)
        join_40543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 14), path_40542, 'join')
        # Calling join(args, kwargs) (line 27)
        join_call_result_40547 = invoke(stypy.reporting.localization.Localization(__file__, 27, 14), join_40543, *[pkg_dir_40544, str_40545], **kwargs_40546)
        
        # Assigning a type to the variable 'two' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'two', join_call_result_40547)
        
        # Call to write_file(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'two' (line 28)
        two_40550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'two', False)
        str_40551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 29), 'str', 'xxx')
        # Processing the call keyword arguments (line 28)
        kwargs_40552 = {}
        # Getting the type of 'self' (line 28)
        self_40548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 28)
        write_file_40549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_40548, 'write_file')
        # Calling write_file(args, kwargs) (line 28)
        write_file_call_result_40553 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), write_file_40549, *[two_40550, str_40551], **kwargs_40552)
        
        
        # Assigning a List to a Attribute (line 30):
        
        # Assigning a List to a Attribute (line 30):
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_40554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        # Getting the type of 'one' (line 30)
        one_40555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), list_40554, one_40555)
        # Adding element type (line 30)
        
        # Obtaining an instance of the builtin type 'tuple' (line 30)
        tuple_40556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 30)
        # Adding element type (line 30)
        # Getting the type of 'inst2' (line 30)
        inst2_40557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 32), 'inst2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 32), tuple_40556, inst2_40557)
        # Adding element type (line 30)
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_40558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        # Getting the type of 'two' (line 30)
        two_40559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 40), 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 39), list_40558, two_40559)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 32), tuple_40556, list_40558)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), list_40554, tuple_40556)
        
        # Getting the type of 'cmd' (line 30)
        cmd_40560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'cmd')
        # Setting the type of the member 'data_files' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), cmd_40560, 'data_files', list_40554)
        
        # Call to assertEqual(...): (line 31)
        # Processing the call arguments (line 31)
        
        # Call to get_inputs(...): (line 31)
        # Processing the call keyword arguments (line 31)
        kwargs_40565 = {}
        # Getting the type of 'cmd' (line 31)
        cmd_40563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'cmd', False)
        # Obtaining the member 'get_inputs' of a type (line 31)
        get_inputs_40564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 25), cmd_40563, 'get_inputs')
        # Calling get_inputs(args, kwargs) (line 31)
        get_inputs_call_result_40566 = invoke(stypy.reporting.localization.Localization(__file__, 31, 25), get_inputs_40564, *[], **kwargs_40565)
        
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_40567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        # Getting the type of 'one' (line 31)
        one_40568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 44), 'one', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 43), list_40567, one_40568)
        # Adding element type (line 31)
        
        # Obtaining an instance of the builtin type 'tuple' (line 31)
        tuple_40569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 31)
        # Adding element type (line 31)
        # Getting the type of 'inst2' (line 31)
        inst2_40570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 50), 'inst2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 50), tuple_40569, inst2_40570)
        # Adding element type (line 31)
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_40571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        # Getting the type of 'two' (line 31)
        two_40572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 58), 'two', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 57), list_40571, two_40572)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 50), tuple_40569, list_40571)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 43), list_40567, tuple_40569)
        
        # Processing the call keyword arguments (line 31)
        kwargs_40573 = {}
        # Getting the type of 'self' (line 31)
        self_40561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 31)
        assertEqual_40562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_40561, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 31)
        assertEqual_call_result_40574 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), assertEqual_40562, *[get_inputs_call_result_40566, list_40567], **kwargs_40573)
        
        
        # Call to ensure_finalized(...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_40577 = {}
        # Getting the type of 'cmd' (line 34)
        cmd_40575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 34)
        ensure_finalized_40576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), cmd_40575, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 34)
        ensure_finalized_call_result_40578 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), ensure_finalized_40576, *[], **kwargs_40577)
        
        
        # Call to run(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_40581 = {}
        # Getting the type of 'cmd' (line 35)
        cmd_40579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 35)
        run_40580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), cmd_40579, 'run')
        # Calling run(args, kwargs) (line 35)
        run_call_result_40582 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), run_40580, *[], **kwargs_40581)
        
        
        # Call to assertEqual(...): (line 38)
        # Processing the call arguments (line 38)
        
        # Call to len(...): (line 38)
        # Processing the call arguments (line 38)
        
        # Call to get_outputs(...): (line 38)
        # Processing the call keyword arguments (line 38)
        kwargs_40588 = {}
        # Getting the type of 'cmd' (line 38)
        cmd_40586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'cmd', False)
        # Obtaining the member 'get_outputs' of a type (line 38)
        get_outputs_40587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 29), cmd_40586, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 38)
        get_outputs_call_result_40589 = invoke(stypy.reporting.localization.Localization(__file__, 38, 29), get_outputs_40587, *[], **kwargs_40588)
        
        # Processing the call keyword arguments (line 38)
        kwargs_40590 = {}
        # Getting the type of 'len' (line 38)
        len_40585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'len', False)
        # Calling len(args, kwargs) (line 38)
        len_call_result_40591 = invoke(stypy.reporting.localization.Localization(__file__, 38, 25), len_40585, *[get_outputs_call_result_40589], **kwargs_40590)
        
        int_40592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 49), 'int')
        # Processing the call keyword arguments (line 38)
        kwargs_40593 = {}
        # Getting the type of 'self' (line 38)
        self_40583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 38)
        assertEqual_40584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_40583, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 38)
        assertEqual_call_result_40594 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), assertEqual_40584, *[len_call_result_40591, int_40592], **kwargs_40593)
        
        
        # Assigning a Subscript to a Name (line 39):
        
        # Assigning a Subscript to a Name (line 39):
        
        # Obtaining the type of the subscript
        int_40595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 34), 'int')
        
        # Call to split(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'two' (line 39)
        two_40599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'two', False)
        # Processing the call keyword arguments (line 39)
        kwargs_40600 = {}
        # Getting the type of 'os' (line 39)
        os_40596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 39)
        path_40597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 15), os_40596, 'path')
        # Obtaining the member 'split' of a type (line 39)
        split_40598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 15), path_40597, 'split')
        # Calling split(args, kwargs) (line 39)
        split_call_result_40601 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), split_40598, *[two_40599], **kwargs_40600)
        
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___40602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 15), split_call_result_40601, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_40603 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), getitem___40602, int_40595)
        
        # Assigning a type to the variable 'rtwo' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'rtwo', subscript_call_result_40603)
        
        # Call to assertTrue(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Call to exists(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Call to join(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'inst2' (line 40)
        inst2_40612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 52), 'inst2', False)
        # Getting the type of 'rtwo' (line 40)
        rtwo_40613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 59), 'rtwo', False)
        # Processing the call keyword arguments (line 40)
        kwargs_40614 = {}
        # Getting the type of 'os' (line 40)
        os_40609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 39), 'os', False)
        # Obtaining the member 'path' of a type (line 40)
        path_40610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 39), os_40609, 'path')
        # Obtaining the member 'join' of a type (line 40)
        join_40611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 39), path_40610, 'join')
        # Calling join(args, kwargs) (line 40)
        join_call_result_40615 = invoke(stypy.reporting.localization.Localization(__file__, 40, 39), join_40611, *[inst2_40612, rtwo_40613], **kwargs_40614)
        
        # Processing the call keyword arguments (line 40)
        kwargs_40616 = {}
        # Getting the type of 'os' (line 40)
        os_40606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 40)
        path_40607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 24), os_40606, 'path')
        # Obtaining the member 'exists' of a type (line 40)
        exists_40608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 24), path_40607, 'exists')
        # Calling exists(args, kwargs) (line 40)
        exists_call_result_40617 = invoke(stypy.reporting.localization.Localization(__file__, 40, 24), exists_40608, *[join_call_result_40615], **kwargs_40616)
        
        # Processing the call keyword arguments (line 40)
        kwargs_40618 = {}
        # Getting the type of 'self' (line 40)
        self_40604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 40)
        assertTrue_40605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_40604, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 40)
        assertTrue_call_result_40619 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), assertTrue_40605, *[exists_call_result_40617], **kwargs_40618)
        
        
        # Assigning a Subscript to a Name (line 41):
        
        # Assigning a Subscript to a Name (line 41):
        
        # Obtaining the type of the subscript
        int_40620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 34), 'int')
        
        # Call to split(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'one' (line 41)
        one_40624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 29), 'one', False)
        # Processing the call keyword arguments (line 41)
        kwargs_40625 = {}
        # Getting the type of 'os' (line 41)
        os_40621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 41)
        path_40622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), os_40621, 'path')
        # Obtaining the member 'split' of a type (line 41)
        split_40623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), path_40622, 'split')
        # Calling split(args, kwargs) (line 41)
        split_call_result_40626 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), split_40623, *[one_40624], **kwargs_40625)
        
        # Obtaining the member '__getitem__' of a type (line 41)
        getitem___40627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), split_call_result_40626, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 41)
        subscript_call_result_40628 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), getitem___40627, int_40620)
        
        # Assigning a type to the variable 'rone' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'rone', subscript_call_result_40628)
        
        # Call to assertTrue(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to exists(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to join(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'inst' (line 42)
        inst_40637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 52), 'inst', False)
        # Getting the type of 'rone' (line 42)
        rone_40638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 58), 'rone', False)
        # Processing the call keyword arguments (line 42)
        kwargs_40639 = {}
        # Getting the type of 'os' (line 42)
        os_40634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 'os', False)
        # Obtaining the member 'path' of a type (line 42)
        path_40635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 39), os_40634, 'path')
        # Obtaining the member 'join' of a type (line 42)
        join_40636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 39), path_40635, 'join')
        # Calling join(args, kwargs) (line 42)
        join_call_result_40640 = invoke(stypy.reporting.localization.Localization(__file__, 42, 39), join_40636, *[inst_40637, rone_40638], **kwargs_40639)
        
        # Processing the call keyword arguments (line 42)
        kwargs_40641 = {}
        # Getting the type of 'os' (line 42)
        os_40631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 42)
        path_40632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), os_40631, 'path')
        # Obtaining the member 'exists' of a type (line 42)
        exists_40633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), path_40632, 'exists')
        # Calling exists(args, kwargs) (line 42)
        exists_call_result_40642 = invoke(stypy.reporting.localization.Localization(__file__, 42, 24), exists_40633, *[join_call_result_40640], **kwargs_40641)
        
        # Processing the call keyword arguments (line 42)
        kwargs_40643 = {}
        # Getting the type of 'self' (line 42)
        self_40629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 42)
        assertTrue_40630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_40629, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 42)
        assertTrue_call_result_40644 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assertTrue_40630, *[exists_call_result_40642], **kwargs_40643)
        
        
        # Assigning a List to a Attribute (line 43):
        
        # Assigning a List to a Attribute (line 43):
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_40645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        
        # Getting the type of 'cmd' (line 43)
        cmd_40646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'cmd')
        # Setting the type of the member 'outfiles' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), cmd_40646, 'outfiles', list_40645)
        
        # Assigning a Num to a Attribute (line 46):
        
        # Assigning a Num to a Attribute (line 46):
        int_40647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'int')
        # Getting the type of 'cmd' (line 46)
        cmd_40648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'cmd')
        # Setting the type of the member 'warn_dir' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), cmd_40648, 'warn_dir', int_40647)
        
        # Call to ensure_finalized(...): (line 47)
        # Processing the call keyword arguments (line 47)
        kwargs_40651 = {}
        # Getting the type of 'cmd' (line 47)
        cmd_40649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 47)
        ensure_finalized_40650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), cmd_40649, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 47)
        ensure_finalized_call_result_40652 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), ensure_finalized_40650, *[], **kwargs_40651)
        
        
        # Call to run(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_40655 = {}
        # Getting the type of 'cmd' (line 48)
        cmd_40653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 48)
        run_40654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), cmd_40653, 'run')
        # Calling run(args, kwargs) (line 48)
        run_call_result_40656 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), run_40654, *[], **kwargs_40655)
        
        
        # Call to assertEqual(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to len(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to get_outputs(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_40662 = {}
        # Getting the type of 'cmd' (line 51)
        cmd_40660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'cmd', False)
        # Obtaining the member 'get_outputs' of a type (line 51)
        get_outputs_40661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 29), cmd_40660, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 51)
        get_outputs_call_result_40663 = invoke(stypy.reporting.localization.Localization(__file__, 51, 29), get_outputs_40661, *[], **kwargs_40662)
        
        # Processing the call keyword arguments (line 51)
        kwargs_40664 = {}
        # Getting the type of 'len' (line 51)
        len_40659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'len', False)
        # Calling len(args, kwargs) (line 51)
        len_call_result_40665 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), len_40659, *[get_outputs_call_result_40663], **kwargs_40664)
        
        int_40666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 49), 'int')
        # Processing the call keyword arguments (line 51)
        kwargs_40667 = {}
        # Getting the type of 'self' (line 51)
        self_40657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 51)
        assertEqual_40658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_40657, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 51)
        assertEqual_call_result_40668 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assertEqual_40658, *[len_call_result_40665, int_40666], **kwargs_40667)
        
        
        # Call to assertTrue(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Call to exists(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Call to join(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'inst2' (line 52)
        inst2_40677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 52), 'inst2', False)
        # Getting the type of 'rtwo' (line 52)
        rtwo_40678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 59), 'rtwo', False)
        # Processing the call keyword arguments (line 52)
        kwargs_40679 = {}
        # Getting the type of 'os' (line 52)
        os_40674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 39), 'os', False)
        # Obtaining the member 'path' of a type (line 52)
        path_40675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 39), os_40674, 'path')
        # Obtaining the member 'join' of a type (line 52)
        join_40676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 39), path_40675, 'join')
        # Calling join(args, kwargs) (line 52)
        join_call_result_40680 = invoke(stypy.reporting.localization.Localization(__file__, 52, 39), join_40676, *[inst2_40677, rtwo_40678], **kwargs_40679)
        
        # Processing the call keyword arguments (line 52)
        kwargs_40681 = {}
        # Getting the type of 'os' (line 52)
        os_40671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 52)
        path_40672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 24), os_40671, 'path')
        # Obtaining the member 'exists' of a type (line 52)
        exists_40673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 24), path_40672, 'exists')
        # Calling exists(args, kwargs) (line 52)
        exists_call_result_40682 = invoke(stypy.reporting.localization.Localization(__file__, 52, 24), exists_40673, *[join_call_result_40680], **kwargs_40681)
        
        # Processing the call keyword arguments (line 52)
        kwargs_40683 = {}
        # Getting the type of 'self' (line 52)
        self_40669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 52)
        assertTrue_40670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_40669, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 52)
        assertTrue_call_result_40684 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), assertTrue_40670, *[exists_call_result_40682], **kwargs_40683)
        
        
        # Call to assertTrue(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to exists(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to join(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'inst' (line 53)
        inst_40693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 52), 'inst', False)
        # Getting the type of 'rone' (line 53)
        rone_40694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 58), 'rone', False)
        # Processing the call keyword arguments (line 53)
        kwargs_40695 = {}
        # Getting the type of 'os' (line 53)
        os_40690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 39), 'os', False)
        # Obtaining the member 'path' of a type (line 53)
        path_40691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 39), os_40690, 'path')
        # Obtaining the member 'join' of a type (line 53)
        join_40692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 39), path_40691, 'join')
        # Calling join(args, kwargs) (line 53)
        join_call_result_40696 = invoke(stypy.reporting.localization.Localization(__file__, 53, 39), join_40692, *[inst_40693, rone_40694], **kwargs_40695)
        
        # Processing the call keyword arguments (line 53)
        kwargs_40697 = {}
        # Getting the type of 'os' (line 53)
        os_40687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 53)
        path_40688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 24), os_40687, 'path')
        # Obtaining the member 'exists' of a type (line 53)
        exists_40689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 24), path_40688, 'exists')
        # Calling exists(args, kwargs) (line 53)
        exists_call_result_40698 = invoke(stypy.reporting.localization.Localization(__file__, 53, 24), exists_40689, *[join_call_result_40696], **kwargs_40697)
        
        # Processing the call keyword arguments (line 53)
        kwargs_40699 = {}
        # Getting the type of 'self' (line 53)
        self_40685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 53)
        assertTrue_40686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_40685, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 53)
        assertTrue_call_result_40700 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), assertTrue_40686, *[exists_call_result_40698], **kwargs_40699)
        
        
        # Assigning a List to a Attribute (line 54):
        
        # Assigning a List to a Attribute (line 54):
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_40701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        
        # Getting the type of 'cmd' (line 54)
        cmd_40702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'cmd')
        # Setting the type of the member 'outfiles' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), cmd_40702, 'outfiles', list_40701)
        
        # Assigning a Call to a Attribute (line 57):
        
        # Assigning a Call to a Attribute (line 57):
        
        # Call to join(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'pkg_dir' (line 57)
        pkg_dir_40706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 32), 'pkg_dir', False)
        str_40707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 41), 'str', 'root')
        # Processing the call keyword arguments (line 57)
        kwargs_40708 = {}
        # Getting the type of 'os' (line 57)
        os_40703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 57)
        path_40704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 19), os_40703, 'path')
        # Obtaining the member 'join' of a type (line 57)
        join_40705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 19), path_40704, 'join')
        # Calling join(args, kwargs) (line 57)
        join_call_result_40709 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), join_40705, *[pkg_dir_40706, str_40707], **kwargs_40708)
        
        # Getting the type of 'cmd' (line 57)
        cmd_40710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'cmd')
        # Setting the type of the member 'root' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), cmd_40710, 'root', join_call_result_40709)
        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Call to join(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'cmd' (line 58)
        cmd_40714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 29), 'cmd', False)
        # Obtaining the member 'install_dir' of a type (line 58)
        install_dir_40715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 29), cmd_40714, 'install_dir')
        str_40716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 46), 'str', 'inst3')
        # Processing the call keyword arguments (line 58)
        kwargs_40717 = {}
        # Getting the type of 'os' (line 58)
        os_40711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 58)
        path_40712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), os_40711, 'path')
        # Obtaining the member 'join' of a type (line 58)
        join_40713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), path_40712, 'join')
        # Calling join(args, kwargs) (line 58)
        join_call_result_40718 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), join_40713, *[install_dir_40715, str_40716], **kwargs_40717)
        
        # Assigning a type to the variable 'inst3' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'inst3', join_call_result_40718)
        
        # Assigning a Call to a Name (line 59):
        
        # Assigning a Call to a Name (line 59):
        
        # Call to join(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'pkg_dir' (line 59)
        pkg_dir_40722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'pkg_dir', False)
        str_40723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 38), 'str', 'inst4')
        # Processing the call keyword arguments (line 59)
        kwargs_40724 = {}
        # Getting the type of 'os' (line 59)
        os_40719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 59)
        path_40720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), os_40719, 'path')
        # Obtaining the member 'join' of a type (line 59)
        join_40721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), path_40720, 'join')
        # Calling join(args, kwargs) (line 59)
        join_call_result_40725 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), join_40721, *[pkg_dir_40722, str_40723], **kwargs_40724)
        
        # Assigning a type to the variable 'inst4' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'inst4', join_call_result_40725)
        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to join(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'cmd' (line 60)
        cmd_40729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'cmd', False)
        # Obtaining the member 'install_dir' of a type (line 60)
        install_dir_40730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 29), cmd_40729, 'install_dir')
        str_40731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 46), 'str', 'three')
        # Processing the call keyword arguments (line 60)
        kwargs_40732 = {}
        # Getting the type of 'os' (line 60)
        os_40726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 60)
        path_40727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), os_40726, 'path')
        # Obtaining the member 'join' of a type (line 60)
        join_40728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), path_40727, 'join')
        # Calling join(args, kwargs) (line 60)
        join_call_result_40733 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), join_40728, *[install_dir_40730, str_40731], **kwargs_40732)
        
        # Assigning a type to the variable 'three' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'three', join_call_result_40733)
        
        # Call to write_file(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'three' (line 61)
        three_40736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'three', False)
        str_40737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 31), 'str', 'xx')
        # Processing the call keyword arguments (line 61)
        kwargs_40738 = {}
        # Getting the type of 'self' (line 61)
        self_40734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 61)
        write_file_40735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_40734, 'write_file')
        # Calling write_file(args, kwargs) (line 61)
        write_file_call_result_40739 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), write_file_40735, *[three_40736, str_40737], **kwargs_40738)
        
        
        # Assigning a List to a Attribute (line 62):
        
        # Assigning a List to a Attribute (line 62):
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_40740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        # Getting the type of 'one' (line 62)
        one_40741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 25), list_40740, one_40741)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'tuple' (line 62)
        tuple_40742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 62)
        # Adding element type (line 62)
        # Getting the type of 'inst2' (line 62)
        inst2_40743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 32), 'inst2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 32), tuple_40742, inst2_40743)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_40744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        # Getting the type of 'two' (line 62)
        two_40745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 40), 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 39), list_40744, two_40745)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 32), tuple_40742, list_40744)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 25), list_40740, tuple_40742)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_40746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        str_40747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'str', 'inst3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 27), tuple_40746, str_40747)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_40748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        # Getting the type of 'three' (line 63)
        three_40749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'three')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 36), list_40748, three_40749)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 27), tuple_40746, list_40748)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 25), list_40740, tuple_40746)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'tuple' (line 64)
        tuple_40750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 64)
        # Adding element type (line 64)
        # Getting the type of 'inst4' (line 64)
        inst4_40751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'inst4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 27), tuple_40750, inst4_40751)
        # Adding element type (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_40752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 27), tuple_40750, list_40752)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 25), list_40740, tuple_40750)
        
        # Getting the type of 'cmd' (line 62)
        cmd_40753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'cmd')
        # Setting the type of the member 'data_files' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), cmd_40753, 'data_files', list_40740)
        
        # Call to ensure_finalized(...): (line 65)
        # Processing the call keyword arguments (line 65)
        kwargs_40756 = {}
        # Getting the type of 'cmd' (line 65)
        cmd_40754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 65)
        ensure_finalized_40755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), cmd_40754, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 65)
        ensure_finalized_call_result_40757 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), ensure_finalized_40755, *[], **kwargs_40756)
        
        
        # Call to run(...): (line 66)
        # Processing the call keyword arguments (line 66)
        kwargs_40760 = {}
        # Getting the type of 'cmd' (line 66)
        cmd_40758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 66)
        run_40759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), cmd_40758, 'run')
        # Calling run(args, kwargs) (line 66)
        run_call_result_40761 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), run_40759, *[], **kwargs_40760)
        
        
        # Call to assertEqual(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Call to len(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Call to get_outputs(...): (line 69)
        # Processing the call keyword arguments (line 69)
        kwargs_40767 = {}
        # Getting the type of 'cmd' (line 69)
        cmd_40765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'cmd', False)
        # Obtaining the member 'get_outputs' of a type (line 69)
        get_outputs_40766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 29), cmd_40765, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 69)
        get_outputs_call_result_40768 = invoke(stypy.reporting.localization.Localization(__file__, 69, 29), get_outputs_40766, *[], **kwargs_40767)
        
        # Processing the call keyword arguments (line 69)
        kwargs_40769 = {}
        # Getting the type of 'len' (line 69)
        len_40764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'len', False)
        # Calling len(args, kwargs) (line 69)
        len_call_result_40770 = invoke(stypy.reporting.localization.Localization(__file__, 69, 25), len_40764, *[get_outputs_call_result_40768], **kwargs_40769)
        
        int_40771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 49), 'int')
        # Processing the call keyword arguments (line 69)
        kwargs_40772 = {}
        # Getting the type of 'self' (line 69)
        self_40762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 69)
        assertEqual_40763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_40762, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 69)
        assertEqual_call_result_40773 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), assertEqual_40763, *[len_call_result_40770, int_40771], **kwargs_40772)
        
        
        # Call to assertTrue(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Call to exists(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Call to join(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'inst2' (line 70)
        inst2_40782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 52), 'inst2', False)
        # Getting the type of 'rtwo' (line 70)
        rtwo_40783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 59), 'rtwo', False)
        # Processing the call keyword arguments (line 70)
        kwargs_40784 = {}
        # Getting the type of 'os' (line 70)
        os_40779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 39), 'os', False)
        # Obtaining the member 'path' of a type (line 70)
        path_40780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 39), os_40779, 'path')
        # Obtaining the member 'join' of a type (line 70)
        join_40781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 39), path_40780, 'join')
        # Calling join(args, kwargs) (line 70)
        join_call_result_40785 = invoke(stypy.reporting.localization.Localization(__file__, 70, 39), join_40781, *[inst2_40782, rtwo_40783], **kwargs_40784)
        
        # Processing the call keyword arguments (line 70)
        kwargs_40786 = {}
        # Getting the type of 'os' (line 70)
        os_40776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 70)
        path_40777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 24), os_40776, 'path')
        # Obtaining the member 'exists' of a type (line 70)
        exists_40778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 24), path_40777, 'exists')
        # Calling exists(args, kwargs) (line 70)
        exists_call_result_40787 = invoke(stypy.reporting.localization.Localization(__file__, 70, 24), exists_40778, *[join_call_result_40785], **kwargs_40786)
        
        # Processing the call keyword arguments (line 70)
        kwargs_40788 = {}
        # Getting the type of 'self' (line 70)
        self_40774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 70)
        assertTrue_40775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_40774, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 70)
        assertTrue_call_result_40789 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), assertTrue_40775, *[exists_call_result_40787], **kwargs_40788)
        
        
        # Call to assertTrue(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Call to exists(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Call to join(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'inst' (line 71)
        inst_40798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 52), 'inst', False)
        # Getting the type of 'rone' (line 71)
        rone_40799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 58), 'rone', False)
        # Processing the call keyword arguments (line 71)
        kwargs_40800 = {}
        # Getting the type of 'os' (line 71)
        os_40795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 39), 'os', False)
        # Obtaining the member 'path' of a type (line 71)
        path_40796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 39), os_40795, 'path')
        # Obtaining the member 'join' of a type (line 71)
        join_40797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 39), path_40796, 'join')
        # Calling join(args, kwargs) (line 71)
        join_call_result_40801 = invoke(stypy.reporting.localization.Localization(__file__, 71, 39), join_40797, *[inst_40798, rone_40799], **kwargs_40800)
        
        # Processing the call keyword arguments (line 71)
        kwargs_40802 = {}
        # Getting the type of 'os' (line 71)
        os_40792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 71)
        path_40793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 24), os_40792, 'path')
        # Obtaining the member 'exists' of a type (line 71)
        exists_40794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 24), path_40793, 'exists')
        # Calling exists(args, kwargs) (line 71)
        exists_call_result_40803 = invoke(stypy.reporting.localization.Localization(__file__, 71, 24), exists_40794, *[join_call_result_40801], **kwargs_40802)
        
        # Processing the call keyword arguments (line 71)
        kwargs_40804 = {}
        # Getting the type of 'self' (line 71)
        self_40790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 71)
        assertTrue_40791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_40790, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 71)
        assertTrue_call_result_40805 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), assertTrue_40791, *[exists_call_result_40803], **kwargs_40804)
        
        
        # ################# End of 'test_simple_run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_run' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_40806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40806)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_run'
        return stypy_return_type_40806


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallDataTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'InstallDataTestCase' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'InstallDataTestCase', InstallDataTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 73, 0, False)
    
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

    
    # Call to makeSuite(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'InstallDataTestCase' (line 74)
    InstallDataTestCase_40809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 30), 'InstallDataTestCase', False)
    # Processing the call keyword arguments (line 74)
    kwargs_40810 = {}
    # Getting the type of 'unittest' (line 74)
    unittest_40807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 74)
    makeSuite_40808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 11), unittest_40807, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 74)
    makeSuite_call_result_40811 = invoke(stypy.reporting.localization.Localization(__file__, 74, 11), makeSuite_40808, *[InstallDataTestCase_40809], **kwargs_40810)
    
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type', makeSuite_call_result_40811)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_40812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_40812)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_40812

# Assigning a type to the variable 'test_suite' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Call to test_suite(...): (line 77)
    # Processing the call keyword arguments (line 77)
    kwargs_40815 = {}
    # Getting the type of 'test_suite' (line 77)
    test_suite_40814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 77)
    test_suite_call_result_40816 = invoke(stypy.reporting.localization.Localization(__file__, 77, 17), test_suite_40814, *[], **kwargs_40815)
    
    # Processing the call keyword arguments (line 77)
    kwargs_40817 = {}
    # Getting the type of 'run_unittest' (line 77)
    run_unittest_40813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 77)
    run_unittest_call_result_40818 = invoke(stypy.reporting.localization.Localization(__file__, 77, 4), run_unittest_40813, *[test_suite_call_result_40816], **kwargs_40817)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
