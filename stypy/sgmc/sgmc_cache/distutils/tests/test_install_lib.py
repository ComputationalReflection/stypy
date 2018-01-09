
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.install_data.'''
2: import os
3: import sys
4: import unittest
5: 
6: from distutils.command.install_lib import install_lib
7: from distutils.extension import Extension
8: from distutils.tests import support
9: from distutils.errors import DistutilsOptionError
10: from test.test_support import run_unittest
11: 
12: class InstallLibTestCase(support.TempdirManager,
13:                          support.LoggingSilencer,
14:                          support.EnvironGuard,
15:                          unittest.TestCase):
16: 
17:     def test_finalize_options(self):
18:         pkg_dir, dist = self.create_dist()
19:         cmd = install_lib(dist)
20: 
21:         cmd.finalize_options()
22:         self.assertEqual(cmd.compile, 1)
23:         self.assertEqual(cmd.optimize, 0)
24: 
25:         # optimize must be 0, 1, or 2
26:         cmd.optimize = 'foo'
27:         self.assertRaises(DistutilsOptionError, cmd.finalize_options)
28:         cmd.optimize = '4'
29:         self.assertRaises(DistutilsOptionError, cmd.finalize_options)
30: 
31:         cmd.optimize = '2'
32:         cmd.finalize_options()
33:         self.assertEqual(cmd.optimize, 2)
34: 
35:     def _setup_byte_compile(self):
36:         pkg_dir, dist = self.create_dist()
37:         cmd = install_lib(dist)
38:         cmd.compile = cmd.optimize = 1
39: 
40:         f = os.path.join(pkg_dir, 'foo.py')
41:         self.write_file(f, '# python file')
42:         cmd.byte_compile([f])
43:         return pkg_dir
44: 
45:     @unittest.skipIf(sys.dont_write_bytecode, 'byte-compile not enabled')
46:     def test_byte_compile(self):
47:         pkg_dir = self._setup_byte_compile()
48:         if sys.flags.optimize < 1:
49:             self.assertTrue(os.path.exists(os.path.join(pkg_dir, 'foo.pyc')))
50:         else:
51:             self.assertTrue(os.path.exists(os.path.join(pkg_dir, 'foo.pyo')))
52: 
53:     def test_get_outputs(self):
54:         pkg_dir, dist = self.create_dist()
55:         cmd = install_lib(dist)
56: 
57:         # setting up a dist environment
58:         cmd.compile = cmd.optimize = 1
59:         cmd.install_dir = pkg_dir
60:         f = os.path.join(pkg_dir, 'foo.py')
61:         self.write_file(f, '# python file')
62:         cmd.distribution.py_modules = [pkg_dir]
63:         cmd.distribution.ext_modules = [Extension('foo', ['xxx'])]
64:         cmd.distribution.packages = [pkg_dir]
65:         cmd.distribution.script_name = 'setup.py'
66: 
67:         # get_output should return 4 elements
68:         self.assertGreaterEqual(len(cmd.get_outputs()), 2)
69: 
70:     def test_get_inputs(self):
71:         pkg_dir, dist = self.create_dist()
72:         cmd = install_lib(dist)
73: 
74:         # setting up a dist environment
75:         cmd.compile = cmd.optimize = 1
76:         cmd.install_dir = pkg_dir
77:         f = os.path.join(pkg_dir, 'foo.py')
78:         self.write_file(f, '# python file')
79:         cmd.distribution.py_modules = [pkg_dir]
80:         cmd.distribution.ext_modules = [Extension('foo', ['xxx'])]
81:         cmd.distribution.packages = [pkg_dir]
82:         cmd.distribution.script_name = 'setup.py'
83: 
84:         # get_input should return 2 elements
85:         self.assertEqual(len(cmd.get_inputs()), 2)
86: 
87:     def test_dont_write_bytecode(self):
88:         # makes sure byte_compile is not used
89:         pkg_dir, dist = self.create_dist()
90:         cmd = install_lib(dist)
91:         cmd.compile = 1
92:         cmd.optimize = 1
93: 
94:         old_dont_write_bytecode = sys.dont_write_bytecode
95:         sys.dont_write_bytecode = True
96:         try:
97:             cmd.byte_compile([])
98:         finally:
99:             sys.dont_write_bytecode = old_dont_write_bytecode
100: 
101:         self.assertIn('byte-compiling is disabled', self.logs[0][1])
102: 
103: def test_suite():
104:     return unittest.makeSuite(InstallLibTestCase)
105: 
106: if __name__ == "__main__":
107:     run_unittest(test_suite())
108: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_40951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.install_data.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import os' statement (line 2)
import os

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import unittest' statement (line 4)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.command.install_lib import install_lib' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_40952 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.install_lib')

if (type(import_40952) is not StypyTypeError):

    if (import_40952 != 'pyd_module'):
        __import__(import_40952)
        sys_modules_40953 = sys.modules[import_40952]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.install_lib', sys_modules_40953.module_type_store, module_type_store, ['install_lib'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_40953, sys_modules_40953.module_type_store, module_type_store)
    else:
        from distutils.command.install_lib import install_lib

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.install_lib', None, module_type_store, ['install_lib'], [install_lib])

else:
    # Assigning a type to the variable 'distutils.command.install_lib' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.install_lib', import_40952)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.extension import Extension' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_40954 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.extension')

if (type(import_40954) is not StypyTypeError):

    if (import_40954 != 'pyd_module'):
        __import__(import_40954)
        sys_modules_40955 = sys.modules[import_40954]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.extension', sys_modules_40955.module_type_store, module_type_store, ['Extension'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_40955, sys_modules_40955.module_type_store, module_type_store)
    else:
        from distutils.extension import Extension

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.extension', None, module_type_store, ['Extension'], [Extension])

else:
    # Assigning a type to the variable 'distutils.extension' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.extension', import_40954)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.tests import support' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_40956 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests')

if (type(import_40956) is not StypyTypeError):

    if (import_40956 != 'pyd_module'):
        __import__(import_40956)
        sys_modules_40957 = sys.modules[import_40956]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', sys_modules_40957.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_40957, sys_modules_40957.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', import_40956)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.errors import DistutilsOptionError' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_40958 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors')

if (type(import_40958) is not StypyTypeError):

    if (import_40958 != 'pyd_module'):
        __import__(import_40958)
        sys_modules_40959 = sys.modules[import_40958]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', sys_modules_40959.module_type_store, module_type_store, ['DistutilsOptionError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_40959, sys_modules_40959.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsOptionError

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', None, module_type_store, ['DistutilsOptionError'], [DistutilsOptionError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', import_40958)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from test.test_support import run_unittest' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_40960 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'test.test_support')

if (type(import_40960) is not StypyTypeError):

    if (import_40960 != 'pyd_module'):
        __import__(import_40960)
        sys_modules_40961 = sys.modules[import_40960]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'test.test_support', sys_modules_40961.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_40961, sys_modules_40961.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'test.test_support', import_40960)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'InstallLibTestCase' class
# Getting the type of 'support' (line 12)
support_40962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 25), 'support')
# Obtaining the member 'TempdirManager' of a type (line 12)
TempdirManager_40963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 25), support_40962, 'TempdirManager')
# Getting the type of 'support' (line 13)
support_40964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 25), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 13)
LoggingSilencer_40965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 25), support_40964, 'LoggingSilencer')
# Getting the type of 'support' (line 14)
support_40966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 25), 'support')
# Obtaining the member 'EnvironGuard' of a type (line 14)
EnvironGuard_40967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 25), support_40966, 'EnvironGuard')
# Getting the type of 'unittest' (line 15)
unittest_40968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 25), 'unittest')
# Obtaining the member 'TestCase' of a type (line 15)
TestCase_40969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 25), unittest_40968, 'TestCase')

class InstallLibTestCase(TempdirManager_40963, LoggingSilencer_40965, EnvironGuard_40967, TestCase_40969, ):

    @norecursion
    def test_finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_finalize_options'
        module_type_store = module_type_store.open_function_context('test_finalize_options', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_localization', localization)
        InstallLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_function_name', 'InstallLibTestCase.test_finalize_options')
        InstallLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        InstallLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallLibTestCase.test_finalize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_finalize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_finalize_options(...)' code ##################

        
        # Assigning a Call to a Tuple (line 18):
        
        # Assigning a Subscript to a Name (line 18):
        
        # Obtaining the type of the subscript
        int_40970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'int')
        
        # Call to create_dist(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_40973 = {}
        # Getting the type of 'self' (line 18)
        self_40971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 18)
        create_dist_40972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 24), self_40971, 'create_dist')
        # Calling create_dist(args, kwargs) (line 18)
        create_dist_call_result_40974 = invoke(stypy.reporting.localization.Localization(__file__, 18, 24), create_dist_40972, *[], **kwargs_40973)
        
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___40975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), create_dist_call_result_40974, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_40976 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), getitem___40975, int_40970)
        
        # Assigning a type to the variable 'tuple_var_assignment_40941' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_40941', subscript_call_result_40976)
        
        # Assigning a Subscript to a Name (line 18):
        
        # Obtaining the type of the subscript
        int_40977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'int')
        
        # Call to create_dist(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_40980 = {}
        # Getting the type of 'self' (line 18)
        self_40978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 18)
        create_dist_40979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 24), self_40978, 'create_dist')
        # Calling create_dist(args, kwargs) (line 18)
        create_dist_call_result_40981 = invoke(stypy.reporting.localization.Localization(__file__, 18, 24), create_dist_40979, *[], **kwargs_40980)
        
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___40982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), create_dist_call_result_40981, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_40983 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), getitem___40982, int_40977)
        
        # Assigning a type to the variable 'tuple_var_assignment_40942' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_40942', subscript_call_result_40983)
        
        # Assigning a Name to a Name (line 18):
        # Getting the type of 'tuple_var_assignment_40941' (line 18)
        tuple_var_assignment_40941_40984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_40941')
        # Assigning a type to the variable 'pkg_dir' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'pkg_dir', tuple_var_assignment_40941_40984)
        
        # Assigning a Name to a Name (line 18):
        # Getting the type of 'tuple_var_assignment_40942' (line 18)
        tuple_var_assignment_40942_40985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_40942')
        # Assigning a type to the variable 'dist' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'dist', tuple_var_assignment_40942_40985)
        
        # Assigning a Call to a Name (line 19):
        
        # Assigning a Call to a Name (line 19):
        
        # Call to install_lib(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'dist' (line 19)
        dist_40987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 26), 'dist', False)
        # Processing the call keyword arguments (line 19)
        kwargs_40988 = {}
        # Getting the type of 'install_lib' (line 19)
        install_lib_40986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'install_lib', False)
        # Calling install_lib(args, kwargs) (line 19)
        install_lib_call_result_40989 = invoke(stypy.reporting.localization.Localization(__file__, 19, 14), install_lib_40986, *[dist_40987], **kwargs_40988)
        
        # Assigning a type to the variable 'cmd' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'cmd', install_lib_call_result_40989)
        
        # Call to finalize_options(...): (line 21)
        # Processing the call keyword arguments (line 21)
        kwargs_40992 = {}
        # Getting the type of 'cmd' (line 21)
        cmd_40990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 21)
        finalize_options_40991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), cmd_40990, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 21)
        finalize_options_call_result_40993 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), finalize_options_40991, *[], **kwargs_40992)
        
        
        # Call to assertEqual(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'cmd' (line 22)
        cmd_40996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'cmd', False)
        # Obtaining the member 'compile' of a type (line 22)
        compile_40997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 25), cmd_40996, 'compile')
        int_40998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 38), 'int')
        # Processing the call keyword arguments (line 22)
        kwargs_40999 = {}
        # Getting the type of 'self' (line 22)
        self_40994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 22)
        assertEqual_40995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_40994, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 22)
        assertEqual_call_result_41000 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), assertEqual_40995, *[compile_40997, int_40998], **kwargs_40999)
        
        
        # Call to assertEqual(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'cmd' (line 23)
        cmd_41003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 25), 'cmd', False)
        # Obtaining the member 'optimize' of a type (line 23)
        optimize_41004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 25), cmd_41003, 'optimize')
        int_41005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 39), 'int')
        # Processing the call keyword arguments (line 23)
        kwargs_41006 = {}
        # Getting the type of 'self' (line 23)
        self_41001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 23)
        assertEqual_41002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_41001, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 23)
        assertEqual_call_result_41007 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), assertEqual_41002, *[optimize_41004, int_41005], **kwargs_41006)
        
        
        # Assigning a Str to a Attribute (line 26):
        
        # Assigning a Str to a Attribute (line 26):
        str_41008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'str', 'foo')
        # Getting the type of 'cmd' (line 26)
        cmd_41009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'cmd')
        # Setting the type of the member 'optimize' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), cmd_41009, 'optimize', str_41008)
        
        # Call to assertRaises(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'DistutilsOptionError' (line 27)
        DistutilsOptionError_41012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 26), 'DistutilsOptionError', False)
        # Getting the type of 'cmd' (line 27)
        cmd_41013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 48), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 27)
        finalize_options_41014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 48), cmd_41013, 'finalize_options')
        # Processing the call keyword arguments (line 27)
        kwargs_41015 = {}
        # Getting the type of 'self' (line 27)
        self_41010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 27)
        assertRaises_41011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_41010, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 27)
        assertRaises_call_result_41016 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assertRaises_41011, *[DistutilsOptionError_41012, finalize_options_41014], **kwargs_41015)
        
        
        # Assigning a Str to a Attribute (line 28):
        
        # Assigning a Str to a Attribute (line 28):
        str_41017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 23), 'str', '4')
        # Getting the type of 'cmd' (line 28)
        cmd_41018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'cmd')
        # Setting the type of the member 'optimize' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), cmd_41018, 'optimize', str_41017)
        
        # Call to assertRaises(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'DistutilsOptionError' (line 29)
        DistutilsOptionError_41021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 26), 'DistutilsOptionError', False)
        # Getting the type of 'cmd' (line 29)
        cmd_41022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 48), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 29)
        finalize_options_41023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 48), cmd_41022, 'finalize_options')
        # Processing the call keyword arguments (line 29)
        kwargs_41024 = {}
        # Getting the type of 'self' (line 29)
        self_41019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 29)
        assertRaises_41020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_41019, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 29)
        assertRaises_call_result_41025 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), assertRaises_41020, *[DistutilsOptionError_41021, finalize_options_41023], **kwargs_41024)
        
        
        # Assigning a Str to a Attribute (line 31):
        
        # Assigning a Str to a Attribute (line 31):
        str_41026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 23), 'str', '2')
        # Getting the type of 'cmd' (line 31)
        cmd_41027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'cmd')
        # Setting the type of the member 'optimize' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), cmd_41027, 'optimize', str_41026)
        
        # Call to finalize_options(...): (line 32)
        # Processing the call keyword arguments (line 32)
        kwargs_41030 = {}
        # Getting the type of 'cmd' (line 32)
        cmd_41028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 32)
        finalize_options_41029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), cmd_41028, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 32)
        finalize_options_call_result_41031 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), finalize_options_41029, *[], **kwargs_41030)
        
        
        # Call to assertEqual(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'cmd' (line 33)
        cmd_41034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 25), 'cmd', False)
        # Obtaining the member 'optimize' of a type (line 33)
        optimize_41035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 25), cmd_41034, 'optimize')
        int_41036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 39), 'int')
        # Processing the call keyword arguments (line 33)
        kwargs_41037 = {}
        # Getting the type of 'self' (line 33)
        self_41032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 33)
        assertEqual_41033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_41032, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 33)
        assertEqual_call_result_41038 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), assertEqual_41033, *[optimize_41035, int_41036], **kwargs_41037)
        
        
        # ################# End of 'test_finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_41039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41039)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_finalize_options'
        return stypy_return_type_41039


    @norecursion
    def _setup_byte_compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_setup_byte_compile'
        module_type_store = module_type_store.open_function_context('_setup_byte_compile', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallLibTestCase._setup_byte_compile.__dict__.__setitem__('stypy_localization', localization)
        InstallLibTestCase._setup_byte_compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallLibTestCase._setup_byte_compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallLibTestCase._setup_byte_compile.__dict__.__setitem__('stypy_function_name', 'InstallLibTestCase._setup_byte_compile')
        InstallLibTestCase._setup_byte_compile.__dict__.__setitem__('stypy_param_names_list', [])
        InstallLibTestCase._setup_byte_compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallLibTestCase._setup_byte_compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallLibTestCase._setup_byte_compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallLibTestCase._setup_byte_compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallLibTestCase._setup_byte_compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallLibTestCase._setup_byte_compile.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallLibTestCase._setup_byte_compile', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_setup_byte_compile', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_setup_byte_compile(...)' code ##################

        
        # Assigning a Call to a Tuple (line 36):
        
        # Assigning a Subscript to a Name (line 36):
        
        # Obtaining the type of the subscript
        int_41040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 8), 'int')
        
        # Call to create_dist(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_41043 = {}
        # Getting the type of 'self' (line 36)
        self_41041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 36)
        create_dist_41042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 24), self_41041, 'create_dist')
        # Calling create_dist(args, kwargs) (line 36)
        create_dist_call_result_41044 = invoke(stypy.reporting.localization.Localization(__file__, 36, 24), create_dist_41042, *[], **kwargs_41043)
        
        # Obtaining the member '__getitem__' of a type (line 36)
        getitem___41045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), create_dist_call_result_41044, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 36)
        subscript_call_result_41046 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), getitem___41045, int_41040)
        
        # Assigning a type to the variable 'tuple_var_assignment_40943' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'tuple_var_assignment_40943', subscript_call_result_41046)
        
        # Assigning a Subscript to a Name (line 36):
        
        # Obtaining the type of the subscript
        int_41047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 8), 'int')
        
        # Call to create_dist(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_41050 = {}
        # Getting the type of 'self' (line 36)
        self_41048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 36)
        create_dist_41049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 24), self_41048, 'create_dist')
        # Calling create_dist(args, kwargs) (line 36)
        create_dist_call_result_41051 = invoke(stypy.reporting.localization.Localization(__file__, 36, 24), create_dist_41049, *[], **kwargs_41050)
        
        # Obtaining the member '__getitem__' of a type (line 36)
        getitem___41052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), create_dist_call_result_41051, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 36)
        subscript_call_result_41053 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), getitem___41052, int_41047)
        
        # Assigning a type to the variable 'tuple_var_assignment_40944' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'tuple_var_assignment_40944', subscript_call_result_41053)
        
        # Assigning a Name to a Name (line 36):
        # Getting the type of 'tuple_var_assignment_40943' (line 36)
        tuple_var_assignment_40943_41054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'tuple_var_assignment_40943')
        # Assigning a type to the variable 'pkg_dir' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'pkg_dir', tuple_var_assignment_40943_41054)
        
        # Assigning a Name to a Name (line 36):
        # Getting the type of 'tuple_var_assignment_40944' (line 36)
        tuple_var_assignment_40944_41055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'tuple_var_assignment_40944')
        # Assigning a type to the variable 'dist' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'dist', tuple_var_assignment_40944_41055)
        
        # Assigning a Call to a Name (line 37):
        
        # Assigning a Call to a Name (line 37):
        
        # Call to install_lib(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'dist' (line 37)
        dist_41057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'dist', False)
        # Processing the call keyword arguments (line 37)
        kwargs_41058 = {}
        # Getting the type of 'install_lib' (line 37)
        install_lib_41056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'install_lib', False)
        # Calling install_lib(args, kwargs) (line 37)
        install_lib_call_result_41059 = invoke(stypy.reporting.localization.Localization(__file__, 37, 14), install_lib_41056, *[dist_41057], **kwargs_41058)
        
        # Assigning a type to the variable 'cmd' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'cmd', install_lib_call_result_41059)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Num to a Attribute (line 38):
        int_41060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 37), 'int')
        # Getting the type of 'cmd' (line 38)
        cmd_41061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'cmd')
        # Setting the type of the member 'optimize' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 22), cmd_41061, 'optimize', int_41060)
        
        # Assigning a Attribute to a Attribute (line 38):
        # Getting the type of 'cmd' (line 38)
        cmd_41062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'cmd')
        # Obtaining the member 'optimize' of a type (line 38)
        optimize_41063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 22), cmd_41062, 'optimize')
        # Getting the type of 'cmd' (line 38)
        cmd_41064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'cmd')
        # Setting the type of the member 'compile' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), cmd_41064, 'compile', optimize_41063)
        
        # Assigning a Call to a Name (line 40):
        
        # Assigning a Call to a Name (line 40):
        
        # Call to join(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'pkg_dir' (line 40)
        pkg_dir_41068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'pkg_dir', False)
        str_41069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 34), 'str', 'foo.py')
        # Processing the call keyword arguments (line 40)
        kwargs_41070 = {}
        # Getting the type of 'os' (line 40)
        os_41065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'os', False)
        # Obtaining the member 'path' of a type (line 40)
        path_41066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), os_41065, 'path')
        # Obtaining the member 'join' of a type (line 40)
        join_41067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), path_41066, 'join')
        # Calling join(args, kwargs) (line 40)
        join_call_result_41071 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), join_41067, *[pkg_dir_41068, str_41069], **kwargs_41070)
        
        # Assigning a type to the variable 'f' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'f', join_call_result_41071)
        
        # Call to write_file(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'f' (line 41)
        f_41074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'f', False)
        str_41075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 27), 'str', '# python file')
        # Processing the call keyword arguments (line 41)
        kwargs_41076 = {}
        # Getting the type of 'self' (line 41)
        self_41072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 41)
        write_file_41073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_41072, 'write_file')
        # Calling write_file(args, kwargs) (line 41)
        write_file_call_result_41077 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), write_file_41073, *[f_41074, str_41075], **kwargs_41076)
        
        
        # Call to byte_compile(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_41080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        # Getting the type of 'f' (line 42)
        f_41081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'f', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 25), list_41080, f_41081)
        
        # Processing the call keyword arguments (line 42)
        kwargs_41082 = {}
        # Getting the type of 'cmd' (line 42)
        cmd_41078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'cmd', False)
        # Obtaining the member 'byte_compile' of a type (line 42)
        byte_compile_41079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), cmd_41078, 'byte_compile')
        # Calling byte_compile(args, kwargs) (line 42)
        byte_compile_call_result_41083 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), byte_compile_41079, *[list_41080], **kwargs_41082)
        
        # Getting the type of 'pkg_dir' (line 43)
        pkg_dir_41084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'pkg_dir')
        # Assigning a type to the variable 'stypy_return_type' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'stypy_return_type', pkg_dir_41084)
        
        # ################# End of '_setup_byte_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_setup_byte_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_41085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41085)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_setup_byte_compile'
        return stypy_return_type_41085


    @norecursion
    def test_byte_compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_byte_compile'
        module_type_store = module_type_store.open_function_context('test_byte_compile', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallLibTestCase.test_byte_compile.__dict__.__setitem__('stypy_localization', localization)
        InstallLibTestCase.test_byte_compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallLibTestCase.test_byte_compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallLibTestCase.test_byte_compile.__dict__.__setitem__('stypy_function_name', 'InstallLibTestCase.test_byte_compile')
        InstallLibTestCase.test_byte_compile.__dict__.__setitem__('stypy_param_names_list', [])
        InstallLibTestCase.test_byte_compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallLibTestCase.test_byte_compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallLibTestCase.test_byte_compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallLibTestCase.test_byte_compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallLibTestCase.test_byte_compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallLibTestCase.test_byte_compile.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallLibTestCase.test_byte_compile', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_byte_compile', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_byte_compile(...)' code ##################

        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to _setup_byte_compile(...): (line 47)
        # Processing the call keyword arguments (line 47)
        kwargs_41088 = {}
        # Getting the type of 'self' (line 47)
        self_41086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'self', False)
        # Obtaining the member '_setup_byte_compile' of a type (line 47)
        _setup_byte_compile_41087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 18), self_41086, '_setup_byte_compile')
        # Calling _setup_byte_compile(args, kwargs) (line 47)
        _setup_byte_compile_call_result_41089 = invoke(stypy.reporting.localization.Localization(__file__, 47, 18), _setup_byte_compile_41087, *[], **kwargs_41088)
        
        # Assigning a type to the variable 'pkg_dir' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'pkg_dir', _setup_byte_compile_call_result_41089)
        
        
        # Getting the type of 'sys' (line 48)
        sys_41090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'sys')
        # Obtaining the member 'flags' of a type (line 48)
        flags_41091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 11), sys_41090, 'flags')
        # Obtaining the member 'optimize' of a type (line 48)
        optimize_41092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 11), flags_41091, 'optimize')
        int_41093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 32), 'int')
        # Applying the binary operator '<' (line 48)
        result_lt_41094 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 11), '<', optimize_41092, int_41093)
        
        # Testing the type of an if condition (line 48)
        if_condition_41095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 8), result_lt_41094)
        # Assigning a type to the variable 'if_condition_41095' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'if_condition_41095', if_condition_41095)
        # SSA begins for if statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assertTrue(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Call to exists(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Call to join(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'pkg_dir' (line 49)
        pkg_dir_41104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 56), 'pkg_dir', False)
        str_41105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 65), 'str', 'foo.pyc')
        # Processing the call keyword arguments (line 49)
        kwargs_41106 = {}
        # Getting the type of 'os' (line 49)
        os_41101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 43), 'os', False)
        # Obtaining the member 'path' of a type (line 49)
        path_41102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 43), os_41101, 'path')
        # Obtaining the member 'join' of a type (line 49)
        join_41103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 43), path_41102, 'join')
        # Calling join(args, kwargs) (line 49)
        join_call_result_41107 = invoke(stypy.reporting.localization.Localization(__file__, 49, 43), join_41103, *[pkg_dir_41104, str_41105], **kwargs_41106)
        
        # Processing the call keyword arguments (line 49)
        kwargs_41108 = {}
        # Getting the type of 'os' (line 49)
        os_41098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 49)
        path_41099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 28), os_41098, 'path')
        # Obtaining the member 'exists' of a type (line 49)
        exists_41100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 28), path_41099, 'exists')
        # Calling exists(args, kwargs) (line 49)
        exists_call_result_41109 = invoke(stypy.reporting.localization.Localization(__file__, 49, 28), exists_41100, *[join_call_result_41107], **kwargs_41108)
        
        # Processing the call keyword arguments (line 49)
        kwargs_41110 = {}
        # Getting the type of 'self' (line 49)
        self_41096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 49)
        assertTrue_41097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_41096, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 49)
        assertTrue_call_result_41111 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), assertTrue_41097, *[exists_call_result_41109], **kwargs_41110)
        
        # SSA branch for the else part of an if statement (line 48)
        module_type_store.open_ssa_branch('else')
        
        # Call to assertTrue(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to exists(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to join(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'pkg_dir' (line 51)
        pkg_dir_41120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 56), 'pkg_dir', False)
        str_41121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 65), 'str', 'foo.pyo')
        # Processing the call keyword arguments (line 51)
        kwargs_41122 = {}
        # Getting the type of 'os' (line 51)
        os_41117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 43), 'os', False)
        # Obtaining the member 'path' of a type (line 51)
        path_41118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 43), os_41117, 'path')
        # Obtaining the member 'join' of a type (line 51)
        join_41119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 43), path_41118, 'join')
        # Calling join(args, kwargs) (line 51)
        join_call_result_41123 = invoke(stypy.reporting.localization.Localization(__file__, 51, 43), join_41119, *[pkg_dir_41120, str_41121], **kwargs_41122)
        
        # Processing the call keyword arguments (line 51)
        kwargs_41124 = {}
        # Getting the type of 'os' (line 51)
        os_41114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 51)
        path_41115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 28), os_41114, 'path')
        # Obtaining the member 'exists' of a type (line 51)
        exists_41116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 28), path_41115, 'exists')
        # Calling exists(args, kwargs) (line 51)
        exists_call_result_41125 = invoke(stypy.reporting.localization.Localization(__file__, 51, 28), exists_41116, *[join_call_result_41123], **kwargs_41124)
        
        # Processing the call keyword arguments (line 51)
        kwargs_41126 = {}
        # Getting the type of 'self' (line 51)
        self_41112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 51)
        assertTrue_41113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_41112, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 51)
        assertTrue_call_result_41127 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), assertTrue_41113, *[exists_call_result_41125], **kwargs_41126)
        
        # SSA join for if statement (line 48)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_byte_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_byte_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_41128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41128)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_byte_compile'
        return stypy_return_type_41128


    @norecursion
    def test_get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_get_outputs'
        module_type_store = module_type_store.open_function_context('test_get_outputs', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallLibTestCase.test_get_outputs.__dict__.__setitem__('stypy_localization', localization)
        InstallLibTestCase.test_get_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallLibTestCase.test_get_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallLibTestCase.test_get_outputs.__dict__.__setitem__('stypy_function_name', 'InstallLibTestCase.test_get_outputs')
        InstallLibTestCase.test_get_outputs.__dict__.__setitem__('stypy_param_names_list', [])
        InstallLibTestCase.test_get_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallLibTestCase.test_get_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallLibTestCase.test_get_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallLibTestCase.test_get_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallLibTestCase.test_get_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallLibTestCase.test_get_outputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallLibTestCase.test_get_outputs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_get_outputs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_get_outputs(...)' code ##################

        
        # Assigning a Call to a Tuple (line 54):
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_41129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'int')
        
        # Call to create_dist(...): (line 54)
        # Processing the call keyword arguments (line 54)
        kwargs_41132 = {}
        # Getting the type of 'self' (line 54)
        self_41130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 54)
        create_dist_41131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), self_41130, 'create_dist')
        # Calling create_dist(args, kwargs) (line 54)
        create_dist_call_result_41133 = invoke(stypy.reporting.localization.Localization(__file__, 54, 24), create_dist_41131, *[], **kwargs_41132)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___41134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), create_dist_call_result_41133, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_41135 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), getitem___41134, int_41129)
        
        # Assigning a type to the variable 'tuple_var_assignment_40945' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_40945', subscript_call_result_41135)
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_41136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'int')
        
        # Call to create_dist(...): (line 54)
        # Processing the call keyword arguments (line 54)
        kwargs_41139 = {}
        # Getting the type of 'self' (line 54)
        self_41137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 54)
        create_dist_41138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), self_41137, 'create_dist')
        # Calling create_dist(args, kwargs) (line 54)
        create_dist_call_result_41140 = invoke(stypy.reporting.localization.Localization(__file__, 54, 24), create_dist_41138, *[], **kwargs_41139)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___41141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), create_dist_call_result_41140, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_41142 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), getitem___41141, int_41136)
        
        # Assigning a type to the variable 'tuple_var_assignment_40946' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_40946', subscript_call_result_41142)
        
        # Assigning a Name to a Name (line 54):
        # Getting the type of 'tuple_var_assignment_40945' (line 54)
        tuple_var_assignment_40945_41143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_40945')
        # Assigning a type to the variable 'pkg_dir' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'pkg_dir', tuple_var_assignment_40945_41143)
        
        # Assigning a Name to a Name (line 54):
        # Getting the type of 'tuple_var_assignment_40946' (line 54)
        tuple_var_assignment_40946_41144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_40946')
        # Assigning a type to the variable 'dist' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'dist', tuple_var_assignment_40946_41144)
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to install_lib(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'dist' (line 55)
        dist_41146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 26), 'dist', False)
        # Processing the call keyword arguments (line 55)
        kwargs_41147 = {}
        # Getting the type of 'install_lib' (line 55)
        install_lib_41145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 14), 'install_lib', False)
        # Calling install_lib(args, kwargs) (line 55)
        install_lib_call_result_41148 = invoke(stypy.reporting.localization.Localization(__file__, 55, 14), install_lib_41145, *[dist_41146], **kwargs_41147)
        
        # Assigning a type to the variable 'cmd' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'cmd', install_lib_call_result_41148)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Num to a Attribute (line 58):
        int_41149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 37), 'int')
        # Getting the type of 'cmd' (line 58)
        cmd_41150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'cmd')
        # Setting the type of the member 'optimize' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 22), cmd_41150, 'optimize', int_41149)
        
        # Assigning a Attribute to a Attribute (line 58):
        # Getting the type of 'cmd' (line 58)
        cmd_41151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'cmd')
        # Obtaining the member 'optimize' of a type (line 58)
        optimize_41152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 22), cmd_41151, 'optimize')
        # Getting the type of 'cmd' (line 58)
        cmd_41153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'cmd')
        # Setting the type of the member 'compile' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), cmd_41153, 'compile', optimize_41152)
        
        # Assigning a Name to a Attribute (line 59):
        
        # Assigning a Name to a Attribute (line 59):
        # Getting the type of 'pkg_dir' (line 59)
        pkg_dir_41154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'pkg_dir')
        # Getting the type of 'cmd' (line 59)
        cmd_41155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'cmd')
        # Setting the type of the member 'install_dir' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), cmd_41155, 'install_dir', pkg_dir_41154)
        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to join(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'pkg_dir' (line 60)
        pkg_dir_41159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'pkg_dir', False)
        str_41160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 34), 'str', 'foo.py')
        # Processing the call keyword arguments (line 60)
        kwargs_41161 = {}
        # Getting the type of 'os' (line 60)
        os_41156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'os', False)
        # Obtaining the member 'path' of a type (line 60)
        path_41157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), os_41156, 'path')
        # Obtaining the member 'join' of a type (line 60)
        join_41158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), path_41157, 'join')
        # Calling join(args, kwargs) (line 60)
        join_call_result_41162 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), join_41158, *[pkg_dir_41159, str_41160], **kwargs_41161)
        
        # Assigning a type to the variable 'f' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'f', join_call_result_41162)
        
        # Call to write_file(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'f' (line 61)
        f_41165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'f', False)
        str_41166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 27), 'str', '# python file')
        # Processing the call keyword arguments (line 61)
        kwargs_41167 = {}
        # Getting the type of 'self' (line 61)
        self_41163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 61)
        write_file_41164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_41163, 'write_file')
        # Calling write_file(args, kwargs) (line 61)
        write_file_call_result_41168 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), write_file_41164, *[f_41165, str_41166], **kwargs_41167)
        
        
        # Assigning a List to a Attribute (line 62):
        
        # Assigning a List to a Attribute (line 62):
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_41169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        # Getting the type of 'pkg_dir' (line 62)
        pkg_dir_41170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 39), 'pkg_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 38), list_41169, pkg_dir_41170)
        
        # Getting the type of 'cmd' (line 62)
        cmd_41171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 62)
        distribution_41172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), cmd_41171, 'distribution')
        # Setting the type of the member 'py_modules' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), distribution_41172, 'py_modules', list_41169)
        
        # Assigning a List to a Attribute (line 63):
        
        # Assigning a List to a Attribute (line 63):
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_41173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        
        # Call to Extension(...): (line 63)
        # Processing the call arguments (line 63)
        str_41175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 50), 'str', 'foo')
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_41176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        str_41177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 58), 'str', 'xxx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 57), list_41176, str_41177)
        
        # Processing the call keyword arguments (line 63)
        kwargs_41178 = {}
        # Getting the type of 'Extension' (line 63)
        Extension_41174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 40), 'Extension', False)
        # Calling Extension(args, kwargs) (line 63)
        Extension_call_result_41179 = invoke(stypy.reporting.localization.Localization(__file__, 63, 40), Extension_41174, *[str_41175, list_41176], **kwargs_41178)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 39), list_41173, Extension_call_result_41179)
        
        # Getting the type of 'cmd' (line 63)
        cmd_41180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 63)
        distribution_41181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), cmd_41180, 'distribution')
        # Setting the type of the member 'ext_modules' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), distribution_41181, 'ext_modules', list_41173)
        
        # Assigning a List to a Attribute (line 64):
        
        # Assigning a List to a Attribute (line 64):
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_41182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        # Getting the type of 'pkg_dir' (line 64)
        pkg_dir_41183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 37), 'pkg_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 36), list_41182, pkg_dir_41183)
        
        # Getting the type of 'cmd' (line 64)
        cmd_41184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 64)
        distribution_41185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), cmd_41184, 'distribution')
        # Setting the type of the member 'packages' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), distribution_41185, 'packages', list_41182)
        
        # Assigning a Str to a Attribute (line 65):
        
        # Assigning a Str to a Attribute (line 65):
        str_41186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 39), 'str', 'setup.py')
        # Getting the type of 'cmd' (line 65)
        cmd_41187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 65)
        distribution_41188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), cmd_41187, 'distribution')
        # Setting the type of the member 'script_name' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), distribution_41188, 'script_name', str_41186)
        
        # Call to assertGreaterEqual(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Call to len(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Call to get_outputs(...): (line 68)
        # Processing the call keyword arguments (line 68)
        kwargs_41194 = {}
        # Getting the type of 'cmd' (line 68)
        cmd_41192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 36), 'cmd', False)
        # Obtaining the member 'get_outputs' of a type (line 68)
        get_outputs_41193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 36), cmd_41192, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 68)
        get_outputs_call_result_41195 = invoke(stypy.reporting.localization.Localization(__file__, 68, 36), get_outputs_41193, *[], **kwargs_41194)
        
        # Processing the call keyword arguments (line 68)
        kwargs_41196 = {}
        # Getting the type of 'len' (line 68)
        len_41191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 32), 'len', False)
        # Calling len(args, kwargs) (line 68)
        len_call_result_41197 = invoke(stypy.reporting.localization.Localization(__file__, 68, 32), len_41191, *[get_outputs_call_result_41195], **kwargs_41196)
        
        int_41198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 56), 'int')
        # Processing the call keyword arguments (line 68)
        kwargs_41199 = {}
        # Getting the type of 'self' (line 68)
        self_41189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self', False)
        # Obtaining the member 'assertGreaterEqual' of a type (line 68)
        assertGreaterEqual_41190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_41189, 'assertGreaterEqual')
        # Calling assertGreaterEqual(args, kwargs) (line 68)
        assertGreaterEqual_call_result_41200 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), assertGreaterEqual_41190, *[len_call_result_41197, int_41198], **kwargs_41199)
        
        
        # ################# End of 'test_get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_41201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41201)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_get_outputs'
        return stypy_return_type_41201


    @norecursion
    def test_get_inputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_get_inputs'
        module_type_store = module_type_store.open_function_context('test_get_inputs', 70, 4, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallLibTestCase.test_get_inputs.__dict__.__setitem__('stypy_localization', localization)
        InstallLibTestCase.test_get_inputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallLibTestCase.test_get_inputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallLibTestCase.test_get_inputs.__dict__.__setitem__('stypy_function_name', 'InstallLibTestCase.test_get_inputs')
        InstallLibTestCase.test_get_inputs.__dict__.__setitem__('stypy_param_names_list', [])
        InstallLibTestCase.test_get_inputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallLibTestCase.test_get_inputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallLibTestCase.test_get_inputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallLibTestCase.test_get_inputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallLibTestCase.test_get_inputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallLibTestCase.test_get_inputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallLibTestCase.test_get_inputs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_get_inputs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_get_inputs(...)' code ##################

        
        # Assigning a Call to a Tuple (line 71):
        
        # Assigning a Subscript to a Name (line 71):
        
        # Obtaining the type of the subscript
        int_41202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'int')
        
        # Call to create_dist(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_41205 = {}
        # Getting the type of 'self' (line 71)
        self_41203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 71)
        create_dist_41204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 24), self_41203, 'create_dist')
        # Calling create_dist(args, kwargs) (line 71)
        create_dist_call_result_41206 = invoke(stypy.reporting.localization.Localization(__file__, 71, 24), create_dist_41204, *[], **kwargs_41205)
        
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___41207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), create_dist_call_result_41206, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_41208 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___41207, int_41202)
        
        # Assigning a type to the variable 'tuple_var_assignment_40947' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_40947', subscript_call_result_41208)
        
        # Assigning a Subscript to a Name (line 71):
        
        # Obtaining the type of the subscript
        int_41209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'int')
        
        # Call to create_dist(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_41212 = {}
        # Getting the type of 'self' (line 71)
        self_41210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 71)
        create_dist_41211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 24), self_41210, 'create_dist')
        # Calling create_dist(args, kwargs) (line 71)
        create_dist_call_result_41213 = invoke(stypy.reporting.localization.Localization(__file__, 71, 24), create_dist_41211, *[], **kwargs_41212)
        
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___41214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), create_dist_call_result_41213, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_41215 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___41214, int_41209)
        
        # Assigning a type to the variable 'tuple_var_assignment_40948' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_40948', subscript_call_result_41215)
        
        # Assigning a Name to a Name (line 71):
        # Getting the type of 'tuple_var_assignment_40947' (line 71)
        tuple_var_assignment_40947_41216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_40947')
        # Assigning a type to the variable 'pkg_dir' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'pkg_dir', tuple_var_assignment_40947_41216)
        
        # Assigning a Name to a Name (line 71):
        # Getting the type of 'tuple_var_assignment_40948' (line 71)
        tuple_var_assignment_40948_41217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_40948')
        # Assigning a type to the variable 'dist' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'dist', tuple_var_assignment_40948_41217)
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to install_lib(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'dist' (line 72)
        dist_41219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'dist', False)
        # Processing the call keyword arguments (line 72)
        kwargs_41220 = {}
        # Getting the type of 'install_lib' (line 72)
        install_lib_41218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 14), 'install_lib', False)
        # Calling install_lib(args, kwargs) (line 72)
        install_lib_call_result_41221 = invoke(stypy.reporting.localization.Localization(__file__, 72, 14), install_lib_41218, *[dist_41219], **kwargs_41220)
        
        # Assigning a type to the variable 'cmd' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'cmd', install_lib_call_result_41221)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Num to a Attribute (line 75):
        int_41222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 37), 'int')
        # Getting the type of 'cmd' (line 75)
        cmd_41223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'cmd')
        # Setting the type of the member 'optimize' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 22), cmd_41223, 'optimize', int_41222)
        
        # Assigning a Attribute to a Attribute (line 75):
        # Getting the type of 'cmd' (line 75)
        cmd_41224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'cmd')
        # Obtaining the member 'optimize' of a type (line 75)
        optimize_41225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 22), cmd_41224, 'optimize')
        # Getting the type of 'cmd' (line 75)
        cmd_41226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'cmd')
        # Setting the type of the member 'compile' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), cmd_41226, 'compile', optimize_41225)
        
        # Assigning a Name to a Attribute (line 76):
        
        # Assigning a Name to a Attribute (line 76):
        # Getting the type of 'pkg_dir' (line 76)
        pkg_dir_41227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'pkg_dir')
        # Getting the type of 'cmd' (line 76)
        cmd_41228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'cmd')
        # Setting the type of the member 'install_dir' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), cmd_41228, 'install_dir', pkg_dir_41227)
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to join(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'pkg_dir' (line 77)
        pkg_dir_41232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'pkg_dir', False)
        str_41233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 34), 'str', 'foo.py')
        # Processing the call keyword arguments (line 77)
        kwargs_41234 = {}
        # Getting the type of 'os' (line 77)
        os_41229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'os', False)
        # Obtaining the member 'path' of a type (line 77)
        path_41230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), os_41229, 'path')
        # Obtaining the member 'join' of a type (line 77)
        join_41231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), path_41230, 'join')
        # Calling join(args, kwargs) (line 77)
        join_call_result_41235 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), join_41231, *[pkg_dir_41232, str_41233], **kwargs_41234)
        
        # Assigning a type to the variable 'f' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'f', join_call_result_41235)
        
        # Call to write_file(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'f' (line 78)
        f_41238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'f', False)
        str_41239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 27), 'str', '# python file')
        # Processing the call keyword arguments (line 78)
        kwargs_41240 = {}
        # Getting the type of 'self' (line 78)
        self_41236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 78)
        write_file_41237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_41236, 'write_file')
        # Calling write_file(args, kwargs) (line 78)
        write_file_call_result_41241 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), write_file_41237, *[f_41238, str_41239], **kwargs_41240)
        
        
        # Assigning a List to a Attribute (line 79):
        
        # Assigning a List to a Attribute (line 79):
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_41242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        # Getting the type of 'pkg_dir' (line 79)
        pkg_dir_41243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 39), 'pkg_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 38), list_41242, pkg_dir_41243)
        
        # Getting the type of 'cmd' (line 79)
        cmd_41244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 79)
        distribution_41245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), cmd_41244, 'distribution')
        # Setting the type of the member 'py_modules' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), distribution_41245, 'py_modules', list_41242)
        
        # Assigning a List to a Attribute (line 80):
        
        # Assigning a List to a Attribute (line 80):
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_41246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        
        # Call to Extension(...): (line 80)
        # Processing the call arguments (line 80)
        str_41248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 50), 'str', 'foo')
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_41249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        str_41250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 58), 'str', 'xxx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 57), list_41249, str_41250)
        
        # Processing the call keyword arguments (line 80)
        kwargs_41251 = {}
        # Getting the type of 'Extension' (line 80)
        Extension_41247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 40), 'Extension', False)
        # Calling Extension(args, kwargs) (line 80)
        Extension_call_result_41252 = invoke(stypy.reporting.localization.Localization(__file__, 80, 40), Extension_41247, *[str_41248, list_41249], **kwargs_41251)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 39), list_41246, Extension_call_result_41252)
        
        # Getting the type of 'cmd' (line 80)
        cmd_41253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 80)
        distribution_41254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), cmd_41253, 'distribution')
        # Setting the type of the member 'ext_modules' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), distribution_41254, 'ext_modules', list_41246)
        
        # Assigning a List to a Attribute (line 81):
        
        # Assigning a List to a Attribute (line 81):
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_41255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        # Getting the type of 'pkg_dir' (line 81)
        pkg_dir_41256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 37), 'pkg_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 36), list_41255, pkg_dir_41256)
        
        # Getting the type of 'cmd' (line 81)
        cmd_41257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 81)
        distribution_41258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), cmd_41257, 'distribution')
        # Setting the type of the member 'packages' of a type (line 81)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), distribution_41258, 'packages', list_41255)
        
        # Assigning a Str to a Attribute (line 82):
        
        # Assigning a Str to a Attribute (line 82):
        str_41259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 39), 'str', 'setup.py')
        # Getting the type of 'cmd' (line 82)
        cmd_41260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 82)
        distribution_41261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), cmd_41260, 'distribution')
        # Setting the type of the member 'script_name' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), distribution_41261, 'script_name', str_41259)
        
        # Call to assertEqual(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Call to len(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Call to get_inputs(...): (line 85)
        # Processing the call keyword arguments (line 85)
        kwargs_41267 = {}
        # Getting the type of 'cmd' (line 85)
        cmd_41265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'cmd', False)
        # Obtaining the member 'get_inputs' of a type (line 85)
        get_inputs_41266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 29), cmd_41265, 'get_inputs')
        # Calling get_inputs(args, kwargs) (line 85)
        get_inputs_call_result_41268 = invoke(stypy.reporting.localization.Localization(__file__, 85, 29), get_inputs_41266, *[], **kwargs_41267)
        
        # Processing the call keyword arguments (line 85)
        kwargs_41269 = {}
        # Getting the type of 'len' (line 85)
        len_41264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'len', False)
        # Calling len(args, kwargs) (line 85)
        len_call_result_41270 = invoke(stypy.reporting.localization.Localization(__file__, 85, 25), len_41264, *[get_inputs_call_result_41268], **kwargs_41269)
        
        int_41271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 48), 'int')
        # Processing the call keyword arguments (line 85)
        kwargs_41272 = {}
        # Getting the type of 'self' (line 85)
        self_41262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 85)
        assertEqual_41263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_41262, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 85)
        assertEqual_call_result_41273 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), assertEqual_41263, *[len_call_result_41270, int_41271], **kwargs_41272)
        
        
        # ################# End of 'test_get_inputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_get_inputs' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_41274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41274)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_get_inputs'
        return stypy_return_type_41274


    @norecursion
    def test_dont_write_bytecode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dont_write_bytecode'
        module_type_store = module_type_store.open_function_context('test_dont_write_bytecode', 87, 4, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallLibTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_localization', localization)
        InstallLibTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallLibTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallLibTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_function_name', 'InstallLibTestCase.test_dont_write_bytecode')
        InstallLibTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_param_names_list', [])
        InstallLibTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallLibTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallLibTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallLibTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallLibTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallLibTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallLibTestCase.test_dont_write_bytecode', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dont_write_bytecode', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dont_write_bytecode(...)' code ##################

        
        # Assigning a Call to a Tuple (line 89):
        
        # Assigning a Subscript to a Name (line 89):
        
        # Obtaining the type of the subscript
        int_41275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'int')
        
        # Call to create_dist(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_41278 = {}
        # Getting the type of 'self' (line 89)
        self_41276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 89)
        create_dist_41277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 24), self_41276, 'create_dist')
        # Calling create_dist(args, kwargs) (line 89)
        create_dist_call_result_41279 = invoke(stypy.reporting.localization.Localization(__file__, 89, 24), create_dist_41277, *[], **kwargs_41278)
        
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___41280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), create_dist_call_result_41279, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_41281 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), getitem___41280, int_41275)
        
        # Assigning a type to the variable 'tuple_var_assignment_40949' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_40949', subscript_call_result_41281)
        
        # Assigning a Subscript to a Name (line 89):
        
        # Obtaining the type of the subscript
        int_41282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'int')
        
        # Call to create_dist(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_41285 = {}
        # Getting the type of 'self' (line 89)
        self_41283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 89)
        create_dist_41284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 24), self_41283, 'create_dist')
        # Calling create_dist(args, kwargs) (line 89)
        create_dist_call_result_41286 = invoke(stypy.reporting.localization.Localization(__file__, 89, 24), create_dist_41284, *[], **kwargs_41285)
        
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___41287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), create_dist_call_result_41286, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_41288 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), getitem___41287, int_41282)
        
        # Assigning a type to the variable 'tuple_var_assignment_40950' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_40950', subscript_call_result_41288)
        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'tuple_var_assignment_40949' (line 89)
        tuple_var_assignment_40949_41289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_40949')
        # Assigning a type to the variable 'pkg_dir' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'pkg_dir', tuple_var_assignment_40949_41289)
        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'tuple_var_assignment_40950' (line 89)
        tuple_var_assignment_40950_41290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_var_assignment_40950')
        # Assigning a type to the variable 'dist' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'dist', tuple_var_assignment_40950_41290)
        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to install_lib(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'dist' (line 90)
        dist_41292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 26), 'dist', False)
        # Processing the call keyword arguments (line 90)
        kwargs_41293 = {}
        # Getting the type of 'install_lib' (line 90)
        install_lib_41291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 14), 'install_lib', False)
        # Calling install_lib(args, kwargs) (line 90)
        install_lib_call_result_41294 = invoke(stypy.reporting.localization.Localization(__file__, 90, 14), install_lib_41291, *[dist_41292], **kwargs_41293)
        
        # Assigning a type to the variable 'cmd' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'cmd', install_lib_call_result_41294)
        
        # Assigning a Num to a Attribute (line 91):
        
        # Assigning a Num to a Attribute (line 91):
        int_41295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 22), 'int')
        # Getting the type of 'cmd' (line 91)
        cmd_41296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'cmd')
        # Setting the type of the member 'compile' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), cmd_41296, 'compile', int_41295)
        
        # Assigning a Num to a Attribute (line 92):
        
        # Assigning a Num to a Attribute (line 92):
        int_41297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 23), 'int')
        # Getting the type of 'cmd' (line 92)
        cmd_41298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'cmd')
        # Setting the type of the member 'optimize' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), cmd_41298, 'optimize', int_41297)
        
        # Assigning a Attribute to a Name (line 94):
        
        # Assigning a Attribute to a Name (line 94):
        # Getting the type of 'sys' (line 94)
        sys_41299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 34), 'sys')
        # Obtaining the member 'dont_write_bytecode' of a type (line 94)
        dont_write_bytecode_41300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 34), sys_41299, 'dont_write_bytecode')
        # Assigning a type to the variable 'old_dont_write_bytecode' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'old_dont_write_bytecode', dont_write_bytecode_41300)
        
        # Assigning a Name to a Attribute (line 95):
        
        # Assigning a Name to a Attribute (line 95):
        # Getting the type of 'True' (line 95)
        True_41301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 34), 'True')
        # Getting the type of 'sys' (line 95)
        sys_41302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'sys')
        # Setting the type of the member 'dont_write_bytecode' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), sys_41302, 'dont_write_bytecode', True_41301)
        
        # Try-finally block (line 96)
        
        # Call to byte_compile(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_41305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        
        # Processing the call keyword arguments (line 97)
        kwargs_41306 = {}
        # Getting the type of 'cmd' (line 97)
        cmd_41303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'cmd', False)
        # Obtaining the member 'byte_compile' of a type (line 97)
        byte_compile_41304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), cmd_41303, 'byte_compile')
        # Calling byte_compile(args, kwargs) (line 97)
        byte_compile_call_result_41307 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), byte_compile_41304, *[list_41305], **kwargs_41306)
        
        
        # finally branch of the try-finally block (line 96)
        
        # Assigning a Name to a Attribute (line 99):
        
        # Assigning a Name to a Attribute (line 99):
        # Getting the type of 'old_dont_write_bytecode' (line 99)
        old_dont_write_bytecode_41308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'old_dont_write_bytecode')
        # Getting the type of 'sys' (line 99)
        sys_41309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'sys')
        # Setting the type of the member 'dont_write_bytecode' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), sys_41309, 'dont_write_bytecode', old_dont_write_bytecode_41308)
        
        
        # Call to assertIn(...): (line 101)
        # Processing the call arguments (line 101)
        str_41312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 22), 'str', 'byte-compiling is disabled')
        
        # Obtaining the type of the subscript
        int_41313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 65), 'int')
        
        # Obtaining the type of the subscript
        int_41314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 62), 'int')
        # Getting the type of 'self' (line 101)
        self_41315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 52), 'self', False)
        # Obtaining the member 'logs' of a type (line 101)
        logs_41316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 52), self_41315, 'logs')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___41317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 52), logs_41316, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_41318 = invoke(stypy.reporting.localization.Localization(__file__, 101, 52), getitem___41317, int_41314)
        
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___41319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 52), subscript_call_result_41318, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_41320 = invoke(stypy.reporting.localization.Localization(__file__, 101, 52), getitem___41319, int_41313)
        
        # Processing the call keyword arguments (line 101)
        kwargs_41321 = {}
        # Getting the type of 'self' (line 101)
        self_41310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 101)
        assertIn_41311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), self_41310, 'assertIn')
        # Calling assertIn(args, kwargs) (line 101)
        assertIn_call_result_41322 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), assertIn_41311, *[str_41312, subscript_call_result_41320], **kwargs_41321)
        
        
        # ################# End of 'test_dont_write_bytecode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dont_write_bytecode' in the type store
        # Getting the type of 'stypy_return_type' (line 87)
        stypy_return_type_41323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41323)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dont_write_bytecode'
        return stypy_return_type_41323


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 12, 0, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallLibTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'InstallLibTestCase' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'InstallLibTestCase', InstallLibTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 103, 0, False)
    
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

    
    # Call to makeSuite(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'InstallLibTestCase' (line 104)
    InstallLibTestCase_41326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'InstallLibTestCase', False)
    # Processing the call keyword arguments (line 104)
    kwargs_41327 = {}
    # Getting the type of 'unittest' (line 104)
    unittest_41324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 104)
    makeSuite_41325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), unittest_41324, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 104)
    makeSuite_call_result_41328 = invoke(stypy.reporting.localization.Localization(__file__, 104, 11), makeSuite_41325, *[InstallLibTestCase_41326], **kwargs_41327)
    
    # Assigning a type to the variable 'stypy_return_type' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type', makeSuite_call_result_41328)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 103)
    stypy_return_type_41329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_41329)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_41329

# Assigning a type to the variable 'test_suite' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 107)
    # Processing the call arguments (line 107)
    
    # Call to test_suite(...): (line 107)
    # Processing the call keyword arguments (line 107)
    kwargs_41332 = {}
    # Getting the type of 'test_suite' (line 107)
    test_suite_41331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 107)
    test_suite_call_result_41333 = invoke(stypy.reporting.localization.Localization(__file__, 107, 17), test_suite_41331, *[], **kwargs_41332)
    
    # Processing the call keyword arguments (line 107)
    kwargs_41334 = {}
    # Getting the type of 'run_unittest' (line 107)
    run_unittest_41330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 107)
    run_unittest_call_result_41335 = invoke(stypy.reporting.localization.Localization(__file__, 107, 4), run_unittest_41330, *[test_suite_call_result_41333], **kwargs_41334)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
