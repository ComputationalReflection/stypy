
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.build_clib.'''
2: import unittest
3: import os
4: import sys
5: 
6: from test.test_support import run_unittest
7: 
8: from distutils.command.build_clib import build_clib
9: from distutils.errors import DistutilsSetupError
10: from distutils.tests import support
11: from distutils.spawn import find_executable
12: 
13: class BuildCLibTestCase(support.TempdirManager,
14:                         support.LoggingSilencer,
15:                         unittest.TestCase):
16: 
17:     def test_check_library_dist(self):
18:         pkg_dir, dist = self.create_dist()
19:         cmd = build_clib(dist)
20: 
21:         # 'libraries' option must be a list
22:         self.assertRaises(DistutilsSetupError, cmd.check_library_list, 'foo')
23: 
24:         # each element of 'libraries' must a 2-tuple
25:         self.assertRaises(DistutilsSetupError, cmd.check_library_list,
26:                           ['foo1', 'foo2'])
27: 
28:         # first element of each tuple in 'libraries'
29:         # must be a string (the library name)
30:         self.assertRaises(DistutilsSetupError, cmd.check_library_list,
31:                           [(1, 'foo1'), ('name', 'foo2')])
32: 
33:         # library name may not contain directory separators
34:         self.assertRaises(DistutilsSetupError, cmd.check_library_list,
35:                           [('name', 'foo1'),
36:                            ('another/name', 'foo2')])
37: 
38:         # second element of each tuple must be a dictionary (build info)
39:         self.assertRaises(DistutilsSetupError, cmd.check_library_list,
40:                           [('name', {}),
41:                            ('another', 'foo2')])
42: 
43:         # those work
44:         libs = [('name', {}), ('name', {'ok': 'good'})]
45:         cmd.check_library_list(libs)
46: 
47:     def test_get_source_files(self):
48:         pkg_dir, dist = self.create_dist()
49:         cmd = build_clib(dist)
50: 
51:         # "in 'libraries' option 'sources' must be present and must be
52:         # a list of source filenames
53:         cmd.libraries = [('name', {})]
54:         self.assertRaises(DistutilsSetupError, cmd.get_source_files)
55: 
56:         cmd.libraries = [('name', {'sources': 1})]
57:         self.assertRaises(DistutilsSetupError, cmd.get_source_files)
58: 
59:         cmd.libraries = [('name', {'sources': ['a', 'b']})]
60:         self.assertEqual(cmd.get_source_files(), ['a', 'b'])
61: 
62:         cmd.libraries = [('name', {'sources': ('a', 'b')})]
63:         self.assertEqual(cmd.get_source_files(), ['a', 'b'])
64: 
65:         cmd.libraries = [('name', {'sources': ('a', 'b')}),
66:                          ('name2', {'sources': ['c', 'd']})]
67:         self.assertEqual(cmd.get_source_files(), ['a', 'b', 'c', 'd'])
68: 
69:     def test_build_libraries(self):
70: 
71:         pkg_dir, dist = self.create_dist()
72:         cmd = build_clib(dist)
73:         class FakeCompiler:
74:             def compile(*args, **kw):
75:                 pass
76:             create_static_lib = compile
77: 
78:         cmd.compiler = FakeCompiler()
79: 
80:         # build_libraries is also doing a bit of typo checking
81:         lib = [('name', {'sources': 'notvalid'})]
82:         self.assertRaises(DistutilsSetupError, cmd.build_libraries, lib)
83: 
84:         lib = [('name', {'sources': list()})]
85:         cmd.build_libraries(lib)
86: 
87:         lib = [('name', {'sources': tuple()})]
88:         cmd.build_libraries(lib)
89: 
90:     def test_finalize_options(self):
91:         pkg_dir, dist = self.create_dist()
92:         cmd = build_clib(dist)
93: 
94:         cmd.include_dirs = 'one-dir'
95:         cmd.finalize_options()
96:         self.assertEqual(cmd.include_dirs, ['one-dir'])
97: 
98:         cmd.include_dirs = None
99:         cmd.finalize_options()
100:         self.assertEqual(cmd.include_dirs, [])
101: 
102:         cmd.distribution.libraries = 'WONTWORK'
103:         self.assertRaises(DistutilsSetupError, cmd.finalize_options)
104: 
105:     @unittest.skipIf(sys.platform == 'win32', "can't test on Windows")
106:     def test_run(self):
107:         pkg_dir, dist = self.create_dist()
108:         cmd = build_clib(dist)
109: 
110:         foo_c = os.path.join(pkg_dir, 'foo.c')
111:         self.write_file(foo_c, 'int main(void) { return 1;}\n')
112:         cmd.libraries = [('foo', {'sources': [foo_c]})]
113: 
114:         build_temp = os.path.join(pkg_dir, 'build')
115:         os.mkdir(build_temp)
116:         cmd.build_temp = build_temp
117:         cmd.build_clib = build_temp
118: 
119:         # before we run the command, we want to make sure
120:         # all commands are present on the system
121:         # by creating a compiler and checking its executables
122:         from distutils.ccompiler import new_compiler
123:         from distutils.sysconfig import customize_compiler
124: 
125:         compiler = new_compiler()
126:         customize_compiler(compiler)
127:         for ccmd in compiler.executables.values():
128:             if ccmd is None:
129:                 continue
130:             if find_executable(ccmd[0]) is None:
131:                 self.skipTest('The %r command is not found' % ccmd[0])
132: 
133:         # this should work
134:         cmd.run()
135: 
136:         # let's check the result
137:         self.assertIn('libfoo.a', os.listdir(build_temp))
138: 
139: def test_suite():
140:     return unittest.makeSuite(BuildCLibTestCase)
141: 
142: if __name__ == "__main__":
143:     run_unittest(test_suite())
144: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_31080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.build_clib.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import unittest' statement (line 2)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from test.test_support import run_unittest' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_31081 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'test.test_support')

if (type(import_31081) is not StypyTypeError):

    if (import_31081 != 'pyd_module'):
        __import__(import_31081)
        sys_modules_31082 = sys.modules[import_31081]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'test.test_support', sys_modules_31082.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_31082, sys_modules_31082.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'test.test_support', import_31081)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.command.build_clib import build_clib' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_31083 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.build_clib')

if (type(import_31083) is not StypyTypeError):

    if (import_31083 != 'pyd_module'):
        __import__(import_31083)
        sys_modules_31084 = sys.modules[import_31083]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.build_clib', sys_modules_31084.module_type_store, module_type_store, ['build_clib'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_31084, sys_modules_31084.module_type_store, module_type_store)
    else:
        from distutils.command.build_clib import build_clib

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.build_clib', None, module_type_store, ['build_clib'], [build_clib])

else:
    # Assigning a type to the variable 'distutils.command.build_clib' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.build_clib', import_31083)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.errors import DistutilsSetupError' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_31085 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors')

if (type(import_31085) is not StypyTypeError):

    if (import_31085 != 'pyd_module'):
        __import__(import_31085)
        sys_modules_31086 = sys.modules[import_31085]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', sys_modules_31086.module_type_store, module_type_store, ['DistutilsSetupError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_31086, sys_modules_31086.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsSetupError

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', None, module_type_store, ['DistutilsSetupError'], [DistutilsSetupError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', import_31085)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.tests import support' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_31087 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests')

if (type(import_31087) is not StypyTypeError):

    if (import_31087 != 'pyd_module'):
        __import__(import_31087)
        sys_modules_31088 = sys.modules[import_31087]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests', sys_modules_31088.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_31088, sys_modules_31088.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests', import_31087)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.spawn import find_executable' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_31089 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.spawn')

if (type(import_31089) is not StypyTypeError):

    if (import_31089 != 'pyd_module'):
        __import__(import_31089)
        sys_modules_31090 = sys.modules[import_31089]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.spawn', sys_modules_31090.module_type_store, module_type_store, ['find_executable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_31090, sys_modules_31090.module_type_store, module_type_store)
    else:
        from distutils.spawn import find_executable

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.spawn', None, module_type_store, ['find_executable'], [find_executable])

else:
    # Assigning a type to the variable 'distutils.spawn' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.spawn', import_31089)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'BuildCLibTestCase' class
# Getting the type of 'support' (line 13)
support_31091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 24), 'support')
# Obtaining the member 'TempdirManager' of a type (line 13)
TempdirManager_31092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 24), support_31091, 'TempdirManager')
# Getting the type of 'support' (line 14)
support_31093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 24), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 14)
LoggingSilencer_31094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 24), support_31093, 'LoggingSilencer')
# Getting the type of 'unittest' (line 15)
unittest_31095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'unittest')
# Obtaining the member 'TestCase' of a type (line 15)
TestCase_31096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 24), unittest_31095, 'TestCase')

class BuildCLibTestCase(TempdirManager_31092, LoggingSilencer_31094, TestCase_31096, ):

    @norecursion
    def test_check_library_dist(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_check_library_dist'
        module_type_store = module_type_store.open_function_context('test_check_library_dist', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildCLibTestCase.test_check_library_dist.__dict__.__setitem__('stypy_localization', localization)
        BuildCLibTestCase.test_check_library_dist.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildCLibTestCase.test_check_library_dist.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildCLibTestCase.test_check_library_dist.__dict__.__setitem__('stypy_function_name', 'BuildCLibTestCase.test_check_library_dist')
        BuildCLibTestCase.test_check_library_dist.__dict__.__setitem__('stypy_param_names_list', [])
        BuildCLibTestCase.test_check_library_dist.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildCLibTestCase.test_check_library_dist.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildCLibTestCase.test_check_library_dist.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildCLibTestCase.test_check_library_dist.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildCLibTestCase.test_check_library_dist.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildCLibTestCase.test_check_library_dist.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildCLibTestCase.test_check_library_dist', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_check_library_dist', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_check_library_dist(...)' code ##################

        
        # Assigning a Call to a Tuple (line 18):
        
        # Assigning a Subscript to a Name (line 18):
        
        # Obtaining the type of the subscript
        int_31097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'int')
        
        # Call to create_dist(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_31100 = {}
        # Getting the type of 'self' (line 18)
        self_31098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 18)
        create_dist_31099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 24), self_31098, 'create_dist')
        # Calling create_dist(args, kwargs) (line 18)
        create_dist_call_result_31101 = invoke(stypy.reporting.localization.Localization(__file__, 18, 24), create_dist_31099, *[], **kwargs_31100)
        
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___31102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), create_dist_call_result_31101, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_31103 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), getitem___31102, int_31097)
        
        # Assigning a type to the variable 'tuple_var_assignment_31070' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_31070', subscript_call_result_31103)
        
        # Assigning a Subscript to a Name (line 18):
        
        # Obtaining the type of the subscript
        int_31104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'int')
        
        # Call to create_dist(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_31107 = {}
        # Getting the type of 'self' (line 18)
        self_31105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 18)
        create_dist_31106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 24), self_31105, 'create_dist')
        # Calling create_dist(args, kwargs) (line 18)
        create_dist_call_result_31108 = invoke(stypy.reporting.localization.Localization(__file__, 18, 24), create_dist_31106, *[], **kwargs_31107)
        
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___31109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), create_dist_call_result_31108, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_31110 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), getitem___31109, int_31104)
        
        # Assigning a type to the variable 'tuple_var_assignment_31071' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_31071', subscript_call_result_31110)
        
        # Assigning a Name to a Name (line 18):
        # Getting the type of 'tuple_var_assignment_31070' (line 18)
        tuple_var_assignment_31070_31111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_31070')
        # Assigning a type to the variable 'pkg_dir' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'pkg_dir', tuple_var_assignment_31070_31111)
        
        # Assigning a Name to a Name (line 18):
        # Getting the type of 'tuple_var_assignment_31071' (line 18)
        tuple_var_assignment_31071_31112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_31071')
        # Assigning a type to the variable 'dist' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'dist', tuple_var_assignment_31071_31112)
        
        # Assigning a Call to a Name (line 19):
        
        # Assigning a Call to a Name (line 19):
        
        # Call to build_clib(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'dist' (line 19)
        dist_31114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'dist', False)
        # Processing the call keyword arguments (line 19)
        kwargs_31115 = {}
        # Getting the type of 'build_clib' (line 19)
        build_clib_31113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'build_clib', False)
        # Calling build_clib(args, kwargs) (line 19)
        build_clib_call_result_31116 = invoke(stypy.reporting.localization.Localization(__file__, 19, 14), build_clib_31113, *[dist_31114], **kwargs_31115)
        
        # Assigning a type to the variable 'cmd' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'cmd', build_clib_call_result_31116)
        
        # Call to assertRaises(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'DistutilsSetupError' (line 22)
        DistutilsSetupError_31119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 22)
        cmd_31120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 47), 'cmd', False)
        # Obtaining the member 'check_library_list' of a type (line 22)
        check_library_list_31121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 47), cmd_31120, 'check_library_list')
        str_31122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 71), 'str', 'foo')
        # Processing the call keyword arguments (line 22)
        kwargs_31123 = {}
        # Getting the type of 'self' (line 22)
        self_31117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 22)
        assertRaises_31118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_31117, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 22)
        assertRaises_call_result_31124 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), assertRaises_31118, *[DistutilsSetupError_31119, check_library_list_31121, str_31122], **kwargs_31123)
        
        
        # Call to assertRaises(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'DistutilsSetupError' (line 25)
        DistutilsSetupError_31127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 25)
        cmd_31128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 47), 'cmd', False)
        # Obtaining the member 'check_library_list' of a type (line 25)
        check_library_list_31129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 47), cmd_31128, 'check_library_list')
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_31130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        str_31131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 27), 'str', 'foo1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 26), list_31130, str_31131)
        # Adding element type (line 26)
        str_31132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 35), 'str', 'foo2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 26), list_31130, str_31132)
        
        # Processing the call keyword arguments (line 25)
        kwargs_31133 = {}
        # Getting the type of 'self' (line 25)
        self_31125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 25)
        assertRaises_31126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_31125, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 25)
        assertRaises_call_result_31134 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), assertRaises_31126, *[DistutilsSetupError_31127, check_library_list_31129, list_31130], **kwargs_31133)
        
        
        # Call to assertRaises(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'DistutilsSetupError' (line 30)
        DistutilsSetupError_31137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 30)
        cmd_31138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 47), 'cmd', False)
        # Obtaining the member 'check_library_list' of a type (line 30)
        check_library_list_31139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 47), cmd_31138, 'check_library_list')
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_31140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        
        # Obtaining an instance of the builtin type 'tuple' (line 31)
        tuple_31141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 31)
        # Adding element type (line 31)
        int_31142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 28), tuple_31141, int_31142)
        # Adding element type (line 31)
        str_31143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'str', 'foo1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 28), tuple_31141, str_31143)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 26), list_31140, tuple_31141)
        # Adding element type (line 31)
        
        # Obtaining an instance of the builtin type 'tuple' (line 31)
        tuple_31144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 31)
        # Adding element type (line 31)
        str_31145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 41), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 41), tuple_31144, str_31145)
        # Adding element type (line 31)
        str_31146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 49), 'str', 'foo2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 41), tuple_31144, str_31146)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 26), list_31140, tuple_31144)
        
        # Processing the call keyword arguments (line 30)
        kwargs_31147 = {}
        # Getting the type of 'self' (line 30)
        self_31135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 30)
        assertRaises_31136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_31135, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 30)
        assertRaises_call_result_31148 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assertRaises_31136, *[DistutilsSetupError_31137, check_library_list_31139, list_31140], **kwargs_31147)
        
        
        # Call to assertRaises(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'DistutilsSetupError' (line 34)
        DistutilsSetupError_31151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 34)
        cmd_31152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 47), 'cmd', False)
        # Obtaining the member 'check_library_list' of a type (line 34)
        check_library_list_31153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 47), cmd_31152, 'check_library_list')
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_31154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        
        # Obtaining an instance of the builtin type 'tuple' (line 35)
        tuple_31155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 35)
        # Adding element type (line 35)
        str_31156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 28), tuple_31155, str_31156)
        # Adding element type (line 35)
        str_31157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'str', 'foo1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 28), tuple_31155, str_31157)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 26), list_31154, tuple_31155)
        # Adding element type (line 35)
        
        # Obtaining an instance of the builtin type 'tuple' (line 36)
        tuple_31158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 36)
        # Adding element type (line 36)
        str_31159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 28), 'str', 'another/name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 28), tuple_31158, str_31159)
        # Adding element type (line 36)
        str_31160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 44), 'str', 'foo2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 28), tuple_31158, str_31160)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 26), list_31154, tuple_31158)
        
        # Processing the call keyword arguments (line 34)
        kwargs_31161 = {}
        # Getting the type of 'self' (line 34)
        self_31149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 34)
        assertRaises_31150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_31149, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 34)
        assertRaises_call_result_31162 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assertRaises_31150, *[DistutilsSetupError_31151, check_library_list_31153, list_31154], **kwargs_31161)
        
        
        # Call to assertRaises(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'DistutilsSetupError' (line 39)
        DistutilsSetupError_31165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 39)
        cmd_31166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 47), 'cmd', False)
        # Obtaining the member 'check_library_list' of a type (line 39)
        check_library_list_31167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 47), cmd_31166, 'check_library_list')
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_31168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_31169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        # Adding element type (line 40)
        str_31170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 28), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 28), tuple_31169, str_31170)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'dict' (line 40)
        dict_31171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 40)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 28), tuple_31169, dict_31171)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 26), list_31168, tuple_31169)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_31172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        str_31173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 28), 'str', 'another')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 28), tuple_31172, str_31173)
        # Adding element type (line 41)
        str_31174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 39), 'str', 'foo2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 28), tuple_31172, str_31174)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 26), list_31168, tuple_31172)
        
        # Processing the call keyword arguments (line 39)
        kwargs_31175 = {}
        # Getting the type of 'self' (line 39)
        self_31163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 39)
        assertRaises_31164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_31163, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 39)
        assertRaises_call_result_31176 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), assertRaises_31164, *[DistutilsSetupError_31165, check_library_list_31167, list_31168], **kwargs_31175)
        
        
        # Assigning a List to a Name (line 44):
        
        # Assigning a List to a Name (line 44):
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_31177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        # Adding element type (line 44)
        
        # Obtaining an instance of the builtin type 'tuple' (line 44)
        tuple_31178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 44)
        # Adding element type (line 44)
        str_31179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 17), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 17), tuple_31178, str_31179)
        # Adding element type (line 44)
        
        # Obtaining an instance of the builtin type 'dict' (line 44)
        dict_31180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 44)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 17), tuple_31178, dict_31180)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 15), list_31177, tuple_31178)
        # Adding element type (line 44)
        
        # Obtaining an instance of the builtin type 'tuple' (line 44)
        tuple_31181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 44)
        # Adding element type (line 44)
        str_31182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 31), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 31), tuple_31181, str_31182)
        # Adding element type (line 44)
        
        # Obtaining an instance of the builtin type 'dict' (line 44)
        dict_31183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 39), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 44)
        # Adding element type (key, value) (line 44)
        str_31184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 40), 'str', 'ok')
        str_31185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 46), 'str', 'good')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 39), dict_31183, (str_31184, str_31185))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 31), tuple_31181, dict_31183)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 15), list_31177, tuple_31181)
        
        # Assigning a type to the variable 'libs' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'libs', list_31177)
        
        # Call to check_library_list(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'libs' (line 45)
        libs_31188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'libs', False)
        # Processing the call keyword arguments (line 45)
        kwargs_31189 = {}
        # Getting the type of 'cmd' (line 45)
        cmd_31186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'cmd', False)
        # Obtaining the member 'check_library_list' of a type (line 45)
        check_library_list_31187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), cmd_31186, 'check_library_list')
        # Calling check_library_list(args, kwargs) (line 45)
        check_library_list_call_result_31190 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), check_library_list_31187, *[libs_31188], **kwargs_31189)
        
        
        # ################# End of 'test_check_library_dist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_check_library_dist' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_31191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_31191)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_check_library_dist'
        return stypy_return_type_31191


    @norecursion
    def test_get_source_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_get_source_files'
        module_type_store = module_type_store.open_function_context('test_get_source_files', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildCLibTestCase.test_get_source_files.__dict__.__setitem__('stypy_localization', localization)
        BuildCLibTestCase.test_get_source_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildCLibTestCase.test_get_source_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildCLibTestCase.test_get_source_files.__dict__.__setitem__('stypy_function_name', 'BuildCLibTestCase.test_get_source_files')
        BuildCLibTestCase.test_get_source_files.__dict__.__setitem__('stypy_param_names_list', [])
        BuildCLibTestCase.test_get_source_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildCLibTestCase.test_get_source_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildCLibTestCase.test_get_source_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildCLibTestCase.test_get_source_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildCLibTestCase.test_get_source_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildCLibTestCase.test_get_source_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildCLibTestCase.test_get_source_files', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_get_source_files', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_get_source_files(...)' code ##################

        
        # Assigning a Call to a Tuple (line 48):
        
        # Assigning a Subscript to a Name (line 48):
        
        # Obtaining the type of the subscript
        int_31192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'int')
        
        # Call to create_dist(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_31195 = {}
        # Getting the type of 'self' (line 48)
        self_31193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 48)
        create_dist_31194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 24), self_31193, 'create_dist')
        # Calling create_dist(args, kwargs) (line 48)
        create_dist_call_result_31196 = invoke(stypy.reporting.localization.Localization(__file__, 48, 24), create_dist_31194, *[], **kwargs_31195)
        
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___31197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), create_dist_call_result_31196, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 48)
        subscript_call_result_31198 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), getitem___31197, int_31192)
        
        # Assigning a type to the variable 'tuple_var_assignment_31072' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_31072', subscript_call_result_31198)
        
        # Assigning a Subscript to a Name (line 48):
        
        # Obtaining the type of the subscript
        int_31199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'int')
        
        # Call to create_dist(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_31202 = {}
        # Getting the type of 'self' (line 48)
        self_31200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 48)
        create_dist_31201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 24), self_31200, 'create_dist')
        # Calling create_dist(args, kwargs) (line 48)
        create_dist_call_result_31203 = invoke(stypy.reporting.localization.Localization(__file__, 48, 24), create_dist_31201, *[], **kwargs_31202)
        
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___31204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), create_dist_call_result_31203, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 48)
        subscript_call_result_31205 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), getitem___31204, int_31199)
        
        # Assigning a type to the variable 'tuple_var_assignment_31073' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_31073', subscript_call_result_31205)
        
        # Assigning a Name to a Name (line 48):
        # Getting the type of 'tuple_var_assignment_31072' (line 48)
        tuple_var_assignment_31072_31206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_31072')
        # Assigning a type to the variable 'pkg_dir' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'pkg_dir', tuple_var_assignment_31072_31206)
        
        # Assigning a Name to a Name (line 48):
        # Getting the type of 'tuple_var_assignment_31073' (line 48)
        tuple_var_assignment_31073_31207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_31073')
        # Assigning a type to the variable 'dist' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'dist', tuple_var_assignment_31073_31207)
        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to build_clib(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'dist' (line 49)
        dist_31209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'dist', False)
        # Processing the call keyword arguments (line 49)
        kwargs_31210 = {}
        # Getting the type of 'build_clib' (line 49)
        build_clib_31208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 14), 'build_clib', False)
        # Calling build_clib(args, kwargs) (line 49)
        build_clib_call_result_31211 = invoke(stypy.reporting.localization.Localization(__file__, 49, 14), build_clib_31208, *[dist_31209], **kwargs_31210)
        
        # Assigning a type to the variable 'cmd' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'cmd', build_clib_call_result_31211)
        
        # Assigning a List to a Attribute (line 53):
        
        # Assigning a List to a Attribute (line 53):
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_31212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_31213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        str_31214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 26), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 26), tuple_31213, str_31214)
        # Adding element type (line 53)
        
        # Obtaining an instance of the builtin type 'dict' (line 53)
        dict_31215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 34), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 53)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 26), tuple_31213, dict_31215)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 24), list_31212, tuple_31213)
        
        # Getting the type of 'cmd' (line 53)
        cmd_31216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'cmd')
        # Setting the type of the member 'libraries' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), cmd_31216, 'libraries', list_31212)
        
        # Call to assertRaises(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'DistutilsSetupError' (line 54)
        DistutilsSetupError_31219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 54)
        cmd_31220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 47), 'cmd', False)
        # Obtaining the member 'get_source_files' of a type (line 54)
        get_source_files_31221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 47), cmd_31220, 'get_source_files')
        # Processing the call keyword arguments (line 54)
        kwargs_31222 = {}
        # Getting the type of 'self' (line 54)
        self_31217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 54)
        assertRaises_31218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_31217, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 54)
        assertRaises_call_result_31223 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assertRaises_31218, *[DistutilsSetupError_31219, get_source_files_31221], **kwargs_31222)
        
        
        # Assigning a List to a Attribute (line 56):
        
        # Assigning a List to a Attribute (line 56):
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_31224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'tuple' (line 56)
        tuple_31225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 56)
        # Adding element type (line 56)
        str_31226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 26), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 26), tuple_31225, str_31226)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'dict' (line 56)
        dict_31227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 34), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 56)
        # Adding element type (key, value) (line 56)
        str_31228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 35), 'str', 'sources')
        int_31229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 46), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 34), dict_31227, (str_31228, int_31229))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 26), tuple_31225, dict_31227)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 24), list_31224, tuple_31225)
        
        # Getting the type of 'cmd' (line 56)
        cmd_31230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'cmd')
        # Setting the type of the member 'libraries' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), cmd_31230, 'libraries', list_31224)
        
        # Call to assertRaises(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'DistutilsSetupError' (line 57)
        DistutilsSetupError_31233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 57)
        cmd_31234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 47), 'cmd', False)
        # Obtaining the member 'get_source_files' of a type (line 57)
        get_source_files_31235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 47), cmd_31234, 'get_source_files')
        # Processing the call keyword arguments (line 57)
        kwargs_31236 = {}
        # Getting the type of 'self' (line 57)
        self_31231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 57)
        assertRaises_31232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_31231, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 57)
        assertRaises_call_result_31237 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), assertRaises_31232, *[DistutilsSetupError_31233, get_source_files_31235], **kwargs_31236)
        
        
        # Assigning a List to a Attribute (line 59):
        
        # Assigning a List to a Attribute (line 59):
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_31238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        # Adding element type (line 59)
        
        # Obtaining an instance of the builtin type 'tuple' (line 59)
        tuple_31239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 59)
        # Adding element type (line 59)
        str_31240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 26), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 26), tuple_31239, str_31240)
        # Adding element type (line 59)
        
        # Obtaining an instance of the builtin type 'dict' (line 59)
        dict_31241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 34), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 59)
        # Adding element type (key, value) (line 59)
        str_31242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 35), 'str', 'sources')
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_31243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        # Adding element type (line 59)
        str_31244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 47), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 46), list_31243, str_31244)
        # Adding element type (line 59)
        str_31245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 52), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 46), list_31243, str_31245)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 34), dict_31241, (str_31242, list_31243))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 26), tuple_31239, dict_31241)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 24), list_31238, tuple_31239)
        
        # Getting the type of 'cmd' (line 59)
        cmd_31246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'cmd')
        # Setting the type of the member 'libraries' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), cmd_31246, 'libraries', list_31238)
        
        # Call to assertEqual(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to get_source_files(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_31251 = {}
        # Getting the type of 'cmd' (line 60)
        cmd_31249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'cmd', False)
        # Obtaining the member 'get_source_files' of a type (line 60)
        get_source_files_31250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 25), cmd_31249, 'get_source_files')
        # Calling get_source_files(args, kwargs) (line 60)
        get_source_files_call_result_31252 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), get_source_files_31250, *[], **kwargs_31251)
        
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_31253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        str_31254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 50), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 49), list_31253, str_31254)
        # Adding element type (line 60)
        str_31255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 55), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 49), list_31253, str_31255)
        
        # Processing the call keyword arguments (line 60)
        kwargs_31256 = {}
        # Getting the type of 'self' (line 60)
        self_31247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 60)
        assertEqual_31248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_31247, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 60)
        assertEqual_call_result_31257 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assertEqual_31248, *[get_source_files_call_result_31252, list_31253], **kwargs_31256)
        
        
        # Assigning a List to a Attribute (line 62):
        
        # Assigning a List to a Attribute (line 62):
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_31258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'tuple' (line 62)
        tuple_31259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 62)
        # Adding element type (line 62)
        str_31260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 26), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 26), tuple_31259, str_31260)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'dict' (line 62)
        dict_31261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 34), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 62)
        # Adding element type (key, value) (line 62)
        str_31262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 35), 'str', 'sources')
        
        # Obtaining an instance of the builtin type 'tuple' (line 62)
        tuple_31263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 62)
        # Adding element type (line 62)
        str_31264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 47), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 47), tuple_31263, str_31264)
        # Adding element type (line 62)
        str_31265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 52), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 47), tuple_31263, str_31265)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 34), dict_31261, (str_31262, tuple_31263))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 26), tuple_31259, dict_31261)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 24), list_31258, tuple_31259)
        
        # Getting the type of 'cmd' (line 62)
        cmd_31266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'cmd')
        # Setting the type of the member 'libraries' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), cmd_31266, 'libraries', list_31258)
        
        # Call to assertEqual(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to get_source_files(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_31271 = {}
        # Getting the type of 'cmd' (line 63)
        cmd_31269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 25), 'cmd', False)
        # Obtaining the member 'get_source_files' of a type (line 63)
        get_source_files_31270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 25), cmd_31269, 'get_source_files')
        # Calling get_source_files(args, kwargs) (line 63)
        get_source_files_call_result_31272 = invoke(stypy.reporting.localization.Localization(__file__, 63, 25), get_source_files_31270, *[], **kwargs_31271)
        
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_31273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        str_31274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 50), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 49), list_31273, str_31274)
        # Adding element type (line 63)
        str_31275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 55), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 49), list_31273, str_31275)
        
        # Processing the call keyword arguments (line 63)
        kwargs_31276 = {}
        # Getting the type of 'self' (line 63)
        self_31267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 63)
        assertEqual_31268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_31267, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 63)
        assertEqual_call_result_31277 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), assertEqual_31268, *[get_source_files_call_result_31272, list_31273], **kwargs_31276)
        
        
        # Assigning a List to a Attribute (line 65):
        
        # Assigning a List to a Attribute (line 65):
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_31278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'tuple' (line 65)
        tuple_31279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 65)
        # Adding element type (line 65)
        str_31280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 26), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 26), tuple_31279, str_31280)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'dict' (line 65)
        dict_31281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 34), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 65)
        # Adding element type (key, value) (line 65)
        str_31282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 35), 'str', 'sources')
        
        # Obtaining an instance of the builtin type 'tuple' (line 65)
        tuple_31283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 65)
        # Adding element type (line 65)
        str_31284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 47), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 47), tuple_31283, str_31284)
        # Adding element type (line 65)
        str_31285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 52), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 47), tuple_31283, str_31285)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 34), dict_31281, (str_31282, tuple_31283))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 26), tuple_31279, dict_31281)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), list_31278, tuple_31279)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_31286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        # Adding element type (line 66)
        str_31287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 26), 'str', 'name2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 26), tuple_31286, str_31287)
        # Adding element type (line 66)
        
        # Obtaining an instance of the builtin type 'dict' (line 66)
        dict_31288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 35), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 66)
        # Adding element type (key, value) (line 66)
        str_31289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 36), 'str', 'sources')
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_31290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        str_31291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 48), 'str', 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 47), list_31290, str_31291)
        # Adding element type (line 66)
        str_31292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 53), 'str', 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 47), list_31290, str_31292)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 35), dict_31288, (str_31289, list_31290))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 26), tuple_31286, dict_31288)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), list_31278, tuple_31286)
        
        # Getting the type of 'cmd' (line 65)
        cmd_31293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'cmd')
        # Setting the type of the member 'libraries' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), cmd_31293, 'libraries', list_31278)
        
        # Call to assertEqual(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to get_source_files(...): (line 67)
        # Processing the call keyword arguments (line 67)
        kwargs_31298 = {}
        # Getting the type of 'cmd' (line 67)
        cmd_31296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'cmd', False)
        # Obtaining the member 'get_source_files' of a type (line 67)
        get_source_files_31297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 25), cmd_31296, 'get_source_files')
        # Calling get_source_files(args, kwargs) (line 67)
        get_source_files_call_result_31299 = invoke(stypy.reporting.localization.Localization(__file__, 67, 25), get_source_files_31297, *[], **kwargs_31298)
        
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_31300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        str_31301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 50), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 49), list_31300, str_31301)
        # Adding element type (line 67)
        str_31302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 55), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 49), list_31300, str_31302)
        # Adding element type (line 67)
        str_31303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 60), 'str', 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 49), list_31300, str_31303)
        # Adding element type (line 67)
        str_31304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 65), 'str', 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 49), list_31300, str_31304)
        
        # Processing the call keyword arguments (line 67)
        kwargs_31305 = {}
        # Getting the type of 'self' (line 67)
        self_31294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 67)
        assertEqual_31295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_31294, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 67)
        assertEqual_call_result_31306 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), assertEqual_31295, *[get_source_files_call_result_31299, list_31300], **kwargs_31305)
        
        
        # ################# End of 'test_get_source_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_get_source_files' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_31307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_31307)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_get_source_files'
        return stypy_return_type_31307


    @norecursion
    def test_build_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_build_libraries'
        module_type_store = module_type_store.open_function_context('test_build_libraries', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildCLibTestCase.test_build_libraries.__dict__.__setitem__('stypy_localization', localization)
        BuildCLibTestCase.test_build_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildCLibTestCase.test_build_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildCLibTestCase.test_build_libraries.__dict__.__setitem__('stypy_function_name', 'BuildCLibTestCase.test_build_libraries')
        BuildCLibTestCase.test_build_libraries.__dict__.__setitem__('stypy_param_names_list', [])
        BuildCLibTestCase.test_build_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildCLibTestCase.test_build_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildCLibTestCase.test_build_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildCLibTestCase.test_build_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildCLibTestCase.test_build_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildCLibTestCase.test_build_libraries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildCLibTestCase.test_build_libraries', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_build_libraries', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_build_libraries(...)' code ##################

        
        # Assigning a Call to a Tuple (line 71):
        
        # Assigning a Subscript to a Name (line 71):
        
        # Obtaining the type of the subscript
        int_31308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'int')
        
        # Call to create_dist(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_31311 = {}
        # Getting the type of 'self' (line 71)
        self_31309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 71)
        create_dist_31310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 24), self_31309, 'create_dist')
        # Calling create_dist(args, kwargs) (line 71)
        create_dist_call_result_31312 = invoke(stypy.reporting.localization.Localization(__file__, 71, 24), create_dist_31310, *[], **kwargs_31311)
        
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___31313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), create_dist_call_result_31312, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_31314 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___31313, int_31308)
        
        # Assigning a type to the variable 'tuple_var_assignment_31074' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_31074', subscript_call_result_31314)
        
        # Assigning a Subscript to a Name (line 71):
        
        # Obtaining the type of the subscript
        int_31315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'int')
        
        # Call to create_dist(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_31318 = {}
        # Getting the type of 'self' (line 71)
        self_31316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 71)
        create_dist_31317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 24), self_31316, 'create_dist')
        # Calling create_dist(args, kwargs) (line 71)
        create_dist_call_result_31319 = invoke(stypy.reporting.localization.Localization(__file__, 71, 24), create_dist_31317, *[], **kwargs_31318)
        
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___31320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), create_dist_call_result_31319, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_31321 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___31320, int_31315)
        
        # Assigning a type to the variable 'tuple_var_assignment_31075' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_31075', subscript_call_result_31321)
        
        # Assigning a Name to a Name (line 71):
        # Getting the type of 'tuple_var_assignment_31074' (line 71)
        tuple_var_assignment_31074_31322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_31074')
        # Assigning a type to the variable 'pkg_dir' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'pkg_dir', tuple_var_assignment_31074_31322)
        
        # Assigning a Name to a Name (line 71):
        # Getting the type of 'tuple_var_assignment_31075' (line 71)
        tuple_var_assignment_31075_31323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'tuple_var_assignment_31075')
        # Assigning a type to the variable 'dist' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'dist', tuple_var_assignment_31075_31323)
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to build_clib(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'dist' (line 72)
        dist_31325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'dist', False)
        # Processing the call keyword arguments (line 72)
        kwargs_31326 = {}
        # Getting the type of 'build_clib' (line 72)
        build_clib_31324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 14), 'build_clib', False)
        # Calling build_clib(args, kwargs) (line 72)
        build_clib_call_result_31327 = invoke(stypy.reporting.localization.Localization(__file__, 72, 14), build_clib_31324, *[dist_31325], **kwargs_31326)
        
        # Assigning a type to the variable 'cmd' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'cmd', build_clib_call_result_31327)
        # Declaration of the 'FakeCompiler' class

        class FakeCompiler:

            @norecursion
            def compile(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'compile'
                module_type_store = module_type_store.open_function_context('compile', 74, 12, False)
                # Assigning a type to the variable 'self' (line 75)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                FakeCompiler.compile.__dict__.__setitem__('stypy_localization', localization)
                FakeCompiler.compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                FakeCompiler.compile.__dict__.__setitem__('stypy_type_store', module_type_store)
                FakeCompiler.compile.__dict__.__setitem__('stypy_function_name', 'FakeCompiler.compile')
                FakeCompiler.compile.__dict__.__setitem__('stypy_param_names_list', [])
                FakeCompiler.compile.__dict__.__setitem__('stypy_varargs_param_name', 'args')
                FakeCompiler.compile.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
                FakeCompiler.compile.__dict__.__setitem__('stypy_call_defaults', defaults)
                FakeCompiler.compile.__dict__.__setitem__('stypy_call_varargs', varargs)
                FakeCompiler.compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                FakeCompiler.compile.__dict__.__setitem__('stypy_declared_arg_number', 0)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeCompiler.compile', [], 'args', 'kw', defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'compile', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'compile(...)' code ##################

                pass
                
                # ################# End of 'compile(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'compile' in the type store
                # Getting the type of 'stypy_return_type' (line 74)
                stypy_return_type_31328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_31328)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'compile'
                return stypy_return_type_31328

            
            # Assigning a Name to a Name (line 76):
            
            # Assigning a Name to a Name (line 76):
            # Getting the type of 'compile' (line 76)
            compile_31329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 32), 'compile')
            # Assigning a type to the variable 'create_static_lib' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'create_static_lib', compile_31329)
        
        # Assigning a type to the variable 'FakeCompiler' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'FakeCompiler', FakeCompiler)
        
        # Assigning a Call to a Attribute (line 78):
        
        # Assigning a Call to a Attribute (line 78):
        
        # Call to FakeCompiler(...): (line 78)
        # Processing the call keyword arguments (line 78)
        kwargs_31331 = {}
        # Getting the type of 'FakeCompiler' (line 78)
        FakeCompiler_31330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'FakeCompiler', False)
        # Calling FakeCompiler(args, kwargs) (line 78)
        FakeCompiler_call_result_31332 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), FakeCompiler_31330, *[], **kwargs_31331)
        
        # Getting the type of 'cmd' (line 78)
        cmd_31333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'cmd')
        # Setting the type of the member 'compiler' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), cmd_31333, 'compiler', FakeCompiler_call_result_31332)
        
        # Assigning a List to a Name (line 81):
        
        # Assigning a List to a Name (line 81):
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_31334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        
        # Obtaining an instance of the builtin type 'tuple' (line 81)
        tuple_31335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 81)
        # Adding element type (line 81)
        str_31336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 16), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 16), tuple_31335, str_31336)
        # Adding element type (line 81)
        
        # Obtaining an instance of the builtin type 'dict' (line 81)
        dict_31337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 81)
        # Adding element type (key, value) (line 81)
        str_31338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'str', 'sources')
        str_31339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 36), 'str', 'notvalid')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 24), dict_31337, (str_31338, str_31339))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 16), tuple_31335, dict_31337)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 14), list_31334, tuple_31335)
        
        # Assigning a type to the variable 'lib' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'lib', list_31334)
        
        # Call to assertRaises(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'DistutilsSetupError' (line 82)
        DistutilsSetupError_31342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 82)
        cmd_31343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 47), 'cmd', False)
        # Obtaining the member 'build_libraries' of a type (line 82)
        build_libraries_31344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 47), cmd_31343, 'build_libraries')
        # Getting the type of 'lib' (line 82)
        lib_31345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 68), 'lib', False)
        # Processing the call keyword arguments (line 82)
        kwargs_31346 = {}
        # Getting the type of 'self' (line 82)
        self_31340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 82)
        assertRaises_31341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_31340, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 82)
        assertRaises_call_result_31347 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), assertRaises_31341, *[DistutilsSetupError_31342, build_libraries_31344, lib_31345], **kwargs_31346)
        
        
        # Assigning a List to a Name (line 84):
        
        # Assigning a List to a Name (line 84):
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_31348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        
        # Obtaining an instance of the builtin type 'tuple' (line 84)
        tuple_31349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 84)
        # Adding element type (line 84)
        str_31350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 16), tuple_31349, str_31350)
        # Adding element type (line 84)
        
        # Obtaining an instance of the builtin type 'dict' (line 84)
        dict_31351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 84)
        # Adding element type (key, value) (line 84)
        str_31352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 25), 'str', 'sources')
        
        # Call to list(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_31354 = {}
        # Getting the type of 'list' (line 84)
        list_31353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 36), 'list', False)
        # Calling list(args, kwargs) (line 84)
        list_call_result_31355 = invoke(stypy.reporting.localization.Localization(__file__, 84, 36), list_31353, *[], **kwargs_31354)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 24), dict_31351, (str_31352, list_call_result_31355))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 16), tuple_31349, dict_31351)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 14), list_31348, tuple_31349)
        
        # Assigning a type to the variable 'lib' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'lib', list_31348)
        
        # Call to build_libraries(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'lib' (line 85)
        lib_31358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'lib', False)
        # Processing the call keyword arguments (line 85)
        kwargs_31359 = {}
        # Getting the type of 'cmd' (line 85)
        cmd_31356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'cmd', False)
        # Obtaining the member 'build_libraries' of a type (line 85)
        build_libraries_31357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), cmd_31356, 'build_libraries')
        # Calling build_libraries(args, kwargs) (line 85)
        build_libraries_call_result_31360 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), build_libraries_31357, *[lib_31358], **kwargs_31359)
        
        
        # Assigning a List to a Name (line 87):
        
        # Assigning a List to a Name (line 87):
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_31361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        
        # Obtaining an instance of the builtin type 'tuple' (line 87)
        tuple_31362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 87)
        # Adding element type (line 87)
        str_31363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 16), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 16), tuple_31362, str_31363)
        # Adding element type (line 87)
        
        # Obtaining an instance of the builtin type 'dict' (line 87)
        dict_31364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 87)
        # Adding element type (key, value) (line 87)
        str_31365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 25), 'str', 'sources')
        
        # Call to tuple(...): (line 87)
        # Processing the call keyword arguments (line 87)
        kwargs_31367 = {}
        # Getting the type of 'tuple' (line 87)
        tuple_31366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 36), 'tuple', False)
        # Calling tuple(args, kwargs) (line 87)
        tuple_call_result_31368 = invoke(stypy.reporting.localization.Localization(__file__, 87, 36), tuple_31366, *[], **kwargs_31367)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 24), dict_31364, (str_31365, tuple_call_result_31368))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 16), tuple_31362, dict_31364)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 14), list_31361, tuple_31362)
        
        # Assigning a type to the variable 'lib' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'lib', list_31361)
        
        # Call to build_libraries(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'lib' (line 88)
        lib_31371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 28), 'lib', False)
        # Processing the call keyword arguments (line 88)
        kwargs_31372 = {}
        # Getting the type of 'cmd' (line 88)
        cmd_31369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'cmd', False)
        # Obtaining the member 'build_libraries' of a type (line 88)
        build_libraries_31370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), cmd_31369, 'build_libraries')
        # Calling build_libraries(args, kwargs) (line 88)
        build_libraries_call_result_31373 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), build_libraries_31370, *[lib_31371], **kwargs_31372)
        
        
        # ################# End of 'test_build_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_build_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_31374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_31374)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_build_libraries'
        return stypy_return_type_31374


    @norecursion
    def test_finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_finalize_options'
        module_type_store = module_type_store.open_function_context('test_finalize_options', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildCLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_localization', localization)
        BuildCLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildCLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildCLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_function_name', 'BuildCLibTestCase.test_finalize_options')
        BuildCLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        BuildCLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildCLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildCLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildCLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildCLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildCLibTestCase.test_finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildCLibTestCase.test_finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 91):
        
        # Assigning a Subscript to a Name (line 91):
        
        # Obtaining the type of the subscript
        int_31375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        
        # Call to create_dist(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_31378 = {}
        # Getting the type of 'self' (line 91)
        self_31376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 91)
        create_dist_31377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 24), self_31376, 'create_dist')
        # Calling create_dist(args, kwargs) (line 91)
        create_dist_call_result_31379 = invoke(stypy.reporting.localization.Localization(__file__, 91, 24), create_dist_31377, *[], **kwargs_31378)
        
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___31380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), create_dist_call_result_31379, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_31381 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), getitem___31380, int_31375)
        
        # Assigning a type to the variable 'tuple_var_assignment_31076' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_31076', subscript_call_result_31381)
        
        # Assigning a Subscript to a Name (line 91):
        
        # Obtaining the type of the subscript
        int_31382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        
        # Call to create_dist(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_31385 = {}
        # Getting the type of 'self' (line 91)
        self_31383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 91)
        create_dist_31384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 24), self_31383, 'create_dist')
        # Calling create_dist(args, kwargs) (line 91)
        create_dist_call_result_31386 = invoke(stypy.reporting.localization.Localization(__file__, 91, 24), create_dist_31384, *[], **kwargs_31385)
        
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___31387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), create_dist_call_result_31386, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_31388 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), getitem___31387, int_31382)
        
        # Assigning a type to the variable 'tuple_var_assignment_31077' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_31077', subscript_call_result_31388)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'tuple_var_assignment_31076' (line 91)
        tuple_var_assignment_31076_31389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_31076')
        # Assigning a type to the variable 'pkg_dir' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'pkg_dir', tuple_var_assignment_31076_31389)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'tuple_var_assignment_31077' (line 91)
        tuple_var_assignment_31077_31390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_31077')
        # Assigning a type to the variable 'dist' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 17), 'dist', tuple_var_assignment_31077_31390)
        
        # Assigning a Call to a Name (line 92):
        
        # Assigning a Call to a Name (line 92):
        
        # Call to build_clib(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'dist' (line 92)
        dist_31392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'dist', False)
        # Processing the call keyword arguments (line 92)
        kwargs_31393 = {}
        # Getting the type of 'build_clib' (line 92)
        build_clib_31391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'build_clib', False)
        # Calling build_clib(args, kwargs) (line 92)
        build_clib_call_result_31394 = invoke(stypy.reporting.localization.Localization(__file__, 92, 14), build_clib_31391, *[dist_31392], **kwargs_31393)
        
        # Assigning a type to the variable 'cmd' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'cmd', build_clib_call_result_31394)
        
        # Assigning a Str to a Attribute (line 94):
        
        # Assigning a Str to a Attribute (line 94):
        str_31395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 27), 'str', 'one-dir')
        # Getting the type of 'cmd' (line 94)
        cmd_31396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'cmd')
        # Setting the type of the member 'include_dirs' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), cmd_31396, 'include_dirs', str_31395)
        
        # Call to finalize_options(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_31399 = {}
        # Getting the type of 'cmd' (line 95)
        cmd_31397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 95)
        finalize_options_31398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), cmd_31397, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 95)
        finalize_options_call_result_31400 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), finalize_options_31398, *[], **kwargs_31399)
        
        
        # Call to assertEqual(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'cmd' (line 96)
        cmd_31403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'cmd', False)
        # Obtaining the member 'include_dirs' of a type (line 96)
        include_dirs_31404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 25), cmd_31403, 'include_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_31405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        str_31406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 44), 'str', 'one-dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 43), list_31405, str_31406)
        
        # Processing the call keyword arguments (line 96)
        kwargs_31407 = {}
        # Getting the type of 'self' (line 96)
        self_31401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 96)
        assertEqual_31402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_31401, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 96)
        assertEqual_call_result_31408 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), assertEqual_31402, *[include_dirs_31404, list_31405], **kwargs_31407)
        
        
        # Assigning a Name to a Attribute (line 98):
        
        # Assigning a Name to a Attribute (line 98):
        # Getting the type of 'None' (line 98)
        None_31409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'None')
        # Getting the type of 'cmd' (line 98)
        cmd_31410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'cmd')
        # Setting the type of the member 'include_dirs' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), cmd_31410, 'include_dirs', None_31409)
        
        # Call to finalize_options(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_31413 = {}
        # Getting the type of 'cmd' (line 99)
        cmd_31411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 99)
        finalize_options_31412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), cmd_31411, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 99)
        finalize_options_call_result_31414 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), finalize_options_31412, *[], **kwargs_31413)
        
        
        # Call to assertEqual(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'cmd' (line 100)
        cmd_31417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'cmd', False)
        # Obtaining the member 'include_dirs' of a type (line 100)
        include_dirs_31418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 25), cmd_31417, 'include_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_31419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        
        # Processing the call keyword arguments (line 100)
        kwargs_31420 = {}
        # Getting the type of 'self' (line 100)
        self_31415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 100)
        assertEqual_31416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_31415, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 100)
        assertEqual_call_result_31421 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), assertEqual_31416, *[include_dirs_31418, list_31419], **kwargs_31420)
        
        
        # Assigning a Str to a Attribute (line 102):
        
        # Assigning a Str to a Attribute (line 102):
        str_31422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 37), 'str', 'WONTWORK')
        # Getting the type of 'cmd' (line 102)
        cmd_31423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 102)
        distribution_31424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), cmd_31423, 'distribution')
        # Setting the type of the member 'libraries' of a type (line 102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), distribution_31424, 'libraries', str_31422)
        
        # Call to assertRaises(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'DistutilsSetupError' (line 103)
        DistutilsSetupError_31427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 103)
        cmd_31428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 47), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 103)
        finalize_options_31429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 47), cmd_31428, 'finalize_options')
        # Processing the call keyword arguments (line 103)
        kwargs_31430 = {}
        # Getting the type of 'self' (line 103)
        self_31425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 103)
        assertRaises_31426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_31425, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 103)
        assertRaises_call_result_31431 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), assertRaises_31426, *[DistutilsSetupError_31427, finalize_options_31429], **kwargs_31430)
        
        
        # ################# End of 'test_finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_31432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_31432)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_finalize_options'
        return stypy_return_type_31432


    @norecursion
    def test_run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_run'
        module_type_store = module_type_store.open_function_context('test_run', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildCLibTestCase.test_run.__dict__.__setitem__('stypy_localization', localization)
        BuildCLibTestCase.test_run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildCLibTestCase.test_run.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildCLibTestCase.test_run.__dict__.__setitem__('stypy_function_name', 'BuildCLibTestCase.test_run')
        BuildCLibTestCase.test_run.__dict__.__setitem__('stypy_param_names_list', [])
        BuildCLibTestCase.test_run.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildCLibTestCase.test_run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildCLibTestCase.test_run.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildCLibTestCase.test_run.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildCLibTestCase.test_run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildCLibTestCase.test_run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildCLibTestCase.test_run', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_run', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_run(...)' code ##################

        
        # Assigning a Call to a Tuple (line 107):
        
        # Assigning a Subscript to a Name (line 107):
        
        # Obtaining the type of the subscript
        int_31433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 8), 'int')
        
        # Call to create_dist(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_31436 = {}
        # Getting the type of 'self' (line 107)
        self_31434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 107)
        create_dist_31435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 24), self_31434, 'create_dist')
        # Calling create_dist(args, kwargs) (line 107)
        create_dist_call_result_31437 = invoke(stypy.reporting.localization.Localization(__file__, 107, 24), create_dist_31435, *[], **kwargs_31436)
        
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___31438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), create_dist_call_result_31437, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_31439 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), getitem___31438, int_31433)
        
        # Assigning a type to the variable 'tuple_var_assignment_31078' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'tuple_var_assignment_31078', subscript_call_result_31439)
        
        # Assigning a Subscript to a Name (line 107):
        
        # Obtaining the type of the subscript
        int_31440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 8), 'int')
        
        # Call to create_dist(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_31443 = {}
        # Getting the type of 'self' (line 107)
        self_31441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 107)
        create_dist_31442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 24), self_31441, 'create_dist')
        # Calling create_dist(args, kwargs) (line 107)
        create_dist_call_result_31444 = invoke(stypy.reporting.localization.Localization(__file__, 107, 24), create_dist_31442, *[], **kwargs_31443)
        
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___31445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), create_dist_call_result_31444, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_31446 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), getitem___31445, int_31440)
        
        # Assigning a type to the variable 'tuple_var_assignment_31079' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'tuple_var_assignment_31079', subscript_call_result_31446)
        
        # Assigning a Name to a Name (line 107):
        # Getting the type of 'tuple_var_assignment_31078' (line 107)
        tuple_var_assignment_31078_31447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'tuple_var_assignment_31078')
        # Assigning a type to the variable 'pkg_dir' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'pkg_dir', tuple_var_assignment_31078_31447)
        
        # Assigning a Name to a Name (line 107):
        # Getting the type of 'tuple_var_assignment_31079' (line 107)
        tuple_var_assignment_31079_31448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'tuple_var_assignment_31079')
        # Assigning a type to the variable 'dist' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'dist', tuple_var_assignment_31079_31448)
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to build_clib(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'dist' (line 108)
        dist_31450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'dist', False)
        # Processing the call keyword arguments (line 108)
        kwargs_31451 = {}
        # Getting the type of 'build_clib' (line 108)
        build_clib_31449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 'build_clib', False)
        # Calling build_clib(args, kwargs) (line 108)
        build_clib_call_result_31452 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), build_clib_31449, *[dist_31450], **kwargs_31451)
        
        # Assigning a type to the variable 'cmd' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'cmd', build_clib_call_result_31452)
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to join(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'pkg_dir' (line 110)
        pkg_dir_31456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'pkg_dir', False)
        str_31457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 38), 'str', 'foo.c')
        # Processing the call keyword arguments (line 110)
        kwargs_31458 = {}
        # Getting the type of 'os' (line 110)
        os_31453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 110)
        path_31454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), os_31453, 'path')
        # Obtaining the member 'join' of a type (line 110)
        join_31455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), path_31454, 'join')
        # Calling join(args, kwargs) (line 110)
        join_call_result_31459 = invoke(stypy.reporting.localization.Localization(__file__, 110, 16), join_31455, *[pkg_dir_31456, str_31457], **kwargs_31458)
        
        # Assigning a type to the variable 'foo_c' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'foo_c', join_call_result_31459)
        
        # Call to write_file(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'foo_c' (line 111)
        foo_c_31462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'foo_c', False)
        str_31463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 31), 'str', 'int main(void) { return 1;}\n')
        # Processing the call keyword arguments (line 111)
        kwargs_31464 = {}
        # Getting the type of 'self' (line 111)
        self_31460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 111)
        write_file_31461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_31460, 'write_file')
        # Calling write_file(args, kwargs) (line 111)
        write_file_call_result_31465 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), write_file_31461, *[foo_c_31462, str_31463], **kwargs_31464)
        
        
        # Assigning a List to a Attribute (line 112):
        
        # Assigning a List to a Attribute (line 112):
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_31466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        
        # Obtaining an instance of the builtin type 'tuple' (line 112)
        tuple_31467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 112)
        # Adding element type (line 112)
        str_31468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 26), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 26), tuple_31467, str_31468)
        # Adding element type (line 112)
        
        # Obtaining an instance of the builtin type 'dict' (line 112)
        dict_31469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 33), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 112)
        # Adding element type (key, value) (line 112)
        str_31470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 34), 'str', 'sources')
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_31471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        # Getting the type of 'foo_c' (line 112)
        foo_c_31472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 46), 'foo_c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 45), list_31471, foo_c_31472)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 33), dict_31469, (str_31470, list_31471))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 26), tuple_31467, dict_31469)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 24), list_31466, tuple_31467)
        
        # Getting the type of 'cmd' (line 112)
        cmd_31473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'cmd')
        # Setting the type of the member 'libraries' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), cmd_31473, 'libraries', list_31466)
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to join(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'pkg_dir' (line 114)
        pkg_dir_31477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 34), 'pkg_dir', False)
        str_31478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 43), 'str', 'build')
        # Processing the call keyword arguments (line 114)
        kwargs_31479 = {}
        # Getting the type of 'os' (line 114)
        os_31474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 114)
        path_31475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 21), os_31474, 'path')
        # Obtaining the member 'join' of a type (line 114)
        join_31476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 21), path_31475, 'join')
        # Calling join(args, kwargs) (line 114)
        join_call_result_31480 = invoke(stypy.reporting.localization.Localization(__file__, 114, 21), join_31476, *[pkg_dir_31477, str_31478], **kwargs_31479)
        
        # Assigning a type to the variable 'build_temp' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'build_temp', join_call_result_31480)
        
        # Call to mkdir(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'build_temp' (line 115)
        build_temp_31483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'build_temp', False)
        # Processing the call keyword arguments (line 115)
        kwargs_31484 = {}
        # Getting the type of 'os' (line 115)
        os_31481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 115)
        mkdir_31482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), os_31481, 'mkdir')
        # Calling mkdir(args, kwargs) (line 115)
        mkdir_call_result_31485 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), mkdir_31482, *[build_temp_31483], **kwargs_31484)
        
        
        # Assigning a Name to a Attribute (line 116):
        
        # Assigning a Name to a Attribute (line 116):
        # Getting the type of 'build_temp' (line 116)
        build_temp_31486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 25), 'build_temp')
        # Getting the type of 'cmd' (line 116)
        cmd_31487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'cmd')
        # Setting the type of the member 'build_temp' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), cmd_31487, 'build_temp', build_temp_31486)
        
        # Assigning a Name to a Attribute (line 117):
        
        # Assigning a Name to a Attribute (line 117):
        # Getting the type of 'build_temp' (line 117)
        build_temp_31488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'build_temp')
        # Getting the type of 'cmd' (line 117)
        cmd_31489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'cmd')
        # Setting the type of the member 'build_clib' of a type (line 117)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), cmd_31489, 'build_clib', build_temp_31488)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 122, 8))
        
        # 'from distutils.ccompiler import new_compiler' statement (line 122)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_31490 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 122, 8), 'distutils.ccompiler')

        if (type(import_31490) is not StypyTypeError):

            if (import_31490 != 'pyd_module'):
                __import__(import_31490)
                sys_modules_31491 = sys.modules[import_31490]
                import_from_module(stypy.reporting.localization.Localization(__file__, 122, 8), 'distutils.ccompiler', sys_modules_31491.module_type_store, module_type_store, ['new_compiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 122, 8), __file__, sys_modules_31491, sys_modules_31491.module_type_store, module_type_store)
            else:
                from distutils.ccompiler import new_compiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 122, 8), 'distutils.ccompiler', None, module_type_store, ['new_compiler'], [new_compiler])

        else:
            # Assigning a type to the variable 'distutils.ccompiler' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'distutils.ccompiler', import_31490)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 123, 8))
        
        # 'from distutils.sysconfig import customize_compiler' statement (line 123)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_31492 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 123, 8), 'distutils.sysconfig')

        if (type(import_31492) is not StypyTypeError):

            if (import_31492 != 'pyd_module'):
                __import__(import_31492)
                sys_modules_31493 = sys.modules[import_31492]
                import_from_module(stypy.reporting.localization.Localization(__file__, 123, 8), 'distutils.sysconfig', sys_modules_31493.module_type_store, module_type_store, ['customize_compiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 123, 8), __file__, sys_modules_31493, sys_modules_31493.module_type_store, module_type_store)
            else:
                from distutils.sysconfig import customize_compiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 123, 8), 'distutils.sysconfig', None, module_type_store, ['customize_compiler'], [customize_compiler])

        else:
            # Assigning a type to the variable 'distutils.sysconfig' (line 123)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'distutils.sysconfig', import_31492)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        
        # Assigning a Call to a Name (line 125):
        
        # Assigning a Call to a Name (line 125):
        
        # Call to new_compiler(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_31495 = {}
        # Getting the type of 'new_compiler' (line 125)
        new_compiler_31494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'new_compiler', False)
        # Calling new_compiler(args, kwargs) (line 125)
        new_compiler_call_result_31496 = invoke(stypy.reporting.localization.Localization(__file__, 125, 19), new_compiler_31494, *[], **kwargs_31495)
        
        # Assigning a type to the variable 'compiler' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'compiler', new_compiler_call_result_31496)
        
        # Call to customize_compiler(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'compiler' (line 126)
        compiler_31498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 27), 'compiler', False)
        # Processing the call keyword arguments (line 126)
        kwargs_31499 = {}
        # Getting the type of 'customize_compiler' (line 126)
        customize_compiler_31497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'customize_compiler', False)
        # Calling customize_compiler(args, kwargs) (line 126)
        customize_compiler_call_result_31500 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), customize_compiler_31497, *[compiler_31498], **kwargs_31499)
        
        
        
        # Call to values(...): (line 127)
        # Processing the call keyword arguments (line 127)
        kwargs_31504 = {}
        # Getting the type of 'compiler' (line 127)
        compiler_31501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'compiler', False)
        # Obtaining the member 'executables' of a type (line 127)
        executables_31502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 20), compiler_31501, 'executables')
        # Obtaining the member 'values' of a type (line 127)
        values_31503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 20), executables_31502, 'values')
        # Calling values(args, kwargs) (line 127)
        values_call_result_31505 = invoke(stypy.reporting.localization.Localization(__file__, 127, 20), values_31503, *[], **kwargs_31504)
        
        # Testing the type of a for loop iterable (line 127)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 127, 8), values_call_result_31505)
        # Getting the type of the for loop variable (line 127)
        for_loop_var_31506 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 127, 8), values_call_result_31505)
        # Assigning a type to the variable 'ccmd' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'ccmd', for_loop_var_31506)
        # SSA begins for a for statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 128)
        # Getting the type of 'ccmd' (line 128)
        ccmd_31507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'ccmd')
        # Getting the type of 'None' (line 128)
        None_31508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'None')
        
        (may_be_31509, more_types_in_union_31510) = may_be_none(ccmd_31507, None_31508)

        if may_be_31509:

            if more_types_in_union_31510:
                # Runtime conditional SSA (line 128)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store


            if more_types_in_union_31510:
                # SSA join for if statement (line 128)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 130)
        
        # Call to find_executable(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Obtaining the type of the subscript
        int_31512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 36), 'int')
        # Getting the type of 'ccmd' (line 130)
        ccmd_31513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 31), 'ccmd', False)
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___31514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 31), ccmd_31513, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_31515 = invoke(stypy.reporting.localization.Localization(__file__, 130, 31), getitem___31514, int_31512)
        
        # Processing the call keyword arguments (line 130)
        kwargs_31516 = {}
        # Getting the type of 'find_executable' (line 130)
        find_executable_31511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'find_executable', False)
        # Calling find_executable(args, kwargs) (line 130)
        find_executable_call_result_31517 = invoke(stypy.reporting.localization.Localization(__file__, 130, 15), find_executable_31511, *[subscript_call_result_31515], **kwargs_31516)
        
        # Getting the type of 'None' (line 130)
        None_31518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 43), 'None')
        
        (may_be_31519, more_types_in_union_31520) = may_be_none(find_executable_call_result_31517, None_31518)

        if may_be_31519:

            if more_types_in_union_31520:
                # Runtime conditional SSA (line 130)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to skipTest(...): (line 131)
            # Processing the call arguments (line 131)
            str_31523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 30), 'str', 'The %r command is not found')
            
            # Obtaining the type of the subscript
            int_31524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 67), 'int')
            # Getting the type of 'ccmd' (line 131)
            ccmd_31525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 62), 'ccmd', False)
            # Obtaining the member '__getitem__' of a type (line 131)
            getitem___31526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 62), ccmd_31525, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 131)
            subscript_call_result_31527 = invoke(stypy.reporting.localization.Localization(__file__, 131, 62), getitem___31526, int_31524)
            
            # Applying the binary operator '%' (line 131)
            result_mod_31528 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 30), '%', str_31523, subscript_call_result_31527)
            
            # Processing the call keyword arguments (line 131)
            kwargs_31529 = {}
            # Getting the type of 'self' (line 131)
            self_31521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'self', False)
            # Obtaining the member 'skipTest' of a type (line 131)
            skipTest_31522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), self_31521, 'skipTest')
            # Calling skipTest(args, kwargs) (line 131)
            skipTest_call_result_31530 = invoke(stypy.reporting.localization.Localization(__file__, 131, 16), skipTest_31522, *[result_mod_31528], **kwargs_31529)
            

            if more_types_in_union_31520:
                # SSA join for if statement (line 130)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to run(...): (line 134)
        # Processing the call keyword arguments (line 134)
        kwargs_31533 = {}
        # Getting the type of 'cmd' (line 134)
        cmd_31531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 134)
        run_31532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), cmd_31531, 'run')
        # Calling run(args, kwargs) (line 134)
        run_call_result_31534 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), run_31532, *[], **kwargs_31533)
        
        
        # Call to assertIn(...): (line 137)
        # Processing the call arguments (line 137)
        str_31537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 22), 'str', 'libfoo.a')
        
        # Call to listdir(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'build_temp' (line 137)
        build_temp_31540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 45), 'build_temp', False)
        # Processing the call keyword arguments (line 137)
        kwargs_31541 = {}
        # Getting the type of 'os' (line 137)
        os_31538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 34), 'os', False)
        # Obtaining the member 'listdir' of a type (line 137)
        listdir_31539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 34), os_31538, 'listdir')
        # Calling listdir(args, kwargs) (line 137)
        listdir_call_result_31542 = invoke(stypy.reporting.localization.Localization(__file__, 137, 34), listdir_31539, *[build_temp_31540], **kwargs_31541)
        
        # Processing the call keyword arguments (line 137)
        kwargs_31543 = {}
        # Getting the type of 'self' (line 137)
        self_31535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 137)
        assertIn_31536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_31535, 'assertIn')
        # Calling assertIn(args, kwargs) (line 137)
        assertIn_call_result_31544 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), assertIn_31536, *[str_31537, listdir_call_result_31542], **kwargs_31543)
        
        
        # ################# End of 'test_run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_run' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_31545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_31545)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_run'
        return stypy_return_type_31545


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildCLibTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BuildCLibTestCase' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'BuildCLibTestCase', BuildCLibTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 139, 0, False)
    
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

    
    # Call to makeSuite(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'BuildCLibTestCase' (line 140)
    BuildCLibTestCase_31548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'BuildCLibTestCase', False)
    # Processing the call keyword arguments (line 140)
    kwargs_31549 = {}
    # Getting the type of 'unittest' (line 140)
    unittest_31546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 140)
    makeSuite_31547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), unittest_31546, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 140)
    makeSuite_call_result_31550 = invoke(stypy.reporting.localization.Localization(__file__, 140, 11), makeSuite_31547, *[BuildCLibTestCase_31548], **kwargs_31549)
    
    # Assigning a type to the variable 'stypy_return_type' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type', makeSuite_call_result_31550)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 139)
    stypy_return_type_31551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31551)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_31551

# Assigning a type to the variable 'test_suite' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 143)
    # Processing the call arguments (line 143)
    
    # Call to test_suite(...): (line 143)
    # Processing the call keyword arguments (line 143)
    kwargs_31554 = {}
    # Getting the type of 'test_suite' (line 143)
    test_suite_31553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 143)
    test_suite_call_result_31555 = invoke(stypy.reporting.localization.Localization(__file__, 143, 17), test_suite_31553, *[], **kwargs_31554)
    
    # Processing the call keyword arguments (line 143)
    kwargs_31556 = {}
    # Getting the type of 'run_unittest' (line 143)
    run_unittest_31552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 143)
    run_unittest_call_result_31557 = invoke(stypy.reporting.localization.Localization(__file__, 143, 4), run_unittest_31552, *[test_suite_call_result_31555], **kwargs_31556)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
