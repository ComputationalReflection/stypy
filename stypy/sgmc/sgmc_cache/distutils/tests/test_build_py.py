
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.build_py.'''
2: 
3: import os
4: import sys
5: import StringIO
6: import unittest
7: 
8: from distutils.command.build_py import build_py
9: from distutils.core import Distribution
10: from distutils.errors import DistutilsFileError
11: 
12: from distutils.tests import support
13: from test.test_support import run_unittest
14: 
15: 
16: class BuildPyTestCase(support.TempdirManager,
17:                       support.LoggingSilencer,
18:                       unittest.TestCase):
19: 
20:     def test_package_data(self):
21:         sources = self.mkdtemp()
22:         f = open(os.path.join(sources, "__init__.py"), "w")
23:         try:
24:             f.write("# Pretend this is a package.")
25:         finally:
26:             f.close()
27:         f = open(os.path.join(sources, "README.txt"), "w")
28:         try:
29:             f.write("Info about this package")
30:         finally:
31:             f.close()
32: 
33:         destination = self.mkdtemp()
34: 
35:         dist = Distribution({"packages": ["pkg"],
36:                              "package_dir": {"pkg": sources}})
37:         # script_name need not exist, it just need to be initialized
38:         dist.script_name = os.path.join(sources, "setup.py")
39:         dist.command_obj["build"] = support.DummyCommand(
40:             force=0,
41:             build_lib=destination)
42:         dist.packages = ["pkg"]
43:         dist.package_data = {"pkg": ["README.txt"]}
44:         dist.package_dir = {"pkg": sources}
45: 
46:         cmd = build_py(dist)
47:         cmd.compile = 1
48:         cmd.ensure_finalized()
49:         self.assertEqual(cmd.package_data, dist.package_data)
50: 
51:         cmd.run()
52: 
53:         # This makes sure the list of outputs includes byte-compiled
54:         # files for Python modules but not for package data files
55:         # (there shouldn't *be* byte-code files for those!).
56:         #
57:         self.assertEqual(len(cmd.get_outputs()), 3)
58:         pkgdest = os.path.join(destination, "pkg")
59:         files = os.listdir(pkgdest)
60:         self.assertIn("__init__.py", files)
61:         self.assertIn("README.txt", files)
62:         # XXX even with -O, distutils writes pyc, not pyo; bug?
63:         if sys.dont_write_bytecode:
64:             self.assertNotIn("__init__.pyc", files)
65:         else:
66:             self.assertIn("__init__.pyc", files)
67: 
68:     def test_empty_package_dir(self):
69:         # See SF 1668596/1720897.
70:         cwd = os.getcwd()
71: 
72:         # create the distribution files.
73:         sources = self.mkdtemp()
74:         open(os.path.join(sources, "__init__.py"), "w").close()
75: 
76:         testdir = os.path.join(sources, "doc")
77:         os.mkdir(testdir)
78:         open(os.path.join(testdir, "testfile"), "w").close()
79: 
80:         os.chdir(sources)
81:         old_stdout = sys.stdout
82:         sys.stdout = StringIO.StringIO()
83: 
84:         try:
85:             dist = Distribution({"packages": ["pkg"],
86:                                  "package_dir": {"pkg": ""},
87:                                  "package_data": {"pkg": ["doc/*"]}})
88:             # script_name need not exist, it just need to be initialized
89:             dist.script_name = os.path.join(sources, "setup.py")
90:             dist.script_args = ["build"]
91:             dist.parse_command_line()
92: 
93:             try:
94:                 dist.run_commands()
95:             except DistutilsFileError:
96:                 self.fail("failed package_data test when package_dir is ''")
97:         finally:
98:             # Restore state.
99:             os.chdir(cwd)
100:             sys.stdout = old_stdout
101: 
102:     def test_dir_in_package_data(self):
103:         '''
104:         A directory in package_data should not be added to the filelist.
105:         '''
106:         # See bug 19286
107:         sources = self.mkdtemp()
108:         pkg_dir = os.path.join(sources, "pkg")
109: 
110:         os.mkdir(pkg_dir)
111:         open(os.path.join(pkg_dir, "__init__.py"), "w").close()
112: 
113:         docdir = os.path.join(pkg_dir, "doc")
114:         os.mkdir(docdir)
115:         open(os.path.join(docdir, "testfile"), "w").close()
116: 
117:         # create the directory that could be incorrectly detected as a file
118:         os.mkdir(os.path.join(docdir, 'otherdir'))
119: 
120:         os.chdir(sources)
121:         dist = Distribution({"packages": ["pkg"],
122:                              "package_data": {"pkg": ["doc/*"]}})
123:         # script_name need not exist, it just need to be initialized
124:         dist.script_name = os.path.join(sources, "setup.py")
125:         dist.script_args = ["build"]
126:         dist.parse_command_line()
127: 
128:         try:
129:             dist.run_commands()
130:         except DistutilsFileError:
131:             self.fail("failed package_data when data dir includes a dir")
132: 
133:     def test_dont_write_bytecode(self):
134:         # makes sure byte_compile is not used
135:         pkg_dir, dist = self.create_dist()
136:         cmd = build_py(dist)
137:         cmd.compile = 1
138:         cmd.optimize = 1
139: 
140:         old_dont_write_bytecode = sys.dont_write_bytecode
141:         sys.dont_write_bytecode = True
142:         try:
143:             cmd.byte_compile([])
144:         finally:
145:             sys.dont_write_bytecode = old_dont_write_bytecode
146: 
147:         self.assertIn('byte-compiling is disabled', self.logs[0][1])
148: 
149: def test_suite():
150:     return unittest.makeSuite(BuildPyTestCase)
151: 
152: if __name__ == "__main__":
153:     run_unittest(test_suite())
154: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_33311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.build_py.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import StringIO' statement (line 5)
import StringIO

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'StringIO', StringIO, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import unittest' statement (line 6)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.command.build_py import build_py' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_33312 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.build_py')

if (type(import_33312) is not StypyTypeError):

    if (import_33312 != 'pyd_module'):
        __import__(import_33312)
        sys_modules_33313 = sys.modules[import_33312]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.build_py', sys_modules_33313.module_type_store, module_type_store, ['build_py'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_33313, sys_modules_33313.module_type_store, module_type_store)
    else:
        from distutils.command.build_py import build_py

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.build_py', None, module_type_store, ['build_py'], [build_py])

else:
    # Assigning a type to the variable 'distutils.command.build_py' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.build_py', import_33312)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.core import Distribution' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_33314 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core')

if (type(import_33314) is not StypyTypeError):

    if (import_33314 != 'pyd_module'):
        __import__(import_33314)
        sys_modules_33315 = sys.modules[import_33314]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core', sys_modules_33315.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_33315, sys_modules_33315.module_type_store, module_type_store)
    else:
        from distutils.core import Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core', None, module_type_store, ['Distribution'], [Distribution])

else:
    # Assigning a type to the variable 'distutils.core' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core', import_33314)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.errors import DistutilsFileError' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_33316 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors')

if (type(import_33316) is not StypyTypeError):

    if (import_33316 != 'pyd_module'):
        __import__(import_33316)
        sys_modules_33317 = sys.modules[import_33316]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', sys_modules_33317.module_type_store, module_type_store, ['DistutilsFileError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_33317, sys_modules_33317.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsFileError

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', None, module_type_store, ['DistutilsFileError'], [DistutilsFileError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', import_33316)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.tests import support' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_33318 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.tests')

if (type(import_33318) is not StypyTypeError):

    if (import_33318 != 'pyd_module'):
        __import__(import_33318)
        sys_modules_33319 = sys.modules[import_33318]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.tests', sys_modules_33319.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_33319, sys_modules_33319.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.tests', import_33318)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from test.test_support import run_unittest' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_33320 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'test.test_support')

if (type(import_33320) is not StypyTypeError):

    if (import_33320 != 'pyd_module'):
        __import__(import_33320)
        sys_modules_33321 = sys.modules[import_33320]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'test.test_support', sys_modules_33321.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_33321, sys_modules_33321.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'test.test_support', import_33320)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'BuildPyTestCase' class
# Getting the type of 'support' (line 16)
support_33322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'support')
# Obtaining the member 'TempdirManager' of a type (line 16)
TempdirManager_33323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 22), support_33322, 'TempdirManager')
# Getting the type of 'support' (line 17)
support_33324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 17)
LoggingSilencer_33325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 22), support_33324, 'LoggingSilencer')
# Getting the type of 'unittest' (line 18)
unittest_33326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'unittest')
# Obtaining the member 'TestCase' of a type (line 18)
TestCase_33327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 22), unittest_33326, 'TestCase')

class BuildPyTestCase(TempdirManager_33323, LoggingSilencer_33325, TestCase_33327, ):

    @norecursion
    def test_package_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_package_data'
        module_type_store = module_type_store.open_function_context('test_package_data', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildPyTestCase.test_package_data.__dict__.__setitem__('stypy_localization', localization)
        BuildPyTestCase.test_package_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildPyTestCase.test_package_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildPyTestCase.test_package_data.__dict__.__setitem__('stypy_function_name', 'BuildPyTestCase.test_package_data')
        BuildPyTestCase.test_package_data.__dict__.__setitem__('stypy_param_names_list', [])
        BuildPyTestCase.test_package_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildPyTestCase.test_package_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildPyTestCase.test_package_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildPyTestCase.test_package_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildPyTestCase.test_package_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildPyTestCase.test_package_data.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildPyTestCase.test_package_data', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_package_data', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_package_data(...)' code ##################

        
        # Assigning a Call to a Name (line 21):
        
        # Assigning a Call to a Name (line 21):
        
        # Call to mkdtemp(...): (line 21)
        # Processing the call keyword arguments (line 21)
        kwargs_33330 = {}
        # Getting the type of 'self' (line 21)
        self_33328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 21)
        mkdtemp_33329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 18), self_33328, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 21)
        mkdtemp_call_result_33331 = invoke(stypy.reporting.localization.Localization(__file__, 21, 18), mkdtemp_33329, *[], **kwargs_33330)
        
        # Assigning a type to the variable 'sources' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'sources', mkdtemp_call_result_33331)
        
        # Assigning a Call to a Name (line 22):
        
        # Assigning a Call to a Name (line 22):
        
        # Call to open(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Call to join(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'sources' (line 22)
        sources_33336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 30), 'sources', False)
        str_33337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 39), 'str', '__init__.py')
        # Processing the call keyword arguments (line 22)
        kwargs_33338 = {}
        # Getting the type of 'os' (line 22)
        os_33333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 22)
        path_33334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), os_33333, 'path')
        # Obtaining the member 'join' of a type (line 22)
        join_33335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), path_33334, 'join')
        # Calling join(args, kwargs) (line 22)
        join_call_result_33339 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), join_33335, *[sources_33336, str_33337], **kwargs_33338)
        
        str_33340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 55), 'str', 'w')
        # Processing the call keyword arguments (line 22)
        kwargs_33341 = {}
        # Getting the type of 'open' (line 22)
        open_33332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'open', False)
        # Calling open(args, kwargs) (line 22)
        open_call_result_33342 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), open_33332, *[join_call_result_33339, str_33340], **kwargs_33341)
        
        # Assigning a type to the variable 'f' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'f', open_call_result_33342)
        
        # Try-finally block (line 23)
        
        # Call to write(...): (line 24)
        # Processing the call arguments (line 24)
        str_33345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'str', '# Pretend this is a package.')
        # Processing the call keyword arguments (line 24)
        kwargs_33346 = {}
        # Getting the type of 'f' (line 24)
        f_33343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 24)
        write_33344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), f_33343, 'write')
        # Calling write(args, kwargs) (line 24)
        write_call_result_33347 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), write_33344, *[str_33345], **kwargs_33346)
        
        
        # finally branch of the try-finally block (line 23)
        
        # Call to close(...): (line 26)
        # Processing the call keyword arguments (line 26)
        kwargs_33350 = {}
        # Getting the type of 'f' (line 26)
        f_33348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 26)
        close_33349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), f_33348, 'close')
        # Calling close(args, kwargs) (line 26)
        close_call_result_33351 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), close_33349, *[], **kwargs_33350)
        
        
        
        # Assigning a Call to a Name (line 27):
        
        # Assigning a Call to a Name (line 27):
        
        # Call to open(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Call to join(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'sources' (line 27)
        sources_33356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 30), 'sources', False)
        str_33357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 39), 'str', 'README.txt')
        # Processing the call keyword arguments (line 27)
        kwargs_33358 = {}
        # Getting the type of 'os' (line 27)
        os_33353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 27)
        path_33354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 17), os_33353, 'path')
        # Obtaining the member 'join' of a type (line 27)
        join_33355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 17), path_33354, 'join')
        # Calling join(args, kwargs) (line 27)
        join_call_result_33359 = invoke(stypy.reporting.localization.Localization(__file__, 27, 17), join_33355, *[sources_33356, str_33357], **kwargs_33358)
        
        str_33360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 54), 'str', 'w')
        # Processing the call keyword arguments (line 27)
        kwargs_33361 = {}
        # Getting the type of 'open' (line 27)
        open_33352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'open', False)
        # Calling open(args, kwargs) (line 27)
        open_call_result_33362 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), open_33352, *[join_call_result_33359, str_33360], **kwargs_33361)
        
        # Assigning a type to the variable 'f' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'f', open_call_result_33362)
        
        # Try-finally block (line 28)
        
        # Call to write(...): (line 29)
        # Processing the call arguments (line 29)
        str_33365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 20), 'str', 'Info about this package')
        # Processing the call keyword arguments (line 29)
        kwargs_33366 = {}
        # Getting the type of 'f' (line 29)
        f_33363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 29)
        write_33364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), f_33363, 'write')
        # Calling write(args, kwargs) (line 29)
        write_call_result_33367 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), write_33364, *[str_33365], **kwargs_33366)
        
        
        # finally branch of the try-finally block (line 28)
        
        # Call to close(...): (line 31)
        # Processing the call keyword arguments (line 31)
        kwargs_33370 = {}
        # Getting the type of 'f' (line 31)
        f_33368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 31)
        close_33369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), f_33368, 'close')
        # Calling close(args, kwargs) (line 31)
        close_call_result_33371 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), close_33369, *[], **kwargs_33370)
        
        
        
        # Assigning a Call to a Name (line 33):
        
        # Assigning a Call to a Name (line 33):
        
        # Call to mkdtemp(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_33374 = {}
        # Getting the type of 'self' (line 33)
        self_33372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 22), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 33)
        mkdtemp_33373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 22), self_33372, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 33)
        mkdtemp_call_result_33375 = invoke(stypy.reporting.localization.Localization(__file__, 33, 22), mkdtemp_33373, *[], **kwargs_33374)
        
        # Assigning a type to the variable 'destination' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'destination', mkdtemp_call_result_33375)
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to Distribution(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Obtaining an instance of the builtin type 'dict' (line 35)
        dict_33377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 35)
        # Adding element type (key, value) (line 35)
        str_33378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 29), 'str', 'packages')
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_33379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        str_33380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 42), 'str', 'pkg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 41), list_33379, str_33380)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 28), dict_33377, (str_33378, list_33379))
        # Adding element type (key, value) (line 35)
        str_33381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 29), 'str', 'package_dir')
        
        # Obtaining an instance of the builtin type 'dict' (line 36)
        dict_33382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 44), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 36)
        # Adding element type (key, value) (line 36)
        str_33383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 45), 'str', 'pkg')
        # Getting the type of 'sources' (line 36)
        sources_33384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 52), 'sources', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 44), dict_33382, (str_33383, sources_33384))
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 28), dict_33377, (str_33381, dict_33382))
        
        # Processing the call keyword arguments (line 35)
        kwargs_33385 = {}
        # Getting the type of 'Distribution' (line 35)
        Distribution_33376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 35)
        Distribution_call_result_33386 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), Distribution_33376, *[dict_33377], **kwargs_33385)
        
        # Assigning a type to the variable 'dist' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'dist', Distribution_call_result_33386)
        
        # Assigning a Call to a Attribute (line 38):
        
        # Assigning a Call to a Attribute (line 38):
        
        # Call to join(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'sources' (line 38)
        sources_33390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 40), 'sources', False)
        str_33391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 49), 'str', 'setup.py')
        # Processing the call keyword arguments (line 38)
        kwargs_33392 = {}
        # Getting the type of 'os' (line 38)
        os_33387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 38)
        path_33388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 27), os_33387, 'path')
        # Obtaining the member 'join' of a type (line 38)
        join_33389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 27), path_33388, 'join')
        # Calling join(args, kwargs) (line 38)
        join_call_result_33393 = invoke(stypy.reporting.localization.Localization(__file__, 38, 27), join_33389, *[sources_33390, str_33391], **kwargs_33392)
        
        # Getting the type of 'dist' (line 38)
        dist_33394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'dist')
        # Setting the type of the member 'script_name' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), dist_33394, 'script_name', join_call_result_33393)
        
        # Assigning a Call to a Subscript (line 39):
        
        # Assigning a Call to a Subscript (line 39):
        
        # Call to DummyCommand(...): (line 39)
        # Processing the call keyword arguments (line 39)
        int_33397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 18), 'int')
        keyword_33398 = int_33397
        # Getting the type of 'destination' (line 41)
        destination_33399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'destination', False)
        keyword_33400 = destination_33399
        kwargs_33401 = {'force': keyword_33398, 'build_lib': keyword_33400}
        # Getting the type of 'support' (line 39)
        support_33395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 36), 'support', False)
        # Obtaining the member 'DummyCommand' of a type (line 39)
        DummyCommand_33396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 36), support_33395, 'DummyCommand')
        # Calling DummyCommand(args, kwargs) (line 39)
        DummyCommand_call_result_33402 = invoke(stypy.reporting.localization.Localization(__file__, 39, 36), DummyCommand_33396, *[], **kwargs_33401)
        
        # Getting the type of 'dist' (line 39)
        dist_33403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'dist')
        # Obtaining the member 'command_obj' of a type (line 39)
        command_obj_33404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), dist_33403, 'command_obj')
        str_33405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'str', 'build')
        # Storing an element on a container (line 39)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 8), command_obj_33404, (str_33405, DummyCommand_call_result_33402))
        
        # Assigning a List to a Attribute (line 42):
        
        # Assigning a List to a Attribute (line 42):
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_33406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        str_33407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'str', 'pkg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 24), list_33406, str_33407)
        
        # Getting the type of 'dist' (line 42)
        dist_33408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'dist')
        # Setting the type of the member 'packages' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), dist_33408, 'packages', list_33406)
        
        # Assigning a Dict to a Attribute (line 43):
        
        # Assigning a Dict to a Attribute (line 43):
        
        # Obtaining an instance of the builtin type 'dict' (line 43)
        dict_33409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 43)
        # Adding element type (key, value) (line 43)
        str_33410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 29), 'str', 'pkg')
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_33411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        str_33412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 37), 'str', 'README.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 36), list_33411, str_33412)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 28), dict_33409, (str_33410, list_33411))
        
        # Getting the type of 'dist' (line 43)
        dist_33413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'dist')
        # Setting the type of the member 'package_data' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), dist_33413, 'package_data', dict_33409)
        
        # Assigning a Dict to a Attribute (line 44):
        
        # Assigning a Dict to a Attribute (line 44):
        
        # Obtaining an instance of the builtin type 'dict' (line 44)
        dict_33414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 44)
        # Adding element type (key, value) (line 44)
        str_33415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 28), 'str', 'pkg')
        # Getting the type of 'sources' (line 44)
        sources_33416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'sources')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 27), dict_33414, (str_33415, sources_33416))
        
        # Getting the type of 'dist' (line 44)
        dist_33417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'dist')
        # Setting the type of the member 'package_dir' of a type (line 44)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), dist_33417, 'package_dir', dict_33414)
        
        # Assigning a Call to a Name (line 46):
        
        # Assigning a Call to a Name (line 46):
        
        # Call to build_py(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'dist' (line 46)
        dist_33419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 23), 'dist', False)
        # Processing the call keyword arguments (line 46)
        kwargs_33420 = {}
        # Getting the type of 'build_py' (line 46)
        build_py_33418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'build_py', False)
        # Calling build_py(args, kwargs) (line 46)
        build_py_call_result_33421 = invoke(stypy.reporting.localization.Localization(__file__, 46, 14), build_py_33418, *[dist_33419], **kwargs_33420)
        
        # Assigning a type to the variable 'cmd' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'cmd', build_py_call_result_33421)
        
        # Assigning a Num to a Attribute (line 47):
        
        # Assigning a Num to a Attribute (line 47):
        int_33422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'int')
        # Getting the type of 'cmd' (line 47)
        cmd_33423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'cmd')
        # Setting the type of the member 'compile' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), cmd_33423, 'compile', int_33422)
        
        # Call to ensure_finalized(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_33426 = {}
        # Getting the type of 'cmd' (line 48)
        cmd_33424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 48)
        ensure_finalized_33425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), cmd_33424, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 48)
        ensure_finalized_call_result_33427 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), ensure_finalized_33425, *[], **kwargs_33426)
        
        
        # Call to assertEqual(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'cmd' (line 49)
        cmd_33430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'cmd', False)
        # Obtaining the member 'package_data' of a type (line 49)
        package_data_33431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 25), cmd_33430, 'package_data')
        # Getting the type of 'dist' (line 49)
        dist_33432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 43), 'dist', False)
        # Obtaining the member 'package_data' of a type (line 49)
        package_data_33433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 43), dist_33432, 'package_data')
        # Processing the call keyword arguments (line 49)
        kwargs_33434 = {}
        # Getting the type of 'self' (line 49)
        self_33428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 49)
        assertEqual_33429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_33428, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 49)
        assertEqual_call_result_33435 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), assertEqual_33429, *[package_data_33431, package_data_33433], **kwargs_33434)
        
        
        # Call to run(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_33438 = {}
        # Getting the type of 'cmd' (line 51)
        cmd_33436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 51)
        run_33437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), cmd_33436, 'run')
        # Calling run(args, kwargs) (line 51)
        run_call_result_33439 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), run_33437, *[], **kwargs_33438)
        
        
        # Call to assertEqual(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Call to len(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Call to get_outputs(...): (line 57)
        # Processing the call keyword arguments (line 57)
        kwargs_33445 = {}
        # Getting the type of 'cmd' (line 57)
        cmd_33443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 'cmd', False)
        # Obtaining the member 'get_outputs' of a type (line 57)
        get_outputs_33444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 29), cmd_33443, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 57)
        get_outputs_call_result_33446 = invoke(stypy.reporting.localization.Localization(__file__, 57, 29), get_outputs_33444, *[], **kwargs_33445)
        
        # Processing the call keyword arguments (line 57)
        kwargs_33447 = {}
        # Getting the type of 'len' (line 57)
        len_33442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'len', False)
        # Calling len(args, kwargs) (line 57)
        len_call_result_33448 = invoke(stypy.reporting.localization.Localization(__file__, 57, 25), len_33442, *[get_outputs_call_result_33446], **kwargs_33447)
        
        int_33449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 49), 'int')
        # Processing the call keyword arguments (line 57)
        kwargs_33450 = {}
        # Getting the type of 'self' (line 57)
        self_33440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 57)
        assertEqual_33441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_33440, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 57)
        assertEqual_call_result_33451 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), assertEqual_33441, *[len_call_result_33448, int_33449], **kwargs_33450)
        
        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Call to join(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'destination' (line 58)
        destination_33455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 31), 'destination', False)
        str_33456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 44), 'str', 'pkg')
        # Processing the call keyword arguments (line 58)
        kwargs_33457 = {}
        # Getting the type of 'os' (line 58)
        os_33452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 58)
        path_33453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 18), os_33452, 'path')
        # Obtaining the member 'join' of a type (line 58)
        join_33454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 18), path_33453, 'join')
        # Calling join(args, kwargs) (line 58)
        join_call_result_33458 = invoke(stypy.reporting.localization.Localization(__file__, 58, 18), join_33454, *[destination_33455, str_33456], **kwargs_33457)
        
        # Assigning a type to the variable 'pkgdest' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'pkgdest', join_call_result_33458)
        
        # Assigning a Call to a Name (line 59):
        
        # Assigning a Call to a Name (line 59):
        
        # Call to listdir(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'pkgdest' (line 59)
        pkgdest_33461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 27), 'pkgdest', False)
        # Processing the call keyword arguments (line 59)
        kwargs_33462 = {}
        # Getting the type of 'os' (line 59)
        os_33459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'os', False)
        # Obtaining the member 'listdir' of a type (line 59)
        listdir_33460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), os_33459, 'listdir')
        # Calling listdir(args, kwargs) (line 59)
        listdir_call_result_33463 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), listdir_33460, *[pkgdest_33461], **kwargs_33462)
        
        # Assigning a type to the variable 'files' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'files', listdir_call_result_33463)
        
        # Call to assertIn(...): (line 60)
        # Processing the call arguments (line 60)
        str_33466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'str', '__init__.py')
        # Getting the type of 'files' (line 60)
        files_33467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'files', False)
        # Processing the call keyword arguments (line 60)
        kwargs_33468 = {}
        # Getting the type of 'self' (line 60)
        self_33464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 60)
        assertIn_33465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_33464, 'assertIn')
        # Calling assertIn(args, kwargs) (line 60)
        assertIn_call_result_33469 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assertIn_33465, *[str_33466, files_33467], **kwargs_33468)
        
        
        # Call to assertIn(...): (line 61)
        # Processing the call arguments (line 61)
        str_33472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 22), 'str', 'README.txt')
        # Getting the type of 'files' (line 61)
        files_33473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 36), 'files', False)
        # Processing the call keyword arguments (line 61)
        kwargs_33474 = {}
        # Getting the type of 'self' (line 61)
        self_33470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 61)
        assertIn_33471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_33470, 'assertIn')
        # Calling assertIn(args, kwargs) (line 61)
        assertIn_call_result_33475 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assertIn_33471, *[str_33472, files_33473], **kwargs_33474)
        
        
        # Getting the type of 'sys' (line 63)
        sys_33476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'sys')
        # Obtaining the member 'dont_write_bytecode' of a type (line 63)
        dont_write_bytecode_33477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 11), sys_33476, 'dont_write_bytecode')
        # Testing the type of an if condition (line 63)
        if_condition_33478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 8), dont_write_bytecode_33477)
        # Assigning a type to the variable 'if_condition_33478' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'if_condition_33478', if_condition_33478)
        # SSA begins for if statement (line 63)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assertNotIn(...): (line 64)
        # Processing the call arguments (line 64)
        str_33481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 29), 'str', '__init__.pyc')
        # Getting the type of 'files' (line 64)
        files_33482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 45), 'files', False)
        # Processing the call keyword arguments (line 64)
        kwargs_33483 = {}
        # Getting the type of 'self' (line 64)
        self_33479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'self', False)
        # Obtaining the member 'assertNotIn' of a type (line 64)
        assertNotIn_33480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), self_33479, 'assertNotIn')
        # Calling assertNotIn(args, kwargs) (line 64)
        assertNotIn_call_result_33484 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), assertNotIn_33480, *[str_33481, files_33482], **kwargs_33483)
        
        # SSA branch for the else part of an if statement (line 63)
        module_type_store.open_ssa_branch('else')
        
        # Call to assertIn(...): (line 66)
        # Processing the call arguments (line 66)
        str_33487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 26), 'str', '__init__.pyc')
        # Getting the type of 'files' (line 66)
        files_33488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 42), 'files', False)
        # Processing the call keyword arguments (line 66)
        kwargs_33489 = {}
        # Getting the type of 'self' (line 66)
        self_33485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 66)
        assertIn_33486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), self_33485, 'assertIn')
        # Calling assertIn(args, kwargs) (line 66)
        assertIn_call_result_33490 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), assertIn_33486, *[str_33487, files_33488], **kwargs_33489)
        
        # SSA join for if statement (line 63)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_package_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_package_data' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_33491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33491)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_package_data'
        return stypy_return_type_33491


    @norecursion
    def test_empty_package_dir(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_empty_package_dir'
        module_type_store = module_type_store.open_function_context('test_empty_package_dir', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildPyTestCase.test_empty_package_dir.__dict__.__setitem__('stypy_localization', localization)
        BuildPyTestCase.test_empty_package_dir.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildPyTestCase.test_empty_package_dir.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildPyTestCase.test_empty_package_dir.__dict__.__setitem__('stypy_function_name', 'BuildPyTestCase.test_empty_package_dir')
        BuildPyTestCase.test_empty_package_dir.__dict__.__setitem__('stypy_param_names_list', [])
        BuildPyTestCase.test_empty_package_dir.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildPyTestCase.test_empty_package_dir.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildPyTestCase.test_empty_package_dir.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildPyTestCase.test_empty_package_dir.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildPyTestCase.test_empty_package_dir.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildPyTestCase.test_empty_package_dir.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildPyTestCase.test_empty_package_dir', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_empty_package_dir', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_empty_package_dir(...)' code ##################

        
        # Assigning a Call to a Name (line 70):
        
        # Assigning a Call to a Name (line 70):
        
        # Call to getcwd(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_33494 = {}
        # Getting the type of 'os' (line 70)
        os_33492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 70)
        getcwd_33493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 14), os_33492, 'getcwd')
        # Calling getcwd(args, kwargs) (line 70)
        getcwd_call_result_33495 = invoke(stypy.reporting.localization.Localization(__file__, 70, 14), getcwd_33493, *[], **kwargs_33494)
        
        # Assigning a type to the variable 'cwd' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'cwd', getcwd_call_result_33495)
        
        # Assigning a Call to a Name (line 73):
        
        # Assigning a Call to a Name (line 73):
        
        # Call to mkdtemp(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_33498 = {}
        # Getting the type of 'self' (line 73)
        self_33496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 73)
        mkdtemp_33497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 18), self_33496, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 73)
        mkdtemp_call_result_33499 = invoke(stypy.reporting.localization.Localization(__file__, 73, 18), mkdtemp_33497, *[], **kwargs_33498)
        
        # Assigning a type to the variable 'sources' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'sources', mkdtemp_call_result_33499)
        
        # Call to close(...): (line 74)
        # Processing the call keyword arguments (line 74)
        kwargs_33512 = {}
        
        # Call to open(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Call to join(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'sources' (line 74)
        sources_33504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'sources', False)
        str_33505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 35), 'str', '__init__.py')
        # Processing the call keyword arguments (line 74)
        kwargs_33506 = {}
        # Getting the type of 'os' (line 74)
        os_33501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'os', False)
        # Obtaining the member 'path' of a type (line 74)
        path_33502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), os_33501, 'path')
        # Obtaining the member 'join' of a type (line 74)
        join_33503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), path_33502, 'join')
        # Calling join(args, kwargs) (line 74)
        join_call_result_33507 = invoke(stypy.reporting.localization.Localization(__file__, 74, 13), join_33503, *[sources_33504, str_33505], **kwargs_33506)
        
        str_33508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 51), 'str', 'w')
        # Processing the call keyword arguments (line 74)
        kwargs_33509 = {}
        # Getting the type of 'open' (line 74)
        open_33500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'open', False)
        # Calling open(args, kwargs) (line 74)
        open_call_result_33510 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), open_33500, *[join_call_result_33507, str_33508], **kwargs_33509)
        
        # Obtaining the member 'close' of a type (line 74)
        close_33511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), open_call_result_33510, 'close')
        # Calling close(args, kwargs) (line 74)
        close_call_result_33513 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), close_33511, *[], **kwargs_33512)
        
        
        # Assigning a Call to a Name (line 76):
        
        # Assigning a Call to a Name (line 76):
        
        # Call to join(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'sources' (line 76)
        sources_33517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'sources', False)
        str_33518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 40), 'str', 'doc')
        # Processing the call keyword arguments (line 76)
        kwargs_33519 = {}
        # Getting the type of 'os' (line 76)
        os_33514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 76)
        path_33515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 18), os_33514, 'path')
        # Obtaining the member 'join' of a type (line 76)
        join_33516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 18), path_33515, 'join')
        # Calling join(args, kwargs) (line 76)
        join_call_result_33520 = invoke(stypy.reporting.localization.Localization(__file__, 76, 18), join_33516, *[sources_33517, str_33518], **kwargs_33519)
        
        # Assigning a type to the variable 'testdir' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'testdir', join_call_result_33520)
        
        # Call to mkdir(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'testdir' (line 77)
        testdir_33523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'testdir', False)
        # Processing the call keyword arguments (line 77)
        kwargs_33524 = {}
        # Getting the type of 'os' (line 77)
        os_33521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 77)
        mkdir_33522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), os_33521, 'mkdir')
        # Calling mkdir(args, kwargs) (line 77)
        mkdir_call_result_33525 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), mkdir_33522, *[testdir_33523], **kwargs_33524)
        
        
        # Call to close(...): (line 78)
        # Processing the call keyword arguments (line 78)
        kwargs_33538 = {}
        
        # Call to open(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Call to join(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'testdir' (line 78)
        testdir_33530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'testdir', False)
        str_33531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 35), 'str', 'testfile')
        # Processing the call keyword arguments (line 78)
        kwargs_33532 = {}
        # Getting the type of 'os' (line 78)
        os_33527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 13), 'os', False)
        # Obtaining the member 'path' of a type (line 78)
        path_33528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 13), os_33527, 'path')
        # Obtaining the member 'join' of a type (line 78)
        join_33529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 13), path_33528, 'join')
        # Calling join(args, kwargs) (line 78)
        join_call_result_33533 = invoke(stypy.reporting.localization.Localization(__file__, 78, 13), join_33529, *[testdir_33530, str_33531], **kwargs_33532)
        
        str_33534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 48), 'str', 'w')
        # Processing the call keyword arguments (line 78)
        kwargs_33535 = {}
        # Getting the type of 'open' (line 78)
        open_33526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'open', False)
        # Calling open(args, kwargs) (line 78)
        open_call_result_33536 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), open_33526, *[join_call_result_33533, str_33534], **kwargs_33535)
        
        # Obtaining the member 'close' of a type (line 78)
        close_33537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), open_call_result_33536, 'close')
        # Calling close(args, kwargs) (line 78)
        close_call_result_33539 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), close_33537, *[], **kwargs_33538)
        
        
        # Call to chdir(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'sources' (line 80)
        sources_33542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'sources', False)
        # Processing the call keyword arguments (line 80)
        kwargs_33543 = {}
        # Getting the type of 'os' (line 80)
        os_33540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 80)
        chdir_33541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), os_33540, 'chdir')
        # Calling chdir(args, kwargs) (line 80)
        chdir_call_result_33544 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), chdir_33541, *[sources_33542], **kwargs_33543)
        
        
        # Assigning a Attribute to a Name (line 81):
        
        # Assigning a Attribute to a Name (line 81):
        # Getting the type of 'sys' (line 81)
        sys_33545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 21), 'sys')
        # Obtaining the member 'stdout' of a type (line 81)
        stdout_33546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 21), sys_33545, 'stdout')
        # Assigning a type to the variable 'old_stdout' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'old_stdout', stdout_33546)
        
        # Assigning a Call to a Attribute (line 82):
        
        # Assigning a Call to a Attribute (line 82):
        
        # Call to StringIO(...): (line 82)
        # Processing the call keyword arguments (line 82)
        kwargs_33549 = {}
        # Getting the type of 'StringIO' (line 82)
        StringIO_33547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'StringIO', False)
        # Obtaining the member 'StringIO' of a type (line 82)
        StringIO_33548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), StringIO_33547, 'StringIO')
        # Calling StringIO(args, kwargs) (line 82)
        StringIO_call_result_33550 = invoke(stypy.reporting.localization.Localization(__file__, 82, 21), StringIO_33548, *[], **kwargs_33549)
        
        # Getting the type of 'sys' (line 82)
        sys_33551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'sys')
        # Setting the type of the member 'stdout' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), sys_33551, 'stdout', StringIO_call_result_33550)
        
        # Try-finally block (line 84)
        
        # Assigning a Call to a Name (line 85):
        
        # Assigning a Call to a Name (line 85):
        
        # Call to Distribution(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Obtaining an instance of the builtin type 'dict' (line 85)
        dict_33553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 32), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 85)
        # Adding element type (key, value) (line 85)
        str_33554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 33), 'str', 'packages')
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_33555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        str_33556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 46), 'str', 'pkg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 45), list_33555, str_33556)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 32), dict_33553, (str_33554, list_33555))
        # Adding element type (key, value) (line 85)
        str_33557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 33), 'str', 'package_dir')
        
        # Obtaining an instance of the builtin type 'dict' (line 86)
        dict_33558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 48), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 86)
        # Adding element type (key, value) (line 86)
        str_33559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 49), 'str', 'pkg')
        str_33560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 56), 'str', '')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 48), dict_33558, (str_33559, str_33560))
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 32), dict_33553, (str_33557, dict_33558))
        # Adding element type (key, value) (line 85)
        str_33561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 33), 'str', 'package_data')
        
        # Obtaining an instance of the builtin type 'dict' (line 87)
        dict_33562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 49), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 87)
        # Adding element type (key, value) (line 87)
        str_33563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 50), 'str', 'pkg')
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_33564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        str_33565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 58), 'str', 'doc/*')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 57), list_33564, str_33565)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 49), dict_33562, (str_33563, list_33564))
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 32), dict_33553, (str_33561, dict_33562))
        
        # Processing the call keyword arguments (line 85)
        kwargs_33566 = {}
        # Getting the type of 'Distribution' (line 85)
        Distribution_33552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 19), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 85)
        Distribution_call_result_33567 = invoke(stypy.reporting.localization.Localization(__file__, 85, 19), Distribution_33552, *[dict_33553], **kwargs_33566)
        
        # Assigning a type to the variable 'dist' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'dist', Distribution_call_result_33567)
        
        # Assigning a Call to a Attribute (line 89):
        
        # Assigning a Call to a Attribute (line 89):
        
        # Call to join(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'sources' (line 89)
        sources_33571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 44), 'sources', False)
        str_33572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 53), 'str', 'setup.py')
        # Processing the call keyword arguments (line 89)
        kwargs_33573 = {}
        # Getting the type of 'os' (line 89)
        os_33568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 89)
        path_33569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 31), os_33568, 'path')
        # Obtaining the member 'join' of a type (line 89)
        join_33570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 31), path_33569, 'join')
        # Calling join(args, kwargs) (line 89)
        join_call_result_33574 = invoke(stypy.reporting.localization.Localization(__file__, 89, 31), join_33570, *[sources_33571, str_33572], **kwargs_33573)
        
        # Getting the type of 'dist' (line 89)
        dist_33575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'dist')
        # Setting the type of the member 'script_name' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), dist_33575, 'script_name', join_call_result_33574)
        
        # Assigning a List to a Attribute (line 90):
        
        # Assigning a List to a Attribute (line 90):
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_33576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        str_33577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 32), 'str', 'build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 31), list_33576, str_33577)
        
        # Getting the type of 'dist' (line 90)
        dist_33578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'dist')
        # Setting the type of the member 'script_args' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), dist_33578, 'script_args', list_33576)
        
        # Call to parse_command_line(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_33581 = {}
        # Getting the type of 'dist' (line 91)
        dist_33579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'dist', False)
        # Obtaining the member 'parse_command_line' of a type (line 91)
        parse_command_line_33580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), dist_33579, 'parse_command_line')
        # Calling parse_command_line(args, kwargs) (line 91)
        parse_command_line_call_result_33582 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), parse_command_line_33580, *[], **kwargs_33581)
        
        
        
        # SSA begins for try-except statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to run_commands(...): (line 94)
        # Processing the call keyword arguments (line 94)
        kwargs_33585 = {}
        # Getting the type of 'dist' (line 94)
        dist_33583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'dist', False)
        # Obtaining the member 'run_commands' of a type (line 94)
        run_commands_33584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 16), dist_33583, 'run_commands')
        # Calling run_commands(args, kwargs) (line 94)
        run_commands_call_result_33586 = invoke(stypy.reporting.localization.Localization(__file__, 94, 16), run_commands_33584, *[], **kwargs_33585)
        
        # SSA branch for the except part of a try statement (line 93)
        # SSA branch for the except 'DistutilsFileError' branch of a try statement (line 93)
        module_type_store.open_ssa_branch('except')
        
        # Call to fail(...): (line 96)
        # Processing the call arguments (line 96)
        str_33589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 26), 'str', "failed package_data test when package_dir is ''")
        # Processing the call keyword arguments (line 96)
        kwargs_33590 = {}
        # Getting the type of 'self' (line 96)
        self_33587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'self', False)
        # Obtaining the member 'fail' of a type (line 96)
        fail_33588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 16), self_33587, 'fail')
        # Calling fail(args, kwargs) (line 96)
        fail_call_result_33591 = invoke(stypy.reporting.localization.Localization(__file__, 96, 16), fail_33588, *[str_33589], **kwargs_33590)
        
        # SSA join for try-except statement (line 93)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # finally branch of the try-finally block (line 84)
        
        # Call to chdir(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'cwd' (line 99)
        cwd_33594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 21), 'cwd', False)
        # Processing the call keyword arguments (line 99)
        kwargs_33595 = {}
        # Getting the type of 'os' (line 99)
        os_33592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 99)
        chdir_33593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), os_33592, 'chdir')
        # Calling chdir(args, kwargs) (line 99)
        chdir_call_result_33596 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), chdir_33593, *[cwd_33594], **kwargs_33595)
        
        
        # Assigning a Name to a Attribute (line 100):
        
        # Assigning a Name to a Attribute (line 100):
        # Getting the type of 'old_stdout' (line 100)
        old_stdout_33597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'old_stdout')
        # Getting the type of 'sys' (line 100)
        sys_33598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'sys')
        # Setting the type of the member 'stdout' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), sys_33598, 'stdout', old_stdout_33597)
        
        
        # ################# End of 'test_empty_package_dir(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_empty_package_dir' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_33599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33599)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_empty_package_dir'
        return stypy_return_type_33599


    @norecursion
    def test_dir_in_package_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dir_in_package_data'
        module_type_store = module_type_store.open_function_context('test_dir_in_package_data', 102, 4, False)
        # Assigning a type to the variable 'self' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildPyTestCase.test_dir_in_package_data.__dict__.__setitem__('stypy_localization', localization)
        BuildPyTestCase.test_dir_in_package_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildPyTestCase.test_dir_in_package_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildPyTestCase.test_dir_in_package_data.__dict__.__setitem__('stypy_function_name', 'BuildPyTestCase.test_dir_in_package_data')
        BuildPyTestCase.test_dir_in_package_data.__dict__.__setitem__('stypy_param_names_list', [])
        BuildPyTestCase.test_dir_in_package_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildPyTestCase.test_dir_in_package_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildPyTestCase.test_dir_in_package_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildPyTestCase.test_dir_in_package_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildPyTestCase.test_dir_in_package_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildPyTestCase.test_dir_in_package_data.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildPyTestCase.test_dir_in_package_data', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dir_in_package_data', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dir_in_package_data(...)' code ##################

        str_33600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, (-1)), 'str', '\n        A directory in package_data should not be added to the filelist.\n        ')
        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to mkdtemp(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_33603 = {}
        # Getting the type of 'self' (line 107)
        self_33601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 107)
        mkdtemp_33602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 18), self_33601, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 107)
        mkdtemp_call_result_33604 = invoke(stypy.reporting.localization.Localization(__file__, 107, 18), mkdtemp_33602, *[], **kwargs_33603)
        
        # Assigning a type to the variable 'sources' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'sources', mkdtemp_call_result_33604)
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to join(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'sources' (line 108)
        sources_33608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 31), 'sources', False)
        str_33609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 40), 'str', 'pkg')
        # Processing the call keyword arguments (line 108)
        kwargs_33610 = {}
        # Getting the type of 'os' (line 108)
        os_33605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 108)
        path_33606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 18), os_33605, 'path')
        # Obtaining the member 'join' of a type (line 108)
        join_33607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 18), path_33606, 'join')
        # Calling join(args, kwargs) (line 108)
        join_call_result_33611 = invoke(stypy.reporting.localization.Localization(__file__, 108, 18), join_33607, *[sources_33608, str_33609], **kwargs_33610)
        
        # Assigning a type to the variable 'pkg_dir' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'pkg_dir', join_call_result_33611)
        
        # Call to mkdir(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'pkg_dir' (line 110)
        pkg_dir_33614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'pkg_dir', False)
        # Processing the call keyword arguments (line 110)
        kwargs_33615 = {}
        # Getting the type of 'os' (line 110)
        os_33612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 110)
        mkdir_33613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), os_33612, 'mkdir')
        # Calling mkdir(args, kwargs) (line 110)
        mkdir_call_result_33616 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), mkdir_33613, *[pkg_dir_33614], **kwargs_33615)
        
        
        # Call to close(...): (line 111)
        # Processing the call keyword arguments (line 111)
        kwargs_33629 = {}
        
        # Call to open(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Call to join(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'pkg_dir' (line 111)
        pkg_dir_33621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'pkg_dir', False)
        str_33622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 35), 'str', '__init__.py')
        # Processing the call keyword arguments (line 111)
        kwargs_33623 = {}
        # Getting the type of 'os' (line 111)
        os_33618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 13), 'os', False)
        # Obtaining the member 'path' of a type (line 111)
        path_33619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 13), os_33618, 'path')
        # Obtaining the member 'join' of a type (line 111)
        join_33620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 13), path_33619, 'join')
        # Calling join(args, kwargs) (line 111)
        join_call_result_33624 = invoke(stypy.reporting.localization.Localization(__file__, 111, 13), join_33620, *[pkg_dir_33621, str_33622], **kwargs_33623)
        
        str_33625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 51), 'str', 'w')
        # Processing the call keyword arguments (line 111)
        kwargs_33626 = {}
        # Getting the type of 'open' (line 111)
        open_33617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'open', False)
        # Calling open(args, kwargs) (line 111)
        open_call_result_33627 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), open_33617, *[join_call_result_33624, str_33625], **kwargs_33626)
        
        # Obtaining the member 'close' of a type (line 111)
        close_33628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), open_call_result_33627, 'close')
        # Calling close(args, kwargs) (line 111)
        close_call_result_33630 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), close_33628, *[], **kwargs_33629)
        
        
        # Assigning a Call to a Name (line 113):
        
        # Assigning a Call to a Name (line 113):
        
        # Call to join(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'pkg_dir' (line 113)
        pkg_dir_33634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'pkg_dir', False)
        str_33635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 39), 'str', 'doc')
        # Processing the call keyword arguments (line 113)
        kwargs_33636 = {}
        # Getting the type of 'os' (line 113)
        os_33631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 113)
        path_33632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 17), os_33631, 'path')
        # Obtaining the member 'join' of a type (line 113)
        join_33633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 17), path_33632, 'join')
        # Calling join(args, kwargs) (line 113)
        join_call_result_33637 = invoke(stypy.reporting.localization.Localization(__file__, 113, 17), join_33633, *[pkg_dir_33634, str_33635], **kwargs_33636)
        
        # Assigning a type to the variable 'docdir' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'docdir', join_call_result_33637)
        
        # Call to mkdir(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'docdir' (line 114)
        docdir_33640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'docdir', False)
        # Processing the call keyword arguments (line 114)
        kwargs_33641 = {}
        # Getting the type of 'os' (line 114)
        os_33638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 114)
        mkdir_33639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), os_33638, 'mkdir')
        # Calling mkdir(args, kwargs) (line 114)
        mkdir_call_result_33642 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), mkdir_33639, *[docdir_33640], **kwargs_33641)
        
        
        # Call to close(...): (line 115)
        # Processing the call keyword arguments (line 115)
        kwargs_33655 = {}
        
        # Call to open(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Call to join(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'docdir' (line 115)
        docdir_33647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 26), 'docdir', False)
        str_33648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 34), 'str', 'testfile')
        # Processing the call keyword arguments (line 115)
        kwargs_33649 = {}
        # Getting the type of 'os' (line 115)
        os_33644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 13), 'os', False)
        # Obtaining the member 'path' of a type (line 115)
        path_33645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 13), os_33644, 'path')
        # Obtaining the member 'join' of a type (line 115)
        join_33646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 13), path_33645, 'join')
        # Calling join(args, kwargs) (line 115)
        join_call_result_33650 = invoke(stypy.reporting.localization.Localization(__file__, 115, 13), join_33646, *[docdir_33647, str_33648], **kwargs_33649)
        
        str_33651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 47), 'str', 'w')
        # Processing the call keyword arguments (line 115)
        kwargs_33652 = {}
        # Getting the type of 'open' (line 115)
        open_33643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'open', False)
        # Calling open(args, kwargs) (line 115)
        open_call_result_33653 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), open_33643, *[join_call_result_33650, str_33651], **kwargs_33652)
        
        # Obtaining the member 'close' of a type (line 115)
        close_33654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), open_call_result_33653, 'close')
        # Calling close(args, kwargs) (line 115)
        close_call_result_33656 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), close_33654, *[], **kwargs_33655)
        
        
        # Call to mkdir(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Call to join(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'docdir' (line 118)
        docdir_33662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 30), 'docdir', False)
        str_33663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 38), 'str', 'otherdir')
        # Processing the call keyword arguments (line 118)
        kwargs_33664 = {}
        # Getting the type of 'os' (line 118)
        os_33659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 118)
        path_33660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 17), os_33659, 'path')
        # Obtaining the member 'join' of a type (line 118)
        join_33661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 17), path_33660, 'join')
        # Calling join(args, kwargs) (line 118)
        join_call_result_33665 = invoke(stypy.reporting.localization.Localization(__file__, 118, 17), join_33661, *[docdir_33662, str_33663], **kwargs_33664)
        
        # Processing the call keyword arguments (line 118)
        kwargs_33666 = {}
        # Getting the type of 'os' (line 118)
        os_33657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 118)
        mkdir_33658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), os_33657, 'mkdir')
        # Calling mkdir(args, kwargs) (line 118)
        mkdir_call_result_33667 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), mkdir_33658, *[join_call_result_33665], **kwargs_33666)
        
        
        # Call to chdir(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'sources' (line 120)
        sources_33670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'sources', False)
        # Processing the call keyword arguments (line 120)
        kwargs_33671 = {}
        # Getting the type of 'os' (line 120)
        os_33668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 120)
        chdir_33669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), os_33668, 'chdir')
        # Calling chdir(args, kwargs) (line 120)
        chdir_call_result_33672 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), chdir_33669, *[sources_33670], **kwargs_33671)
        
        
        # Assigning a Call to a Name (line 121):
        
        # Assigning a Call to a Name (line 121):
        
        # Call to Distribution(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Obtaining an instance of the builtin type 'dict' (line 121)
        dict_33674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 121)
        # Adding element type (key, value) (line 121)
        str_33675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 29), 'str', 'packages')
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_33676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        str_33677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 42), 'str', 'pkg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 41), list_33676, str_33677)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 28), dict_33674, (str_33675, list_33676))
        # Adding element type (key, value) (line 121)
        str_33678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 29), 'str', 'package_data')
        
        # Obtaining an instance of the builtin type 'dict' (line 122)
        dict_33679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 45), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 122)
        # Adding element type (key, value) (line 122)
        str_33680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 46), 'str', 'pkg')
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_33681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        # Adding element type (line 122)
        str_33682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 54), 'str', 'doc/*')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 53), list_33681, str_33682)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 45), dict_33679, (str_33680, list_33681))
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 28), dict_33674, (str_33678, dict_33679))
        
        # Processing the call keyword arguments (line 121)
        kwargs_33683 = {}
        # Getting the type of 'Distribution' (line 121)
        Distribution_33673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 121)
        Distribution_call_result_33684 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), Distribution_33673, *[dict_33674], **kwargs_33683)
        
        # Assigning a type to the variable 'dist' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'dist', Distribution_call_result_33684)
        
        # Assigning a Call to a Attribute (line 124):
        
        # Assigning a Call to a Attribute (line 124):
        
        # Call to join(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'sources' (line 124)
        sources_33688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'sources', False)
        str_33689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 49), 'str', 'setup.py')
        # Processing the call keyword arguments (line 124)
        kwargs_33690 = {}
        # Getting the type of 'os' (line 124)
        os_33685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 124)
        path_33686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 27), os_33685, 'path')
        # Obtaining the member 'join' of a type (line 124)
        join_33687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 27), path_33686, 'join')
        # Calling join(args, kwargs) (line 124)
        join_call_result_33691 = invoke(stypy.reporting.localization.Localization(__file__, 124, 27), join_33687, *[sources_33688, str_33689], **kwargs_33690)
        
        # Getting the type of 'dist' (line 124)
        dist_33692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'dist')
        # Setting the type of the member 'script_name' of a type (line 124)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), dist_33692, 'script_name', join_call_result_33691)
        
        # Assigning a List to a Attribute (line 125):
        
        # Assigning a List to a Attribute (line 125):
        
        # Obtaining an instance of the builtin type 'list' (line 125)
        list_33693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 125)
        # Adding element type (line 125)
        str_33694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 28), 'str', 'build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 27), list_33693, str_33694)
        
        # Getting the type of 'dist' (line 125)
        dist_33695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'dist')
        # Setting the type of the member 'script_args' of a type (line 125)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), dist_33695, 'script_args', list_33693)
        
        # Call to parse_command_line(...): (line 126)
        # Processing the call keyword arguments (line 126)
        kwargs_33698 = {}
        # Getting the type of 'dist' (line 126)
        dist_33696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'dist', False)
        # Obtaining the member 'parse_command_line' of a type (line 126)
        parse_command_line_33697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), dist_33696, 'parse_command_line')
        # Calling parse_command_line(args, kwargs) (line 126)
        parse_command_line_call_result_33699 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), parse_command_line_33697, *[], **kwargs_33698)
        
        
        
        # SSA begins for try-except statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to run_commands(...): (line 129)
        # Processing the call keyword arguments (line 129)
        kwargs_33702 = {}
        # Getting the type of 'dist' (line 129)
        dist_33700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'dist', False)
        # Obtaining the member 'run_commands' of a type (line 129)
        run_commands_33701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), dist_33700, 'run_commands')
        # Calling run_commands(args, kwargs) (line 129)
        run_commands_call_result_33703 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), run_commands_33701, *[], **kwargs_33702)
        
        # SSA branch for the except part of a try statement (line 128)
        # SSA branch for the except 'DistutilsFileError' branch of a try statement (line 128)
        module_type_store.open_ssa_branch('except')
        
        # Call to fail(...): (line 131)
        # Processing the call arguments (line 131)
        str_33706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 22), 'str', 'failed package_data when data dir includes a dir')
        # Processing the call keyword arguments (line 131)
        kwargs_33707 = {}
        # Getting the type of 'self' (line 131)
        self_33704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 131)
        fail_33705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), self_33704, 'fail')
        # Calling fail(args, kwargs) (line 131)
        fail_call_result_33708 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), fail_33705, *[str_33706], **kwargs_33707)
        
        # SSA join for try-except statement (line 128)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dir_in_package_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dir_in_package_data' in the type store
        # Getting the type of 'stypy_return_type' (line 102)
        stypy_return_type_33709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33709)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dir_in_package_data'
        return stypy_return_type_33709


    @norecursion
    def test_dont_write_bytecode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dont_write_bytecode'
        module_type_store = module_type_store.open_function_context('test_dont_write_bytecode', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildPyTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_localization', localization)
        BuildPyTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildPyTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildPyTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_function_name', 'BuildPyTestCase.test_dont_write_bytecode')
        BuildPyTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_param_names_list', [])
        BuildPyTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildPyTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildPyTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildPyTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildPyTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildPyTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildPyTestCase.test_dont_write_bytecode', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 135):
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_33710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'int')
        
        # Call to create_dist(...): (line 135)
        # Processing the call keyword arguments (line 135)
        kwargs_33713 = {}
        # Getting the type of 'self' (line 135)
        self_33711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 135)
        create_dist_33712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 24), self_33711, 'create_dist')
        # Calling create_dist(args, kwargs) (line 135)
        create_dist_call_result_33714 = invoke(stypy.reporting.localization.Localization(__file__, 135, 24), create_dist_33712, *[], **kwargs_33713)
        
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___33715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), create_dist_call_result_33714, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_33716 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), getitem___33715, int_33710)
        
        # Assigning a type to the variable 'tuple_var_assignment_33309' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_33309', subscript_call_result_33716)
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_33717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'int')
        
        # Call to create_dist(...): (line 135)
        # Processing the call keyword arguments (line 135)
        kwargs_33720 = {}
        # Getting the type of 'self' (line 135)
        self_33718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 135)
        create_dist_33719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 24), self_33718, 'create_dist')
        # Calling create_dist(args, kwargs) (line 135)
        create_dist_call_result_33721 = invoke(stypy.reporting.localization.Localization(__file__, 135, 24), create_dist_33719, *[], **kwargs_33720)
        
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___33722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), create_dist_call_result_33721, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_33723 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), getitem___33722, int_33717)
        
        # Assigning a type to the variable 'tuple_var_assignment_33310' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_33310', subscript_call_result_33723)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'tuple_var_assignment_33309' (line 135)
        tuple_var_assignment_33309_33724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_33309')
        # Assigning a type to the variable 'pkg_dir' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'pkg_dir', tuple_var_assignment_33309_33724)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'tuple_var_assignment_33310' (line 135)
        tuple_var_assignment_33310_33725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_33310')
        # Assigning a type to the variable 'dist' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'dist', tuple_var_assignment_33310_33725)
        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to build_py(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'dist' (line 136)
        dist_33727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'dist', False)
        # Processing the call keyword arguments (line 136)
        kwargs_33728 = {}
        # Getting the type of 'build_py' (line 136)
        build_py_33726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 14), 'build_py', False)
        # Calling build_py(args, kwargs) (line 136)
        build_py_call_result_33729 = invoke(stypy.reporting.localization.Localization(__file__, 136, 14), build_py_33726, *[dist_33727], **kwargs_33728)
        
        # Assigning a type to the variable 'cmd' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'cmd', build_py_call_result_33729)
        
        # Assigning a Num to a Attribute (line 137):
        
        # Assigning a Num to a Attribute (line 137):
        int_33730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 22), 'int')
        # Getting the type of 'cmd' (line 137)
        cmd_33731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'cmd')
        # Setting the type of the member 'compile' of a type (line 137)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), cmd_33731, 'compile', int_33730)
        
        # Assigning a Num to a Attribute (line 138):
        
        # Assigning a Num to a Attribute (line 138):
        int_33732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'int')
        # Getting the type of 'cmd' (line 138)
        cmd_33733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'cmd')
        # Setting the type of the member 'optimize' of a type (line 138)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), cmd_33733, 'optimize', int_33732)
        
        # Assigning a Attribute to a Name (line 140):
        
        # Assigning a Attribute to a Name (line 140):
        # Getting the type of 'sys' (line 140)
        sys_33734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 34), 'sys')
        # Obtaining the member 'dont_write_bytecode' of a type (line 140)
        dont_write_bytecode_33735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 34), sys_33734, 'dont_write_bytecode')
        # Assigning a type to the variable 'old_dont_write_bytecode' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'old_dont_write_bytecode', dont_write_bytecode_33735)
        
        # Assigning a Name to a Attribute (line 141):
        
        # Assigning a Name to a Attribute (line 141):
        # Getting the type of 'True' (line 141)
        True_33736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 'True')
        # Getting the type of 'sys' (line 141)
        sys_33737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'sys')
        # Setting the type of the member 'dont_write_bytecode' of a type (line 141)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), sys_33737, 'dont_write_bytecode', True_33736)
        
        # Try-finally block (line 142)
        
        # Call to byte_compile(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_33740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        
        # Processing the call keyword arguments (line 143)
        kwargs_33741 = {}
        # Getting the type of 'cmd' (line 143)
        cmd_33738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'cmd', False)
        # Obtaining the member 'byte_compile' of a type (line 143)
        byte_compile_33739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 12), cmd_33738, 'byte_compile')
        # Calling byte_compile(args, kwargs) (line 143)
        byte_compile_call_result_33742 = invoke(stypy.reporting.localization.Localization(__file__, 143, 12), byte_compile_33739, *[list_33740], **kwargs_33741)
        
        
        # finally branch of the try-finally block (line 142)
        
        # Assigning a Name to a Attribute (line 145):
        
        # Assigning a Name to a Attribute (line 145):
        # Getting the type of 'old_dont_write_bytecode' (line 145)
        old_dont_write_bytecode_33743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 38), 'old_dont_write_bytecode')
        # Getting the type of 'sys' (line 145)
        sys_33744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'sys')
        # Setting the type of the member 'dont_write_bytecode' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), sys_33744, 'dont_write_bytecode', old_dont_write_bytecode_33743)
        
        
        # Call to assertIn(...): (line 147)
        # Processing the call arguments (line 147)
        str_33747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 22), 'str', 'byte-compiling is disabled')
        
        # Obtaining the type of the subscript
        int_33748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 65), 'int')
        
        # Obtaining the type of the subscript
        int_33749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 62), 'int')
        # Getting the type of 'self' (line 147)
        self_33750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 52), 'self', False)
        # Obtaining the member 'logs' of a type (line 147)
        logs_33751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 52), self_33750, 'logs')
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___33752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 52), logs_33751, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 147)
        subscript_call_result_33753 = invoke(stypy.reporting.localization.Localization(__file__, 147, 52), getitem___33752, int_33749)
        
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___33754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 52), subscript_call_result_33753, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 147)
        subscript_call_result_33755 = invoke(stypy.reporting.localization.Localization(__file__, 147, 52), getitem___33754, int_33748)
        
        # Processing the call keyword arguments (line 147)
        kwargs_33756 = {}
        # Getting the type of 'self' (line 147)
        self_33745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 147)
        assertIn_33746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_33745, 'assertIn')
        # Calling assertIn(args, kwargs) (line 147)
        assertIn_call_result_33757 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), assertIn_33746, *[str_33747, subscript_call_result_33755], **kwargs_33756)
        
        
        # ################# End of 'test_dont_write_bytecode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dont_write_bytecode' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_33758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33758)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dont_write_bytecode'
        return stypy_return_type_33758


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 16, 0, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildPyTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BuildPyTestCase' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'BuildPyTestCase', BuildPyTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 149, 0, False)
    
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

    
    # Call to makeSuite(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'BuildPyTestCase' (line 150)
    BuildPyTestCase_33761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'BuildPyTestCase', False)
    # Processing the call keyword arguments (line 150)
    kwargs_33762 = {}
    # Getting the type of 'unittest' (line 150)
    unittest_33759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 150)
    makeSuite_33760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 11), unittest_33759, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 150)
    makeSuite_call_result_33763 = invoke(stypy.reporting.localization.Localization(__file__, 150, 11), makeSuite_33760, *[BuildPyTestCase_33761], **kwargs_33762)
    
    # Assigning a type to the variable 'stypy_return_type' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type', makeSuite_call_result_33763)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 149)
    stypy_return_type_33764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33764)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_33764

# Assigning a type to the variable 'test_suite' (line 149)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 153)
    # Processing the call arguments (line 153)
    
    # Call to test_suite(...): (line 153)
    # Processing the call keyword arguments (line 153)
    kwargs_33767 = {}
    # Getting the type of 'test_suite' (line 153)
    test_suite_33766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 153)
    test_suite_call_result_33768 = invoke(stypy.reporting.localization.Localization(__file__, 153, 17), test_suite_33766, *[], **kwargs_33767)
    
    # Processing the call keyword arguments (line 153)
    kwargs_33769 = {}
    # Getting the type of 'run_unittest' (line 153)
    run_unittest_33765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 153)
    run_unittest_call_result_33770 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), run_unittest_33765, *[test_suite_call_result_33768], **kwargs_33769)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
