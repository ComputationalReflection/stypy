
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.bdist_rpm.'''
2: 
3: import unittest
4: import sys
5: import os
6: import tempfile
7: import shutil
8: 
9: from test.test_support import run_unittest
10: 
11: try:
12:     import zlib
13: except ImportError:
14:     zlib = None
15: 
16: from distutils.core import Distribution
17: from distutils.command.bdist_rpm import bdist_rpm
18: from distutils.tests import support
19: from distutils.spawn import find_executable
20: from distutils import spawn
21: from distutils.errors import DistutilsExecError
22: 
23: SETUP_PY = '''\
24: from distutils.core import setup
25: import foo
26: 
27: setup(name='foo', version='0.1', py_modules=['foo'],
28:       url='xxx', author='xxx', author_email='xxx')
29: 
30: '''
31: 
32: class BuildRpmTestCase(support.TempdirManager,
33:                        support.EnvironGuard,
34:                        support.LoggingSilencer,
35:                        unittest.TestCase):
36: 
37:     def setUp(self):
38:         super(BuildRpmTestCase, self).setUp()
39:         self.old_location = os.getcwd()
40:         self.old_sys_argv = sys.argv, sys.argv[:]
41: 
42:     def tearDown(self):
43:         os.chdir(self.old_location)
44:         sys.argv = self.old_sys_argv[0]
45:         sys.argv[:] = self.old_sys_argv[1]
46:         super(BuildRpmTestCase, self).tearDown()
47: 
48:     # XXX I am unable yet to make this test work without
49:     # spurious sdtout/stderr output under Mac OS X
50:     @unittest.skipUnless(sys.platform.startswith('linux'),
51:                          'spurious sdtout/stderr output under Mac OS X')
52:     @unittest.skipUnless(zlib, "requires zlib")
53:     @unittest.skipIf(find_executable('rpm') is None,
54:                      'the rpm command is not found')
55:     @unittest.skipIf(find_executable('rpmbuild') is None,
56:                      'the rpmbuild command is not found')
57:     def test_quiet(self):
58:         # let's create a package
59:         tmp_dir = self.mkdtemp()
60:         os.environ['HOME'] = tmp_dir   # to confine dir '.rpmdb' creation
61:         pkg_dir = os.path.join(tmp_dir, 'foo')
62:         os.mkdir(pkg_dir)
63:         self.write_file((pkg_dir, 'setup.py'), SETUP_PY)
64:         self.write_file((pkg_dir, 'foo.py'), '#')
65:         self.write_file((pkg_dir, 'MANIFEST.in'), 'include foo.py')
66:         self.write_file((pkg_dir, 'README'), '')
67: 
68:         dist = Distribution({'name': 'foo', 'version': '0.1',
69:                              'py_modules': ['foo'],
70:                              'url': 'xxx', 'author': 'xxx',
71:                              'author_email': 'xxx'})
72:         dist.script_name = 'setup.py'
73:         os.chdir(pkg_dir)
74: 
75:         sys.argv = ['setup.py']
76:         cmd = bdist_rpm(dist)
77:         cmd.fix_python = True
78: 
79:         # running in quiet mode
80:         cmd.quiet = 1
81:         cmd.ensure_finalized()
82:         cmd.run()
83: 
84:         dist_created = os.listdir(os.path.join(pkg_dir, 'dist'))
85:         self.assertIn('foo-0.1-1.noarch.rpm', dist_created)
86: 
87:         # bug #2945: upload ignores bdist_rpm files
88:         self.assertIn(('bdist_rpm', 'any', 'dist/foo-0.1-1.src.rpm'), dist.dist_files)
89:         self.assertIn(('bdist_rpm', 'any', 'dist/foo-0.1-1.noarch.rpm'), dist.dist_files)
90: 
91:     # XXX I am unable yet to make this test work without
92:     # spurious sdtout/stderr output under Mac OS X
93:     @unittest.skipUnless(sys.platform.startswith('linux'),
94:                          'spurious sdtout/stderr output under Mac OS X')
95:     @unittest.skipUnless(zlib, "requires zlib")
96:     # http://bugs.python.org/issue1533164
97:     @unittest.skipIf(find_executable('rpm') is None,
98:                      'the rpm command is not found')
99:     @unittest.skipIf(find_executable('rpmbuild') is None,
100:                      'the rpmbuild command is not found')
101:     def test_no_optimize_flag(self):
102:         # let's create a package that brakes bdist_rpm
103:         tmp_dir = self.mkdtemp()
104:         os.environ['HOME'] = tmp_dir   # to confine dir '.rpmdb' creation
105:         pkg_dir = os.path.join(tmp_dir, 'foo')
106:         os.mkdir(pkg_dir)
107:         self.write_file((pkg_dir, 'setup.py'), SETUP_PY)
108:         self.write_file((pkg_dir, 'foo.py'), '#')
109:         self.write_file((pkg_dir, 'MANIFEST.in'), 'include foo.py')
110:         self.write_file((pkg_dir, 'README'), '')
111: 
112:         dist = Distribution({'name': 'foo', 'version': '0.1',
113:                              'py_modules': ['foo'],
114:                              'url': 'xxx', 'author': 'xxx',
115:                              'author_email': 'xxx'})
116:         dist.script_name = 'setup.py'
117:         os.chdir(pkg_dir)
118: 
119:         sys.argv = ['setup.py']
120:         cmd = bdist_rpm(dist)
121:         cmd.fix_python = True
122: 
123:         cmd.quiet = 1
124:         cmd.ensure_finalized()
125:         cmd.run()
126: 
127:         dist_created = os.listdir(os.path.join(pkg_dir, 'dist'))
128:         self.assertIn('foo-0.1-1.noarch.rpm', dist_created)
129: 
130:         # bug #2945: upload ignores bdist_rpm files
131:         self.assertIn(('bdist_rpm', 'any', 'dist/foo-0.1-1.src.rpm'), dist.dist_files)
132:         self.assertIn(('bdist_rpm', 'any', 'dist/foo-0.1-1.noarch.rpm'), dist.dist_files)
133: 
134:         os.remove(os.path.join(pkg_dir, 'dist', 'foo-0.1-1.noarch.rpm'))
135: 
136: def test_suite():
137:     return unittest.makeSuite(BuildRpmTestCase)
138: 
139: if __name__ == '__main__':
140:     run_unittest(test_suite())
141: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_30457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.bdist_rpm.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import unittest' statement (line 3)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import os' statement (line 5)
import os

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import tempfile' statement (line 6)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'tempfile', tempfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import shutil' statement (line 7)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from test.test_support import run_unittest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30458 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support')

if (type(import_30458) is not StypyTypeError):

    if (import_30458 != 'pyd_module'):
        __import__(import_30458)
        sys_modules_30459 = sys.modules[import_30458]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', sys_modules_30459.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_30459, sys_modules_30459.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', import_30458)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')



# SSA begins for try-except statement (line 11)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 4))

# 'import zlib' statement (line 12)
import zlib

import_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'zlib', zlib, module_type_store)

# SSA branch for the except part of a try statement (line 11)
# SSA branch for the except 'ImportError' branch of a try statement (line 11)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 14):
# Getting the type of 'None' (line 14)
None_30460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'None')
# Assigning a type to the variable 'zlib' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'zlib', None_30460)
# SSA join for try-except statement (line 11)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils.core import Distribution' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30461 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.core')

if (type(import_30461) is not StypyTypeError):

    if (import_30461 != 'pyd_module'):
        __import__(import_30461)
        sys_modules_30462 = sys.modules[import_30461]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.core', sys_modules_30462.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_30462, sys_modules_30462.module_type_store, module_type_store)
    else:
        from distutils.core import Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.core', None, module_type_store, ['Distribution'], [Distribution])

else:
    # Assigning a type to the variable 'distutils.core' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.core', import_30461)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils.command.bdist_rpm import bdist_rpm' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30463 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.command.bdist_rpm')

if (type(import_30463) is not StypyTypeError):

    if (import_30463 != 'pyd_module'):
        __import__(import_30463)
        sys_modules_30464 = sys.modules[import_30463]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.command.bdist_rpm', sys_modules_30464.module_type_store, module_type_store, ['bdist_rpm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_30464, sys_modules_30464.module_type_store, module_type_store)
    else:
        from distutils.command.bdist_rpm import bdist_rpm

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.command.bdist_rpm', None, module_type_store, ['bdist_rpm'], [bdist_rpm])

else:
    # Assigning a type to the variable 'distutils.command.bdist_rpm' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.command.bdist_rpm', import_30463)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from distutils.tests import support' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30465 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.tests')

if (type(import_30465) is not StypyTypeError):

    if (import_30465 != 'pyd_module'):
        __import__(import_30465)
        sys_modules_30466 = sys.modules[import_30465]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.tests', sys_modules_30466.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_30466, sys_modules_30466.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.tests', import_30465)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from distutils.spawn import find_executable' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30467 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.spawn')

if (type(import_30467) is not StypyTypeError):

    if (import_30467 != 'pyd_module'):
        __import__(import_30467)
        sys_modules_30468 = sys.modules[import_30467]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.spawn', sys_modules_30468.module_type_store, module_type_store, ['find_executable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_30468, sys_modules_30468.module_type_store, module_type_store)
    else:
        from distutils.spawn import find_executable

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.spawn', None, module_type_store, ['find_executable'], [find_executable])

else:
    # Assigning a type to the variable 'distutils.spawn' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.spawn', import_30467)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from distutils import spawn' statement (line 20)
try:
    from distutils import spawn

except:
    spawn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils', None, module_type_store, ['spawn'], [spawn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from distutils.errors import DistutilsExecError' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30469 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.errors')

if (type(import_30469) is not StypyTypeError):

    if (import_30469 != 'pyd_module'):
        __import__(import_30469)
        sys_modules_30470 = sys.modules[import_30469]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.errors', sys_modules_30470.module_type_store, module_type_store, ['DistutilsExecError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_30470, sys_modules_30470.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsExecError

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.errors', None, module_type_store, ['DistutilsExecError'], [DistutilsExecError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.errors', import_30469)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Assigning a Str to a Name (line 23):
str_30471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'str', "from distutils.core import setup\nimport foo\n\nsetup(name='foo', version='0.1', py_modules=['foo'],\n      url='xxx', author='xxx', author_email='xxx')\n\n")
# Assigning a type to the variable 'SETUP_PY' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'SETUP_PY', str_30471)
# Declaration of the 'BuildRpmTestCase' class
# Getting the type of 'support' (line 32)
support_30472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'support')
# Obtaining the member 'TempdirManager' of a type (line 32)
TempdirManager_30473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 23), support_30472, 'TempdirManager')
# Getting the type of 'support' (line 33)
support_30474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'support')
# Obtaining the member 'EnvironGuard' of a type (line 33)
EnvironGuard_30475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 23), support_30474, 'EnvironGuard')
# Getting the type of 'support' (line 34)
support_30476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 34)
LoggingSilencer_30477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 23), support_30476, 'LoggingSilencer')
# Getting the type of 'unittest' (line 35)
unittest_30478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'unittest')
# Obtaining the member 'TestCase' of a type (line 35)
TestCase_30479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 23), unittest_30478, 'TestCase')

class BuildRpmTestCase(TempdirManager_30473, EnvironGuard_30475, LoggingSilencer_30477, TestCase_30479, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildRpmTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        BuildRpmTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildRpmTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildRpmTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'BuildRpmTestCase.setUp')
        BuildRpmTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        BuildRpmTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildRpmTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildRpmTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildRpmTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildRpmTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildRpmTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildRpmTestCase.setUp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setUp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setUp(...)' code ##################

        
        # Call to setUp(...): (line 38)
        # Processing the call keyword arguments (line 38)
        kwargs_30486 = {}
        
        # Call to super(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'BuildRpmTestCase' (line 38)
        BuildRpmTestCase_30481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'BuildRpmTestCase', False)
        # Getting the type of 'self' (line 38)
        self_30482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 32), 'self', False)
        # Processing the call keyword arguments (line 38)
        kwargs_30483 = {}
        # Getting the type of 'super' (line 38)
        super_30480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'super', False)
        # Calling super(args, kwargs) (line 38)
        super_call_result_30484 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), super_30480, *[BuildRpmTestCase_30481, self_30482], **kwargs_30483)
        
        # Obtaining the member 'setUp' of a type (line 38)
        setUp_30485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), super_call_result_30484, 'setUp')
        # Calling setUp(args, kwargs) (line 38)
        setUp_call_result_30487 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), setUp_30485, *[], **kwargs_30486)
        
        
        # Assigning a Call to a Attribute (line 39):
        
        # Call to getcwd(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_30490 = {}
        # Getting the type of 'os' (line 39)
        os_30488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 28), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 39)
        getcwd_30489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 28), os_30488, 'getcwd')
        # Calling getcwd(args, kwargs) (line 39)
        getcwd_call_result_30491 = invoke(stypy.reporting.localization.Localization(__file__, 39, 28), getcwd_30489, *[], **kwargs_30490)
        
        # Getting the type of 'self' (line 39)
        self_30492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'old_location' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_30492, 'old_location', getcwd_call_result_30491)
        
        # Assigning a Tuple to a Attribute (line 40):
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_30493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        # Adding element type (line 40)
        # Getting the type of 'sys' (line 40)
        sys_30494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'sys')
        # Obtaining the member 'argv' of a type (line 40)
        argv_30495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 28), sys_30494, 'argv')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 28), tuple_30493, argv_30495)
        # Adding element type (line 40)
        
        # Obtaining the type of the subscript
        slice_30496 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 40, 38), None, None, None)
        # Getting the type of 'sys' (line 40)
        sys_30497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 38), 'sys')
        # Obtaining the member 'argv' of a type (line 40)
        argv_30498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 38), sys_30497, 'argv')
        # Obtaining the member '__getitem__' of a type (line 40)
        getitem___30499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 38), argv_30498, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 40)
        subscript_call_result_30500 = invoke(stypy.reporting.localization.Localization(__file__, 40, 38), getitem___30499, slice_30496)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 28), tuple_30493, subscript_call_result_30500)
        
        # Getting the type of 'self' (line 40)
        self_30501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'old_sys_argv' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_30501, 'old_sys_argv', tuple_30493)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_30502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30502)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_30502


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildRpmTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        BuildRpmTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildRpmTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildRpmTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'BuildRpmTestCase.tearDown')
        BuildRpmTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        BuildRpmTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildRpmTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildRpmTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildRpmTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildRpmTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildRpmTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildRpmTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tearDown', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tearDown(...)' code ##################

        
        # Call to chdir(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'self' (line 43)
        self_30505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'self', False)
        # Obtaining the member 'old_location' of a type (line 43)
        old_location_30506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 17), self_30505, 'old_location')
        # Processing the call keyword arguments (line 43)
        kwargs_30507 = {}
        # Getting the type of 'os' (line 43)
        os_30503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 43)
        chdir_30504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), os_30503, 'chdir')
        # Calling chdir(args, kwargs) (line 43)
        chdir_call_result_30508 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), chdir_30504, *[old_location_30506], **kwargs_30507)
        
        
        # Assigning a Subscript to a Attribute (line 44):
        
        # Obtaining the type of the subscript
        int_30509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 37), 'int')
        # Getting the type of 'self' (line 44)
        self_30510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'self')
        # Obtaining the member 'old_sys_argv' of a type (line 44)
        old_sys_argv_30511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 19), self_30510, 'old_sys_argv')
        # Obtaining the member '__getitem__' of a type (line 44)
        getitem___30512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 19), old_sys_argv_30511, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 44)
        subscript_call_result_30513 = invoke(stypy.reporting.localization.Localization(__file__, 44, 19), getitem___30512, int_30509)
        
        # Getting the type of 'sys' (line 44)
        sys_30514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'sys')
        # Setting the type of the member 'argv' of a type (line 44)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), sys_30514, 'argv', subscript_call_result_30513)
        
        # Assigning a Subscript to a Subscript (line 45):
        
        # Obtaining the type of the subscript
        int_30515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 40), 'int')
        # Getting the type of 'self' (line 45)
        self_30516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 22), 'self')
        # Obtaining the member 'old_sys_argv' of a type (line 45)
        old_sys_argv_30517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 22), self_30516, 'old_sys_argv')
        # Obtaining the member '__getitem__' of a type (line 45)
        getitem___30518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 22), old_sys_argv_30517, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 45)
        subscript_call_result_30519 = invoke(stypy.reporting.localization.Localization(__file__, 45, 22), getitem___30518, int_30515)
        
        # Getting the type of 'sys' (line 45)
        sys_30520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'sys')
        # Obtaining the member 'argv' of a type (line 45)
        argv_30521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), sys_30520, 'argv')
        slice_30522 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 45, 8), None, None, None)
        # Storing an element on a container (line 45)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 8), argv_30521, (slice_30522, subscript_call_result_30519))
        
        # Call to tearDown(...): (line 46)
        # Processing the call keyword arguments (line 46)
        kwargs_30529 = {}
        
        # Call to super(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'BuildRpmTestCase' (line 46)
        BuildRpmTestCase_30524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'BuildRpmTestCase', False)
        # Getting the type of 'self' (line 46)
        self_30525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 32), 'self', False)
        # Processing the call keyword arguments (line 46)
        kwargs_30526 = {}
        # Getting the type of 'super' (line 46)
        super_30523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'super', False)
        # Calling super(args, kwargs) (line 46)
        super_call_result_30527 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), super_30523, *[BuildRpmTestCase_30524, self_30525], **kwargs_30526)
        
        # Obtaining the member 'tearDown' of a type (line 46)
        tearDown_30528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), super_call_result_30527, 'tearDown')
        # Calling tearDown(args, kwargs) (line 46)
        tearDown_call_result_30530 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), tearDown_30528, *[], **kwargs_30529)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_30531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30531)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_30531


    @norecursion
    def test_quiet(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_quiet'
        module_type_store = module_type_store.open_function_context('test_quiet', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildRpmTestCase.test_quiet.__dict__.__setitem__('stypy_localization', localization)
        BuildRpmTestCase.test_quiet.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildRpmTestCase.test_quiet.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildRpmTestCase.test_quiet.__dict__.__setitem__('stypy_function_name', 'BuildRpmTestCase.test_quiet')
        BuildRpmTestCase.test_quiet.__dict__.__setitem__('stypy_param_names_list', [])
        BuildRpmTestCase.test_quiet.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildRpmTestCase.test_quiet.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildRpmTestCase.test_quiet.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildRpmTestCase.test_quiet.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildRpmTestCase.test_quiet.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildRpmTestCase.test_quiet.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildRpmTestCase.test_quiet', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_quiet', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_quiet(...)' code ##################

        
        # Assigning a Call to a Name (line 59):
        
        # Call to mkdtemp(...): (line 59)
        # Processing the call keyword arguments (line 59)
        kwargs_30534 = {}
        # Getting the type of 'self' (line 59)
        self_30532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 59)
        mkdtemp_30533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 18), self_30532, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 59)
        mkdtemp_call_result_30535 = invoke(stypy.reporting.localization.Localization(__file__, 59, 18), mkdtemp_30533, *[], **kwargs_30534)
        
        # Assigning a type to the variable 'tmp_dir' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'tmp_dir', mkdtemp_call_result_30535)
        
        # Assigning a Name to a Subscript (line 60):
        # Getting the type of 'tmp_dir' (line 60)
        tmp_dir_30536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'tmp_dir')
        # Getting the type of 'os' (line 60)
        os_30537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'os')
        # Obtaining the member 'environ' of a type (line 60)
        environ_30538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), os_30537, 'environ')
        str_30539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'str', 'HOME')
        # Storing an element on a container (line 60)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 8), environ_30538, (str_30539, tmp_dir_30536))
        
        # Assigning a Call to a Name (line 61):
        
        # Call to join(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'tmp_dir' (line 61)
        tmp_dir_30543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 31), 'tmp_dir', False)
        str_30544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 40), 'str', 'foo')
        # Processing the call keyword arguments (line 61)
        kwargs_30545 = {}
        # Getting the type of 'os' (line 61)
        os_30540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 61)
        path_30541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 18), os_30540, 'path')
        # Obtaining the member 'join' of a type (line 61)
        join_30542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 18), path_30541, 'join')
        # Calling join(args, kwargs) (line 61)
        join_call_result_30546 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), join_30542, *[tmp_dir_30543, str_30544], **kwargs_30545)
        
        # Assigning a type to the variable 'pkg_dir' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'pkg_dir', join_call_result_30546)
        
        # Call to mkdir(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'pkg_dir' (line 62)
        pkg_dir_30549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'pkg_dir', False)
        # Processing the call keyword arguments (line 62)
        kwargs_30550 = {}
        # Getting the type of 'os' (line 62)
        os_30547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 62)
        mkdir_30548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), os_30547, 'mkdir')
        # Calling mkdir(args, kwargs) (line 62)
        mkdir_call_result_30551 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), mkdir_30548, *[pkg_dir_30549], **kwargs_30550)
        
        
        # Call to write_file(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_30554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        # Getting the type of 'pkg_dir' (line 63)
        pkg_dir_30555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 25), 'pkg_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 25), tuple_30554, pkg_dir_30555)
        # Adding element type (line 63)
        str_30556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 34), 'str', 'setup.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 25), tuple_30554, str_30556)
        
        # Getting the type of 'SETUP_PY' (line 63)
        SETUP_PY_30557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 47), 'SETUP_PY', False)
        # Processing the call keyword arguments (line 63)
        kwargs_30558 = {}
        # Getting the type of 'self' (line 63)
        self_30552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 63)
        write_file_30553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_30552, 'write_file')
        # Calling write_file(args, kwargs) (line 63)
        write_file_call_result_30559 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), write_file_30553, *[tuple_30554, SETUP_PY_30557], **kwargs_30558)
        
        
        # Call to write_file(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Obtaining an instance of the builtin type 'tuple' (line 64)
        tuple_30562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 64)
        # Adding element type (line 64)
        # Getting the type of 'pkg_dir' (line 64)
        pkg_dir_30563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'pkg_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 25), tuple_30562, pkg_dir_30563)
        # Adding element type (line 64)
        str_30564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 34), 'str', 'foo.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 25), tuple_30562, str_30564)
        
        str_30565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 45), 'str', '#')
        # Processing the call keyword arguments (line 64)
        kwargs_30566 = {}
        # Getting the type of 'self' (line 64)
        self_30560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 64)
        write_file_30561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_30560, 'write_file')
        # Calling write_file(args, kwargs) (line 64)
        write_file_call_result_30567 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), write_file_30561, *[tuple_30562, str_30565], **kwargs_30566)
        
        
        # Call to write_file(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Obtaining an instance of the builtin type 'tuple' (line 65)
        tuple_30570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 65)
        # Adding element type (line 65)
        # Getting the type of 'pkg_dir' (line 65)
        pkg_dir_30571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'pkg_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 25), tuple_30570, pkg_dir_30571)
        # Adding element type (line 65)
        str_30572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 34), 'str', 'MANIFEST.in')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 25), tuple_30570, str_30572)
        
        str_30573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 50), 'str', 'include foo.py')
        # Processing the call keyword arguments (line 65)
        kwargs_30574 = {}
        # Getting the type of 'self' (line 65)
        self_30568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 65)
        write_file_30569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_30568, 'write_file')
        # Calling write_file(args, kwargs) (line 65)
        write_file_call_result_30575 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), write_file_30569, *[tuple_30570, str_30573], **kwargs_30574)
        
        
        # Call to write_file(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_30578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        # Adding element type (line 66)
        # Getting the type of 'pkg_dir' (line 66)
        pkg_dir_30579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'pkg_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 25), tuple_30578, pkg_dir_30579)
        # Adding element type (line 66)
        str_30580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 34), 'str', 'README')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 25), tuple_30578, str_30580)
        
        str_30581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 45), 'str', '')
        # Processing the call keyword arguments (line 66)
        kwargs_30582 = {}
        # Getting the type of 'self' (line 66)
        self_30576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 66)
        write_file_30577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_30576, 'write_file')
        # Calling write_file(args, kwargs) (line 66)
        write_file_call_result_30583 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), write_file_30577, *[tuple_30578, str_30581], **kwargs_30582)
        
        
        # Assigning a Call to a Name (line 68):
        
        # Call to Distribution(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Obtaining an instance of the builtin type 'dict' (line 68)
        dict_30585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 68)
        # Adding element type (key, value) (line 68)
        str_30586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'str', 'name')
        str_30587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 37), 'str', 'foo')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), dict_30585, (str_30586, str_30587))
        # Adding element type (key, value) (line 68)
        str_30588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 44), 'str', 'version')
        str_30589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 55), 'str', '0.1')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), dict_30585, (str_30588, str_30589))
        # Adding element type (key, value) (line 68)
        str_30590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 29), 'str', 'py_modules')
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_30591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        str_30592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 44), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 43), list_30591, str_30592)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), dict_30585, (str_30590, list_30591))
        # Adding element type (key, value) (line 68)
        str_30593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 29), 'str', 'url')
        str_30594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 36), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), dict_30585, (str_30593, str_30594))
        # Adding element type (key, value) (line 68)
        str_30595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 43), 'str', 'author')
        str_30596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 53), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), dict_30585, (str_30595, str_30596))
        # Adding element type (key, value) (line 68)
        str_30597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'str', 'author_email')
        str_30598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 45), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), dict_30585, (str_30597, str_30598))
        
        # Processing the call keyword arguments (line 68)
        kwargs_30599 = {}
        # Getting the type of 'Distribution' (line 68)
        Distribution_30584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 68)
        Distribution_call_result_30600 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), Distribution_30584, *[dict_30585], **kwargs_30599)
        
        # Assigning a type to the variable 'dist' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'dist', Distribution_call_result_30600)
        
        # Assigning a Str to a Attribute (line 72):
        str_30601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 27), 'str', 'setup.py')
        # Getting the type of 'dist' (line 72)
        dist_30602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'dist')
        # Setting the type of the member 'script_name' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), dist_30602, 'script_name', str_30601)
        
        # Call to chdir(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'pkg_dir' (line 73)
        pkg_dir_30605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'pkg_dir', False)
        # Processing the call keyword arguments (line 73)
        kwargs_30606 = {}
        # Getting the type of 'os' (line 73)
        os_30603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 73)
        chdir_30604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), os_30603, 'chdir')
        # Calling chdir(args, kwargs) (line 73)
        chdir_call_result_30607 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), chdir_30604, *[pkg_dir_30605], **kwargs_30606)
        
        
        # Assigning a List to a Attribute (line 75):
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_30608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        str_30609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 20), 'str', 'setup.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 19), list_30608, str_30609)
        
        # Getting the type of 'sys' (line 75)
        sys_30610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'sys')
        # Setting the type of the member 'argv' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), sys_30610, 'argv', list_30608)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to bdist_rpm(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'dist' (line 76)
        dist_30612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'dist', False)
        # Processing the call keyword arguments (line 76)
        kwargs_30613 = {}
        # Getting the type of 'bdist_rpm' (line 76)
        bdist_rpm_30611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'bdist_rpm', False)
        # Calling bdist_rpm(args, kwargs) (line 76)
        bdist_rpm_call_result_30614 = invoke(stypy.reporting.localization.Localization(__file__, 76, 14), bdist_rpm_30611, *[dist_30612], **kwargs_30613)
        
        # Assigning a type to the variable 'cmd' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'cmd', bdist_rpm_call_result_30614)
        
        # Assigning a Name to a Attribute (line 77):
        # Getting the type of 'True' (line 77)
        True_30615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'True')
        # Getting the type of 'cmd' (line 77)
        cmd_30616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'cmd')
        # Setting the type of the member 'fix_python' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), cmd_30616, 'fix_python', True_30615)
        
        # Assigning a Num to a Attribute (line 80):
        int_30617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'int')
        # Getting the type of 'cmd' (line 80)
        cmd_30618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'cmd')
        # Setting the type of the member 'quiet' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), cmd_30618, 'quiet', int_30617)
        
        # Call to ensure_finalized(...): (line 81)
        # Processing the call keyword arguments (line 81)
        kwargs_30621 = {}
        # Getting the type of 'cmd' (line 81)
        cmd_30619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 81)
        ensure_finalized_30620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), cmd_30619, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 81)
        ensure_finalized_call_result_30622 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), ensure_finalized_30620, *[], **kwargs_30621)
        
        
        # Call to run(...): (line 82)
        # Processing the call keyword arguments (line 82)
        kwargs_30625 = {}
        # Getting the type of 'cmd' (line 82)
        cmd_30623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 82)
        run_30624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), cmd_30623, 'run')
        # Calling run(args, kwargs) (line 82)
        run_call_result_30626 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), run_30624, *[], **kwargs_30625)
        
        
        # Assigning a Call to a Name (line 84):
        
        # Call to listdir(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Call to join(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'pkg_dir' (line 84)
        pkg_dir_30632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 47), 'pkg_dir', False)
        str_30633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 56), 'str', 'dist')
        # Processing the call keyword arguments (line 84)
        kwargs_30634 = {}
        # Getting the type of 'os' (line 84)
        os_30629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 84)
        path_30630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 34), os_30629, 'path')
        # Obtaining the member 'join' of a type (line 84)
        join_30631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 34), path_30630, 'join')
        # Calling join(args, kwargs) (line 84)
        join_call_result_30635 = invoke(stypy.reporting.localization.Localization(__file__, 84, 34), join_30631, *[pkg_dir_30632, str_30633], **kwargs_30634)
        
        # Processing the call keyword arguments (line 84)
        kwargs_30636 = {}
        # Getting the type of 'os' (line 84)
        os_30627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'os', False)
        # Obtaining the member 'listdir' of a type (line 84)
        listdir_30628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), os_30627, 'listdir')
        # Calling listdir(args, kwargs) (line 84)
        listdir_call_result_30637 = invoke(stypy.reporting.localization.Localization(__file__, 84, 23), listdir_30628, *[join_call_result_30635], **kwargs_30636)
        
        # Assigning a type to the variable 'dist_created' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'dist_created', listdir_call_result_30637)
        
        # Call to assertIn(...): (line 85)
        # Processing the call arguments (line 85)
        str_30640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'str', 'foo-0.1-1.noarch.rpm')
        # Getting the type of 'dist_created' (line 85)
        dist_created_30641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 46), 'dist_created', False)
        # Processing the call keyword arguments (line 85)
        kwargs_30642 = {}
        # Getting the type of 'self' (line 85)
        self_30638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 85)
        assertIn_30639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_30638, 'assertIn')
        # Calling assertIn(args, kwargs) (line 85)
        assertIn_call_result_30643 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), assertIn_30639, *[str_30640, dist_created_30641], **kwargs_30642)
        
        
        # Call to assertIn(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_30646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        str_30647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 23), 'str', 'bdist_rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 23), tuple_30646, str_30647)
        # Adding element type (line 88)
        str_30648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 36), 'str', 'any')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 23), tuple_30646, str_30648)
        # Adding element type (line 88)
        str_30649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 43), 'str', 'dist/foo-0.1-1.src.rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 23), tuple_30646, str_30649)
        
        # Getting the type of 'dist' (line 88)
        dist_30650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 70), 'dist', False)
        # Obtaining the member 'dist_files' of a type (line 88)
        dist_files_30651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 70), dist_30650, 'dist_files')
        # Processing the call keyword arguments (line 88)
        kwargs_30652 = {}
        # Getting the type of 'self' (line 88)
        self_30644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 88)
        assertIn_30645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_30644, 'assertIn')
        # Calling assertIn(args, kwargs) (line 88)
        assertIn_call_result_30653 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), assertIn_30645, *[tuple_30646, dist_files_30651], **kwargs_30652)
        
        
        # Call to assertIn(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Obtaining an instance of the builtin type 'tuple' (line 89)
        tuple_30656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 89)
        # Adding element type (line 89)
        str_30657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 23), 'str', 'bdist_rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 23), tuple_30656, str_30657)
        # Adding element type (line 89)
        str_30658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 36), 'str', 'any')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 23), tuple_30656, str_30658)
        # Adding element type (line 89)
        str_30659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 43), 'str', 'dist/foo-0.1-1.noarch.rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 23), tuple_30656, str_30659)
        
        # Getting the type of 'dist' (line 89)
        dist_30660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 73), 'dist', False)
        # Obtaining the member 'dist_files' of a type (line 89)
        dist_files_30661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 73), dist_30660, 'dist_files')
        # Processing the call keyword arguments (line 89)
        kwargs_30662 = {}
        # Getting the type of 'self' (line 89)
        self_30654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 89)
        assertIn_30655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_30654, 'assertIn')
        # Calling assertIn(args, kwargs) (line 89)
        assertIn_call_result_30663 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assertIn_30655, *[tuple_30656, dist_files_30661], **kwargs_30662)
        
        
        # ################# End of 'test_quiet(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_quiet' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_30664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30664)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_quiet'
        return stypy_return_type_30664


    @norecursion
    def test_no_optimize_flag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_no_optimize_flag'
        module_type_store = module_type_store.open_function_context('test_no_optimize_flag', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildRpmTestCase.test_no_optimize_flag.__dict__.__setitem__('stypy_localization', localization)
        BuildRpmTestCase.test_no_optimize_flag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildRpmTestCase.test_no_optimize_flag.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildRpmTestCase.test_no_optimize_flag.__dict__.__setitem__('stypy_function_name', 'BuildRpmTestCase.test_no_optimize_flag')
        BuildRpmTestCase.test_no_optimize_flag.__dict__.__setitem__('stypy_param_names_list', [])
        BuildRpmTestCase.test_no_optimize_flag.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildRpmTestCase.test_no_optimize_flag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildRpmTestCase.test_no_optimize_flag.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildRpmTestCase.test_no_optimize_flag.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildRpmTestCase.test_no_optimize_flag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildRpmTestCase.test_no_optimize_flag.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildRpmTestCase.test_no_optimize_flag', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_no_optimize_flag', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_no_optimize_flag(...)' code ##################

        
        # Assigning a Call to a Name (line 103):
        
        # Call to mkdtemp(...): (line 103)
        # Processing the call keyword arguments (line 103)
        kwargs_30667 = {}
        # Getting the type of 'self' (line 103)
        self_30665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 103)
        mkdtemp_30666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 18), self_30665, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 103)
        mkdtemp_call_result_30668 = invoke(stypy.reporting.localization.Localization(__file__, 103, 18), mkdtemp_30666, *[], **kwargs_30667)
        
        # Assigning a type to the variable 'tmp_dir' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tmp_dir', mkdtemp_call_result_30668)
        
        # Assigning a Name to a Subscript (line 104):
        # Getting the type of 'tmp_dir' (line 104)
        tmp_dir_30669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'tmp_dir')
        # Getting the type of 'os' (line 104)
        os_30670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'os')
        # Obtaining the member 'environ' of a type (line 104)
        environ_30671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), os_30670, 'environ')
        str_30672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 19), 'str', 'HOME')
        # Storing an element on a container (line 104)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 8), environ_30671, (str_30672, tmp_dir_30669))
        
        # Assigning a Call to a Name (line 105):
        
        # Call to join(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'tmp_dir' (line 105)
        tmp_dir_30676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 31), 'tmp_dir', False)
        str_30677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 40), 'str', 'foo')
        # Processing the call keyword arguments (line 105)
        kwargs_30678 = {}
        # Getting the type of 'os' (line 105)
        os_30673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 105)
        path_30674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 18), os_30673, 'path')
        # Obtaining the member 'join' of a type (line 105)
        join_30675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 18), path_30674, 'join')
        # Calling join(args, kwargs) (line 105)
        join_call_result_30679 = invoke(stypy.reporting.localization.Localization(__file__, 105, 18), join_30675, *[tmp_dir_30676, str_30677], **kwargs_30678)
        
        # Assigning a type to the variable 'pkg_dir' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'pkg_dir', join_call_result_30679)
        
        # Call to mkdir(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'pkg_dir' (line 106)
        pkg_dir_30682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'pkg_dir', False)
        # Processing the call keyword arguments (line 106)
        kwargs_30683 = {}
        # Getting the type of 'os' (line 106)
        os_30680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 106)
        mkdir_30681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), os_30680, 'mkdir')
        # Calling mkdir(args, kwargs) (line 106)
        mkdir_call_result_30684 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), mkdir_30681, *[pkg_dir_30682], **kwargs_30683)
        
        
        # Call to write_file(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Obtaining an instance of the builtin type 'tuple' (line 107)
        tuple_30687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 107)
        # Adding element type (line 107)
        # Getting the type of 'pkg_dir' (line 107)
        pkg_dir_30688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 25), 'pkg_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 25), tuple_30687, pkg_dir_30688)
        # Adding element type (line 107)
        str_30689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 34), 'str', 'setup.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 25), tuple_30687, str_30689)
        
        # Getting the type of 'SETUP_PY' (line 107)
        SETUP_PY_30690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 47), 'SETUP_PY', False)
        # Processing the call keyword arguments (line 107)
        kwargs_30691 = {}
        # Getting the type of 'self' (line 107)
        self_30685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 107)
        write_file_30686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), self_30685, 'write_file')
        # Calling write_file(args, kwargs) (line 107)
        write_file_call_result_30692 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), write_file_30686, *[tuple_30687, SETUP_PY_30690], **kwargs_30691)
        
        
        # Call to write_file(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_30695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        # Getting the type of 'pkg_dir' (line 108)
        pkg_dir_30696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'pkg_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 25), tuple_30695, pkg_dir_30696)
        # Adding element type (line 108)
        str_30697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 34), 'str', 'foo.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 25), tuple_30695, str_30697)
        
        str_30698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 45), 'str', '#')
        # Processing the call keyword arguments (line 108)
        kwargs_30699 = {}
        # Getting the type of 'self' (line 108)
        self_30693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 108)
        write_file_30694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_30693, 'write_file')
        # Calling write_file(args, kwargs) (line 108)
        write_file_call_result_30700 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), write_file_30694, *[tuple_30695, str_30698], **kwargs_30699)
        
        
        # Call to write_file(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining an instance of the builtin type 'tuple' (line 109)
        tuple_30703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 109)
        # Adding element type (line 109)
        # Getting the type of 'pkg_dir' (line 109)
        pkg_dir_30704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 'pkg_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 25), tuple_30703, pkg_dir_30704)
        # Adding element type (line 109)
        str_30705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 34), 'str', 'MANIFEST.in')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 25), tuple_30703, str_30705)
        
        str_30706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 50), 'str', 'include foo.py')
        # Processing the call keyword arguments (line 109)
        kwargs_30707 = {}
        # Getting the type of 'self' (line 109)
        self_30701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 109)
        write_file_30702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_30701, 'write_file')
        # Calling write_file(args, kwargs) (line 109)
        write_file_call_result_30708 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), write_file_30702, *[tuple_30703, str_30706], **kwargs_30707)
        
        
        # Call to write_file(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Obtaining an instance of the builtin type 'tuple' (line 110)
        tuple_30711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 110)
        # Adding element type (line 110)
        # Getting the type of 'pkg_dir' (line 110)
        pkg_dir_30712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'pkg_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 25), tuple_30711, pkg_dir_30712)
        # Adding element type (line 110)
        str_30713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 34), 'str', 'README')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 25), tuple_30711, str_30713)
        
        str_30714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 45), 'str', '')
        # Processing the call keyword arguments (line 110)
        kwargs_30715 = {}
        # Getting the type of 'self' (line 110)
        self_30709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 110)
        write_file_30710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_30709, 'write_file')
        # Calling write_file(args, kwargs) (line 110)
        write_file_call_result_30716 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), write_file_30710, *[tuple_30711, str_30714], **kwargs_30715)
        
        
        # Assigning a Call to a Name (line 112):
        
        # Call to Distribution(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Obtaining an instance of the builtin type 'dict' (line 112)
        dict_30718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 112)
        # Adding element type (key, value) (line 112)
        str_30719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 29), 'str', 'name')
        str_30720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 37), 'str', 'foo')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 28), dict_30718, (str_30719, str_30720))
        # Adding element type (key, value) (line 112)
        str_30721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 44), 'str', 'version')
        str_30722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 55), 'str', '0.1')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 28), dict_30718, (str_30721, str_30722))
        # Adding element type (key, value) (line 112)
        str_30723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 29), 'str', 'py_modules')
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_30724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        str_30725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 44), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 43), list_30724, str_30725)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 28), dict_30718, (str_30723, list_30724))
        # Adding element type (key, value) (line 112)
        str_30726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 29), 'str', 'url')
        str_30727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 36), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 28), dict_30718, (str_30726, str_30727))
        # Adding element type (key, value) (line 112)
        str_30728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 43), 'str', 'author')
        str_30729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 53), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 28), dict_30718, (str_30728, str_30729))
        # Adding element type (key, value) (line 112)
        str_30730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 29), 'str', 'author_email')
        str_30731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 45), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 28), dict_30718, (str_30730, str_30731))
        
        # Processing the call keyword arguments (line 112)
        kwargs_30732 = {}
        # Getting the type of 'Distribution' (line 112)
        Distribution_30717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 112)
        Distribution_call_result_30733 = invoke(stypy.reporting.localization.Localization(__file__, 112, 15), Distribution_30717, *[dict_30718], **kwargs_30732)
        
        # Assigning a type to the variable 'dist' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'dist', Distribution_call_result_30733)
        
        # Assigning a Str to a Attribute (line 116):
        str_30734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 27), 'str', 'setup.py')
        # Getting the type of 'dist' (line 116)
        dist_30735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'dist')
        # Setting the type of the member 'script_name' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), dist_30735, 'script_name', str_30734)
        
        # Call to chdir(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'pkg_dir' (line 117)
        pkg_dir_30738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'pkg_dir', False)
        # Processing the call keyword arguments (line 117)
        kwargs_30739 = {}
        # Getting the type of 'os' (line 117)
        os_30736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 117)
        chdir_30737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), os_30736, 'chdir')
        # Calling chdir(args, kwargs) (line 117)
        chdir_call_result_30740 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), chdir_30737, *[pkg_dir_30738], **kwargs_30739)
        
        
        # Assigning a List to a Attribute (line 119):
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_30741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        str_30742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 20), 'str', 'setup.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 19), list_30741, str_30742)
        
        # Getting the type of 'sys' (line 119)
        sys_30743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'sys')
        # Setting the type of the member 'argv' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), sys_30743, 'argv', list_30741)
        
        # Assigning a Call to a Name (line 120):
        
        # Call to bdist_rpm(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'dist' (line 120)
        dist_30745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 'dist', False)
        # Processing the call keyword arguments (line 120)
        kwargs_30746 = {}
        # Getting the type of 'bdist_rpm' (line 120)
        bdist_rpm_30744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), 'bdist_rpm', False)
        # Calling bdist_rpm(args, kwargs) (line 120)
        bdist_rpm_call_result_30747 = invoke(stypy.reporting.localization.Localization(__file__, 120, 14), bdist_rpm_30744, *[dist_30745], **kwargs_30746)
        
        # Assigning a type to the variable 'cmd' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'cmd', bdist_rpm_call_result_30747)
        
        # Assigning a Name to a Attribute (line 121):
        # Getting the type of 'True' (line 121)
        True_30748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 25), 'True')
        # Getting the type of 'cmd' (line 121)
        cmd_30749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'cmd')
        # Setting the type of the member 'fix_python' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), cmd_30749, 'fix_python', True_30748)
        
        # Assigning a Num to a Attribute (line 123):
        int_30750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 20), 'int')
        # Getting the type of 'cmd' (line 123)
        cmd_30751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'cmd')
        # Setting the type of the member 'quiet' of a type (line 123)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), cmd_30751, 'quiet', int_30750)
        
        # Call to ensure_finalized(...): (line 124)
        # Processing the call keyword arguments (line 124)
        kwargs_30754 = {}
        # Getting the type of 'cmd' (line 124)
        cmd_30752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 124)
        ensure_finalized_30753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), cmd_30752, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 124)
        ensure_finalized_call_result_30755 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), ensure_finalized_30753, *[], **kwargs_30754)
        
        
        # Call to run(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_30758 = {}
        # Getting the type of 'cmd' (line 125)
        cmd_30756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 125)
        run_30757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), cmd_30756, 'run')
        # Calling run(args, kwargs) (line 125)
        run_call_result_30759 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), run_30757, *[], **kwargs_30758)
        
        
        # Assigning a Call to a Name (line 127):
        
        # Call to listdir(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Call to join(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'pkg_dir' (line 127)
        pkg_dir_30765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 47), 'pkg_dir', False)
        str_30766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 56), 'str', 'dist')
        # Processing the call keyword arguments (line 127)
        kwargs_30767 = {}
        # Getting the type of 'os' (line 127)
        os_30762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 127)
        path_30763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 34), os_30762, 'path')
        # Obtaining the member 'join' of a type (line 127)
        join_30764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 34), path_30763, 'join')
        # Calling join(args, kwargs) (line 127)
        join_call_result_30768 = invoke(stypy.reporting.localization.Localization(__file__, 127, 34), join_30764, *[pkg_dir_30765, str_30766], **kwargs_30767)
        
        # Processing the call keyword arguments (line 127)
        kwargs_30769 = {}
        # Getting the type of 'os' (line 127)
        os_30760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'os', False)
        # Obtaining the member 'listdir' of a type (line 127)
        listdir_30761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 23), os_30760, 'listdir')
        # Calling listdir(args, kwargs) (line 127)
        listdir_call_result_30770 = invoke(stypy.reporting.localization.Localization(__file__, 127, 23), listdir_30761, *[join_call_result_30768], **kwargs_30769)
        
        # Assigning a type to the variable 'dist_created' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'dist_created', listdir_call_result_30770)
        
        # Call to assertIn(...): (line 128)
        # Processing the call arguments (line 128)
        str_30773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 22), 'str', 'foo-0.1-1.noarch.rpm')
        # Getting the type of 'dist_created' (line 128)
        dist_created_30774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 46), 'dist_created', False)
        # Processing the call keyword arguments (line 128)
        kwargs_30775 = {}
        # Getting the type of 'self' (line 128)
        self_30771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 128)
        assertIn_30772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_30771, 'assertIn')
        # Calling assertIn(args, kwargs) (line 128)
        assertIn_call_result_30776 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), assertIn_30772, *[str_30773, dist_created_30774], **kwargs_30775)
        
        
        # Call to assertIn(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Obtaining an instance of the builtin type 'tuple' (line 131)
        tuple_30779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 131)
        # Adding element type (line 131)
        str_30780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 23), 'str', 'bdist_rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), tuple_30779, str_30780)
        # Adding element type (line 131)
        str_30781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 36), 'str', 'any')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), tuple_30779, str_30781)
        # Adding element type (line 131)
        str_30782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 43), 'str', 'dist/foo-0.1-1.src.rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), tuple_30779, str_30782)
        
        # Getting the type of 'dist' (line 131)
        dist_30783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 70), 'dist', False)
        # Obtaining the member 'dist_files' of a type (line 131)
        dist_files_30784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 70), dist_30783, 'dist_files')
        # Processing the call keyword arguments (line 131)
        kwargs_30785 = {}
        # Getting the type of 'self' (line 131)
        self_30777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 131)
        assertIn_30778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_30777, 'assertIn')
        # Calling assertIn(args, kwargs) (line 131)
        assertIn_call_result_30786 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), assertIn_30778, *[tuple_30779, dist_files_30784], **kwargs_30785)
        
        
        # Call to assertIn(...): (line 132)
        # Processing the call arguments (line 132)
        
        # Obtaining an instance of the builtin type 'tuple' (line 132)
        tuple_30789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 132)
        # Adding element type (line 132)
        str_30790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 23), 'str', 'bdist_rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 23), tuple_30789, str_30790)
        # Adding element type (line 132)
        str_30791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 36), 'str', 'any')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 23), tuple_30789, str_30791)
        # Adding element type (line 132)
        str_30792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 43), 'str', 'dist/foo-0.1-1.noarch.rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 23), tuple_30789, str_30792)
        
        # Getting the type of 'dist' (line 132)
        dist_30793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 73), 'dist', False)
        # Obtaining the member 'dist_files' of a type (line 132)
        dist_files_30794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 73), dist_30793, 'dist_files')
        # Processing the call keyword arguments (line 132)
        kwargs_30795 = {}
        # Getting the type of 'self' (line 132)
        self_30787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 132)
        assertIn_30788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), self_30787, 'assertIn')
        # Calling assertIn(args, kwargs) (line 132)
        assertIn_call_result_30796 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), assertIn_30788, *[tuple_30789, dist_files_30794], **kwargs_30795)
        
        
        # Call to remove(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Call to join(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'pkg_dir' (line 134)
        pkg_dir_30802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 31), 'pkg_dir', False)
        str_30803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 40), 'str', 'dist')
        str_30804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 48), 'str', 'foo-0.1-1.noarch.rpm')
        # Processing the call keyword arguments (line 134)
        kwargs_30805 = {}
        # Getting the type of 'os' (line 134)
        os_30799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 134)
        path_30800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 18), os_30799, 'path')
        # Obtaining the member 'join' of a type (line 134)
        join_30801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 18), path_30800, 'join')
        # Calling join(args, kwargs) (line 134)
        join_call_result_30806 = invoke(stypy.reporting.localization.Localization(__file__, 134, 18), join_30801, *[pkg_dir_30802, str_30803, str_30804], **kwargs_30805)
        
        # Processing the call keyword arguments (line 134)
        kwargs_30807 = {}
        # Getting the type of 'os' (line 134)
        os_30797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'os', False)
        # Obtaining the member 'remove' of a type (line 134)
        remove_30798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), os_30797, 'remove')
        # Calling remove(args, kwargs) (line 134)
        remove_call_result_30808 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), remove_30798, *[join_call_result_30806], **kwargs_30807)
        
        
        # ################# End of 'test_no_optimize_flag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_no_optimize_flag' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_30809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30809)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_no_optimize_flag'
        return stypy_return_type_30809


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 32, 0, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildRpmTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BuildRpmTestCase' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'BuildRpmTestCase', BuildRpmTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 136, 0, False)
    
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

    
    # Call to makeSuite(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'BuildRpmTestCase' (line 137)
    BuildRpmTestCase_30812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 30), 'BuildRpmTestCase', False)
    # Processing the call keyword arguments (line 137)
    kwargs_30813 = {}
    # Getting the type of 'unittest' (line 137)
    unittest_30810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 137)
    makeSuite_30811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), unittest_30810, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 137)
    makeSuite_call_result_30814 = invoke(stypy.reporting.localization.Localization(__file__, 137, 11), makeSuite_30811, *[BuildRpmTestCase_30812], **kwargs_30813)
    
    # Assigning a type to the variable 'stypy_return_type' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type', makeSuite_call_result_30814)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 136)
    stypy_return_type_30815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30815)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_30815

# Assigning a type to the variable 'test_suite' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 140)
    # Processing the call arguments (line 140)
    
    # Call to test_suite(...): (line 140)
    # Processing the call keyword arguments (line 140)
    kwargs_30818 = {}
    # Getting the type of 'test_suite' (line 140)
    test_suite_30817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 140)
    test_suite_call_result_30819 = invoke(stypy.reporting.localization.Localization(__file__, 140, 17), test_suite_30817, *[], **kwargs_30818)
    
    # Processing the call keyword arguments (line 140)
    kwargs_30820 = {}
    # Getting the type of 'run_unittest' (line 140)
    run_unittest_30816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 140)
    run_unittest_call_result_30821 = invoke(stypy.reporting.localization.Localization(__file__, 140, 4), run_unittest_30816, *[test_suite_call_result_30819], **kwargs_30820)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
