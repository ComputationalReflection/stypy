
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.bdist_dumb.'''
2: 
3: import os
4: import sys
5: import zipfile
6: import unittest
7: from test.test_support import run_unittest
8: 
9: # zlib is not used here, but if it's not available
10: # test_simple_built will fail
11: try:
12:     import zlib
13: except ImportError:
14:     zlib = None
15: 
16: from distutils.core import Distribution
17: from distutils.command.bdist_dumb import bdist_dumb
18: from distutils.tests import support
19: 
20: SETUP_PY = '''\
21: from distutils.core import setup
22: import foo
23: 
24: setup(name='foo', version='0.1', py_modules=['foo'],
25:       url='xxx', author='xxx', author_email='xxx')
26: 
27: '''
28: 
29: class BuildDumbTestCase(support.TempdirManager,
30:                         support.LoggingSilencer,
31:                         support.EnvironGuard,
32:                         unittest.TestCase):
33: 
34:     def setUp(self):
35:         super(BuildDumbTestCase, self).setUp()
36:         self.old_location = os.getcwd()
37:         self.old_sys_argv = sys.argv, sys.argv[:]
38: 
39:     def tearDown(self):
40:         os.chdir(self.old_location)
41:         sys.argv = self.old_sys_argv[0]
42:         sys.argv[:] = self.old_sys_argv[1]
43:         super(BuildDumbTestCase, self).tearDown()
44: 
45:     @unittest.skipUnless(zlib, "requires zlib")
46:     def test_simple_built(self):
47: 
48:         # let's create a simple package
49:         tmp_dir = self.mkdtemp()
50:         pkg_dir = os.path.join(tmp_dir, 'foo')
51:         os.mkdir(pkg_dir)
52:         self.write_file((pkg_dir, 'setup.py'), SETUP_PY)
53:         self.write_file((pkg_dir, 'foo.py'), '#')
54:         self.write_file((pkg_dir, 'MANIFEST.in'), 'include foo.py')
55:         self.write_file((pkg_dir, 'README'), '')
56: 
57:         dist = Distribution({'name': 'foo', 'version': '0.1',
58:                              'py_modules': ['foo'],
59:                              'url': 'xxx', 'author': 'xxx',
60:                              'author_email': 'xxx'})
61:         dist.script_name = 'setup.py'
62:         os.chdir(pkg_dir)
63: 
64:         sys.argv = ['setup.py']
65:         cmd = bdist_dumb(dist)
66: 
67:         # so the output is the same no matter
68:         # what is the platform
69:         cmd.format = 'zip'
70: 
71:         cmd.ensure_finalized()
72:         cmd.run()
73: 
74:         # see what we have
75:         dist_created = os.listdir(os.path.join(pkg_dir, 'dist'))
76:         base = "%s.%s.zip" % (dist.get_fullname(), cmd.plat_name)
77:         if os.name == 'os2':
78:             base = base.replace(':', '-')
79: 
80:         self.assertEqual(dist_created, [base])
81: 
82:         # now let's check what we have in the zip file
83:         fp = zipfile.ZipFile(os.path.join('dist', base))
84:         try:
85:             contents = fp.namelist()
86:         finally:
87:             fp.close()
88: 
89:         contents = sorted(os.path.basename(fn) for fn in contents)
90:         wanted = ['foo-0.1-py%s.%s.egg-info' % sys.version_info[:2], 'foo.py']
91:         if not sys.dont_write_bytecode:
92:             wanted.append('foo.pyc')
93:         self.assertEqual(contents, sorted(wanted))
94: 
95:     def test_finalize_options(self):
96:         pkg_dir, dist = self.create_dist()
97:         os.chdir(pkg_dir)
98:         cmd = bdist_dumb(dist)
99:         self.assertEqual(cmd.bdist_dir, None)
100:         cmd.finalize_options()
101: 
102:         # bdist_dir is initialized to bdist_base/dumb if not set
103:         base = cmd.get_finalized_command('bdist').bdist_base
104:         self.assertEqual(cmd.bdist_dir, os.path.join(base, 'dumb'))
105: 
106:         # the format is set to a default value depending on the os.name
107:         default = cmd.default_format[os.name]
108:         self.assertEqual(cmd.format, default)
109: 
110: def test_suite():
111:     return unittest.makeSuite(BuildDumbTestCase)
112: 
113: if __name__ == '__main__':
114:     run_unittest(test_suite())
115: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_30066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.bdist_dumb.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import zipfile' statement (line 5)
import zipfile

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'zipfile', zipfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import unittest' statement (line 6)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from test.test_support import run_unittest' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30067 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'test.test_support')

if (type(import_30067) is not StypyTypeError):

    if (import_30067 != 'pyd_module'):
        __import__(import_30067)
        sys_modules_30068 = sys.modules[import_30067]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'test.test_support', sys_modules_30068.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_30068, sys_modules_30068.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'test.test_support', import_30067)

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

# Assigning a Name to a Name (line 14):
# Getting the type of 'None' (line 14)
None_30069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'None')
# Assigning a type to the variable 'zlib' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'zlib', None_30069)
# SSA join for try-except statement (line 11)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils.core import Distribution' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30070 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.core')

if (type(import_30070) is not StypyTypeError):

    if (import_30070 != 'pyd_module'):
        __import__(import_30070)
        sys_modules_30071 = sys.modules[import_30070]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.core', sys_modules_30071.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_30071, sys_modules_30071.module_type_store, module_type_store)
    else:
        from distutils.core import Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.core', None, module_type_store, ['Distribution'], [Distribution])

else:
    # Assigning a type to the variable 'distutils.core' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.core', import_30070)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils.command.bdist_dumb import bdist_dumb' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30072 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.command.bdist_dumb')

if (type(import_30072) is not StypyTypeError):

    if (import_30072 != 'pyd_module'):
        __import__(import_30072)
        sys_modules_30073 = sys.modules[import_30072]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.command.bdist_dumb', sys_modules_30073.module_type_store, module_type_store, ['bdist_dumb'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_30073, sys_modules_30073.module_type_store, module_type_store)
    else:
        from distutils.command.bdist_dumb import bdist_dumb

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.command.bdist_dumb', None, module_type_store, ['bdist_dumb'], [bdist_dumb])

else:
    # Assigning a type to the variable 'distutils.command.bdist_dumb' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.command.bdist_dumb', import_30072)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from distutils.tests import support' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30074 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.tests')

if (type(import_30074) is not StypyTypeError):

    if (import_30074 != 'pyd_module'):
        __import__(import_30074)
        sys_modules_30075 = sys.modules[import_30074]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.tests', sys_modules_30075.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_30075, sys_modules_30075.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.tests', import_30074)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Assigning a Str to a Name (line 20):

# Assigning a Str to a Name (line 20):
str_30076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', "from distutils.core import setup\nimport foo\n\nsetup(name='foo', version='0.1', py_modules=['foo'],\n      url='xxx', author='xxx', author_email='xxx')\n\n")
# Assigning a type to the variable 'SETUP_PY' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'SETUP_PY', str_30076)
# Declaration of the 'BuildDumbTestCase' class
# Getting the type of 'support' (line 29)
support_30077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'support')
# Obtaining the member 'TempdirManager' of a type (line 29)
TempdirManager_30078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 24), support_30077, 'TempdirManager')
# Getting the type of 'support' (line 30)
support_30079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 30)
LoggingSilencer_30080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 24), support_30079, 'LoggingSilencer')
# Getting the type of 'support' (line 31)
support_30081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'support')
# Obtaining the member 'EnvironGuard' of a type (line 31)
EnvironGuard_30082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 24), support_30081, 'EnvironGuard')
# Getting the type of 'unittest' (line 32)
unittest_30083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'unittest')
# Obtaining the member 'TestCase' of a type (line 32)
TestCase_30084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 24), unittest_30083, 'TestCase')

class BuildDumbTestCase(TempdirManager_30078, LoggingSilencer_30080, EnvironGuard_30082, TestCase_30084, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildDumbTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        BuildDumbTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildDumbTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildDumbTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'BuildDumbTestCase.setUp')
        BuildDumbTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        BuildDumbTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildDumbTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildDumbTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildDumbTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildDumbTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildDumbTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildDumbTestCase.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to setUp(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_30091 = {}
        
        # Call to super(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'BuildDumbTestCase' (line 35)
        BuildDumbTestCase_30086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'BuildDumbTestCase', False)
        # Getting the type of 'self' (line 35)
        self_30087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'self', False)
        # Processing the call keyword arguments (line 35)
        kwargs_30088 = {}
        # Getting the type of 'super' (line 35)
        super_30085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'super', False)
        # Calling super(args, kwargs) (line 35)
        super_call_result_30089 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), super_30085, *[BuildDumbTestCase_30086, self_30087], **kwargs_30088)
        
        # Obtaining the member 'setUp' of a type (line 35)
        setUp_30090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), super_call_result_30089, 'setUp')
        # Calling setUp(args, kwargs) (line 35)
        setUp_call_result_30092 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), setUp_30090, *[], **kwargs_30091)
        
        
        # Assigning a Call to a Attribute (line 36):
        
        # Assigning a Call to a Attribute (line 36):
        
        # Call to getcwd(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_30095 = {}
        # Getting the type of 'os' (line 36)
        os_30093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 28), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 36)
        getcwd_30094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 28), os_30093, 'getcwd')
        # Calling getcwd(args, kwargs) (line 36)
        getcwd_call_result_30096 = invoke(stypy.reporting.localization.Localization(__file__, 36, 28), getcwd_30094, *[], **kwargs_30095)
        
        # Getting the type of 'self' (line 36)
        self_30097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self')
        # Setting the type of the member 'old_location' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_30097, 'old_location', getcwd_call_result_30096)
        
        # Assigning a Tuple to a Attribute (line 37):
        
        # Assigning a Tuple to a Attribute (line 37):
        
        # Obtaining an instance of the builtin type 'tuple' (line 37)
        tuple_30098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 37)
        # Adding element type (line 37)
        # Getting the type of 'sys' (line 37)
        sys_30099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 28), 'sys')
        # Obtaining the member 'argv' of a type (line 37)
        argv_30100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 28), sys_30099, 'argv')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 28), tuple_30098, argv_30100)
        # Adding element type (line 37)
        
        # Obtaining the type of the subscript
        slice_30101 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 37, 38), None, None, None)
        # Getting the type of 'sys' (line 37)
        sys_30102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 38), 'sys')
        # Obtaining the member 'argv' of a type (line 37)
        argv_30103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 38), sys_30102, 'argv')
        # Obtaining the member '__getitem__' of a type (line 37)
        getitem___30104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 38), argv_30103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 37)
        subscript_call_result_30105 = invoke(stypy.reporting.localization.Localization(__file__, 37, 38), getitem___30104, slice_30101)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 28), tuple_30098, subscript_call_result_30105)
        
        # Getting the type of 'self' (line 37)
        self_30106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member 'old_sys_argv' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_30106, 'old_sys_argv', tuple_30098)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_30107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30107)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_30107


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildDumbTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        BuildDumbTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildDumbTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildDumbTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'BuildDumbTestCase.tearDown')
        BuildDumbTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        BuildDumbTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildDumbTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildDumbTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildDumbTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildDumbTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildDumbTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildDumbTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to chdir(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_30110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'self', False)
        # Obtaining the member 'old_location' of a type (line 40)
        old_location_30111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 17), self_30110, 'old_location')
        # Processing the call keyword arguments (line 40)
        kwargs_30112 = {}
        # Getting the type of 'os' (line 40)
        os_30108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 40)
        chdir_30109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), os_30108, 'chdir')
        # Calling chdir(args, kwargs) (line 40)
        chdir_call_result_30113 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), chdir_30109, *[old_location_30111], **kwargs_30112)
        
        
        # Assigning a Subscript to a Attribute (line 41):
        
        # Assigning a Subscript to a Attribute (line 41):
        
        # Obtaining the type of the subscript
        int_30114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 37), 'int')
        # Getting the type of 'self' (line 41)
        self_30115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'self')
        # Obtaining the member 'old_sys_argv' of a type (line 41)
        old_sys_argv_30116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), self_30115, 'old_sys_argv')
        # Obtaining the member '__getitem__' of a type (line 41)
        getitem___30117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), old_sys_argv_30116, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 41)
        subscript_call_result_30118 = invoke(stypy.reporting.localization.Localization(__file__, 41, 19), getitem___30117, int_30114)
        
        # Getting the type of 'sys' (line 41)
        sys_30119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'sys')
        # Setting the type of the member 'argv' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), sys_30119, 'argv', subscript_call_result_30118)
        
        # Assigning a Subscript to a Subscript (line 42):
        
        # Assigning a Subscript to a Subscript (line 42):
        
        # Obtaining the type of the subscript
        int_30120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 40), 'int')
        # Getting the type of 'self' (line 42)
        self_30121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'self')
        # Obtaining the member 'old_sys_argv' of a type (line 42)
        old_sys_argv_30122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 22), self_30121, 'old_sys_argv')
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___30123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 22), old_sys_argv_30122, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_30124 = invoke(stypy.reporting.localization.Localization(__file__, 42, 22), getitem___30123, int_30120)
        
        # Getting the type of 'sys' (line 42)
        sys_30125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'sys')
        # Obtaining the member 'argv' of a type (line 42)
        argv_30126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), sys_30125, 'argv')
        slice_30127 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 8), None, None, None)
        # Storing an element on a container (line 42)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 8), argv_30126, (slice_30127, subscript_call_result_30124))
        
        # Call to tearDown(...): (line 43)
        # Processing the call keyword arguments (line 43)
        kwargs_30134 = {}
        
        # Call to super(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'BuildDumbTestCase' (line 43)
        BuildDumbTestCase_30129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'BuildDumbTestCase', False)
        # Getting the type of 'self' (line 43)
        self_30130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 33), 'self', False)
        # Processing the call keyword arguments (line 43)
        kwargs_30131 = {}
        # Getting the type of 'super' (line 43)
        super_30128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'super', False)
        # Calling super(args, kwargs) (line 43)
        super_call_result_30132 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), super_30128, *[BuildDumbTestCase_30129, self_30130], **kwargs_30131)
        
        # Obtaining the member 'tearDown' of a type (line 43)
        tearDown_30133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), super_call_result_30132, 'tearDown')
        # Calling tearDown(args, kwargs) (line 43)
        tearDown_call_result_30135 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), tearDown_30133, *[], **kwargs_30134)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_30136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30136)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_30136


    @norecursion
    def test_simple_built(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_built'
        module_type_store = module_type_store.open_function_context('test_simple_built', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildDumbTestCase.test_simple_built.__dict__.__setitem__('stypy_localization', localization)
        BuildDumbTestCase.test_simple_built.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildDumbTestCase.test_simple_built.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildDumbTestCase.test_simple_built.__dict__.__setitem__('stypy_function_name', 'BuildDumbTestCase.test_simple_built')
        BuildDumbTestCase.test_simple_built.__dict__.__setitem__('stypy_param_names_list', [])
        BuildDumbTestCase.test_simple_built.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildDumbTestCase.test_simple_built.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildDumbTestCase.test_simple_built.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildDumbTestCase.test_simple_built.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildDumbTestCase.test_simple_built.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildDumbTestCase.test_simple_built.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildDumbTestCase.test_simple_built', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_built', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_built(...)' code ##################

        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to mkdtemp(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_30139 = {}
        # Getting the type of 'self' (line 49)
        self_30137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 49)
        mkdtemp_30138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 18), self_30137, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 49)
        mkdtemp_call_result_30140 = invoke(stypy.reporting.localization.Localization(__file__, 49, 18), mkdtemp_30138, *[], **kwargs_30139)
        
        # Assigning a type to the variable 'tmp_dir' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tmp_dir', mkdtemp_call_result_30140)
        
        # Assigning a Call to a Name (line 50):
        
        # Assigning a Call to a Name (line 50):
        
        # Call to join(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'tmp_dir' (line 50)
        tmp_dir_30144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 31), 'tmp_dir', False)
        str_30145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 40), 'str', 'foo')
        # Processing the call keyword arguments (line 50)
        kwargs_30146 = {}
        # Getting the type of 'os' (line 50)
        os_30141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 50)
        path_30142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 18), os_30141, 'path')
        # Obtaining the member 'join' of a type (line 50)
        join_30143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 18), path_30142, 'join')
        # Calling join(args, kwargs) (line 50)
        join_call_result_30147 = invoke(stypy.reporting.localization.Localization(__file__, 50, 18), join_30143, *[tmp_dir_30144, str_30145], **kwargs_30146)
        
        # Assigning a type to the variable 'pkg_dir' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'pkg_dir', join_call_result_30147)
        
        # Call to mkdir(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'pkg_dir' (line 51)
        pkg_dir_30150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'pkg_dir', False)
        # Processing the call keyword arguments (line 51)
        kwargs_30151 = {}
        # Getting the type of 'os' (line 51)
        os_30148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 51)
        mkdir_30149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), os_30148, 'mkdir')
        # Calling mkdir(args, kwargs) (line 51)
        mkdir_call_result_30152 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), mkdir_30149, *[pkg_dir_30150], **kwargs_30151)
        
        
        # Call to write_file(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_30155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        # Getting the type of 'pkg_dir' (line 52)
        pkg_dir_30156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 25), 'pkg_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 25), tuple_30155, pkg_dir_30156)
        # Adding element type (line 52)
        str_30157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 34), 'str', 'setup.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 25), tuple_30155, str_30157)
        
        # Getting the type of 'SETUP_PY' (line 52)
        SETUP_PY_30158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 47), 'SETUP_PY', False)
        # Processing the call keyword arguments (line 52)
        kwargs_30159 = {}
        # Getting the type of 'self' (line 52)
        self_30153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 52)
        write_file_30154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_30153, 'write_file')
        # Calling write_file(args, kwargs) (line 52)
        write_file_call_result_30160 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), write_file_30154, *[tuple_30155, SETUP_PY_30158], **kwargs_30159)
        
        
        # Call to write_file(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_30163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        # Getting the type of 'pkg_dir' (line 53)
        pkg_dir_30164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'pkg_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 25), tuple_30163, pkg_dir_30164)
        # Adding element type (line 53)
        str_30165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 34), 'str', 'foo.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 25), tuple_30163, str_30165)
        
        str_30166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 45), 'str', '#')
        # Processing the call keyword arguments (line 53)
        kwargs_30167 = {}
        # Getting the type of 'self' (line 53)
        self_30161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 53)
        write_file_30162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_30161, 'write_file')
        # Calling write_file(args, kwargs) (line 53)
        write_file_call_result_30168 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), write_file_30162, *[tuple_30163, str_30166], **kwargs_30167)
        
        
        # Call to write_file(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_30171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        # Getting the type of 'pkg_dir' (line 54)
        pkg_dir_30172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'pkg_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 25), tuple_30171, pkg_dir_30172)
        # Adding element type (line 54)
        str_30173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'str', 'MANIFEST.in')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 25), tuple_30171, str_30173)
        
        str_30174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 50), 'str', 'include foo.py')
        # Processing the call keyword arguments (line 54)
        kwargs_30175 = {}
        # Getting the type of 'self' (line 54)
        self_30169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 54)
        write_file_30170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_30169, 'write_file')
        # Calling write_file(args, kwargs) (line 54)
        write_file_call_result_30176 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), write_file_30170, *[tuple_30171, str_30174], **kwargs_30175)
        
        
        # Call to write_file(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Obtaining an instance of the builtin type 'tuple' (line 55)
        tuple_30179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 55)
        # Adding element type (line 55)
        # Getting the type of 'pkg_dir' (line 55)
        pkg_dir_30180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'pkg_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 25), tuple_30179, pkg_dir_30180)
        # Adding element type (line 55)
        str_30181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'str', 'README')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 25), tuple_30179, str_30181)
        
        str_30182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 45), 'str', '')
        # Processing the call keyword arguments (line 55)
        kwargs_30183 = {}
        # Getting the type of 'self' (line 55)
        self_30177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 55)
        write_file_30178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_30177, 'write_file')
        # Calling write_file(args, kwargs) (line 55)
        write_file_call_result_30184 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), write_file_30178, *[tuple_30179, str_30182], **kwargs_30183)
        
        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to Distribution(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Obtaining an instance of the builtin type 'dict' (line 57)
        dict_30186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 57)
        # Adding element type (key, value) (line 57)
        str_30187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 29), 'str', 'name')
        str_30188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 37), 'str', 'foo')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 28), dict_30186, (str_30187, str_30188))
        # Adding element type (key, value) (line 57)
        str_30189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 44), 'str', 'version')
        str_30190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 55), 'str', '0.1')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 28), dict_30186, (str_30189, str_30190))
        # Adding element type (key, value) (line 57)
        str_30191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 29), 'str', 'py_modules')
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_30192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        str_30193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 44), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 43), list_30192, str_30193)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 28), dict_30186, (str_30191, list_30192))
        # Adding element type (key, value) (line 57)
        str_30194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 29), 'str', 'url')
        str_30195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 28), dict_30186, (str_30194, str_30195))
        # Adding element type (key, value) (line 57)
        str_30196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 43), 'str', 'author')
        str_30197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 53), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 28), dict_30186, (str_30196, str_30197))
        # Adding element type (key, value) (line 57)
        str_30198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 29), 'str', 'author_email')
        str_30199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 45), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 28), dict_30186, (str_30198, str_30199))
        
        # Processing the call keyword arguments (line 57)
        kwargs_30200 = {}
        # Getting the type of 'Distribution' (line 57)
        Distribution_30185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 57)
        Distribution_call_result_30201 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), Distribution_30185, *[dict_30186], **kwargs_30200)
        
        # Assigning a type to the variable 'dist' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'dist', Distribution_call_result_30201)
        
        # Assigning a Str to a Attribute (line 61):
        
        # Assigning a Str to a Attribute (line 61):
        str_30202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 27), 'str', 'setup.py')
        # Getting the type of 'dist' (line 61)
        dist_30203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'dist')
        # Setting the type of the member 'script_name' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), dist_30203, 'script_name', str_30202)
        
        # Call to chdir(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'pkg_dir' (line 62)
        pkg_dir_30206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'pkg_dir', False)
        # Processing the call keyword arguments (line 62)
        kwargs_30207 = {}
        # Getting the type of 'os' (line 62)
        os_30204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 62)
        chdir_30205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), os_30204, 'chdir')
        # Calling chdir(args, kwargs) (line 62)
        chdir_call_result_30208 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), chdir_30205, *[pkg_dir_30206], **kwargs_30207)
        
        
        # Assigning a List to a Attribute (line 64):
        
        # Assigning a List to a Attribute (line 64):
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_30209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        str_30210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 20), 'str', 'setup.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 19), list_30209, str_30210)
        
        # Getting the type of 'sys' (line 64)
        sys_30211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'sys')
        # Setting the type of the member 'argv' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), sys_30211, 'argv', list_30209)
        
        # Assigning a Call to a Name (line 65):
        
        # Assigning a Call to a Name (line 65):
        
        # Call to bdist_dumb(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'dist' (line 65)
        dist_30213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'dist', False)
        # Processing the call keyword arguments (line 65)
        kwargs_30214 = {}
        # Getting the type of 'bdist_dumb' (line 65)
        bdist_dumb_30212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 14), 'bdist_dumb', False)
        # Calling bdist_dumb(args, kwargs) (line 65)
        bdist_dumb_call_result_30215 = invoke(stypy.reporting.localization.Localization(__file__, 65, 14), bdist_dumb_30212, *[dist_30213], **kwargs_30214)
        
        # Assigning a type to the variable 'cmd' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'cmd', bdist_dumb_call_result_30215)
        
        # Assigning a Str to a Attribute (line 69):
        
        # Assigning a Str to a Attribute (line 69):
        str_30216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 21), 'str', 'zip')
        # Getting the type of 'cmd' (line 69)
        cmd_30217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'cmd')
        # Setting the type of the member 'format' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), cmd_30217, 'format', str_30216)
        
        # Call to ensure_finalized(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_30220 = {}
        # Getting the type of 'cmd' (line 71)
        cmd_30218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 71)
        ensure_finalized_30219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), cmd_30218, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 71)
        ensure_finalized_call_result_30221 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), ensure_finalized_30219, *[], **kwargs_30220)
        
        
        # Call to run(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_30224 = {}
        # Getting the type of 'cmd' (line 72)
        cmd_30222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 72)
        run_30223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), cmd_30222, 'run')
        # Calling run(args, kwargs) (line 72)
        run_call_result_30225 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), run_30223, *[], **kwargs_30224)
        
        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to listdir(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Call to join(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'pkg_dir' (line 75)
        pkg_dir_30231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 47), 'pkg_dir', False)
        str_30232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 56), 'str', 'dist')
        # Processing the call keyword arguments (line 75)
        kwargs_30233 = {}
        # Getting the type of 'os' (line 75)
        os_30228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 75)
        path_30229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 34), os_30228, 'path')
        # Obtaining the member 'join' of a type (line 75)
        join_30230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 34), path_30229, 'join')
        # Calling join(args, kwargs) (line 75)
        join_call_result_30234 = invoke(stypy.reporting.localization.Localization(__file__, 75, 34), join_30230, *[pkg_dir_30231, str_30232], **kwargs_30233)
        
        # Processing the call keyword arguments (line 75)
        kwargs_30235 = {}
        # Getting the type of 'os' (line 75)
        os_30226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'os', False)
        # Obtaining the member 'listdir' of a type (line 75)
        listdir_30227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 23), os_30226, 'listdir')
        # Calling listdir(args, kwargs) (line 75)
        listdir_call_result_30236 = invoke(stypy.reporting.localization.Localization(__file__, 75, 23), listdir_30227, *[join_call_result_30234], **kwargs_30235)
        
        # Assigning a type to the variable 'dist_created' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'dist_created', listdir_call_result_30236)
        
        # Assigning a BinOp to a Name (line 76):
        
        # Assigning a BinOp to a Name (line 76):
        str_30237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 15), 'str', '%s.%s.zip')
        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_30238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        
        # Call to get_fullname(...): (line 76)
        # Processing the call keyword arguments (line 76)
        kwargs_30241 = {}
        # Getting the type of 'dist' (line 76)
        dist_30239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), 'dist', False)
        # Obtaining the member 'get_fullname' of a type (line 76)
        get_fullname_30240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 30), dist_30239, 'get_fullname')
        # Calling get_fullname(args, kwargs) (line 76)
        get_fullname_call_result_30242 = invoke(stypy.reporting.localization.Localization(__file__, 76, 30), get_fullname_30240, *[], **kwargs_30241)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 30), tuple_30238, get_fullname_call_result_30242)
        # Adding element type (line 76)
        # Getting the type of 'cmd' (line 76)
        cmd_30243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 51), 'cmd')
        # Obtaining the member 'plat_name' of a type (line 76)
        plat_name_30244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 51), cmd_30243, 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 30), tuple_30238, plat_name_30244)
        
        # Applying the binary operator '%' (line 76)
        result_mod_30245 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 15), '%', str_30237, tuple_30238)
        
        # Assigning a type to the variable 'base' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'base', result_mod_30245)
        
        
        # Getting the type of 'os' (line 77)
        os_30246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'os')
        # Obtaining the member 'name' of a type (line 77)
        name_30247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 11), os_30246, 'name')
        str_30248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 22), 'str', 'os2')
        # Applying the binary operator '==' (line 77)
        result_eq_30249 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 11), '==', name_30247, str_30248)
        
        # Testing the type of an if condition (line 77)
        if_condition_30250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 8), result_eq_30249)
        # Assigning a type to the variable 'if_condition_30250' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'if_condition_30250', if_condition_30250)
        # SSA begins for if statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 78):
        
        # Assigning a Call to a Name (line 78):
        
        # Call to replace(...): (line 78)
        # Processing the call arguments (line 78)
        str_30253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 32), 'str', ':')
        str_30254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 37), 'str', '-')
        # Processing the call keyword arguments (line 78)
        kwargs_30255 = {}
        # Getting the type of 'base' (line 78)
        base_30251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'base', False)
        # Obtaining the member 'replace' of a type (line 78)
        replace_30252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 19), base_30251, 'replace')
        # Calling replace(args, kwargs) (line 78)
        replace_call_result_30256 = invoke(stypy.reporting.localization.Localization(__file__, 78, 19), replace_30252, *[str_30253, str_30254], **kwargs_30255)
        
        # Assigning a type to the variable 'base' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'base', replace_call_result_30256)
        # SSA join for if statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertEqual(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'dist_created' (line 80)
        dist_created_30259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'dist_created', False)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_30260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        # Getting the type of 'base' (line 80)
        base_30261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 40), 'base', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 39), list_30260, base_30261)
        
        # Processing the call keyword arguments (line 80)
        kwargs_30262 = {}
        # Getting the type of 'self' (line 80)
        self_30257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 80)
        assertEqual_30258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_30257, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 80)
        assertEqual_call_result_30263 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assertEqual_30258, *[dist_created_30259, list_30260], **kwargs_30262)
        
        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to ZipFile(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Call to join(...): (line 83)
        # Processing the call arguments (line 83)
        str_30269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 42), 'str', 'dist')
        # Getting the type of 'base' (line 83)
        base_30270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 50), 'base', False)
        # Processing the call keyword arguments (line 83)
        kwargs_30271 = {}
        # Getting the type of 'os' (line 83)
        os_30266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 83)
        path_30267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 29), os_30266, 'path')
        # Obtaining the member 'join' of a type (line 83)
        join_30268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 29), path_30267, 'join')
        # Calling join(args, kwargs) (line 83)
        join_call_result_30272 = invoke(stypy.reporting.localization.Localization(__file__, 83, 29), join_30268, *[str_30269, base_30270], **kwargs_30271)
        
        # Processing the call keyword arguments (line 83)
        kwargs_30273 = {}
        # Getting the type of 'zipfile' (line 83)
        zipfile_30264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'zipfile', False)
        # Obtaining the member 'ZipFile' of a type (line 83)
        ZipFile_30265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 13), zipfile_30264, 'ZipFile')
        # Calling ZipFile(args, kwargs) (line 83)
        ZipFile_call_result_30274 = invoke(stypy.reporting.localization.Localization(__file__, 83, 13), ZipFile_30265, *[join_call_result_30272], **kwargs_30273)
        
        # Assigning a type to the variable 'fp' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'fp', ZipFile_call_result_30274)
        
        # Try-finally block (line 84)
        
        # Assigning a Call to a Name (line 85):
        
        # Assigning a Call to a Name (line 85):
        
        # Call to namelist(...): (line 85)
        # Processing the call keyword arguments (line 85)
        kwargs_30277 = {}
        # Getting the type of 'fp' (line 85)
        fp_30275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'fp', False)
        # Obtaining the member 'namelist' of a type (line 85)
        namelist_30276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 23), fp_30275, 'namelist')
        # Calling namelist(args, kwargs) (line 85)
        namelist_call_result_30278 = invoke(stypy.reporting.localization.Localization(__file__, 85, 23), namelist_30276, *[], **kwargs_30277)
        
        # Assigning a type to the variable 'contents' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'contents', namelist_call_result_30278)
        
        # finally branch of the try-finally block (line 84)
        
        # Call to close(...): (line 87)
        # Processing the call keyword arguments (line 87)
        kwargs_30281 = {}
        # Getting the type of 'fp' (line 87)
        fp_30279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'fp', False)
        # Obtaining the member 'close' of a type (line 87)
        close_30280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), fp_30279, 'close')
        # Calling close(args, kwargs) (line 87)
        close_call_result_30282 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), close_30280, *[], **kwargs_30281)
        
        
        
        # Assigning a Call to a Name (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to sorted(...): (line 89)
        # Processing the call arguments (line 89)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 89, 26, True)
        # Calculating comprehension expression
        # Getting the type of 'contents' (line 89)
        contents_30290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 57), 'contents', False)
        comprehension_30291 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 26), contents_30290)
        # Assigning a type to the variable 'fn' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'fn', comprehension_30291)
        
        # Call to basename(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'fn' (line 89)
        fn_30287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 43), 'fn', False)
        # Processing the call keyword arguments (line 89)
        kwargs_30288 = {}
        # Getting the type of 'os' (line 89)
        os_30284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 89)
        path_30285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 26), os_30284, 'path')
        # Obtaining the member 'basename' of a type (line 89)
        basename_30286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 26), path_30285, 'basename')
        # Calling basename(args, kwargs) (line 89)
        basename_call_result_30289 = invoke(stypy.reporting.localization.Localization(__file__, 89, 26), basename_30286, *[fn_30287], **kwargs_30288)
        
        list_30292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 26), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 26), list_30292, basename_call_result_30289)
        # Processing the call keyword arguments (line 89)
        kwargs_30293 = {}
        # Getting the type of 'sorted' (line 89)
        sorted_30283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'sorted', False)
        # Calling sorted(args, kwargs) (line 89)
        sorted_call_result_30294 = invoke(stypy.reporting.localization.Localization(__file__, 89, 19), sorted_30283, *[list_30292], **kwargs_30293)
        
        # Assigning a type to the variable 'contents' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'contents', sorted_call_result_30294)
        
        # Assigning a List to a Name (line 90):
        
        # Assigning a List to a Name (line 90):
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_30295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        str_30296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'str', 'foo-0.1-py%s.%s.egg-info')
        
        # Obtaining the type of the subscript
        int_30297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 65), 'int')
        slice_30298 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 90, 47), None, int_30297, None)
        # Getting the type of 'sys' (line 90)
        sys_30299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 47), 'sys')
        # Obtaining the member 'version_info' of a type (line 90)
        version_info_30300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 47), sys_30299, 'version_info')
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___30301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 47), version_info_30300, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_30302 = invoke(stypy.reporting.localization.Localization(__file__, 90, 47), getitem___30301, slice_30298)
        
        # Applying the binary operator '%' (line 90)
        result_mod_30303 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 18), '%', str_30296, subscript_call_result_30302)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_30295, result_mod_30303)
        # Adding element type (line 90)
        str_30304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 69), 'str', 'foo.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_30295, str_30304)
        
        # Assigning a type to the variable 'wanted' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'wanted', list_30295)
        
        
        # Getting the type of 'sys' (line 91)
        sys_30305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'sys')
        # Obtaining the member 'dont_write_bytecode' of a type (line 91)
        dont_write_bytecode_30306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 15), sys_30305, 'dont_write_bytecode')
        # Applying the 'not' unary operator (line 91)
        result_not__30307 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 11), 'not', dont_write_bytecode_30306)
        
        # Testing the type of an if condition (line 91)
        if_condition_30308 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 8), result_not__30307)
        # Assigning a type to the variable 'if_condition_30308' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'if_condition_30308', if_condition_30308)
        # SSA begins for if statement (line 91)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 92)
        # Processing the call arguments (line 92)
        str_30311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 26), 'str', 'foo.pyc')
        # Processing the call keyword arguments (line 92)
        kwargs_30312 = {}
        # Getting the type of 'wanted' (line 92)
        wanted_30309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'wanted', False)
        # Obtaining the member 'append' of a type (line 92)
        append_30310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), wanted_30309, 'append')
        # Calling append(args, kwargs) (line 92)
        append_call_result_30313 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), append_30310, *[str_30311], **kwargs_30312)
        
        # SSA join for if statement (line 91)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertEqual(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'contents' (line 93)
        contents_30316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 25), 'contents', False)
        
        # Call to sorted(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'wanted' (line 93)
        wanted_30318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 42), 'wanted', False)
        # Processing the call keyword arguments (line 93)
        kwargs_30319 = {}
        # Getting the type of 'sorted' (line 93)
        sorted_30317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 35), 'sorted', False)
        # Calling sorted(args, kwargs) (line 93)
        sorted_call_result_30320 = invoke(stypy.reporting.localization.Localization(__file__, 93, 35), sorted_30317, *[wanted_30318], **kwargs_30319)
        
        # Processing the call keyword arguments (line 93)
        kwargs_30321 = {}
        # Getting the type of 'self' (line 93)
        self_30314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 93)
        assertEqual_30315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), self_30314, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 93)
        assertEqual_call_result_30322 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), assertEqual_30315, *[contents_30316, sorted_call_result_30320], **kwargs_30321)
        
        
        # ################# End of 'test_simple_built(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_built' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_30323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30323)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_built'
        return stypy_return_type_30323


    @norecursion
    def test_finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_finalize_options'
        module_type_store = module_type_store.open_function_context('test_finalize_options', 95, 4, False)
        # Assigning a type to the variable 'self' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildDumbTestCase.test_finalize_options.__dict__.__setitem__('stypy_localization', localization)
        BuildDumbTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildDumbTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildDumbTestCase.test_finalize_options.__dict__.__setitem__('stypy_function_name', 'BuildDumbTestCase.test_finalize_options')
        BuildDumbTestCase.test_finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        BuildDumbTestCase.test_finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildDumbTestCase.test_finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildDumbTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildDumbTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildDumbTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildDumbTestCase.test_finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildDumbTestCase.test_finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 96):
        
        # Assigning a Subscript to a Name (line 96):
        
        # Obtaining the type of the subscript
        int_30324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'int')
        
        # Call to create_dist(...): (line 96)
        # Processing the call keyword arguments (line 96)
        kwargs_30327 = {}
        # Getting the type of 'self' (line 96)
        self_30325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 96)
        create_dist_30326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 24), self_30325, 'create_dist')
        # Calling create_dist(args, kwargs) (line 96)
        create_dist_call_result_30328 = invoke(stypy.reporting.localization.Localization(__file__, 96, 24), create_dist_30326, *[], **kwargs_30327)
        
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___30329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), create_dist_call_result_30328, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_30330 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), getitem___30329, int_30324)
        
        # Assigning a type to the variable 'tuple_var_assignment_30064' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_30064', subscript_call_result_30330)
        
        # Assigning a Subscript to a Name (line 96):
        
        # Obtaining the type of the subscript
        int_30331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'int')
        
        # Call to create_dist(...): (line 96)
        # Processing the call keyword arguments (line 96)
        kwargs_30334 = {}
        # Getting the type of 'self' (line 96)
        self_30332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 96)
        create_dist_30333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 24), self_30332, 'create_dist')
        # Calling create_dist(args, kwargs) (line 96)
        create_dist_call_result_30335 = invoke(stypy.reporting.localization.Localization(__file__, 96, 24), create_dist_30333, *[], **kwargs_30334)
        
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___30336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), create_dist_call_result_30335, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_30337 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), getitem___30336, int_30331)
        
        # Assigning a type to the variable 'tuple_var_assignment_30065' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_30065', subscript_call_result_30337)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'tuple_var_assignment_30064' (line 96)
        tuple_var_assignment_30064_30338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_30064')
        # Assigning a type to the variable 'pkg_dir' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'pkg_dir', tuple_var_assignment_30064_30338)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'tuple_var_assignment_30065' (line 96)
        tuple_var_assignment_30065_30339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_30065')
        # Assigning a type to the variable 'dist' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'dist', tuple_var_assignment_30065_30339)
        
        # Call to chdir(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'pkg_dir' (line 97)
        pkg_dir_30342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 17), 'pkg_dir', False)
        # Processing the call keyword arguments (line 97)
        kwargs_30343 = {}
        # Getting the type of 'os' (line 97)
        os_30340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 97)
        chdir_30341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), os_30340, 'chdir')
        # Calling chdir(args, kwargs) (line 97)
        chdir_call_result_30344 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), chdir_30341, *[pkg_dir_30342], **kwargs_30343)
        
        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to bdist_dumb(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'dist' (line 98)
        dist_30346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'dist', False)
        # Processing the call keyword arguments (line 98)
        kwargs_30347 = {}
        # Getting the type of 'bdist_dumb' (line 98)
        bdist_dumb_30345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'bdist_dumb', False)
        # Calling bdist_dumb(args, kwargs) (line 98)
        bdist_dumb_call_result_30348 = invoke(stypy.reporting.localization.Localization(__file__, 98, 14), bdist_dumb_30345, *[dist_30346], **kwargs_30347)
        
        # Assigning a type to the variable 'cmd' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'cmd', bdist_dumb_call_result_30348)
        
        # Call to assertEqual(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'cmd' (line 99)
        cmd_30351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'cmd', False)
        # Obtaining the member 'bdist_dir' of a type (line 99)
        bdist_dir_30352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 25), cmd_30351, 'bdist_dir')
        # Getting the type of 'None' (line 99)
        None_30353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 40), 'None', False)
        # Processing the call keyword arguments (line 99)
        kwargs_30354 = {}
        # Getting the type of 'self' (line 99)
        self_30349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 99)
        assertEqual_30350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_30349, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 99)
        assertEqual_call_result_30355 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), assertEqual_30350, *[bdist_dir_30352, None_30353], **kwargs_30354)
        
        
        # Call to finalize_options(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_30358 = {}
        # Getting the type of 'cmd' (line 100)
        cmd_30356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 100)
        finalize_options_30357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), cmd_30356, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 100)
        finalize_options_call_result_30359 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), finalize_options_30357, *[], **kwargs_30358)
        
        
        # Assigning a Attribute to a Name (line 103):
        
        # Assigning a Attribute to a Name (line 103):
        
        # Call to get_finalized_command(...): (line 103)
        # Processing the call arguments (line 103)
        str_30362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 41), 'str', 'bdist')
        # Processing the call keyword arguments (line 103)
        kwargs_30363 = {}
        # Getting the type of 'cmd' (line 103)
        cmd_30360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'cmd', False)
        # Obtaining the member 'get_finalized_command' of a type (line 103)
        get_finalized_command_30361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), cmd_30360, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 103)
        get_finalized_command_call_result_30364 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), get_finalized_command_30361, *[str_30362], **kwargs_30363)
        
        # Obtaining the member 'bdist_base' of a type (line 103)
        bdist_base_30365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), get_finalized_command_call_result_30364, 'bdist_base')
        # Assigning a type to the variable 'base' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'base', bdist_base_30365)
        
        # Call to assertEqual(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'cmd' (line 104)
        cmd_30368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'cmd', False)
        # Obtaining the member 'bdist_dir' of a type (line 104)
        bdist_dir_30369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 25), cmd_30368, 'bdist_dir')
        
        # Call to join(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'base' (line 104)
        base_30373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 53), 'base', False)
        str_30374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 59), 'str', 'dumb')
        # Processing the call keyword arguments (line 104)
        kwargs_30375 = {}
        # Getting the type of 'os' (line 104)
        os_30370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 40), 'os', False)
        # Obtaining the member 'path' of a type (line 104)
        path_30371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 40), os_30370, 'path')
        # Obtaining the member 'join' of a type (line 104)
        join_30372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 40), path_30371, 'join')
        # Calling join(args, kwargs) (line 104)
        join_call_result_30376 = invoke(stypy.reporting.localization.Localization(__file__, 104, 40), join_30372, *[base_30373, str_30374], **kwargs_30375)
        
        # Processing the call keyword arguments (line 104)
        kwargs_30377 = {}
        # Getting the type of 'self' (line 104)
        self_30366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 104)
        assertEqual_30367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_30366, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 104)
        assertEqual_call_result_30378 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), assertEqual_30367, *[bdist_dir_30369, join_call_result_30376], **kwargs_30377)
        
        
        # Assigning a Subscript to a Name (line 107):
        
        # Assigning a Subscript to a Name (line 107):
        
        # Obtaining the type of the subscript
        # Getting the type of 'os' (line 107)
        os_30379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 37), 'os')
        # Obtaining the member 'name' of a type (line 107)
        name_30380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 37), os_30379, 'name')
        # Getting the type of 'cmd' (line 107)
        cmd_30381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 18), 'cmd')
        # Obtaining the member 'default_format' of a type (line 107)
        default_format_30382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 18), cmd_30381, 'default_format')
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___30383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 18), default_format_30382, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_30384 = invoke(stypy.reporting.localization.Localization(__file__, 107, 18), getitem___30383, name_30380)
        
        # Assigning a type to the variable 'default' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'default', subscript_call_result_30384)
        
        # Call to assertEqual(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'cmd' (line 108)
        cmd_30387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'cmd', False)
        # Obtaining the member 'format' of a type (line 108)
        format_30388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 25), cmd_30387, 'format')
        # Getting the type of 'default' (line 108)
        default_30389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 37), 'default', False)
        # Processing the call keyword arguments (line 108)
        kwargs_30390 = {}
        # Getting the type of 'self' (line 108)
        self_30385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 108)
        assertEqual_30386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_30385, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 108)
        assertEqual_call_result_30391 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), assertEqual_30386, *[format_30388, default_30389], **kwargs_30390)
        
        
        # ################# End of 'test_finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 95)
        stypy_return_type_30392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30392)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_finalize_options'
        return stypy_return_type_30392


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 29, 0, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildDumbTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BuildDumbTestCase' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'BuildDumbTestCase', BuildDumbTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 110, 0, False)
    
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

    
    # Call to makeSuite(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'BuildDumbTestCase' (line 111)
    BuildDumbTestCase_30395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'BuildDumbTestCase', False)
    # Processing the call keyword arguments (line 111)
    kwargs_30396 = {}
    # Getting the type of 'unittest' (line 111)
    unittest_30393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 111)
    makeSuite_30394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 11), unittest_30393, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 111)
    makeSuite_call_result_30397 = invoke(stypy.reporting.localization.Localization(__file__, 111, 11), makeSuite_30394, *[BuildDumbTestCase_30395], **kwargs_30396)
    
    # Assigning a type to the variable 'stypy_return_type' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type', makeSuite_call_result_30397)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 110)
    stypy_return_type_30398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30398)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_30398

# Assigning a type to the variable 'test_suite' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 114)
    # Processing the call arguments (line 114)
    
    # Call to test_suite(...): (line 114)
    # Processing the call keyword arguments (line 114)
    kwargs_30401 = {}
    # Getting the type of 'test_suite' (line 114)
    test_suite_30400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 114)
    test_suite_call_result_30402 = invoke(stypy.reporting.localization.Localization(__file__, 114, 17), test_suite_30400, *[], **kwargs_30401)
    
    # Processing the call keyword arguments (line 114)
    kwargs_30403 = {}
    # Getting the type of 'run_unittest' (line 114)
    run_unittest_30399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 114)
    run_unittest_call_result_30404 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), run_unittest_30399, *[test_suite_call_result_30402], **kwargs_30403)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
