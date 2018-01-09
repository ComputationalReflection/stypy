
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.sysconfig.'''
2: import os
3: import test
4: import unittest
5: import shutil
6: import subprocess
7: import sys
8: import textwrap
9: 
10: from distutils import sysconfig
11: from distutils.tests import support
12: from test.test_support import TESTFN
13: 
14: class SysconfigTestCase(support.EnvironGuard,
15:                         unittest.TestCase):
16:     def setUp(self):
17:         super(SysconfigTestCase, self).setUp()
18:         self.makefile = None
19: 
20:     def tearDown(self):
21:         if self.makefile is not None:
22:             os.unlink(self.makefile)
23:         self.cleanup_testfn()
24:         super(SysconfigTestCase, self).tearDown()
25: 
26:     def cleanup_testfn(self):
27:         path = test.test_support.TESTFN
28:         if os.path.isfile(path):
29:             os.remove(path)
30:         elif os.path.isdir(path):
31:             shutil.rmtree(path)
32: 
33:     def test_get_python_lib(self):
34:         lib_dir = sysconfig.get_python_lib()
35:         # XXX doesn't work on Linux when Python was never installed before
36:         #self.assertTrue(os.path.isdir(lib_dir), lib_dir)
37:         # test for pythonxx.lib?
38:         self.assertNotEqual(sysconfig.get_python_lib(),
39:                             sysconfig.get_python_lib(prefix=TESTFN))
40:         _sysconfig = __import__('sysconfig')
41:         res = sysconfig.get_python_lib(True, True)
42:         self.assertEqual(_sysconfig.get_path('platstdlib'), res)
43: 
44:     def test_get_python_inc(self):
45:         inc_dir = sysconfig.get_python_inc()
46:         # This is not much of a test.  We make sure Python.h exists
47:         # in the directory returned by get_python_inc() but we don't know
48:         # it is the correct file.
49:         self.assertTrue(os.path.isdir(inc_dir), inc_dir)
50:         python_h = os.path.join(inc_dir, "Python.h")
51:         self.assertTrue(os.path.isfile(python_h), python_h)
52: 
53:     def test_parse_makefile_base(self):
54:         self.makefile = test.test_support.TESTFN
55:         fd = open(self.makefile, 'w')
56:         try:
57:             fd.write(r"CONFIG_ARGS=  '--arg1=optarg1' 'ENV=LIB'" '\n')
58:             fd.write('VAR=$OTHER\nOTHER=foo')
59:         finally:
60:             fd.close()
61:         d = sysconfig.parse_makefile(self.makefile)
62:         self.assertEqual(d, {'CONFIG_ARGS': "'--arg1=optarg1' 'ENV=LIB'",
63:                              'OTHER': 'foo'})
64: 
65:     def test_parse_makefile_literal_dollar(self):
66:         self.makefile = test.test_support.TESTFN
67:         fd = open(self.makefile, 'w')
68:         try:
69:             fd.write(r"CONFIG_ARGS=  '--arg1=optarg1' 'ENV=\$$LIB'" '\n')
70:             fd.write('VAR=$OTHER\nOTHER=foo')
71:         finally:
72:             fd.close()
73:         d = sysconfig.parse_makefile(self.makefile)
74:         self.assertEqual(d, {'CONFIG_ARGS': r"'--arg1=optarg1' 'ENV=\$LIB'",
75:                              'OTHER': 'foo'})
76: 
77: 
78:     def test_sysconfig_module(self):
79:         import sysconfig as global_sysconfig
80:         self.assertEqual(global_sysconfig.get_config_var('CFLAGS'), sysconfig.get_config_var('CFLAGS'))
81:         self.assertEqual(global_sysconfig.get_config_var('LDFLAGS'), sysconfig.get_config_var('LDFLAGS'))
82: 
83:     @unittest.skipIf(sysconfig.get_config_var('CUSTOMIZED_OSX_COMPILER'),'compiler flags customized')
84:     def test_sysconfig_compiler_vars(self):
85:         # On OS X, binary installers support extension module building on
86:         # various levels of the operating system with differing Xcode
87:         # configurations.  This requires customization of some of the
88:         # compiler configuration directives to suit the environment on
89:         # the installed machine.  Some of these customizations may require
90:         # running external programs and, so, are deferred until needed by
91:         # the first extension module build.  With Python 3.3, only
92:         # the Distutils version of sysconfig is used for extension module
93:         # builds, which happens earlier in the Distutils tests.  This may
94:         # cause the following tests to fail since no tests have caused
95:         # the global version of sysconfig to call the customization yet.
96:         # The solution for now is to simply skip this test in this case.
97:         # The longer-term solution is to only have one version of sysconfig.
98: 
99:         import sysconfig as global_sysconfig
100:         if sysconfig.get_config_var('CUSTOMIZED_OSX_COMPILER'):
101:             self.skipTest('compiler flags customized')
102:         self.assertEqual(global_sysconfig.get_config_var('LDSHARED'), sysconfig.get_config_var('LDSHARED'))
103:         self.assertEqual(global_sysconfig.get_config_var('CC'), sysconfig.get_config_var('CC'))
104: 
105:     def test_customize_compiler_before_get_config_vars(self):
106:         # Issue #21923: test that a Distribution compiler
107:         # instance can be called without an explicit call to
108:         # get_config_vars().
109:         with open(TESTFN, 'w') as f:
110:             f.writelines(textwrap.dedent('''\
111:                 from distutils.core import Distribution
112:                 config = Distribution().get_command_obj('config')
113:                 # try_compile may pass or it may fail if no compiler
114:                 # is found but it should not raise an exception.
115:                 rc = config.try_compile('int x;')
116:                 '''))
117:         p = subprocess.Popen([str(sys.executable), TESTFN],
118:                 stdout=subprocess.PIPE,
119:                 stderr=subprocess.STDOUT,
120:                 universal_newlines=True)
121:         outs, errs = p.communicate()
122:         self.assertEqual(0, p.returncode, "Subprocess failed: " + outs)
123: 
124: 
125: def test_suite():
126:     suite = unittest.TestSuite()
127:     suite.addTest(unittest.makeSuite(SysconfigTestCase))
128:     return suite
129: 
130: 
131: if __name__ == '__main__':
132:     test.test_support.run_unittest(test_suite())
133: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_44214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.sysconfig.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import os' statement (line 2)
import os

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import test' statement (line 3)
import test

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'test', test, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import unittest' statement (line 4)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import shutil' statement (line 5)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import subprocess' statement (line 6)
import subprocess

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'subprocess', subprocess, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import sys' statement (line 7)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import textwrap' statement (line 8)
import textwrap

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'textwrap', textwrap, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils import sysconfig' statement (line 10)
try:
    from distutils import sysconfig

except:
    sysconfig = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils', None, module_type_store, ['sysconfig'], [sysconfig])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.tests import support' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_44215 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests')

if (type(import_44215) is not StypyTypeError):

    if (import_44215 != 'pyd_module'):
        __import__(import_44215)
        sys_modules_44216 = sys.modules[import_44215]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests', sys_modules_44216.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_44216, sys_modules_44216.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests', import_44215)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from test.test_support import TESTFN' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_44217 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'test.test_support')

if (type(import_44217) is not StypyTypeError):

    if (import_44217 != 'pyd_module'):
        __import__(import_44217)
        sys_modules_44218 = sys.modules[import_44217]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'test.test_support', sys_modules_44218.module_type_store, module_type_store, ['TESTFN'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_44218, sys_modules_44218.module_type_store, module_type_store)
    else:
        from test.test_support import TESTFN

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'test.test_support', None, module_type_store, ['TESTFN'], [TESTFN])

else:
    # Assigning a type to the variable 'test.test_support' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'test.test_support', import_44217)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'SysconfigTestCase' class
# Getting the type of 'support' (line 14)
support_44219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 24), 'support')
# Obtaining the member 'EnvironGuard' of a type (line 14)
EnvironGuard_44220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 24), support_44219, 'EnvironGuard')
# Getting the type of 'unittest' (line 15)
unittest_44221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'unittest')
# Obtaining the member 'TestCase' of a type (line 15)
TestCase_44222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 24), unittest_44221, 'TestCase')

class SysconfigTestCase(EnvironGuard_44220, TestCase_44222, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SysconfigTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        SysconfigTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SysconfigTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        SysconfigTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'SysconfigTestCase.setUp')
        SysconfigTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        SysconfigTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        SysconfigTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SysconfigTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        SysconfigTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        SysconfigTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SysconfigTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SysconfigTestCase.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to setUp(...): (line 17)
        # Processing the call keyword arguments (line 17)
        kwargs_44229 = {}
        
        # Call to super(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'SysconfigTestCase' (line 17)
        SysconfigTestCase_44224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'SysconfigTestCase', False)
        # Getting the type of 'self' (line 17)
        self_44225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 33), 'self', False)
        # Processing the call keyword arguments (line 17)
        kwargs_44226 = {}
        # Getting the type of 'super' (line 17)
        super_44223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'super', False)
        # Calling super(args, kwargs) (line 17)
        super_call_result_44227 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), super_44223, *[SysconfigTestCase_44224, self_44225], **kwargs_44226)
        
        # Obtaining the member 'setUp' of a type (line 17)
        setUp_44228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), super_call_result_44227, 'setUp')
        # Calling setUp(args, kwargs) (line 17)
        setUp_call_result_44230 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), setUp_44228, *[], **kwargs_44229)
        
        
        # Assigning a Name to a Attribute (line 18):
        
        # Assigning a Name to a Attribute (line 18):
        # Getting the type of 'None' (line 18)
        None_44231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'None')
        # Getting the type of 'self' (line 18)
        self_44232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self')
        # Setting the type of the member 'makefile' of a type (line 18)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), self_44232, 'makefile', None_44231)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_44233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44233)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_44233


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SysconfigTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        SysconfigTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SysconfigTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        SysconfigTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'SysconfigTestCase.tearDown')
        SysconfigTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        SysconfigTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        SysconfigTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SysconfigTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        SysconfigTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        SysconfigTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SysconfigTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SysconfigTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 21)
        self_44234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'self')
        # Obtaining the member 'makefile' of a type (line 21)
        makefile_44235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 11), self_44234, 'makefile')
        # Getting the type of 'None' (line 21)
        None_44236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 32), 'None')
        # Applying the binary operator 'isnot' (line 21)
        result_is_not_44237 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 11), 'isnot', makefile_44235, None_44236)
        
        # Testing the type of an if condition (line 21)
        if_condition_44238 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 8), result_is_not_44237)
        # Assigning a type to the variable 'if_condition_44238' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'if_condition_44238', if_condition_44238)
        # SSA begins for if statement (line 21)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to unlink(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'self' (line 22)
        self_44241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 22), 'self', False)
        # Obtaining the member 'makefile' of a type (line 22)
        makefile_44242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 22), self_44241, 'makefile')
        # Processing the call keyword arguments (line 22)
        kwargs_44243 = {}
        # Getting the type of 'os' (line 22)
        os_44239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'os', False)
        # Obtaining the member 'unlink' of a type (line 22)
        unlink_44240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 12), os_44239, 'unlink')
        # Calling unlink(args, kwargs) (line 22)
        unlink_call_result_44244 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), unlink_44240, *[makefile_44242], **kwargs_44243)
        
        # SSA join for if statement (line 21)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to cleanup_testfn(...): (line 23)
        # Processing the call keyword arguments (line 23)
        kwargs_44247 = {}
        # Getting the type of 'self' (line 23)
        self_44245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self', False)
        # Obtaining the member 'cleanup_testfn' of a type (line 23)
        cleanup_testfn_44246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_44245, 'cleanup_testfn')
        # Calling cleanup_testfn(args, kwargs) (line 23)
        cleanup_testfn_call_result_44248 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), cleanup_testfn_44246, *[], **kwargs_44247)
        
        
        # Call to tearDown(...): (line 24)
        # Processing the call keyword arguments (line 24)
        kwargs_44255 = {}
        
        # Call to super(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'SysconfigTestCase' (line 24)
        SysconfigTestCase_44250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 14), 'SysconfigTestCase', False)
        # Getting the type of 'self' (line 24)
        self_44251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 33), 'self', False)
        # Processing the call keyword arguments (line 24)
        kwargs_44252 = {}
        # Getting the type of 'super' (line 24)
        super_44249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'super', False)
        # Calling super(args, kwargs) (line 24)
        super_call_result_44253 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), super_44249, *[SysconfigTestCase_44250, self_44251], **kwargs_44252)
        
        # Obtaining the member 'tearDown' of a type (line 24)
        tearDown_44254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), super_call_result_44253, 'tearDown')
        # Calling tearDown(args, kwargs) (line 24)
        tearDown_call_result_44256 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), tearDown_44254, *[], **kwargs_44255)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_44257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44257)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_44257


    @norecursion
    def cleanup_testfn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cleanup_testfn'
        module_type_store = module_type_store.open_function_context('cleanup_testfn', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SysconfigTestCase.cleanup_testfn.__dict__.__setitem__('stypy_localization', localization)
        SysconfigTestCase.cleanup_testfn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SysconfigTestCase.cleanup_testfn.__dict__.__setitem__('stypy_type_store', module_type_store)
        SysconfigTestCase.cleanup_testfn.__dict__.__setitem__('stypy_function_name', 'SysconfigTestCase.cleanup_testfn')
        SysconfigTestCase.cleanup_testfn.__dict__.__setitem__('stypy_param_names_list', [])
        SysconfigTestCase.cleanup_testfn.__dict__.__setitem__('stypy_varargs_param_name', None)
        SysconfigTestCase.cleanup_testfn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SysconfigTestCase.cleanup_testfn.__dict__.__setitem__('stypy_call_defaults', defaults)
        SysconfigTestCase.cleanup_testfn.__dict__.__setitem__('stypy_call_varargs', varargs)
        SysconfigTestCase.cleanup_testfn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SysconfigTestCase.cleanup_testfn.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SysconfigTestCase.cleanup_testfn', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cleanup_testfn', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cleanup_testfn(...)' code ##################

        
        # Assigning a Attribute to a Name (line 27):
        
        # Assigning a Attribute to a Name (line 27):
        # Getting the type of 'test' (line 27)
        test_44258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'test')
        # Obtaining the member 'test_support' of a type (line 27)
        test_support_44259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 15), test_44258, 'test_support')
        # Obtaining the member 'TESTFN' of a type (line 27)
        TESTFN_44260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 15), test_support_44259, 'TESTFN')
        # Assigning a type to the variable 'path' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'path', TESTFN_44260)
        
        
        # Call to isfile(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'path' (line 28)
        path_44264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'path', False)
        # Processing the call keyword arguments (line 28)
        kwargs_44265 = {}
        # Getting the type of 'os' (line 28)
        os_44261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 28)
        path_44262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 11), os_44261, 'path')
        # Obtaining the member 'isfile' of a type (line 28)
        isfile_44263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 11), path_44262, 'isfile')
        # Calling isfile(args, kwargs) (line 28)
        isfile_call_result_44266 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), isfile_44263, *[path_44264], **kwargs_44265)
        
        # Testing the type of an if condition (line 28)
        if_condition_44267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 8), isfile_call_result_44266)
        # Assigning a type to the variable 'if_condition_44267' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'if_condition_44267', if_condition_44267)
        # SSA begins for if statement (line 28)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'path' (line 29)
        path_44270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'path', False)
        # Processing the call keyword arguments (line 29)
        kwargs_44271 = {}
        # Getting the type of 'os' (line 29)
        os_44268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'os', False)
        # Obtaining the member 'remove' of a type (line 29)
        remove_44269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), os_44268, 'remove')
        # Calling remove(args, kwargs) (line 29)
        remove_call_result_44272 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), remove_44269, *[path_44270], **kwargs_44271)
        
        # SSA branch for the else part of an if statement (line 28)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isdir(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'path' (line 30)
        path_44276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 27), 'path', False)
        # Processing the call keyword arguments (line 30)
        kwargs_44277 = {}
        # Getting the type of 'os' (line 30)
        os_44273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 13), 'os', False)
        # Obtaining the member 'path' of a type (line 30)
        path_44274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 13), os_44273, 'path')
        # Obtaining the member 'isdir' of a type (line 30)
        isdir_44275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 13), path_44274, 'isdir')
        # Calling isdir(args, kwargs) (line 30)
        isdir_call_result_44278 = invoke(stypy.reporting.localization.Localization(__file__, 30, 13), isdir_44275, *[path_44276], **kwargs_44277)
        
        # Testing the type of an if condition (line 30)
        if_condition_44279 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 13), isdir_call_result_44278)
        # Assigning a type to the variable 'if_condition_44279' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 13), 'if_condition_44279', if_condition_44279)
        # SSA begins for if statement (line 30)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to rmtree(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'path' (line 31)
        path_44282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'path', False)
        # Processing the call keyword arguments (line 31)
        kwargs_44283 = {}
        # Getting the type of 'shutil' (line 31)
        shutil_44280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'shutil', False)
        # Obtaining the member 'rmtree' of a type (line 31)
        rmtree_44281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), shutil_44280, 'rmtree')
        # Calling rmtree(args, kwargs) (line 31)
        rmtree_call_result_44284 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), rmtree_44281, *[path_44282], **kwargs_44283)
        
        # SSA join for if statement (line 30)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 28)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'cleanup_testfn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cleanup_testfn' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_44285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44285)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cleanup_testfn'
        return stypy_return_type_44285


    @norecursion
    def test_get_python_lib(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_get_python_lib'
        module_type_store = module_type_store.open_function_context('test_get_python_lib', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SysconfigTestCase.test_get_python_lib.__dict__.__setitem__('stypy_localization', localization)
        SysconfigTestCase.test_get_python_lib.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SysconfigTestCase.test_get_python_lib.__dict__.__setitem__('stypy_type_store', module_type_store)
        SysconfigTestCase.test_get_python_lib.__dict__.__setitem__('stypy_function_name', 'SysconfigTestCase.test_get_python_lib')
        SysconfigTestCase.test_get_python_lib.__dict__.__setitem__('stypy_param_names_list', [])
        SysconfigTestCase.test_get_python_lib.__dict__.__setitem__('stypy_varargs_param_name', None)
        SysconfigTestCase.test_get_python_lib.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SysconfigTestCase.test_get_python_lib.__dict__.__setitem__('stypy_call_defaults', defaults)
        SysconfigTestCase.test_get_python_lib.__dict__.__setitem__('stypy_call_varargs', varargs)
        SysconfigTestCase.test_get_python_lib.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SysconfigTestCase.test_get_python_lib.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SysconfigTestCase.test_get_python_lib', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_get_python_lib', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_get_python_lib(...)' code ##################

        
        # Assigning a Call to a Name (line 34):
        
        # Assigning a Call to a Name (line 34):
        
        # Call to get_python_lib(...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_44288 = {}
        # Getting the type of 'sysconfig' (line 34)
        sysconfig_44286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'sysconfig', False)
        # Obtaining the member 'get_python_lib' of a type (line 34)
        get_python_lib_44287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 18), sysconfig_44286, 'get_python_lib')
        # Calling get_python_lib(args, kwargs) (line 34)
        get_python_lib_call_result_44289 = invoke(stypy.reporting.localization.Localization(__file__, 34, 18), get_python_lib_44287, *[], **kwargs_44288)
        
        # Assigning a type to the variable 'lib_dir' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'lib_dir', get_python_lib_call_result_44289)
        
        # Call to assertNotEqual(...): (line 38)
        # Processing the call arguments (line 38)
        
        # Call to get_python_lib(...): (line 38)
        # Processing the call keyword arguments (line 38)
        kwargs_44294 = {}
        # Getting the type of 'sysconfig' (line 38)
        sysconfig_44292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 28), 'sysconfig', False)
        # Obtaining the member 'get_python_lib' of a type (line 38)
        get_python_lib_44293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 28), sysconfig_44292, 'get_python_lib')
        # Calling get_python_lib(args, kwargs) (line 38)
        get_python_lib_call_result_44295 = invoke(stypy.reporting.localization.Localization(__file__, 38, 28), get_python_lib_44293, *[], **kwargs_44294)
        
        
        # Call to get_python_lib(...): (line 39)
        # Processing the call keyword arguments (line 39)
        # Getting the type of 'TESTFN' (line 39)
        TESTFN_44298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 60), 'TESTFN', False)
        keyword_44299 = TESTFN_44298
        kwargs_44300 = {'prefix': keyword_44299}
        # Getting the type of 'sysconfig' (line 39)
        sysconfig_44296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 28), 'sysconfig', False)
        # Obtaining the member 'get_python_lib' of a type (line 39)
        get_python_lib_44297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 28), sysconfig_44296, 'get_python_lib')
        # Calling get_python_lib(args, kwargs) (line 39)
        get_python_lib_call_result_44301 = invoke(stypy.reporting.localization.Localization(__file__, 39, 28), get_python_lib_44297, *[], **kwargs_44300)
        
        # Processing the call keyword arguments (line 38)
        kwargs_44302 = {}
        # Getting the type of 'self' (line 38)
        self_44290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self', False)
        # Obtaining the member 'assertNotEqual' of a type (line 38)
        assertNotEqual_44291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_44290, 'assertNotEqual')
        # Calling assertNotEqual(args, kwargs) (line 38)
        assertNotEqual_call_result_44303 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), assertNotEqual_44291, *[get_python_lib_call_result_44295, get_python_lib_call_result_44301], **kwargs_44302)
        
        
        # Assigning a Call to a Name (line 40):
        
        # Assigning a Call to a Name (line 40):
        
        # Call to __import__(...): (line 40)
        # Processing the call arguments (line 40)
        str_44305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 32), 'str', 'sysconfig')
        # Processing the call keyword arguments (line 40)
        kwargs_44306 = {}
        # Getting the type of '__import__' (line 40)
        import___44304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 21), '__import__', False)
        # Calling __import__(args, kwargs) (line 40)
        import___call_result_44307 = invoke(stypy.reporting.localization.Localization(__file__, 40, 21), import___44304, *[str_44305], **kwargs_44306)
        
        # Assigning a type to the variable '_sysconfig' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), '_sysconfig', import___call_result_44307)
        
        # Assigning a Call to a Name (line 41):
        
        # Assigning a Call to a Name (line 41):
        
        # Call to get_python_lib(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'True' (line 41)
        True_44310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 39), 'True', False)
        # Getting the type of 'True' (line 41)
        True_44311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 45), 'True', False)
        # Processing the call keyword arguments (line 41)
        kwargs_44312 = {}
        # Getting the type of 'sysconfig' (line 41)
        sysconfig_44308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'sysconfig', False)
        # Obtaining the member 'get_python_lib' of a type (line 41)
        get_python_lib_44309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 14), sysconfig_44308, 'get_python_lib')
        # Calling get_python_lib(args, kwargs) (line 41)
        get_python_lib_call_result_44313 = invoke(stypy.reporting.localization.Localization(__file__, 41, 14), get_python_lib_44309, *[True_44310, True_44311], **kwargs_44312)
        
        # Assigning a type to the variable 'res' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'res', get_python_lib_call_result_44313)
        
        # Call to assertEqual(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to get_path(...): (line 42)
        # Processing the call arguments (line 42)
        str_44318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 45), 'str', 'platstdlib')
        # Processing the call keyword arguments (line 42)
        kwargs_44319 = {}
        # Getting the type of '_sysconfig' (line 42)
        _sysconfig_44316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 25), '_sysconfig', False)
        # Obtaining the member 'get_path' of a type (line 42)
        get_path_44317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 25), _sysconfig_44316, 'get_path')
        # Calling get_path(args, kwargs) (line 42)
        get_path_call_result_44320 = invoke(stypy.reporting.localization.Localization(__file__, 42, 25), get_path_44317, *[str_44318], **kwargs_44319)
        
        # Getting the type of 'res' (line 42)
        res_44321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 60), 'res', False)
        # Processing the call keyword arguments (line 42)
        kwargs_44322 = {}
        # Getting the type of 'self' (line 42)
        self_44314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 42)
        assertEqual_44315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_44314, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 42)
        assertEqual_call_result_44323 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assertEqual_44315, *[get_path_call_result_44320, res_44321], **kwargs_44322)
        
        
        # ################# End of 'test_get_python_lib(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_get_python_lib' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_44324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44324)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_get_python_lib'
        return stypy_return_type_44324


    @norecursion
    def test_get_python_inc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_get_python_inc'
        module_type_store = module_type_store.open_function_context('test_get_python_inc', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SysconfigTestCase.test_get_python_inc.__dict__.__setitem__('stypy_localization', localization)
        SysconfigTestCase.test_get_python_inc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SysconfigTestCase.test_get_python_inc.__dict__.__setitem__('stypy_type_store', module_type_store)
        SysconfigTestCase.test_get_python_inc.__dict__.__setitem__('stypy_function_name', 'SysconfigTestCase.test_get_python_inc')
        SysconfigTestCase.test_get_python_inc.__dict__.__setitem__('stypy_param_names_list', [])
        SysconfigTestCase.test_get_python_inc.__dict__.__setitem__('stypy_varargs_param_name', None)
        SysconfigTestCase.test_get_python_inc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SysconfigTestCase.test_get_python_inc.__dict__.__setitem__('stypy_call_defaults', defaults)
        SysconfigTestCase.test_get_python_inc.__dict__.__setitem__('stypy_call_varargs', varargs)
        SysconfigTestCase.test_get_python_inc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SysconfigTestCase.test_get_python_inc.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SysconfigTestCase.test_get_python_inc', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_get_python_inc', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_get_python_inc(...)' code ##################

        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to get_python_inc(...): (line 45)
        # Processing the call keyword arguments (line 45)
        kwargs_44327 = {}
        # Getting the type of 'sysconfig' (line 45)
        sysconfig_44325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'sysconfig', False)
        # Obtaining the member 'get_python_inc' of a type (line 45)
        get_python_inc_44326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 18), sysconfig_44325, 'get_python_inc')
        # Calling get_python_inc(args, kwargs) (line 45)
        get_python_inc_call_result_44328 = invoke(stypy.reporting.localization.Localization(__file__, 45, 18), get_python_inc_44326, *[], **kwargs_44327)
        
        # Assigning a type to the variable 'inc_dir' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'inc_dir', get_python_inc_call_result_44328)
        
        # Call to assertTrue(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Call to isdir(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'inc_dir' (line 49)
        inc_dir_44334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'inc_dir', False)
        # Processing the call keyword arguments (line 49)
        kwargs_44335 = {}
        # Getting the type of 'os' (line 49)
        os_44331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 49)
        path_44332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 24), os_44331, 'path')
        # Obtaining the member 'isdir' of a type (line 49)
        isdir_44333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 24), path_44332, 'isdir')
        # Calling isdir(args, kwargs) (line 49)
        isdir_call_result_44336 = invoke(stypy.reporting.localization.Localization(__file__, 49, 24), isdir_44333, *[inc_dir_44334], **kwargs_44335)
        
        # Getting the type of 'inc_dir' (line 49)
        inc_dir_44337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 48), 'inc_dir', False)
        # Processing the call keyword arguments (line 49)
        kwargs_44338 = {}
        # Getting the type of 'self' (line 49)
        self_44329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 49)
        assertTrue_44330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_44329, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 49)
        assertTrue_call_result_44339 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), assertTrue_44330, *[isdir_call_result_44336, inc_dir_44337], **kwargs_44338)
        
        
        # Assigning a Call to a Name (line 50):
        
        # Assigning a Call to a Name (line 50):
        
        # Call to join(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'inc_dir' (line 50)
        inc_dir_44343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 32), 'inc_dir', False)
        str_44344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 41), 'str', 'Python.h')
        # Processing the call keyword arguments (line 50)
        kwargs_44345 = {}
        # Getting the type of 'os' (line 50)
        os_44340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 50)
        path_44341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 19), os_44340, 'path')
        # Obtaining the member 'join' of a type (line 50)
        join_44342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 19), path_44341, 'join')
        # Calling join(args, kwargs) (line 50)
        join_call_result_44346 = invoke(stypy.reporting.localization.Localization(__file__, 50, 19), join_44342, *[inc_dir_44343, str_44344], **kwargs_44345)
        
        # Assigning a type to the variable 'python_h' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'python_h', join_call_result_44346)
        
        # Call to assertTrue(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to isfile(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'python_h' (line 51)
        python_h_44352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 39), 'python_h', False)
        # Processing the call keyword arguments (line 51)
        kwargs_44353 = {}
        # Getting the type of 'os' (line 51)
        os_44349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 51)
        path_44350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 24), os_44349, 'path')
        # Obtaining the member 'isfile' of a type (line 51)
        isfile_44351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 24), path_44350, 'isfile')
        # Calling isfile(args, kwargs) (line 51)
        isfile_call_result_44354 = invoke(stypy.reporting.localization.Localization(__file__, 51, 24), isfile_44351, *[python_h_44352], **kwargs_44353)
        
        # Getting the type of 'python_h' (line 51)
        python_h_44355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 50), 'python_h', False)
        # Processing the call keyword arguments (line 51)
        kwargs_44356 = {}
        # Getting the type of 'self' (line 51)
        self_44347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 51)
        assertTrue_44348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_44347, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 51)
        assertTrue_call_result_44357 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assertTrue_44348, *[isfile_call_result_44354, python_h_44355], **kwargs_44356)
        
        
        # ################# End of 'test_get_python_inc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_get_python_inc' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_44358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44358)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_get_python_inc'
        return stypy_return_type_44358


    @norecursion
    def test_parse_makefile_base(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_parse_makefile_base'
        module_type_store = module_type_store.open_function_context('test_parse_makefile_base', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SysconfigTestCase.test_parse_makefile_base.__dict__.__setitem__('stypy_localization', localization)
        SysconfigTestCase.test_parse_makefile_base.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SysconfigTestCase.test_parse_makefile_base.__dict__.__setitem__('stypy_type_store', module_type_store)
        SysconfigTestCase.test_parse_makefile_base.__dict__.__setitem__('stypy_function_name', 'SysconfigTestCase.test_parse_makefile_base')
        SysconfigTestCase.test_parse_makefile_base.__dict__.__setitem__('stypy_param_names_list', [])
        SysconfigTestCase.test_parse_makefile_base.__dict__.__setitem__('stypy_varargs_param_name', None)
        SysconfigTestCase.test_parse_makefile_base.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SysconfigTestCase.test_parse_makefile_base.__dict__.__setitem__('stypy_call_defaults', defaults)
        SysconfigTestCase.test_parse_makefile_base.__dict__.__setitem__('stypy_call_varargs', varargs)
        SysconfigTestCase.test_parse_makefile_base.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SysconfigTestCase.test_parse_makefile_base.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SysconfigTestCase.test_parse_makefile_base', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_parse_makefile_base', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_parse_makefile_base(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 54):
        
        # Assigning a Attribute to a Attribute (line 54):
        # Getting the type of 'test' (line 54)
        test_44359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'test')
        # Obtaining the member 'test_support' of a type (line 54)
        test_support_44360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), test_44359, 'test_support')
        # Obtaining the member 'TESTFN' of a type (line 54)
        TESTFN_44361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), test_support_44360, 'TESTFN')
        # Getting the type of 'self' (line 54)
        self_44362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self')
        # Setting the type of the member 'makefile' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_44362, 'makefile', TESTFN_44361)
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to open(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'self' (line 55)
        self_44364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 18), 'self', False)
        # Obtaining the member 'makefile' of a type (line 55)
        makefile_44365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 18), self_44364, 'makefile')
        str_44366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 33), 'str', 'w')
        # Processing the call keyword arguments (line 55)
        kwargs_44367 = {}
        # Getting the type of 'open' (line 55)
        open_44363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 13), 'open', False)
        # Calling open(args, kwargs) (line 55)
        open_call_result_44368 = invoke(stypy.reporting.localization.Localization(__file__, 55, 13), open_44363, *[makefile_44365, str_44366], **kwargs_44367)
        
        # Assigning a type to the variable 'fd' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'fd', open_call_result_44368)
        
        # Try-finally block (line 56)
        
        # Call to write(...): (line 57)
        # Processing the call arguments (line 57)
        str_44371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 21), 'str', "CONFIG_ARGS=  '--arg1=optarg1' 'ENV=LIB'\n")
        # Processing the call keyword arguments (line 57)
        kwargs_44372 = {}
        # Getting the type of 'fd' (line 57)
        fd_44369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'fd', False)
        # Obtaining the member 'write' of a type (line 57)
        write_44370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), fd_44369, 'write')
        # Calling write(args, kwargs) (line 57)
        write_call_result_44373 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), write_44370, *[str_44371], **kwargs_44372)
        
        
        # Call to write(...): (line 58)
        # Processing the call arguments (line 58)
        str_44376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'str', 'VAR=$OTHER\nOTHER=foo')
        # Processing the call keyword arguments (line 58)
        kwargs_44377 = {}
        # Getting the type of 'fd' (line 58)
        fd_44374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'fd', False)
        # Obtaining the member 'write' of a type (line 58)
        write_44375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), fd_44374, 'write')
        # Calling write(args, kwargs) (line 58)
        write_call_result_44378 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), write_44375, *[str_44376], **kwargs_44377)
        
        
        # finally branch of the try-finally block (line 56)
        
        # Call to close(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_44381 = {}
        # Getting the type of 'fd' (line 60)
        fd_44379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'fd', False)
        # Obtaining the member 'close' of a type (line 60)
        close_44380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), fd_44379, 'close')
        # Calling close(args, kwargs) (line 60)
        close_call_result_44382 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), close_44380, *[], **kwargs_44381)
        
        
        
        # Assigning a Call to a Name (line 61):
        
        # Assigning a Call to a Name (line 61):
        
        # Call to parse_makefile(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'self' (line 61)
        self_44385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 37), 'self', False)
        # Obtaining the member 'makefile' of a type (line 61)
        makefile_44386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 37), self_44385, 'makefile')
        # Processing the call keyword arguments (line 61)
        kwargs_44387 = {}
        # Getting the type of 'sysconfig' (line 61)
        sysconfig_44383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'sysconfig', False)
        # Obtaining the member 'parse_makefile' of a type (line 61)
        parse_makefile_44384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), sysconfig_44383, 'parse_makefile')
        # Calling parse_makefile(args, kwargs) (line 61)
        parse_makefile_call_result_44388 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), parse_makefile_44384, *[makefile_44386], **kwargs_44387)
        
        # Assigning a type to the variable 'd' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'd', parse_makefile_call_result_44388)
        
        # Call to assertEqual(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'd' (line 62)
        d_44391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'd', False)
        
        # Obtaining an instance of the builtin type 'dict' (line 62)
        dict_44392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 62)
        # Adding element type (key, value) (line 62)
        str_44393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 29), 'str', 'CONFIG_ARGS')
        str_44394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 44), 'str', "'--arg1=optarg1' 'ENV=LIB'")
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), dict_44392, (str_44393, str_44394))
        # Adding element type (key, value) (line 62)
        str_44395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 29), 'str', 'OTHER')
        str_44396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 38), 'str', 'foo')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), dict_44392, (str_44395, str_44396))
        
        # Processing the call keyword arguments (line 62)
        kwargs_44397 = {}
        # Getting the type of 'self' (line 62)
        self_44389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 62)
        assertEqual_44390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_44389, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 62)
        assertEqual_call_result_44398 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), assertEqual_44390, *[d_44391, dict_44392], **kwargs_44397)
        
        
        # ################# End of 'test_parse_makefile_base(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_parse_makefile_base' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_44399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44399)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_parse_makefile_base'
        return stypy_return_type_44399


    @norecursion
    def test_parse_makefile_literal_dollar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_parse_makefile_literal_dollar'
        module_type_store = module_type_store.open_function_context('test_parse_makefile_literal_dollar', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SysconfigTestCase.test_parse_makefile_literal_dollar.__dict__.__setitem__('stypy_localization', localization)
        SysconfigTestCase.test_parse_makefile_literal_dollar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SysconfigTestCase.test_parse_makefile_literal_dollar.__dict__.__setitem__('stypy_type_store', module_type_store)
        SysconfigTestCase.test_parse_makefile_literal_dollar.__dict__.__setitem__('stypy_function_name', 'SysconfigTestCase.test_parse_makefile_literal_dollar')
        SysconfigTestCase.test_parse_makefile_literal_dollar.__dict__.__setitem__('stypy_param_names_list', [])
        SysconfigTestCase.test_parse_makefile_literal_dollar.__dict__.__setitem__('stypy_varargs_param_name', None)
        SysconfigTestCase.test_parse_makefile_literal_dollar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SysconfigTestCase.test_parse_makefile_literal_dollar.__dict__.__setitem__('stypy_call_defaults', defaults)
        SysconfigTestCase.test_parse_makefile_literal_dollar.__dict__.__setitem__('stypy_call_varargs', varargs)
        SysconfigTestCase.test_parse_makefile_literal_dollar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SysconfigTestCase.test_parse_makefile_literal_dollar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SysconfigTestCase.test_parse_makefile_literal_dollar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_parse_makefile_literal_dollar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_parse_makefile_literal_dollar(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 66):
        
        # Assigning a Attribute to a Attribute (line 66):
        # Getting the type of 'test' (line 66)
        test_44400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'test')
        # Obtaining the member 'test_support' of a type (line 66)
        test_support_44401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 24), test_44400, 'test_support')
        # Obtaining the member 'TESTFN' of a type (line 66)
        TESTFN_44402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 24), test_support_44401, 'TESTFN')
        # Getting the type of 'self' (line 66)
        self_44403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member 'makefile' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_44403, 'makefile', TESTFN_44402)
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to open(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'self' (line 67)
        self_44405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'self', False)
        # Obtaining the member 'makefile' of a type (line 67)
        makefile_44406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 18), self_44405, 'makefile')
        str_44407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 33), 'str', 'w')
        # Processing the call keyword arguments (line 67)
        kwargs_44408 = {}
        # Getting the type of 'open' (line 67)
        open_44404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'open', False)
        # Calling open(args, kwargs) (line 67)
        open_call_result_44409 = invoke(stypy.reporting.localization.Localization(__file__, 67, 13), open_44404, *[makefile_44406, str_44407], **kwargs_44408)
        
        # Assigning a type to the variable 'fd' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'fd', open_call_result_44409)
        
        # Try-finally block (line 68)
        
        # Call to write(...): (line 69)
        # Processing the call arguments (line 69)
        str_44412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 21), 'str', "CONFIG_ARGS=  '--arg1=optarg1' 'ENV=\\$$LIB'\n")
        # Processing the call keyword arguments (line 69)
        kwargs_44413 = {}
        # Getting the type of 'fd' (line 69)
        fd_44410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'fd', False)
        # Obtaining the member 'write' of a type (line 69)
        write_44411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), fd_44410, 'write')
        # Calling write(args, kwargs) (line 69)
        write_call_result_44414 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), write_44411, *[str_44412], **kwargs_44413)
        
        
        # Call to write(...): (line 70)
        # Processing the call arguments (line 70)
        str_44417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 21), 'str', 'VAR=$OTHER\nOTHER=foo')
        # Processing the call keyword arguments (line 70)
        kwargs_44418 = {}
        # Getting the type of 'fd' (line 70)
        fd_44415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'fd', False)
        # Obtaining the member 'write' of a type (line 70)
        write_44416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), fd_44415, 'write')
        # Calling write(args, kwargs) (line 70)
        write_call_result_44419 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), write_44416, *[str_44417], **kwargs_44418)
        
        
        # finally branch of the try-finally block (line 68)
        
        # Call to close(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_44422 = {}
        # Getting the type of 'fd' (line 72)
        fd_44420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'fd', False)
        # Obtaining the member 'close' of a type (line 72)
        close_44421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), fd_44420, 'close')
        # Calling close(args, kwargs) (line 72)
        close_call_result_44423 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), close_44421, *[], **kwargs_44422)
        
        
        
        # Assigning a Call to a Name (line 73):
        
        # Assigning a Call to a Name (line 73):
        
        # Call to parse_makefile(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'self' (line 73)
        self_44426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 37), 'self', False)
        # Obtaining the member 'makefile' of a type (line 73)
        makefile_44427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 37), self_44426, 'makefile')
        # Processing the call keyword arguments (line 73)
        kwargs_44428 = {}
        # Getting the type of 'sysconfig' (line 73)
        sysconfig_44424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'sysconfig', False)
        # Obtaining the member 'parse_makefile' of a type (line 73)
        parse_makefile_44425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 12), sysconfig_44424, 'parse_makefile')
        # Calling parse_makefile(args, kwargs) (line 73)
        parse_makefile_call_result_44429 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), parse_makefile_44425, *[makefile_44427], **kwargs_44428)
        
        # Assigning a type to the variable 'd' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'd', parse_makefile_call_result_44429)
        
        # Call to assertEqual(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'd' (line 74)
        d_44432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'd', False)
        
        # Obtaining an instance of the builtin type 'dict' (line 74)
        dict_44433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 74)
        # Adding element type (key, value) (line 74)
        str_44434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 29), 'str', 'CONFIG_ARGS')
        str_44435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 44), 'str', "'--arg1=optarg1' 'ENV=\\$LIB'")
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 28), dict_44433, (str_44434, str_44435))
        # Adding element type (key, value) (line 74)
        str_44436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 29), 'str', 'OTHER')
        str_44437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 38), 'str', 'foo')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 28), dict_44433, (str_44436, str_44437))
        
        # Processing the call keyword arguments (line 74)
        kwargs_44438 = {}
        # Getting the type of 'self' (line 74)
        self_44430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 74)
        assertEqual_44431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_44430, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 74)
        assertEqual_call_result_44439 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), assertEqual_44431, *[d_44432, dict_44433], **kwargs_44438)
        
        
        # ################# End of 'test_parse_makefile_literal_dollar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_parse_makefile_literal_dollar' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_44440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44440)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_parse_makefile_literal_dollar'
        return stypy_return_type_44440


    @norecursion
    def test_sysconfig_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sysconfig_module'
        module_type_store = module_type_store.open_function_context('test_sysconfig_module', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SysconfigTestCase.test_sysconfig_module.__dict__.__setitem__('stypy_localization', localization)
        SysconfigTestCase.test_sysconfig_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SysconfigTestCase.test_sysconfig_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        SysconfigTestCase.test_sysconfig_module.__dict__.__setitem__('stypy_function_name', 'SysconfigTestCase.test_sysconfig_module')
        SysconfigTestCase.test_sysconfig_module.__dict__.__setitem__('stypy_param_names_list', [])
        SysconfigTestCase.test_sysconfig_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        SysconfigTestCase.test_sysconfig_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SysconfigTestCase.test_sysconfig_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        SysconfigTestCase.test_sysconfig_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        SysconfigTestCase.test_sysconfig_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SysconfigTestCase.test_sysconfig_module.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SysconfigTestCase.test_sysconfig_module', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sysconfig_module', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sysconfig_module(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 79, 8))
        
        # 'import sysconfig' statement (line 79)
        import sysconfig as global_sysconfig

        import_module(stypy.reporting.localization.Localization(__file__, 79, 8), 'global_sysconfig', global_sysconfig, module_type_store)
        
        
        # Call to assertEqual(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Call to get_config_var(...): (line 80)
        # Processing the call arguments (line 80)
        str_44445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 57), 'str', 'CFLAGS')
        # Processing the call keyword arguments (line 80)
        kwargs_44446 = {}
        # Getting the type of 'global_sysconfig' (line 80)
        global_sysconfig_44443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'global_sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 80)
        get_config_var_44444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 25), global_sysconfig_44443, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 80)
        get_config_var_call_result_44447 = invoke(stypy.reporting.localization.Localization(__file__, 80, 25), get_config_var_44444, *[str_44445], **kwargs_44446)
        
        
        # Call to get_config_var(...): (line 80)
        # Processing the call arguments (line 80)
        str_44450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 93), 'str', 'CFLAGS')
        # Processing the call keyword arguments (line 80)
        kwargs_44451 = {}
        # Getting the type of 'sysconfig' (line 80)
        sysconfig_44448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 68), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 80)
        get_config_var_44449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 68), sysconfig_44448, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 80)
        get_config_var_call_result_44452 = invoke(stypy.reporting.localization.Localization(__file__, 80, 68), get_config_var_44449, *[str_44450], **kwargs_44451)
        
        # Processing the call keyword arguments (line 80)
        kwargs_44453 = {}
        # Getting the type of 'self' (line 80)
        self_44441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 80)
        assertEqual_44442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_44441, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 80)
        assertEqual_call_result_44454 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assertEqual_44442, *[get_config_var_call_result_44447, get_config_var_call_result_44452], **kwargs_44453)
        
        
        # Call to assertEqual(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Call to get_config_var(...): (line 81)
        # Processing the call arguments (line 81)
        str_44459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 57), 'str', 'LDFLAGS')
        # Processing the call keyword arguments (line 81)
        kwargs_44460 = {}
        # Getting the type of 'global_sysconfig' (line 81)
        global_sysconfig_44457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 25), 'global_sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 81)
        get_config_var_44458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 25), global_sysconfig_44457, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 81)
        get_config_var_call_result_44461 = invoke(stypy.reporting.localization.Localization(__file__, 81, 25), get_config_var_44458, *[str_44459], **kwargs_44460)
        
        
        # Call to get_config_var(...): (line 81)
        # Processing the call arguments (line 81)
        str_44464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 94), 'str', 'LDFLAGS')
        # Processing the call keyword arguments (line 81)
        kwargs_44465 = {}
        # Getting the type of 'sysconfig' (line 81)
        sysconfig_44462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 69), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 81)
        get_config_var_44463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 69), sysconfig_44462, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 81)
        get_config_var_call_result_44466 = invoke(stypy.reporting.localization.Localization(__file__, 81, 69), get_config_var_44463, *[str_44464], **kwargs_44465)
        
        # Processing the call keyword arguments (line 81)
        kwargs_44467 = {}
        # Getting the type of 'self' (line 81)
        self_44455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 81)
        assertEqual_44456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_44455, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 81)
        assertEqual_call_result_44468 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), assertEqual_44456, *[get_config_var_call_result_44461, get_config_var_call_result_44466], **kwargs_44467)
        
        
        # ################# End of 'test_sysconfig_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sysconfig_module' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_44469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44469)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sysconfig_module'
        return stypy_return_type_44469


    @norecursion
    def test_sysconfig_compiler_vars(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sysconfig_compiler_vars'
        module_type_store = module_type_store.open_function_context('test_sysconfig_compiler_vars', 83, 4, False)
        # Assigning a type to the variable 'self' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SysconfigTestCase.test_sysconfig_compiler_vars.__dict__.__setitem__('stypy_localization', localization)
        SysconfigTestCase.test_sysconfig_compiler_vars.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SysconfigTestCase.test_sysconfig_compiler_vars.__dict__.__setitem__('stypy_type_store', module_type_store)
        SysconfigTestCase.test_sysconfig_compiler_vars.__dict__.__setitem__('stypy_function_name', 'SysconfigTestCase.test_sysconfig_compiler_vars')
        SysconfigTestCase.test_sysconfig_compiler_vars.__dict__.__setitem__('stypy_param_names_list', [])
        SysconfigTestCase.test_sysconfig_compiler_vars.__dict__.__setitem__('stypy_varargs_param_name', None)
        SysconfigTestCase.test_sysconfig_compiler_vars.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SysconfigTestCase.test_sysconfig_compiler_vars.__dict__.__setitem__('stypy_call_defaults', defaults)
        SysconfigTestCase.test_sysconfig_compiler_vars.__dict__.__setitem__('stypy_call_varargs', varargs)
        SysconfigTestCase.test_sysconfig_compiler_vars.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SysconfigTestCase.test_sysconfig_compiler_vars.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SysconfigTestCase.test_sysconfig_compiler_vars', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sysconfig_compiler_vars', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sysconfig_compiler_vars(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 99, 8))
        
        # 'import sysconfig' statement (line 99)
        import sysconfig as global_sysconfig

        import_module(stypy.reporting.localization.Localization(__file__, 99, 8), 'global_sysconfig', global_sysconfig, module_type_store)
        
        
        
        # Call to get_config_var(...): (line 100)
        # Processing the call arguments (line 100)
        str_44472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 36), 'str', 'CUSTOMIZED_OSX_COMPILER')
        # Processing the call keyword arguments (line 100)
        kwargs_44473 = {}
        # Getting the type of 'sysconfig' (line 100)
        sysconfig_44470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 100)
        get_config_var_44471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 11), sysconfig_44470, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 100)
        get_config_var_call_result_44474 = invoke(stypy.reporting.localization.Localization(__file__, 100, 11), get_config_var_44471, *[str_44472], **kwargs_44473)
        
        # Testing the type of an if condition (line 100)
        if_condition_44475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 8), get_config_var_call_result_44474)
        # Assigning a type to the variable 'if_condition_44475' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'if_condition_44475', if_condition_44475)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to skipTest(...): (line 101)
        # Processing the call arguments (line 101)
        str_44478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'str', 'compiler flags customized')
        # Processing the call keyword arguments (line 101)
        kwargs_44479 = {}
        # Getting the type of 'self' (line 101)
        self_44476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'self', False)
        # Obtaining the member 'skipTest' of a type (line 101)
        skipTest_44477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), self_44476, 'skipTest')
        # Calling skipTest(args, kwargs) (line 101)
        skipTest_call_result_44480 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), skipTest_44477, *[str_44478], **kwargs_44479)
        
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertEqual(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to get_config_var(...): (line 102)
        # Processing the call arguments (line 102)
        str_44485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 57), 'str', 'LDSHARED')
        # Processing the call keyword arguments (line 102)
        kwargs_44486 = {}
        # Getting the type of 'global_sysconfig' (line 102)
        global_sysconfig_44483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'global_sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 102)
        get_config_var_44484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 25), global_sysconfig_44483, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 102)
        get_config_var_call_result_44487 = invoke(stypy.reporting.localization.Localization(__file__, 102, 25), get_config_var_44484, *[str_44485], **kwargs_44486)
        
        
        # Call to get_config_var(...): (line 102)
        # Processing the call arguments (line 102)
        str_44490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 95), 'str', 'LDSHARED')
        # Processing the call keyword arguments (line 102)
        kwargs_44491 = {}
        # Getting the type of 'sysconfig' (line 102)
        sysconfig_44488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 70), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 102)
        get_config_var_44489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 70), sysconfig_44488, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 102)
        get_config_var_call_result_44492 = invoke(stypy.reporting.localization.Localization(__file__, 102, 70), get_config_var_44489, *[str_44490], **kwargs_44491)
        
        # Processing the call keyword arguments (line 102)
        kwargs_44493 = {}
        # Getting the type of 'self' (line 102)
        self_44481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 102)
        assertEqual_44482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), self_44481, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 102)
        assertEqual_call_result_44494 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), assertEqual_44482, *[get_config_var_call_result_44487, get_config_var_call_result_44492], **kwargs_44493)
        
        
        # Call to assertEqual(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to get_config_var(...): (line 103)
        # Processing the call arguments (line 103)
        str_44499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 57), 'str', 'CC')
        # Processing the call keyword arguments (line 103)
        kwargs_44500 = {}
        # Getting the type of 'global_sysconfig' (line 103)
        global_sysconfig_44497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 25), 'global_sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 103)
        get_config_var_44498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 25), global_sysconfig_44497, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 103)
        get_config_var_call_result_44501 = invoke(stypy.reporting.localization.Localization(__file__, 103, 25), get_config_var_44498, *[str_44499], **kwargs_44500)
        
        
        # Call to get_config_var(...): (line 103)
        # Processing the call arguments (line 103)
        str_44504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 89), 'str', 'CC')
        # Processing the call keyword arguments (line 103)
        kwargs_44505 = {}
        # Getting the type of 'sysconfig' (line 103)
        sysconfig_44502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 64), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 103)
        get_config_var_44503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 64), sysconfig_44502, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 103)
        get_config_var_call_result_44506 = invoke(stypy.reporting.localization.Localization(__file__, 103, 64), get_config_var_44503, *[str_44504], **kwargs_44505)
        
        # Processing the call keyword arguments (line 103)
        kwargs_44507 = {}
        # Getting the type of 'self' (line 103)
        self_44495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 103)
        assertEqual_44496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_44495, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 103)
        assertEqual_call_result_44508 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), assertEqual_44496, *[get_config_var_call_result_44501, get_config_var_call_result_44506], **kwargs_44507)
        
        
        # ################# End of 'test_sysconfig_compiler_vars(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sysconfig_compiler_vars' in the type store
        # Getting the type of 'stypy_return_type' (line 83)
        stypy_return_type_44509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44509)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sysconfig_compiler_vars'
        return stypy_return_type_44509


    @norecursion
    def test_customize_compiler_before_get_config_vars(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_customize_compiler_before_get_config_vars'
        module_type_store = module_type_store.open_function_context('test_customize_compiler_before_get_config_vars', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SysconfigTestCase.test_customize_compiler_before_get_config_vars.__dict__.__setitem__('stypy_localization', localization)
        SysconfigTestCase.test_customize_compiler_before_get_config_vars.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SysconfigTestCase.test_customize_compiler_before_get_config_vars.__dict__.__setitem__('stypy_type_store', module_type_store)
        SysconfigTestCase.test_customize_compiler_before_get_config_vars.__dict__.__setitem__('stypy_function_name', 'SysconfigTestCase.test_customize_compiler_before_get_config_vars')
        SysconfigTestCase.test_customize_compiler_before_get_config_vars.__dict__.__setitem__('stypy_param_names_list', [])
        SysconfigTestCase.test_customize_compiler_before_get_config_vars.__dict__.__setitem__('stypy_varargs_param_name', None)
        SysconfigTestCase.test_customize_compiler_before_get_config_vars.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SysconfigTestCase.test_customize_compiler_before_get_config_vars.__dict__.__setitem__('stypy_call_defaults', defaults)
        SysconfigTestCase.test_customize_compiler_before_get_config_vars.__dict__.__setitem__('stypy_call_varargs', varargs)
        SysconfigTestCase.test_customize_compiler_before_get_config_vars.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SysconfigTestCase.test_customize_compiler_before_get_config_vars.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SysconfigTestCase.test_customize_compiler_before_get_config_vars', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_customize_compiler_before_get_config_vars', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_customize_compiler_before_get_config_vars(...)' code ##################

        
        # Call to open(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'TESTFN' (line 109)
        TESTFN_44511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 18), 'TESTFN', False)
        str_44512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 26), 'str', 'w')
        # Processing the call keyword arguments (line 109)
        kwargs_44513 = {}
        # Getting the type of 'open' (line 109)
        open_44510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'open', False)
        # Calling open(args, kwargs) (line 109)
        open_call_result_44514 = invoke(stypy.reporting.localization.Localization(__file__, 109, 13), open_44510, *[TESTFN_44511, str_44512], **kwargs_44513)
        
        with_44515 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 109, 13), open_call_result_44514, 'with parameter', '__enter__', '__exit__')

        if with_44515:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 109)
            enter___44516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 13), open_call_result_44514, '__enter__')
            with_enter_44517 = invoke(stypy.reporting.localization.Localization(__file__, 109, 13), enter___44516)
            # Assigning a type to the variable 'f' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'f', with_enter_44517)
            
            # Call to writelines(...): (line 110)
            # Processing the call arguments (line 110)
            
            # Call to dedent(...): (line 110)
            # Processing the call arguments (line 110)
            str_44522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, (-1)), 'str', "                from distutils.core import Distribution\n                config = Distribution().get_command_obj('config')\n                # try_compile may pass or it may fail if no compiler\n                # is found but it should not raise an exception.\n                rc = config.try_compile('int x;')\n                ")
            # Processing the call keyword arguments (line 110)
            kwargs_44523 = {}
            # Getting the type of 'textwrap' (line 110)
            textwrap_44520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'textwrap', False)
            # Obtaining the member 'dedent' of a type (line 110)
            dedent_44521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 25), textwrap_44520, 'dedent')
            # Calling dedent(args, kwargs) (line 110)
            dedent_call_result_44524 = invoke(stypy.reporting.localization.Localization(__file__, 110, 25), dedent_44521, *[str_44522], **kwargs_44523)
            
            # Processing the call keyword arguments (line 110)
            kwargs_44525 = {}
            # Getting the type of 'f' (line 110)
            f_44518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'f', False)
            # Obtaining the member 'writelines' of a type (line 110)
            writelines_44519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), f_44518, 'writelines')
            # Calling writelines(args, kwargs) (line 110)
            writelines_call_result_44526 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), writelines_44519, *[dedent_call_result_44524], **kwargs_44525)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 109)
            exit___44527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 13), open_call_result_44514, '__exit__')
            with_exit_44528 = invoke(stypy.reporting.localization.Localization(__file__, 109, 13), exit___44527, None, None, None)

        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to Popen(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_44531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        
        # Call to str(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'sys' (line 117)
        sys_44533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 34), 'sys', False)
        # Obtaining the member 'executable' of a type (line 117)
        executable_44534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 34), sys_44533, 'executable')
        # Processing the call keyword arguments (line 117)
        kwargs_44535 = {}
        # Getting the type of 'str' (line 117)
        str_44532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 30), 'str', False)
        # Calling str(args, kwargs) (line 117)
        str_call_result_44536 = invoke(stypy.reporting.localization.Localization(__file__, 117, 30), str_44532, *[executable_44534], **kwargs_44535)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 29), list_44531, str_call_result_44536)
        # Adding element type (line 117)
        # Getting the type of 'TESTFN' (line 117)
        TESTFN_44537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 51), 'TESTFN', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 29), list_44531, TESTFN_44537)
        
        # Processing the call keyword arguments (line 117)
        # Getting the type of 'subprocess' (line 118)
        subprocess_44538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 23), 'subprocess', False)
        # Obtaining the member 'PIPE' of a type (line 118)
        PIPE_44539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 23), subprocess_44538, 'PIPE')
        keyword_44540 = PIPE_44539
        # Getting the type of 'subprocess' (line 119)
        subprocess_44541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'subprocess', False)
        # Obtaining the member 'STDOUT' of a type (line 119)
        STDOUT_44542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 23), subprocess_44541, 'STDOUT')
        keyword_44543 = STDOUT_44542
        # Getting the type of 'True' (line 120)
        True_44544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 35), 'True', False)
        keyword_44545 = True_44544
        kwargs_44546 = {'universal_newlines': keyword_44545, 'stderr': keyword_44543, 'stdout': keyword_44540}
        # Getting the type of 'subprocess' (line 117)
        subprocess_44529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'subprocess', False)
        # Obtaining the member 'Popen' of a type (line 117)
        Popen_44530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), subprocess_44529, 'Popen')
        # Calling Popen(args, kwargs) (line 117)
        Popen_call_result_44547 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), Popen_44530, *[list_44531], **kwargs_44546)
        
        # Assigning a type to the variable 'p' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'p', Popen_call_result_44547)
        
        # Assigning a Call to a Tuple (line 121):
        
        # Assigning a Subscript to a Name (line 121):
        
        # Obtaining the type of the subscript
        int_44548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 8), 'int')
        
        # Call to communicate(...): (line 121)
        # Processing the call keyword arguments (line 121)
        kwargs_44551 = {}
        # Getting the type of 'p' (line 121)
        p_44549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'p', False)
        # Obtaining the member 'communicate' of a type (line 121)
        communicate_44550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 21), p_44549, 'communicate')
        # Calling communicate(args, kwargs) (line 121)
        communicate_call_result_44552 = invoke(stypy.reporting.localization.Localization(__file__, 121, 21), communicate_44550, *[], **kwargs_44551)
        
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___44553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), communicate_call_result_44552, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_44554 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), getitem___44553, int_44548)
        
        # Assigning a type to the variable 'tuple_var_assignment_44212' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'tuple_var_assignment_44212', subscript_call_result_44554)
        
        # Assigning a Subscript to a Name (line 121):
        
        # Obtaining the type of the subscript
        int_44555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 8), 'int')
        
        # Call to communicate(...): (line 121)
        # Processing the call keyword arguments (line 121)
        kwargs_44558 = {}
        # Getting the type of 'p' (line 121)
        p_44556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'p', False)
        # Obtaining the member 'communicate' of a type (line 121)
        communicate_44557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 21), p_44556, 'communicate')
        # Calling communicate(args, kwargs) (line 121)
        communicate_call_result_44559 = invoke(stypy.reporting.localization.Localization(__file__, 121, 21), communicate_44557, *[], **kwargs_44558)
        
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___44560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), communicate_call_result_44559, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_44561 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), getitem___44560, int_44555)
        
        # Assigning a type to the variable 'tuple_var_assignment_44213' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'tuple_var_assignment_44213', subscript_call_result_44561)
        
        # Assigning a Name to a Name (line 121):
        # Getting the type of 'tuple_var_assignment_44212' (line 121)
        tuple_var_assignment_44212_44562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'tuple_var_assignment_44212')
        # Assigning a type to the variable 'outs' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'outs', tuple_var_assignment_44212_44562)
        
        # Assigning a Name to a Name (line 121):
        # Getting the type of 'tuple_var_assignment_44213' (line 121)
        tuple_var_assignment_44213_44563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'tuple_var_assignment_44213')
        # Assigning a type to the variable 'errs' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 14), 'errs', tuple_var_assignment_44213_44563)
        
        # Call to assertEqual(...): (line 122)
        # Processing the call arguments (line 122)
        int_44566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 25), 'int')
        # Getting the type of 'p' (line 122)
        p_44567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'p', False)
        # Obtaining the member 'returncode' of a type (line 122)
        returncode_44568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 28), p_44567, 'returncode')
        str_44569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 42), 'str', 'Subprocess failed: ')
        # Getting the type of 'outs' (line 122)
        outs_44570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 66), 'outs', False)
        # Applying the binary operator '+' (line 122)
        result_add_44571 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 42), '+', str_44569, outs_44570)
        
        # Processing the call keyword arguments (line 122)
        kwargs_44572 = {}
        # Getting the type of 'self' (line 122)
        self_44564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 122)
        assertEqual_44565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_44564, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 122)
        assertEqual_call_result_44573 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), assertEqual_44565, *[int_44566, returncode_44568, result_add_44571], **kwargs_44572)
        
        
        # ################# End of 'test_customize_compiler_before_get_config_vars(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_customize_compiler_before_get_config_vars' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_44574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44574)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_customize_compiler_before_get_config_vars'
        return stypy_return_type_44574


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SysconfigTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SysconfigTestCase' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'SysconfigTestCase', SysconfigTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 125, 0, False)
    
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

    
    # Assigning a Call to a Name (line 126):
    
    # Assigning a Call to a Name (line 126):
    
    # Call to TestSuite(...): (line 126)
    # Processing the call keyword arguments (line 126)
    kwargs_44577 = {}
    # Getting the type of 'unittest' (line 126)
    unittest_44575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'unittest', False)
    # Obtaining the member 'TestSuite' of a type (line 126)
    TestSuite_44576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), unittest_44575, 'TestSuite')
    # Calling TestSuite(args, kwargs) (line 126)
    TestSuite_call_result_44578 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), TestSuite_44576, *[], **kwargs_44577)
    
    # Assigning a type to the variable 'suite' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'suite', TestSuite_call_result_44578)
    
    # Call to addTest(...): (line 127)
    # Processing the call arguments (line 127)
    
    # Call to makeSuite(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'SysconfigTestCase' (line 127)
    SysconfigTestCase_44583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 37), 'SysconfigTestCase', False)
    # Processing the call keyword arguments (line 127)
    kwargs_44584 = {}
    # Getting the type of 'unittest' (line 127)
    unittest_44581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 18), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 127)
    makeSuite_44582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 18), unittest_44581, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 127)
    makeSuite_call_result_44585 = invoke(stypy.reporting.localization.Localization(__file__, 127, 18), makeSuite_44582, *[SysconfigTestCase_44583], **kwargs_44584)
    
    # Processing the call keyword arguments (line 127)
    kwargs_44586 = {}
    # Getting the type of 'suite' (line 127)
    suite_44579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'suite', False)
    # Obtaining the member 'addTest' of a type (line 127)
    addTest_44580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 4), suite_44579, 'addTest')
    # Calling addTest(args, kwargs) (line 127)
    addTest_call_result_44587 = invoke(stypy.reporting.localization.Localization(__file__, 127, 4), addTest_44580, *[makeSuite_call_result_44585], **kwargs_44586)
    
    # Getting the type of 'suite' (line 128)
    suite_44588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'suite')
    # Assigning a type to the variable 'stypy_return_type' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type', suite_44588)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 125)
    stypy_return_type_44589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44589)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_44589

# Assigning a type to the variable 'test_suite' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 132)
    # Processing the call arguments (line 132)
    
    # Call to test_suite(...): (line 132)
    # Processing the call keyword arguments (line 132)
    kwargs_44594 = {}
    # Getting the type of 'test_suite' (line 132)
    test_suite_44593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 35), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 132)
    test_suite_call_result_44595 = invoke(stypy.reporting.localization.Localization(__file__, 132, 35), test_suite_44593, *[], **kwargs_44594)
    
    # Processing the call keyword arguments (line 132)
    kwargs_44596 = {}
    # Getting the type of 'test' (line 132)
    test_44590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'test', False)
    # Obtaining the member 'test_support' of a type (line 132)
    test_support_44591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 4), test_44590, 'test_support')
    # Obtaining the member 'run_unittest' of a type (line 132)
    run_unittest_44592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 4), test_support_44591, 'run_unittest')
    # Calling run_unittest(args, kwargs) (line 132)
    run_unittest_call_result_44597 = invoke(stypy.reporting.localization.Localization(__file__, 132, 4), run_unittest_44592, *[test_suite_call_result_44595], **kwargs_44596)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
