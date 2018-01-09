
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.unixccompiler.'''
2: import os
3: import sys
4: import unittest
5: from test.test_support import EnvironmentVarGuard, run_unittest
6: 
7: from distutils import sysconfig
8: from distutils.unixccompiler import UnixCCompiler
9: 
10: class UnixCCompilerTestCase(unittest.TestCase):
11: 
12:     def setUp(self):
13:         self._backup_platform = sys.platform
14:         self._backup_get_config_var = sysconfig.get_config_var
15:         class CompilerWrapper(UnixCCompiler):
16:             def rpath_foo(self):
17:                 return self.runtime_library_dir_option('/foo')
18:         self.cc = CompilerWrapper()
19: 
20:     def tearDown(self):
21:         sys.platform = self._backup_platform
22:         sysconfig.get_config_var = self._backup_get_config_var
23: 
24:     @unittest.skipIf(sys.platform == 'win32', "can't test on Windows")
25:     def test_runtime_libdir_option(self):
26:         # Issue#5900
27:         #
28:         # Ensure RUNPATH is added to extension modules with RPATH if
29:         # GNU ld is used
30: 
31:         # darwin
32:         sys.platform = 'darwin'
33:         self.assertEqual(self.cc.rpath_foo(), '-L/foo')
34: 
35:         # hp-ux
36:         sys.platform = 'hp-ux'
37:         old_gcv = sysconfig.get_config_var
38:         def gcv(v):
39:             return 'xxx'
40:         sysconfig.get_config_var = gcv
41:         self.assertEqual(self.cc.rpath_foo(), ['+s', '-L/foo'])
42: 
43:         def gcv(v):
44:             return 'gcc'
45:         sysconfig.get_config_var = gcv
46:         self.assertEqual(self.cc.rpath_foo(), ['-Wl,+s', '-L/foo'])
47: 
48:         def gcv(v):
49:             return 'g++'
50:         sysconfig.get_config_var = gcv
51:         self.assertEqual(self.cc.rpath_foo(), ['-Wl,+s', '-L/foo'])
52: 
53:         sysconfig.get_config_var = old_gcv
54: 
55:         # irix646
56:         sys.platform = 'irix646'
57:         self.assertEqual(self.cc.rpath_foo(), ['-rpath', '/foo'])
58: 
59:         # osf1V5
60:         sys.platform = 'osf1V5'
61:         self.assertEqual(self.cc.rpath_foo(), ['-rpath', '/foo'])
62: 
63:         # GCC GNULD
64:         sys.platform = 'bar'
65:         def gcv(v):
66:             if v == 'CC':
67:                 return 'gcc'
68:             elif v == 'GNULD':
69:                 return 'yes'
70:         sysconfig.get_config_var = gcv
71:         self.assertEqual(self.cc.rpath_foo(), '-Wl,-R/foo')
72: 
73:         # GCC non-GNULD
74:         sys.platform = 'bar'
75:         def gcv(v):
76:             if v == 'CC':
77:                 return 'gcc'
78:             elif v == 'GNULD':
79:                 return 'no'
80:         sysconfig.get_config_var = gcv
81:         self.assertEqual(self.cc.rpath_foo(), '-Wl,-R/foo')
82: 
83:         # GCC GNULD with fully qualified configuration prefix
84:         # see #7617
85:         sys.platform = 'bar'
86:         def gcv(v):
87:             if v == 'CC':
88:                 return 'x86_64-pc-linux-gnu-gcc-4.4.2'
89:             elif v == 'GNULD':
90:                 return 'yes'
91:         sysconfig.get_config_var = gcv
92:         self.assertEqual(self.cc.rpath_foo(), '-Wl,-R/foo')
93: 
94: 
95:         # non-GCC GNULD
96:         sys.platform = 'bar'
97:         def gcv(v):
98:             if v == 'CC':
99:                 return 'cc'
100:             elif v == 'GNULD':
101:                 return 'yes'
102:         sysconfig.get_config_var = gcv
103:         self.assertEqual(self.cc.rpath_foo(), '-R/foo')
104: 
105:         # non-GCC non-GNULD
106:         sys.platform = 'bar'
107:         def gcv(v):
108:             if v == 'CC':
109:                 return 'cc'
110:             elif v == 'GNULD':
111:                 return 'no'
112:         sysconfig.get_config_var = gcv
113:         self.assertEqual(self.cc.rpath_foo(), '-R/foo')
114: 
115:         # AIX C/C++ linker
116:         sys.platform = 'aix'
117:         def gcv(v):
118:             return 'xxx'
119:         sysconfig.get_config_var = gcv
120:         self.assertEqual(self.cc.rpath_foo(), '-R/foo')
121: 
122:     @unittest.skipUnless(sys.platform == 'darwin', 'test only relevant for OS X')
123:     def test_osx_cc_overrides_ldshared(self):
124:         # Issue #18080:
125:         # ensure that setting CC env variable also changes default linker
126:         def gcv(v):
127:             if v == 'LDSHARED':
128:                 return 'gcc-4.2 -bundle -undefined dynamic_lookup '
129:             return 'gcc-4.2'
130:         sysconfig.get_config_var = gcv
131:         with EnvironmentVarGuard() as env:
132:             env['CC'] = 'my_cc'
133:             del env['LDSHARED']
134:             sysconfig.customize_compiler(self.cc)
135:         self.assertEqual(self.cc.linker_so[0], 'my_cc')
136: 
137:     @unittest.skipUnless(sys.platform == 'darwin', 'test only relevant for OS X')
138:     def test_osx_explicit_ldshared(self):
139:         # Issue #18080:
140:         # ensure that setting CC env variable does not change
141:         #   explicit LDSHARED setting for linker
142:         def gcv(v):
143:             if v == 'LDSHARED':
144:                 return 'gcc-4.2 -bundle -undefined dynamic_lookup '
145:             return 'gcc-4.2'
146:         sysconfig.get_config_var = gcv
147:         with EnvironmentVarGuard() as env:
148:             env['CC'] = 'my_cc'
149:             env['LDSHARED'] = 'my_ld -bundle -dynamic'
150:             sysconfig.customize_compiler(self.cc)
151:         self.assertEqual(self.cc.linker_so[0], 'my_ld')
152: 
153: 
154: def test_suite():
155:     return unittest.makeSuite(UnixCCompilerTestCase)
156: 
157: if __name__ == "__main__":
158:     run_unittest(test_suite())
159: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_44813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.unixccompiler.')
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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from test.test_support import EnvironmentVarGuard, run_unittest' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_44814 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support')

if (type(import_44814) is not StypyTypeError):

    if (import_44814 != 'pyd_module'):
        __import__(import_44814)
        sys_modules_44815 = sys.modules[import_44814]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', sys_modules_44815.module_type_store, module_type_store, ['EnvironmentVarGuard', 'run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_44815, sys_modules_44815.module_type_store, module_type_store)
    else:
        from test.test_support import EnvironmentVarGuard, run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', None, module_type_store, ['EnvironmentVarGuard', 'run_unittest'], [EnvironmentVarGuard, run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', import_44814)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils import sysconfig' statement (line 7)
try:
    from distutils import sysconfig

except:
    sysconfig = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils', None, module_type_store, ['sysconfig'], [sysconfig])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.unixccompiler import UnixCCompiler' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_44816 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.unixccompiler')

if (type(import_44816) is not StypyTypeError):

    if (import_44816 != 'pyd_module'):
        __import__(import_44816)
        sys_modules_44817 = sys.modules[import_44816]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.unixccompiler', sys_modules_44817.module_type_store, module_type_store, ['UnixCCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_44817, sys_modules_44817.module_type_store, module_type_store)
    else:
        from distutils.unixccompiler import UnixCCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.unixccompiler', None, module_type_store, ['UnixCCompiler'], [UnixCCompiler])

else:
    # Assigning a type to the variable 'distutils.unixccompiler' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.unixccompiler', import_44816)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'UnixCCompilerTestCase' class
# Getting the type of 'unittest' (line 10)
unittest_44818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 28), 'unittest')
# Obtaining the member 'TestCase' of a type (line 10)
TestCase_44819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 28), unittest_44818, 'TestCase')

class UnixCCompilerTestCase(TestCase_44819, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnixCCompilerTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompilerTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompilerTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompilerTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'UnixCCompilerTestCase.setUp')
        UnixCCompilerTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        UnixCCompilerTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompilerTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompilerTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompilerTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompilerTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompilerTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompilerTestCase.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Attribute (line 13):
        # Getting the type of 'sys' (line 13)
        sys_44820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 32), 'sys')
        # Obtaining the member 'platform' of a type (line 13)
        platform_44821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 32), sys_44820, 'platform')
        # Getting the type of 'self' (line 13)
        self_44822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self')
        # Setting the type of the member '_backup_platform' of a type (line 13)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), self_44822, '_backup_platform', platform_44821)
        
        # Assigning a Attribute to a Attribute (line 14):
        # Getting the type of 'sysconfig' (line 14)
        sysconfig_44823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 38), 'sysconfig')
        # Obtaining the member 'get_config_var' of a type (line 14)
        get_config_var_44824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 38), sysconfig_44823, 'get_config_var')
        # Getting the type of 'self' (line 14)
        self_44825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self')
        # Setting the type of the member '_backup_get_config_var' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_44825, '_backup_get_config_var', get_config_var_44824)
        # Declaration of the 'CompilerWrapper' class
        # Getting the type of 'UnixCCompiler' (line 15)
        UnixCCompiler_44826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 30), 'UnixCCompiler')

        class CompilerWrapper(UnixCCompiler_44826, ):

            @norecursion
            def rpath_foo(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'rpath_foo'
                module_type_store = module_type_store.open_function_context('rpath_foo', 16, 12, False)
                # Assigning a type to the variable 'self' (line 17)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                CompilerWrapper.rpath_foo.__dict__.__setitem__('stypy_localization', localization)
                CompilerWrapper.rpath_foo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                CompilerWrapper.rpath_foo.__dict__.__setitem__('stypy_type_store', module_type_store)
                CompilerWrapper.rpath_foo.__dict__.__setitem__('stypy_function_name', 'CompilerWrapper.rpath_foo')
                CompilerWrapper.rpath_foo.__dict__.__setitem__('stypy_param_names_list', [])
                CompilerWrapper.rpath_foo.__dict__.__setitem__('stypy_varargs_param_name', None)
                CompilerWrapper.rpath_foo.__dict__.__setitem__('stypy_kwargs_param_name', None)
                CompilerWrapper.rpath_foo.__dict__.__setitem__('stypy_call_defaults', defaults)
                CompilerWrapper.rpath_foo.__dict__.__setitem__('stypy_call_varargs', varargs)
                CompilerWrapper.rpath_foo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                CompilerWrapper.rpath_foo.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompilerWrapper.rpath_foo', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'rpath_foo', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'rpath_foo(...)' code ##################

                
                # Call to runtime_library_dir_option(...): (line 17)
                # Processing the call arguments (line 17)
                str_44829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 55), 'str', '/foo')
                # Processing the call keyword arguments (line 17)
                kwargs_44830 = {}
                # Getting the type of 'self' (line 17)
                self_44827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 'self', False)
                # Obtaining the member 'runtime_library_dir_option' of a type (line 17)
                runtime_library_dir_option_44828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 23), self_44827, 'runtime_library_dir_option')
                # Calling runtime_library_dir_option(args, kwargs) (line 17)
                runtime_library_dir_option_call_result_44831 = invoke(stypy.reporting.localization.Localization(__file__, 17, 23), runtime_library_dir_option_44828, *[str_44829], **kwargs_44830)
                
                # Assigning a type to the variable 'stypy_return_type' (line 17)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'stypy_return_type', runtime_library_dir_option_call_result_44831)
                
                # ################# End of 'rpath_foo(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'rpath_foo' in the type store
                # Getting the type of 'stypy_return_type' (line 16)
                stypy_return_type_44832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_44832)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'rpath_foo'
                return stypy_return_type_44832

        
        # Assigning a type to the variable 'CompilerWrapper' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'CompilerWrapper', CompilerWrapper)
        
        # Assigning a Call to a Attribute (line 18):
        
        # Call to CompilerWrapper(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_44834 = {}
        # Getting the type of 'CompilerWrapper' (line 18)
        CompilerWrapper_44833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 18), 'CompilerWrapper', False)
        # Calling CompilerWrapper(args, kwargs) (line 18)
        CompilerWrapper_call_result_44835 = invoke(stypy.reporting.localization.Localization(__file__, 18, 18), CompilerWrapper_44833, *[], **kwargs_44834)
        
        # Getting the type of 'self' (line 18)
        self_44836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self')
        # Setting the type of the member 'cc' of a type (line 18)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), self_44836, 'cc', CompilerWrapper_call_result_44835)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_44837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44837)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_44837


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
        UnixCCompilerTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompilerTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompilerTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompilerTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'UnixCCompilerTestCase.tearDown')
        UnixCCompilerTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        UnixCCompilerTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompilerTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompilerTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompilerTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompilerTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompilerTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompilerTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Attribute (line 21):
        # Getting the type of 'self' (line 21)
        self_44838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), 'self')
        # Obtaining the member '_backup_platform' of a type (line 21)
        _backup_platform_44839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 23), self_44838, '_backup_platform')
        # Getting the type of 'sys' (line 21)
        sys_44840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'sys')
        # Setting the type of the member 'platform' of a type (line 21)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), sys_44840, 'platform', _backup_platform_44839)
        
        # Assigning a Attribute to a Attribute (line 22):
        # Getting the type of 'self' (line 22)
        self_44841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 35), 'self')
        # Obtaining the member '_backup_get_config_var' of a type (line 22)
        _backup_get_config_var_44842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 35), self_44841, '_backup_get_config_var')
        # Getting the type of 'sysconfig' (line 22)
        sysconfig_44843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'sysconfig')
        # Setting the type of the member 'get_config_var' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), sysconfig_44843, 'get_config_var', _backup_get_config_var_44842)
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_44844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44844)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_44844


    @norecursion
    def test_runtime_libdir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_runtime_libdir_option'
        module_type_store = module_type_store.open_function_context('test_runtime_libdir_option', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnixCCompilerTestCase.test_runtime_libdir_option.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompilerTestCase.test_runtime_libdir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompilerTestCase.test_runtime_libdir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompilerTestCase.test_runtime_libdir_option.__dict__.__setitem__('stypy_function_name', 'UnixCCompilerTestCase.test_runtime_libdir_option')
        UnixCCompilerTestCase.test_runtime_libdir_option.__dict__.__setitem__('stypy_param_names_list', [])
        UnixCCompilerTestCase.test_runtime_libdir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompilerTestCase.test_runtime_libdir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompilerTestCase.test_runtime_libdir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompilerTestCase.test_runtime_libdir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompilerTestCase.test_runtime_libdir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompilerTestCase.test_runtime_libdir_option.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompilerTestCase.test_runtime_libdir_option', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_runtime_libdir_option', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_runtime_libdir_option(...)' code ##################

        
        # Assigning a Str to a Attribute (line 32):
        str_44845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'str', 'darwin')
        # Getting the type of 'sys' (line 32)
        sys_44846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'sys')
        # Setting the type of the member 'platform' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), sys_44846, 'platform', str_44845)
        
        # Call to assertEqual(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Call to rpath_foo(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_44852 = {}
        # Getting the type of 'self' (line 33)
        self_44849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 33)
        cc_44850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 25), self_44849, 'cc')
        # Obtaining the member 'rpath_foo' of a type (line 33)
        rpath_foo_44851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 25), cc_44850, 'rpath_foo')
        # Calling rpath_foo(args, kwargs) (line 33)
        rpath_foo_call_result_44853 = invoke(stypy.reporting.localization.Localization(__file__, 33, 25), rpath_foo_44851, *[], **kwargs_44852)
        
        str_44854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 46), 'str', '-L/foo')
        # Processing the call keyword arguments (line 33)
        kwargs_44855 = {}
        # Getting the type of 'self' (line 33)
        self_44847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 33)
        assertEqual_44848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_44847, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 33)
        assertEqual_call_result_44856 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), assertEqual_44848, *[rpath_foo_call_result_44853, str_44854], **kwargs_44855)
        
        
        # Assigning a Str to a Attribute (line 36):
        str_44857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'str', 'hp-ux')
        # Getting the type of 'sys' (line 36)
        sys_44858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'sys')
        # Setting the type of the member 'platform' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), sys_44858, 'platform', str_44857)
        
        # Assigning a Attribute to a Name (line 37):
        # Getting the type of 'sysconfig' (line 37)
        sysconfig_44859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'sysconfig')
        # Obtaining the member 'get_config_var' of a type (line 37)
        get_config_var_44860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 18), sysconfig_44859, 'get_config_var')
        # Assigning a type to the variable 'old_gcv' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'old_gcv', get_config_var_44860)

        @norecursion
        def gcv(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'gcv'
            module_type_store = module_type_store.open_function_context('gcv', 38, 8, False)
            
            # Passed parameters checking function
            gcv.stypy_localization = localization
            gcv.stypy_type_of_self = None
            gcv.stypy_type_store = module_type_store
            gcv.stypy_function_name = 'gcv'
            gcv.stypy_param_names_list = ['v']
            gcv.stypy_varargs_param_name = None
            gcv.stypy_kwargs_param_name = None
            gcv.stypy_call_defaults = defaults
            gcv.stypy_call_varargs = varargs
            gcv.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'gcv', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'gcv', localization, ['v'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'gcv(...)' code ##################

            str_44861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'str', 'xxx')
            # Assigning a type to the variable 'stypy_return_type' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'stypy_return_type', str_44861)
            
            # ################# End of 'gcv(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'gcv' in the type store
            # Getting the type of 'stypy_return_type' (line 38)
            stypy_return_type_44862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_44862)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'gcv'
            return stypy_return_type_44862

        # Assigning a type to the variable 'gcv' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'gcv', gcv)
        
        # Assigning a Name to a Attribute (line 40):
        # Getting the type of 'gcv' (line 40)
        gcv_44863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'gcv')
        # Getting the type of 'sysconfig' (line 40)
        sysconfig_44864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'sysconfig')
        # Setting the type of the member 'get_config_var' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), sysconfig_44864, 'get_config_var', gcv_44863)
        
        # Call to assertEqual(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Call to rpath_foo(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_44870 = {}
        # Getting the type of 'self' (line 41)
        self_44867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 41)
        cc_44868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 25), self_44867, 'cc')
        # Obtaining the member 'rpath_foo' of a type (line 41)
        rpath_foo_44869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 25), cc_44868, 'rpath_foo')
        # Calling rpath_foo(args, kwargs) (line 41)
        rpath_foo_call_result_44871 = invoke(stypy.reporting.localization.Localization(__file__, 41, 25), rpath_foo_44869, *[], **kwargs_44870)
        
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_44872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        str_44873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 47), 'str', '+s')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 46), list_44872, str_44873)
        # Adding element type (line 41)
        str_44874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 53), 'str', '-L/foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 46), list_44872, str_44874)
        
        # Processing the call keyword arguments (line 41)
        kwargs_44875 = {}
        # Getting the type of 'self' (line 41)
        self_44865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 41)
        assertEqual_44866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_44865, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 41)
        assertEqual_call_result_44876 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), assertEqual_44866, *[rpath_foo_call_result_44871, list_44872], **kwargs_44875)
        

        @norecursion
        def gcv(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'gcv'
            module_type_store = module_type_store.open_function_context('gcv', 43, 8, False)
            
            # Passed parameters checking function
            gcv.stypy_localization = localization
            gcv.stypy_type_of_self = None
            gcv.stypy_type_store = module_type_store
            gcv.stypy_function_name = 'gcv'
            gcv.stypy_param_names_list = ['v']
            gcv.stypy_varargs_param_name = None
            gcv.stypy_kwargs_param_name = None
            gcv.stypy_call_defaults = defaults
            gcv.stypy_call_varargs = varargs
            gcv.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'gcv', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'gcv', localization, ['v'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'gcv(...)' code ##################

            str_44877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'str', 'gcc')
            # Assigning a type to the variable 'stypy_return_type' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'stypy_return_type', str_44877)
            
            # ################# End of 'gcv(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'gcv' in the type store
            # Getting the type of 'stypy_return_type' (line 43)
            stypy_return_type_44878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_44878)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'gcv'
            return stypy_return_type_44878

        # Assigning a type to the variable 'gcv' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'gcv', gcv)
        
        # Assigning a Name to a Attribute (line 45):
        # Getting the type of 'gcv' (line 45)
        gcv_44879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 35), 'gcv')
        # Getting the type of 'sysconfig' (line 45)
        sysconfig_44880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'sysconfig')
        # Setting the type of the member 'get_config_var' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), sysconfig_44880, 'get_config_var', gcv_44879)
        
        # Call to assertEqual(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Call to rpath_foo(...): (line 46)
        # Processing the call keyword arguments (line 46)
        kwargs_44886 = {}
        # Getting the type of 'self' (line 46)
        self_44883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 46)
        cc_44884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 25), self_44883, 'cc')
        # Obtaining the member 'rpath_foo' of a type (line 46)
        rpath_foo_44885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 25), cc_44884, 'rpath_foo')
        # Calling rpath_foo(args, kwargs) (line 46)
        rpath_foo_call_result_44887 = invoke(stypy.reporting.localization.Localization(__file__, 46, 25), rpath_foo_44885, *[], **kwargs_44886)
        
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_44888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        str_44889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 47), 'str', '-Wl,+s')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 46), list_44888, str_44889)
        # Adding element type (line 46)
        str_44890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 57), 'str', '-L/foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 46), list_44888, str_44890)
        
        # Processing the call keyword arguments (line 46)
        kwargs_44891 = {}
        # Getting the type of 'self' (line 46)
        self_44881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 46)
        assertEqual_44882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_44881, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 46)
        assertEqual_call_result_44892 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), assertEqual_44882, *[rpath_foo_call_result_44887, list_44888], **kwargs_44891)
        

        @norecursion
        def gcv(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'gcv'
            module_type_store = module_type_store.open_function_context('gcv', 48, 8, False)
            
            # Passed parameters checking function
            gcv.stypy_localization = localization
            gcv.stypy_type_of_self = None
            gcv.stypy_type_store = module_type_store
            gcv.stypy_function_name = 'gcv'
            gcv.stypy_param_names_list = ['v']
            gcv.stypy_varargs_param_name = None
            gcv.stypy_kwargs_param_name = None
            gcv.stypy_call_defaults = defaults
            gcv.stypy_call_varargs = varargs
            gcv.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'gcv', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'gcv', localization, ['v'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'gcv(...)' code ##################

            str_44893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 19), 'str', 'g++')
            # Assigning a type to the variable 'stypy_return_type' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'stypy_return_type', str_44893)
            
            # ################# End of 'gcv(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'gcv' in the type store
            # Getting the type of 'stypy_return_type' (line 48)
            stypy_return_type_44894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_44894)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'gcv'
            return stypy_return_type_44894

        # Assigning a type to the variable 'gcv' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'gcv', gcv)
        
        # Assigning a Name to a Attribute (line 50):
        # Getting the type of 'gcv' (line 50)
        gcv_44895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'gcv')
        # Getting the type of 'sysconfig' (line 50)
        sysconfig_44896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'sysconfig')
        # Setting the type of the member 'get_config_var' of a type (line 50)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), sysconfig_44896, 'get_config_var', gcv_44895)
        
        # Call to assertEqual(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to rpath_foo(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_44902 = {}
        # Getting the type of 'self' (line 51)
        self_44899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 51)
        cc_44900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), self_44899, 'cc')
        # Obtaining the member 'rpath_foo' of a type (line 51)
        rpath_foo_44901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), cc_44900, 'rpath_foo')
        # Calling rpath_foo(args, kwargs) (line 51)
        rpath_foo_call_result_44903 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), rpath_foo_44901, *[], **kwargs_44902)
        
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_44904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        str_44905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 47), 'str', '-Wl,+s')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 46), list_44904, str_44905)
        # Adding element type (line 51)
        str_44906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 57), 'str', '-L/foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 46), list_44904, str_44906)
        
        # Processing the call keyword arguments (line 51)
        kwargs_44907 = {}
        # Getting the type of 'self' (line 51)
        self_44897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 51)
        assertEqual_44898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_44897, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 51)
        assertEqual_call_result_44908 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assertEqual_44898, *[rpath_foo_call_result_44903, list_44904], **kwargs_44907)
        
        
        # Assigning a Name to a Attribute (line 53):
        # Getting the type of 'old_gcv' (line 53)
        old_gcv_44909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 35), 'old_gcv')
        # Getting the type of 'sysconfig' (line 53)
        sysconfig_44910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'sysconfig')
        # Setting the type of the member 'get_config_var' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), sysconfig_44910, 'get_config_var', old_gcv_44909)
        
        # Assigning a Str to a Attribute (line 56):
        str_44911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'str', 'irix646')
        # Getting the type of 'sys' (line 56)
        sys_44912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'sys')
        # Setting the type of the member 'platform' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), sys_44912, 'platform', str_44911)
        
        # Call to assertEqual(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Call to rpath_foo(...): (line 57)
        # Processing the call keyword arguments (line 57)
        kwargs_44918 = {}
        # Getting the type of 'self' (line 57)
        self_44915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 57)
        cc_44916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), self_44915, 'cc')
        # Obtaining the member 'rpath_foo' of a type (line 57)
        rpath_foo_44917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), cc_44916, 'rpath_foo')
        # Calling rpath_foo(args, kwargs) (line 57)
        rpath_foo_call_result_44919 = invoke(stypy.reporting.localization.Localization(__file__, 57, 25), rpath_foo_44917, *[], **kwargs_44918)
        
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_44920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        str_44921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 47), 'str', '-rpath')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 46), list_44920, str_44921)
        # Adding element type (line 57)
        str_44922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 57), 'str', '/foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 46), list_44920, str_44922)
        
        # Processing the call keyword arguments (line 57)
        kwargs_44923 = {}
        # Getting the type of 'self' (line 57)
        self_44913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 57)
        assertEqual_44914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_44913, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 57)
        assertEqual_call_result_44924 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), assertEqual_44914, *[rpath_foo_call_result_44919, list_44920], **kwargs_44923)
        
        
        # Assigning a Str to a Attribute (line 60):
        str_44925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 23), 'str', 'osf1V5')
        # Getting the type of 'sys' (line 60)
        sys_44926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'sys')
        # Setting the type of the member 'platform' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), sys_44926, 'platform', str_44925)
        
        # Call to assertEqual(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Call to rpath_foo(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_44932 = {}
        # Getting the type of 'self' (line 61)
        self_44929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 61)
        cc_44930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 25), self_44929, 'cc')
        # Obtaining the member 'rpath_foo' of a type (line 61)
        rpath_foo_44931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 25), cc_44930, 'rpath_foo')
        # Calling rpath_foo(args, kwargs) (line 61)
        rpath_foo_call_result_44933 = invoke(stypy.reporting.localization.Localization(__file__, 61, 25), rpath_foo_44931, *[], **kwargs_44932)
        
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_44934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        str_44935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 47), 'str', '-rpath')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 46), list_44934, str_44935)
        # Adding element type (line 61)
        str_44936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 57), 'str', '/foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 46), list_44934, str_44936)
        
        # Processing the call keyword arguments (line 61)
        kwargs_44937 = {}
        # Getting the type of 'self' (line 61)
        self_44927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 61)
        assertEqual_44928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_44927, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 61)
        assertEqual_call_result_44938 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assertEqual_44928, *[rpath_foo_call_result_44933, list_44934], **kwargs_44937)
        
        
        # Assigning a Str to a Attribute (line 64):
        str_44939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 23), 'str', 'bar')
        # Getting the type of 'sys' (line 64)
        sys_44940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'sys')
        # Setting the type of the member 'platform' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), sys_44940, 'platform', str_44939)

        @norecursion
        def gcv(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'gcv'
            module_type_store = module_type_store.open_function_context('gcv', 65, 8, False)
            
            # Passed parameters checking function
            gcv.stypy_localization = localization
            gcv.stypy_type_of_self = None
            gcv.stypy_type_store = module_type_store
            gcv.stypy_function_name = 'gcv'
            gcv.stypy_param_names_list = ['v']
            gcv.stypy_varargs_param_name = None
            gcv.stypy_kwargs_param_name = None
            gcv.stypy_call_defaults = defaults
            gcv.stypy_call_varargs = varargs
            gcv.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'gcv', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'gcv', localization, ['v'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'gcv(...)' code ##################

            
            
            # Getting the type of 'v' (line 66)
            v_44941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 15), 'v')
            str_44942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 20), 'str', 'CC')
            # Applying the binary operator '==' (line 66)
            result_eq_44943 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 15), '==', v_44941, str_44942)
            
            # Testing the type of an if condition (line 66)
            if_condition_44944 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 12), result_eq_44943)
            # Assigning a type to the variable 'if_condition_44944' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'if_condition_44944', if_condition_44944)
            # SSA begins for if statement (line 66)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_44945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 23), 'str', 'gcc')
            # Assigning a type to the variable 'stypy_return_type' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'stypy_return_type', str_44945)
            # SSA branch for the else part of an if statement (line 66)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'v' (line 68)
            v_44946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'v')
            str_44947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 22), 'str', 'GNULD')
            # Applying the binary operator '==' (line 68)
            result_eq_44948 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 17), '==', v_44946, str_44947)
            
            # Testing the type of an if condition (line 68)
            if_condition_44949 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 17), result_eq_44948)
            # Assigning a type to the variable 'if_condition_44949' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'if_condition_44949', if_condition_44949)
            # SSA begins for if statement (line 68)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_44950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'str', 'yes')
            # Assigning a type to the variable 'stypy_return_type' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'stypy_return_type', str_44950)
            # SSA join for if statement (line 68)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 66)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'gcv(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'gcv' in the type store
            # Getting the type of 'stypy_return_type' (line 65)
            stypy_return_type_44951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_44951)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'gcv'
            return stypy_return_type_44951

        # Assigning a type to the variable 'gcv' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'gcv', gcv)
        
        # Assigning a Name to a Attribute (line 70):
        # Getting the type of 'gcv' (line 70)
        gcv_44952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'gcv')
        # Getting the type of 'sysconfig' (line 70)
        sysconfig_44953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'sysconfig')
        # Setting the type of the member 'get_config_var' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), sysconfig_44953, 'get_config_var', gcv_44952)
        
        # Call to assertEqual(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Call to rpath_foo(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_44959 = {}
        # Getting the type of 'self' (line 71)
        self_44956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 71)
        cc_44957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 25), self_44956, 'cc')
        # Obtaining the member 'rpath_foo' of a type (line 71)
        rpath_foo_44958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 25), cc_44957, 'rpath_foo')
        # Calling rpath_foo(args, kwargs) (line 71)
        rpath_foo_call_result_44960 = invoke(stypy.reporting.localization.Localization(__file__, 71, 25), rpath_foo_44958, *[], **kwargs_44959)
        
        str_44961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 46), 'str', '-Wl,-R/foo')
        # Processing the call keyword arguments (line 71)
        kwargs_44962 = {}
        # Getting the type of 'self' (line 71)
        self_44954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 71)
        assertEqual_44955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_44954, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 71)
        assertEqual_call_result_44963 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), assertEqual_44955, *[rpath_foo_call_result_44960, str_44961], **kwargs_44962)
        
        
        # Assigning a Str to a Attribute (line 74):
        str_44964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 23), 'str', 'bar')
        # Getting the type of 'sys' (line 74)
        sys_44965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'sys')
        # Setting the type of the member 'platform' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), sys_44965, 'platform', str_44964)

        @norecursion
        def gcv(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'gcv'
            module_type_store = module_type_store.open_function_context('gcv', 75, 8, False)
            
            # Passed parameters checking function
            gcv.stypy_localization = localization
            gcv.stypy_type_of_self = None
            gcv.stypy_type_store = module_type_store
            gcv.stypy_function_name = 'gcv'
            gcv.stypy_param_names_list = ['v']
            gcv.stypy_varargs_param_name = None
            gcv.stypy_kwargs_param_name = None
            gcv.stypy_call_defaults = defaults
            gcv.stypy_call_varargs = varargs
            gcv.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'gcv', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'gcv', localization, ['v'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'gcv(...)' code ##################

            
            
            # Getting the type of 'v' (line 76)
            v_44966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'v')
            str_44967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 20), 'str', 'CC')
            # Applying the binary operator '==' (line 76)
            result_eq_44968 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 15), '==', v_44966, str_44967)
            
            # Testing the type of an if condition (line 76)
            if_condition_44969 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 12), result_eq_44968)
            # Assigning a type to the variable 'if_condition_44969' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'if_condition_44969', if_condition_44969)
            # SSA begins for if statement (line 76)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_44970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 23), 'str', 'gcc')
            # Assigning a type to the variable 'stypy_return_type' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'stypy_return_type', str_44970)
            # SSA branch for the else part of an if statement (line 76)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'v' (line 78)
            v_44971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'v')
            str_44972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 22), 'str', 'GNULD')
            # Applying the binary operator '==' (line 78)
            result_eq_44973 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 17), '==', v_44971, str_44972)
            
            # Testing the type of an if condition (line 78)
            if_condition_44974 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 17), result_eq_44973)
            # Assigning a type to the variable 'if_condition_44974' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'if_condition_44974', if_condition_44974)
            # SSA begins for if statement (line 78)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_44975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 23), 'str', 'no')
            # Assigning a type to the variable 'stypy_return_type' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'stypy_return_type', str_44975)
            # SSA join for if statement (line 78)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 76)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'gcv(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'gcv' in the type store
            # Getting the type of 'stypy_return_type' (line 75)
            stypy_return_type_44976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_44976)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'gcv'
            return stypy_return_type_44976

        # Assigning a type to the variable 'gcv' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'gcv', gcv)
        
        # Assigning a Name to a Attribute (line 80):
        # Getting the type of 'gcv' (line 80)
        gcv_44977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'gcv')
        # Getting the type of 'sysconfig' (line 80)
        sysconfig_44978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'sysconfig')
        # Setting the type of the member 'get_config_var' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), sysconfig_44978, 'get_config_var', gcv_44977)
        
        # Call to assertEqual(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Call to rpath_foo(...): (line 81)
        # Processing the call keyword arguments (line 81)
        kwargs_44984 = {}
        # Getting the type of 'self' (line 81)
        self_44981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 81)
        cc_44982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 25), self_44981, 'cc')
        # Obtaining the member 'rpath_foo' of a type (line 81)
        rpath_foo_44983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 25), cc_44982, 'rpath_foo')
        # Calling rpath_foo(args, kwargs) (line 81)
        rpath_foo_call_result_44985 = invoke(stypy.reporting.localization.Localization(__file__, 81, 25), rpath_foo_44983, *[], **kwargs_44984)
        
        str_44986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 46), 'str', '-Wl,-R/foo')
        # Processing the call keyword arguments (line 81)
        kwargs_44987 = {}
        # Getting the type of 'self' (line 81)
        self_44979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 81)
        assertEqual_44980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_44979, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 81)
        assertEqual_call_result_44988 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), assertEqual_44980, *[rpath_foo_call_result_44985, str_44986], **kwargs_44987)
        
        
        # Assigning a Str to a Attribute (line 85):
        str_44989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 23), 'str', 'bar')
        # Getting the type of 'sys' (line 85)
        sys_44990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'sys')
        # Setting the type of the member 'platform' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), sys_44990, 'platform', str_44989)

        @norecursion
        def gcv(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'gcv'
            module_type_store = module_type_store.open_function_context('gcv', 86, 8, False)
            
            # Passed parameters checking function
            gcv.stypy_localization = localization
            gcv.stypy_type_of_self = None
            gcv.stypy_type_store = module_type_store
            gcv.stypy_function_name = 'gcv'
            gcv.stypy_param_names_list = ['v']
            gcv.stypy_varargs_param_name = None
            gcv.stypy_kwargs_param_name = None
            gcv.stypy_call_defaults = defaults
            gcv.stypy_call_varargs = varargs
            gcv.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'gcv', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'gcv', localization, ['v'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'gcv(...)' code ##################

            
            
            # Getting the type of 'v' (line 87)
            v_44991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'v')
            str_44992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'str', 'CC')
            # Applying the binary operator '==' (line 87)
            result_eq_44993 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 15), '==', v_44991, str_44992)
            
            # Testing the type of an if condition (line 87)
            if_condition_44994 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 12), result_eq_44993)
            # Assigning a type to the variable 'if_condition_44994' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'if_condition_44994', if_condition_44994)
            # SSA begins for if statement (line 87)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_44995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 23), 'str', 'x86_64-pc-linux-gnu-gcc-4.4.2')
            # Assigning a type to the variable 'stypy_return_type' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'stypy_return_type', str_44995)
            # SSA branch for the else part of an if statement (line 87)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'v' (line 89)
            v_44996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'v')
            str_44997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'str', 'GNULD')
            # Applying the binary operator '==' (line 89)
            result_eq_44998 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 17), '==', v_44996, str_44997)
            
            # Testing the type of an if condition (line 89)
            if_condition_44999 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 17), result_eq_44998)
            # Assigning a type to the variable 'if_condition_44999' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'if_condition_44999', if_condition_44999)
            # SSA begins for if statement (line 89)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_45000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'str', 'yes')
            # Assigning a type to the variable 'stypy_return_type' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'stypy_return_type', str_45000)
            # SSA join for if statement (line 89)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 87)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'gcv(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'gcv' in the type store
            # Getting the type of 'stypy_return_type' (line 86)
            stypy_return_type_45001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_45001)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'gcv'
            return stypy_return_type_45001

        # Assigning a type to the variable 'gcv' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'gcv', gcv)
        
        # Assigning a Name to a Attribute (line 91):
        # Getting the type of 'gcv' (line 91)
        gcv_45002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 35), 'gcv')
        # Getting the type of 'sysconfig' (line 91)
        sysconfig_45003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'sysconfig')
        # Setting the type of the member 'get_config_var' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), sysconfig_45003, 'get_config_var', gcv_45002)
        
        # Call to assertEqual(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to rpath_foo(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_45009 = {}
        # Getting the type of 'self' (line 92)
        self_45006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 92)
        cc_45007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 25), self_45006, 'cc')
        # Obtaining the member 'rpath_foo' of a type (line 92)
        rpath_foo_45008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 25), cc_45007, 'rpath_foo')
        # Calling rpath_foo(args, kwargs) (line 92)
        rpath_foo_call_result_45010 = invoke(stypy.reporting.localization.Localization(__file__, 92, 25), rpath_foo_45008, *[], **kwargs_45009)
        
        str_45011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 46), 'str', '-Wl,-R/foo')
        # Processing the call keyword arguments (line 92)
        kwargs_45012 = {}
        # Getting the type of 'self' (line 92)
        self_45004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 92)
        assertEqual_45005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_45004, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 92)
        assertEqual_call_result_45013 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), assertEqual_45005, *[rpath_foo_call_result_45010, str_45011], **kwargs_45012)
        
        
        # Assigning a Str to a Attribute (line 96):
        str_45014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 23), 'str', 'bar')
        # Getting the type of 'sys' (line 96)
        sys_45015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'sys')
        # Setting the type of the member 'platform' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), sys_45015, 'platform', str_45014)

        @norecursion
        def gcv(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'gcv'
            module_type_store = module_type_store.open_function_context('gcv', 97, 8, False)
            
            # Passed parameters checking function
            gcv.stypy_localization = localization
            gcv.stypy_type_of_self = None
            gcv.stypy_type_store = module_type_store
            gcv.stypy_function_name = 'gcv'
            gcv.stypy_param_names_list = ['v']
            gcv.stypy_varargs_param_name = None
            gcv.stypy_kwargs_param_name = None
            gcv.stypy_call_defaults = defaults
            gcv.stypy_call_varargs = varargs
            gcv.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'gcv', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'gcv', localization, ['v'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'gcv(...)' code ##################

            
            
            # Getting the type of 'v' (line 98)
            v_45016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'v')
            str_45017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 20), 'str', 'CC')
            # Applying the binary operator '==' (line 98)
            result_eq_45018 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 15), '==', v_45016, str_45017)
            
            # Testing the type of an if condition (line 98)
            if_condition_45019 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 12), result_eq_45018)
            # Assigning a type to the variable 'if_condition_45019' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'if_condition_45019', if_condition_45019)
            # SSA begins for if statement (line 98)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_45020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 23), 'str', 'cc')
            # Assigning a type to the variable 'stypy_return_type' (line 99)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'stypy_return_type', str_45020)
            # SSA branch for the else part of an if statement (line 98)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'v' (line 100)
            v_45021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 17), 'v')
            str_45022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 22), 'str', 'GNULD')
            # Applying the binary operator '==' (line 100)
            result_eq_45023 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 17), '==', v_45021, str_45022)
            
            # Testing the type of an if condition (line 100)
            if_condition_45024 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 17), result_eq_45023)
            # Assigning a type to the variable 'if_condition_45024' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 17), 'if_condition_45024', if_condition_45024)
            # SSA begins for if statement (line 100)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_45025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 23), 'str', 'yes')
            # Assigning a type to the variable 'stypy_return_type' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'stypy_return_type', str_45025)
            # SSA join for if statement (line 100)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 98)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'gcv(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'gcv' in the type store
            # Getting the type of 'stypy_return_type' (line 97)
            stypy_return_type_45026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_45026)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'gcv'
            return stypy_return_type_45026

        # Assigning a type to the variable 'gcv' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'gcv', gcv)
        
        # Assigning a Name to a Attribute (line 102):
        # Getting the type of 'gcv' (line 102)
        gcv_45027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 35), 'gcv')
        # Getting the type of 'sysconfig' (line 102)
        sysconfig_45028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'sysconfig')
        # Setting the type of the member 'get_config_var' of a type (line 102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), sysconfig_45028, 'get_config_var', gcv_45027)
        
        # Call to assertEqual(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to rpath_foo(...): (line 103)
        # Processing the call keyword arguments (line 103)
        kwargs_45034 = {}
        # Getting the type of 'self' (line 103)
        self_45031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 103)
        cc_45032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 25), self_45031, 'cc')
        # Obtaining the member 'rpath_foo' of a type (line 103)
        rpath_foo_45033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 25), cc_45032, 'rpath_foo')
        # Calling rpath_foo(args, kwargs) (line 103)
        rpath_foo_call_result_45035 = invoke(stypy.reporting.localization.Localization(__file__, 103, 25), rpath_foo_45033, *[], **kwargs_45034)
        
        str_45036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 46), 'str', '-R/foo')
        # Processing the call keyword arguments (line 103)
        kwargs_45037 = {}
        # Getting the type of 'self' (line 103)
        self_45029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 103)
        assertEqual_45030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_45029, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 103)
        assertEqual_call_result_45038 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), assertEqual_45030, *[rpath_foo_call_result_45035, str_45036], **kwargs_45037)
        
        
        # Assigning a Str to a Attribute (line 106):
        str_45039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 23), 'str', 'bar')
        # Getting the type of 'sys' (line 106)
        sys_45040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'sys')
        # Setting the type of the member 'platform' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), sys_45040, 'platform', str_45039)

        @norecursion
        def gcv(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'gcv'
            module_type_store = module_type_store.open_function_context('gcv', 107, 8, False)
            
            # Passed parameters checking function
            gcv.stypy_localization = localization
            gcv.stypy_type_of_self = None
            gcv.stypy_type_store = module_type_store
            gcv.stypy_function_name = 'gcv'
            gcv.stypy_param_names_list = ['v']
            gcv.stypy_varargs_param_name = None
            gcv.stypy_kwargs_param_name = None
            gcv.stypy_call_defaults = defaults
            gcv.stypy_call_varargs = varargs
            gcv.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'gcv', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'gcv', localization, ['v'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'gcv(...)' code ##################

            
            
            # Getting the type of 'v' (line 108)
            v_45041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'v')
            str_45042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 20), 'str', 'CC')
            # Applying the binary operator '==' (line 108)
            result_eq_45043 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 15), '==', v_45041, str_45042)
            
            # Testing the type of an if condition (line 108)
            if_condition_45044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 12), result_eq_45043)
            # Assigning a type to the variable 'if_condition_45044' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'if_condition_45044', if_condition_45044)
            # SSA begins for if statement (line 108)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_45045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'str', 'cc')
            # Assigning a type to the variable 'stypy_return_type' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'stypy_return_type', str_45045)
            # SSA branch for the else part of an if statement (line 108)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'v' (line 110)
            v_45046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'v')
            str_45047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 22), 'str', 'GNULD')
            # Applying the binary operator '==' (line 110)
            result_eq_45048 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 17), '==', v_45046, str_45047)
            
            # Testing the type of an if condition (line 110)
            if_condition_45049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 17), result_eq_45048)
            # Assigning a type to the variable 'if_condition_45049' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'if_condition_45049', if_condition_45049)
            # SSA begins for if statement (line 110)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_45050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 23), 'str', 'no')
            # Assigning a type to the variable 'stypy_return_type' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'stypy_return_type', str_45050)
            # SSA join for if statement (line 110)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 108)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'gcv(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'gcv' in the type store
            # Getting the type of 'stypy_return_type' (line 107)
            stypy_return_type_45051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_45051)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'gcv'
            return stypy_return_type_45051

        # Assigning a type to the variable 'gcv' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'gcv', gcv)
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'gcv' (line 112)
        gcv_45052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 35), 'gcv')
        # Getting the type of 'sysconfig' (line 112)
        sysconfig_45053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'sysconfig')
        # Setting the type of the member 'get_config_var' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), sysconfig_45053, 'get_config_var', gcv_45052)
        
        # Call to assertEqual(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to rpath_foo(...): (line 113)
        # Processing the call keyword arguments (line 113)
        kwargs_45059 = {}
        # Getting the type of 'self' (line 113)
        self_45056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 113)
        cc_45057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 25), self_45056, 'cc')
        # Obtaining the member 'rpath_foo' of a type (line 113)
        rpath_foo_45058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 25), cc_45057, 'rpath_foo')
        # Calling rpath_foo(args, kwargs) (line 113)
        rpath_foo_call_result_45060 = invoke(stypy.reporting.localization.Localization(__file__, 113, 25), rpath_foo_45058, *[], **kwargs_45059)
        
        str_45061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 46), 'str', '-R/foo')
        # Processing the call keyword arguments (line 113)
        kwargs_45062 = {}
        # Getting the type of 'self' (line 113)
        self_45054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 113)
        assertEqual_45055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_45054, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 113)
        assertEqual_call_result_45063 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), assertEqual_45055, *[rpath_foo_call_result_45060, str_45061], **kwargs_45062)
        
        
        # Assigning a Str to a Attribute (line 116):
        str_45064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 23), 'str', 'aix')
        # Getting the type of 'sys' (line 116)
        sys_45065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'sys')
        # Setting the type of the member 'platform' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), sys_45065, 'platform', str_45064)

        @norecursion
        def gcv(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'gcv'
            module_type_store = module_type_store.open_function_context('gcv', 117, 8, False)
            
            # Passed parameters checking function
            gcv.stypy_localization = localization
            gcv.stypy_type_of_self = None
            gcv.stypy_type_store = module_type_store
            gcv.stypy_function_name = 'gcv'
            gcv.stypy_param_names_list = ['v']
            gcv.stypy_varargs_param_name = None
            gcv.stypy_kwargs_param_name = None
            gcv.stypy_call_defaults = defaults
            gcv.stypy_call_varargs = varargs
            gcv.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'gcv', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'gcv', localization, ['v'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'gcv(...)' code ##################

            str_45066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 19), 'str', 'xxx')
            # Assigning a type to the variable 'stypy_return_type' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'stypy_return_type', str_45066)
            
            # ################# End of 'gcv(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'gcv' in the type store
            # Getting the type of 'stypy_return_type' (line 117)
            stypy_return_type_45067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_45067)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'gcv'
            return stypy_return_type_45067

        # Assigning a type to the variable 'gcv' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'gcv', gcv)
        
        # Assigning a Name to a Attribute (line 119):
        # Getting the type of 'gcv' (line 119)
        gcv_45068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'gcv')
        # Getting the type of 'sysconfig' (line 119)
        sysconfig_45069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'sysconfig')
        # Setting the type of the member 'get_config_var' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), sysconfig_45069, 'get_config_var', gcv_45068)
        
        # Call to assertEqual(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Call to rpath_foo(...): (line 120)
        # Processing the call keyword arguments (line 120)
        kwargs_45075 = {}
        # Getting the type of 'self' (line 120)
        self_45072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 120)
        cc_45073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 25), self_45072, 'cc')
        # Obtaining the member 'rpath_foo' of a type (line 120)
        rpath_foo_45074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 25), cc_45073, 'rpath_foo')
        # Calling rpath_foo(args, kwargs) (line 120)
        rpath_foo_call_result_45076 = invoke(stypy.reporting.localization.Localization(__file__, 120, 25), rpath_foo_45074, *[], **kwargs_45075)
        
        str_45077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 46), 'str', '-R/foo')
        # Processing the call keyword arguments (line 120)
        kwargs_45078 = {}
        # Getting the type of 'self' (line 120)
        self_45070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 120)
        assertEqual_45071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_45070, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 120)
        assertEqual_call_result_45079 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), assertEqual_45071, *[rpath_foo_call_result_45076, str_45077], **kwargs_45078)
        
        
        # ################# End of 'test_runtime_libdir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_runtime_libdir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_45080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45080)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_runtime_libdir_option'
        return stypy_return_type_45080


    @norecursion
    def test_osx_cc_overrides_ldshared(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_osx_cc_overrides_ldshared'
        module_type_store = module_type_store.open_function_context('test_osx_cc_overrides_ldshared', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnixCCompilerTestCase.test_osx_cc_overrides_ldshared.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompilerTestCase.test_osx_cc_overrides_ldshared.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompilerTestCase.test_osx_cc_overrides_ldshared.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompilerTestCase.test_osx_cc_overrides_ldshared.__dict__.__setitem__('stypy_function_name', 'UnixCCompilerTestCase.test_osx_cc_overrides_ldshared')
        UnixCCompilerTestCase.test_osx_cc_overrides_ldshared.__dict__.__setitem__('stypy_param_names_list', [])
        UnixCCompilerTestCase.test_osx_cc_overrides_ldshared.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompilerTestCase.test_osx_cc_overrides_ldshared.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompilerTestCase.test_osx_cc_overrides_ldshared.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompilerTestCase.test_osx_cc_overrides_ldshared.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompilerTestCase.test_osx_cc_overrides_ldshared.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompilerTestCase.test_osx_cc_overrides_ldshared.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompilerTestCase.test_osx_cc_overrides_ldshared', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_osx_cc_overrides_ldshared', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_osx_cc_overrides_ldshared(...)' code ##################


        @norecursion
        def gcv(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'gcv'
            module_type_store = module_type_store.open_function_context('gcv', 126, 8, False)
            
            # Passed parameters checking function
            gcv.stypy_localization = localization
            gcv.stypy_type_of_self = None
            gcv.stypy_type_store = module_type_store
            gcv.stypy_function_name = 'gcv'
            gcv.stypy_param_names_list = ['v']
            gcv.stypy_varargs_param_name = None
            gcv.stypy_kwargs_param_name = None
            gcv.stypy_call_defaults = defaults
            gcv.stypy_call_varargs = varargs
            gcv.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'gcv', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'gcv', localization, ['v'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'gcv(...)' code ##################

            
            
            # Getting the type of 'v' (line 127)
            v_45081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'v')
            str_45082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 20), 'str', 'LDSHARED')
            # Applying the binary operator '==' (line 127)
            result_eq_45083 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 15), '==', v_45081, str_45082)
            
            # Testing the type of an if condition (line 127)
            if_condition_45084 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 12), result_eq_45083)
            # Assigning a type to the variable 'if_condition_45084' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'if_condition_45084', if_condition_45084)
            # SSA begins for if statement (line 127)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_45085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 23), 'str', 'gcc-4.2 -bundle -undefined dynamic_lookup ')
            # Assigning a type to the variable 'stypy_return_type' (line 128)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'stypy_return_type', str_45085)
            # SSA join for if statement (line 127)
            module_type_store = module_type_store.join_ssa_context()
            
            str_45086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 19), 'str', 'gcc-4.2')
            # Assigning a type to the variable 'stypy_return_type' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'stypy_return_type', str_45086)
            
            # ################# End of 'gcv(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'gcv' in the type store
            # Getting the type of 'stypy_return_type' (line 126)
            stypy_return_type_45087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_45087)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'gcv'
            return stypy_return_type_45087

        # Assigning a type to the variable 'gcv' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'gcv', gcv)
        
        # Assigning a Name to a Attribute (line 130):
        # Getting the type of 'gcv' (line 130)
        gcv_45088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'gcv')
        # Getting the type of 'sysconfig' (line 130)
        sysconfig_45089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'sysconfig')
        # Setting the type of the member 'get_config_var' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), sysconfig_45089, 'get_config_var', gcv_45088)
        
        # Call to EnvironmentVarGuard(...): (line 131)
        # Processing the call keyword arguments (line 131)
        kwargs_45091 = {}
        # Getting the type of 'EnvironmentVarGuard' (line 131)
        EnvironmentVarGuard_45090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 13), 'EnvironmentVarGuard', False)
        # Calling EnvironmentVarGuard(args, kwargs) (line 131)
        EnvironmentVarGuard_call_result_45092 = invoke(stypy.reporting.localization.Localization(__file__, 131, 13), EnvironmentVarGuard_45090, *[], **kwargs_45091)
        
        with_45093 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 131, 13), EnvironmentVarGuard_call_result_45092, 'with parameter', '__enter__', '__exit__')

        if with_45093:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 131)
            enter___45094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 13), EnvironmentVarGuard_call_result_45092, '__enter__')
            with_enter_45095 = invoke(stypy.reporting.localization.Localization(__file__, 131, 13), enter___45094)
            # Assigning a type to the variable 'env' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 13), 'env', with_enter_45095)
            
            # Assigning a Str to a Subscript (line 132):
            str_45096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 24), 'str', 'my_cc')
            # Getting the type of 'env' (line 132)
            env_45097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'env')
            str_45098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 16), 'str', 'CC')
            # Storing an element on a container (line 132)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 12), env_45097, (str_45098, str_45096))
            # Deleting a member
            # Getting the type of 'env' (line 133)
            env_45099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'env')
            
            # Obtaining the type of the subscript
            str_45100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 20), 'str', 'LDSHARED')
            # Getting the type of 'env' (line 133)
            env_45101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'env')
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___45102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), env_45101, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_45103 = invoke(stypy.reporting.localization.Localization(__file__, 133, 16), getitem___45102, str_45100)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 12), env_45099, subscript_call_result_45103)
            
            # Call to customize_compiler(...): (line 134)
            # Processing the call arguments (line 134)
            # Getting the type of 'self' (line 134)
            self_45106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 41), 'self', False)
            # Obtaining the member 'cc' of a type (line 134)
            cc_45107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 41), self_45106, 'cc')
            # Processing the call keyword arguments (line 134)
            kwargs_45108 = {}
            # Getting the type of 'sysconfig' (line 134)
            sysconfig_45104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'sysconfig', False)
            # Obtaining the member 'customize_compiler' of a type (line 134)
            customize_compiler_45105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 12), sysconfig_45104, 'customize_compiler')
            # Calling customize_compiler(args, kwargs) (line 134)
            customize_compiler_call_result_45109 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), customize_compiler_45105, *[cc_45107], **kwargs_45108)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 131)
            exit___45110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 13), EnvironmentVarGuard_call_result_45092, '__exit__')
            with_exit_45111 = invoke(stypy.reporting.localization.Localization(__file__, 131, 13), exit___45110, None, None, None)

        
        # Call to assertEqual(...): (line 135)
        # Processing the call arguments (line 135)
        
        # Obtaining the type of the subscript
        int_45114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 43), 'int')
        # Getting the type of 'self' (line 135)
        self_45115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 135)
        cc_45116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 25), self_45115, 'cc')
        # Obtaining the member 'linker_so' of a type (line 135)
        linker_so_45117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 25), cc_45116, 'linker_so')
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___45118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 25), linker_so_45117, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_45119 = invoke(stypy.reporting.localization.Localization(__file__, 135, 25), getitem___45118, int_45114)
        
        str_45120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 47), 'str', 'my_cc')
        # Processing the call keyword arguments (line 135)
        kwargs_45121 = {}
        # Getting the type of 'self' (line 135)
        self_45112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 135)
        assertEqual_45113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), self_45112, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 135)
        assertEqual_call_result_45122 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), assertEqual_45113, *[subscript_call_result_45119, str_45120], **kwargs_45121)
        
        
        # ################# End of 'test_osx_cc_overrides_ldshared(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_osx_cc_overrides_ldshared' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_45123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45123)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_osx_cc_overrides_ldshared'
        return stypy_return_type_45123


    @norecursion
    def test_osx_explicit_ldshared(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_osx_explicit_ldshared'
        module_type_store = module_type_store.open_function_context('test_osx_explicit_ldshared', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnixCCompilerTestCase.test_osx_explicit_ldshared.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompilerTestCase.test_osx_explicit_ldshared.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompilerTestCase.test_osx_explicit_ldshared.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompilerTestCase.test_osx_explicit_ldshared.__dict__.__setitem__('stypy_function_name', 'UnixCCompilerTestCase.test_osx_explicit_ldshared')
        UnixCCompilerTestCase.test_osx_explicit_ldshared.__dict__.__setitem__('stypy_param_names_list', [])
        UnixCCompilerTestCase.test_osx_explicit_ldshared.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompilerTestCase.test_osx_explicit_ldshared.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompilerTestCase.test_osx_explicit_ldshared.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompilerTestCase.test_osx_explicit_ldshared.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompilerTestCase.test_osx_explicit_ldshared.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompilerTestCase.test_osx_explicit_ldshared.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompilerTestCase.test_osx_explicit_ldshared', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_osx_explicit_ldshared', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_osx_explicit_ldshared(...)' code ##################


        @norecursion
        def gcv(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'gcv'
            module_type_store = module_type_store.open_function_context('gcv', 142, 8, False)
            
            # Passed parameters checking function
            gcv.stypy_localization = localization
            gcv.stypy_type_of_self = None
            gcv.stypy_type_store = module_type_store
            gcv.stypy_function_name = 'gcv'
            gcv.stypy_param_names_list = ['v']
            gcv.stypy_varargs_param_name = None
            gcv.stypy_kwargs_param_name = None
            gcv.stypy_call_defaults = defaults
            gcv.stypy_call_varargs = varargs
            gcv.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'gcv', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'gcv', localization, ['v'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'gcv(...)' code ##################

            
            
            # Getting the type of 'v' (line 143)
            v_45124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'v')
            str_45125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'str', 'LDSHARED')
            # Applying the binary operator '==' (line 143)
            result_eq_45126 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 15), '==', v_45124, str_45125)
            
            # Testing the type of an if condition (line 143)
            if_condition_45127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 12), result_eq_45126)
            # Assigning a type to the variable 'if_condition_45127' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'if_condition_45127', if_condition_45127)
            # SSA begins for if statement (line 143)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_45128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 23), 'str', 'gcc-4.2 -bundle -undefined dynamic_lookup ')
            # Assigning a type to the variable 'stypy_return_type' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'stypy_return_type', str_45128)
            # SSA join for if statement (line 143)
            module_type_store = module_type_store.join_ssa_context()
            
            str_45129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 19), 'str', 'gcc-4.2')
            # Assigning a type to the variable 'stypy_return_type' (line 145)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'stypy_return_type', str_45129)
            
            # ################# End of 'gcv(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'gcv' in the type store
            # Getting the type of 'stypy_return_type' (line 142)
            stypy_return_type_45130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_45130)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'gcv'
            return stypy_return_type_45130

        # Assigning a type to the variable 'gcv' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'gcv', gcv)
        
        # Assigning a Name to a Attribute (line 146):
        # Getting the type of 'gcv' (line 146)
        gcv_45131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 35), 'gcv')
        # Getting the type of 'sysconfig' (line 146)
        sysconfig_45132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'sysconfig')
        # Setting the type of the member 'get_config_var' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), sysconfig_45132, 'get_config_var', gcv_45131)
        
        # Call to EnvironmentVarGuard(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_45134 = {}
        # Getting the type of 'EnvironmentVarGuard' (line 147)
        EnvironmentVarGuard_45133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 13), 'EnvironmentVarGuard', False)
        # Calling EnvironmentVarGuard(args, kwargs) (line 147)
        EnvironmentVarGuard_call_result_45135 = invoke(stypy.reporting.localization.Localization(__file__, 147, 13), EnvironmentVarGuard_45133, *[], **kwargs_45134)
        
        with_45136 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 147, 13), EnvironmentVarGuard_call_result_45135, 'with parameter', '__enter__', '__exit__')

        if with_45136:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 147)
            enter___45137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 13), EnvironmentVarGuard_call_result_45135, '__enter__')
            with_enter_45138 = invoke(stypy.reporting.localization.Localization(__file__, 147, 13), enter___45137)
            # Assigning a type to the variable 'env' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 13), 'env', with_enter_45138)
            
            # Assigning a Str to a Subscript (line 148):
            str_45139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 24), 'str', 'my_cc')
            # Getting the type of 'env' (line 148)
            env_45140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'env')
            str_45141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 16), 'str', 'CC')
            # Storing an element on a container (line 148)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 12), env_45140, (str_45141, str_45139))
            
            # Assigning a Str to a Subscript (line 149):
            str_45142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 30), 'str', 'my_ld -bundle -dynamic')
            # Getting the type of 'env' (line 149)
            env_45143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'env')
            str_45144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 16), 'str', 'LDSHARED')
            # Storing an element on a container (line 149)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 12), env_45143, (str_45144, str_45142))
            
            # Call to customize_compiler(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'self' (line 150)
            self_45147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 41), 'self', False)
            # Obtaining the member 'cc' of a type (line 150)
            cc_45148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 41), self_45147, 'cc')
            # Processing the call keyword arguments (line 150)
            kwargs_45149 = {}
            # Getting the type of 'sysconfig' (line 150)
            sysconfig_45145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'sysconfig', False)
            # Obtaining the member 'customize_compiler' of a type (line 150)
            customize_compiler_45146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 12), sysconfig_45145, 'customize_compiler')
            # Calling customize_compiler(args, kwargs) (line 150)
            customize_compiler_call_result_45150 = invoke(stypy.reporting.localization.Localization(__file__, 150, 12), customize_compiler_45146, *[cc_45148], **kwargs_45149)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 147)
            exit___45151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 13), EnvironmentVarGuard_call_result_45135, '__exit__')
            with_exit_45152 = invoke(stypy.reporting.localization.Localization(__file__, 147, 13), exit___45151, None, None, None)

        
        # Call to assertEqual(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Obtaining the type of the subscript
        int_45155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 43), 'int')
        # Getting the type of 'self' (line 151)
        self_45156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'self', False)
        # Obtaining the member 'cc' of a type (line 151)
        cc_45157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 25), self_45156, 'cc')
        # Obtaining the member 'linker_so' of a type (line 151)
        linker_so_45158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 25), cc_45157, 'linker_so')
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___45159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 25), linker_so_45158, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_45160 = invoke(stypy.reporting.localization.Localization(__file__, 151, 25), getitem___45159, int_45155)
        
        str_45161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 47), 'str', 'my_ld')
        # Processing the call keyword arguments (line 151)
        kwargs_45162 = {}
        # Getting the type of 'self' (line 151)
        self_45153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 151)
        assertEqual_45154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), self_45153, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 151)
        assertEqual_call_result_45163 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), assertEqual_45154, *[subscript_call_result_45160, str_45161], **kwargs_45162)
        
        
        # ################# End of 'test_osx_explicit_ldshared(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_osx_explicit_ldshared' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_45164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45164)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_osx_explicit_ldshared'
        return stypy_return_type_45164


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 0, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompilerTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'UnixCCompilerTestCase' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'UnixCCompilerTestCase', UnixCCompilerTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 154, 0, False)
    
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

    
    # Call to makeSuite(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'UnixCCompilerTestCase' (line 155)
    UnixCCompilerTestCase_45167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 30), 'UnixCCompilerTestCase', False)
    # Processing the call keyword arguments (line 155)
    kwargs_45168 = {}
    # Getting the type of 'unittest' (line 155)
    unittest_45165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 155)
    makeSuite_45166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 11), unittest_45165, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 155)
    makeSuite_call_result_45169 = invoke(stypy.reporting.localization.Localization(__file__, 155, 11), makeSuite_45166, *[UnixCCompilerTestCase_45167], **kwargs_45168)
    
    # Assigning a type to the variable 'stypy_return_type' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type', makeSuite_call_result_45169)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 154)
    stypy_return_type_45170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_45170)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_45170

# Assigning a type to the variable 'test_suite' (line 154)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 158)
    # Processing the call arguments (line 158)
    
    # Call to test_suite(...): (line 158)
    # Processing the call keyword arguments (line 158)
    kwargs_45173 = {}
    # Getting the type of 'test_suite' (line 158)
    test_suite_45172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 158)
    test_suite_call_result_45174 = invoke(stypy.reporting.localization.Localization(__file__, 158, 17), test_suite_45172, *[], **kwargs_45173)
    
    # Processing the call keyword arguments (line 158)
    kwargs_45175 = {}
    # Getting the type of 'run_unittest' (line 158)
    run_unittest_45171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 158)
    run_unittest_call_result_45176 = invoke(stypy.reporting.localization.Localization(__file__, 158, 4), run_unittest_45171, *[test_suite_call_result_45174], **kwargs_45175)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
