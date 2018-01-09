
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.ccompiler.'''
2: import os
3: import unittest
4: from test.test_support import captured_stdout
5: 
6: from distutils.ccompiler import (gen_lib_options, CCompiler,
7:                                  get_default_compiler)
8: from distutils.sysconfig import customize_compiler
9: from distutils import debug
10: from distutils.tests import support
11: 
12: class FakeCompiler(object):
13:     def library_dir_option(self, dir):
14:         return "-L" + dir
15: 
16:     def runtime_library_dir_option(self, dir):
17:         return ["-cool", "-R" + dir]
18: 
19:     def find_library_file(self, dirs, lib, debug=0):
20:         return 'found'
21: 
22:     def library_option(self, lib):
23:         return "-l" + lib
24: 
25: class CCompilerTestCase(support.EnvironGuard, unittest.TestCase):
26: 
27:     def test_gen_lib_options(self):
28:         compiler = FakeCompiler()
29:         libdirs = ['lib1', 'lib2']
30:         runlibdirs = ['runlib1']
31:         libs = [os.path.join('dir', 'name'), 'name2']
32: 
33:         opts = gen_lib_options(compiler, libdirs, runlibdirs, libs)
34:         wanted = ['-Llib1', '-Llib2', '-cool', '-Rrunlib1', 'found',
35:                   '-lname2']
36:         self.assertEqual(opts, wanted)
37: 
38:     def test_debug_print(self):
39: 
40:         class MyCCompiler(CCompiler):
41:             executables = {}
42: 
43:         compiler = MyCCompiler()
44:         with captured_stdout() as stdout:
45:             compiler.debug_print('xxx')
46:         stdout.seek(0)
47:         self.assertEqual(stdout.read(), '')
48: 
49:         debug.DEBUG = True
50:         try:
51:             with captured_stdout() as stdout:
52:                 compiler.debug_print('xxx')
53:             stdout.seek(0)
54:             self.assertEqual(stdout.read(), 'xxx\n')
55:         finally:
56:             debug.DEBUG = False
57: 
58:     @unittest.skipUnless(get_default_compiler() == 'unix',
59:                          'not testing if default compiler is not unix')
60:     def test_customize_compiler(self):
61:         os.environ['AR'] = 'my_ar'
62:         os.environ['ARFLAGS'] = '-arflags'
63: 
64:         # make sure AR gets caught
65:         class compiler:
66:             compiler_type = 'unix'
67: 
68:             def set_executables(self, **kw):
69:                 self.exes = kw
70: 
71:         comp = compiler()
72:         customize_compiler(comp)
73:         self.assertEqual(comp.exes['archiver'], 'my_ar -arflags')
74: 
75: def test_suite():
76:     return unittest.makeSuite(CCompilerTestCase)
77: 
78: if __name__ == "__main__":
79:     unittest.main(defaultTest="test_suite")
80: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_34038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.ccompiler.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import os' statement (line 2)
import os

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import unittest' statement (line 3)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from test.test_support import captured_stdout' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34039 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support')

if (type(import_34039) is not StypyTypeError):

    if (import_34039 != 'pyd_module'):
        __import__(import_34039)
        sys_modules_34040 = sys.modules[import_34039]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', sys_modules_34040.module_type_store, module_type_store, ['captured_stdout'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_34040, sys_modules_34040.module_type_store, module_type_store)
    else:
        from test.test_support import captured_stdout

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', None, module_type_store, ['captured_stdout'], [captured_stdout])

else:
    # Assigning a type to the variable 'test.test_support' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', import_34039)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.ccompiler import gen_lib_options, CCompiler, get_default_compiler' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34041 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.ccompiler')

if (type(import_34041) is not StypyTypeError):

    if (import_34041 != 'pyd_module'):
        __import__(import_34041)
        sys_modules_34042 = sys.modules[import_34041]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.ccompiler', sys_modules_34042.module_type_store, module_type_store, ['gen_lib_options', 'CCompiler', 'get_default_compiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_34042, sys_modules_34042.module_type_store, module_type_store)
    else:
        from distutils.ccompiler import gen_lib_options, CCompiler, get_default_compiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.ccompiler', None, module_type_store, ['gen_lib_options', 'CCompiler', 'get_default_compiler'], [gen_lib_options, CCompiler, get_default_compiler])

else:
    # Assigning a type to the variable 'distutils.ccompiler' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.ccompiler', import_34041)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.sysconfig import customize_compiler' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34043 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.sysconfig')

if (type(import_34043) is not StypyTypeError):

    if (import_34043 != 'pyd_module'):
        __import__(import_34043)
        sys_modules_34044 = sys.modules[import_34043]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.sysconfig', sys_modules_34044.module_type_store, module_type_store, ['customize_compiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_34044, sys_modules_34044.module_type_store, module_type_store)
    else:
        from distutils.sysconfig import customize_compiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.sysconfig', None, module_type_store, ['customize_compiler'], [customize_compiler])

else:
    # Assigning a type to the variable 'distutils.sysconfig' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.sysconfig', import_34043)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils import debug' statement (line 9)
try:
    from distutils import debug

except:
    debug = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils', None, module_type_store, ['debug'], [debug])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.tests import support' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34045 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests')

if (type(import_34045) is not StypyTypeError):

    if (import_34045 != 'pyd_module'):
        __import__(import_34045)
        sys_modules_34046 = sys.modules[import_34045]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests', sys_modules_34046.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_34046, sys_modules_34046.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests', import_34045)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'FakeCompiler' class

class FakeCompiler(object, ):

    @norecursion
    def library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_dir_option'
        module_type_store = module_type_store.open_function_context('library_dir_option', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FakeCompiler.library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        FakeCompiler.library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FakeCompiler.library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeCompiler.library_dir_option.__dict__.__setitem__('stypy_function_name', 'FakeCompiler.library_dir_option')
        FakeCompiler.library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        FakeCompiler.library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        FakeCompiler.library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FakeCompiler.library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeCompiler.library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeCompiler.library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeCompiler.library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeCompiler.library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'library_dir_option', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'library_dir_option(...)' code ##################

        str_34047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'str', '-L')
        # Getting the type of 'dir' (line 14)
        dir_34048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 22), 'dir')
        # Applying the binary operator '+' (line 14)
        result_add_34049 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 15), '+', str_34047, dir_34048)
        
        # Assigning a type to the variable 'stypy_return_type' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'stypy_return_type', result_add_34049)
        
        # ################# End of 'library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_34050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34050)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_dir_option'
        return stypy_return_type_34050


    @norecursion
    def runtime_library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runtime_library_dir_option'
        module_type_store = module_type_store.open_function_context('runtime_library_dir_option', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FakeCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        FakeCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FakeCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_function_name', 'FakeCompiler.runtime_library_dir_option')
        FakeCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        FakeCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        FakeCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FakeCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeCompiler.runtime_library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'runtime_library_dir_option', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'runtime_library_dir_option(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_34051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        str_34052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 16), 'str', '-cool')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 15), list_34051, str_34052)
        # Adding element type (line 17)
        str_34053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'str', '-R')
        # Getting the type of 'dir' (line 17)
        dir_34054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 32), 'dir')
        # Applying the binary operator '+' (line 17)
        result_add_34055 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 25), '+', str_34053, dir_34054)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 15), list_34051, result_add_34055)
        
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type', list_34051)
        
        # ################# End of 'runtime_library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runtime_library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_34056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34056)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runtime_library_dir_option'
        return stypy_return_type_34056


    @norecursion
    def find_library_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_34057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 49), 'int')
        defaults = [int_34057]
        # Create a new context for function 'find_library_file'
        module_type_store = module_type_store.open_function_context('find_library_file', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FakeCompiler.find_library_file.__dict__.__setitem__('stypy_localization', localization)
        FakeCompiler.find_library_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FakeCompiler.find_library_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeCompiler.find_library_file.__dict__.__setitem__('stypy_function_name', 'FakeCompiler.find_library_file')
        FakeCompiler.find_library_file.__dict__.__setitem__('stypy_param_names_list', ['dirs', 'lib', 'debug'])
        FakeCompiler.find_library_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        FakeCompiler.find_library_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FakeCompiler.find_library_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeCompiler.find_library_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeCompiler.find_library_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeCompiler.find_library_file.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeCompiler.find_library_file', ['dirs', 'lib', 'debug'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_library_file', localization, ['dirs', 'lib', 'debug'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_library_file(...)' code ##################

        str_34058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'str', 'found')
        # Assigning a type to the variable 'stypy_return_type' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'stypy_return_type', str_34058)
        
        # ################# End of 'find_library_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_library_file' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_34059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34059)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_library_file'
        return stypy_return_type_34059


    @norecursion
    def library_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_option'
        module_type_store = module_type_store.open_function_context('library_option', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FakeCompiler.library_option.__dict__.__setitem__('stypy_localization', localization)
        FakeCompiler.library_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FakeCompiler.library_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeCompiler.library_option.__dict__.__setitem__('stypy_function_name', 'FakeCompiler.library_option')
        FakeCompiler.library_option.__dict__.__setitem__('stypy_param_names_list', ['lib'])
        FakeCompiler.library_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        FakeCompiler.library_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FakeCompiler.library_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeCompiler.library_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeCompiler.library_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeCompiler.library_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeCompiler.library_option', ['lib'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'library_option', localization, ['lib'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'library_option(...)' code ##################

        str_34060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'str', '-l')
        # Getting the type of 'lib' (line 23)
        lib_34061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 22), 'lib')
        # Applying the binary operator '+' (line 23)
        result_add_34062 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 15), '+', str_34060, lib_34061)
        
        # Assigning a type to the variable 'stypy_return_type' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', result_add_34062)
        
        # ################# End of 'library_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_option' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_34063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34063)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_option'
        return stypy_return_type_34063


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FakeCompiler' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'FakeCompiler', FakeCompiler)
# Declaration of the 'CCompilerTestCase' class
# Getting the type of 'support' (line 25)
support_34064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'support')
# Obtaining the member 'EnvironGuard' of a type (line 25)
EnvironGuard_34065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), support_34064, 'EnvironGuard')
# Getting the type of 'unittest' (line 25)
unittest_34066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 46), 'unittest')
# Obtaining the member 'TestCase' of a type (line 25)
TestCase_34067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 46), unittest_34066, 'TestCase')

class CCompilerTestCase(EnvironGuard_34065, TestCase_34067, ):

    @norecursion
    def test_gen_lib_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_gen_lib_options'
        module_type_store = module_type_store.open_function_context('test_gen_lib_options', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompilerTestCase.test_gen_lib_options.__dict__.__setitem__('stypy_localization', localization)
        CCompilerTestCase.test_gen_lib_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompilerTestCase.test_gen_lib_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompilerTestCase.test_gen_lib_options.__dict__.__setitem__('stypy_function_name', 'CCompilerTestCase.test_gen_lib_options')
        CCompilerTestCase.test_gen_lib_options.__dict__.__setitem__('stypy_param_names_list', [])
        CCompilerTestCase.test_gen_lib_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompilerTestCase.test_gen_lib_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompilerTestCase.test_gen_lib_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompilerTestCase.test_gen_lib_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompilerTestCase.test_gen_lib_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompilerTestCase.test_gen_lib_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompilerTestCase.test_gen_lib_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_gen_lib_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_gen_lib_options(...)' code ##################

        
        # Assigning a Call to a Name (line 28):
        
        # Call to FakeCompiler(...): (line 28)
        # Processing the call keyword arguments (line 28)
        kwargs_34069 = {}
        # Getting the type of 'FakeCompiler' (line 28)
        FakeCompiler_34068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'FakeCompiler', False)
        # Calling FakeCompiler(args, kwargs) (line 28)
        FakeCompiler_call_result_34070 = invoke(stypy.reporting.localization.Localization(__file__, 28, 19), FakeCompiler_34068, *[], **kwargs_34069)
        
        # Assigning a type to the variable 'compiler' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'compiler', FakeCompiler_call_result_34070)
        
        # Assigning a List to a Name (line 29):
        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_34071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        # Adding element type (line 29)
        str_34072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'str', 'lib1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 18), list_34071, str_34072)
        # Adding element type (line 29)
        str_34073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'str', 'lib2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 18), list_34071, str_34073)
        
        # Assigning a type to the variable 'libdirs' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'libdirs', list_34071)
        
        # Assigning a List to a Name (line 30):
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_34074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        str_34075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 22), 'str', 'runlib1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), list_34074, str_34075)
        
        # Assigning a type to the variable 'runlibdirs' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'runlibdirs', list_34074)
        
        # Assigning a List to a Name (line 31):
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_34076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        
        # Call to join(...): (line 31)
        # Processing the call arguments (line 31)
        str_34080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'str', 'dir')
        str_34081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'str', 'name')
        # Processing the call keyword arguments (line 31)
        kwargs_34082 = {}
        # Getting the type of 'os' (line 31)
        os_34077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 31)
        path_34078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), os_34077, 'path')
        # Obtaining the member 'join' of a type (line 31)
        join_34079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), path_34078, 'join')
        # Calling join(args, kwargs) (line 31)
        join_call_result_34083 = invoke(stypy.reporting.localization.Localization(__file__, 31, 16), join_34079, *[str_34080, str_34081], **kwargs_34082)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 15), list_34076, join_call_result_34083)
        # Adding element type (line 31)
        str_34084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 45), 'str', 'name2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 15), list_34076, str_34084)
        
        # Assigning a type to the variable 'libs' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'libs', list_34076)
        
        # Assigning a Call to a Name (line 33):
        
        # Call to gen_lib_options(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'compiler' (line 33)
        compiler_34086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 31), 'compiler', False)
        # Getting the type of 'libdirs' (line 33)
        libdirs_34087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 41), 'libdirs', False)
        # Getting the type of 'runlibdirs' (line 33)
        runlibdirs_34088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 50), 'runlibdirs', False)
        # Getting the type of 'libs' (line 33)
        libs_34089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 62), 'libs', False)
        # Processing the call keyword arguments (line 33)
        kwargs_34090 = {}
        # Getting the type of 'gen_lib_options' (line 33)
        gen_lib_options_34085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'gen_lib_options', False)
        # Calling gen_lib_options(args, kwargs) (line 33)
        gen_lib_options_call_result_34091 = invoke(stypy.reporting.localization.Localization(__file__, 33, 15), gen_lib_options_34085, *[compiler_34086, libdirs_34087, runlibdirs_34088, libs_34089], **kwargs_34090)
        
        # Assigning a type to the variable 'opts' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'opts', gen_lib_options_call_result_34091)
        
        # Assigning a List to a Name (line 34):
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_34092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        # Adding element type (line 34)
        str_34093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 18), 'str', '-Llib1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 17), list_34092, str_34093)
        # Adding element type (line 34)
        str_34094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 28), 'str', '-Llib2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 17), list_34092, str_34094)
        # Adding element type (line 34)
        str_34095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 38), 'str', '-cool')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 17), list_34092, str_34095)
        # Adding element type (line 34)
        str_34096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 47), 'str', '-Rrunlib1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 17), list_34092, str_34096)
        # Adding element type (line 34)
        str_34097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 60), 'str', 'found')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 17), list_34092, str_34097)
        # Adding element type (line 34)
        str_34098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 18), 'str', '-lname2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 17), list_34092, str_34098)
        
        # Assigning a type to the variable 'wanted' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'wanted', list_34092)
        
        # Call to assertEqual(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'opts' (line 36)
        opts_34101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'opts', False)
        # Getting the type of 'wanted' (line 36)
        wanted_34102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'wanted', False)
        # Processing the call keyword arguments (line 36)
        kwargs_34103 = {}
        # Getting the type of 'self' (line 36)
        self_34099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 36)
        assertEqual_34100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_34099, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 36)
        assertEqual_call_result_34104 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assertEqual_34100, *[opts_34101, wanted_34102], **kwargs_34103)
        
        
        # ################# End of 'test_gen_lib_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_gen_lib_options' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_34105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34105)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_gen_lib_options'
        return stypy_return_type_34105


    @norecursion
    def test_debug_print(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_debug_print'
        module_type_store = module_type_store.open_function_context('test_debug_print', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompilerTestCase.test_debug_print.__dict__.__setitem__('stypy_localization', localization)
        CCompilerTestCase.test_debug_print.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompilerTestCase.test_debug_print.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompilerTestCase.test_debug_print.__dict__.__setitem__('stypy_function_name', 'CCompilerTestCase.test_debug_print')
        CCompilerTestCase.test_debug_print.__dict__.__setitem__('stypy_param_names_list', [])
        CCompilerTestCase.test_debug_print.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompilerTestCase.test_debug_print.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompilerTestCase.test_debug_print.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompilerTestCase.test_debug_print.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompilerTestCase.test_debug_print.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompilerTestCase.test_debug_print.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompilerTestCase.test_debug_print', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_debug_print', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_debug_print(...)' code ##################

        # Declaration of the 'MyCCompiler' class
        # Getting the type of 'CCompiler' (line 40)
        CCompiler_34106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'CCompiler')

        class MyCCompiler(CCompiler_34106, ):
            
            # Assigning a Dict to a Name (line 41):
            
            # Obtaining an instance of the builtin type 'dict' (line 41)
            dict_34107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 41)
            
            # Assigning a type to the variable 'executables' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'executables', dict_34107)
        
        # Assigning a type to the variable 'MyCCompiler' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'MyCCompiler', MyCCompiler)
        
        # Assigning a Call to a Name (line 43):
        
        # Call to MyCCompiler(...): (line 43)
        # Processing the call keyword arguments (line 43)
        kwargs_34109 = {}
        # Getting the type of 'MyCCompiler' (line 43)
        MyCCompiler_34108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'MyCCompiler', False)
        # Calling MyCCompiler(args, kwargs) (line 43)
        MyCCompiler_call_result_34110 = invoke(stypy.reporting.localization.Localization(__file__, 43, 19), MyCCompiler_34108, *[], **kwargs_34109)
        
        # Assigning a type to the variable 'compiler' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'compiler', MyCCompiler_call_result_34110)
        
        # Call to captured_stdout(...): (line 44)
        # Processing the call keyword arguments (line 44)
        kwargs_34112 = {}
        # Getting the type of 'captured_stdout' (line 44)
        captured_stdout_34111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'captured_stdout', False)
        # Calling captured_stdout(args, kwargs) (line 44)
        captured_stdout_call_result_34113 = invoke(stypy.reporting.localization.Localization(__file__, 44, 13), captured_stdout_34111, *[], **kwargs_34112)
        
        with_34114 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 44, 13), captured_stdout_call_result_34113, 'with parameter', '__enter__', '__exit__')

        if with_34114:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 44)
            enter___34115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 13), captured_stdout_call_result_34113, '__enter__')
            with_enter_34116 = invoke(stypy.reporting.localization.Localization(__file__, 44, 13), enter___34115)
            # Assigning a type to the variable 'stdout' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'stdout', with_enter_34116)
            
            # Call to debug_print(...): (line 45)
            # Processing the call arguments (line 45)
            str_34119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 33), 'str', 'xxx')
            # Processing the call keyword arguments (line 45)
            kwargs_34120 = {}
            # Getting the type of 'compiler' (line 45)
            compiler_34117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'compiler', False)
            # Obtaining the member 'debug_print' of a type (line 45)
            debug_print_34118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), compiler_34117, 'debug_print')
            # Calling debug_print(args, kwargs) (line 45)
            debug_print_call_result_34121 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), debug_print_34118, *[str_34119], **kwargs_34120)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 44)
            exit___34122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 13), captured_stdout_call_result_34113, '__exit__')
            with_exit_34123 = invoke(stypy.reporting.localization.Localization(__file__, 44, 13), exit___34122, None, None, None)

        
        # Call to seek(...): (line 46)
        # Processing the call arguments (line 46)
        int_34126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'int')
        # Processing the call keyword arguments (line 46)
        kwargs_34127 = {}
        # Getting the type of 'stdout' (line 46)
        stdout_34124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'stdout', False)
        # Obtaining the member 'seek' of a type (line 46)
        seek_34125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), stdout_34124, 'seek')
        # Calling seek(args, kwargs) (line 46)
        seek_call_result_34128 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), seek_34125, *[int_34126], **kwargs_34127)
        
        
        # Call to assertEqual(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Call to read(...): (line 47)
        # Processing the call keyword arguments (line 47)
        kwargs_34133 = {}
        # Getting the type of 'stdout' (line 47)
        stdout_34131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'stdout', False)
        # Obtaining the member 'read' of a type (line 47)
        read_34132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 25), stdout_34131, 'read')
        # Calling read(args, kwargs) (line 47)
        read_call_result_34134 = invoke(stypy.reporting.localization.Localization(__file__, 47, 25), read_34132, *[], **kwargs_34133)
        
        str_34135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 40), 'str', '')
        # Processing the call keyword arguments (line 47)
        kwargs_34136 = {}
        # Getting the type of 'self' (line 47)
        self_34129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 47)
        assertEqual_34130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_34129, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 47)
        assertEqual_call_result_34137 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assertEqual_34130, *[read_call_result_34134, str_34135], **kwargs_34136)
        
        
        # Assigning a Name to a Attribute (line 49):
        # Getting the type of 'True' (line 49)
        True_34138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'True')
        # Getting the type of 'debug' (line 49)
        debug_34139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'debug')
        # Setting the type of the member 'DEBUG' of a type (line 49)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), debug_34139, 'DEBUG', True_34138)
        
        # Try-finally block (line 50)
        
        # Call to captured_stdout(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_34141 = {}
        # Getting the type of 'captured_stdout' (line 51)
        captured_stdout_34140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'captured_stdout', False)
        # Calling captured_stdout(args, kwargs) (line 51)
        captured_stdout_call_result_34142 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), captured_stdout_34140, *[], **kwargs_34141)
        
        with_34143 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 51, 17), captured_stdout_call_result_34142, 'with parameter', '__enter__', '__exit__')

        if with_34143:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 51)
            enter___34144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 17), captured_stdout_call_result_34142, '__enter__')
            with_enter_34145 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), enter___34144)
            # Assigning a type to the variable 'stdout' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'stdout', with_enter_34145)
            
            # Call to debug_print(...): (line 52)
            # Processing the call arguments (line 52)
            str_34148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 37), 'str', 'xxx')
            # Processing the call keyword arguments (line 52)
            kwargs_34149 = {}
            # Getting the type of 'compiler' (line 52)
            compiler_34146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'compiler', False)
            # Obtaining the member 'debug_print' of a type (line 52)
            debug_print_34147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), compiler_34146, 'debug_print')
            # Calling debug_print(args, kwargs) (line 52)
            debug_print_call_result_34150 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), debug_print_34147, *[str_34148], **kwargs_34149)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 51)
            exit___34151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 17), captured_stdout_call_result_34142, '__exit__')
            with_exit_34152 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), exit___34151, None, None, None)

        
        # Call to seek(...): (line 53)
        # Processing the call arguments (line 53)
        int_34155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 24), 'int')
        # Processing the call keyword arguments (line 53)
        kwargs_34156 = {}
        # Getting the type of 'stdout' (line 53)
        stdout_34153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'stdout', False)
        # Obtaining the member 'seek' of a type (line 53)
        seek_34154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 12), stdout_34153, 'seek')
        # Calling seek(args, kwargs) (line 53)
        seek_call_result_34157 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), seek_34154, *[int_34155], **kwargs_34156)
        
        
        # Call to assertEqual(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Call to read(...): (line 54)
        # Processing the call keyword arguments (line 54)
        kwargs_34162 = {}
        # Getting the type of 'stdout' (line 54)
        stdout_34160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'stdout', False)
        # Obtaining the member 'read' of a type (line 54)
        read_34161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 29), stdout_34160, 'read')
        # Calling read(args, kwargs) (line 54)
        read_call_result_34163 = invoke(stypy.reporting.localization.Localization(__file__, 54, 29), read_34161, *[], **kwargs_34162)
        
        str_34164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 44), 'str', 'xxx\n')
        # Processing the call keyword arguments (line 54)
        kwargs_34165 = {}
        # Getting the type of 'self' (line 54)
        self_34158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 54)
        assertEqual_34159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), self_34158, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 54)
        assertEqual_call_result_34166 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), assertEqual_34159, *[read_call_result_34163, str_34164], **kwargs_34165)
        
        
        # finally branch of the try-finally block (line 50)
        
        # Assigning a Name to a Attribute (line 56):
        # Getting the type of 'False' (line 56)
        False_34167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'False')
        # Getting the type of 'debug' (line 56)
        debug_34168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'debug')
        # Setting the type of the member 'DEBUG' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), debug_34168, 'DEBUG', False_34167)
        
        
        # ################# End of 'test_debug_print(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_debug_print' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_34169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34169)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_debug_print'
        return stypy_return_type_34169


    @norecursion
    def test_customize_compiler(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_customize_compiler'
        module_type_store = module_type_store.open_function_context('test_customize_compiler', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompilerTestCase.test_customize_compiler.__dict__.__setitem__('stypy_localization', localization)
        CCompilerTestCase.test_customize_compiler.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompilerTestCase.test_customize_compiler.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompilerTestCase.test_customize_compiler.__dict__.__setitem__('stypy_function_name', 'CCompilerTestCase.test_customize_compiler')
        CCompilerTestCase.test_customize_compiler.__dict__.__setitem__('stypy_param_names_list', [])
        CCompilerTestCase.test_customize_compiler.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompilerTestCase.test_customize_compiler.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompilerTestCase.test_customize_compiler.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompilerTestCase.test_customize_compiler.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompilerTestCase.test_customize_compiler.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompilerTestCase.test_customize_compiler.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompilerTestCase.test_customize_compiler', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_customize_compiler', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_customize_compiler(...)' code ##################

        
        # Assigning a Str to a Subscript (line 61):
        str_34170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 27), 'str', 'my_ar')
        # Getting the type of 'os' (line 61)
        os_34171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'os')
        # Obtaining the member 'environ' of a type (line 61)
        environ_34172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), os_34171, 'environ')
        str_34173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'str', 'AR')
        # Storing an element on a container (line 61)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 8), environ_34172, (str_34173, str_34170))
        
        # Assigning a Str to a Subscript (line 62):
        str_34174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 32), 'str', '-arflags')
        # Getting the type of 'os' (line 62)
        os_34175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'os')
        # Obtaining the member 'environ' of a type (line 62)
        environ_34176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), os_34175, 'environ')
        str_34177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 19), 'str', 'ARFLAGS')
        # Storing an element on a container (line 62)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 8), environ_34176, (str_34177, str_34174))
        # Declaration of the 'compiler' class

        class compiler:
            
            # Assigning a Str to a Name (line 66):
            str_34178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 28), 'str', 'unix')
            # Assigning a type to the variable 'compiler_type' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'compiler_type', str_34178)

            @norecursion
            def set_executables(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'set_executables'
                module_type_store = module_type_store.open_function_context('set_executables', 68, 12, False)
                # Assigning a type to the variable 'self' (line 69)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                compiler.set_executables.__dict__.__setitem__('stypy_localization', localization)
                compiler.set_executables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                compiler.set_executables.__dict__.__setitem__('stypy_type_store', module_type_store)
                compiler.set_executables.__dict__.__setitem__('stypy_function_name', 'compiler.set_executables')
                compiler.set_executables.__dict__.__setitem__('stypy_param_names_list', [])
                compiler.set_executables.__dict__.__setitem__('stypy_varargs_param_name', None)
                compiler.set_executables.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
                compiler.set_executables.__dict__.__setitem__('stypy_call_defaults', defaults)
                compiler.set_executables.__dict__.__setitem__('stypy_call_varargs', varargs)
                compiler.set_executables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                compiler.set_executables.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'compiler.set_executables', [], None, 'kw', defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'set_executables', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'set_executables(...)' code ##################

                
                # Assigning a Name to a Attribute (line 69):
                # Getting the type of 'kw' (line 69)
                kw_34179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 28), 'kw')
                # Getting the type of 'self' (line 69)
                self_34180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'self')
                # Setting the type of the member 'exes' of a type (line 69)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), self_34180, 'exes', kw_34179)
                
                # ################# End of 'set_executables(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'set_executables' in the type store
                # Getting the type of 'stypy_return_type' (line 68)
                stypy_return_type_34181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_34181)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'set_executables'
                return stypy_return_type_34181

        
        # Assigning a type to the variable 'compiler' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'compiler', compiler)
        
        # Assigning a Call to a Name (line 71):
        
        # Call to compiler(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_34183 = {}
        # Getting the type of 'compiler' (line 71)
        compiler_34182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'compiler', False)
        # Calling compiler(args, kwargs) (line 71)
        compiler_call_result_34184 = invoke(stypy.reporting.localization.Localization(__file__, 71, 15), compiler_34182, *[], **kwargs_34183)
        
        # Assigning a type to the variable 'comp' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'comp', compiler_call_result_34184)
        
        # Call to customize_compiler(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'comp' (line 72)
        comp_34186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'comp', False)
        # Processing the call keyword arguments (line 72)
        kwargs_34187 = {}
        # Getting the type of 'customize_compiler' (line 72)
        customize_compiler_34185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'customize_compiler', False)
        # Calling customize_compiler(args, kwargs) (line 72)
        customize_compiler_call_result_34188 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), customize_compiler_34185, *[comp_34186], **kwargs_34187)
        
        
        # Call to assertEqual(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Obtaining the type of the subscript
        str_34191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 35), 'str', 'archiver')
        # Getting the type of 'comp' (line 73)
        comp_34192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 25), 'comp', False)
        # Obtaining the member 'exes' of a type (line 73)
        exes_34193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 25), comp_34192, 'exes')
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___34194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 25), exes_34193, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_34195 = invoke(stypy.reporting.localization.Localization(__file__, 73, 25), getitem___34194, str_34191)
        
        str_34196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 48), 'str', 'my_ar -arflags')
        # Processing the call keyword arguments (line 73)
        kwargs_34197 = {}
        # Getting the type of 'self' (line 73)
        self_34189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 73)
        assertEqual_34190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_34189, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 73)
        assertEqual_call_result_34198 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), assertEqual_34190, *[subscript_call_result_34195, str_34196], **kwargs_34197)
        
        
        # ################# End of 'test_customize_compiler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_customize_compiler' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_34199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34199)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_customize_compiler'
        return stypy_return_type_34199


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 25, 0, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompilerTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CCompilerTestCase' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'CCompilerTestCase', CCompilerTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 75, 0, False)
    
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

    
    # Call to makeSuite(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'CCompilerTestCase' (line 76)
    CCompilerTestCase_34202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), 'CCompilerTestCase', False)
    # Processing the call keyword arguments (line 76)
    kwargs_34203 = {}
    # Getting the type of 'unittest' (line 76)
    unittest_34200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 76)
    makeSuite_34201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 11), unittest_34200, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 76)
    makeSuite_call_result_34204 = invoke(stypy.reporting.localization.Localization(__file__, 76, 11), makeSuite_34201, *[CCompilerTestCase_34202], **kwargs_34203)
    
    # Assigning a type to the variable 'stypy_return_type' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type', makeSuite_call_result_34204)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_34205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34205)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_34205

# Assigning a type to the variable 'test_suite' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 79)
    # Processing the call keyword arguments (line 79)
    str_34208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 30), 'str', 'test_suite')
    keyword_34209 = str_34208
    kwargs_34210 = {'defaultTest': keyword_34209}
    # Getting the type of 'unittest' (line 79)
    unittest_34206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'unittest', False)
    # Obtaining the member 'main' of a type (line 79)
    main_34207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 4), unittest_34206, 'main')
    # Calling main(args, kwargs) (line 79)
    main_call_result_34211 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), main_34207, *[], **kwargs_34210)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
