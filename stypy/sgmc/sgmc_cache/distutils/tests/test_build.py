
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.build.'''
2: import unittest
3: import os
4: import sys
5: from test.test_support import run_unittest
6: 
7: from distutils.command.build import build
8: from distutils.tests import support
9: from sysconfig import get_platform
10: 
11: class BuildTestCase(support.TempdirManager,
12:                     support.LoggingSilencer,
13:                     unittest.TestCase):
14: 
15:     def test_finalize_options(self):
16:         pkg_dir, dist = self.create_dist()
17:         cmd = build(dist)
18:         cmd.finalize_options()
19: 
20:         # if not specified, plat_name gets the current platform
21:         self.assertEqual(cmd.plat_name, get_platform())
22: 
23:         # build_purelib is build + lib
24:         wanted = os.path.join(cmd.build_base, 'lib')
25:         self.assertEqual(cmd.build_purelib, wanted)
26: 
27:         # build_platlib is 'build/lib.platform-x.x[-pydebug]'
28:         # examples:
29:         #   build/lib.macosx-10.3-i386-2.7
30:         plat_spec = '.%s-%s' % (cmd.plat_name, sys.version[0:3])
31:         if hasattr(sys, 'gettotalrefcount'):
32:             self.assertTrue(cmd.build_platlib.endswith('-pydebug'))
33:             plat_spec += '-pydebug'
34:         wanted = os.path.join(cmd.build_base, 'lib' + plat_spec)
35:         self.assertEqual(cmd.build_platlib, wanted)
36: 
37:         # by default, build_lib = build_purelib
38:         self.assertEqual(cmd.build_lib, cmd.build_purelib)
39: 
40:         # build_temp is build/temp.<plat>
41:         wanted = os.path.join(cmd.build_base, 'temp' + plat_spec)
42:         self.assertEqual(cmd.build_temp, wanted)
43: 
44:         # build_scripts is build/scripts-x.x
45:         wanted = os.path.join(cmd.build_base, 'scripts-' +  sys.version[0:3])
46:         self.assertEqual(cmd.build_scripts, wanted)
47: 
48:         # executable is os.path.normpath(sys.executable)
49:         self.assertEqual(cmd.executable, os.path.normpath(sys.executable))
50: 
51: def test_suite():
52:     return unittest.makeSuite(BuildTestCase)
53: 
54: if __name__ == "__main__":
55:     run_unittest(test_suite())
56: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_30889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.build.')
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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from test.test_support import run_unittest' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30890 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support')

if (type(import_30890) is not StypyTypeError):

    if (import_30890 != 'pyd_module'):
        __import__(import_30890)
        sys_modules_30891 = sys.modules[import_30890]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', sys_modules_30891.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_30891, sys_modules_30891.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', import_30890)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.command.build import build' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30892 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.build')

if (type(import_30892) is not StypyTypeError):

    if (import_30892 != 'pyd_module'):
        __import__(import_30892)
        sys_modules_30893 = sys.modules[import_30892]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.build', sys_modules_30893.module_type_store, module_type_store, ['build'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_30893, sys_modules_30893.module_type_store, module_type_store)
    else:
        from distutils.command.build import build

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.build', None, module_type_store, ['build'], [build])

else:
    # Assigning a type to the variable 'distutils.command.build' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.build', import_30892)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.tests import support' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_30894 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests')

if (type(import_30894) is not StypyTypeError):

    if (import_30894 != 'pyd_module'):
        __import__(import_30894)
        sys_modules_30895 = sys.modules[import_30894]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', sys_modules_30895.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_30895, sys_modules_30895.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', import_30894)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from sysconfig import get_platform' statement (line 9)
try:
    from sysconfig import get_platform

except:
    get_platform = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sysconfig', None, module_type_store, ['get_platform'], [get_platform])

# Declaration of the 'BuildTestCase' class
# Getting the type of 'support' (line 11)
support_30896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 20), 'support')
# Obtaining the member 'TempdirManager' of a type (line 11)
TempdirManager_30897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 20), support_30896, 'TempdirManager')
# Getting the type of 'support' (line 12)
support_30898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 12)
LoggingSilencer_30899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 20), support_30898, 'LoggingSilencer')
# Getting the type of 'unittest' (line 13)
unittest_30900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'unittest')
# Obtaining the member 'TestCase' of a type (line 13)
TestCase_30901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 20), unittest_30900, 'TestCase')

class BuildTestCase(TempdirManager_30897, LoggingSilencer_30899, TestCase_30901, ):

    @norecursion
    def test_finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_finalize_options'
        module_type_store = module_type_store.open_function_context('test_finalize_options', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildTestCase.test_finalize_options.__dict__.__setitem__('stypy_localization', localization)
        BuildTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildTestCase.test_finalize_options.__dict__.__setitem__('stypy_function_name', 'BuildTestCase.test_finalize_options')
        BuildTestCase.test_finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        BuildTestCase.test_finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildTestCase.test_finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildTestCase.test_finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildTestCase.test_finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 16):
        
        # Assigning a Subscript to a Name (line 16):
        
        # Obtaining the type of the subscript
        int_30902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'int')
        
        # Call to create_dist(...): (line 16)
        # Processing the call keyword arguments (line 16)
        kwargs_30905 = {}
        # Getting the type of 'self' (line 16)
        self_30903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 16)
        create_dist_30904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 24), self_30903, 'create_dist')
        # Calling create_dist(args, kwargs) (line 16)
        create_dist_call_result_30906 = invoke(stypy.reporting.localization.Localization(__file__, 16, 24), create_dist_30904, *[], **kwargs_30905)
        
        # Obtaining the member '__getitem__' of a type (line 16)
        getitem___30907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), create_dist_call_result_30906, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 16)
        subscript_call_result_30908 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), getitem___30907, int_30902)
        
        # Assigning a type to the variable 'tuple_var_assignment_30887' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_30887', subscript_call_result_30908)
        
        # Assigning a Subscript to a Name (line 16):
        
        # Obtaining the type of the subscript
        int_30909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'int')
        
        # Call to create_dist(...): (line 16)
        # Processing the call keyword arguments (line 16)
        kwargs_30912 = {}
        # Getting the type of 'self' (line 16)
        self_30910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 16)
        create_dist_30911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 24), self_30910, 'create_dist')
        # Calling create_dist(args, kwargs) (line 16)
        create_dist_call_result_30913 = invoke(stypy.reporting.localization.Localization(__file__, 16, 24), create_dist_30911, *[], **kwargs_30912)
        
        # Obtaining the member '__getitem__' of a type (line 16)
        getitem___30914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), create_dist_call_result_30913, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 16)
        subscript_call_result_30915 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), getitem___30914, int_30909)
        
        # Assigning a type to the variable 'tuple_var_assignment_30888' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_30888', subscript_call_result_30915)
        
        # Assigning a Name to a Name (line 16):
        # Getting the type of 'tuple_var_assignment_30887' (line 16)
        tuple_var_assignment_30887_30916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_30887')
        # Assigning a type to the variable 'pkg_dir' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'pkg_dir', tuple_var_assignment_30887_30916)
        
        # Assigning a Name to a Name (line 16):
        # Getting the type of 'tuple_var_assignment_30888' (line 16)
        tuple_var_assignment_30888_30917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_30888')
        # Assigning a type to the variable 'dist' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'dist', tuple_var_assignment_30888_30917)
        
        # Assigning a Call to a Name (line 17):
        
        # Assigning a Call to a Name (line 17):
        
        # Call to build(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'dist' (line 17)
        dist_30919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'dist', False)
        # Processing the call keyword arguments (line 17)
        kwargs_30920 = {}
        # Getting the type of 'build' (line 17)
        build_30918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'build', False)
        # Calling build(args, kwargs) (line 17)
        build_call_result_30921 = invoke(stypy.reporting.localization.Localization(__file__, 17, 14), build_30918, *[dist_30919], **kwargs_30920)
        
        # Assigning a type to the variable 'cmd' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'cmd', build_call_result_30921)
        
        # Call to finalize_options(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_30924 = {}
        # Getting the type of 'cmd' (line 18)
        cmd_30922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 18)
        finalize_options_30923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), cmd_30922, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 18)
        finalize_options_call_result_30925 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), finalize_options_30923, *[], **kwargs_30924)
        
        
        # Call to assertEqual(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'cmd' (line 21)
        cmd_30928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 25), 'cmd', False)
        # Obtaining the member 'plat_name' of a type (line 21)
        plat_name_30929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 25), cmd_30928, 'plat_name')
        
        # Call to get_platform(...): (line 21)
        # Processing the call keyword arguments (line 21)
        kwargs_30931 = {}
        # Getting the type of 'get_platform' (line 21)
        get_platform_30930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 40), 'get_platform', False)
        # Calling get_platform(args, kwargs) (line 21)
        get_platform_call_result_30932 = invoke(stypy.reporting.localization.Localization(__file__, 21, 40), get_platform_30930, *[], **kwargs_30931)
        
        # Processing the call keyword arguments (line 21)
        kwargs_30933 = {}
        # Getting the type of 'self' (line 21)
        self_30926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 21)
        assertEqual_30927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_30926, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 21)
        assertEqual_call_result_30934 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assertEqual_30927, *[plat_name_30929, get_platform_call_result_30932], **kwargs_30933)
        
        
        # Assigning a Call to a Name (line 24):
        
        # Assigning a Call to a Name (line 24):
        
        # Call to join(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'cmd' (line 24)
        cmd_30938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 30), 'cmd', False)
        # Obtaining the member 'build_base' of a type (line 24)
        build_base_30939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 30), cmd_30938, 'build_base')
        str_30940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 46), 'str', 'lib')
        # Processing the call keyword arguments (line 24)
        kwargs_30941 = {}
        # Getting the type of 'os' (line 24)
        os_30935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 24)
        path_30936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 17), os_30935, 'path')
        # Obtaining the member 'join' of a type (line 24)
        join_30937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 17), path_30936, 'join')
        # Calling join(args, kwargs) (line 24)
        join_call_result_30942 = invoke(stypy.reporting.localization.Localization(__file__, 24, 17), join_30937, *[build_base_30939, str_30940], **kwargs_30941)
        
        # Assigning a type to the variable 'wanted' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'wanted', join_call_result_30942)
        
        # Call to assertEqual(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'cmd' (line 25)
        cmd_30945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'cmd', False)
        # Obtaining the member 'build_purelib' of a type (line 25)
        build_purelib_30946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 25), cmd_30945, 'build_purelib')
        # Getting the type of 'wanted' (line 25)
        wanted_30947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 44), 'wanted', False)
        # Processing the call keyword arguments (line 25)
        kwargs_30948 = {}
        # Getting the type of 'self' (line 25)
        self_30943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 25)
        assertEqual_30944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_30943, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 25)
        assertEqual_call_result_30949 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), assertEqual_30944, *[build_purelib_30946, wanted_30947], **kwargs_30948)
        
        
        # Assigning a BinOp to a Name (line 30):
        
        # Assigning a BinOp to a Name (line 30):
        str_30950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 20), 'str', '.%s-%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 30)
        tuple_30951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 30)
        # Adding element type (line 30)
        # Getting the type of 'cmd' (line 30)
        cmd_30952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 32), 'cmd')
        # Obtaining the member 'plat_name' of a type (line 30)
        plat_name_30953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 32), cmd_30952, 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 32), tuple_30951, plat_name_30953)
        # Adding element type (line 30)
        
        # Obtaining the type of the subscript
        int_30954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 59), 'int')
        int_30955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 61), 'int')
        slice_30956 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 30, 47), int_30954, int_30955, None)
        # Getting the type of 'sys' (line 30)
        sys_30957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 47), 'sys')
        # Obtaining the member 'version' of a type (line 30)
        version_30958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 47), sys_30957, 'version')
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___30959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 47), version_30958, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_30960 = invoke(stypy.reporting.localization.Localization(__file__, 30, 47), getitem___30959, slice_30956)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 32), tuple_30951, subscript_call_result_30960)
        
        # Applying the binary operator '%' (line 30)
        result_mod_30961 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 20), '%', str_30950, tuple_30951)
        
        # Assigning a type to the variable 'plat_spec' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'plat_spec', result_mod_30961)
        
        # Type idiom detected: calculating its left and rigth part (line 31)
        str_30962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 24), 'str', 'gettotalrefcount')
        # Getting the type of 'sys' (line 31)
        sys_30963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'sys')
        
        (may_be_30964, more_types_in_union_30965) = may_provide_member(str_30962, sys_30963)

        if may_be_30964:

            if more_types_in_union_30965:
                # Runtime conditional SSA (line 31)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'sys' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'sys', remove_not_member_provider_from_union(sys_30963, 'gettotalrefcount'))
            
            # Call to assertTrue(...): (line 32)
            # Processing the call arguments (line 32)
            
            # Call to endswith(...): (line 32)
            # Processing the call arguments (line 32)
            str_30971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 55), 'str', '-pydebug')
            # Processing the call keyword arguments (line 32)
            kwargs_30972 = {}
            # Getting the type of 'cmd' (line 32)
            cmd_30968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 28), 'cmd', False)
            # Obtaining the member 'build_platlib' of a type (line 32)
            build_platlib_30969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 28), cmd_30968, 'build_platlib')
            # Obtaining the member 'endswith' of a type (line 32)
            endswith_30970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 28), build_platlib_30969, 'endswith')
            # Calling endswith(args, kwargs) (line 32)
            endswith_call_result_30973 = invoke(stypy.reporting.localization.Localization(__file__, 32, 28), endswith_30970, *[str_30971], **kwargs_30972)
            
            # Processing the call keyword arguments (line 32)
            kwargs_30974 = {}
            # Getting the type of 'self' (line 32)
            self_30966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'self', False)
            # Obtaining the member 'assertTrue' of a type (line 32)
            assertTrue_30967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), self_30966, 'assertTrue')
            # Calling assertTrue(args, kwargs) (line 32)
            assertTrue_call_result_30975 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), assertTrue_30967, *[endswith_call_result_30973], **kwargs_30974)
            
            
            # Getting the type of 'plat_spec' (line 33)
            plat_spec_30976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'plat_spec')
            str_30977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'str', '-pydebug')
            # Applying the binary operator '+=' (line 33)
            result_iadd_30978 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 12), '+=', plat_spec_30976, str_30977)
            # Assigning a type to the variable 'plat_spec' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'plat_spec', result_iadd_30978)
            

            if more_types_in_union_30965:
                # SSA join for if statement (line 31)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 34):
        
        # Assigning a Call to a Name (line 34):
        
        # Call to join(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'cmd' (line 34)
        cmd_30982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'cmd', False)
        # Obtaining the member 'build_base' of a type (line 34)
        build_base_30983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 30), cmd_30982, 'build_base')
        str_30984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 46), 'str', 'lib')
        # Getting the type of 'plat_spec' (line 34)
        plat_spec_30985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 54), 'plat_spec', False)
        # Applying the binary operator '+' (line 34)
        result_add_30986 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 46), '+', str_30984, plat_spec_30985)
        
        # Processing the call keyword arguments (line 34)
        kwargs_30987 = {}
        # Getting the type of 'os' (line 34)
        os_30979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 34)
        path_30980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 17), os_30979, 'path')
        # Obtaining the member 'join' of a type (line 34)
        join_30981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 17), path_30980, 'join')
        # Calling join(args, kwargs) (line 34)
        join_call_result_30988 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), join_30981, *[build_base_30983, result_add_30986], **kwargs_30987)
        
        # Assigning a type to the variable 'wanted' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'wanted', join_call_result_30988)
        
        # Call to assertEqual(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'cmd' (line 35)
        cmd_30991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 'cmd', False)
        # Obtaining the member 'build_platlib' of a type (line 35)
        build_platlib_30992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 25), cmd_30991, 'build_platlib')
        # Getting the type of 'wanted' (line 35)
        wanted_30993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 44), 'wanted', False)
        # Processing the call keyword arguments (line 35)
        kwargs_30994 = {}
        # Getting the type of 'self' (line 35)
        self_30989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 35)
        assertEqual_30990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_30989, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 35)
        assertEqual_call_result_30995 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), assertEqual_30990, *[build_platlib_30992, wanted_30993], **kwargs_30994)
        
        
        # Call to assertEqual(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'cmd' (line 38)
        cmd_30998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'cmd', False)
        # Obtaining the member 'build_lib' of a type (line 38)
        build_lib_30999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 25), cmd_30998, 'build_lib')
        # Getting the type of 'cmd' (line 38)
        cmd_31000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 40), 'cmd', False)
        # Obtaining the member 'build_purelib' of a type (line 38)
        build_purelib_31001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 40), cmd_31000, 'build_purelib')
        # Processing the call keyword arguments (line 38)
        kwargs_31002 = {}
        # Getting the type of 'self' (line 38)
        self_30996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 38)
        assertEqual_30997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_30996, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 38)
        assertEqual_call_result_31003 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), assertEqual_30997, *[build_lib_30999, build_purelib_31001], **kwargs_31002)
        
        
        # Assigning a Call to a Name (line 41):
        
        # Assigning a Call to a Name (line 41):
        
        # Call to join(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'cmd' (line 41)
        cmd_31007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 30), 'cmd', False)
        # Obtaining the member 'build_base' of a type (line 41)
        build_base_31008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 30), cmd_31007, 'build_base')
        str_31009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 46), 'str', 'temp')
        # Getting the type of 'plat_spec' (line 41)
        plat_spec_31010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 55), 'plat_spec', False)
        # Applying the binary operator '+' (line 41)
        result_add_31011 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 46), '+', str_31009, plat_spec_31010)
        
        # Processing the call keyword arguments (line 41)
        kwargs_31012 = {}
        # Getting the type of 'os' (line 41)
        os_31004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 41)
        path_31005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 17), os_31004, 'path')
        # Obtaining the member 'join' of a type (line 41)
        join_31006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 17), path_31005, 'join')
        # Calling join(args, kwargs) (line 41)
        join_call_result_31013 = invoke(stypy.reporting.localization.Localization(__file__, 41, 17), join_31006, *[build_base_31008, result_add_31011], **kwargs_31012)
        
        # Assigning a type to the variable 'wanted' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'wanted', join_call_result_31013)
        
        # Call to assertEqual(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'cmd' (line 42)
        cmd_31016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 25), 'cmd', False)
        # Obtaining the member 'build_temp' of a type (line 42)
        build_temp_31017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 25), cmd_31016, 'build_temp')
        # Getting the type of 'wanted' (line 42)
        wanted_31018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 41), 'wanted', False)
        # Processing the call keyword arguments (line 42)
        kwargs_31019 = {}
        # Getting the type of 'self' (line 42)
        self_31014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 42)
        assertEqual_31015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_31014, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 42)
        assertEqual_call_result_31020 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assertEqual_31015, *[build_temp_31017, wanted_31018], **kwargs_31019)
        
        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to join(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'cmd' (line 45)
        cmd_31024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 30), 'cmd', False)
        # Obtaining the member 'build_base' of a type (line 45)
        build_base_31025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 30), cmd_31024, 'build_base')
        str_31026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 46), 'str', 'scripts-')
        
        # Obtaining the type of the subscript
        int_31027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 72), 'int')
        int_31028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 74), 'int')
        slice_31029 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 45, 60), int_31027, int_31028, None)
        # Getting the type of 'sys' (line 45)
        sys_31030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 60), 'sys', False)
        # Obtaining the member 'version' of a type (line 45)
        version_31031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 60), sys_31030, 'version')
        # Obtaining the member '__getitem__' of a type (line 45)
        getitem___31032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 60), version_31031, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 45)
        subscript_call_result_31033 = invoke(stypy.reporting.localization.Localization(__file__, 45, 60), getitem___31032, slice_31029)
        
        # Applying the binary operator '+' (line 45)
        result_add_31034 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 46), '+', str_31026, subscript_call_result_31033)
        
        # Processing the call keyword arguments (line 45)
        kwargs_31035 = {}
        # Getting the type of 'os' (line 45)
        os_31021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 45)
        path_31022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 17), os_31021, 'path')
        # Obtaining the member 'join' of a type (line 45)
        join_31023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 17), path_31022, 'join')
        # Calling join(args, kwargs) (line 45)
        join_call_result_31036 = invoke(stypy.reporting.localization.Localization(__file__, 45, 17), join_31023, *[build_base_31025, result_add_31034], **kwargs_31035)
        
        # Assigning a type to the variable 'wanted' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'wanted', join_call_result_31036)
        
        # Call to assertEqual(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'cmd' (line 46)
        cmd_31039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'cmd', False)
        # Obtaining the member 'build_scripts' of a type (line 46)
        build_scripts_31040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 25), cmd_31039, 'build_scripts')
        # Getting the type of 'wanted' (line 46)
        wanted_31041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 44), 'wanted', False)
        # Processing the call keyword arguments (line 46)
        kwargs_31042 = {}
        # Getting the type of 'self' (line 46)
        self_31037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 46)
        assertEqual_31038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_31037, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 46)
        assertEqual_call_result_31043 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), assertEqual_31038, *[build_scripts_31040, wanted_31041], **kwargs_31042)
        
        
        # Call to assertEqual(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'cmd' (line 49)
        cmd_31046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'cmd', False)
        # Obtaining the member 'executable' of a type (line 49)
        executable_31047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 25), cmd_31046, 'executable')
        
        # Call to normpath(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'sys' (line 49)
        sys_31051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 58), 'sys', False)
        # Obtaining the member 'executable' of a type (line 49)
        executable_31052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 58), sys_31051, 'executable')
        # Processing the call keyword arguments (line 49)
        kwargs_31053 = {}
        # Getting the type of 'os' (line 49)
        os_31048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 41), 'os', False)
        # Obtaining the member 'path' of a type (line 49)
        path_31049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 41), os_31048, 'path')
        # Obtaining the member 'normpath' of a type (line 49)
        normpath_31050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 41), path_31049, 'normpath')
        # Calling normpath(args, kwargs) (line 49)
        normpath_call_result_31054 = invoke(stypy.reporting.localization.Localization(__file__, 49, 41), normpath_31050, *[executable_31052], **kwargs_31053)
        
        # Processing the call keyword arguments (line 49)
        kwargs_31055 = {}
        # Getting the type of 'self' (line 49)
        self_31044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 49)
        assertEqual_31045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_31044, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 49)
        assertEqual_call_result_31056 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), assertEqual_31045, *[executable_31047, normpath_call_result_31054], **kwargs_31055)
        
        
        # ################# End of 'test_finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_31057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_31057)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_finalize_options'
        return stypy_return_type_31057


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BuildTestCase' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'BuildTestCase', BuildTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 51, 0, False)
    
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

    
    # Call to makeSuite(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'BuildTestCase' (line 52)
    BuildTestCase_31060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 30), 'BuildTestCase', False)
    # Processing the call keyword arguments (line 52)
    kwargs_31061 = {}
    # Getting the type of 'unittest' (line 52)
    unittest_31058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 52)
    makeSuite_31059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 11), unittest_31058, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 52)
    makeSuite_call_result_31062 = invoke(stypy.reporting.localization.Localization(__file__, 52, 11), makeSuite_31059, *[BuildTestCase_31060], **kwargs_31061)
    
    # Assigning a type to the variable 'stypy_return_type' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type', makeSuite_call_result_31062)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_31063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31063)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_31063

# Assigning a type to the variable 'test_suite' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 55)
    # Processing the call arguments (line 55)
    
    # Call to test_suite(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_31066 = {}
    # Getting the type of 'test_suite' (line 55)
    test_suite_31065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 55)
    test_suite_call_result_31067 = invoke(stypy.reporting.localization.Localization(__file__, 55, 17), test_suite_31065, *[], **kwargs_31066)
    
    # Processing the call keyword arguments (line 55)
    kwargs_31068 = {}
    # Getting the type of 'run_unittest' (line 55)
    run_unittest_31064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 55)
    run_unittest_call_result_31069 = invoke(stypy.reporting.localization.Localization(__file__, 55, 4), run_unittest_31064, *[test_suite_call_result_31067], **kwargs_31068)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
