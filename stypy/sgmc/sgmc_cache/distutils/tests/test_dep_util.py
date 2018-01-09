
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.dep_util.'''
2: import unittest
3: import os
4: import time
5: 
6: from distutils.dep_util import newer, newer_pairwise, newer_group
7: from distutils.errors import DistutilsFileError
8: from distutils.tests import support
9: from test.test_support import run_unittest
10: 
11: class DepUtilTestCase(support.TempdirManager, unittest.TestCase):
12: 
13:     def test_newer(self):
14: 
15:         tmpdir = self.mkdtemp()
16:         new_file = os.path.join(tmpdir, 'new')
17:         old_file = os.path.abspath(__file__)
18: 
19:         # Raise DistutilsFileError if 'new_file' does not exist.
20:         self.assertRaises(DistutilsFileError, newer, new_file, old_file)
21: 
22:         # Return true if 'new_file' exists and is more recently modified than
23:         # 'old_file', or if 'new_file' exists and 'old_file' doesn't.
24:         self.write_file(new_file)
25:         self.assertTrue(newer(new_file, 'I_dont_exist'))
26:         self.assertTrue(newer(new_file, old_file))
27: 
28:         # Return false if both exist and 'old_file' is the same age or younger
29:         # than 'new_file'.
30:         self.assertFalse(newer(old_file, new_file))
31: 
32:     def test_newer_pairwise(self):
33:         tmpdir = self.mkdtemp()
34:         sources = os.path.join(tmpdir, 'sources')
35:         targets = os.path.join(tmpdir, 'targets')
36:         os.mkdir(sources)
37:         os.mkdir(targets)
38:         one = os.path.join(sources, 'one')
39:         two = os.path.join(sources, 'two')
40:         three = os.path.abspath(__file__)    # I am the old file
41:         four = os.path.join(targets, 'four')
42:         self.write_file(one)
43:         self.write_file(two)
44:         self.write_file(four)
45: 
46:         self.assertEqual(newer_pairwise([one, two], [three, four]),
47:                          ([one],[three]))
48: 
49:     def test_newer_group(self):
50:         tmpdir = self.mkdtemp()
51:         sources = os.path.join(tmpdir, 'sources')
52:         os.mkdir(sources)
53:         one = os.path.join(sources, 'one')
54:         two = os.path.join(sources, 'two')
55:         three = os.path.join(sources, 'three')
56:         old_file = os.path.abspath(__file__)
57: 
58:         # return true if 'old_file' is out-of-date with respect to any file
59:         # listed in 'sources'.
60:         self.write_file(one)
61:         self.write_file(two)
62:         self.write_file(three)
63:         self.assertTrue(newer_group([one, two, three], old_file))
64:         self.assertFalse(newer_group([one, two, old_file], three))
65: 
66:         # missing handling
67:         os.remove(one)
68:         self.assertRaises(OSError, newer_group, [one, two, old_file], three)
69: 
70:         self.assertFalse(newer_group([one, two, old_file], three,
71:                                      missing='ignore'))
72: 
73:         self.assertTrue(newer_group([one, two, old_file], three,
74:                                     missing='newer'))
75: 
76: 
77: def test_suite():
78:     return unittest.makeSuite(DepUtilTestCase)
79: 
80: if __name__ == "__main__":
81:     run_unittest(test_suite())
82: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_35925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.dep_util.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import unittest' statement (line 2)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import time' statement (line 4)
import time

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.dep_util import newer, newer_pairwise, newer_group' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35926 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.dep_util')

if (type(import_35926) is not StypyTypeError):

    if (import_35926 != 'pyd_module'):
        __import__(import_35926)
        sys_modules_35927 = sys.modules[import_35926]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.dep_util', sys_modules_35927.module_type_store, module_type_store, ['newer', 'newer_pairwise', 'newer_group'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_35927, sys_modules_35927.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer, newer_pairwise, newer_group

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.dep_util', None, module_type_store, ['newer', 'newer_pairwise', 'newer_group'], [newer, newer_pairwise, newer_group])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.dep_util', import_35926)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.errors import DistutilsFileError' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35928 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.errors')

if (type(import_35928) is not StypyTypeError):

    if (import_35928 != 'pyd_module'):
        __import__(import_35928)
        sys_modules_35929 = sys.modules[import_35928]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.errors', sys_modules_35929.module_type_store, module_type_store, ['DistutilsFileError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_35929, sys_modules_35929.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsFileError

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.errors', None, module_type_store, ['DistutilsFileError'], [DistutilsFileError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.errors', import_35928)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.tests import support' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35930 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests')

if (type(import_35930) is not StypyTypeError):

    if (import_35930 != 'pyd_module'):
        __import__(import_35930)
        sys_modules_35931 = sys.modules[import_35930]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', sys_modules_35931.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_35931, sys_modules_35931.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', import_35930)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from test.test_support import run_unittest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35932 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support')

if (type(import_35932) is not StypyTypeError):

    if (import_35932 != 'pyd_module'):
        __import__(import_35932)
        sys_modules_35933 = sys.modules[import_35932]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', sys_modules_35933.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_35933, sys_modules_35933.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', import_35932)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'DepUtilTestCase' class
# Getting the type of 'support' (line 11)
support_35934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 22), 'support')
# Obtaining the member 'TempdirManager' of a type (line 11)
TempdirManager_35935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 22), support_35934, 'TempdirManager')
# Getting the type of 'unittest' (line 11)
unittest_35936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 46), 'unittest')
# Obtaining the member 'TestCase' of a type (line 11)
TestCase_35937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 46), unittest_35936, 'TestCase')

class DepUtilTestCase(TempdirManager_35935, TestCase_35937, ):

    @norecursion
    def test_newer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_newer'
        module_type_store = module_type_store.open_function_context('test_newer', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DepUtilTestCase.test_newer.__dict__.__setitem__('stypy_localization', localization)
        DepUtilTestCase.test_newer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DepUtilTestCase.test_newer.__dict__.__setitem__('stypy_type_store', module_type_store)
        DepUtilTestCase.test_newer.__dict__.__setitem__('stypy_function_name', 'DepUtilTestCase.test_newer')
        DepUtilTestCase.test_newer.__dict__.__setitem__('stypy_param_names_list', [])
        DepUtilTestCase.test_newer.__dict__.__setitem__('stypy_varargs_param_name', None)
        DepUtilTestCase.test_newer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DepUtilTestCase.test_newer.__dict__.__setitem__('stypy_call_defaults', defaults)
        DepUtilTestCase.test_newer.__dict__.__setitem__('stypy_call_varargs', varargs)
        DepUtilTestCase.test_newer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DepUtilTestCase.test_newer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DepUtilTestCase.test_newer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_newer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_newer(...)' code ##################

        
        # Assigning a Call to a Name (line 15):
        
        # Call to mkdtemp(...): (line 15)
        # Processing the call keyword arguments (line 15)
        kwargs_35940 = {}
        # Getting the type of 'self' (line 15)
        self_35938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 15)
        mkdtemp_35939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 17), self_35938, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 15)
        mkdtemp_call_result_35941 = invoke(stypy.reporting.localization.Localization(__file__, 15, 17), mkdtemp_35939, *[], **kwargs_35940)
        
        # Assigning a type to the variable 'tmpdir' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'tmpdir', mkdtemp_call_result_35941)
        
        # Assigning a Call to a Name (line 16):
        
        # Call to join(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'tmpdir' (line 16)
        tmpdir_35945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 32), 'tmpdir', False)
        str_35946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 40), 'str', 'new')
        # Processing the call keyword arguments (line 16)
        kwargs_35947 = {}
        # Getting the type of 'os' (line 16)
        os_35942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 16)
        path_35943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 19), os_35942, 'path')
        # Obtaining the member 'join' of a type (line 16)
        join_35944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 19), path_35943, 'join')
        # Calling join(args, kwargs) (line 16)
        join_call_result_35948 = invoke(stypy.reporting.localization.Localization(__file__, 16, 19), join_35944, *[tmpdir_35945, str_35946], **kwargs_35947)
        
        # Assigning a type to the variable 'new_file' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'new_file', join_call_result_35948)
        
        # Assigning a Call to a Name (line 17):
        
        # Call to abspath(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of '__file__' (line 17)
        file___35952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 35), '__file__', False)
        # Processing the call keyword arguments (line 17)
        kwargs_35953 = {}
        # Getting the type of 'os' (line 17)
        os_35949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 17)
        path_35950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 19), os_35949, 'path')
        # Obtaining the member 'abspath' of a type (line 17)
        abspath_35951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 19), path_35950, 'abspath')
        # Calling abspath(args, kwargs) (line 17)
        abspath_call_result_35954 = invoke(stypy.reporting.localization.Localization(__file__, 17, 19), abspath_35951, *[file___35952], **kwargs_35953)
        
        # Assigning a type to the variable 'old_file' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'old_file', abspath_call_result_35954)
        
        # Call to assertRaises(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'DistutilsFileError' (line 20)
        DistutilsFileError_35957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 26), 'DistutilsFileError', False)
        # Getting the type of 'newer' (line 20)
        newer_35958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 46), 'newer', False)
        # Getting the type of 'new_file' (line 20)
        new_file_35959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 53), 'new_file', False)
        # Getting the type of 'old_file' (line 20)
        old_file_35960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 63), 'old_file', False)
        # Processing the call keyword arguments (line 20)
        kwargs_35961 = {}
        # Getting the type of 'self' (line 20)
        self_35955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 20)
        assertRaises_35956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_35955, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 20)
        assertRaises_call_result_35962 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), assertRaises_35956, *[DistutilsFileError_35957, newer_35958, new_file_35959, old_file_35960], **kwargs_35961)
        
        
        # Call to write_file(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'new_file' (line 24)
        new_file_35965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'new_file', False)
        # Processing the call keyword arguments (line 24)
        kwargs_35966 = {}
        # Getting the type of 'self' (line 24)
        self_35963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 24)
        write_file_35964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_35963, 'write_file')
        # Calling write_file(args, kwargs) (line 24)
        write_file_call_result_35967 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), write_file_35964, *[new_file_35965], **kwargs_35966)
        
        
        # Call to assertTrue(...): (line 25)
        # Processing the call arguments (line 25)
        
        # Call to newer(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'new_file' (line 25)
        new_file_35971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 30), 'new_file', False)
        str_35972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 40), 'str', 'I_dont_exist')
        # Processing the call keyword arguments (line 25)
        kwargs_35973 = {}
        # Getting the type of 'newer' (line 25)
        newer_35970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'newer', False)
        # Calling newer(args, kwargs) (line 25)
        newer_call_result_35974 = invoke(stypy.reporting.localization.Localization(__file__, 25, 24), newer_35970, *[new_file_35971, str_35972], **kwargs_35973)
        
        # Processing the call keyword arguments (line 25)
        kwargs_35975 = {}
        # Getting the type of 'self' (line 25)
        self_35968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 25)
        assertTrue_35969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_35968, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 25)
        assertTrue_call_result_35976 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), assertTrue_35969, *[newer_call_result_35974], **kwargs_35975)
        
        
        # Call to assertTrue(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Call to newer(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'new_file' (line 26)
        new_file_35980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 30), 'new_file', False)
        # Getting the type of 'old_file' (line 26)
        old_file_35981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 40), 'old_file', False)
        # Processing the call keyword arguments (line 26)
        kwargs_35982 = {}
        # Getting the type of 'newer' (line 26)
        newer_35979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 24), 'newer', False)
        # Calling newer(args, kwargs) (line 26)
        newer_call_result_35983 = invoke(stypy.reporting.localization.Localization(__file__, 26, 24), newer_35979, *[new_file_35980, old_file_35981], **kwargs_35982)
        
        # Processing the call keyword arguments (line 26)
        kwargs_35984 = {}
        # Getting the type of 'self' (line 26)
        self_35977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 26)
        assertTrue_35978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_35977, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 26)
        assertTrue_call_result_35985 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), assertTrue_35978, *[newer_call_result_35983], **kwargs_35984)
        
        
        # Call to assertFalse(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Call to newer(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'old_file' (line 30)
        old_file_35989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 31), 'old_file', False)
        # Getting the type of 'new_file' (line 30)
        new_file_35990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 41), 'new_file', False)
        # Processing the call keyword arguments (line 30)
        kwargs_35991 = {}
        # Getting the type of 'newer' (line 30)
        newer_35988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'newer', False)
        # Calling newer(args, kwargs) (line 30)
        newer_call_result_35992 = invoke(stypy.reporting.localization.Localization(__file__, 30, 25), newer_35988, *[old_file_35989, new_file_35990], **kwargs_35991)
        
        # Processing the call keyword arguments (line 30)
        kwargs_35993 = {}
        # Getting the type of 'self' (line 30)
        self_35986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 30)
        assertFalse_35987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_35986, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 30)
        assertFalse_call_result_35994 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assertFalse_35987, *[newer_call_result_35992], **kwargs_35993)
        
        
        # ################# End of 'test_newer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_newer' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_35995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35995)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_newer'
        return stypy_return_type_35995


    @norecursion
    def test_newer_pairwise(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_newer_pairwise'
        module_type_store = module_type_store.open_function_context('test_newer_pairwise', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DepUtilTestCase.test_newer_pairwise.__dict__.__setitem__('stypy_localization', localization)
        DepUtilTestCase.test_newer_pairwise.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DepUtilTestCase.test_newer_pairwise.__dict__.__setitem__('stypy_type_store', module_type_store)
        DepUtilTestCase.test_newer_pairwise.__dict__.__setitem__('stypy_function_name', 'DepUtilTestCase.test_newer_pairwise')
        DepUtilTestCase.test_newer_pairwise.__dict__.__setitem__('stypy_param_names_list', [])
        DepUtilTestCase.test_newer_pairwise.__dict__.__setitem__('stypy_varargs_param_name', None)
        DepUtilTestCase.test_newer_pairwise.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DepUtilTestCase.test_newer_pairwise.__dict__.__setitem__('stypy_call_defaults', defaults)
        DepUtilTestCase.test_newer_pairwise.__dict__.__setitem__('stypy_call_varargs', varargs)
        DepUtilTestCase.test_newer_pairwise.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DepUtilTestCase.test_newer_pairwise.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DepUtilTestCase.test_newer_pairwise', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_newer_pairwise', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_newer_pairwise(...)' code ##################

        
        # Assigning a Call to a Name (line 33):
        
        # Call to mkdtemp(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_35998 = {}
        # Getting the type of 'self' (line 33)
        self_35996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 33)
        mkdtemp_35997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 17), self_35996, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 33)
        mkdtemp_call_result_35999 = invoke(stypy.reporting.localization.Localization(__file__, 33, 17), mkdtemp_35997, *[], **kwargs_35998)
        
        # Assigning a type to the variable 'tmpdir' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'tmpdir', mkdtemp_call_result_35999)
        
        # Assigning a Call to a Name (line 34):
        
        # Call to join(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'tmpdir' (line 34)
        tmpdir_36003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 31), 'tmpdir', False)
        str_36004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 39), 'str', 'sources')
        # Processing the call keyword arguments (line 34)
        kwargs_36005 = {}
        # Getting the type of 'os' (line 34)
        os_36000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 34)
        path_36001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 18), os_36000, 'path')
        # Obtaining the member 'join' of a type (line 34)
        join_36002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 18), path_36001, 'join')
        # Calling join(args, kwargs) (line 34)
        join_call_result_36006 = invoke(stypy.reporting.localization.Localization(__file__, 34, 18), join_36002, *[tmpdir_36003, str_36004], **kwargs_36005)
        
        # Assigning a type to the variable 'sources' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'sources', join_call_result_36006)
        
        # Assigning a Call to a Name (line 35):
        
        # Call to join(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'tmpdir' (line 35)
        tmpdir_36010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 31), 'tmpdir', False)
        str_36011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 39), 'str', 'targets')
        # Processing the call keyword arguments (line 35)
        kwargs_36012 = {}
        # Getting the type of 'os' (line 35)
        os_36007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 35)
        path_36008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 18), os_36007, 'path')
        # Obtaining the member 'join' of a type (line 35)
        join_36009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 18), path_36008, 'join')
        # Calling join(args, kwargs) (line 35)
        join_call_result_36013 = invoke(stypy.reporting.localization.Localization(__file__, 35, 18), join_36009, *[tmpdir_36010, str_36011], **kwargs_36012)
        
        # Assigning a type to the variable 'targets' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'targets', join_call_result_36013)
        
        # Call to mkdir(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'sources' (line 36)
        sources_36016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'sources', False)
        # Processing the call keyword arguments (line 36)
        kwargs_36017 = {}
        # Getting the type of 'os' (line 36)
        os_36014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 36)
        mkdir_36015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), os_36014, 'mkdir')
        # Calling mkdir(args, kwargs) (line 36)
        mkdir_call_result_36018 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), mkdir_36015, *[sources_36016], **kwargs_36017)
        
        
        # Call to mkdir(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'targets' (line 37)
        targets_36021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'targets', False)
        # Processing the call keyword arguments (line 37)
        kwargs_36022 = {}
        # Getting the type of 'os' (line 37)
        os_36019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 37)
        mkdir_36020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), os_36019, 'mkdir')
        # Calling mkdir(args, kwargs) (line 37)
        mkdir_call_result_36023 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), mkdir_36020, *[targets_36021], **kwargs_36022)
        
        
        # Assigning a Call to a Name (line 38):
        
        # Call to join(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'sources' (line 38)
        sources_36027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'sources', False)
        str_36028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'str', 'one')
        # Processing the call keyword arguments (line 38)
        kwargs_36029 = {}
        # Getting the type of 'os' (line 38)
        os_36024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'os', False)
        # Obtaining the member 'path' of a type (line 38)
        path_36025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 14), os_36024, 'path')
        # Obtaining the member 'join' of a type (line 38)
        join_36026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 14), path_36025, 'join')
        # Calling join(args, kwargs) (line 38)
        join_call_result_36030 = invoke(stypy.reporting.localization.Localization(__file__, 38, 14), join_36026, *[sources_36027, str_36028], **kwargs_36029)
        
        # Assigning a type to the variable 'one' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'one', join_call_result_36030)
        
        # Assigning a Call to a Name (line 39):
        
        # Call to join(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'sources' (line 39)
        sources_36034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'sources', False)
        str_36035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'str', 'two')
        # Processing the call keyword arguments (line 39)
        kwargs_36036 = {}
        # Getting the type of 'os' (line 39)
        os_36031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'os', False)
        # Obtaining the member 'path' of a type (line 39)
        path_36032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 14), os_36031, 'path')
        # Obtaining the member 'join' of a type (line 39)
        join_36033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 14), path_36032, 'join')
        # Calling join(args, kwargs) (line 39)
        join_call_result_36037 = invoke(stypy.reporting.localization.Localization(__file__, 39, 14), join_36033, *[sources_36034, str_36035], **kwargs_36036)
        
        # Assigning a type to the variable 'two' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'two', join_call_result_36037)
        
        # Assigning a Call to a Name (line 40):
        
        # Call to abspath(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of '__file__' (line 40)
        file___36041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 32), '__file__', False)
        # Processing the call keyword arguments (line 40)
        kwargs_36042 = {}
        # Getting the type of 'os' (line 40)
        os_36038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 40)
        path_36039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 16), os_36038, 'path')
        # Obtaining the member 'abspath' of a type (line 40)
        abspath_36040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 16), path_36039, 'abspath')
        # Calling abspath(args, kwargs) (line 40)
        abspath_call_result_36043 = invoke(stypy.reporting.localization.Localization(__file__, 40, 16), abspath_36040, *[file___36041], **kwargs_36042)
        
        # Assigning a type to the variable 'three' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'three', abspath_call_result_36043)
        
        # Assigning a Call to a Name (line 41):
        
        # Call to join(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'targets' (line 41)
        targets_36047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'targets', False)
        str_36048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 37), 'str', 'four')
        # Processing the call keyword arguments (line 41)
        kwargs_36049 = {}
        # Getting the type of 'os' (line 41)
        os_36044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 41)
        path_36045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), os_36044, 'path')
        # Obtaining the member 'join' of a type (line 41)
        join_36046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), path_36045, 'join')
        # Calling join(args, kwargs) (line 41)
        join_call_result_36050 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), join_36046, *[targets_36047, str_36048], **kwargs_36049)
        
        # Assigning a type to the variable 'four' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'four', join_call_result_36050)
        
        # Call to write_file(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'one' (line 42)
        one_36053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'one', False)
        # Processing the call keyword arguments (line 42)
        kwargs_36054 = {}
        # Getting the type of 'self' (line 42)
        self_36051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 42)
        write_file_36052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_36051, 'write_file')
        # Calling write_file(args, kwargs) (line 42)
        write_file_call_result_36055 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), write_file_36052, *[one_36053], **kwargs_36054)
        
        
        # Call to write_file(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'two' (line 43)
        two_36058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'two', False)
        # Processing the call keyword arguments (line 43)
        kwargs_36059 = {}
        # Getting the type of 'self' (line 43)
        self_36056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 43)
        write_file_36057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_36056, 'write_file')
        # Calling write_file(args, kwargs) (line 43)
        write_file_call_result_36060 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), write_file_36057, *[two_36058], **kwargs_36059)
        
        
        # Call to write_file(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'four' (line 44)
        four_36063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'four', False)
        # Processing the call keyword arguments (line 44)
        kwargs_36064 = {}
        # Getting the type of 'self' (line 44)
        self_36061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 44)
        write_file_36062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), self_36061, 'write_file')
        # Calling write_file(args, kwargs) (line 44)
        write_file_call_result_36065 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), write_file_36062, *[four_36063], **kwargs_36064)
        
        
        # Call to assertEqual(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Call to newer_pairwise(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_36069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        # Getting the type of 'one' (line 46)
        one_36070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 41), 'one', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 40), list_36069, one_36070)
        # Adding element type (line 46)
        # Getting the type of 'two' (line 46)
        two_36071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'two', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 40), list_36069, two_36071)
        
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_36072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        # Getting the type of 'three' (line 46)
        three_36073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 53), 'three', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 52), list_36072, three_36073)
        # Adding element type (line 46)
        # Getting the type of 'four' (line 46)
        four_36074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 60), 'four', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 52), list_36072, four_36074)
        
        # Processing the call keyword arguments (line 46)
        kwargs_36075 = {}
        # Getting the type of 'newer_pairwise' (line 46)
        newer_pairwise_36068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'newer_pairwise', False)
        # Calling newer_pairwise(args, kwargs) (line 46)
        newer_pairwise_call_result_36076 = invoke(stypy.reporting.localization.Localization(__file__, 46, 25), newer_pairwise_36068, *[list_36069, list_36072], **kwargs_36075)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_36077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_36078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        # Getting the type of 'one' (line 47)
        one_36079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'one', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 26), list_36078, one_36079)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 26), tuple_36077, list_36078)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_36080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        # Getting the type of 'three' (line 47)
        three_36081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'three', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 32), list_36080, three_36081)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 26), tuple_36077, list_36080)
        
        # Processing the call keyword arguments (line 46)
        kwargs_36082 = {}
        # Getting the type of 'self' (line 46)
        self_36066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 46)
        assertEqual_36067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_36066, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 46)
        assertEqual_call_result_36083 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), assertEqual_36067, *[newer_pairwise_call_result_36076, tuple_36077], **kwargs_36082)
        
        
        # ################# End of 'test_newer_pairwise(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_newer_pairwise' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_36084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36084)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_newer_pairwise'
        return stypy_return_type_36084


    @norecursion
    def test_newer_group(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_newer_group'
        module_type_store = module_type_store.open_function_context('test_newer_group', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DepUtilTestCase.test_newer_group.__dict__.__setitem__('stypy_localization', localization)
        DepUtilTestCase.test_newer_group.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DepUtilTestCase.test_newer_group.__dict__.__setitem__('stypy_type_store', module_type_store)
        DepUtilTestCase.test_newer_group.__dict__.__setitem__('stypy_function_name', 'DepUtilTestCase.test_newer_group')
        DepUtilTestCase.test_newer_group.__dict__.__setitem__('stypy_param_names_list', [])
        DepUtilTestCase.test_newer_group.__dict__.__setitem__('stypy_varargs_param_name', None)
        DepUtilTestCase.test_newer_group.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DepUtilTestCase.test_newer_group.__dict__.__setitem__('stypy_call_defaults', defaults)
        DepUtilTestCase.test_newer_group.__dict__.__setitem__('stypy_call_varargs', varargs)
        DepUtilTestCase.test_newer_group.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DepUtilTestCase.test_newer_group.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DepUtilTestCase.test_newer_group', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_newer_group', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_newer_group(...)' code ##################

        
        # Assigning a Call to a Name (line 50):
        
        # Call to mkdtemp(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_36087 = {}
        # Getting the type of 'self' (line 50)
        self_36085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 50)
        mkdtemp_36086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 17), self_36085, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 50)
        mkdtemp_call_result_36088 = invoke(stypy.reporting.localization.Localization(__file__, 50, 17), mkdtemp_36086, *[], **kwargs_36087)
        
        # Assigning a type to the variable 'tmpdir' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'tmpdir', mkdtemp_call_result_36088)
        
        # Assigning a Call to a Name (line 51):
        
        # Call to join(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'tmpdir' (line 51)
        tmpdir_36092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'tmpdir', False)
        str_36093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 39), 'str', 'sources')
        # Processing the call keyword arguments (line 51)
        kwargs_36094 = {}
        # Getting the type of 'os' (line 51)
        os_36089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 51)
        path_36090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 18), os_36089, 'path')
        # Obtaining the member 'join' of a type (line 51)
        join_36091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 18), path_36090, 'join')
        # Calling join(args, kwargs) (line 51)
        join_call_result_36095 = invoke(stypy.reporting.localization.Localization(__file__, 51, 18), join_36091, *[tmpdir_36092, str_36093], **kwargs_36094)
        
        # Assigning a type to the variable 'sources' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'sources', join_call_result_36095)
        
        # Call to mkdir(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'sources' (line 52)
        sources_36098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 17), 'sources', False)
        # Processing the call keyword arguments (line 52)
        kwargs_36099 = {}
        # Getting the type of 'os' (line 52)
        os_36096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 52)
        mkdir_36097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), os_36096, 'mkdir')
        # Calling mkdir(args, kwargs) (line 52)
        mkdir_call_result_36100 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), mkdir_36097, *[sources_36098], **kwargs_36099)
        
        
        # Assigning a Call to a Name (line 53):
        
        # Call to join(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'sources' (line 53)
        sources_36104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'sources', False)
        str_36105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 36), 'str', 'one')
        # Processing the call keyword arguments (line 53)
        kwargs_36106 = {}
        # Getting the type of 'os' (line 53)
        os_36101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'os', False)
        # Obtaining the member 'path' of a type (line 53)
        path_36102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 14), os_36101, 'path')
        # Obtaining the member 'join' of a type (line 53)
        join_36103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 14), path_36102, 'join')
        # Calling join(args, kwargs) (line 53)
        join_call_result_36107 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), join_36103, *[sources_36104, str_36105], **kwargs_36106)
        
        # Assigning a type to the variable 'one' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'one', join_call_result_36107)
        
        # Assigning a Call to a Name (line 54):
        
        # Call to join(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'sources' (line 54)
        sources_36111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 27), 'sources', False)
        str_36112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 36), 'str', 'two')
        # Processing the call keyword arguments (line 54)
        kwargs_36113 = {}
        # Getting the type of 'os' (line 54)
        os_36108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 14), 'os', False)
        # Obtaining the member 'path' of a type (line 54)
        path_36109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 14), os_36108, 'path')
        # Obtaining the member 'join' of a type (line 54)
        join_36110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 14), path_36109, 'join')
        # Calling join(args, kwargs) (line 54)
        join_call_result_36114 = invoke(stypy.reporting.localization.Localization(__file__, 54, 14), join_36110, *[sources_36111, str_36112], **kwargs_36113)
        
        # Assigning a type to the variable 'two' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'two', join_call_result_36114)
        
        # Assigning a Call to a Name (line 55):
        
        # Call to join(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'sources' (line 55)
        sources_36118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 29), 'sources', False)
        str_36119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'str', 'three')
        # Processing the call keyword arguments (line 55)
        kwargs_36120 = {}
        # Getting the type of 'os' (line 55)
        os_36115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 55)
        path_36116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), os_36115, 'path')
        # Obtaining the member 'join' of a type (line 55)
        join_36117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), path_36116, 'join')
        # Calling join(args, kwargs) (line 55)
        join_call_result_36121 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), join_36117, *[sources_36118, str_36119], **kwargs_36120)
        
        # Assigning a type to the variable 'three' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'three', join_call_result_36121)
        
        # Assigning a Call to a Name (line 56):
        
        # Call to abspath(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of '__file__' (line 56)
        file___36125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 35), '__file__', False)
        # Processing the call keyword arguments (line 56)
        kwargs_36126 = {}
        # Getting the type of 'os' (line 56)
        os_36122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 56)
        path_36123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 19), os_36122, 'path')
        # Obtaining the member 'abspath' of a type (line 56)
        abspath_36124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 19), path_36123, 'abspath')
        # Calling abspath(args, kwargs) (line 56)
        abspath_call_result_36127 = invoke(stypy.reporting.localization.Localization(__file__, 56, 19), abspath_36124, *[file___36125], **kwargs_36126)
        
        # Assigning a type to the variable 'old_file' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'old_file', abspath_call_result_36127)
        
        # Call to write_file(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'one' (line 60)
        one_36130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 'one', False)
        # Processing the call keyword arguments (line 60)
        kwargs_36131 = {}
        # Getting the type of 'self' (line 60)
        self_36128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 60)
        write_file_36129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_36128, 'write_file')
        # Calling write_file(args, kwargs) (line 60)
        write_file_call_result_36132 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), write_file_36129, *[one_36130], **kwargs_36131)
        
        
        # Call to write_file(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'two' (line 61)
        two_36135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'two', False)
        # Processing the call keyword arguments (line 61)
        kwargs_36136 = {}
        # Getting the type of 'self' (line 61)
        self_36133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 61)
        write_file_36134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_36133, 'write_file')
        # Calling write_file(args, kwargs) (line 61)
        write_file_call_result_36137 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), write_file_36134, *[two_36135], **kwargs_36136)
        
        
        # Call to write_file(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'three' (line 62)
        three_36140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'three', False)
        # Processing the call keyword arguments (line 62)
        kwargs_36141 = {}
        # Getting the type of 'self' (line 62)
        self_36138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 62)
        write_file_36139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_36138, 'write_file')
        # Calling write_file(args, kwargs) (line 62)
        write_file_call_result_36142 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), write_file_36139, *[three_36140], **kwargs_36141)
        
        
        # Call to assertTrue(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to newer_group(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_36146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        # Getting the type of 'one' (line 63)
        one_36147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'one', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 36), list_36146, one_36147)
        # Adding element type (line 63)
        # Getting the type of 'two' (line 63)
        two_36148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 42), 'two', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 36), list_36146, two_36148)
        # Adding element type (line 63)
        # Getting the type of 'three' (line 63)
        three_36149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 47), 'three', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 36), list_36146, three_36149)
        
        # Getting the type of 'old_file' (line 63)
        old_file_36150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 55), 'old_file', False)
        # Processing the call keyword arguments (line 63)
        kwargs_36151 = {}
        # Getting the type of 'newer_group' (line 63)
        newer_group_36145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'newer_group', False)
        # Calling newer_group(args, kwargs) (line 63)
        newer_group_call_result_36152 = invoke(stypy.reporting.localization.Localization(__file__, 63, 24), newer_group_36145, *[list_36146, old_file_36150], **kwargs_36151)
        
        # Processing the call keyword arguments (line 63)
        kwargs_36153 = {}
        # Getting the type of 'self' (line 63)
        self_36143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 63)
        assertTrue_36144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_36143, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 63)
        assertTrue_call_result_36154 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), assertTrue_36144, *[newer_group_call_result_36152], **kwargs_36153)
        
        
        # Call to assertFalse(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to newer_group(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_36158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        # Getting the type of 'one' (line 64)
        one_36159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 38), 'one', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 37), list_36158, one_36159)
        # Adding element type (line 64)
        # Getting the type of 'two' (line 64)
        two_36160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 43), 'two', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 37), list_36158, two_36160)
        # Adding element type (line 64)
        # Getting the type of 'old_file' (line 64)
        old_file_36161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 48), 'old_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 37), list_36158, old_file_36161)
        
        # Getting the type of 'three' (line 64)
        three_36162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 59), 'three', False)
        # Processing the call keyword arguments (line 64)
        kwargs_36163 = {}
        # Getting the type of 'newer_group' (line 64)
        newer_group_36157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'newer_group', False)
        # Calling newer_group(args, kwargs) (line 64)
        newer_group_call_result_36164 = invoke(stypy.reporting.localization.Localization(__file__, 64, 25), newer_group_36157, *[list_36158, three_36162], **kwargs_36163)
        
        # Processing the call keyword arguments (line 64)
        kwargs_36165 = {}
        # Getting the type of 'self' (line 64)
        self_36155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 64)
        assertFalse_36156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_36155, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 64)
        assertFalse_call_result_36166 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assertFalse_36156, *[newer_group_call_result_36164], **kwargs_36165)
        
        
        # Call to remove(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'one' (line 67)
        one_36169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'one', False)
        # Processing the call keyword arguments (line 67)
        kwargs_36170 = {}
        # Getting the type of 'os' (line 67)
        os_36167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'os', False)
        # Obtaining the member 'remove' of a type (line 67)
        remove_36168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), os_36167, 'remove')
        # Calling remove(args, kwargs) (line 67)
        remove_call_result_36171 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), remove_36168, *[one_36169], **kwargs_36170)
        
        
        # Call to assertRaises(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'OSError' (line 68)
        OSError_36174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 26), 'OSError', False)
        # Getting the type of 'newer_group' (line 68)
        newer_group_36175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 35), 'newer_group', False)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_36176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        # Getting the type of 'one' (line 68)
        one_36177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 49), 'one', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 48), list_36176, one_36177)
        # Adding element type (line 68)
        # Getting the type of 'two' (line 68)
        two_36178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 54), 'two', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 48), list_36176, two_36178)
        # Adding element type (line 68)
        # Getting the type of 'old_file' (line 68)
        old_file_36179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 59), 'old_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 48), list_36176, old_file_36179)
        
        # Getting the type of 'three' (line 68)
        three_36180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 70), 'three', False)
        # Processing the call keyword arguments (line 68)
        kwargs_36181 = {}
        # Getting the type of 'self' (line 68)
        self_36172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 68)
        assertRaises_36173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_36172, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 68)
        assertRaises_call_result_36182 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), assertRaises_36173, *[OSError_36174, newer_group_36175, list_36176, three_36180], **kwargs_36181)
        
        
        # Call to assertFalse(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Call to newer_group(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_36186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        # Getting the type of 'one' (line 70)
        one_36187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'one', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 37), list_36186, one_36187)
        # Adding element type (line 70)
        # Getting the type of 'two' (line 70)
        two_36188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 43), 'two', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 37), list_36186, two_36188)
        # Adding element type (line 70)
        # Getting the type of 'old_file' (line 70)
        old_file_36189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 48), 'old_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 37), list_36186, old_file_36189)
        
        # Getting the type of 'three' (line 70)
        three_36190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 59), 'three', False)
        # Processing the call keyword arguments (line 70)
        str_36191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 45), 'str', 'ignore')
        keyword_36192 = str_36191
        kwargs_36193 = {'missing': keyword_36192}
        # Getting the type of 'newer_group' (line 70)
        newer_group_36185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'newer_group', False)
        # Calling newer_group(args, kwargs) (line 70)
        newer_group_call_result_36194 = invoke(stypy.reporting.localization.Localization(__file__, 70, 25), newer_group_36185, *[list_36186, three_36190], **kwargs_36193)
        
        # Processing the call keyword arguments (line 70)
        kwargs_36195 = {}
        # Getting the type of 'self' (line 70)
        self_36183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 70)
        assertFalse_36184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_36183, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 70)
        assertFalse_call_result_36196 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), assertFalse_36184, *[newer_group_call_result_36194], **kwargs_36195)
        
        
        # Call to assertTrue(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to newer_group(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_36200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        # Getting the type of 'one' (line 73)
        one_36201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 37), 'one', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 36), list_36200, one_36201)
        # Adding element type (line 73)
        # Getting the type of 'two' (line 73)
        two_36202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 42), 'two', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 36), list_36200, two_36202)
        # Adding element type (line 73)
        # Getting the type of 'old_file' (line 73)
        old_file_36203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 47), 'old_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 36), list_36200, old_file_36203)
        
        # Getting the type of 'three' (line 73)
        three_36204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 58), 'three', False)
        # Processing the call keyword arguments (line 73)
        str_36205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 44), 'str', 'newer')
        keyword_36206 = str_36205
        kwargs_36207 = {'missing': keyword_36206}
        # Getting the type of 'newer_group' (line 73)
        newer_group_36199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'newer_group', False)
        # Calling newer_group(args, kwargs) (line 73)
        newer_group_call_result_36208 = invoke(stypy.reporting.localization.Localization(__file__, 73, 24), newer_group_36199, *[list_36200, three_36204], **kwargs_36207)
        
        # Processing the call keyword arguments (line 73)
        kwargs_36209 = {}
        # Getting the type of 'self' (line 73)
        self_36197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 73)
        assertTrue_36198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_36197, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 73)
        assertTrue_call_result_36210 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), assertTrue_36198, *[newer_group_call_result_36208], **kwargs_36209)
        
        
        # ################# End of 'test_newer_group(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_newer_group' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_36211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36211)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_newer_group'
        return stypy_return_type_36211


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DepUtilTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DepUtilTestCase' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'DepUtilTestCase', DepUtilTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 77, 0, False)
    
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

    
    # Call to makeSuite(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'DepUtilTestCase' (line 78)
    DepUtilTestCase_36214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'DepUtilTestCase', False)
    # Processing the call keyword arguments (line 78)
    kwargs_36215 = {}
    # Getting the type of 'unittest' (line 78)
    unittest_36212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 78)
    makeSuite_36213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 11), unittest_36212, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 78)
    makeSuite_call_result_36216 = invoke(stypy.reporting.localization.Localization(__file__, 78, 11), makeSuite_36213, *[DepUtilTestCase_36214], **kwargs_36215)
    
    # Assigning a type to the variable 'stypy_return_type' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type', makeSuite_call_result_36216)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_36217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36217)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_36217

# Assigning a type to the variable 'test_suite' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 81)
    # Processing the call arguments (line 81)
    
    # Call to test_suite(...): (line 81)
    # Processing the call keyword arguments (line 81)
    kwargs_36220 = {}
    # Getting the type of 'test_suite' (line 81)
    test_suite_36219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 81)
    test_suite_call_result_36221 = invoke(stypy.reporting.localization.Localization(__file__, 81, 17), test_suite_36219, *[], **kwargs_36220)
    
    # Processing the call keyword arguments (line 81)
    kwargs_36222 = {}
    # Getting the type of 'run_unittest' (line 81)
    run_unittest_36218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 81)
    run_unittest_call_result_36223 = invoke(stypy.reporting.localization.Localization(__file__, 81, 4), run_unittest_36218, *[test_suite_call_result_36221], **kwargs_36222)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
