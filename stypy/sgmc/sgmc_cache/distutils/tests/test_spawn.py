
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.spawn.'''
2: import unittest
3: import os
4: import time
5: from test.test_support import captured_stdout, run_unittest
6: 
7: from distutils.spawn import _nt_quote_args
8: from distutils.spawn import spawn, find_executable
9: from distutils.errors import DistutilsExecError
10: from distutils.tests import support
11: 
12: class SpawnTestCase(support.TempdirManager,
13:                     support.LoggingSilencer,
14:                     unittest.TestCase):
15: 
16:     def test_nt_quote_args(self):
17: 
18:         for (args, wanted) in ((['with space', 'nospace'],
19:                                 ['"with space"', 'nospace']),
20:                                (['nochange', 'nospace'],
21:                                 ['nochange', 'nospace'])):
22:             res = _nt_quote_args(args)
23:             self.assertEqual(res, wanted)
24: 
25: 
26:     @unittest.skipUnless(os.name in ('nt', 'posix'),
27:                          'Runs only under posix or nt')
28:     def test_spawn(self):
29:         tmpdir = self.mkdtemp()
30: 
31:         # creating something executable
32:         # through the shell that returns 1
33:         if os.name == 'posix':
34:             exe = os.path.join(tmpdir, 'foo.sh')
35:             self.write_file(exe, '#!/bin/sh\nexit 1')
36:             os.chmod(exe, 0777)
37:         else:
38:             exe = os.path.join(tmpdir, 'foo.bat')
39:             self.write_file(exe, 'exit 1')
40: 
41:         os.chmod(exe, 0777)
42:         self.assertRaises(DistutilsExecError, spawn, [exe])
43: 
44:         # now something that works
45:         if os.name == 'posix':
46:             exe = os.path.join(tmpdir, 'foo.sh')
47:             self.write_file(exe, '#!/bin/sh\nexit 0')
48:             os.chmod(exe, 0777)
49:         else:
50:             exe = os.path.join(tmpdir, 'foo.bat')
51:             self.write_file(exe, 'exit 0')
52: 
53:         os.chmod(exe, 0777)
54:         spawn([exe])  # should work without any error
55: 
56: def test_suite():
57:     return unittest.makeSuite(SpawnTestCase)
58: 
59: if __name__ == "__main__":
60:     run_unittest(test_suite())
61: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_44052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.spawn.')
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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from test.test_support import captured_stdout, run_unittest' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_44053 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support')

if (type(import_44053) is not StypyTypeError):

    if (import_44053 != 'pyd_module'):
        __import__(import_44053)
        sys_modules_44054 = sys.modules[import_44053]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', sys_modules_44054.module_type_store, module_type_store, ['captured_stdout', 'run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_44054, sys_modules_44054.module_type_store, module_type_store)
    else:
        from test.test_support import captured_stdout, run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', None, module_type_store, ['captured_stdout', 'run_unittest'], [captured_stdout, run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', import_44053)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.spawn import _nt_quote_args' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_44055 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.spawn')

if (type(import_44055) is not StypyTypeError):

    if (import_44055 != 'pyd_module'):
        __import__(import_44055)
        sys_modules_44056 = sys.modules[import_44055]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.spawn', sys_modules_44056.module_type_store, module_type_store, ['_nt_quote_args'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_44056, sys_modules_44056.module_type_store, module_type_store)
    else:
        from distutils.spawn import _nt_quote_args

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.spawn', None, module_type_store, ['_nt_quote_args'], [_nt_quote_args])

else:
    # Assigning a type to the variable 'distutils.spawn' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.spawn', import_44055)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.spawn import spawn, find_executable' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_44057 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.spawn')

if (type(import_44057) is not StypyTypeError):

    if (import_44057 != 'pyd_module'):
        __import__(import_44057)
        sys_modules_44058 = sys.modules[import_44057]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.spawn', sys_modules_44058.module_type_store, module_type_store, ['spawn', 'find_executable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_44058, sys_modules_44058.module_type_store, module_type_store)
    else:
        from distutils.spawn import spawn, find_executable

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.spawn', None, module_type_store, ['spawn', 'find_executable'], [spawn, find_executable])

else:
    # Assigning a type to the variable 'distutils.spawn' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.spawn', import_44057)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.errors import DistutilsExecError' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_44059 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors')

if (type(import_44059) is not StypyTypeError):

    if (import_44059 != 'pyd_module'):
        __import__(import_44059)
        sys_modules_44060 = sys.modules[import_44059]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', sys_modules_44060.module_type_store, module_type_store, ['DistutilsExecError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_44060, sys_modules_44060.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsExecError

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', None, module_type_store, ['DistutilsExecError'], [DistutilsExecError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', import_44059)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.tests import support' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_44061 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests')

if (type(import_44061) is not StypyTypeError):

    if (import_44061 != 'pyd_module'):
        __import__(import_44061)
        sys_modules_44062 = sys.modules[import_44061]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests', sys_modules_44062.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_44062, sys_modules_44062.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.tests', import_44061)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'SpawnTestCase' class
# Getting the type of 'support' (line 12)
support_44063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'support')
# Obtaining the member 'TempdirManager' of a type (line 12)
TempdirManager_44064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 20), support_44063, 'TempdirManager')
# Getting the type of 'support' (line 13)
support_44065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 13)
LoggingSilencer_44066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 20), support_44065, 'LoggingSilencer')
# Getting the type of 'unittest' (line 14)
unittest_44067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 20), 'unittest')
# Obtaining the member 'TestCase' of a type (line 14)
TestCase_44068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 20), unittest_44067, 'TestCase')

class SpawnTestCase(TempdirManager_44064, LoggingSilencer_44066, TestCase_44068, ):

    @norecursion
    def test_nt_quote_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_nt_quote_args'
        module_type_store = module_type_store.open_function_context('test_nt_quote_args', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SpawnTestCase.test_nt_quote_args.__dict__.__setitem__('stypy_localization', localization)
        SpawnTestCase.test_nt_quote_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SpawnTestCase.test_nt_quote_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        SpawnTestCase.test_nt_quote_args.__dict__.__setitem__('stypy_function_name', 'SpawnTestCase.test_nt_quote_args')
        SpawnTestCase.test_nt_quote_args.__dict__.__setitem__('stypy_param_names_list', [])
        SpawnTestCase.test_nt_quote_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        SpawnTestCase.test_nt_quote_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SpawnTestCase.test_nt_quote_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        SpawnTestCase.test_nt_quote_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        SpawnTestCase.test_nt_quote_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SpawnTestCase.test_nt_quote_args.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SpawnTestCase.test_nt_quote_args', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_nt_quote_args', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_nt_quote_args(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 18)
        tuple_44069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 18)
        # Adding element type (line 18)
        
        # Obtaining an instance of the builtin type 'tuple' (line 18)
        tuple_44070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 18)
        # Adding element type (line 18)
        
        # Obtaining an instance of the builtin type 'list' (line 18)
        list_44071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 18)
        # Adding element type (line 18)
        str_44072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 33), 'str', 'with space')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 32), list_44071, str_44072)
        # Adding element type (line 18)
        str_44073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 47), 'str', 'nospace')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 32), list_44071, str_44073)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 32), tuple_44070, list_44071)
        # Adding element type (line 18)
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_44074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        # Adding element type (line 19)
        str_44075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 33), 'str', '"with space"')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 32), list_44074, str_44075)
        # Adding element type (line 19)
        str_44076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 49), 'str', 'nospace')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 32), list_44074, str_44076)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 32), tuple_44070, list_44074)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 31), tuple_44069, tuple_44070)
        # Adding element type (line 18)
        
        # Obtaining an instance of the builtin type 'tuple' (line 20)
        tuple_44077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 20)
        # Adding element type (line 20)
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_44078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        # Adding element type (line 20)
        str_44079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 33), 'str', 'nochange')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 32), list_44078, str_44079)
        # Adding element type (line 20)
        str_44080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 45), 'str', 'nospace')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 32), list_44078, str_44080)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 32), tuple_44077, list_44078)
        # Adding element type (line 20)
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_44081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        str_44082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 33), 'str', 'nochange')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 32), list_44081, str_44082)
        # Adding element type (line 21)
        str_44083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 45), 'str', 'nospace')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 32), list_44081, str_44083)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 32), tuple_44077, list_44081)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 31), tuple_44069, tuple_44077)
        
        # Testing the type of a for loop iterable (line 18)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 18, 8), tuple_44069)
        # Getting the type of the for loop variable (line 18)
        for_loop_var_44084 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 18, 8), tuple_44069)
        # Assigning a type to the variable 'args' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'args', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 8), for_loop_var_44084))
        # Assigning a type to the variable 'wanted' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'wanted', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 8), for_loop_var_44084))
        # SSA begins for a for statement (line 18)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 22):
        
        # Call to _nt_quote_args(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'args' (line 22)
        args_44086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 33), 'args', False)
        # Processing the call keyword arguments (line 22)
        kwargs_44087 = {}
        # Getting the type of '_nt_quote_args' (line 22)
        _nt_quote_args_44085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), '_nt_quote_args', False)
        # Calling _nt_quote_args(args, kwargs) (line 22)
        _nt_quote_args_call_result_44088 = invoke(stypy.reporting.localization.Localization(__file__, 22, 18), _nt_quote_args_44085, *[args_44086], **kwargs_44087)
        
        # Assigning a type to the variable 'res' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'res', _nt_quote_args_call_result_44088)
        
        # Call to assertEqual(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'res' (line 23)
        res_44091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 29), 'res', False)
        # Getting the type of 'wanted' (line 23)
        wanted_44092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 34), 'wanted', False)
        # Processing the call keyword arguments (line 23)
        kwargs_44093 = {}
        # Getting the type of 'self' (line 23)
        self_44089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 23)
        assertEqual_44090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), self_44089, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 23)
        assertEqual_call_result_44094 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), assertEqual_44090, *[res_44091, wanted_44092], **kwargs_44093)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_nt_quote_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_nt_quote_args' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_44095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44095)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_nt_quote_args'
        return stypy_return_type_44095


    @norecursion
    def test_spawn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spawn'
        module_type_store = module_type_store.open_function_context('test_spawn', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SpawnTestCase.test_spawn.__dict__.__setitem__('stypy_localization', localization)
        SpawnTestCase.test_spawn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SpawnTestCase.test_spawn.__dict__.__setitem__('stypy_type_store', module_type_store)
        SpawnTestCase.test_spawn.__dict__.__setitem__('stypy_function_name', 'SpawnTestCase.test_spawn')
        SpawnTestCase.test_spawn.__dict__.__setitem__('stypy_param_names_list', [])
        SpawnTestCase.test_spawn.__dict__.__setitem__('stypy_varargs_param_name', None)
        SpawnTestCase.test_spawn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SpawnTestCase.test_spawn.__dict__.__setitem__('stypy_call_defaults', defaults)
        SpawnTestCase.test_spawn.__dict__.__setitem__('stypy_call_varargs', varargs)
        SpawnTestCase.test_spawn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SpawnTestCase.test_spawn.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SpawnTestCase.test_spawn', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spawn', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spawn(...)' code ##################

        
        # Assigning a Call to a Name (line 29):
        
        # Call to mkdtemp(...): (line 29)
        # Processing the call keyword arguments (line 29)
        kwargs_44098 = {}
        # Getting the type of 'self' (line 29)
        self_44096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 29)
        mkdtemp_44097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 17), self_44096, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 29)
        mkdtemp_call_result_44099 = invoke(stypy.reporting.localization.Localization(__file__, 29, 17), mkdtemp_44097, *[], **kwargs_44098)
        
        # Assigning a type to the variable 'tmpdir' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'tmpdir', mkdtemp_call_result_44099)
        
        
        # Getting the type of 'os' (line 33)
        os_44100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'os')
        # Obtaining the member 'name' of a type (line 33)
        name_44101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 11), os_44100, 'name')
        str_44102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 22), 'str', 'posix')
        # Applying the binary operator '==' (line 33)
        result_eq_44103 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 11), '==', name_44101, str_44102)
        
        # Testing the type of an if condition (line 33)
        if_condition_44104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 8), result_eq_44103)
        # Assigning a type to the variable 'if_condition_44104' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'if_condition_44104', if_condition_44104)
        # SSA begins for if statement (line 33)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 34):
        
        # Call to join(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'tmpdir' (line 34)
        tmpdir_44108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 31), 'tmpdir', False)
        str_44109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 39), 'str', 'foo.sh')
        # Processing the call keyword arguments (line 34)
        kwargs_44110 = {}
        # Getting the type of 'os' (line 34)
        os_44105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 34)
        path_44106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 18), os_44105, 'path')
        # Obtaining the member 'join' of a type (line 34)
        join_44107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 18), path_44106, 'join')
        # Calling join(args, kwargs) (line 34)
        join_call_result_44111 = invoke(stypy.reporting.localization.Localization(__file__, 34, 18), join_44107, *[tmpdir_44108, str_44109], **kwargs_44110)
        
        # Assigning a type to the variable 'exe' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'exe', join_call_result_44111)
        
        # Call to write_file(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'exe' (line 35)
        exe_44114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 28), 'exe', False)
        str_44115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 33), 'str', '#!/bin/sh\nexit 1')
        # Processing the call keyword arguments (line 35)
        kwargs_44116 = {}
        # Getting the type of 'self' (line 35)
        self_44112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'self', False)
        # Obtaining the member 'write_file' of a type (line 35)
        write_file_44113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), self_44112, 'write_file')
        # Calling write_file(args, kwargs) (line 35)
        write_file_call_result_44117 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), write_file_44113, *[exe_44114, str_44115], **kwargs_44116)
        
        
        # Call to chmod(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'exe' (line 36)
        exe_44120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'exe', False)
        int_44121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 26), 'int')
        # Processing the call keyword arguments (line 36)
        kwargs_44122 = {}
        # Getting the type of 'os' (line 36)
        os_44118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'os', False)
        # Obtaining the member 'chmod' of a type (line 36)
        chmod_44119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), os_44118, 'chmod')
        # Calling chmod(args, kwargs) (line 36)
        chmod_call_result_44123 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), chmod_44119, *[exe_44120, int_44121], **kwargs_44122)
        
        # SSA branch for the else part of an if statement (line 33)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 38):
        
        # Call to join(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'tmpdir' (line 38)
        tmpdir_44127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 31), 'tmpdir', False)
        str_44128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 39), 'str', 'foo.bat')
        # Processing the call keyword arguments (line 38)
        kwargs_44129 = {}
        # Getting the type of 'os' (line 38)
        os_44124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 38)
        path_44125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 18), os_44124, 'path')
        # Obtaining the member 'join' of a type (line 38)
        join_44126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 18), path_44125, 'join')
        # Calling join(args, kwargs) (line 38)
        join_call_result_44130 = invoke(stypy.reporting.localization.Localization(__file__, 38, 18), join_44126, *[tmpdir_44127, str_44128], **kwargs_44129)
        
        # Assigning a type to the variable 'exe' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'exe', join_call_result_44130)
        
        # Call to write_file(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'exe' (line 39)
        exe_44133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 28), 'exe', False)
        str_44134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'str', 'exit 1')
        # Processing the call keyword arguments (line 39)
        kwargs_44135 = {}
        # Getting the type of 'self' (line 39)
        self_44131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'self', False)
        # Obtaining the member 'write_file' of a type (line 39)
        write_file_44132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), self_44131, 'write_file')
        # Calling write_file(args, kwargs) (line 39)
        write_file_call_result_44136 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), write_file_44132, *[exe_44133, str_44134], **kwargs_44135)
        
        # SSA join for if statement (line 33)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to chmod(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'exe' (line 41)
        exe_44139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'exe', False)
        int_44140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'int')
        # Processing the call keyword arguments (line 41)
        kwargs_44141 = {}
        # Getting the type of 'os' (line 41)
        os_44137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'os', False)
        # Obtaining the member 'chmod' of a type (line 41)
        chmod_44138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), os_44137, 'chmod')
        # Calling chmod(args, kwargs) (line 41)
        chmod_call_result_44142 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), chmod_44138, *[exe_44139, int_44140], **kwargs_44141)
        
        
        # Call to assertRaises(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'DistutilsExecError' (line 42)
        DistutilsExecError_44145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'DistutilsExecError', False)
        # Getting the type of 'spawn' (line 42)
        spawn_44146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 46), 'spawn', False)
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_44147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        # Getting the type of 'exe' (line 42)
        exe_44148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 54), 'exe', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 53), list_44147, exe_44148)
        
        # Processing the call keyword arguments (line 42)
        kwargs_44149 = {}
        # Getting the type of 'self' (line 42)
        self_44143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 42)
        assertRaises_44144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_44143, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 42)
        assertRaises_call_result_44150 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assertRaises_44144, *[DistutilsExecError_44145, spawn_44146, list_44147], **kwargs_44149)
        
        
        
        # Getting the type of 'os' (line 45)
        os_44151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'os')
        # Obtaining the member 'name' of a type (line 45)
        name_44152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 11), os_44151, 'name')
        str_44153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 22), 'str', 'posix')
        # Applying the binary operator '==' (line 45)
        result_eq_44154 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), '==', name_44152, str_44153)
        
        # Testing the type of an if condition (line 45)
        if_condition_44155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 8), result_eq_44154)
        # Assigning a type to the variable 'if_condition_44155' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'if_condition_44155', if_condition_44155)
        # SSA begins for if statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 46):
        
        # Call to join(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'tmpdir' (line 46)
        tmpdir_44159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 31), 'tmpdir', False)
        str_44160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 39), 'str', 'foo.sh')
        # Processing the call keyword arguments (line 46)
        kwargs_44161 = {}
        # Getting the type of 'os' (line 46)
        os_44156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 46)
        path_44157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 18), os_44156, 'path')
        # Obtaining the member 'join' of a type (line 46)
        join_44158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 18), path_44157, 'join')
        # Calling join(args, kwargs) (line 46)
        join_call_result_44162 = invoke(stypy.reporting.localization.Localization(__file__, 46, 18), join_44158, *[tmpdir_44159, str_44160], **kwargs_44161)
        
        # Assigning a type to the variable 'exe' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'exe', join_call_result_44162)
        
        # Call to write_file(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'exe' (line 47)
        exe_44165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 28), 'exe', False)
        str_44166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 33), 'str', '#!/bin/sh\nexit 0')
        # Processing the call keyword arguments (line 47)
        kwargs_44167 = {}
        # Getting the type of 'self' (line 47)
        self_44163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'self', False)
        # Obtaining the member 'write_file' of a type (line 47)
        write_file_44164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), self_44163, 'write_file')
        # Calling write_file(args, kwargs) (line 47)
        write_file_call_result_44168 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), write_file_44164, *[exe_44165, str_44166], **kwargs_44167)
        
        
        # Call to chmod(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'exe' (line 48)
        exe_44171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'exe', False)
        int_44172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 26), 'int')
        # Processing the call keyword arguments (line 48)
        kwargs_44173 = {}
        # Getting the type of 'os' (line 48)
        os_44169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'os', False)
        # Obtaining the member 'chmod' of a type (line 48)
        chmod_44170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), os_44169, 'chmod')
        # Calling chmod(args, kwargs) (line 48)
        chmod_call_result_44174 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), chmod_44170, *[exe_44171, int_44172], **kwargs_44173)
        
        # SSA branch for the else part of an if statement (line 45)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 50):
        
        # Call to join(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'tmpdir' (line 50)
        tmpdir_44178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 31), 'tmpdir', False)
        str_44179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 39), 'str', 'foo.bat')
        # Processing the call keyword arguments (line 50)
        kwargs_44180 = {}
        # Getting the type of 'os' (line 50)
        os_44175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 50)
        path_44176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 18), os_44175, 'path')
        # Obtaining the member 'join' of a type (line 50)
        join_44177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 18), path_44176, 'join')
        # Calling join(args, kwargs) (line 50)
        join_call_result_44181 = invoke(stypy.reporting.localization.Localization(__file__, 50, 18), join_44177, *[tmpdir_44178, str_44179], **kwargs_44180)
        
        # Assigning a type to the variable 'exe' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'exe', join_call_result_44181)
        
        # Call to write_file(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'exe' (line 51)
        exe_44184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'exe', False)
        str_44185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 33), 'str', 'exit 0')
        # Processing the call keyword arguments (line 51)
        kwargs_44186 = {}
        # Getting the type of 'self' (line 51)
        self_44182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self', False)
        # Obtaining the member 'write_file' of a type (line 51)
        write_file_44183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_44182, 'write_file')
        # Calling write_file(args, kwargs) (line 51)
        write_file_call_result_44187 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), write_file_44183, *[exe_44184, str_44185], **kwargs_44186)
        
        # SSA join for if statement (line 45)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to chmod(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'exe' (line 53)
        exe_44190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'exe', False)
        int_44191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 22), 'int')
        # Processing the call keyword arguments (line 53)
        kwargs_44192 = {}
        # Getting the type of 'os' (line 53)
        os_44188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'os', False)
        # Obtaining the member 'chmod' of a type (line 53)
        chmod_44189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), os_44188, 'chmod')
        # Calling chmod(args, kwargs) (line 53)
        chmod_call_result_44193 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), chmod_44189, *[exe_44190, int_44191], **kwargs_44192)
        
        
        # Call to spawn(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_44195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        # Getting the type of 'exe' (line 54)
        exe_44196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'exe', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 14), list_44195, exe_44196)
        
        # Processing the call keyword arguments (line 54)
        kwargs_44197 = {}
        # Getting the type of 'spawn' (line 54)
        spawn_44194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'spawn', False)
        # Calling spawn(args, kwargs) (line 54)
        spawn_call_result_44198 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), spawn_44194, *[list_44195], **kwargs_44197)
        
        
        # ################# End of 'test_spawn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spawn' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_44199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44199)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spawn'
        return stypy_return_type_44199


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SpawnTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SpawnTestCase' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'SpawnTestCase', SpawnTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 56, 0, False)
    
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

    
    # Call to makeSuite(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'SpawnTestCase' (line 57)
    SpawnTestCase_44202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'SpawnTestCase', False)
    # Processing the call keyword arguments (line 57)
    kwargs_44203 = {}
    # Getting the type of 'unittest' (line 57)
    unittest_44200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 57)
    makeSuite_44201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), unittest_44200, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 57)
    makeSuite_call_result_44204 = invoke(stypy.reporting.localization.Localization(__file__, 57, 11), makeSuite_44201, *[SpawnTestCase_44202], **kwargs_44203)
    
    # Assigning a type to the variable 'stypy_return_type' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type', makeSuite_call_result_44204)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 56)
    stypy_return_type_44205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44205)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_44205

# Assigning a type to the variable 'test_suite' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Call to test_suite(...): (line 60)
    # Processing the call keyword arguments (line 60)
    kwargs_44208 = {}
    # Getting the type of 'test_suite' (line 60)
    test_suite_44207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 60)
    test_suite_call_result_44209 = invoke(stypy.reporting.localization.Localization(__file__, 60, 17), test_suite_44207, *[], **kwargs_44208)
    
    # Processing the call keyword arguments (line 60)
    kwargs_44210 = {}
    # Getting the type of 'run_unittest' (line 60)
    run_unittest_44206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 60)
    run_unittest_call_result_44211 = invoke(stypy.reporting.localization.Localization(__file__, 60, 4), run_unittest_44206, *[test_suite_call_result_44209], **kwargs_44210)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
