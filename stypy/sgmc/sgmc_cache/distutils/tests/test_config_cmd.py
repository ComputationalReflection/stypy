
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.config.'''
2: import unittest
3: import os
4: import sys
5: from test.test_support import run_unittest
6: 
7: from distutils.command.config import dump_file, config
8: from distutils.tests import support
9: from distutils import log
10: 
11: class ConfigTestCase(support.LoggingSilencer,
12:                      support.TempdirManager,
13:                      unittest.TestCase):
14: 
15:     def _info(self, msg, *args):
16:         for line in msg.splitlines():
17:             self._logs.append(line)
18: 
19:     def setUp(self):
20:         super(ConfigTestCase, self).setUp()
21:         self._logs = []
22:         self.old_log = log.info
23:         log.info = self._info
24: 
25:     def tearDown(self):
26:         log.info = self.old_log
27:         super(ConfigTestCase, self).tearDown()
28: 
29:     def test_dump_file(self):
30:         this_file = os.path.splitext(__file__)[0] + '.py'
31:         f = open(this_file)
32:         try:
33:             numlines = len(f.readlines())
34:         finally:
35:             f.close()
36: 
37:         dump_file(this_file, 'I am the header')
38:         self.assertEqual(len(self._logs), numlines+1)
39: 
40:     @unittest.skipIf(sys.platform == 'win32', "can't test on Windows")
41:     def test_search_cpp(self):
42:         pkg_dir, dist = self.create_dist()
43:         cmd = config(dist)
44: 
45:         # simple pattern searches
46:         match = cmd.search_cpp(pattern='xxx', body='/* xxx */')
47:         self.assertEqual(match, 0)
48: 
49:         match = cmd.search_cpp(pattern='_configtest', body='/* xxx */')
50:         self.assertEqual(match, 1)
51: 
52:     def test_finalize_options(self):
53:         # finalize_options does a bit of transformation
54:         # on options
55:         pkg_dir, dist = self.create_dist()
56:         cmd = config(dist)
57:         cmd.include_dirs = 'one%stwo' % os.pathsep
58:         cmd.libraries = 'one'
59:         cmd.library_dirs = 'three%sfour' % os.pathsep
60:         cmd.ensure_finalized()
61: 
62:         self.assertEqual(cmd.include_dirs, ['one', 'two'])
63:         self.assertEqual(cmd.libraries, ['one'])
64:         self.assertEqual(cmd.library_dirs, ['three', 'four'])
65: 
66:     def test_clean(self):
67:         # _clean removes files
68:         tmp_dir = self.mkdtemp()
69:         f1 = os.path.join(tmp_dir, 'one')
70:         f2 = os.path.join(tmp_dir, 'two')
71: 
72:         self.write_file(f1, 'xxx')
73:         self.write_file(f2, 'xxx')
74: 
75:         for f in (f1, f2):
76:             self.assertTrue(os.path.exists(f))
77: 
78:         pkg_dir, dist = self.create_dist()
79:         cmd = config(dist)
80:         cmd._clean(f1, f2)
81: 
82:         for f in (f1, f2):
83:             self.assertFalse(os.path.exists(f))
84: 
85: def test_suite():
86:     return unittest.makeSuite(ConfigTestCase)
87: 
88: if __name__ == "__main__":
89:     run_unittest(test_suite())
90: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_35336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.config.')
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
import_35337 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support')

if (type(import_35337) is not StypyTypeError):

    if (import_35337 != 'pyd_module'):
        __import__(import_35337)
        sys_modules_35338 = sys.modules[import_35337]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', sys_modules_35338.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_35338, sys_modules_35338.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', import_35337)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.command.config import dump_file, config' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35339 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.config')

if (type(import_35339) is not StypyTypeError):

    if (import_35339 != 'pyd_module'):
        __import__(import_35339)
        sys_modules_35340 = sys.modules[import_35339]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.config', sys_modules_35340.module_type_store, module_type_store, ['dump_file', 'config'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_35340, sys_modules_35340.module_type_store, module_type_store)
    else:
        from distutils.command.config import dump_file, config

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.config', None, module_type_store, ['dump_file', 'config'], [dump_file, config])

else:
    # Assigning a type to the variable 'distutils.command.config' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.config', import_35339)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.tests import support' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35341 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests')

if (type(import_35341) is not StypyTypeError):

    if (import_35341 != 'pyd_module'):
        __import__(import_35341)
        sys_modules_35342 = sys.modules[import_35341]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', sys_modules_35342.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_35342, sys_modules_35342.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', import_35341)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils import log' statement (line 9)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils', None, module_type_store, ['log'], [log])

# Declaration of the 'ConfigTestCase' class
# Getting the type of 'support' (line 11)
support_35343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 21), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 11)
LoggingSilencer_35344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 21), support_35343, 'LoggingSilencer')
# Getting the type of 'support' (line 12)
support_35345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 'support')
# Obtaining the member 'TempdirManager' of a type (line 12)
TempdirManager_35346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 21), support_35345, 'TempdirManager')
# Getting the type of 'unittest' (line 13)
unittest_35347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 21), 'unittest')
# Obtaining the member 'TestCase' of a type (line 13)
TestCase_35348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 21), unittest_35347, 'TestCase')

class ConfigTestCase(LoggingSilencer_35344, TempdirManager_35346, TestCase_35348, ):

    @norecursion
    def _info(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_info'
        module_type_store = module_type_store.open_function_context('_info', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ConfigTestCase._info.__dict__.__setitem__('stypy_localization', localization)
        ConfigTestCase._info.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ConfigTestCase._info.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConfigTestCase._info.__dict__.__setitem__('stypy_function_name', 'ConfigTestCase._info')
        ConfigTestCase._info.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        ConfigTestCase._info.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        ConfigTestCase._info.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConfigTestCase._info.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConfigTestCase._info.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConfigTestCase._info.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConfigTestCase._info.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigTestCase._info', ['msg'], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_info', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_info(...)' code ##################

        
        
        # Call to splitlines(...): (line 16)
        # Processing the call keyword arguments (line 16)
        kwargs_35351 = {}
        # Getting the type of 'msg' (line 16)
        msg_35349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'msg', False)
        # Obtaining the member 'splitlines' of a type (line 16)
        splitlines_35350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 20), msg_35349, 'splitlines')
        # Calling splitlines(args, kwargs) (line 16)
        splitlines_call_result_35352 = invoke(stypy.reporting.localization.Localization(__file__, 16, 20), splitlines_35350, *[], **kwargs_35351)
        
        # Testing the type of a for loop iterable (line 16)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 16, 8), splitlines_call_result_35352)
        # Getting the type of the for loop variable (line 16)
        for_loop_var_35353 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 16, 8), splitlines_call_result_35352)
        # Assigning a type to the variable 'line' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'line', for_loop_var_35353)
        # SSA begins for a for statement (line 16)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'line' (line 17)
        line_35357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 30), 'line', False)
        # Processing the call keyword arguments (line 17)
        kwargs_35358 = {}
        # Getting the type of 'self' (line 17)
        self_35354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'self', False)
        # Obtaining the member '_logs' of a type (line 17)
        _logs_35355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 12), self_35354, '_logs')
        # Obtaining the member 'append' of a type (line 17)
        append_35356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 12), _logs_35355, 'append')
        # Calling append(args, kwargs) (line 17)
        append_call_result_35359 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), append_35356, *[line_35357], **kwargs_35358)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_info(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_info' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_35360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35360)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_info'
        return stypy_return_type_35360


    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ConfigTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        ConfigTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ConfigTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConfigTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'ConfigTestCase.setUp')
        ConfigTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        ConfigTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConfigTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConfigTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConfigTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConfigTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConfigTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigTestCase.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to setUp(...): (line 20)
        # Processing the call keyword arguments (line 20)
        kwargs_35367 = {}
        
        # Call to super(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'ConfigTestCase' (line 20)
        ConfigTestCase_35362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'ConfigTestCase', False)
        # Getting the type of 'self' (line 20)
        self_35363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 30), 'self', False)
        # Processing the call keyword arguments (line 20)
        kwargs_35364 = {}
        # Getting the type of 'super' (line 20)
        super_35361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'super', False)
        # Calling super(args, kwargs) (line 20)
        super_call_result_35365 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), super_35361, *[ConfigTestCase_35362, self_35363], **kwargs_35364)
        
        # Obtaining the member 'setUp' of a type (line 20)
        setUp_35366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), super_call_result_35365, 'setUp')
        # Calling setUp(args, kwargs) (line 20)
        setUp_call_result_35368 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), setUp_35366, *[], **kwargs_35367)
        
        
        # Assigning a List to a Attribute (line 21):
        
        # Assigning a List to a Attribute (line 21):
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_35369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        
        # Getting the type of 'self' (line 21)
        self_35370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self')
        # Setting the type of the member '_logs' of a type (line 21)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_35370, '_logs', list_35369)
        
        # Assigning a Attribute to a Attribute (line 22):
        
        # Assigning a Attribute to a Attribute (line 22):
        # Getting the type of 'log' (line 22)
        log_35371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'log')
        # Obtaining the member 'info' of a type (line 22)
        info_35372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 23), log_35371, 'info')
        # Getting the type of 'self' (line 22)
        self_35373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member 'old_log' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_35373, 'old_log', info_35372)
        
        # Assigning a Attribute to a Attribute (line 23):
        
        # Assigning a Attribute to a Attribute (line 23):
        # Getting the type of 'self' (line 23)
        self_35374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'self')
        # Obtaining the member '_info' of a type (line 23)
        _info_35375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 19), self_35374, '_info')
        # Getting the type of 'log' (line 23)
        log_35376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'log')
        # Setting the type of the member 'info' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), log_35376, 'info', _info_35375)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_35377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35377)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_35377


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ConfigTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        ConfigTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ConfigTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConfigTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'ConfigTestCase.tearDown')
        ConfigTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        ConfigTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConfigTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConfigTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConfigTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConfigTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConfigTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Attribute (line 26):
        
        # Assigning a Attribute to a Attribute (line 26):
        # Getting the type of 'self' (line 26)
        self_35378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 19), 'self')
        # Obtaining the member 'old_log' of a type (line 26)
        old_log_35379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 19), self_35378, 'old_log')
        # Getting the type of 'log' (line 26)
        log_35380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'log')
        # Setting the type of the member 'info' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), log_35380, 'info', old_log_35379)
        
        # Call to tearDown(...): (line 27)
        # Processing the call keyword arguments (line 27)
        kwargs_35387 = {}
        
        # Call to super(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'ConfigTestCase' (line 27)
        ConfigTestCase_35382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 'ConfigTestCase', False)
        # Getting the type of 'self' (line 27)
        self_35383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 30), 'self', False)
        # Processing the call keyword arguments (line 27)
        kwargs_35384 = {}
        # Getting the type of 'super' (line 27)
        super_35381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'super', False)
        # Calling super(args, kwargs) (line 27)
        super_call_result_35385 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), super_35381, *[ConfigTestCase_35382, self_35383], **kwargs_35384)
        
        # Obtaining the member 'tearDown' of a type (line 27)
        tearDown_35386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), super_call_result_35385, 'tearDown')
        # Calling tearDown(args, kwargs) (line 27)
        tearDown_call_result_35388 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), tearDown_35386, *[], **kwargs_35387)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_35389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35389)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_35389


    @norecursion
    def test_dump_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dump_file'
        module_type_store = module_type_store.open_function_context('test_dump_file', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ConfigTestCase.test_dump_file.__dict__.__setitem__('stypy_localization', localization)
        ConfigTestCase.test_dump_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ConfigTestCase.test_dump_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConfigTestCase.test_dump_file.__dict__.__setitem__('stypy_function_name', 'ConfigTestCase.test_dump_file')
        ConfigTestCase.test_dump_file.__dict__.__setitem__('stypy_param_names_list', [])
        ConfigTestCase.test_dump_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConfigTestCase.test_dump_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConfigTestCase.test_dump_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConfigTestCase.test_dump_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConfigTestCase.test_dump_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConfigTestCase.test_dump_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigTestCase.test_dump_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dump_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dump_file(...)' code ##################

        
        # Assigning a BinOp to a Name (line 30):
        
        # Assigning a BinOp to a Name (line 30):
        
        # Obtaining the type of the subscript
        int_35390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 47), 'int')
        
        # Call to splitext(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of '__file__' (line 30)
        file___35394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 37), '__file__', False)
        # Processing the call keyword arguments (line 30)
        kwargs_35395 = {}
        # Getting the type of 'os' (line 30)
        os_35391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 30)
        path_35392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 20), os_35391, 'path')
        # Obtaining the member 'splitext' of a type (line 30)
        splitext_35393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 20), path_35392, 'splitext')
        # Calling splitext(args, kwargs) (line 30)
        splitext_call_result_35396 = invoke(stypy.reporting.localization.Localization(__file__, 30, 20), splitext_35393, *[file___35394], **kwargs_35395)
        
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___35397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 20), splitext_call_result_35396, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_35398 = invoke(stypy.reporting.localization.Localization(__file__, 30, 20), getitem___35397, int_35390)
        
        str_35399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 52), 'str', '.py')
        # Applying the binary operator '+' (line 30)
        result_add_35400 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 20), '+', subscript_call_result_35398, str_35399)
        
        # Assigning a type to the variable 'this_file' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'this_file', result_add_35400)
        
        # Assigning a Call to a Name (line 31):
        
        # Assigning a Call to a Name (line 31):
        
        # Call to open(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'this_file' (line 31)
        this_file_35402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'this_file', False)
        # Processing the call keyword arguments (line 31)
        kwargs_35403 = {}
        # Getting the type of 'open' (line 31)
        open_35401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'open', False)
        # Calling open(args, kwargs) (line 31)
        open_call_result_35404 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), open_35401, *[this_file_35402], **kwargs_35403)
        
        # Assigning a type to the variable 'f' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'f', open_call_result_35404)
        
        # Try-finally block (line 32)
        
        # Assigning a Call to a Name (line 33):
        
        # Assigning a Call to a Name (line 33):
        
        # Call to len(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Call to readlines(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_35408 = {}
        # Getting the type of 'f' (line 33)
        f_35406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 27), 'f', False)
        # Obtaining the member 'readlines' of a type (line 33)
        readlines_35407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 27), f_35406, 'readlines')
        # Calling readlines(args, kwargs) (line 33)
        readlines_call_result_35409 = invoke(stypy.reporting.localization.Localization(__file__, 33, 27), readlines_35407, *[], **kwargs_35408)
        
        # Processing the call keyword arguments (line 33)
        kwargs_35410 = {}
        # Getting the type of 'len' (line 33)
        len_35405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'len', False)
        # Calling len(args, kwargs) (line 33)
        len_call_result_35411 = invoke(stypy.reporting.localization.Localization(__file__, 33, 23), len_35405, *[readlines_call_result_35409], **kwargs_35410)
        
        # Assigning a type to the variable 'numlines' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'numlines', len_call_result_35411)
        
        # finally branch of the try-finally block (line 32)
        
        # Call to close(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_35414 = {}
        # Getting the type of 'f' (line 35)
        f_35412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 35)
        close_35413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), f_35412, 'close')
        # Calling close(args, kwargs) (line 35)
        close_call_result_35415 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), close_35413, *[], **kwargs_35414)
        
        
        
        # Call to dump_file(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'this_file' (line 37)
        this_file_35417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'this_file', False)
        str_35418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 29), 'str', 'I am the header')
        # Processing the call keyword arguments (line 37)
        kwargs_35419 = {}
        # Getting the type of 'dump_file' (line 37)
        dump_file_35416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'dump_file', False)
        # Calling dump_file(args, kwargs) (line 37)
        dump_file_call_result_35420 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), dump_file_35416, *[this_file_35417, str_35418], **kwargs_35419)
        
        
        # Call to assertEqual(...): (line 38)
        # Processing the call arguments (line 38)
        
        # Call to len(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'self' (line 38)
        self_35424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'self', False)
        # Obtaining the member '_logs' of a type (line 38)
        _logs_35425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 29), self_35424, '_logs')
        # Processing the call keyword arguments (line 38)
        kwargs_35426 = {}
        # Getting the type of 'len' (line 38)
        len_35423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'len', False)
        # Calling len(args, kwargs) (line 38)
        len_call_result_35427 = invoke(stypy.reporting.localization.Localization(__file__, 38, 25), len_35423, *[_logs_35425], **kwargs_35426)
        
        # Getting the type of 'numlines' (line 38)
        numlines_35428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 42), 'numlines', False)
        int_35429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 51), 'int')
        # Applying the binary operator '+' (line 38)
        result_add_35430 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 42), '+', numlines_35428, int_35429)
        
        # Processing the call keyword arguments (line 38)
        kwargs_35431 = {}
        # Getting the type of 'self' (line 38)
        self_35421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 38)
        assertEqual_35422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_35421, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 38)
        assertEqual_call_result_35432 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), assertEqual_35422, *[len_call_result_35427, result_add_35430], **kwargs_35431)
        
        
        # ################# End of 'test_dump_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dump_file' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_35433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35433)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dump_file'
        return stypy_return_type_35433


    @norecursion
    def test_search_cpp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_search_cpp'
        module_type_store = module_type_store.open_function_context('test_search_cpp', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ConfigTestCase.test_search_cpp.__dict__.__setitem__('stypy_localization', localization)
        ConfigTestCase.test_search_cpp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ConfigTestCase.test_search_cpp.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConfigTestCase.test_search_cpp.__dict__.__setitem__('stypy_function_name', 'ConfigTestCase.test_search_cpp')
        ConfigTestCase.test_search_cpp.__dict__.__setitem__('stypy_param_names_list', [])
        ConfigTestCase.test_search_cpp.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConfigTestCase.test_search_cpp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConfigTestCase.test_search_cpp.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConfigTestCase.test_search_cpp.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConfigTestCase.test_search_cpp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConfigTestCase.test_search_cpp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigTestCase.test_search_cpp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_search_cpp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_search_cpp(...)' code ##################

        
        # Assigning a Call to a Tuple (line 42):
        
        # Assigning a Subscript to a Name (line 42):
        
        # Obtaining the type of the subscript
        int_35434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 8), 'int')
        
        # Call to create_dist(...): (line 42)
        # Processing the call keyword arguments (line 42)
        kwargs_35437 = {}
        # Getting the type of 'self' (line 42)
        self_35435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 42)
        create_dist_35436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), self_35435, 'create_dist')
        # Calling create_dist(args, kwargs) (line 42)
        create_dist_call_result_35438 = invoke(stypy.reporting.localization.Localization(__file__, 42, 24), create_dist_35436, *[], **kwargs_35437)
        
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___35439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), create_dist_call_result_35438, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_35440 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), getitem___35439, int_35434)
        
        # Assigning a type to the variable 'tuple_var_assignment_35330' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'tuple_var_assignment_35330', subscript_call_result_35440)
        
        # Assigning a Subscript to a Name (line 42):
        
        # Obtaining the type of the subscript
        int_35441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 8), 'int')
        
        # Call to create_dist(...): (line 42)
        # Processing the call keyword arguments (line 42)
        kwargs_35444 = {}
        # Getting the type of 'self' (line 42)
        self_35442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 42)
        create_dist_35443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), self_35442, 'create_dist')
        # Calling create_dist(args, kwargs) (line 42)
        create_dist_call_result_35445 = invoke(stypy.reporting.localization.Localization(__file__, 42, 24), create_dist_35443, *[], **kwargs_35444)
        
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___35446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), create_dist_call_result_35445, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_35447 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), getitem___35446, int_35441)
        
        # Assigning a type to the variable 'tuple_var_assignment_35331' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'tuple_var_assignment_35331', subscript_call_result_35447)
        
        # Assigning a Name to a Name (line 42):
        # Getting the type of 'tuple_var_assignment_35330' (line 42)
        tuple_var_assignment_35330_35448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'tuple_var_assignment_35330')
        # Assigning a type to the variable 'pkg_dir' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'pkg_dir', tuple_var_assignment_35330_35448)
        
        # Assigning a Name to a Name (line 42):
        # Getting the type of 'tuple_var_assignment_35331' (line 42)
        tuple_var_assignment_35331_35449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'tuple_var_assignment_35331')
        # Assigning a type to the variable 'dist' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'dist', tuple_var_assignment_35331_35449)
        
        # Assigning a Call to a Name (line 43):
        
        # Assigning a Call to a Name (line 43):
        
        # Call to config(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'dist' (line 43)
        dist_35451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'dist', False)
        # Processing the call keyword arguments (line 43)
        kwargs_35452 = {}
        # Getting the type of 'config' (line 43)
        config_35450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'config', False)
        # Calling config(args, kwargs) (line 43)
        config_call_result_35453 = invoke(stypy.reporting.localization.Localization(__file__, 43, 14), config_35450, *[dist_35451], **kwargs_35452)
        
        # Assigning a type to the variable 'cmd' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'cmd', config_call_result_35453)
        
        # Assigning a Call to a Name (line 46):
        
        # Assigning a Call to a Name (line 46):
        
        # Call to search_cpp(...): (line 46)
        # Processing the call keyword arguments (line 46)
        str_35456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 39), 'str', 'xxx')
        keyword_35457 = str_35456
        str_35458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 51), 'str', '/* xxx */')
        keyword_35459 = str_35458
        kwargs_35460 = {'body': keyword_35459, 'pattern': keyword_35457}
        # Getting the type of 'cmd' (line 46)
        cmd_35454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'cmd', False)
        # Obtaining the member 'search_cpp' of a type (line 46)
        search_cpp_35455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 16), cmd_35454, 'search_cpp')
        # Calling search_cpp(args, kwargs) (line 46)
        search_cpp_call_result_35461 = invoke(stypy.reporting.localization.Localization(__file__, 46, 16), search_cpp_35455, *[], **kwargs_35460)
        
        # Assigning a type to the variable 'match' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'match', search_cpp_call_result_35461)
        
        # Call to assertEqual(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'match' (line 47)
        match_35464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'match', False)
        int_35465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 32), 'int')
        # Processing the call keyword arguments (line 47)
        kwargs_35466 = {}
        # Getting the type of 'self' (line 47)
        self_35462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 47)
        assertEqual_35463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_35462, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 47)
        assertEqual_call_result_35467 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assertEqual_35463, *[match_35464, int_35465], **kwargs_35466)
        
        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to search_cpp(...): (line 49)
        # Processing the call keyword arguments (line 49)
        str_35470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 39), 'str', '_configtest')
        keyword_35471 = str_35470
        str_35472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 59), 'str', '/* xxx */')
        keyword_35473 = str_35472
        kwargs_35474 = {'body': keyword_35473, 'pattern': keyword_35471}
        # Getting the type of 'cmd' (line 49)
        cmd_35468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'cmd', False)
        # Obtaining the member 'search_cpp' of a type (line 49)
        search_cpp_35469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), cmd_35468, 'search_cpp')
        # Calling search_cpp(args, kwargs) (line 49)
        search_cpp_call_result_35475 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), search_cpp_35469, *[], **kwargs_35474)
        
        # Assigning a type to the variable 'match' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'match', search_cpp_call_result_35475)
        
        # Call to assertEqual(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'match' (line 50)
        match_35478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'match', False)
        int_35479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 32), 'int')
        # Processing the call keyword arguments (line 50)
        kwargs_35480 = {}
        # Getting the type of 'self' (line 50)
        self_35476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 50)
        assertEqual_35477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), self_35476, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 50)
        assertEqual_call_result_35481 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), assertEqual_35477, *[match_35478, int_35479], **kwargs_35480)
        
        
        # ################# End of 'test_search_cpp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_search_cpp' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_35482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35482)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_search_cpp'
        return stypy_return_type_35482


    @norecursion
    def test_finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_finalize_options'
        module_type_store = module_type_store.open_function_context('test_finalize_options', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ConfigTestCase.test_finalize_options.__dict__.__setitem__('stypy_localization', localization)
        ConfigTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ConfigTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConfigTestCase.test_finalize_options.__dict__.__setitem__('stypy_function_name', 'ConfigTestCase.test_finalize_options')
        ConfigTestCase.test_finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        ConfigTestCase.test_finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConfigTestCase.test_finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConfigTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConfigTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConfigTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConfigTestCase.test_finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigTestCase.test_finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 55):
        
        # Assigning a Subscript to a Name (line 55):
        
        # Obtaining the type of the subscript
        int_35483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'int')
        
        # Call to create_dist(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_35486 = {}
        # Getting the type of 'self' (line 55)
        self_35484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 55)
        create_dist_35485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 24), self_35484, 'create_dist')
        # Calling create_dist(args, kwargs) (line 55)
        create_dist_call_result_35487 = invoke(stypy.reporting.localization.Localization(__file__, 55, 24), create_dist_35485, *[], **kwargs_35486)
        
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___35488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), create_dist_call_result_35487, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_35489 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), getitem___35488, int_35483)
        
        # Assigning a type to the variable 'tuple_var_assignment_35332' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_35332', subscript_call_result_35489)
        
        # Assigning a Subscript to a Name (line 55):
        
        # Obtaining the type of the subscript
        int_35490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'int')
        
        # Call to create_dist(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_35493 = {}
        # Getting the type of 'self' (line 55)
        self_35491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 55)
        create_dist_35492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 24), self_35491, 'create_dist')
        # Calling create_dist(args, kwargs) (line 55)
        create_dist_call_result_35494 = invoke(stypy.reporting.localization.Localization(__file__, 55, 24), create_dist_35492, *[], **kwargs_35493)
        
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___35495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), create_dist_call_result_35494, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_35496 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), getitem___35495, int_35490)
        
        # Assigning a type to the variable 'tuple_var_assignment_35333' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_35333', subscript_call_result_35496)
        
        # Assigning a Name to a Name (line 55):
        # Getting the type of 'tuple_var_assignment_35332' (line 55)
        tuple_var_assignment_35332_35497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_35332')
        # Assigning a type to the variable 'pkg_dir' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'pkg_dir', tuple_var_assignment_35332_35497)
        
        # Assigning a Name to a Name (line 55):
        # Getting the type of 'tuple_var_assignment_35333' (line 55)
        tuple_var_assignment_35333_35498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'tuple_var_assignment_35333')
        # Assigning a type to the variable 'dist' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'dist', tuple_var_assignment_35333_35498)
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to config(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'dist' (line 56)
        dist_35500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'dist', False)
        # Processing the call keyword arguments (line 56)
        kwargs_35501 = {}
        # Getting the type of 'config' (line 56)
        config_35499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'config', False)
        # Calling config(args, kwargs) (line 56)
        config_call_result_35502 = invoke(stypy.reporting.localization.Localization(__file__, 56, 14), config_35499, *[dist_35500], **kwargs_35501)
        
        # Assigning a type to the variable 'cmd' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'cmd', config_call_result_35502)
        
        # Assigning a BinOp to a Attribute (line 57):
        
        # Assigning a BinOp to a Attribute (line 57):
        str_35503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 27), 'str', 'one%stwo')
        # Getting the type of 'os' (line 57)
        os_35504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 40), 'os')
        # Obtaining the member 'pathsep' of a type (line 57)
        pathsep_35505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 40), os_35504, 'pathsep')
        # Applying the binary operator '%' (line 57)
        result_mod_35506 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 27), '%', str_35503, pathsep_35505)
        
        # Getting the type of 'cmd' (line 57)
        cmd_35507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'cmd')
        # Setting the type of the member 'include_dirs' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), cmd_35507, 'include_dirs', result_mod_35506)
        
        # Assigning a Str to a Attribute (line 58):
        
        # Assigning a Str to a Attribute (line 58):
        str_35508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 24), 'str', 'one')
        # Getting the type of 'cmd' (line 58)
        cmd_35509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'cmd')
        # Setting the type of the member 'libraries' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), cmd_35509, 'libraries', str_35508)
        
        # Assigning a BinOp to a Attribute (line 59):
        
        # Assigning a BinOp to a Attribute (line 59):
        str_35510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 27), 'str', 'three%sfour')
        # Getting the type of 'os' (line 59)
        os_35511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 43), 'os')
        # Obtaining the member 'pathsep' of a type (line 59)
        pathsep_35512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 43), os_35511, 'pathsep')
        # Applying the binary operator '%' (line 59)
        result_mod_35513 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 27), '%', str_35510, pathsep_35512)
        
        # Getting the type of 'cmd' (line 59)
        cmd_35514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'cmd')
        # Setting the type of the member 'library_dirs' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), cmd_35514, 'library_dirs', result_mod_35513)
        
        # Call to ensure_finalized(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_35517 = {}
        # Getting the type of 'cmd' (line 60)
        cmd_35515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 60)
        ensure_finalized_35516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), cmd_35515, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 60)
        ensure_finalized_call_result_35518 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), ensure_finalized_35516, *[], **kwargs_35517)
        
        
        # Call to assertEqual(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'cmd' (line 62)
        cmd_35521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'cmd', False)
        # Obtaining the member 'include_dirs' of a type (line 62)
        include_dirs_35522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 25), cmd_35521, 'include_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_35523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        str_35524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 44), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 43), list_35523, str_35524)
        # Adding element type (line 62)
        str_35525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 51), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 43), list_35523, str_35525)
        
        # Processing the call keyword arguments (line 62)
        kwargs_35526 = {}
        # Getting the type of 'self' (line 62)
        self_35519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 62)
        assertEqual_35520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_35519, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 62)
        assertEqual_call_result_35527 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), assertEqual_35520, *[include_dirs_35522, list_35523], **kwargs_35526)
        
        
        # Call to assertEqual(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'cmd' (line 63)
        cmd_35530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 25), 'cmd', False)
        # Obtaining the member 'libraries' of a type (line 63)
        libraries_35531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 25), cmd_35530, 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_35532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        str_35533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 41), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 40), list_35532, str_35533)
        
        # Processing the call keyword arguments (line 63)
        kwargs_35534 = {}
        # Getting the type of 'self' (line 63)
        self_35528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 63)
        assertEqual_35529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_35528, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 63)
        assertEqual_call_result_35535 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), assertEqual_35529, *[libraries_35531, list_35532], **kwargs_35534)
        
        
        # Call to assertEqual(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'cmd' (line 64)
        cmd_35538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'cmd', False)
        # Obtaining the member 'library_dirs' of a type (line 64)
        library_dirs_35539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 25), cmd_35538, 'library_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_35540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        str_35541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 44), 'str', 'three')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 43), list_35540, str_35541)
        # Adding element type (line 64)
        str_35542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 53), 'str', 'four')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 43), list_35540, str_35542)
        
        # Processing the call keyword arguments (line 64)
        kwargs_35543 = {}
        # Getting the type of 'self' (line 64)
        self_35536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 64)
        assertEqual_35537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_35536, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 64)
        assertEqual_call_result_35544 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assertEqual_35537, *[library_dirs_35539, list_35540], **kwargs_35543)
        
        
        # ################# End of 'test_finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_35545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35545)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_finalize_options'
        return stypy_return_type_35545


    @norecursion
    def test_clean(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_clean'
        module_type_store = module_type_store.open_function_context('test_clean', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ConfigTestCase.test_clean.__dict__.__setitem__('stypy_localization', localization)
        ConfigTestCase.test_clean.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ConfigTestCase.test_clean.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConfigTestCase.test_clean.__dict__.__setitem__('stypy_function_name', 'ConfigTestCase.test_clean')
        ConfigTestCase.test_clean.__dict__.__setitem__('stypy_param_names_list', [])
        ConfigTestCase.test_clean.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConfigTestCase.test_clean.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConfigTestCase.test_clean.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConfigTestCase.test_clean.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConfigTestCase.test_clean.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConfigTestCase.test_clean.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigTestCase.test_clean', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_clean', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_clean(...)' code ##################

        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to mkdtemp(...): (line 68)
        # Processing the call keyword arguments (line 68)
        kwargs_35548 = {}
        # Getting the type of 'self' (line 68)
        self_35546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 68)
        mkdtemp_35547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 18), self_35546, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 68)
        mkdtemp_call_result_35549 = invoke(stypy.reporting.localization.Localization(__file__, 68, 18), mkdtemp_35547, *[], **kwargs_35548)
        
        # Assigning a type to the variable 'tmp_dir' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tmp_dir', mkdtemp_call_result_35549)
        
        # Assigning a Call to a Name (line 69):
        
        # Assigning a Call to a Name (line 69):
        
        # Call to join(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'tmp_dir' (line 69)
        tmp_dir_35553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'tmp_dir', False)
        str_35554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 35), 'str', 'one')
        # Processing the call keyword arguments (line 69)
        kwargs_35555 = {}
        # Getting the type of 'os' (line 69)
        os_35550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'os', False)
        # Obtaining the member 'path' of a type (line 69)
        path_35551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), os_35550, 'path')
        # Obtaining the member 'join' of a type (line 69)
        join_35552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), path_35551, 'join')
        # Calling join(args, kwargs) (line 69)
        join_call_result_35556 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), join_35552, *[tmp_dir_35553, str_35554], **kwargs_35555)
        
        # Assigning a type to the variable 'f1' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'f1', join_call_result_35556)
        
        # Assigning a Call to a Name (line 70):
        
        # Assigning a Call to a Name (line 70):
        
        # Call to join(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'tmp_dir' (line 70)
        tmp_dir_35560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 26), 'tmp_dir', False)
        str_35561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 35), 'str', 'two')
        # Processing the call keyword arguments (line 70)
        kwargs_35562 = {}
        # Getting the type of 'os' (line 70)
        os_35557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 13), 'os', False)
        # Obtaining the member 'path' of a type (line 70)
        path_35558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 13), os_35557, 'path')
        # Obtaining the member 'join' of a type (line 70)
        join_35559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 13), path_35558, 'join')
        # Calling join(args, kwargs) (line 70)
        join_call_result_35563 = invoke(stypy.reporting.localization.Localization(__file__, 70, 13), join_35559, *[tmp_dir_35560, str_35561], **kwargs_35562)
        
        # Assigning a type to the variable 'f2' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'f2', join_call_result_35563)
        
        # Call to write_file(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'f1' (line 72)
        f1_35566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'f1', False)
        str_35567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 28), 'str', 'xxx')
        # Processing the call keyword arguments (line 72)
        kwargs_35568 = {}
        # Getting the type of 'self' (line 72)
        self_35564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 72)
        write_file_35565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_35564, 'write_file')
        # Calling write_file(args, kwargs) (line 72)
        write_file_call_result_35569 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), write_file_35565, *[f1_35566, str_35567], **kwargs_35568)
        
        
        # Call to write_file(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'f2' (line 73)
        f2_35572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'f2', False)
        str_35573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'str', 'xxx')
        # Processing the call keyword arguments (line 73)
        kwargs_35574 = {}
        # Getting the type of 'self' (line 73)
        self_35570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 73)
        write_file_35571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_35570, 'write_file')
        # Calling write_file(args, kwargs) (line 73)
        write_file_call_result_35575 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), write_file_35571, *[f2_35572, str_35573], **kwargs_35574)
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_35576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        # Getting the type of 'f1' (line 75)
        f1_35577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'f1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 18), tuple_35576, f1_35577)
        # Adding element type (line 75)
        # Getting the type of 'f2' (line 75)
        f2_35578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'f2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 18), tuple_35576, f2_35578)
        
        # Testing the type of a for loop iterable (line 75)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 75, 8), tuple_35576)
        # Getting the type of the for loop variable (line 75)
        for_loop_var_35579 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 75, 8), tuple_35576)
        # Assigning a type to the variable 'f' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'f', for_loop_var_35579)
        # SSA begins for a for statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertTrue(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Call to exists(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'f' (line 76)
        f_35585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 43), 'f', False)
        # Processing the call keyword arguments (line 76)
        kwargs_35586 = {}
        # Getting the type of 'os' (line 76)
        os_35582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 76)
        path_35583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 28), os_35582, 'path')
        # Obtaining the member 'exists' of a type (line 76)
        exists_35584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 28), path_35583, 'exists')
        # Calling exists(args, kwargs) (line 76)
        exists_call_result_35587 = invoke(stypy.reporting.localization.Localization(__file__, 76, 28), exists_35584, *[f_35585], **kwargs_35586)
        
        # Processing the call keyword arguments (line 76)
        kwargs_35588 = {}
        # Getting the type of 'self' (line 76)
        self_35580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 76)
        assertTrue_35581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), self_35580, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 76)
        assertTrue_call_result_35589 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), assertTrue_35581, *[exists_call_result_35587], **kwargs_35588)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 78):
        
        # Assigning a Subscript to a Name (line 78):
        
        # Obtaining the type of the subscript
        int_35590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 8), 'int')
        
        # Call to create_dist(...): (line 78)
        # Processing the call keyword arguments (line 78)
        kwargs_35593 = {}
        # Getting the type of 'self' (line 78)
        self_35591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 78)
        create_dist_35592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 24), self_35591, 'create_dist')
        # Calling create_dist(args, kwargs) (line 78)
        create_dist_call_result_35594 = invoke(stypy.reporting.localization.Localization(__file__, 78, 24), create_dist_35592, *[], **kwargs_35593)
        
        # Obtaining the member '__getitem__' of a type (line 78)
        getitem___35595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), create_dist_call_result_35594, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 78)
        subscript_call_result_35596 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), getitem___35595, int_35590)
        
        # Assigning a type to the variable 'tuple_var_assignment_35334' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'tuple_var_assignment_35334', subscript_call_result_35596)
        
        # Assigning a Subscript to a Name (line 78):
        
        # Obtaining the type of the subscript
        int_35597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 8), 'int')
        
        # Call to create_dist(...): (line 78)
        # Processing the call keyword arguments (line 78)
        kwargs_35600 = {}
        # Getting the type of 'self' (line 78)
        self_35598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 78)
        create_dist_35599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 24), self_35598, 'create_dist')
        # Calling create_dist(args, kwargs) (line 78)
        create_dist_call_result_35601 = invoke(stypy.reporting.localization.Localization(__file__, 78, 24), create_dist_35599, *[], **kwargs_35600)
        
        # Obtaining the member '__getitem__' of a type (line 78)
        getitem___35602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), create_dist_call_result_35601, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 78)
        subscript_call_result_35603 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), getitem___35602, int_35597)
        
        # Assigning a type to the variable 'tuple_var_assignment_35335' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'tuple_var_assignment_35335', subscript_call_result_35603)
        
        # Assigning a Name to a Name (line 78):
        # Getting the type of 'tuple_var_assignment_35334' (line 78)
        tuple_var_assignment_35334_35604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'tuple_var_assignment_35334')
        # Assigning a type to the variable 'pkg_dir' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'pkg_dir', tuple_var_assignment_35334_35604)
        
        # Assigning a Name to a Name (line 78):
        # Getting the type of 'tuple_var_assignment_35335' (line 78)
        tuple_var_assignment_35335_35605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'tuple_var_assignment_35335')
        # Assigning a type to the variable 'dist' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'dist', tuple_var_assignment_35335_35605)
        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to config(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'dist' (line 79)
        dist_35607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'dist', False)
        # Processing the call keyword arguments (line 79)
        kwargs_35608 = {}
        # Getting the type of 'config' (line 79)
        config_35606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 14), 'config', False)
        # Calling config(args, kwargs) (line 79)
        config_call_result_35609 = invoke(stypy.reporting.localization.Localization(__file__, 79, 14), config_35606, *[dist_35607], **kwargs_35608)
        
        # Assigning a type to the variable 'cmd' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'cmd', config_call_result_35609)
        
        # Call to _clean(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'f1' (line 80)
        f1_35612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'f1', False)
        # Getting the type of 'f2' (line 80)
        f2_35613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'f2', False)
        # Processing the call keyword arguments (line 80)
        kwargs_35614 = {}
        # Getting the type of 'cmd' (line 80)
        cmd_35610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'cmd', False)
        # Obtaining the member '_clean' of a type (line 80)
        _clean_35611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), cmd_35610, '_clean')
        # Calling _clean(args, kwargs) (line 80)
        _clean_call_result_35615 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), _clean_35611, *[f1_35612, f2_35613], **kwargs_35614)
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 82)
        tuple_35616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 82)
        # Adding element type (line 82)
        # Getting the type of 'f1' (line 82)
        f1_35617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'f1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 18), tuple_35616, f1_35617)
        # Adding element type (line 82)
        # Getting the type of 'f2' (line 82)
        f2_35618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'f2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 18), tuple_35616, f2_35618)
        
        # Testing the type of a for loop iterable (line 82)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 82, 8), tuple_35616)
        # Getting the type of the for loop variable (line 82)
        for_loop_var_35619 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 82, 8), tuple_35616)
        # Assigning a type to the variable 'f' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'f', for_loop_var_35619)
        # SSA begins for a for statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertFalse(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Call to exists(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'f' (line 83)
        f_35625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 44), 'f', False)
        # Processing the call keyword arguments (line 83)
        kwargs_35626 = {}
        # Getting the type of 'os' (line 83)
        os_35622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 83)
        path_35623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 29), os_35622, 'path')
        # Obtaining the member 'exists' of a type (line 83)
        exists_35624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 29), path_35623, 'exists')
        # Calling exists(args, kwargs) (line 83)
        exists_call_result_35627 = invoke(stypy.reporting.localization.Localization(__file__, 83, 29), exists_35624, *[f_35625], **kwargs_35626)
        
        # Processing the call keyword arguments (line 83)
        kwargs_35628 = {}
        # Getting the type of 'self' (line 83)
        self_35620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 83)
        assertFalse_35621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), self_35620, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 83)
        assertFalse_call_result_35629 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), assertFalse_35621, *[exists_call_result_35627], **kwargs_35628)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_clean(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_clean' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_35630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35630)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_clean'
        return stypy_return_type_35630


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ConfigTestCase' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'ConfigTestCase', ConfigTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 85, 0, False)
    
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

    
    # Call to makeSuite(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'ConfigTestCase' (line 86)
    ConfigTestCase_35633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 30), 'ConfigTestCase', False)
    # Processing the call keyword arguments (line 86)
    kwargs_35634 = {}
    # Getting the type of 'unittest' (line 86)
    unittest_35631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 86)
    makeSuite_35632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 11), unittest_35631, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 86)
    makeSuite_call_result_35635 = invoke(stypy.reporting.localization.Localization(__file__, 86, 11), makeSuite_35632, *[ConfigTestCase_35633], **kwargs_35634)
    
    # Assigning a type to the variable 'stypy_return_type' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type', makeSuite_call_result_35635)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_35636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35636)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_35636

# Assigning a type to the variable 'test_suite' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 89)
    # Processing the call arguments (line 89)
    
    # Call to test_suite(...): (line 89)
    # Processing the call keyword arguments (line 89)
    kwargs_35639 = {}
    # Getting the type of 'test_suite' (line 89)
    test_suite_35638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 89)
    test_suite_call_result_35640 = invoke(stypy.reporting.localization.Localization(__file__, 89, 17), test_suite_35638, *[], **kwargs_35639)
    
    # Processing the call keyword arguments (line 89)
    kwargs_35641 = {}
    # Getting the type of 'run_unittest' (line 89)
    run_unittest_35637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 89)
    run_unittest_call_result_35642 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), run_unittest_35637, *[test_suite_call_result_35640], **kwargs_35641)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
