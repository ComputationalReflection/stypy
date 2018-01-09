
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.cmd.'''
2: import unittest
3: import os
4: from test.test_support import captured_stdout, run_unittest
5: 
6: from distutils.cmd import Command
7: from distutils.dist import Distribution
8: from distutils.errors import DistutilsOptionError
9: from distutils import debug
10: 
11: class MyCmd(Command):
12:     def initialize_options(self):
13:         pass
14: 
15: class CommandTestCase(unittest.TestCase):
16: 
17:     def setUp(self):
18:         dist = Distribution()
19:         self.cmd = MyCmd(dist)
20: 
21:     def test_ensure_string_list(self):
22: 
23:         cmd = self.cmd
24:         cmd.not_string_list = ['one', 2, 'three']
25:         cmd.yes_string_list = ['one', 'two', 'three']
26:         cmd.not_string_list2 = object()
27:         cmd.yes_string_list2 = 'ok'
28:         cmd.ensure_string_list('yes_string_list')
29:         cmd.ensure_string_list('yes_string_list2')
30: 
31:         self.assertRaises(DistutilsOptionError,
32:                           cmd.ensure_string_list, 'not_string_list')
33: 
34:         self.assertRaises(DistutilsOptionError,
35:                           cmd.ensure_string_list, 'not_string_list2')
36: 
37:         cmd.option1 = 'ok,dok'
38:         cmd.ensure_string_list('option1')
39:         self.assertEqual(cmd.option1, ['ok', 'dok'])
40: 
41:         cmd.option2 = ['xxx', 'www']
42:         cmd.ensure_string_list('option2')
43: 
44:         cmd.option3 = ['ok', 2]
45:         self.assertRaises(DistutilsOptionError, cmd.ensure_string_list,
46:                           'option3')
47: 
48: 
49:     def test_make_file(self):
50: 
51:         cmd = self.cmd
52: 
53:         # making sure it raises when infiles is not a string or a list/tuple
54:         self.assertRaises(TypeError, cmd.make_file,
55:                           infiles=1, outfile='', func='func', args=())
56: 
57:         # making sure execute gets called properly
58:         def _execute(func, args, exec_msg, level):
59:             self.assertEqual(exec_msg, 'generating out from in')
60:         cmd.force = True
61:         cmd.execute = _execute
62:         cmd.make_file(infiles='in', outfile='out', func='func', args=())
63: 
64:     def test_dump_options(self):
65: 
66:         msgs = []
67:         def _announce(msg, level):
68:             msgs.append(msg)
69:         cmd = self.cmd
70:         cmd.announce = _announce
71:         cmd.option1 = 1
72:         cmd.option2 = 1
73:         cmd.user_options = [('option1', '', ''), ('option2', '', '')]
74:         cmd.dump_options()
75: 
76:         wanted = ["command options for 'MyCmd':", '  option1 = 1',
77:                   '  option2 = 1']
78:         self.assertEqual(msgs, wanted)
79: 
80:     def test_ensure_string(self):
81:         cmd = self.cmd
82:         cmd.option1 = 'ok'
83:         cmd.ensure_string('option1')
84: 
85:         cmd.option2 = None
86:         cmd.ensure_string('option2', 'xxx')
87:         self.assertTrue(hasattr(cmd, 'option2'))
88: 
89:         cmd.option3 = 1
90:         self.assertRaises(DistutilsOptionError, cmd.ensure_string, 'option3')
91: 
92:     def test_ensure_filename(self):
93:         cmd = self.cmd
94:         cmd.option1 = __file__
95:         cmd.ensure_filename('option1')
96:         cmd.option2 = 'xxx'
97:         self.assertRaises(DistutilsOptionError, cmd.ensure_filename, 'option2')
98: 
99:     def test_ensure_dirname(self):
100:         cmd = self.cmd
101:         cmd.option1 = os.path.dirname(__file__) or os.curdir
102:         cmd.ensure_dirname('option1')
103:         cmd.option2 = 'xxx'
104:         self.assertRaises(DistutilsOptionError, cmd.ensure_dirname, 'option2')
105: 
106:     def test_debug_print(self):
107:         cmd = self.cmd
108:         with captured_stdout() as stdout:
109:             cmd.debug_print('xxx')
110:         stdout.seek(0)
111:         self.assertEqual(stdout.read(), '')
112: 
113:         debug.DEBUG = True
114:         try:
115:             with captured_stdout() as stdout:
116:                 cmd.debug_print('xxx')
117:             stdout.seek(0)
118:             self.assertEqual(stdout.read(), 'xxx\n')
119:         finally:
120:             debug.DEBUG = False
121: 
122: def test_suite():
123:     return unittest.makeSuite(CommandTestCase)
124: 
125: if __name__ == '__main__':
126:     run_unittest(test_suite())
127: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_34743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.cmd.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import unittest' statement (line 2)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from test.test_support import captured_stdout, run_unittest' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34744 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support')

if (type(import_34744) is not StypyTypeError):

    if (import_34744 != 'pyd_module'):
        __import__(import_34744)
        sys_modules_34745 = sys.modules[import_34744]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', sys_modules_34745.module_type_store, module_type_store, ['captured_stdout', 'run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_34745, sys_modules_34745.module_type_store, module_type_store)
    else:
        from test.test_support import captured_stdout, run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', None, module_type_store, ['captured_stdout', 'run_unittest'], [captured_stdout, run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', import_34744)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.cmd import Command' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34746 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.cmd')

if (type(import_34746) is not StypyTypeError):

    if (import_34746 != 'pyd_module'):
        __import__(import_34746)
        sys_modules_34747 = sys.modules[import_34746]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.cmd', sys_modules_34747.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_34747, sys_modules_34747.module_type_store, module_type_store)
    else:
        from distutils.cmd import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.cmd', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.cmd' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.cmd', import_34746)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.dist import Distribution' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34748 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.dist')

if (type(import_34748) is not StypyTypeError):

    if (import_34748 != 'pyd_module'):
        __import__(import_34748)
        sys_modules_34749 = sys.modules[import_34748]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.dist', sys_modules_34749.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_34749, sys_modules_34749.module_type_store, module_type_store)
    else:
        from distutils.dist import Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.dist', None, module_type_store, ['Distribution'], [Distribution])

else:
    # Assigning a type to the variable 'distutils.dist' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.dist', import_34748)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.errors import DistutilsOptionError' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34750 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.errors')

if (type(import_34750) is not StypyTypeError):

    if (import_34750 != 'pyd_module'):
        __import__(import_34750)
        sys_modules_34751 = sys.modules[import_34750]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.errors', sys_modules_34751.module_type_store, module_type_store, ['DistutilsOptionError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_34751, sys_modules_34751.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsOptionError

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.errors', None, module_type_store, ['DistutilsOptionError'], [DistutilsOptionError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.errors', import_34750)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils import debug' statement (line 9)
try:
    from distutils import debug

except:
    debug = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils', None, module_type_store, ['debug'], [debug])

# Declaration of the 'MyCmd' class
# Getting the type of 'Command' (line 11)
Command_34752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'Command')

class MyCmd(Command_34752, ):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MyCmd.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        MyCmd.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MyCmd.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        MyCmd.initialize_options.__dict__.__setitem__('stypy_function_name', 'MyCmd.initialize_options')
        MyCmd.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        MyCmd.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        MyCmd.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MyCmd.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        MyCmd.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        MyCmd.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MyCmd.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MyCmd.initialize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'initialize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'initialize_options(...)' code ##################

        pass
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_34753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34753)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_34753


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MyCmd.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MyCmd' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'MyCmd', MyCmd)
# Declaration of the 'CommandTestCase' class
# Getting the type of 'unittest' (line 15)
unittest_34754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'unittest')
# Obtaining the member 'TestCase' of a type (line 15)
TestCase_34755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 22), unittest_34754, 'TestCase')

class CommandTestCase(TestCase_34755, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CommandTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        CommandTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CommandTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        CommandTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'CommandTestCase.setUp')
        CommandTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        CommandTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        CommandTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CommandTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        CommandTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        CommandTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CommandTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommandTestCase.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 18):
        
        # Call to Distribution(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_34757 = {}
        # Getting the type of 'Distribution' (line 18)
        Distribution_34756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 18)
        Distribution_call_result_34758 = invoke(stypy.reporting.localization.Localization(__file__, 18, 15), Distribution_34756, *[], **kwargs_34757)
        
        # Assigning a type to the variable 'dist' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'dist', Distribution_call_result_34758)
        
        # Assigning a Call to a Attribute (line 19):
        
        # Call to MyCmd(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'dist' (line 19)
        dist_34760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'dist', False)
        # Processing the call keyword arguments (line 19)
        kwargs_34761 = {}
        # Getting the type of 'MyCmd' (line 19)
        MyCmd_34759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'MyCmd', False)
        # Calling MyCmd(args, kwargs) (line 19)
        MyCmd_call_result_34762 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), MyCmd_34759, *[dist_34760], **kwargs_34761)
        
        # Getting the type of 'self' (line 19)
        self_34763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self')
        # Setting the type of the member 'cmd' of a type (line 19)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), self_34763, 'cmd', MyCmd_call_result_34762)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_34764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34764)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_34764


    @norecursion
    def test_ensure_string_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ensure_string_list'
        module_type_store = module_type_store.open_function_context('test_ensure_string_list', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CommandTestCase.test_ensure_string_list.__dict__.__setitem__('stypy_localization', localization)
        CommandTestCase.test_ensure_string_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CommandTestCase.test_ensure_string_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        CommandTestCase.test_ensure_string_list.__dict__.__setitem__('stypy_function_name', 'CommandTestCase.test_ensure_string_list')
        CommandTestCase.test_ensure_string_list.__dict__.__setitem__('stypy_param_names_list', [])
        CommandTestCase.test_ensure_string_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        CommandTestCase.test_ensure_string_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CommandTestCase.test_ensure_string_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        CommandTestCase.test_ensure_string_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        CommandTestCase.test_ensure_string_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CommandTestCase.test_ensure_string_list.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommandTestCase.test_ensure_string_list', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ensure_string_list', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ensure_string_list(...)' code ##################

        
        # Assigning a Attribute to a Name (line 23):
        # Getting the type of 'self' (line 23)
        self_34765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 14), 'self')
        # Obtaining the member 'cmd' of a type (line 23)
        cmd_34766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 14), self_34765, 'cmd')
        # Assigning a type to the variable 'cmd' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'cmd', cmd_34766)
        
        # Assigning a List to a Attribute (line 24):
        
        # Obtaining an instance of the builtin type 'list' (line 24)
        list_34767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 24)
        # Adding element type (line 24)
        str_34768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 31), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 30), list_34767, str_34768)
        # Adding element type (line 24)
        int_34769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 30), list_34767, int_34769)
        # Adding element type (line 24)
        str_34770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 41), 'str', 'three')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 30), list_34767, str_34770)
        
        # Getting the type of 'cmd' (line 24)
        cmd_34771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'cmd')
        # Setting the type of the member 'not_string_list' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), cmd_34771, 'not_string_list', list_34767)
        
        # Assigning a List to a Attribute (line 25):
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_34772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        str_34773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 31), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 30), list_34772, str_34773)
        # Adding element type (line 25)
        str_34774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 38), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 30), list_34772, str_34774)
        # Adding element type (line 25)
        str_34775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 45), 'str', 'three')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 30), list_34772, str_34775)
        
        # Getting the type of 'cmd' (line 25)
        cmd_34776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'cmd')
        # Setting the type of the member 'yes_string_list' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), cmd_34776, 'yes_string_list', list_34772)
        
        # Assigning a Call to a Attribute (line 26):
        
        # Call to object(...): (line 26)
        # Processing the call keyword arguments (line 26)
        kwargs_34778 = {}
        # Getting the type of 'object' (line 26)
        object_34777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'object', False)
        # Calling object(args, kwargs) (line 26)
        object_call_result_34779 = invoke(stypy.reporting.localization.Localization(__file__, 26, 31), object_34777, *[], **kwargs_34778)
        
        # Getting the type of 'cmd' (line 26)
        cmd_34780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'cmd')
        # Setting the type of the member 'not_string_list2' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), cmd_34780, 'not_string_list2', object_call_result_34779)
        
        # Assigning a Str to a Attribute (line 27):
        str_34781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 31), 'str', 'ok')
        # Getting the type of 'cmd' (line 27)
        cmd_34782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'cmd')
        # Setting the type of the member 'yes_string_list2' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), cmd_34782, 'yes_string_list2', str_34781)
        
        # Call to ensure_string_list(...): (line 28)
        # Processing the call arguments (line 28)
        str_34785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 31), 'str', 'yes_string_list')
        # Processing the call keyword arguments (line 28)
        kwargs_34786 = {}
        # Getting the type of 'cmd' (line 28)
        cmd_34783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'cmd', False)
        # Obtaining the member 'ensure_string_list' of a type (line 28)
        ensure_string_list_34784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), cmd_34783, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 28)
        ensure_string_list_call_result_34787 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), ensure_string_list_34784, *[str_34785], **kwargs_34786)
        
        
        # Call to ensure_string_list(...): (line 29)
        # Processing the call arguments (line 29)
        str_34790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 31), 'str', 'yes_string_list2')
        # Processing the call keyword arguments (line 29)
        kwargs_34791 = {}
        # Getting the type of 'cmd' (line 29)
        cmd_34788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'cmd', False)
        # Obtaining the member 'ensure_string_list' of a type (line 29)
        ensure_string_list_34789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), cmd_34788, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 29)
        ensure_string_list_call_result_34792 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), ensure_string_list_34789, *[str_34790], **kwargs_34791)
        
        
        # Call to assertRaises(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'DistutilsOptionError' (line 31)
        DistutilsOptionError_34795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'DistutilsOptionError', False)
        # Getting the type of 'cmd' (line 32)
        cmd_34796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'cmd', False)
        # Obtaining the member 'ensure_string_list' of a type (line 32)
        ensure_string_list_34797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 26), cmd_34796, 'ensure_string_list')
        str_34798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 50), 'str', 'not_string_list')
        # Processing the call keyword arguments (line 31)
        kwargs_34799 = {}
        # Getting the type of 'self' (line 31)
        self_34793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 31)
        assertRaises_34794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_34793, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 31)
        assertRaises_call_result_34800 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), assertRaises_34794, *[DistutilsOptionError_34795, ensure_string_list_34797, str_34798], **kwargs_34799)
        
        
        # Call to assertRaises(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'DistutilsOptionError' (line 34)
        DistutilsOptionError_34803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'DistutilsOptionError', False)
        # Getting the type of 'cmd' (line 35)
        cmd_34804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'cmd', False)
        # Obtaining the member 'ensure_string_list' of a type (line 35)
        ensure_string_list_34805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 26), cmd_34804, 'ensure_string_list')
        str_34806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 50), 'str', 'not_string_list2')
        # Processing the call keyword arguments (line 34)
        kwargs_34807 = {}
        # Getting the type of 'self' (line 34)
        self_34801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 34)
        assertRaises_34802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_34801, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 34)
        assertRaises_call_result_34808 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assertRaises_34802, *[DistutilsOptionError_34803, ensure_string_list_34805, str_34806], **kwargs_34807)
        
        
        # Assigning a Str to a Attribute (line 37):
        str_34809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 22), 'str', 'ok,dok')
        # Getting the type of 'cmd' (line 37)
        cmd_34810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'cmd')
        # Setting the type of the member 'option1' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), cmd_34810, 'option1', str_34809)
        
        # Call to ensure_string_list(...): (line 38)
        # Processing the call arguments (line 38)
        str_34813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 31), 'str', 'option1')
        # Processing the call keyword arguments (line 38)
        kwargs_34814 = {}
        # Getting the type of 'cmd' (line 38)
        cmd_34811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'cmd', False)
        # Obtaining the member 'ensure_string_list' of a type (line 38)
        ensure_string_list_34812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), cmd_34811, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 38)
        ensure_string_list_call_result_34815 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), ensure_string_list_34812, *[str_34813], **kwargs_34814)
        
        
        # Call to assertEqual(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'cmd' (line 39)
        cmd_34818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'cmd', False)
        # Obtaining the member 'option1' of a type (line 39)
        option1_34819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 25), cmd_34818, 'option1')
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_34820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        # Adding element type (line 39)
        str_34821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 39), 'str', 'ok')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 38), list_34820, str_34821)
        # Adding element type (line 39)
        str_34822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 45), 'str', 'dok')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 38), list_34820, str_34822)
        
        # Processing the call keyword arguments (line 39)
        kwargs_34823 = {}
        # Getting the type of 'self' (line 39)
        self_34816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 39)
        assertEqual_34817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_34816, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 39)
        assertEqual_call_result_34824 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), assertEqual_34817, *[option1_34819, list_34820], **kwargs_34823)
        
        
        # Assigning a List to a Attribute (line 41):
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_34825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        str_34826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'str', 'xxx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 22), list_34825, str_34826)
        # Adding element type (line 41)
        str_34827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'str', 'www')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 22), list_34825, str_34827)
        
        # Getting the type of 'cmd' (line 41)
        cmd_34828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'cmd')
        # Setting the type of the member 'option2' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), cmd_34828, 'option2', list_34825)
        
        # Call to ensure_string_list(...): (line 42)
        # Processing the call arguments (line 42)
        str_34831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 31), 'str', 'option2')
        # Processing the call keyword arguments (line 42)
        kwargs_34832 = {}
        # Getting the type of 'cmd' (line 42)
        cmd_34829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'cmd', False)
        # Obtaining the member 'ensure_string_list' of a type (line 42)
        ensure_string_list_34830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), cmd_34829, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 42)
        ensure_string_list_call_result_34833 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), ensure_string_list_34830, *[str_34831], **kwargs_34832)
        
        
        # Assigning a List to a Attribute (line 44):
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_34834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        # Adding element type (line 44)
        str_34835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'str', 'ok')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 22), list_34834, str_34835)
        # Adding element type (line 44)
        int_34836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 22), list_34834, int_34836)
        
        # Getting the type of 'cmd' (line 44)
        cmd_34837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'cmd')
        # Setting the type of the member 'option3' of a type (line 44)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), cmd_34837, 'option3', list_34834)
        
        # Call to assertRaises(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'DistutilsOptionError' (line 45)
        DistutilsOptionError_34840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'DistutilsOptionError', False)
        # Getting the type of 'cmd' (line 45)
        cmd_34841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 48), 'cmd', False)
        # Obtaining the member 'ensure_string_list' of a type (line 45)
        ensure_string_list_34842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 48), cmd_34841, 'ensure_string_list')
        str_34843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 26), 'str', 'option3')
        # Processing the call keyword arguments (line 45)
        kwargs_34844 = {}
        # Getting the type of 'self' (line 45)
        self_34838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 45)
        assertRaises_34839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_34838, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 45)
        assertRaises_call_result_34845 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), assertRaises_34839, *[DistutilsOptionError_34840, ensure_string_list_34842, str_34843], **kwargs_34844)
        
        
        # ################# End of 'test_ensure_string_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ensure_string_list' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_34846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34846)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ensure_string_list'
        return stypy_return_type_34846


    @norecursion
    def test_make_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_make_file'
        module_type_store = module_type_store.open_function_context('test_make_file', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CommandTestCase.test_make_file.__dict__.__setitem__('stypy_localization', localization)
        CommandTestCase.test_make_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CommandTestCase.test_make_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        CommandTestCase.test_make_file.__dict__.__setitem__('stypy_function_name', 'CommandTestCase.test_make_file')
        CommandTestCase.test_make_file.__dict__.__setitem__('stypy_param_names_list', [])
        CommandTestCase.test_make_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        CommandTestCase.test_make_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CommandTestCase.test_make_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        CommandTestCase.test_make_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        CommandTestCase.test_make_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CommandTestCase.test_make_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommandTestCase.test_make_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_make_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_make_file(...)' code ##################

        
        # Assigning a Attribute to a Name (line 51):
        # Getting the type of 'self' (line 51)
        self_34847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'self')
        # Obtaining the member 'cmd' of a type (line 51)
        cmd_34848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 14), self_34847, 'cmd')
        # Assigning a type to the variable 'cmd' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'cmd', cmd_34848)
        
        # Call to assertRaises(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'TypeError' (line 54)
        TypeError_34851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 26), 'TypeError', False)
        # Getting the type of 'cmd' (line 54)
        cmd_34852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'cmd', False)
        # Obtaining the member 'make_file' of a type (line 54)
        make_file_34853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 37), cmd_34852, 'make_file')
        # Processing the call keyword arguments (line 54)
        int_34854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'int')
        keyword_34855 = int_34854
        str_34856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 45), 'str', '')
        keyword_34857 = str_34856
        str_34858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 54), 'str', 'func')
        keyword_34859 = str_34858
        
        # Obtaining an instance of the builtin type 'tuple' (line 55)
        tuple_34860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 55)
        
        keyword_34861 = tuple_34860
        kwargs_34862 = {'outfile': keyword_34857, 'args': keyword_34861, 'infiles': keyword_34855, 'func': keyword_34859}
        # Getting the type of 'self' (line 54)
        self_34849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 54)
        assertRaises_34850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_34849, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 54)
        assertRaises_call_result_34863 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assertRaises_34850, *[TypeError_34851, make_file_34853], **kwargs_34862)
        

        @norecursion
        def _execute(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_execute'
            module_type_store = module_type_store.open_function_context('_execute', 58, 8, False)
            
            # Passed parameters checking function
            _execute.stypy_localization = localization
            _execute.stypy_type_of_self = None
            _execute.stypy_type_store = module_type_store
            _execute.stypy_function_name = '_execute'
            _execute.stypy_param_names_list = ['func', 'args', 'exec_msg', 'level']
            _execute.stypy_varargs_param_name = None
            _execute.stypy_kwargs_param_name = None
            _execute.stypy_call_defaults = defaults
            _execute.stypy_call_varargs = varargs
            _execute.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_execute', ['func', 'args', 'exec_msg', 'level'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_execute', localization, ['func', 'args', 'exec_msg', 'level'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_execute(...)' code ##################

            
            # Call to assertEqual(...): (line 59)
            # Processing the call arguments (line 59)
            # Getting the type of 'exec_msg' (line 59)
            exec_msg_34866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'exec_msg', False)
            str_34867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 39), 'str', 'generating out from in')
            # Processing the call keyword arguments (line 59)
            kwargs_34868 = {}
            # Getting the type of 'self' (line 59)
            self_34864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'self', False)
            # Obtaining the member 'assertEqual' of a type (line 59)
            assertEqual_34865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 12), self_34864, 'assertEqual')
            # Calling assertEqual(args, kwargs) (line 59)
            assertEqual_call_result_34869 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), assertEqual_34865, *[exec_msg_34866, str_34867], **kwargs_34868)
            
            
            # ################# End of '_execute(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_execute' in the type store
            # Getting the type of 'stypy_return_type' (line 58)
            stypy_return_type_34870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_34870)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_execute'
            return stypy_return_type_34870

        # Assigning a type to the variable '_execute' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), '_execute', _execute)
        
        # Assigning a Name to a Attribute (line 60):
        # Getting the type of 'True' (line 60)
        True_34871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'True')
        # Getting the type of 'cmd' (line 60)
        cmd_34872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'cmd')
        # Setting the type of the member 'force' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), cmd_34872, 'force', True_34871)
        
        # Assigning a Name to a Attribute (line 61):
        # Getting the type of '_execute' (line 61)
        _execute_34873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 22), '_execute')
        # Getting the type of 'cmd' (line 61)
        cmd_34874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'cmd')
        # Setting the type of the member 'execute' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), cmd_34874, 'execute', _execute_34873)
        
        # Call to make_file(...): (line 62)
        # Processing the call keyword arguments (line 62)
        str_34877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 30), 'str', 'in')
        keyword_34878 = str_34877
        str_34879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 44), 'str', 'out')
        keyword_34880 = str_34879
        str_34881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 56), 'str', 'func')
        keyword_34882 = str_34881
        
        # Obtaining an instance of the builtin type 'tuple' (line 62)
        tuple_34883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 69), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 62)
        
        keyword_34884 = tuple_34883
        kwargs_34885 = {'outfile': keyword_34880, 'args': keyword_34884, 'infiles': keyword_34878, 'func': keyword_34882}
        # Getting the type of 'cmd' (line 62)
        cmd_34875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'cmd', False)
        # Obtaining the member 'make_file' of a type (line 62)
        make_file_34876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), cmd_34875, 'make_file')
        # Calling make_file(args, kwargs) (line 62)
        make_file_call_result_34886 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), make_file_34876, *[], **kwargs_34885)
        
        
        # ################# End of 'test_make_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_make_file' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_34887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34887)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_make_file'
        return stypy_return_type_34887


    @norecursion
    def test_dump_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dump_options'
        module_type_store = module_type_store.open_function_context('test_dump_options', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CommandTestCase.test_dump_options.__dict__.__setitem__('stypy_localization', localization)
        CommandTestCase.test_dump_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CommandTestCase.test_dump_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        CommandTestCase.test_dump_options.__dict__.__setitem__('stypy_function_name', 'CommandTestCase.test_dump_options')
        CommandTestCase.test_dump_options.__dict__.__setitem__('stypy_param_names_list', [])
        CommandTestCase.test_dump_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        CommandTestCase.test_dump_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CommandTestCase.test_dump_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        CommandTestCase.test_dump_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        CommandTestCase.test_dump_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CommandTestCase.test_dump_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommandTestCase.test_dump_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dump_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dump_options(...)' code ##################

        
        # Assigning a List to a Name (line 66):
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_34888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        
        # Assigning a type to the variable 'msgs' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'msgs', list_34888)

        @norecursion
        def _announce(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_announce'
            module_type_store = module_type_store.open_function_context('_announce', 67, 8, False)
            
            # Passed parameters checking function
            _announce.stypy_localization = localization
            _announce.stypy_type_of_self = None
            _announce.stypy_type_store = module_type_store
            _announce.stypy_function_name = '_announce'
            _announce.stypy_param_names_list = ['msg', 'level']
            _announce.stypy_varargs_param_name = None
            _announce.stypy_kwargs_param_name = None
            _announce.stypy_call_defaults = defaults
            _announce.stypy_call_varargs = varargs
            _announce.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_announce', ['msg', 'level'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_announce', localization, ['msg', 'level'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_announce(...)' code ##################

            
            # Call to append(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of 'msg' (line 68)
            msg_34891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'msg', False)
            # Processing the call keyword arguments (line 68)
            kwargs_34892 = {}
            # Getting the type of 'msgs' (line 68)
            msgs_34889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'msgs', False)
            # Obtaining the member 'append' of a type (line 68)
            append_34890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), msgs_34889, 'append')
            # Calling append(args, kwargs) (line 68)
            append_call_result_34893 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), append_34890, *[msg_34891], **kwargs_34892)
            
            
            # ################# End of '_announce(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_announce' in the type store
            # Getting the type of 'stypy_return_type' (line 67)
            stypy_return_type_34894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_34894)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_announce'
            return stypy_return_type_34894

        # Assigning a type to the variable '_announce' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), '_announce', _announce)
        
        # Assigning a Attribute to a Name (line 69):
        # Getting the type of 'self' (line 69)
        self_34895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'self')
        # Obtaining the member 'cmd' of a type (line 69)
        cmd_34896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 14), self_34895, 'cmd')
        # Assigning a type to the variable 'cmd' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'cmd', cmd_34896)
        
        # Assigning a Name to a Attribute (line 70):
        # Getting the type of '_announce' (line 70)
        _announce_34897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), '_announce')
        # Getting the type of 'cmd' (line 70)
        cmd_34898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'cmd')
        # Setting the type of the member 'announce' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), cmd_34898, 'announce', _announce_34897)
        
        # Assigning a Num to a Attribute (line 71):
        int_34899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 22), 'int')
        # Getting the type of 'cmd' (line 71)
        cmd_34900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'cmd')
        # Setting the type of the member 'option1' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), cmd_34900, 'option1', int_34899)
        
        # Assigning a Num to a Attribute (line 72):
        int_34901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 22), 'int')
        # Getting the type of 'cmd' (line 72)
        cmd_34902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'cmd')
        # Setting the type of the member 'option2' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), cmd_34902, 'option2', int_34901)
        
        # Assigning a List to a Attribute (line 73):
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_34903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        
        # Obtaining an instance of the builtin type 'tuple' (line 73)
        tuple_34904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 73)
        # Adding element type (line 73)
        str_34905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 29), 'str', 'option1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 29), tuple_34904, str_34905)
        # Adding element type (line 73)
        str_34906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 40), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 29), tuple_34904, str_34906)
        # Adding element type (line 73)
        str_34907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 44), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 29), tuple_34904, str_34907)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 27), list_34903, tuple_34904)
        # Adding element type (line 73)
        
        # Obtaining an instance of the builtin type 'tuple' (line 73)
        tuple_34908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 73)
        # Adding element type (line 73)
        str_34909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 50), 'str', 'option2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 50), tuple_34908, str_34909)
        # Adding element type (line 73)
        str_34910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 61), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 50), tuple_34908, str_34910)
        # Adding element type (line 73)
        str_34911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 65), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 50), tuple_34908, str_34911)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 27), list_34903, tuple_34908)
        
        # Getting the type of 'cmd' (line 73)
        cmd_34912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'cmd')
        # Setting the type of the member 'user_options' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), cmd_34912, 'user_options', list_34903)
        
        # Call to dump_options(...): (line 74)
        # Processing the call keyword arguments (line 74)
        kwargs_34915 = {}
        # Getting the type of 'cmd' (line 74)
        cmd_34913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'cmd', False)
        # Obtaining the member 'dump_options' of a type (line 74)
        dump_options_34914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), cmd_34913, 'dump_options')
        # Calling dump_options(args, kwargs) (line 74)
        dump_options_call_result_34916 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), dump_options_34914, *[], **kwargs_34915)
        
        
        # Assigning a List to a Name (line 76):
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_34917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        str_34918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 18), 'str', "command options for 'MyCmd':")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 17), list_34917, str_34918)
        # Adding element type (line 76)
        str_34919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 50), 'str', '  option1 = 1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 17), list_34917, str_34919)
        # Adding element type (line 76)
        str_34920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 18), 'str', '  option2 = 1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 17), list_34917, str_34920)
        
        # Assigning a type to the variable 'wanted' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'wanted', list_34917)
        
        # Call to assertEqual(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'msgs' (line 78)
        msgs_34923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'msgs', False)
        # Getting the type of 'wanted' (line 78)
        wanted_34924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 31), 'wanted', False)
        # Processing the call keyword arguments (line 78)
        kwargs_34925 = {}
        # Getting the type of 'self' (line 78)
        self_34921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 78)
        assertEqual_34922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_34921, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 78)
        assertEqual_call_result_34926 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), assertEqual_34922, *[msgs_34923, wanted_34924], **kwargs_34925)
        
        
        # ################# End of 'test_dump_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dump_options' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_34927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34927)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dump_options'
        return stypy_return_type_34927


    @norecursion
    def test_ensure_string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ensure_string'
        module_type_store = module_type_store.open_function_context('test_ensure_string', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CommandTestCase.test_ensure_string.__dict__.__setitem__('stypy_localization', localization)
        CommandTestCase.test_ensure_string.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CommandTestCase.test_ensure_string.__dict__.__setitem__('stypy_type_store', module_type_store)
        CommandTestCase.test_ensure_string.__dict__.__setitem__('stypy_function_name', 'CommandTestCase.test_ensure_string')
        CommandTestCase.test_ensure_string.__dict__.__setitem__('stypy_param_names_list', [])
        CommandTestCase.test_ensure_string.__dict__.__setitem__('stypy_varargs_param_name', None)
        CommandTestCase.test_ensure_string.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CommandTestCase.test_ensure_string.__dict__.__setitem__('stypy_call_defaults', defaults)
        CommandTestCase.test_ensure_string.__dict__.__setitem__('stypy_call_varargs', varargs)
        CommandTestCase.test_ensure_string.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CommandTestCase.test_ensure_string.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommandTestCase.test_ensure_string', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ensure_string', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ensure_string(...)' code ##################

        
        # Assigning a Attribute to a Name (line 81):
        # Getting the type of 'self' (line 81)
        self_34928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 14), 'self')
        # Obtaining the member 'cmd' of a type (line 81)
        cmd_34929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 14), self_34928, 'cmd')
        # Assigning a type to the variable 'cmd' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'cmd', cmd_34929)
        
        # Assigning a Str to a Attribute (line 82):
        str_34930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'str', 'ok')
        # Getting the type of 'cmd' (line 82)
        cmd_34931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'cmd')
        # Setting the type of the member 'option1' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), cmd_34931, 'option1', str_34930)
        
        # Call to ensure_string(...): (line 83)
        # Processing the call arguments (line 83)
        str_34934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 26), 'str', 'option1')
        # Processing the call keyword arguments (line 83)
        kwargs_34935 = {}
        # Getting the type of 'cmd' (line 83)
        cmd_34932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'cmd', False)
        # Obtaining the member 'ensure_string' of a type (line 83)
        ensure_string_34933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), cmd_34932, 'ensure_string')
        # Calling ensure_string(args, kwargs) (line 83)
        ensure_string_call_result_34936 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), ensure_string_34933, *[str_34934], **kwargs_34935)
        
        
        # Assigning a Name to a Attribute (line 85):
        # Getting the type of 'None' (line 85)
        None_34937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 22), 'None')
        # Getting the type of 'cmd' (line 85)
        cmd_34938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'cmd')
        # Setting the type of the member 'option2' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), cmd_34938, 'option2', None_34937)
        
        # Call to ensure_string(...): (line 86)
        # Processing the call arguments (line 86)
        str_34941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 26), 'str', 'option2')
        str_34942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 37), 'str', 'xxx')
        # Processing the call keyword arguments (line 86)
        kwargs_34943 = {}
        # Getting the type of 'cmd' (line 86)
        cmd_34939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'cmd', False)
        # Obtaining the member 'ensure_string' of a type (line 86)
        ensure_string_34940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), cmd_34939, 'ensure_string')
        # Calling ensure_string(args, kwargs) (line 86)
        ensure_string_call_result_34944 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), ensure_string_34940, *[str_34941, str_34942], **kwargs_34943)
        
        
        # Call to assertTrue(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Call to hasattr(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'cmd' (line 87)
        cmd_34948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 32), 'cmd', False)
        str_34949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 37), 'str', 'option2')
        # Processing the call keyword arguments (line 87)
        kwargs_34950 = {}
        # Getting the type of 'hasattr' (line 87)
        hasattr_34947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 87)
        hasattr_call_result_34951 = invoke(stypy.reporting.localization.Localization(__file__, 87, 24), hasattr_34947, *[cmd_34948, str_34949], **kwargs_34950)
        
        # Processing the call keyword arguments (line 87)
        kwargs_34952 = {}
        # Getting the type of 'self' (line 87)
        self_34945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 87)
        assertTrue_34946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_34945, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 87)
        assertTrue_call_result_34953 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), assertTrue_34946, *[hasattr_call_result_34951], **kwargs_34952)
        
        
        # Assigning a Num to a Attribute (line 89):
        int_34954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'int')
        # Getting the type of 'cmd' (line 89)
        cmd_34955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'cmd')
        # Setting the type of the member 'option3' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), cmd_34955, 'option3', int_34954)
        
        # Call to assertRaises(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'DistutilsOptionError' (line 90)
        DistutilsOptionError_34958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 26), 'DistutilsOptionError', False)
        # Getting the type of 'cmd' (line 90)
        cmd_34959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 48), 'cmd', False)
        # Obtaining the member 'ensure_string' of a type (line 90)
        ensure_string_34960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 48), cmd_34959, 'ensure_string')
        str_34961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 67), 'str', 'option3')
        # Processing the call keyword arguments (line 90)
        kwargs_34962 = {}
        # Getting the type of 'self' (line 90)
        self_34956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 90)
        assertRaises_34957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_34956, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 90)
        assertRaises_call_result_34963 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), assertRaises_34957, *[DistutilsOptionError_34958, ensure_string_34960, str_34961], **kwargs_34962)
        
        
        # ################# End of 'test_ensure_string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ensure_string' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_34964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34964)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ensure_string'
        return stypy_return_type_34964


    @norecursion
    def test_ensure_filename(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ensure_filename'
        module_type_store = module_type_store.open_function_context('test_ensure_filename', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CommandTestCase.test_ensure_filename.__dict__.__setitem__('stypy_localization', localization)
        CommandTestCase.test_ensure_filename.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CommandTestCase.test_ensure_filename.__dict__.__setitem__('stypy_type_store', module_type_store)
        CommandTestCase.test_ensure_filename.__dict__.__setitem__('stypy_function_name', 'CommandTestCase.test_ensure_filename')
        CommandTestCase.test_ensure_filename.__dict__.__setitem__('stypy_param_names_list', [])
        CommandTestCase.test_ensure_filename.__dict__.__setitem__('stypy_varargs_param_name', None)
        CommandTestCase.test_ensure_filename.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CommandTestCase.test_ensure_filename.__dict__.__setitem__('stypy_call_defaults', defaults)
        CommandTestCase.test_ensure_filename.__dict__.__setitem__('stypy_call_varargs', varargs)
        CommandTestCase.test_ensure_filename.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CommandTestCase.test_ensure_filename.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommandTestCase.test_ensure_filename', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ensure_filename', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ensure_filename(...)' code ##################

        
        # Assigning a Attribute to a Name (line 93):
        # Getting the type of 'self' (line 93)
        self_34965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 14), 'self')
        # Obtaining the member 'cmd' of a type (line 93)
        cmd_34966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 14), self_34965, 'cmd')
        # Assigning a type to the variable 'cmd' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'cmd', cmd_34966)
        
        # Assigning a Name to a Attribute (line 94):
        # Getting the type of '__file__' (line 94)
        file___34967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 22), '__file__')
        # Getting the type of 'cmd' (line 94)
        cmd_34968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'cmd')
        # Setting the type of the member 'option1' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), cmd_34968, 'option1', file___34967)
        
        # Call to ensure_filename(...): (line 95)
        # Processing the call arguments (line 95)
        str_34971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 28), 'str', 'option1')
        # Processing the call keyword arguments (line 95)
        kwargs_34972 = {}
        # Getting the type of 'cmd' (line 95)
        cmd_34969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'cmd', False)
        # Obtaining the member 'ensure_filename' of a type (line 95)
        ensure_filename_34970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), cmd_34969, 'ensure_filename')
        # Calling ensure_filename(args, kwargs) (line 95)
        ensure_filename_call_result_34973 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), ensure_filename_34970, *[str_34971], **kwargs_34972)
        
        
        # Assigning a Str to a Attribute (line 96):
        str_34974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 22), 'str', 'xxx')
        # Getting the type of 'cmd' (line 96)
        cmd_34975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'cmd')
        # Setting the type of the member 'option2' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), cmd_34975, 'option2', str_34974)
        
        # Call to assertRaises(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'DistutilsOptionError' (line 97)
        DistutilsOptionError_34978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'DistutilsOptionError', False)
        # Getting the type of 'cmd' (line 97)
        cmd_34979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 48), 'cmd', False)
        # Obtaining the member 'ensure_filename' of a type (line 97)
        ensure_filename_34980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 48), cmd_34979, 'ensure_filename')
        str_34981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 69), 'str', 'option2')
        # Processing the call keyword arguments (line 97)
        kwargs_34982 = {}
        # Getting the type of 'self' (line 97)
        self_34976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 97)
        assertRaises_34977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_34976, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 97)
        assertRaises_call_result_34983 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assertRaises_34977, *[DistutilsOptionError_34978, ensure_filename_34980, str_34981], **kwargs_34982)
        
        
        # ################# End of 'test_ensure_filename(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ensure_filename' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_34984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34984)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ensure_filename'
        return stypy_return_type_34984


    @norecursion
    def test_ensure_dirname(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ensure_dirname'
        module_type_store = module_type_store.open_function_context('test_ensure_dirname', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CommandTestCase.test_ensure_dirname.__dict__.__setitem__('stypy_localization', localization)
        CommandTestCase.test_ensure_dirname.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CommandTestCase.test_ensure_dirname.__dict__.__setitem__('stypy_type_store', module_type_store)
        CommandTestCase.test_ensure_dirname.__dict__.__setitem__('stypy_function_name', 'CommandTestCase.test_ensure_dirname')
        CommandTestCase.test_ensure_dirname.__dict__.__setitem__('stypy_param_names_list', [])
        CommandTestCase.test_ensure_dirname.__dict__.__setitem__('stypy_varargs_param_name', None)
        CommandTestCase.test_ensure_dirname.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CommandTestCase.test_ensure_dirname.__dict__.__setitem__('stypy_call_defaults', defaults)
        CommandTestCase.test_ensure_dirname.__dict__.__setitem__('stypy_call_varargs', varargs)
        CommandTestCase.test_ensure_dirname.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CommandTestCase.test_ensure_dirname.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommandTestCase.test_ensure_dirname', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ensure_dirname', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ensure_dirname(...)' code ##################

        
        # Assigning a Attribute to a Name (line 100):
        # Getting the type of 'self' (line 100)
        self_34985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 'self')
        # Obtaining the member 'cmd' of a type (line 100)
        cmd_34986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 14), self_34985, 'cmd')
        # Assigning a type to the variable 'cmd' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'cmd', cmd_34986)
        
        # Assigning a BoolOp to a Attribute (line 101):
        
        # Evaluating a boolean operation
        
        # Call to dirname(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of '__file__' (line 101)
        file___34990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 38), '__file__', False)
        # Processing the call keyword arguments (line 101)
        kwargs_34991 = {}
        # Getting the type of 'os' (line 101)
        os_34987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 101)
        path_34988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 22), os_34987, 'path')
        # Obtaining the member 'dirname' of a type (line 101)
        dirname_34989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 22), path_34988, 'dirname')
        # Calling dirname(args, kwargs) (line 101)
        dirname_call_result_34992 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), dirname_34989, *[file___34990], **kwargs_34991)
        
        # Getting the type of 'os' (line 101)
        os_34993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'os')
        # Obtaining the member 'curdir' of a type (line 101)
        curdir_34994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 51), os_34993, 'curdir')
        # Applying the binary operator 'or' (line 101)
        result_or_keyword_34995 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 22), 'or', dirname_call_result_34992, curdir_34994)
        
        # Getting the type of 'cmd' (line 101)
        cmd_34996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'cmd')
        # Setting the type of the member 'option1' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), cmd_34996, 'option1', result_or_keyword_34995)
        
        # Call to ensure_dirname(...): (line 102)
        # Processing the call arguments (line 102)
        str_34999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 27), 'str', 'option1')
        # Processing the call keyword arguments (line 102)
        kwargs_35000 = {}
        # Getting the type of 'cmd' (line 102)
        cmd_34997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'cmd', False)
        # Obtaining the member 'ensure_dirname' of a type (line 102)
        ensure_dirname_34998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), cmd_34997, 'ensure_dirname')
        # Calling ensure_dirname(args, kwargs) (line 102)
        ensure_dirname_call_result_35001 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), ensure_dirname_34998, *[str_34999], **kwargs_35000)
        
        
        # Assigning a Str to a Attribute (line 103):
        str_35002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 22), 'str', 'xxx')
        # Getting the type of 'cmd' (line 103)
        cmd_35003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'cmd')
        # Setting the type of the member 'option2' of a type (line 103)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), cmd_35003, 'option2', str_35002)
        
        # Call to assertRaises(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'DistutilsOptionError' (line 104)
        DistutilsOptionError_35006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 26), 'DistutilsOptionError', False)
        # Getting the type of 'cmd' (line 104)
        cmd_35007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 48), 'cmd', False)
        # Obtaining the member 'ensure_dirname' of a type (line 104)
        ensure_dirname_35008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 48), cmd_35007, 'ensure_dirname')
        str_35009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 68), 'str', 'option2')
        # Processing the call keyword arguments (line 104)
        kwargs_35010 = {}
        # Getting the type of 'self' (line 104)
        self_35004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 104)
        assertRaises_35005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_35004, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 104)
        assertRaises_call_result_35011 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), assertRaises_35005, *[DistutilsOptionError_35006, ensure_dirname_35008, str_35009], **kwargs_35010)
        
        
        # ################# End of 'test_ensure_dirname(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ensure_dirname' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_35012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35012)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ensure_dirname'
        return stypy_return_type_35012


    @norecursion
    def test_debug_print(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_debug_print'
        module_type_store = module_type_store.open_function_context('test_debug_print', 106, 4, False)
        # Assigning a type to the variable 'self' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CommandTestCase.test_debug_print.__dict__.__setitem__('stypy_localization', localization)
        CommandTestCase.test_debug_print.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CommandTestCase.test_debug_print.__dict__.__setitem__('stypy_type_store', module_type_store)
        CommandTestCase.test_debug_print.__dict__.__setitem__('stypy_function_name', 'CommandTestCase.test_debug_print')
        CommandTestCase.test_debug_print.__dict__.__setitem__('stypy_param_names_list', [])
        CommandTestCase.test_debug_print.__dict__.__setitem__('stypy_varargs_param_name', None)
        CommandTestCase.test_debug_print.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CommandTestCase.test_debug_print.__dict__.__setitem__('stypy_call_defaults', defaults)
        CommandTestCase.test_debug_print.__dict__.__setitem__('stypy_call_varargs', varargs)
        CommandTestCase.test_debug_print.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CommandTestCase.test_debug_print.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommandTestCase.test_debug_print', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Name (line 107):
        # Getting the type of 'self' (line 107)
        self_35013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 14), 'self')
        # Obtaining the member 'cmd' of a type (line 107)
        cmd_35014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 14), self_35013, 'cmd')
        # Assigning a type to the variable 'cmd' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'cmd', cmd_35014)
        
        # Call to captured_stdout(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_35016 = {}
        # Getting the type of 'captured_stdout' (line 108)
        captured_stdout_35015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'captured_stdout', False)
        # Calling captured_stdout(args, kwargs) (line 108)
        captured_stdout_call_result_35017 = invoke(stypy.reporting.localization.Localization(__file__, 108, 13), captured_stdout_35015, *[], **kwargs_35016)
        
        with_35018 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 108, 13), captured_stdout_call_result_35017, 'with parameter', '__enter__', '__exit__')

        if with_35018:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 108)
            enter___35019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 13), captured_stdout_call_result_35017, '__enter__')
            with_enter_35020 = invoke(stypy.reporting.localization.Localization(__file__, 108, 13), enter___35019)
            # Assigning a type to the variable 'stdout' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'stdout', with_enter_35020)
            
            # Call to debug_print(...): (line 109)
            # Processing the call arguments (line 109)
            str_35023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 28), 'str', 'xxx')
            # Processing the call keyword arguments (line 109)
            kwargs_35024 = {}
            # Getting the type of 'cmd' (line 109)
            cmd_35021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'cmd', False)
            # Obtaining the member 'debug_print' of a type (line 109)
            debug_print_35022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), cmd_35021, 'debug_print')
            # Calling debug_print(args, kwargs) (line 109)
            debug_print_call_result_35025 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), debug_print_35022, *[str_35023], **kwargs_35024)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 108)
            exit___35026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 13), captured_stdout_call_result_35017, '__exit__')
            with_exit_35027 = invoke(stypy.reporting.localization.Localization(__file__, 108, 13), exit___35026, None, None, None)

        
        # Call to seek(...): (line 110)
        # Processing the call arguments (line 110)
        int_35030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 20), 'int')
        # Processing the call keyword arguments (line 110)
        kwargs_35031 = {}
        # Getting the type of 'stdout' (line 110)
        stdout_35028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'stdout', False)
        # Obtaining the member 'seek' of a type (line 110)
        seek_35029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), stdout_35028, 'seek')
        # Calling seek(args, kwargs) (line 110)
        seek_call_result_35032 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), seek_35029, *[int_35030], **kwargs_35031)
        
        
        # Call to assertEqual(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Call to read(...): (line 111)
        # Processing the call keyword arguments (line 111)
        kwargs_35037 = {}
        # Getting the type of 'stdout' (line 111)
        stdout_35035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'stdout', False)
        # Obtaining the member 'read' of a type (line 111)
        read_35036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 25), stdout_35035, 'read')
        # Calling read(args, kwargs) (line 111)
        read_call_result_35038 = invoke(stypy.reporting.localization.Localization(__file__, 111, 25), read_35036, *[], **kwargs_35037)
        
        str_35039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 40), 'str', '')
        # Processing the call keyword arguments (line 111)
        kwargs_35040 = {}
        # Getting the type of 'self' (line 111)
        self_35033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 111)
        assertEqual_35034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_35033, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 111)
        assertEqual_call_result_35041 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), assertEqual_35034, *[read_call_result_35038, str_35039], **kwargs_35040)
        
        
        # Assigning a Name to a Attribute (line 113):
        # Getting the type of 'True' (line 113)
        True_35042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 22), 'True')
        # Getting the type of 'debug' (line 113)
        debug_35043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'debug')
        # Setting the type of the member 'DEBUG' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), debug_35043, 'DEBUG', True_35042)
        
        # Try-finally block (line 114)
        
        # Call to captured_stdout(...): (line 115)
        # Processing the call keyword arguments (line 115)
        kwargs_35045 = {}
        # Getting the type of 'captured_stdout' (line 115)
        captured_stdout_35044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'captured_stdout', False)
        # Calling captured_stdout(args, kwargs) (line 115)
        captured_stdout_call_result_35046 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), captured_stdout_35044, *[], **kwargs_35045)
        
        with_35047 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 115, 17), captured_stdout_call_result_35046, 'with parameter', '__enter__', '__exit__')

        if with_35047:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 115)
            enter___35048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 17), captured_stdout_call_result_35046, '__enter__')
            with_enter_35049 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), enter___35048)
            # Assigning a type to the variable 'stdout' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'stdout', with_enter_35049)
            
            # Call to debug_print(...): (line 116)
            # Processing the call arguments (line 116)
            str_35052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 32), 'str', 'xxx')
            # Processing the call keyword arguments (line 116)
            kwargs_35053 = {}
            # Getting the type of 'cmd' (line 116)
            cmd_35050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'cmd', False)
            # Obtaining the member 'debug_print' of a type (line 116)
            debug_print_35051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 16), cmd_35050, 'debug_print')
            # Calling debug_print(args, kwargs) (line 116)
            debug_print_call_result_35054 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), debug_print_35051, *[str_35052], **kwargs_35053)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 115)
            exit___35055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 17), captured_stdout_call_result_35046, '__exit__')
            with_exit_35056 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), exit___35055, None, None, None)

        
        # Call to seek(...): (line 117)
        # Processing the call arguments (line 117)
        int_35059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 24), 'int')
        # Processing the call keyword arguments (line 117)
        kwargs_35060 = {}
        # Getting the type of 'stdout' (line 117)
        stdout_35057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'stdout', False)
        # Obtaining the member 'seek' of a type (line 117)
        seek_35058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), stdout_35057, 'seek')
        # Calling seek(args, kwargs) (line 117)
        seek_call_result_35061 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), seek_35058, *[int_35059], **kwargs_35060)
        
        
        # Call to assertEqual(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Call to read(...): (line 118)
        # Processing the call keyword arguments (line 118)
        kwargs_35066 = {}
        # Getting the type of 'stdout' (line 118)
        stdout_35064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 29), 'stdout', False)
        # Obtaining the member 'read' of a type (line 118)
        read_35065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 29), stdout_35064, 'read')
        # Calling read(args, kwargs) (line 118)
        read_call_result_35067 = invoke(stypy.reporting.localization.Localization(__file__, 118, 29), read_35065, *[], **kwargs_35066)
        
        str_35068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 44), 'str', 'xxx\n')
        # Processing the call keyword arguments (line 118)
        kwargs_35069 = {}
        # Getting the type of 'self' (line 118)
        self_35062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 118)
        assertEqual_35063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), self_35062, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 118)
        assertEqual_call_result_35070 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), assertEqual_35063, *[read_call_result_35067, str_35068], **kwargs_35069)
        
        
        # finally branch of the try-finally block (line 114)
        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'False' (line 120)
        False_35071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 26), 'False')
        # Getting the type of 'debug' (line 120)
        debug_35072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'debug')
        # Setting the type of the member 'DEBUG' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), debug_35072, 'DEBUG', False_35071)
        
        
        # ################# End of 'test_debug_print(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_debug_print' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_35073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35073)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_debug_print'
        return stypy_return_type_35073


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 0, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommandTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CommandTestCase' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'CommandTestCase', CommandTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 122, 0, False)
    
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

    
    # Call to makeSuite(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'CommandTestCase' (line 123)
    CommandTestCase_35076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 30), 'CommandTestCase', False)
    # Processing the call keyword arguments (line 123)
    kwargs_35077 = {}
    # Getting the type of 'unittest' (line 123)
    unittest_35074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 123)
    makeSuite_35075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 11), unittest_35074, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 123)
    makeSuite_call_result_35078 = invoke(stypy.reporting.localization.Localization(__file__, 123, 11), makeSuite_35075, *[CommandTestCase_35076], **kwargs_35077)
    
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type', makeSuite_call_result_35078)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 122)
    stypy_return_type_35079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35079)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_35079

# Assigning a type to the variable 'test_suite' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 126)
    # Processing the call arguments (line 126)
    
    # Call to test_suite(...): (line 126)
    # Processing the call keyword arguments (line 126)
    kwargs_35082 = {}
    # Getting the type of 'test_suite' (line 126)
    test_suite_35081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 126)
    test_suite_call_result_35083 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), test_suite_35081, *[], **kwargs_35082)
    
    # Processing the call keyword arguments (line 126)
    kwargs_35084 = {}
    # Getting the type of 'run_unittest' (line 126)
    run_unittest_35080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 126)
    run_unittest_call_result_35085 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), run_unittest_35080, *[test_suite_call_result_35083], **kwargs_35084)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
