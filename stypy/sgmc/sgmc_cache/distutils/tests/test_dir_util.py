
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.dir_util.'''
2: import unittest
3: import os
4: import stat
5: import shutil
6: import sys
7: 
8: from distutils.dir_util import (mkpath, remove_tree, create_tree, copy_tree,
9:                                 ensure_relative)
10: 
11: from distutils import log
12: from distutils.tests import support
13: from test.test_support import run_unittest
14: 
15: class DirUtilTestCase(support.TempdirManager, unittest.TestCase):
16: 
17:     def _log(self, msg, *args):
18:         if len(args) > 0:
19:             self._logs.append(msg % args)
20:         else:
21:             self._logs.append(msg)
22: 
23:     def setUp(self):
24:         super(DirUtilTestCase, self).setUp()
25:         self._logs = []
26:         tmp_dir = self.mkdtemp()
27:         self.root_target = os.path.join(tmp_dir, 'deep')
28:         self.target = os.path.join(self.root_target, 'here')
29:         self.target2 = os.path.join(tmp_dir, 'deep2')
30:         self.old_log = log.info
31:         log.info = self._log
32: 
33:     def tearDown(self):
34:         log.info = self.old_log
35:         super(DirUtilTestCase, self).tearDown()
36: 
37:     def test_mkpath_remove_tree_verbosity(self):
38: 
39:         mkpath(self.target, verbose=0)
40:         wanted = []
41:         self.assertEqual(self._logs, wanted)
42:         remove_tree(self.root_target, verbose=0)
43: 
44:         mkpath(self.target, verbose=1)
45:         wanted = ['creating %s' % self.root_target,
46:                   'creating %s' % self.target]
47:         self.assertEqual(self._logs, wanted)
48:         self._logs = []
49: 
50:         remove_tree(self.root_target, verbose=1)
51:         wanted = ["removing '%s' (and everything under it)" % self.root_target]
52:         self.assertEqual(self._logs, wanted)
53: 
54:     @unittest.skipIf(sys.platform.startswith('win'),
55:                         "This test is only appropriate for POSIX-like systems.")
56:     def test_mkpath_with_custom_mode(self):
57:         # Get and set the current umask value for testing mode bits.
58:         umask = os.umask(0o002)
59:         os.umask(umask)
60:         mkpath(self.target, 0o700)
61:         self.assertEqual(
62:             stat.S_IMODE(os.stat(self.target).st_mode), 0o700 & ~umask)
63:         mkpath(self.target2, 0o555)
64:         self.assertEqual(
65:             stat.S_IMODE(os.stat(self.target2).st_mode), 0o555 & ~umask)
66: 
67:     def test_create_tree_verbosity(self):
68: 
69:         create_tree(self.root_target, ['one', 'two', 'three'], verbose=0)
70:         self.assertEqual(self._logs, [])
71:         remove_tree(self.root_target, verbose=0)
72: 
73:         wanted = ['creating %s' % self.root_target]
74:         create_tree(self.root_target, ['one', 'two', 'three'], verbose=1)
75:         self.assertEqual(self._logs, wanted)
76: 
77:         remove_tree(self.root_target, verbose=0)
78: 
79: 
80:     def test_copy_tree_verbosity(self):
81: 
82:         mkpath(self.target, verbose=0)
83: 
84:         copy_tree(self.target, self.target2, verbose=0)
85:         self.assertEqual(self._logs, [])
86: 
87:         remove_tree(self.root_target, verbose=0)
88: 
89:         mkpath(self.target, verbose=0)
90:         a_file = os.path.join(self.target, 'ok.txt')
91:         f = open(a_file, 'w')
92:         try:
93:             f.write('some content')
94:         finally:
95:             f.close()
96: 
97:         wanted = ['copying %s -> %s' % (a_file, self.target2)]
98:         copy_tree(self.target, self.target2, verbose=1)
99:         self.assertEqual(self._logs, wanted)
100: 
101:         remove_tree(self.root_target, verbose=0)
102:         remove_tree(self.target2, verbose=0)
103: 
104:     def test_copy_tree_skips_nfs_temp_files(self):
105:         mkpath(self.target, verbose=0)
106: 
107:         a_file = os.path.join(self.target, 'ok.txt')
108:         nfs_file = os.path.join(self.target, '.nfs123abc')
109:         for f in a_file, nfs_file:
110:             fh = open(f, 'w')
111:             try:
112:                 fh.write('some content')
113:             finally:
114:                 fh.close()
115: 
116:         copy_tree(self.target, self.target2)
117:         self.assertEqual(os.listdir(self.target2), ['ok.txt'])
118: 
119:         remove_tree(self.root_target, verbose=0)
120:         remove_tree(self.target2, verbose=0)
121: 
122:     def test_ensure_relative(self):
123:         if os.sep == '/':
124:             self.assertEqual(ensure_relative('/home/foo'), 'home/foo')
125:             self.assertEqual(ensure_relative('some/path'), 'some/path')
126:         else:   # \\
127:             self.assertEqual(ensure_relative('c:\\home\\foo'), 'c:home\\foo')
128:             self.assertEqual(ensure_relative('home\\foo'), 'home\\foo')
129: 
130: def test_suite():
131:     return unittest.makeSuite(DirUtilTestCase)
132: 
133: if __name__ == "__main__":
134:     run_unittest(test_suite())
135: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_36224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.dir_util.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import unittest' statement (line 2)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import stat' statement (line 4)
import stat

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stat', stat, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import shutil' statement (line 5)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import sys' statement (line 6)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.dir_util import mkpath, remove_tree, create_tree, copy_tree, ensure_relative' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_36225 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.dir_util')

if (type(import_36225) is not StypyTypeError):

    if (import_36225 != 'pyd_module'):
        __import__(import_36225)
        sys_modules_36226 = sys.modules[import_36225]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.dir_util', sys_modules_36226.module_type_store, module_type_store, ['mkpath', 'remove_tree', 'create_tree', 'copy_tree', 'ensure_relative'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_36226, sys_modules_36226.module_type_store, module_type_store)
    else:
        from distutils.dir_util import mkpath, remove_tree, create_tree, copy_tree, ensure_relative

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.dir_util', None, module_type_store, ['mkpath', 'remove_tree', 'create_tree', 'copy_tree', 'ensure_relative'], [mkpath, remove_tree, create_tree, copy_tree, ensure_relative])

else:
    # Assigning a type to the variable 'distutils.dir_util' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.dir_util', import_36225)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils import log' statement (line 11)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.tests import support' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_36227 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.tests')

if (type(import_36227) is not StypyTypeError):

    if (import_36227 != 'pyd_module'):
        __import__(import_36227)
        sys_modules_36228 = sys.modules[import_36227]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.tests', sys_modules_36228.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_36228, sys_modules_36228.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.tests', import_36227)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from test.test_support import run_unittest' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_36229 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'test.test_support')

if (type(import_36229) is not StypyTypeError):

    if (import_36229 != 'pyd_module'):
        __import__(import_36229)
        sys_modules_36230 = sys.modules[import_36229]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'test.test_support', sys_modules_36230.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_36230, sys_modules_36230.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'test.test_support', import_36229)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'DirUtilTestCase' class
# Getting the type of 'support' (line 15)
support_36231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'support')
# Obtaining the member 'TempdirManager' of a type (line 15)
TempdirManager_36232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 22), support_36231, 'TempdirManager')
# Getting the type of 'unittest' (line 15)
unittest_36233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 46), 'unittest')
# Obtaining the member 'TestCase' of a type (line 15)
TestCase_36234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 46), unittest_36233, 'TestCase')

class DirUtilTestCase(TempdirManager_36232, TestCase_36234, ):

    @norecursion
    def _log(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_log'
        module_type_store = module_type_store.open_function_context('_log', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DirUtilTestCase._log.__dict__.__setitem__('stypy_localization', localization)
        DirUtilTestCase._log.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DirUtilTestCase._log.__dict__.__setitem__('stypy_type_store', module_type_store)
        DirUtilTestCase._log.__dict__.__setitem__('stypy_function_name', 'DirUtilTestCase._log')
        DirUtilTestCase._log.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        DirUtilTestCase._log.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        DirUtilTestCase._log.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DirUtilTestCase._log.__dict__.__setitem__('stypy_call_defaults', defaults)
        DirUtilTestCase._log.__dict__.__setitem__('stypy_call_varargs', varargs)
        DirUtilTestCase._log.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DirUtilTestCase._log.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DirUtilTestCase._log', ['msg'], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_log', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_log(...)' code ##################

        
        
        
        # Call to len(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'args' (line 18)
        args_36236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'args', False)
        # Processing the call keyword arguments (line 18)
        kwargs_36237 = {}
        # Getting the type of 'len' (line 18)
        len_36235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'len', False)
        # Calling len(args, kwargs) (line 18)
        len_call_result_36238 = invoke(stypy.reporting.localization.Localization(__file__, 18, 11), len_36235, *[args_36236], **kwargs_36237)
        
        int_36239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 23), 'int')
        # Applying the binary operator '>' (line 18)
        result_gt_36240 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 11), '>', len_call_result_36238, int_36239)
        
        # Testing the type of an if condition (line 18)
        if_condition_36241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 8), result_gt_36240)
        # Assigning a type to the variable 'if_condition_36241' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'if_condition_36241', if_condition_36241)
        # SSA begins for if statement (line 18)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'msg' (line 19)
        msg_36245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 30), 'msg', False)
        # Getting the type of 'args' (line 19)
        args_36246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 36), 'args', False)
        # Applying the binary operator '%' (line 19)
        result_mod_36247 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 30), '%', msg_36245, args_36246)
        
        # Processing the call keyword arguments (line 19)
        kwargs_36248 = {}
        # Getting the type of 'self' (line 19)
        self_36242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'self', False)
        # Obtaining the member '_logs' of a type (line 19)
        _logs_36243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), self_36242, '_logs')
        # Obtaining the member 'append' of a type (line 19)
        append_36244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), _logs_36243, 'append')
        # Calling append(args, kwargs) (line 19)
        append_call_result_36249 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), append_36244, *[result_mod_36247], **kwargs_36248)
        
        # SSA branch for the else part of an if statement (line 18)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'msg' (line 21)
        msg_36253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 30), 'msg', False)
        # Processing the call keyword arguments (line 21)
        kwargs_36254 = {}
        # Getting the type of 'self' (line 21)
        self_36250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'self', False)
        # Obtaining the member '_logs' of a type (line 21)
        _logs_36251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 12), self_36250, '_logs')
        # Obtaining the member 'append' of a type (line 21)
        append_36252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 12), _logs_36251, 'append')
        # Calling append(args, kwargs) (line 21)
        append_call_result_36255 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), append_36252, *[msg_36253], **kwargs_36254)
        
        # SSA join for if statement (line 18)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_log(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_log' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_36256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36256)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_log'
        return stypy_return_type_36256


    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DirUtilTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        DirUtilTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DirUtilTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        DirUtilTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'DirUtilTestCase.setUp')
        DirUtilTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        DirUtilTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        DirUtilTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DirUtilTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        DirUtilTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        DirUtilTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DirUtilTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DirUtilTestCase.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to setUp(...): (line 24)
        # Processing the call keyword arguments (line 24)
        kwargs_36263 = {}
        
        # Call to super(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'DirUtilTestCase' (line 24)
        DirUtilTestCase_36258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 14), 'DirUtilTestCase', False)
        # Getting the type of 'self' (line 24)
        self_36259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 31), 'self', False)
        # Processing the call keyword arguments (line 24)
        kwargs_36260 = {}
        # Getting the type of 'super' (line 24)
        super_36257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'super', False)
        # Calling super(args, kwargs) (line 24)
        super_call_result_36261 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), super_36257, *[DirUtilTestCase_36258, self_36259], **kwargs_36260)
        
        # Obtaining the member 'setUp' of a type (line 24)
        setUp_36262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), super_call_result_36261, 'setUp')
        # Calling setUp(args, kwargs) (line 24)
        setUp_call_result_36264 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), setUp_36262, *[], **kwargs_36263)
        
        
        # Assigning a List to a Attribute (line 25):
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_36265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        
        # Getting the type of 'self' (line 25)
        self_36266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member '_logs' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_36266, '_logs', list_36265)
        
        # Assigning a Call to a Name (line 26):
        
        # Call to mkdtemp(...): (line 26)
        # Processing the call keyword arguments (line 26)
        kwargs_36269 = {}
        # Getting the type of 'self' (line 26)
        self_36267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 26)
        mkdtemp_36268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 18), self_36267, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 26)
        mkdtemp_call_result_36270 = invoke(stypy.reporting.localization.Localization(__file__, 26, 18), mkdtemp_36268, *[], **kwargs_36269)
        
        # Assigning a type to the variable 'tmp_dir' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'tmp_dir', mkdtemp_call_result_36270)
        
        # Assigning a Call to a Attribute (line 27):
        
        # Call to join(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'tmp_dir' (line 27)
        tmp_dir_36274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 40), 'tmp_dir', False)
        str_36275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 49), 'str', 'deep')
        # Processing the call keyword arguments (line 27)
        kwargs_36276 = {}
        # Getting the type of 'os' (line 27)
        os_36271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 27)
        path_36272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 27), os_36271, 'path')
        # Obtaining the member 'join' of a type (line 27)
        join_36273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 27), path_36272, 'join')
        # Calling join(args, kwargs) (line 27)
        join_call_result_36277 = invoke(stypy.reporting.localization.Localization(__file__, 27, 27), join_36273, *[tmp_dir_36274, str_36275], **kwargs_36276)
        
        # Getting the type of 'self' (line 27)
        self_36278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member 'root_target' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_36278, 'root_target', join_call_result_36277)
        
        # Assigning a Call to a Attribute (line 28):
        
        # Call to join(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'self' (line 28)
        self_36282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 35), 'self', False)
        # Obtaining the member 'root_target' of a type (line 28)
        root_target_36283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 35), self_36282, 'root_target')
        str_36284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 53), 'str', 'here')
        # Processing the call keyword arguments (line 28)
        kwargs_36285 = {}
        # Getting the type of 'os' (line 28)
        os_36279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 28)
        path_36280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 22), os_36279, 'path')
        # Obtaining the member 'join' of a type (line 28)
        join_36281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 22), path_36280, 'join')
        # Calling join(args, kwargs) (line 28)
        join_call_result_36286 = invoke(stypy.reporting.localization.Localization(__file__, 28, 22), join_36281, *[root_target_36283, str_36284], **kwargs_36285)
        
        # Getting the type of 'self' (line 28)
        self_36287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'target' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_36287, 'target', join_call_result_36286)
        
        # Assigning a Call to a Attribute (line 29):
        
        # Call to join(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'tmp_dir' (line 29)
        tmp_dir_36291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 36), 'tmp_dir', False)
        str_36292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 45), 'str', 'deep2')
        # Processing the call keyword arguments (line 29)
        kwargs_36293 = {}
        # Getting the type of 'os' (line 29)
        os_36288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 29)
        path_36289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 23), os_36288, 'path')
        # Obtaining the member 'join' of a type (line 29)
        join_36290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 23), path_36289, 'join')
        # Calling join(args, kwargs) (line 29)
        join_call_result_36294 = invoke(stypy.reporting.localization.Localization(__file__, 29, 23), join_36290, *[tmp_dir_36291, str_36292], **kwargs_36293)
        
        # Getting the type of 'self' (line 29)
        self_36295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'target2' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_36295, 'target2', join_call_result_36294)
        
        # Assigning a Attribute to a Attribute (line 30):
        # Getting the type of 'log' (line 30)
        log_36296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'log')
        # Obtaining the member 'info' of a type (line 30)
        info_36297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 23), log_36296, 'info')
        # Getting the type of 'self' (line 30)
        self_36298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'old_log' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_36298, 'old_log', info_36297)
        
        # Assigning a Attribute to a Attribute (line 31):
        # Getting the type of 'self' (line 31)
        self_36299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'self')
        # Obtaining the member '_log' of a type (line 31)
        _log_36300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 19), self_36299, '_log')
        # Getting the type of 'log' (line 31)
        log_36301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'log')
        # Setting the type of the member 'info' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), log_36301, 'info', _log_36300)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_36302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36302)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_36302


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DirUtilTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        DirUtilTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DirUtilTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        DirUtilTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'DirUtilTestCase.tearDown')
        DirUtilTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        DirUtilTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        DirUtilTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DirUtilTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        DirUtilTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        DirUtilTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DirUtilTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DirUtilTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Attribute (line 34):
        # Getting the type of 'self' (line 34)
        self_36303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'self')
        # Obtaining the member 'old_log' of a type (line 34)
        old_log_36304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 19), self_36303, 'old_log')
        # Getting the type of 'log' (line 34)
        log_36305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'log')
        # Setting the type of the member 'info' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), log_36305, 'info', old_log_36304)
        
        # Call to tearDown(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_36312 = {}
        
        # Call to super(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'DirUtilTestCase' (line 35)
        DirUtilTestCase_36307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'DirUtilTestCase', False)
        # Getting the type of 'self' (line 35)
        self_36308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 31), 'self', False)
        # Processing the call keyword arguments (line 35)
        kwargs_36309 = {}
        # Getting the type of 'super' (line 35)
        super_36306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'super', False)
        # Calling super(args, kwargs) (line 35)
        super_call_result_36310 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), super_36306, *[DirUtilTestCase_36307, self_36308], **kwargs_36309)
        
        # Obtaining the member 'tearDown' of a type (line 35)
        tearDown_36311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), super_call_result_36310, 'tearDown')
        # Calling tearDown(args, kwargs) (line 35)
        tearDown_call_result_36313 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), tearDown_36311, *[], **kwargs_36312)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_36314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36314)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_36314


    @norecursion
    def test_mkpath_remove_tree_verbosity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_mkpath_remove_tree_verbosity'
        module_type_store = module_type_store.open_function_context('test_mkpath_remove_tree_verbosity', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DirUtilTestCase.test_mkpath_remove_tree_verbosity.__dict__.__setitem__('stypy_localization', localization)
        DirUtilTestCase.test_mkpath_remove_tree_verbosity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DirUtilTestCase.test_mkpath_remove_tree_verbosity.__dict__.__setitem__('stypy_type_store', module_type_store)
        DirUtilTestCase.test_mkpath_remove_tree_verbosity.__dict__.__setitem__('stypy_function_name', 'DirUtilTestCase.test_mkpath_remove_tree_verbosity')
        DirUtilTestCase.test_mkpath_remove_tree_verbosity.__dict__.__setitem__('stypy_param_names_list', [])
        DirUtilTestCase.test_mkpath_remove_tree_verbosity.__dict__.__setitem__('stypy_varargs_param_name', None)
        DirUtilTestCase.test_mkpath_remove_tree_verbosity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DirUtilTestCase.test_mkpath_remove_tree_verbosity.__dict__.__setitem__('stypy_call_defaults', defaults)
        DirUtilTestCase.test_mkpath_remove_tree_verbosity.__dict__.__setitem__('stypy_call_varargs', varargs)
        DirUtilTestCase.test_mkpath_remove_tree_verbosity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DirUtilTestCase.test_mkpath_remove_tree_verbosity.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DirUtilTestCase.test_mkpath_remove_tree_verbosity', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_mkpath_remove_tree_verbosity', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_mkpath_remove_tree_verbosity(...)' code ##################

        
        # Call to mkpath(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'self' (line 39)
        self_36316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'self', False)
        # Obtaining the member 'target' of a type (line 39)
        target_36317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 15), self_36316, 'target')
        # Processing the call keyword arguments (line 39)
        int_36318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'int')
        keyword_36319 = int_36318
        kwargs_36320 = {'verbose': keyword_36319}
        # Getting the type of 'mkpath' (line 39)
        mkpath_36315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'mkpath', False)
        # Calling mkpath(args, kwargs) (line 39)
        mkpath_call_result_36321 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), mkpath_36315, *[target_36317], **kwargs_36320)
        
        
        # Assigning a List to a Name (line 40):
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_36322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        
        # Assigning a type to the variable 'wanted' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'wanted', list_36322)
        
        # Call to assertEqual(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'self' (line 41)
        self_36325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'self', False)
        # Obtaining the member '_logs' of a type (line 41)
        _logs_36326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 25), self_36325, '_logs')
        # Getting the type of 'wanted' (line 41)
        wanted_36327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 37), 'wanted', False)
        # Processing the call keyword arguments (line 41)
        kwargs_36328 = {}
        # Getting the type of 'self' (line 41)
        self_36323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 41)
        assertEqual_36324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_36323, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 41)
        assertEqual_call_result_36329 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), assertEqual_36324, *[_logs_36326, wanted_36327], **kwargs_36328)
        
        
        # Call to remove_tree(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'self' (line 42)
        self_36331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'self', False)
        # Obtaining the member 'root_target' of a type (line 42)
        root_target_36332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 20), self_36331, 'root_target')
        # Processing the call keyword arguments (line 42)
        int_36333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 46), 'int')
        keyword_36334 = int_36333
        kwargs_36335 = {'verbose': keyword_36334}
        # Getting the type of 'remove_tree' (line 42)
        remove_tree_36330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 42)
        remove_tree_call_result_36336 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), remove_tree_36330, *[root_target_36332], **kwargs_36335)
        
        
        # Call to mkpath(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'self' (line 44)
        self_36338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'self', False)
        # Obtaining the member 'target' of a type (line 44)
        target_36339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 15), self_36338, 'target')
        # Processing the call keyword arguments (line 44)
        int_36340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 36), 'int')
        keyword_36341 = int_36340
        kwargs_36342 = {'verbose': keyword_36341}
        # Getting the type of 'mkpath' (line 44)
        mkpath_36337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'mkpath', False)
        # Calling mkpath(args, kwargs) (line 44)
        mkpath_call_result_36343 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), mkpath_36337, *[target_36339], **kwargs_36342)
        
        
        # Assigning a List to a Name (line 45):
        
        # Obtaining an instance of the builtin type 'list' (line 45)
        list_36344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 45)
        # Adding element type (line 45)
        str_36345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 18), 'str', 'creating %s')
        # Getting the type of 'self' (line 45)
        self_36346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), 'self')
        # Obtaining the member 'root_target' of a type (line 45)
        root_target_36347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 34), self_36346, 'root_target')
        # Applying the binary operator '%' (line 45)
        result_mod_36348 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 18), '%', str_36345, root_target_36347)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 17), list_36344, result_mod_36348)
        # Adding element type (line 45)
        str_36349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 18), 'str', 'creating %s')
        # Getting the type of 'self' (line 46)
        self_36350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'self')
        # Obtaining the member 'target' of a type (line 46)
        target_36351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 34), self_36350, 'target')
        # Applying the binary operator '%' (line 46)
        result_mod_36352 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 18), '%', str_36349, target_36351)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 17), list_36344, result_mod_36352)
        
        # Assigning a type to the variable 'wanted' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'wanted', list_36344)
        
        # Call to assertEqual(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'self' (line 47)
        self_36355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'self', False)
        # Obtaining the member '_logs' of a type (line 47)
        _logs_36356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 25), self_36355, '_logs')
        # Getting the type of 'wanted' (line 47)
        wanted_36357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'wanted', False)
        # Processing the call keyword arguments (line 47)
        kwargs_36358 = {}
        # Getting the type of 'self' (line 47)
        self_36353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 47)
        assertEqual_36354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_36353, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 47)
        assertEqual_call_result_36359 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assertEqual_36354, *[_logs_36356, wanted_36357], **kwargs_36358)
        
        
        # Assigning a List to a Attribute (line 48):
        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_36360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        
        # Getting the type of 'self' (line 48)
        self_36361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member '_logs' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_36361, '_logs', list_36360)
        
        # Call to remove_tree(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'self' (line 50)
        self_36363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'self', False)
        # Obtaining the member 'root_target' of a type (line 50)
        root_target_36364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 20), self_36363, 'root_target')
        # Processing the call keyword arguments (line 50)
        int_36365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 46), 'int')
        keyword_36366 = int_36365
        kwargs_36367 = {'verbose': keyword_36366}
        # Getting the type of 'remove_tree' (line 50)
        remove_tree_36362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 50)
        remove_tree_call_result_36368 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), remove_tree_36362, *[root_target_36364], **kwargs_36367)
        
        
        # Assigning a List to a Name (line 51):
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_36369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        str_36370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 18), 'str', "removing '%s' (and everything under it)")
        # Getting the type of 'self' (line 51)
        self_36371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 62), 'self')
        # Obtaining the member 'root_target' of a type (line 51)
        root_target_36372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 62), self_36371, 'root_target')
        # Applying the binary operator '%' (line 51)
        result_mod_36373 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 18), '%', str_36370, root_target_36372)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 17), list_36369, result_mod_36373)
        
        # Assigning a type to the variable 'wanted' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'wanted', list_36369)
        
        # Call to assertEqual(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'self' (line 52)
        self_36376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 25), 'self', False)
        # Obtaining the member '_logs' of a type (line 52)
        _logs_36377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 25), self_36376, '_logs')
        # Getting the type of 'wanted' (line 52)
        wanted_36378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 37), 'wanted', False)
        # Processing the call keyword arguments (line 52)
        kwargs_36379 = {}
        # Getting the type of 'self' (line 52)
        self_36374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 52)
        assertEqual_36375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_36374, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 52)
        assertEqual_call_result_36380 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), assertEqual_36375, *[_logs_36377, wanted_36378], **kwargs_36379)
        
        
        # ################# End of 'test_mkpath_remove_tree_verbosity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_mkpath_remove_tree_verbosity' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_36381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36381)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_mkpath_remove_tree_verbosity'
        return stypy_return_type_36381


    @norecursion
    def test_mkpath_with_custom_mode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_mkpath_with_custom_mode'
        module_type_store = module_type_store.open_function_context('test_mkpath_with_custom_mode', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DirUtilTestCase.test_mkpath_with_custom_mode.__dict__.__setitem__('stypy_localization', localization)
        DirUtilTestCase.test_mkpath_with_custom_mode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DirUtilTestCase.test_mkpath_with_custom_mode.__dict__.__setitem__('stypy_type_store', module_type_store)
        DirUtilTestCase.test_mkpath_with_custom_mode.__dict__.__setitem__('stypy_function_name', 'DirUtilTestCase.test_mkpath_with_custom_mode')
        DirUtilTestCase.test_mkpath_with_custom_mode.__dict__.__setitem__('stypy_param_names_list', [])
        DirUtilTestCase.test_mkpath_with_custom_mode.__dict__.__setitem__('stypy_varargs_param_name', None)
        DirUtilTestCase.test_mkpath_with_custom_mode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DirUtilTestCase.test_mkpath_with_custom_mode.__dict__.__setitem__('stypy_call_defaults', defaults)
        DirUtilTestCase.test_mkpath_with_custom_mode.__dict__.__setitem__('stypy_call_varargs', varargs)
        DirUtilTestCase.test_mkpath_with_custom_mode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DirUtilTestCase.test_mkpath_with_custom_mode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DirUtilTestCase.test_mkpath_with_custom_mode', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_mkpath_with_custom_mode', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_mkpath_with_custom_mode(...)' code ##################

        
        # Assigning a Call to a Name (line 58):
        
        # Call to umask(...): (line 58)
        # Processing the call arguments (line 58)
        int_36384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'int')
        # Processing the call keyword arguments (line 58)
        kwargs_36385 = {}
        # Getting the type of 'os' (line 58)
        os_36382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'os', False)
        # Obtaining the member 'umask' of a type (line 58)
        umask_36383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), os_36382, 'umask')
        # Calling umask(args, kwargs) (line 58)
        umask_call_result_36386 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), umask_36383, *[int_36384], **kwargs_36385)
        
        # Assigning a type to the variable 'umask' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'umask', umask_call_result_36386)
        
        # Call to umask(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'umask' (line 59)
        umask_36389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 17), 'umask', False)
        # Processing the call keyword arguments (line 59)
        kwargs_36390 = {}
        # Getting the type of 'os' (line 59)
        os_36387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'os', False)
        # Obtaining the member 'umask' of a type (line 59)
        umask_36388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), os_36387, 'umask')
        # Calling umask(args, kwargs) (line 59)
        umask_call_result_36391 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), umask_36388, *[umask_36389], **kwargs_36390)
        
        
        # Call to mkpath(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'self' (line 60)
        self_36393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'self', False)
        # Obtaining the member 'target' of a type (line 60)
        target_36394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 15), self_36393, 'target')
        int_36395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'int')
        # Processing the call keyword arguments (line 60)
        kwargs_36396 = {}
        # Getting the type of 'mkpath' (line 60)
        mkpath_36392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'mkpath', False)
        # Calling mkpath(args, kwargs) (line 60)
        mkpath_call_result_36397 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), mkpath_36392, *[target_36394, int_36395], **kwargs_36396)
        
        
        # Call to assertEqual(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Call to S_IMODE(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to stat(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_36404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 33), 'self', False)
        # Obtaining the member 'target' of a type (line 62)
        target_36405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 33), self_36404, 'target')
        # Processing the call keyword arguments (line 62)
        kwargs_36406 = {}
        # Getting the type of 'os' (line 62)
        os_36402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'os', False)
        # Obtaining the member 'stat' of a type (line 62)
        stat_36403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 25), os_36402, 'stat')
        # Calling stat(args, kwargs) (line 62)
        stat_call_result_36407 = invoke(stypy.reporting.localization.Localization(__file__, 62, 25), stat_36403, *[target_36405], **kwargs_36406)
        
        # Obtaining the member 'st_mode' of a type (line 62)
        st_mode_36408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 25), stat_call_result_36407, 'st_mode')
        # Processing the call keyword arguments (line 62)
        kwargs_36409 = {}
        # Getting the type of 'stat' (line 62)
        stat_36400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'stat', False)
        # Obtaining the member 'S_IMODE' of a type (line 62)
        S_IMODE_36401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), stat_36400, 'S_IMODE')
        # Calling S_IMODE(args, kwargs) (line 62)
        S_IMODE_call_result_36410 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), S_IMODE_36401, *[st_mode_36408], **kwargs_36409)
        
        int_36411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 56), 'int')
        
        # Getting the type of 'umask' (line 62)
        umask_36412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 65), 'umask', False)
        # Applying the '~' unary operator (line 62)
        result_inv_36413 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 64), '~', umask_36412)
        
        # Applying the binary operator '&' (line 62)
        result_and__36414 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 56), '&', int_36411, result_inv_36413)
        
        # Processing the call keyword arguments (line 61)
        kwargs_36415 = {}
        # Getting the type of 'self' (line 61)
        self_36398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 61)
        assertEqual_36399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_36398, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 61)
        assertEqual_call_result_36416 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assertEqual_36399, *[S_IMODE_call_result_36410, result_and__36414], **kwargs_36415)
        
        
        # Call to mkpath(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'self' (line 63)
        self_36418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'self', False)
        # Obtaining the member 'target2' of a type (line 63)
        target2_36419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 15), self_36418, 'target2')
        int_36420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 29), 'int')
        # Processing the call keyword arguments (line 63)
        kwargs_36421 = {}
        # Getting the type of 'mkpath' (line 63)
        mkpath_36417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'mkpath', False)
        # Calling mkpath(args, kwargs) (line 63)
        mkpath_call_result_36422 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), mkpath_36417, *[target2_36419, int_36420], **kwargs_36421)
        
        
        # Call to assertEqual(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to S_IMODE(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Call to stat(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_36429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 33), 'self', False)
        # Obtaining the member 'target2' of a type (line 65)
        target2_36430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 33), self_36429, 'target2')
        # Processing the call keyword arguments (line 65)
        kwargs_36431 = {}
        # Getting the type of 'os' (line 65)
        os_36427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'os', False)
        # Obtaining the member 'stat' of a type (line 65)
        stat_36428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 25), os_36427, 'stat')
        # Calling stat(args, kwargs) (line 65)
        stat_call_result_36432 = invoke(stypy.reporting.localization.Localization(__file__, 65, 25), stat_36428, *[target2_36430], **kwargs_36431)
        
        # Obtaining the member 'st_mode' of a type (line 65)
        st_mode_36433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 25), stat_call_result_36432, 'st_mode')
        # Processing the call keyword arguments (line 65)
        kwargs_36434 = {}
        # Getting the type of 'stat' (line 65)
        stat_36425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'stat', False)
        # Obtaining the member 'S_IMODE' of a type (line 65)
        S_IMODE_36426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), stat_36425, 'S_IMODE')
        # Calling S_IMODE(args, kwargs) (line 65)
        S_IMODE_call_result_36435 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), S_IMODE_36426, *[st_mode_36433], **kwargs_36434)
        
        int_36436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 57), 'int')
        
        # Getting the type of 'umask' (line 65)
        umask_36437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 66), 'umask', False)
        # Applying the '~' unary operator (line 65)
        result_inv_36438 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 65), '~', umask_36437)
        
        # Applying the binary operator '&' (line 65)
        result_and__36439 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 57), '&', int_36436, result_inv_36438)
        
        # Processing the call keyword arguments (line 64)
        kwargs_36440 = {}
        # Getting the type of 'self' (line 64)
        self_36423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 64)
        assertEqual_36424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_36423, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 64)
        assertEqual_call_result_36441 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assertEqual_36424, *[S_IMODE_call_result_36435, result_and__36439], **kwargs_36440)
        
        
        # ################# End of 'test_mkpath_with_custom_mode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_mkpath_with_custom_mode' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_36442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36442)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_mkpath_with_custom_mode'
        return stypy_return_type_36442


    @norecursion
    def test_create_tree_verbosity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_create_tree_verbosity'
        module_type_store = module_type_store.open_function_context('test_create_tree_verbosity', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DirUtilTestCase.test_create_tree_verbosity.__dict__.__setitem__('stypy_localization', localization)
        DirUtilTestCase.test_create_tree_verbosity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DirUtilTestCase.test_create_tree_verbosity.__dict__.__setitem__('stypy_type_store', module_type_store)
        DirUtilTestCase.test_create_tree_verbosity.__dict__.__setitem__('stypy_function_name', 'DirUtilTestCase.test_create_tree_verbosity')
        DirUtilTestCase.test_create_tree_verbosity.__dict__.__setitem__('stypy_param_names_list', [])
        DirUtilTestCase.test_create_tree_verbosity.__dict__.__setitem__('stypy_varargs_param_name', None)
        DirUtilTestCase.test_create_tree_verbosity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DirUtilTestCase.test_create_tree_verbosity.__dict__.__setitem__('stypy_call_defaults', defaults)
        DirUtilTestCase.test_create_tree_verbosity.__dict__.__setitem__('stypy_call_varargs', varargs)
        DirUtilTestCase.test_create_tree_verbosity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DirUtilTestCase.test_create_tree_verbosity.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DirUtilTestCase.test_create_tree_verbosity', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_create_tree_verbosity', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_create_tree_verbosity(...)' code ##################

        
        # Call to create_tree(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'self' (line 69)
        self_36444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'self', False)
        # Obtaining the member 'root_target' of a type (line 69)
        root_target_36445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 20), self_36444, 'root_target')
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_36446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        str_36447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 39), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 38), list_36446, str_36447)
        # Adding element type (line 69)
        str_36448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 46), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 38), list_36446, str_36448)
        # Adding element type (line 69)
        str_36449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 53), 'str', 'three')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 38), list_36446, str_36449)
        
        # Processing the call keyword arguments (line 69)
        int_36450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 71), 'int')
        keyword_36451 = int_36450
        kwargs_36452 = {'verbose': keyword_36451}
        # Getting the type of 'create_tree' (line 69)
        create_tree_36443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'create_tree', False)
        # Calling create_tree(args, kwargs) (line 69)
        create_tree_call_result_36453 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), create_tree_36443, *[root_target_36445, list_36446], **kwargs_36452)
        
        
        # Call to assertEqual(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'self' (line 70)
        self_36456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'self', False)
        # Obtaining the member '_logs' of a type (line 70)
        _logs_36457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 25), self_36456, '_logs')
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_36458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        
        # Processing the call keyword arguments (line 70)
        kwargs_36459 = {}
        # Getting the type of 'self' (line 70)
        self_36454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 70)
        assertEqual_36455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_36454, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 70)
        assertEqual_call_result_36460 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), assertEqual_36455, *[_logs_36457, list_36458], **kwargs_36459)
        
        
        # Call to remove_tree(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'self' (line 71)
        self_36462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'self', False)
        # Obtaining the member 'root_target' of a type (line 71)
        root_target_36463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 20), self_36462, 'root_target')
        # Processing the call keyword arguments (line 71)
        int_36464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 46), 'int')
        keyword_36465 = int_36464
        kwargs_36466 = {'verbose': keyword_36465}
        # Getting the type of 'remove_tree' (line 71)
        remove_tree_36461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 71)
        remove_tree_call_result_36467 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), remove_tree_36461, *[root_target_36463], **kwargs_36466)
        
        
        # Assigning a List to a Name (line 73):
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_36468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        str_36469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 18), 'str', 'creating %s')
        # Getting the type of 'self' (line 73)
        self_36470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 34), 'self')
        # Obtaining the member 'root_target' of a type (line 73)
        root_target_36471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 34), self_36470, 'root_target')
        # Applying the binary operator '%' (line 73)
        result_mod_36472 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 18), '%', str_36469, root_target_36471)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 17), list_36468, result_mod_36472)
        
        # Assigning a type to the variable 'wanted' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'wanted', list_36468)
        
        # Call to create_tree(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'self' (line 74)
        self_36474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'self', False)
        # Obtaining the member 'root_target' of a type (line 74)
        root_target_36475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 20), self_36474, 'root_target')
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_36476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        str_36477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 39), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 38), list_36476, str_36477)
        # Adding element type (line 74)
        str_36478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 46), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 38), list_36476, str_36478)
        # Adding element type (line 74)
        str_36479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 53), 'str', 'three')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 38), list_36476, str_36479)
        
        # Processing the call keyword arguments (line 74)
        int_36480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 71), 'int')
        keyword_36481 = int_36480
        kwargs_36482 = {'verbose': keyword_36481}
        # Getting the type of 'create_tree' (line 74)
        create_tree_36473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'create_tree', False)
        # Calling create_tree(args, kwargs) (line 74)
        create_tree_call_result_36483 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), create_tree_36473, *[root_target_36475, list_36476], **kwargs_36482)
        
        
        # Call to assertEqual(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'self' (line 75)
        self_36486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 25), 'self', False)
        # Obtaining the member '_logs' of a type (line 75)
        _logs_36487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 25), self_36486, '_logs')
        # Getting the type of 'wanted' (line 75)
        wanted_36488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 37), 'wanted', False)
        # Processing the call keyword arguments (line 75)
        kwargs_36489 = {}
        # Getting the type of 'self' (line 75)
        self_36484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 75)
        assertEqual_36485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_36484, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 75)
        assertEqual_call_result_36490 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), assertEqual_36485, *[_logs_36487, wanted_36488], **kwargs_36489)
        
        
        # Call to remove_tree(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'self' (line 77)
        self_36492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'self', False)
        # Obtaining the member 'root_target' of a type (line 77)
        root_target_36493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 20), self_36492, 'root_target')
        # Processing the call keyword arguments (line 77)
        int_36494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 46), 'int')
        keyword_36495 = int_36494
        kwargs_36496 = {'verbose': keyword_36495}
        # Getting the type of 'remove_tree' (line 77)
        remove_tree_36491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 77)
        remove_tree_call_result_36497 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), remove_tree_36491, *[root_target_36493], **kwargs_36496)
        
        
        # ################# End of 'test_create_tree_verbosity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_create_tree_verbosity' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_36498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36498)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_create_tree_verbosity'
        return stypy_return_type_36498


    @norecursion
    def test_copy_tree_verbosity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_copy_tree_verbosity'
        module_type_store = module_type_store.open_function_context('test_copy_tree_verbosity', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DirUtilTestCase.test_copy_tree_verbosity.__dict__.__setitem__('stypy_localization', localization)
        DirUtilTestCase.test_copy_tree_verbosity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DirUtilTestCase.test_copy_tree_verbosity.__dict__.__setitem__('stypy_type_store', module_type_store)
        DirUtilTestCase.test_copy_tree_verbosity.__dict__.__setitem__('stypy_function_name', 'DirUtilTestCase.test_copy_tree_verbosity')
        DirUtilTestCase.test_copy_tree_verbosity.__dict__.__setitem__('stypy_param_names_list', [])
        DirUtilTestCase.test_copy_tree_verbosity.__dict__.__setitem__('stypy_varargs_param_name', None)
        DirUtilTestCase.test_copy_tree_verbosity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DirUtilTestCase.test_copy_tree_verbosity.__dict__.__setitem__('stypy_call_defaults', defaults)
        DirUtilTestCase.test_copy_tree_verbosity.__dict__.__setitem__('stypy_call_varargs', varargs)
        DirUtilTestCase.test_copy_tree_verbosity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DirUtilTestCase.test_copy_tree_verbosity.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DirUtilTestCase.test_copy_tree_verbosity', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_copy_tree_verbosity', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_copy_tree_verbosity(...)' code ##################

        
        # Call to mkpath(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'self' (line 82)
        self_36500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'self', False)
        # Obtaining the member 'target' of a type (line 82)
        target_36501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), self_36500, 'target')
        # Processing the call keyword arguments (line 82)
        int_36502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 36), 'int')
        keyword_36503 = int_36502
        kwargs_36504 = {'verbose': keyword_36503}
        # Getting the type of 'mkpath' (line 82)
        mkpath_36499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'mkpath', False)
        # Calling mkpath(args, kwargs) (line 82)
        mkpath_call_result_36505 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), mkpath_36499, *[target_36501], **kwargs_36504)
        
        
        # Call to copy_tree(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'self' (line 84)
        self_36507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'self', False)
        # Obtaining the member 'target' of a type (line 84)
        target_36508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 18), self_36507, 'target')
        # Getting the type of 'self' (line 84)
        self_36509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'self', False)
        # Obtaining the member 'target2' of a type (line 84)
        target2_36510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 31), self_36509, 'target2')
        # Processing the call keyword arguments (line 84)
        int_36511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 53), 'int')
        keyword_36512 = int_36511
        kwargs_36513 = {'verbose': keyword_36512}
        # Getting the type of 'copy_tree' (line 84)
        copy_tree_36506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'copy_tree', False)
        # Calling copy_tree(args, kwargs) (line 84)
        copy_tree_call_result_36514 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), copy_tree_36506, *[target_36508, target2_36510], **kwargs_36513)
        
        
        # Call to assertEqual(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'self' (line 85)
        self_36517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'self', False)
        # Obtaining the member '_logs' of a type (line 85)
        _logs_36518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 25), self_36517, '_logs')
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_36519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        
        # Processing the call keyword arguments (line 85)
        kwargs_36520 = {}
        # Getting the type of 'self' (line 85)
        self_36515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 85)
        assertEqual_36516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_36515, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 85)
        assertEqual_call_result_36521 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), assertEqual_36516, *[_logs_36518, list_36519], **kwargs_36520)
        
        
        # Call to remove_tree(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'self' (line 87)
        self_36523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'self', False)
        # Obtaining the member 'root_target' of a type (line 87)
        root_target_36524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 20), self_36523, 'root_target')
        # Processing the call keyword arguments (line 87)
        int_36525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 46), 'int')
        keyword_36526 = int_36525
        kwargs_36527 = {'verbose': keyword_36526}
        # Getting the type of 'remove_tree' (line 87)
        remove_tree_36522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 87)
        remove_tree_call_result_36528 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), remove_tree_36522, *[root_target_36524], **kwargs_36527)
        
        
        # Call to mkpath(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'self' (line 89)
        self_36530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'self', False)
        # Obtaining the member 'target' of a type (line 89)
        target_36531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 15), self_36530, 'target')
        # Processing the call keyword arguments (line 89)
        int_36532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 36), 'int')
        keyword_36533 = int_36532
        kwargs_36534 = {'verbose': keyword_36533}
        # Getting the type of 'mkpath' (line 89)
        mkpath_36529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'mkpath', False)
        # Calling mkpath(args, kwargs) (line 89)
        mkpath_call_result_36535 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), mkpath_36529, *[target_36531], **kwargs_36534)
        
        
        # Assigning a Call to a Name (line 90):
        
        # Call to join(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'self' (line 90)
        self_36539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'self', False)
        # Obtaining the member 'target' of a type (line 90)
        target_36540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 30), self_36539, 'target')
        str_36541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 43), 'str', 'ok.txt')
        # Processing the call keyword arguments (line 90)
        kwargs_36542 = {}
        # Getting the type of 'os' (line 90)
        os_36536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 90)
        path_36537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 17), os_36536, 'path')
        # Obtaining the member 'join' of a type (line 90)
        join_36538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 17), path_36537, 'join')
        # Calling join(args, kwargs) (line 90)
        join_call_result_36543 = invoke(stypy.reporting.localization.Localization(__file__, 90, 17), join_36538, *[target_36540, str_36541], **kwargs_36542)
        
        # Assigning a type to the variable 'a_file' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'a_file', join_call_result_36543)
        
        # Assigning a Call to a Name (line 91):
        
        # Call to open(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'a_file' (line 91)
        a_file_36545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 17), 'a_file', False)
        str_36546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 25), 'str', 'w')
        # Processing the call keyword arguments (line 91)
        kwargs_36547 = {}
        # Getting the type of 'open' (line 91)
        open_36544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'open', False)
        # Calling open(args, kwargs) (line 91)
        open_call_result_36548 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), open_36544, *[a_file_36545, str_36546], **kwargs_36547)
        
        # Assigning a type to the variable 'f' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'f', open_call_result_36548)
        
        # Try-finally block (line 92)
        
        # Call to write(...): (line 93)
        # Processing the call arguments (line 93)
        str_36551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'str', 'some content')
        # Processing the call keyword arguments (line 93)
        kwargs_36552 = {}
        # Getting the type of 'f' (line 93)
        f_36549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 93)
        write_36550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), f_36549, 'write')
        # Calling write(args, kwargs) (line 93)
        write_call_result_36553 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), write_36550, *[str_36551], **kwargs_36552)
        
        
        # finally branch of the try-finally block (line 92)
        
        # Call to close(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_36556 = {}
        # Getting the type of 'f' (line 95)
        f_36554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 95)
        close_36555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), f_36554, 'close')
        # Calling close(args, kwargs) (line 95)
        close_call_result_36557 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), close_36555, *[], **kwargs_36556)
        
        
        
        # Assigning a List to a Name (line 97):
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_36558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        str_36559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 18), 'str', 'copying %s -> %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 97)
        tuple_36560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 97)
        # Adding element type (line 97)
        # Getting the type of 'a_file' (line 97)
        a_file_36561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'a_file')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 40), tuple_36560, a_file_36561)
        # Adding element type (line 97)
        # Getting the type of 'self' (line 97)
        self_36562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 48), 'self')
        # Obtaining the member 'target2' of a type (line 97)
        target2_36563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 48), self_36562, 'target2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 40), tuple_36560, target2_36563)
        
        # Applying the binary operator '%' (line 97)
        result_mod_36564 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 18), '%', str_36559, tuple_36560)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 17), list_36558, result_mod_36564)
        
        # Assigning a type to the variable 'wanted' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'wanted', list_36558)
        
        # Call to copy_tree(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'self' (line 98)
        self_36566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'self', False)
        # Obtaining the member 'target' of a type (line 98)
        target_36567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 18), self_36566, 'target')
        # Getting the type of 'self' (line 98)
        self_36568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'self', False)
        # Obtaining the member 'target2' of a type (line 98)
        target2_36569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 31), self_36568, 'target2')
        # Processing the call keyword arguments (line 98)
        int_36570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 53), 'int')
        keyword_36571 = int_36570
        kwargs_36572 = {'verbose': keyword_36571}
        # Getting the type of 'copy_tree' (line 98)
        copy_tree_36565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'copy_tree', False)
        # Calling copy_tree(args, kwargs) (line 98)
        copy_tree_call_result_36573 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), copy_tree_36565, *[target_36567, target2_36569], **kwargs_36572)
        
        
        # Call to assertEqual(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'self' (line 99)
        self_36576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'self', False)
        # Obtaining the member '_logs' of a type (line 99)
        _logs_36577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 25), self_36576, '_logs')
        # Getting the type of 'wanted' (line 99)
        wanted_36578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 37), 'wanted', False)
        # Processing the call keyword arguments (line 99)
        kwargs_36579 = {}
        # Getting the type of 'self' (line 99)
        self_36574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 99)
        assertEqual_36575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_36574, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 99)
        assertEqual_call_result_36580 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), assertEqual_36575, *[_logs_36577, wanted_36578], **kwargs_36579)
        
        
        # Call to remove_tree(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'self' (line 101)
        self_36582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'self', False)
        # Obtaining the member 'root_target' of a type (line 101)
        root_target_36583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 20), self_36582, 'root_target')
        # Processing the call keyword arguments (line 101)
        int_36584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 46), 'int')
        keyword_36585 = int_36584
        kwargs_36586 = {'verbose': keyword_36585}
        # Getting the type of 'remove_tree' (line 101)
        remove_tree_36581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 101)
        remove_tree_call_result_36587 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), remove_tree_36581, *[root_target_36583], **kwargs_36586)
        
        
        # Call to remove_tree(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'self' (line 102)
        self_36589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'self', False)
        # Obtaining the member 'target2' of a type (line 102)
        target2_36590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 20), self_36589, 'target2')
        # Processing the call keyword arguments (line 102)
        int_36591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 42), 'int')
        keyword_36592 = int_36591
        kwargs_36593 = {'verbose': keyword_36592}
        # Getting the type of 'remove_tree' (line 102)
        remove_tree_36588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 102)
        remove_tree_call_result_36594 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), remove_tree_36588, *[target2_36590], **kwargs_36593)
        
        
        # ################# End of 'test_copy_tree_verbosity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_copy_tree_verbosity' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_36595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36595)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_copy_tree_verbosity'
        return stypy_return_type_36595


    @norecursion
    def test_copy_tree_skips_nfs_temp_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_copy_tree_skips_nfs_temp_files'
        module_type_store = module_type_store.open_function_context('test_copy_tree_skips_nfs_temp_files', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DirUtilTestCase.test_copy_tree_skips_nfs_temp_files.__dict__.__setitem__('stypy_localization', localization)
        DirUtilTestCase.test_copy_tree_skips_nfs_temp_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DirUtilTestCase.test_copy_tree_skips_nfs_temp_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        DirUtilTestCase.test_copy_tree_skips_nfs_temp_files.__dict__.__setitem__('stypy_function_name', 'DirUtilTestCase.test_copy_tree_skips_nfs_temp_files')
        DirUtilTestCase.test_copy_tree_skips_nfs_temp_files.__dict__.__setitem__('stypy_param_names_list', [])
        DirUtilTestCase.test_copy_tree_skips_nfs_temp_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        DirUtilTestCase.test_copy_tree_skips_nfs_temp_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DirUtilTestCase.test_copy_tree_skips_nfs_temp_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        DirUtilTestCase.test_copy_tree_skips_nfs_temp_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        DirUtilTestCase.test_copy_tree_skips_nfs_temp_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DirUtilTestCase.test_copy_tree_skips_nfs_temp_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DirUtilTestCase.test_copy_tree_skips_nfs_temp_files', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_copy_tree_skips_nfs_temp_files', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_copy_tree_skips_nfs_temp_files(...)' code ##################

        
        # Call to mkpath(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'self' (line 105)
        self_36597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'self', False)
        # Obtaining the member 'target' of a type (line 105)
        target_36598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 15), self_36597, 'target')
        # Processing the call keyword arguments (line 105)
        int_36599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 36), 'int')
        keyword_36600 = int_36599
        kwargs_36601 = {'verbose': keyword_36600}
        # Getting the type of 'mkpath' (line 105)
        mkpath_36596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'mkpath', False)
        # Calling mkpath(args, kwargs) (line 105)
        mkpath_call_result_36602 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), mkpath_36596, *[target_36598], **kwargs_36601)
        
        
        # Assigning a Call to a Name (line 107):
        
        # Call to join(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'self' (line 107)
        self_36606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), 'self', False)
        # Obtaining the member 'target' of a type (line 107)
        target_36607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 30), self_36606, 'target')
        str_36608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 43), 'str', 'ok.txt')
        # Processing the call keyword arguments (line 107)
        kwargs_36609 = {}
        # Getting the type of 'os' (line 107)
        os_36603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 107)
        path_36604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 17), os_36603, 'path')
        # Obtaining the member 'join' of a type (line 107)
        join_36605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 17), path_36604, 'join')
        # Calling join(args, kwargs) (line 107)
        join_call_result_36610 = invoke(stypy.reporting.localization.Localization(__file__, 107, 17), join_36605, *[target_36607, str_36608], **kwargs_36609)
        
        # Assigning a type to the variable 'a_file' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'a_file', join_call_result_36610)
        
        # Assigning a Call to a Name (line 108):
        
        # Call to join(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'self' (line 108)
        self_36614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 32), 'self', False)
        # Obtaining the member 'target' of a type (line 108)
        target_36615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 32), self_36614, 'target')
        str_36616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 45), 'str', '.nfs123abc')
        # Processing the call keyword arguments (line 108)
        kwargs_36617 = {}
        # Getting the type of 'os' (line 108)
        os_36611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 108)
        path_36612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 19), os_36611, 'path')
        # Obtaining the member 'join' of a type (line 108)
        join_36613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 19), path_36612, 'join')
        # Calling join(args, kwargs) (line 108)
        join_call_result_36618 = invoke(stypy.reporting.localization.Localization(__file__, 108, 19), join_36613, *[target_36615, str_36616], **kwargs_36617)
        
        # Assigning a type to the variable 'nfs_file' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'nfs_file', join_call_result_36618)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 109)
        tuple_36619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 109)
        # Adding element type (line 109)
        # Getting the type of 'a_file' (line 109)
        a_file_36620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'a_file')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 17), tuple_36619, a_file_36620)
        # Adding element type (line 109)
        # Getting the type of 'nfs_file' (line 109)
        nfs_file_36621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 'nfs_file')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 17), tuple_36619, nfs_file_36621)
        
        # Testing the type of a for loop iterable (line 109)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 8), tuple_36619)
        # Getting the type of the for loop variable (line 109)
        for_loop_var_36622 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 8), tuple_36619)
        # Assigning a type to the variable 'f' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'f', for_loop_var_36622)
        # SSA begins for a for statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 110):
        
        # Call to open(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'f' (line 110)
        f_36624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'f', False)
        str_36625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'str', 'w')
        # Processing the call keyword arguments (line 110)
        kwargs_36626 = {}
        # Getting the type of 'open' (line 110)
        open_36623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'open', False)
        # Calling open(args, kwargs) (line 110)
        open_call_result_36627 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), open_36623, *[f_36624, str_36625], **kwargs_36626)
        
        # Assigning a type to the variable 'fh' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'fh', open_call_result_36627)
        
        # Try-finally block (line 111)
        
        # Call to write(...): (line 112)
        # Processing the call arguments (line 112)
        str_36630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 25), 'str', 'some content')
        # Processing the call keyword arguments (line 112)
        kwargs_36631 = {}
        # Getting the type of 'fh' (line 112)
        fh_36628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'fh', False)
        # Obtaining the member 'write' of a type (line 112)
        write_36629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), fh_36628, 'write')
        # Calling write(args, kwargs) (line 112)
        write_call_result_36632 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), write_36629, *[str_36630], **kwargs_36631)
        
        
        # finally branch of the try-finally block (line 111)
        
        # Call to close(...): (line 114)
        # Processing the call keyword arguments (line 114)
        kwargs_36635 = {}
        # Getting the type of 'fh' (line 114)
        fh_36633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'fh', False)
        # Obtaining the member 'close' of a type (line 114)
        close_36634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 16), fh_36633, 'close')
        # Calling close(args, kwargs) (line 114)
        close_call_result_36636 = invoke(stypy.reporting.localization.Localization(__file__, 114, 16), close_36634, *[], **kwargs_36635)
        
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to copy_tree(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'self' (line 116)
        self_36638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'self', False)
        # Obtaining the member 'target' of a type (line 116)
        target_36639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 18), self_36638, 'target')
        # Getting the type of 'self' (line 116)
        self_36640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'self', False)
        # Obtaining the member 'target2' of a type (line 116)
        target2_36641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 31), self_36640, 'target2')
        # Processing the call keyword arguments (line 116)
        kwargs_36642 = {}
        # Getting the type of 'copy_tree' (line 116)
        copy_tree_36637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'copy_tree', False)
        # Calling copy_tree(args, kwargs) (line 116)
        copy_tree_call_result_36643 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), copy_tree_36637, *[target_36639, target2_36641], **kwargs_36642)
        
        
        # Call to assertEqual(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Call to listdir(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'self' (line 117)
        self_36648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 36), 'self', False)
        # Obtaining the member 'target2' of a type (line 117)
        target2_36649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 36), self_36648, 'target2')
        # Processing the call keyword arguments (line 117)
        kwargs_36650 = {}
        # Getting the type of 'os' (line 117)
        os_36646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'os', False)
        # Obtaining the member 'listdir' of a type (line 117)
        listdir_36647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 25), os_36646, 'listdir')
        # Calling listdir(args, kwargs) (line 117)
        listdir_call_result_36651 = invoke(stypy.reporting.localization.Localization(__file__, 117, 25), listdir_36647, *[target2_36649], **kwargs_36650)
        
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_36652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        str_36653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 52), 'str', 'ok.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 51), list_36652, str_36653)
        
        # Processing the call keyword arguments (line 117)
        kwargs_36654 = {}
        # Getting the type of 'self' (line 117)
        self_36644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 117)
        assertEqual_36645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_36644, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 117)
        assertEqual_call_result_36655 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), assertEqual_36645, *[listdir_call_result_36651, list_36652], **kwargs_36654)
        
        
        # Call to remove_tree(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'self' (line 119)
        self_36657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'self', False)
        # Obtaining the member 'root_target' of a type (line 119)
        root_target_36658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 20), self_36657, 'root_target')
        # Processing the call keyword arguments (line 119)
        int_36659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 46), 'int')
        keyword_36660 = int_36659
        kwargs_36661 = {'verbose': keyword_36660}
        # Getting the type of 'remove_tree' (line 119)
        remove_tree_36656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 119)
        remove_tree_call_result_36662 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), remove_tree_36656, *[root_target_36658], **kwargs_36661)
        
        
        # Call to remove_tree(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'self' (line 120)
        self_36664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'self', False)
        # Obtaining the member 'target2' of a type (line 120)
        target2_36665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 20), self_36664, 'target2')
        # Processing the call keyword arguments (line 120)
        int_36666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 42), 'int')
        keyword_36667 = int_36666
        kwargs_36668 = {'verbose': keyword_36667}
        # Getting the type of 'remove_tree' (line 120)
        remove_tree_36663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 120)
        remove_tree_call_result_36669 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), remove_tree_36663, *[target2_36665], **kwargs_36668)
        
        
        # ################# End of 'test_copy_tree_skips_nfs_temp_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_copy_tree_skips_nfs_temp_files' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_36670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36670)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_copy_tree_skips_nfs_temp_files'
        return stypy_return_type_36670


    @norecursion
    def test_ensure_relative(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ensure_relative'
        module_type_store = module_type_store.open_function_context('test_ensure_relative', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DirUtilTestCase.test_ensure_relative.__dict__.__setitem__('stypy_localization', localization)
        DirUtilTestCase.test_ensure_relative.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DirUtilTestCase.test_ensure_relative.__dict__.__setitem__('stypy_type_store', module_type_store)
        DirUtilTestCase.test_ensure_relative.__dict__.__setitem__('stypy_function_name', 'DirUtilTestCase.test_ensure_relative')
        DirUtilTestCase.test_ensure_relative.__dict__.__setitem__('stypy_param_names_list', [])
        DirUtilTestCase.test_ensure_relative.__dict__.__setitem__('stypy_varargs_param_name', None)
        DirUtilTestCase.test_ensure_relative.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DirUtilTestCase.test_ensure_relative.__dict__.__setitem__('stypy_call_defaults', defaults)
        DirUtilTestCase.test_ensure_relative.__dict__.__setitem__('stypy_call_varargs', varargs)
        DirUtilTestCase.test_ensure_relative.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DirUtilTestCase.test_ensure_relative.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DirUtilTestCase.test_ensure_relative', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ensure_relative', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ensure_relative(...)' code ##################

        
        
        # Getting the type of 'os' (line 123)
        os_36671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'os')
        # Obtaining the member 'sep' of a type (line 123)
        sep_36672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 11), os_36671, 'sep')
        str_36673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 21), 'str', '/')
        # Applying the binary operator '==' (line 123)
        result_eq_36674 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 11), '==', sep_36672, str_36673)
        
        # Testing the type of an if condition (line 123)
        if_condition_36675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 8), result_eq_36674)
        # Assigning a type to the variable 'if_condition_36675' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'if_condition_36675', if_condition_36675)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assertEqual(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Call to ensure_relative(...): (line 124)
        # Processing the call arguments (line 124)
        str_36679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 45), 'str', '/home/foo')
        # Processing the call keyword arguments (line 124)
        kwargs_36680 = {}
        # Getting the type of 'ensure_relative' (line 124)
        ensure_relative_36678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 29), 'ensure_relative', False)
        # Calling ensure_relative(args, kwargs) (line 124)
        ensure_relative_call_result_36681 = invoke(stypy.reporting.localization.Localization(__file__, 124, 29), ensure_relative_36678, *[str_36679], **kwargs_36680)
        
        str_36682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 59), 'str', 'home/foo')
        # Processing the call keyword arguments (line 124)
        kwargs_36683 = {}
        # Getting the type of 'self' (line 124)
        self_36676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 124)
        assertEqual_36677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), self_36676, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 124)
        assertEqual_call_result_36684 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), assertEqual_36677, *[ensure_relative_call_result_36681, str_36682], **kwargs_36683)
        
        
        # Call to assertEqual(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Call to ensure_relative(...): (line 125)
        # Processing the call arguments (line 125)
        str_36688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 45), 'str', 'some/path')
        # Processing the call keyword arguments (line 125)
        kwargs_36689 = {}
        # Getting the type of 'ensure_relative' (line 125)
        ensure_relative_36687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 29), 'ensure_relative', False)
        # Calling ensure_relative(args, kwargs) (line 125)
        ensure_relative_call_result_36690 = invoke(stypy.reporting.localization.Localization(__file__, 125, 29), ensure_relative_36687, *[str_36688], **kwargs_36689)
        
        str_36691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 59), 'str', 'some/path')
        # Processing the call keyword arguments (line 125)
        kwargs_36692 = {}
        # Getting the type of 'self' (line 125)
        self_36685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 125)
        assertEqual_36686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), self_36685, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 125)
        assertEqual_call_result_36693 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), assertEqual_36686, *[ensure_relative_call_result_36690, str_36691], **kwargs_36692)
        
        # SSA branch for the else part of an if statement (line 123)
        module_type_store.open_ssa_branch('else')
        
        # Call to assertEqual(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Call to ensure_relative(...): (line 127)
        # Processing the call arguments (line 127)
        str_36697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 45), 'str', 'c:\\home\\foo')
        # Processing the call keyword arguments (line 127)
        kwargs_36698 = {}
        # Getting the type of 'ensure_relative' (line 127)
        ensure_relative_36696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 29), 'ensure_relative', False)
        # Calling ensure_relative(args, kwargs) (line 127)
        ensure_relative_call_result_36699 = invoke(stypy.reporting.localization.Localization(__file__, 127, 29), ensure_relative_36696, *[str_36697], **kwargs_36698)
        
        str_36700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 63), 'str', 'c:home\\foo')
        # Processing the call keyword arguments (line 127)
        kwargs_36701 = {}
        # Getting the type of 'self' (line 127)
        self_36694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 127)
        assertEqual_36695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), self_36694, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 127)
        assertEqual_call_result_36702 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), assertEqual_36695, *[ensure_relative_call_result_36699, str_36700], **kwargs_36701)
        
        
        # Call to assertEqual(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Call to ensure_relative(...): (line 128)
        # Processing the call arguments (line 128)
        str_36706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 45), 'str', 'home\\foo')
        # Processing the call keyword arguments (line 128)
        kwargs_36707 = {}
        # Getting the type of 'ensure_relative' (line 128)
        ensure_relative_36705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'ensure_relative', False)
        # Calling ensure_relative(args, kwargs) (line 128)
        ensure_relative_call_result_36708 = invoke(stypy.reporting.localization.Localization(__file__, 128, 29), ensure_relative_36705, *[str_36706], **kwargs_36707)
        
        str_36709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 59), 'str', 'home\\foo')
        # Processing the call keyword arguments (line 128)
        kwargs_36710 = {}
        # Getting the type of 'self' (line 128)
        self_36703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 128)
        assertEqual_36704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), self_36703, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 128)
        assertEqual_call_result_36711 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), assertEqual_36704, *[ensure_relative_call_result_36708, str_36709], **kwargs_36710)
        
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_ensure_relative(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ensure_relative' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_36712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36712)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ensure_relative'
        return stypy_return_type_36712


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DirUtilTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DirUtilTestCase' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'DirUtilTestCase', DirUtilTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 130, 0, False)
    
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

    
    # Call to makeSuite(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'DirUtilTestCase' (line 131)
    DirUtilTestCase_36715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'DirUtilTestCase', False)
    # Processing the call keyword arguments (line 131)
    kwargs_36716 = {}
    # Getting the type of 'unittest' (line 131)
    unittest_36713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 131)
    makeSuite_36714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 11), unittest_36713, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 131)
    makeSuite_call_result_36717 = invoke(stypy.reporting.localization.Localization(__file__, 131, 11), makeSuite_36714, *[DirUtilTestCase_36715], **kwargs_36716)
    
    # Assigning a type to the variable 'stypy_return_type' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type', makeSuite_call_result_36717)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 130)
    stypy_return_type_36718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36718)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_36718

# Assigning a type to the variable 'test_suite' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 134)
    # Processing the call arguments (line 134)
    
    # Call to test_suite(...): (line 134)
    # Processing the call keyword arguments (line 134)
    kwargs_36721 = {}
    # Getting the type of 'test_suite' (line 134)
    test_suite_36720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 134)
    test_suite_call_result_36722 = invoke(stypy.reporting.localization.Localization(__file__, 134, 17), test_suite_36720, *[], **kwargs_36721)
    
    # Processing the call keyword arguments (line 134)
    kwargs_36723 = {}
    # Getting the type of 'run_unittest' (line 134)
    run_unittest_36719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 134)
    run_unittest_call_result_36724 = invoke(stypy.reporting.localization.Localization(__file__, 134, 4), run_unittest_36719, *[test_suite_call_result_36722], **kwargs_36723)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
