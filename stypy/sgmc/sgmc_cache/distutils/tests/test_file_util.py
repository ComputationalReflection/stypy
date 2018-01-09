
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.file_util.'''
2: import unittest
3: import os
4: import shutil
5: 
6: from distutils.file_util import move_file, write_file, copy_file
7: from distutils import log
8: from distutils.tests import support
9: from test.test_support import run_unittest
10: 
11: 
12: requires_os_link = unittest.skipUnless(hasattr(os, "link"),
13:                                        "test requires os.link()")
14: 
15: 
16: class FileUtilTestCase(support.TempdirManager, unittest.TestCase):
17: 
18:     def _log(self, msg, *args):
19:         if len(args) > 0:
20:             self._logs.append(msg % args)
21:         else:
22:             self._logs.append(msg)
23: 
24:     def setUp(self):
25:         super(FileUtilTestCase, self).setUp()
26:         self._logs = []
27:         self.old_log = log.info
28:         log.info = self._log
29:         tmp_dir = self.mkdtemp()
30:         self.source = os.path.join(tmp_dir, 'f1')
31:         self.target = os.path.join(tmp_dir, 'f2')
32:         self.target_dir = os.path.join(tmp_dir, 'd1')
33: 
34:     def tearDown(self):
35:         log.info = self.old_log
36:         super(FileUtilTestCase, self).tearDown()
37: 
38:     def test_move_file_verbosity(self):
39:         f = open(self.source, 'w')
40:         try:
41:             f.write('some content')
42:         finally:
43:             f.close()
44: 
45:         move_file(self.source, self.target, verbose=0)
46:         wanted = []
47:         self.assertEqual(self._logs, wanted)
48: 
49:         # back to original state
50:         move_file(self.target, self.source, verbose=0)
51: 
52:         move_file(self.source, self.target, verbose=1)
53:         wanted = ['moving %s -> %s' % (self.source, self.target)]
54:         self.assertEqual(self._logs, wanted)
55: 
56:         # back to original state
57:         move_file(self.target, self.source, verbose=0)
58: 
59:         self._logs = []
60:         # now the target is a dir
61:         os.mkdir(self.target_dir)
62:         move_file(self.source, self.target_dir, verbose=1)
63:         wanted = ['moving %s -> %s' % (self.source, self.target_dir)]
64:         self.assertEqual(self._logs, wanted)
65: 
66:     def test_write_file(self):
67:         lines = ['a', 'b', 'c']
68:         dir = self.mkdtemp()
69:         foo = os.path.join(dir, 'foo')
70:         write_file(foo, lines)
71:         content = [line.strip() for line in open(foo).readlines()]
72:         self.assertEqual(content, lines)
73: 
74:     def test_copy_file(self):
75:         src_dir = self.mkdtemp()
76:         foo = os.path.join(src_dir, 'foo')
77:         write_file(foo, 'content')
78:         dst_dir = self.mkdtemp()
79:         copy_file(foo, dst_dir)
80:         self.assertTrue(os.path.exists(os.path.join(dst_dir, 'foo')))
81: 
82:     @requires_os_link
83:     def test_copy_file_hard_link(self):
84:         with open(self.source, 'w') as f:
85:             f.write('some content')
86:         st = os.stat(self.source)
87:         copy_file(self.source, self.target, link='hard')
88:         st2 = os.stat(self.source)
89:         st3 = os.stat(self.target)
90:         self.assertTrue(os.path.samestat(st, st2), (st, st2))
91:         self.assertTrue(os.path.samestat(st2, st3), (st2, st3))
92:         with open(self.source, 'r') as f:
93:             self.assertEqual(f.read(), 'some content')
94: 
95:     @requires_os_link
96:     def test_copy_file_hard_link_failure(self):
97:         # If hard linking fails, copy_file() falls back on copying file
98:         # (some special filesystems don't support hard linking even under
99:         #  Unix, see issue #8876).
100:         with open(self.source, 'w') as f:
101:             f.write('some content')
102:         st = os.stat(self.source)
103:         def _os_link(*args):
104:             raise OSError(0, "linking unsupported")
105:         old_link = os.link
106:         os.link = _os_link
107:         try:
108:             copy_file(self.source, self.target, link='hard')
109:         finally:
110:             os.link = old_link
111:         st2 = os.stat(self.source)
112:         st3 = os.stat(self.target)
113:         self.assertTrue(os.path.samestat(st, st2), (st, st2))
114:         self.assertFalse(os.path.samestat(st2, st3), (st2, st3))
115:         for fn in (self.source, self.target):
116:             with open(fn, 'r') as f:
117:                 self.assertEqual(f.read(), 'some content')
118: 
119: 
120: def test_suite():
121:     return unittest.makeSuite(FileUtilTestCase)
122: 
123: if __name__ == "__main__":
124:     run_unittest(test_suite())
125: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_39158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.file_util.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import unittest' statement (line 2)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import shutil' statement (line 4)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.file_util import move_file, write_file, copy_file' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_39159 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.file_util')

if (type(import_39159) is not StypyTypeError):

    if (import_39159 != 'pyd_module'):
        __import__(import_39159)
        sys_modules_39160 = sys.modules[import_39159]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.file_util', sys_modules_39160.module_type_store, module_type_store, ['move_file', 'write_file', 'copy_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_39160, sys_modules_39160.module_type_store, module_type_store)
    else:
        from distutils.file_util import move_file, write_file, copy_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.file_util', None, module_type_store, ['move_file', 'write_file', 'copy_file'], [move_file, write_file, copy_file])

else:
    # Assigning a type to the variable 'distutils.file_util' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.file_util', import_39159)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils import log' statement (line 7)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.tests import support' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_39161 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests')

if (type(import_39161) is not StypyTypeError):

    if (import_39161 != 'pyd_module'):
        __import__(import_39161)
        sys_modules_39162 = sys.modules[import_39161]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', sys_modules_39162.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_39162, sys_modules_39162.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', import_39161)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from test.test_support import run_unittest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_39163 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support')

if (type(import_39163) is not StypyTypeError):

    if (import_39163 != 'pyd_module'):
        __import__(import_39163)
        sys_modules_39164 = sys.modules[import_39163]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', sys_modules_39164.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_39164, sys_modules_39164.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', import_39163)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Assigning a Call to a Name (line 12):

# Call to skipUnless(...): (line 12)
# Processing the call arguments (line 12)

# Call to hasattr(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'os' (line 12)
os_39168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 47), 'os', False)
str_39169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 51), 'str', 'link')
# Processing the call keyword arguments (line 12)
kwargs_39170 = {}
# Getting the type of 'hasattr' (line 12)
hasattr_39167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 39), 'hasattr', False)
# Calling hasattr(args, kwargs) (line 12)
hasattr_call_result_39171 = invoke(stypy.reporting.localization.Localization(__file__, 12, 39), hasattr_39167, *[os_39168, str_39169], **kwargs_39170)

str_39172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 39), 'str', 'test requires os.link()')
# Processing the call keyword arguments (line 12)
kwargs_39173 = {}
# Getting the type of 'unittest' (line 12)
unittest_39165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'unittest', False)
# Obtaining the member 'skipUnless' of a type (line 12)
skipUnless_39166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 19), unittest_39165, 'skipUnless')
# Calling skipUnless(args, kwargs) (line 12)
skipUnless_call_result_39174 = invoke(stypy.reporting.localization.Localization(__file__, 12, 19), skipUnless_39166, *[hasattr_call_result_39171, str_39172], **kwargs_39173)

# Assigning a type to the variable 'requires_os_link' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'requires_os_link', skipUnless_call_result_39174)
# Declaration of the 'FileUtilTestCase' class
# Getting the type of 'support' (line 16)
support_39175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'support')
# Obtaining the member 'TempdirManager' of a type (line 16)
TempdirManager_39176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 23), support_39175, 'TempdirManager')
# Getting the type of 'unittest' (line 16)
unittest_39177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 47), 'unittest')
# Obtaining the member 'TestCase' of a type (line 16)
TestCase_39178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 47), unittest_39177, 'TestCase')

class FileUtilTestCase(TempdirManager_39176, TestCase_39178, ):

    @norecursion
    def _log(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_log'
        module_type_store = module_type_store.open_function_context('_log', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileUtilTestCase._log.__dict__.__setitem__('stypy_localization', localization)
        FileUtilTestCase._log.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileUtilTestCase._log.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileUtilTestCase._log.__dict__.__setitem__('stypy_function_name', 'FileUtilTestCase._log')
        FileUtilTestCase._log.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        FileUtilTestCase._log.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FileUtilTestCase._log.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileUtilTestCase._log.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileUtilTestCase._log.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileUtilTestCase._log.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileUtilTestCase._log.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileUtilTestCase._log', ['msg'], 'args', None, defaults, varargs, kwargs)

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

        
        
        
        # Call to len(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'args' (line 19)
        args_39180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'args', False)
        # Processing the call keyword arguments (line 19)
        kwargs_39181 = {}
        # Getting the type of 'len' (line 19)
        len_39179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'len', False)
        # Calling len(args, kwargs) (line 19)
        len_call_result_39182 = invoke(stypy.reporting.localization.Localization(__file__, 19, 11), len_39179, *[args_39180], **kwargs_39181)
        
        int_39183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'int')
        # Applying the binary operator '>' (line 19)
        result_gt_39184 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 11), '>', len_call_result_39182, int_39183)
        
        # Testing the type of an if condition (line 19)
        if_condition_39185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 8), result_gt_39184)
        # Assigning a type to the variable 'if_condition_39185' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'if_condition_39185', if_condition_39185)
        # SSA begins for if statement (line 19)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'msg' (line 20)
        msg_39189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 30), 'msg', False)
        # Getting the type of 'args' (line 20)
        args_39190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 36), 'args', False)
        # Applying the binary operator '%' (line 20)
        result_mod_39191 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 30), '%', msg_39189, args_39190)
        
        # Processing the call keyword arguments (line 20)
        kwargs_39192 = {}
        # Getting the type of 'self' (line 20)
        self_39186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'self', False)
        # Obtaining the member '_logs' of a type (line 20)
        _logs_39187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), self_39186, '_logs')
        # Obtaining the member 'append' of a type (line 20)
        append_39188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), _logs_39187, 'append')
        # Calling append(args, kwargs) (line 20)
        append_call_result_39193 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), append_39188, *[result_mod_39191], **kwargs_39192)
        
        # SSA branch for the else part of an if statement (line 19)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'msg' (line 22)
        msg_39197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 30), 'msg', False)
        # Processing the call keyword arguments (line 22)
        kwargs_39198 = {}
        # Getting the type of 'self' (line 22)
        self_39194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'self', False)
        # Obtaining the member '_logs' of a type (line 22)
        _logs_39195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 12), self_39194, '_logs')
        # Obtaining the member 'append' of a type (line 22)
        append_39196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 12), _logs_39195, 'append')
        # Calling append(args, kwargs) (line 22)
        append_call_result_39199 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), append_39196, *[msg_39197], **kwargs_39198)
        
        # SSA join for if statement (line 19)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_log(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_log' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_39200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39200)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_log'
        return stypy_return_type_39200


    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileUtilTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        FileUtilTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileUtilTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileUtilTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'FileUtilTestCase.setUp')
        FileUtilTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        FileUtilTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileUtilTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileUtilTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileUtilTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileUtilTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileUtilTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileUtilTestCase.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to setUp(...): (line 25)
        # Processing the call keyword arguments (line 25)
        kwargs_39207 = {}
        
        # Call to super(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'FileUtilTestCase' (line 25)
        FileUtilTestCase_39202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 14), 'FileUtilTestCase', False)
        # Getting the type of 'self' (line 25)
        self_39203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 32), 'self', False)
        # Processing the call keyword arguments (line 25)
        kwargs_39204 = {}
        # Getting the type of 'super' (line 25)
        super_39201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'super', False)
        # Calling super(args, kwargs) (line 25)
        super_call_result_39205 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), super_39201, *[FileUtilTestCase_39202, self_39203], **kwargs_39204)
        
        # Obtaining the member 'setUp' of a type (line 25)
        setUp_39206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), super_call_result_39205, 'setUp')
        # Calling setUp(args, kwargs) (line 25)
        setUp_call_result_39208 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), setUp_39206, *[], **kwargs_39207)
        
        
        # Assigning a List to a Attribute (line 26):
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_39209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        
        # Getting the type of 'self' (line 26)
        self_39210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member '_logs' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_39210, '_logs', list_39209)
        
        # Assigning a Attribute to a Attribute (line 27):
        # Getting the type of 'log' (line 27)
        log_39211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 23), 'log')
        # Obtaining the member 'info' of a type (line 27)
        info_39212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 23), log_39211, 'info')
        # Getting the type of 'self' (line 27)
        self_39213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member 'old_log' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_39213, 'old_log', info_39212)
        
        # Assigning a Attribute to a Attribute (line 28):
        # Getting the type of 'self' (line 28)
        self_39214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'self')
        # Obtaining the member '_log' of a type (line 28)
        _log_39215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 19), self_39214, '_log')
        # Getting the type of 'log' (line 28)
        log_39216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'log')
        # Setting the type of the member 'info' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), log_39216, 'info', _log_39215)
        
        # Assigning a Call to a Name (line 29):
        
        # Call to mkdtemp(...): (line 29)
        # Processing the call keyword arguments (line 29)
        kwargs_39219 = {}
        # Getting the type of 'self' (line 29)
        self_39217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 29)
        mkdtemp_39218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 18), self_39217, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 29)
        mkdtemp_call_result_39220 = invoke(stypy.reporting.localization.Localization(__file__, 29, 18), mkdtemp_39218, *[], **kwargs_39219)
        
        # Assigning a type to the variable 'tmp_dir' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'tmp_dir', mkdtemp_call_result_39220)
        
        # Assigning a Call to a Attribute (line 30):
        
        # Call to join(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'tmp_dir' (line 30)
        tmp_dir_39224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 35), 'tmp_dir', False)
        str_39225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 44), 'str', 'f1')
        # Processing the call keyword arguments (line 30)
        kwargs_39226 = {}
        # Getting the type of 'os' (line 30)
        os_39221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 30)
        path_39222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 22), os_39221, 'path')
        # Obtaining the member 'join' of a type (line 30)
        join_39223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 22), path_39222, 'join')
        # Calling join(args, kwargs) (line 30)
        join_call_result_39227 = invoke(stypy.reporting.localization.Localization(__file__, 30, 22), join_39223, *[tmp_dir_39224, str_39225], **kwargs_39226)
        
        # Getting the type of 'self' (line 30)
        self_39228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'source' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_39228, 'source', join_call_result_39227)
        
        # Assigning a Call to a Attribute (line 31):
        
        # Call to join(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'tmp_dir' (line 31)
        tmp_dir_39232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 35), 'tmp_dir', False)
        str_39233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 44), 'str', 'f2')
        # Processing the call keyword arguments (line 31)
        kwargs_39234 = {}
        # Getting the type of 'os' (line 31)
        os_39229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 31)
        path_39230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 22), os_39229, 'path')
        # Obtaining the member 'join' of a type (line 31)
        join_39231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 22), path_39230, 'join')
        # Calling join(args, kwargs) (line 31)
        join_call_result_39235 = invoke(stypy.reporting.localization.Localization(__file__, 31, 22), join_39231, *[tmp_dir_39232, str_39233], **kwargs_39234)
        
        # Getting the type of 'self' (line 31)
        self_39236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'target' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_39236, 'target', join_call_result_39235)
        
        # Assigning a Call to a Attribute (line 32):
        
        # Call to join(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'tmp_dir' (line 32)
        tmp_dir_39240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 39), 'tmp_dir', False)
        str_39241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 48), 'str', 'd1')
        # Processing the call keyword arguments (line 32)
        kwargs_39242 = {}
        # Getting the type of 'os' (line 32)
        os_39237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 32)
        path_39238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 26), os_39237, 'path')
        # Obtaining the member 'join' of a type (line 32)
        join_39239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 26), path_39238, 'join')
        # Calling join(args, kwargs) (line 32)
        join_call_result_39243 = invoke(stypy.reporting.localization.Localization(__file__, 32, 26), join_39239, *[tmp_dir_39240, str_39241], **kwargs_39242)
        
        # Getting the type of 'self' (line 32)
        self_39244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'target_dir' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_39244, 'target_dir', join_call_result_39243)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_39245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39245)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_39245


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileUtilTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        FileUtilTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileUtilTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileUtilTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'FileUtilTestCase.tearDown')
        FileUtilTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        FileUtilTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileUtilTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileUtilTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileUtilTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileUtilTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileUtilTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileUtilTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Attribute (line 35):
        # Getting the type of 'self' (line 35)
        self_39246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'self')
        # Obtaining the member 'old_log' of a type (line 35)
        old_log_39247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 19), self_39246, 'old_log')
        # Getting the type of 'log' (line 35)
        log_39248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'log')
        # Setting the type of the member 'info' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), log_39248, 'info', old_log_39247)
        
        # Call to tearDown(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_39255 = {}
        
        # Call to super(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'FileUtilTestCase' (line 36)
        FileUtilTestCase_39250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 14), 'FileUtilTestCase', False)
        # Getting the type of 'self' (line 36)
        self_39251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 32), 'self', False)
        # Processing the call keyword arguments (line 36)
        kwargs_39252 = {}
        # Getting the type of 'super' (line 36)
        super_39249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'super', False)
        # Calling super(args, kwargs) (line 36)
        super_call_result_39253 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), super_39249, *[FileUtilTestCase_39250, self_39251], **kwargs_39252)
        
        # Obtaining the member 'tearDown' of a type (line 36)
        tearDown_39254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), super_call_result_39253, 'tearDown')
        # Calling tearDown(args, kwargs) (line 36)
        tearDown_call_result_39256 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), tearDown_39254, *[], **kwargs_39255)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_39257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39257)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_39257


    @norecursion
    def test_move_file_verbosity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_move_file_verbosity'
        module_type_store = module_type_store.open_function_context('test_move_file_verbosity', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileUtilTestCase.test_move_file_verbosity.__dict__.__setitem__('stypy_localization', localization)
        FileUtilTestCase.test_move_file_verbosity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileUtilTestCase.test_move_file_verbosity.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileUtilTestCase.test_move_file_verbosity.__dict__.__setitem__('stypy_function_name', 'FileUtilTestCase.test_move_file_verbosity')
        FileUtilTestCase.test_move_file_verbosity.__dict__.__setitem__('stypy_param_names_list', [])
        FileUtilTestCase.test_move_file_verbosity.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileUtilTestCase.test_move_file_verbosity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileUtilTestCase.test_move_file_verbosity.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileUtilTestCase.test_move_file_verbosity.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileUtilTestCase.test_move_file_verbosity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileUtilTestCase.test_move_file_verbosity.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileUtilTestCase.test_move_file_verbosity', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_move_file_verbosity', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_move_file_verbosity(...)' code ##################

        
        # Assigning a Call to a Name (line 39):
        
        # Call to open(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'self' (line 39)
        self_39259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'self', False)
        # Obtaining the member 'source' of a type (line 39)
        source_39260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 17), self_39259, 'source')
        str_39261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'str', 'w')
        # Processing the call keyword arguments (line 39)
        kwargs_39262 = {}
        # Getting the type of 'open' (line 39)
        open_39258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'open', False)
        # Calling open(args, kwargs) (line 39)
        open_call_result_39263 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), open_39258, *[source_39260, str_39261], **kwargs_39262)
        
        # Assigning a type to the variable 'f' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'f', open_call_result_39263)
        
        # Try-finally block (line 40)
        
        # Call to write(...): (line 41)
        # Processing the call arguments (line 41)
        str_39266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'str', 'some content')
        # Processing the call keyword arguments (line 41)
        kwargs_39267 = {}
        # Getting the type of 'f' (line 41)
        f_39264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 41)
        write_39265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), f_39264, 'write')
        # Calling write(args, kwargs) (line 41)
        write_call_result_39268 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), write_39265, *[str_39266], **kwargs_39267)
        
        
        # finally branch of the try-finally block (line 40)
        
        # Call to close(...): (line 43)
        # Processing the call keyword arguments (line 43)
        kwargs_39271 = {}
        # Getting the type of 'f' (line 43)
        f_39269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 43)
        close_39270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), f_39269, 'close')
        # Calling close(args, kwargs) (line 43)
        close_call_result_39272 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), close_39270, *[], **kwargs_39271)
        
        
        
        # Call to move_file(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'self' (line 45)
        self_39274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'self', False)
        # Obtaining the member 'source' of a type (line 45)
        source_39275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 18), self_39274, 'source')
        # Getting the type of 'self' (line 45)
        self_39276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'self', False)
        # Obtaining the member 'target' of a type (line 45)
        target_39277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 31), self_39276, 'target')
        # Processing the call keyword arguments (line 45)
        int_39278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 52), 'int')
        keyword_39279 = int_39278
        kwargs_39280 = {'verbose': keyword_39279}
        # Getting the type of 'move_file' (line 45)
        move_file_39273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'move_file', False)
        # Calling move_file(args, kwargs) (line 45)
        move_file_call_result_39281 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), move_file_39273, *[source_39275, target_39277], **kwargs_39280)
        
        
        # Assigning a List to a Name (line 46):
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_39282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        
        # Assigning a type to the variable 'wanted' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'wanted', list_39282)
        
        # Call to assertEqual(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'self' (line 47)
        self_39285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'self', False)
        # Obtaining the member '_logs' of a type (line 47)
        _logs_39286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 25), self_39285, '_logs')
        # Getting the type of 'wanted' (line 47)
        wanted_39287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'wanted', False)
        # Processing the call keyword arguments (line 47)
        kwargs_39288 = {}
        # Getting the type of 'self' (line 47)
        self_39283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 47)
        assertEqual_39284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_39283, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 47)
        assertEqual_call_result_39289 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assertEqual_39284, *[_logs_39286, wanted_39287], **kwargs_39288)
        
        
        # Call to move_file(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'self' (line 50)
        self_39291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'self', False)
        # Obtaining the member 'target' of a type (line 50)
        target_39292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 18), self_39291, 'target')
        # Getting the type of 'self' (line 50)
        self_39293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 31), 'self', False)
        # Obtaining the member 'source' of a type (line 50)
        source_39294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 31), self_39293, 'source')
        # Processing the call keyword arguments (line 50)
        int_39295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 52), 'int')
        keyword_39296 = int_39295
        kwargs_39297 = {'verbose': keyword_39296}
        # Getting the type of 'move_file' (line 50)
        move_file_39290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'move_file', False)
        # Calling move_file(args, kwargs) (line 50)
        move_file_call_result_39298 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), move_file_39290, *[target_39292, source_39294], **kwargs_39297)
        
        
        # Call to move_file(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'self' (line 52)
        self_39300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'self', False)
        # Obtaining the member 'source' of a type (line 52)
        source_39301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 18), self_39300, 'source')
        # Getting the type of 'self' (line 52)
        self_39302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 31), 'self', False)
        # Obtaining the member 'target' of a type (line 52)
        target_39303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 31), self_39302, 'target')
        # Processing the call keyword arguments (line 52)
        int_39304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 52), 'int')
        keyword_39305 = int_39304
        kwargs_39306 = {'verbose': keyword_39305}
        # Getting the type of 'move_file' (line 52)
        move_file_39299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'move_file', False)
        # Calling move_file(args, kwargs) (line 52)
        move_file_call_result_39307 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), move_file_39299, *[source_39301, target_39303], **kwargs_39306)
        
        
        # Assigning a List to a Name (line 53):
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_39308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        str_39309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 18), 'str', 'moving %s -> %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_39310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        # Getting the type of 'self' (line 53)
        self_39311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 39), 'self')
        # Obtaining the member 'source' of a type (line 53)
        source_39312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 39), self_39311, 'source')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 39), tuple_39310, source_39312)
        # Adding element type (line 53)
        # Getting the type of 'self' (line 53)
        self_39313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 52), 'self')
        # Obtaining the member 'target' of a type (line 53)
        target_39314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 52), self_39313, 'target')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 39), tuple_39310, target_39314)
        
        # Applying the binary operator '%' (line 53)
        result_mod_39315 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 18), '%', str_39309, tuple_39310)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 17), list_39308, result_mod_39315)
        
        # Assigning a type to the variable 'wanted' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'wanted', list_39308)
        
        # Call to assertEqual(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_39318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'self', False)
        # Obtaining the member '_logs' of a type (line 54)
        _logs_39319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 25), self_39318, '_logs')
        # Getting the type of 'wanted' (line 54)
        wanted_39320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'wanted', False)
        # Processing the call keyword arguments (line 54)
        kwargs_39321 = {}
        # Getting the type of 'self' (line 54)
        self_39316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 54)
        assertEqual_39317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_39316, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 54)
        assertEqual_call_result_39322 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assertEqual_39317, *[_logs_39319, wanted_39320], **kwargs_39321)
        
        
        # Call to move_file(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'self' (line 57)
        self_39324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'self', False)
        # Obtaining the member 'target' of a type (line 57)
        target_39325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 18), self_39324, 'target')
        # Getting the type of 'self' (line 57)
        self_39326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 31), 'self', False)
        # Obtaining the member 'source' of a type (line 57)
        source_39327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 31), self_39326, 'source')
        # Processing the call keyword arguments (line 57)
        int_39328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 52), 'int')
        keyword_39329 = int_39328
        kwargs_39330 = {'verbose': keyword_39329}
        # Getting the type of 'move_file' (line 57)
        move_file_39323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'move_file', False)
        # Calling move_file(args, kwargs) (line 57)
        move_file_call_result_39331 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), move_file_39323, *[target_39325, source_39327], **kwargs_39330)
        
        
        # Assigning a List to a Attribute (line 59):
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_39332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        
        # Getting the type of 'self' (line 59)
        self_39333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member '_logs' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_39333, '_logs', list_39332)
        
        # Call to mkdir(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'self' (line 61)
        self_39336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'self', False)
        # Obtaining the member 'target_dir' of a type (line 61)
        target_dir_39337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 17), self_39336, 'target_dir')
        # Processing the call keyword arguments (line 61)
        kwargs_39338 = {}
        # Getting the type of 'os' (line 61)
        os_39334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 61)
        mkdir_39335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), os_39334, 'mkdir')
        # Calling mkdir(args, kwargs) (line 61)
        mkdir_call_result_39339 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), mkdir_39335, *[target_dir_39337], **kwargs_39338)
        
        
        # Call to move_file(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_39341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'self', False)
        # Obtaining the member 'source' of a type (line 62)
        source_39342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 18), self_39341, 'source')
        # Getting the type of 'self' (line 62)
        self_39343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'self', False)
        # Obtaining the member 'target_dir' of a type (line 62)
        target_dir_39344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 31), self_39343, 'target_dir')
        # Processing the call keyword arguments (line 62)
        int_39345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 56), 'int')
        keyword_39346 = int_39345
        kwargs_39347 = {'verbose': keyword_39346}
        # Getting the type of 'move_file' (line 62)
        move_file_39340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'move_file', False)
        # Calling move_file(args, kwargs) (line 62)
        move_file_call_result_39348 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), move_file_39340, *[source_39342, target_dir_39344], **kwargs_39347)
        
        
        # Assigning a List to a Name (line 63):
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_39349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        str_39350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'str', 'moving %s -> %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_39351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        # Getting the type of 'self' (line 63)
        self_39352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 39), 'self')
        # Obtaining the member 'source' of a type (line 63)
        source_39353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 39), self_39352, 'source')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 39), tuple_39351, source_39353)
        # Adding element type (line 63)
        # Getting the type of 'self' (line 63)
        self_39354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 52), 'self')
        # Obtaining the member 'target_dir' of a type (line 63)
        target_dir_39355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 52), self_39354, 'target_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 39), tuple_39351, target_dir_39355)
        
        # Applying the binary operator '%' (line 63)
        result_mod_39356 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 18), '%', str_39350, tuple_39351)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 17), list_39349, result_mod_39356)
        
        # Assigning a type to the variable 'wanted' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'wanted', list_39349)
        
        # Call to assertEqual(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'self' (line 64)
        self_39359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'self', False)
        # Obtaining the member '_logs' of a type (line 64)
        _logs_39360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 25), self_39359, '_logs')
        # Getting the type of 'wanted' (line 64)
        wanted_39361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 37), 'wanted', False)
        # Processing the call keyword arguments (line 64)
        kwargs_39362 = {}
        # Getting the type of 'self' (line 64)
        self_39357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 64)
        assertEqual_39358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_39357, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 64)
        assertEqual_call_result_39363 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assertEqual_39358, *[_logs_39360, wanted_39361], **kwargs_39362)
        
        
        # ################# End of 'test_move_file_verbosity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_move_file_verbosity' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_39364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39364)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_move_file_verbosity'
        return stypy_return_type_39364


    @norecursion
    def test_write_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_write_file'
        module_type_store = module_type_store.open_function_context('test_write_file', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileUtilTestCase.test_write_file.__dict__.__setitem__('stypy_localization', localization)
        FileUtilTestCase.test_write_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileUtilTestCase.test_write_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileUtilTestCase.test_write_file.__dict__.__setitem__('stypy_function_name', 'FileUtilTestCase.test_write_file')
        FileUtilTestCase.test_write_file.__dict__.__setitem__('stypy_param_names_list', [])
        FileUtilTestCase.test_write_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileUtilTestCase.test_write_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileUtilTestCase.test_write_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileUtilTestCase.test_write_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileUtilTestCase.test_write_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileUtilTestCase.test_write_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileUtilTestCase.test_write_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_write_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_write_file(...)' code ##################

        
        # Assigning a List to a Name (line 67):
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_39365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        str_39366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 17), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 16), list_39365, str_39366)
        # Adding element type (line 67)
        str_39367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 22), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 16), list_39365, str_39367)
        # Adding element type (line 67)
        str_39368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'str', 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 16), list_39365, str_39368)
        
        # Assigning a type to the variable 'lines' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'lines', list_39365)
        
        # Assigning a Call to a Name (line 68):
        
        # Call to mkdtemp(...): (line 68)
        # Processing the call keyword arguments (line 68)
        kwargs_39371 = {}
        # Getting the type of 'self' (line 68)
        self_39369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 68)
        mkdtemp_39370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 14), self_39369, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 68)
        mkdtemp_call_result_39372 = invoke(stypy.reporting.localization.Localization(__file__, 68, 14), mkdtemp_39370, *[], **kwargs_39371)
        
        # Assigning a type to the variable 'dir' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'dir', mkdtemp_call_result_39372)
        
        # Assigning a Call to a Name (line 69):
        
        # Call to join(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'dir' (line 69)
        dir_39376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'dir', False)
        str_39377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 32), 'str', 'foo')
        # Processing the call keyword arguments (line 69)
        kwargs_39378 = {}
        # Getting the type of 'os' (line 69)
        os_39373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'os', False)
        # Obtaining the member 'path' of a type (line 69)
        path_39374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 14), os_39373, 'path')
        # Obtaining the member 'join' of a type (line 69)
        join_39375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 14), path_39374, 'join')
        # Calling join(args, kwargs) (line 69)
        join_call_result_39379 = invoke(stypy.reporting.localization.Localization(__file__, 69, 14), join_39375, *[dir_39376, str_39377], **kwargs_39378)
        
        # Assigning a type to the variable 'foo' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'foo', join_call_result_39379)
        
        # Call to write_file(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'foo' (line 70)
        foo_39381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'foo', False)
        # Getting the type of 'lines' (line 70)
        lines_39382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'lines', False)
        # Processing the call keyword arguments (line 70)
        kwargs_39383 = {}
        # Getting the type of 'write_file' (line 70)
        write_file_39380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'write_file', False)
        # Calling write_file(args, kwargs) (line 70)
        write_file_call_result_39384 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), write_file_39380, *[foo_39381, lines_39382], **kwargs_39383)
        
        
        # Assigning a ListComp to a Name (line 71):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to readlines(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_39394 = {}
        
        # Call to open(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'foo' (line 71)
        foo_39390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 49), 'foo', False)
        # Processing the call keyword arguments (line 71)
        kwargs_39391 = {}
        # Getting the type of 'open' (line 71)
        open_39389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 44), 'open', False)
        # Calling open(args, kwargs) (line 71)
        open_call_result_39392 = invoke(stypy.reporting.localization.Localization(__file__, 71, 44), open_39389, *[foo_39390], **kwargs_39391)
        
        # Obtaining the member 'readlines' of a type (line 71)
        readlines_39393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 44), open_call_result_39392, 'readlines')
        # Calling readlines(args, kwargs) (line 71)
        readlines_call_result_39395 = invoke(stypy.reporting.localization.Localization(__file__, 71, 44), readlines_39393, *[], **kwargs_39394)
        
        comprehension_39396 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 19), readlines_call_result_39395)
        # Assigning a type to the variable 'line' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'line', comprehension_39396)
        
        # Call to strip(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_39387 = {}
        # Getting the type of 'line' (line 71)
        line_39385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'line', False)
        # Obtaining the member 'strip' of a type (line 71)
        strip_39386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), line_39385, 'strip')
        # Calling strip(args, kwargs) (line 71)
        strip_call_result_39388 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), strip_39386, *[], **kwargs_39387)
        
        list_39397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 19), list_39397, strip_call_result_39388)
        # Assigning a type to the variable 'content' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'content', list_39397)
        
        # Call to assertEqual(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'content' (line 72)
        content_39400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'content', False)
        # Getting the type of 'lines' (line 72)
        lines_39401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 34), 'lines', False)
        # Processing the call keyword arguments (line 72)
        kwargs_39402 = {}
        # Getting the type of 'self' (line 72)
        self_39398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 72)
        assertEqual_39399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_39398, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 72)
        assertEqual_call_result_39403 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), assertEqual_39399, *[content_39400, lines_39401], **kwargs_39402)
        
        
        # ################# End of 'test_write_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_write_file' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_39404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39404)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_write_file'
        return stypy_return_type_39404


    @norecursion
    def test_copy_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_copy_file'
        module_type_store = module_type_store.open_function_context('test_copy_file', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileUtilTestCase.test_copy_file.__dict__.__setitem__('stypy_localization', localization)
        FileUtilTestCase.test_copy_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileUtilTestCase.test_copy_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileUtilTestCase.test_copy_file.__dict__.__setitem__('stypy_function_name', 'FileUtilTestCase.test_copy_file')
        FileUtilTestCase.test_copy_file.__dict__.__setitem__('stypy_param_names_list', [])
        FileUtilTestCase.test_copy_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileUtilTestCase.test_copy_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileUtilTestCase.test_copy_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileUtilTestCase.test_copy_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileUtilTestCase.test_copy_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileUtilTestCase.test_copy_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileUtilTestCase.test_copy_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_copy_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_copy_file(...)' code ##################

        
        # Assigning a Call to a Name (line 75):
        
        # Call to mkdtemp(...): (line 75)
        # Processing the call keyword arguments (line 75)
        kwargs_39407 = {}
        # Getting the type of 'self' (line 75)
        self_39405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 75)
        mkdtemp_39406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 18), self_39405, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 75)
        mkdtemp_call_result_39408 = invoke(stypy.reporting.localization.Localization(__file__, 75, 18), mkdtemp_39406, *[], **kwargs_39407)
        
        # Assigning a type to the variable 'src_dir' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'src_dir', mkdtemp_call_result_39408)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to join(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'src_dir' (line 76)
        src_dir_39412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'src_dir', False)
        str_39413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 36), 'str', 'foo')
        # Processing the call keyword arguments (line 76)
        kwargs_39414 = {}
        # Getting the type of 'os' (line 76)
        os_39409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'os', False)
        # Obtaining the member 'path' of a type (line 76)
        path_39410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 14), os_39409, 'path')
        # Obtaining the member 'join' of a type (line 76)
        join_39411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 14), path_39410, 'join')
        # Calling join(args, kwargs) (line 76)
        join_call_result_39415 = invoke(stypy.reporting.localization.Localization(__file__, 76, 14), join_39411, *[src_dir_39412, str_39413], **kwargs_39414)
        
        # Assigning a type to the variable 'foo' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'foo', join_call_result_39415)
        
        # Call to write_file(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'foo' (line 77)
        foo_39417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'foo', False)
        str_39418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 24), 'str', 'content')
        # Processing the call keyword arguments (line 77)
        kwargs_39419 = {}
        # Getting the type of 'write_file' (line 77)
        write_file_39416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'write_file', False)
        # Calling write_file(args, kwargs) (line 77)
        write_file_call_result_39420 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), write_file_39416, *[foo_39417, str_39418], **kwargs_39419)
        
        
        # Assigning a Call to a Name (line 78):
        
        # Call to mkdtemp(...): (line 78)
        # Processing the call keyword arguments (line 78)
        kwargs_39423 = {}
        # Getting the type of 'self' (line 78)
        self_39421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 78)
        mkdtemp_39422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 18), self_39421, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 78)
        mkdtemp_call_result_39424 = invoke(stypy.reporting.localization.Localization(__file__, 78, 18), mkdtemp_39422, *[], **kwargs_39423)
        
        # Assigning a type to the variable 'dst_dir' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'dst_dir', mkdtemp_call_result_39424)
        
        # Call to copy_file(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'foo' (line 79)
        foo_39426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'foo', False)
        # Getting the type of 'dst_dir' (line 79)
        dst_dir_39427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'dst_dir', False)
        # Processing the call keyword arguments (line 79)
        kwargs_39428 = {}
        # Getting the type of 'copy_file' (line 79)
        copy_file_39425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'copy_file', False)
        # Calling copy_file(args, kwargs) (line 79)
        copy_file_call_result_39429 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), copy_file_39425, *[foo_39426, dst_dir_39427], **kwargs_39428)
        
        
        # Call to assertTrue(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Call to exists(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Call to join(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'dst_dir' (line 80)
        dst_dir_39438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 52), 'dst_dir', False)
        str_39439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 61), 'str', 'foo')
        # Processing the call keyword arguments (line 80)
        kwargs_39440 = {}
        # Getting the type of 'os' (line 80)
        os_39435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 39), 'os', False)
        # Obtaining the member 'path' of a type (line 80)
        path_39436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 39), os_39435, 'path')
        # Obtaining the member 'join' of a type (line 80)
        join_39437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 39), path_39436, 'join')
        # Calling join(args, kwargs) (line 80)
        join_call_result_39441 = invoke(stypy.reporting.localization.Localization(__file__, 80, 39), join_39437, *[dst_dir_39438, str_39439], **kwargs_39440)
        
        # Processing the call keyword arguments (line 80)
        kwargs_39442 = {}
        # Getting the type of 'os' (line 80)
        os_39432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 80)
        path_39433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 24), os_39432, 'path')
        # Obtaining the member 'exists' of a type (line 80)
        exists_39434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 24), path_39433, 'exists')
        # Calling exists(args, kwargs) (line 80)
        exists_call_result_39443 = invoke(stypy.reporting.localization.Localization(__file__, 80, 24), exists_39434, *[join_call_result_39441], **kwargs_39442)
        
        # Processing the call keyword arguments (line 80)
        kwargs_39444 = {}
        # Getting the type of 'self' (line 80)
        self_39430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 80)
        assertTrue_39431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_39430, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 80)
        assertTrue_call_result_39445 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assertTrue_39431, *[exists_call_result_39443], **kwargs_39444)
        
        
        # ################# End of 'test_copy_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_copy_file' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_39446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39446)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_copy_file'
        return stypy_return_type_39446


    @norecursion
    def test_copy_file_hard_link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_copy_file_hard_link'
        module_type_store = module_type_store.open_function_context('test_copy_file_hard_link', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileUtilTestCase.test_copy_file_hard_link.__dict__.__setitem__('stypy_localization', localization)
        FileUtilTestCase.test_copy_file_hard_link.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileUtilTestCase.test_copy_file_hard_link.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileUtilTestCase.test_copy_file_hard_link.__dict__.__setitem__('stypy_function_name', 'FileUtilTestCase.test_copy_file_hard_link')
        FileUtilTestCase.test_copy_file_hard_link.__dict__.__setitem__('stypy_param_names_list', [])
        FileUtilTestCase.test_copy_file_hard_link.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileUtilTestCase.test_copy_file_hard_link.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileUtilTestCase.test_copy_file_hard_link.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileUtilTestCase.test_copy_file_hard_link.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileUtilTestCase.test_copy_file_hard_link.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileUtilTestCase.test_copy_file_hard_link.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileUtilTestCase.test_copy_file_hard_link', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_copy_file_hard_link', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_copy_file_hard_link(...)' code ##################

        
        # Call to open(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'self' (line 84)
        self_39448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'self', False)
        # Obtaining the member 'source' of a type (line 84)
        source_39449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 18), self_39448, 'source')
        str_39450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'str', 'w')
        # Processing the call keyword arguments (line 84)
        kwargs_39451 = {}
        # Getting the type of 'open' (line 84)
        open_39447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'open', False)
        # Calling open(args, kwargs) (line 84)
        open_call_result_39452 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), open_39447, *[source_39449, str_39450], **kwargs_39451)
        
        with_39453 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 84, 13), open_call_result_39452, 'with parameter', '__enter__', '__exit__')

        if with_39453:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 84)
            enter___39454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 13), open_call_result_39452, '__enter__')
            with_enter_39455 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), enter___39454)
            # Assigning a type to the variable 'f' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'f', with_enter_39455)
            
            # Call to write(...): (line 85)
            # Processing the call arguments (line 85)
            str_39458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 20), 'str', 'some content')
            # Processing the call keyword arguments (line 85)
            kwargs_39459 = {}
            # Getting the type of 'f' (line 85)
            f_39456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'f', False)
            # Obtaining the member 'write' of a type (line 85)
            write_39457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), f_39456, 'write')
            # Calling write(args, kwargs) (line 85)
            write_call_result_39460 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), write_39457, *[str_39458], **kwargs_39459)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 84)
            exit___39461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 13), open_call_result_39452, '__exit__')
            with_exit_39462 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), exit___39461, None, None, None)

        
        # Assigning a Call to a Name (line 86):
        
        # Call to stat(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'self' (line 86)
        self_39465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'self', False)
        # Obtaining the member 'source' of a type (line 86)
        source_39466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 21), self_39465, 'source')
        # Processing the call keyword arguments (line 86)
        kwargs_39467 = {}
        # Getting the type of 'os' (line 86)
        os_39463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'os', False)
        # Obtaining the member 'stat' of a type (line 86)
        stat_39464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 13), os_39463, 'stat')
        # Calling stat(args, kwargs) (line 86)
        stat_call_result_39468 = invoke(stypy.reporting.localization.Localization(__file__, 86, 13), stat_39464, *[source_39466], **kwargs_39467)
        
        # Assigning a type to the variable 'st' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'st', stat_call_result_39468)
        
        # Call to copy_file(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'self' (line 87)
        self_39470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), 'self', False)
        # Obtaining the member 'source' of a type (line 87)
        source_39471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 18), self_39470, 'source')
        # Getting the type of 'self' (line 87)
        self_39472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 31), 'self', False)
        # Obtaining the member 'target' of a type (line 87)
        target_39473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 31), self_39472, 'target')
        # Processing the call keyword arguments (line 87)
        str_39474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 49), 'str', 'hard')
        keyword_39475 = str_39474
        kwargs_39476 = {'link': keyword_39475}
        # Getting the type of 'copy_file' (line 87)
        copy_file_39469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'copy_file', False)
        # Calling copy_file(args, kwargs) (line 87)
        copy_file_call_result_39477 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), copy_file_39469, *[source_39471, target_39473], **kwargs_39476)
        
        
        # Assigning a Call to a Name (line 88):
        
        # Call to stat(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'self' (line 88)
        self_39480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'self', False)
        # Obtaining the member 'source' of a type (line 88)
        source_39481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 22), self_39480, 'source')
        # Processing the call keyword arguments (line 88)
        kwargs_39482 = {}
        # Getting the type of 'os' (line 88)
        os_39478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'os', False)
        # Obtaining the member 'stat' of a type (line 88)
        stat_39479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 14), os_39478, 'stat')
        # Calling stat(args, kwargs) (line 88)
        stat_call_result_39483 = invoke(stypy.reporting.localization.Localization(__file__, 88, 14), stat_39479, *[source_39481], **kwargs_39482)
        
        # Assigning a type to the variable 'st2' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'st2', stat_call_result_39483)
        
        # Assigning a Call to a Name (line 89):
        
        # Call to stat(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'self' (line 89)
        self_39486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'self', False)
        # Obtaining the member 'target' of a type (line 89)
        target_39487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 22), self_39486, 'target')
        # Processing the call keyword arguments (line 89)
        kwargs_39488 = {}
        # Getting the type of 'os' (line 89)
        os_39484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 14), 'os', False)
        # Obtaining the member 'stat' of a type (line 89)
        stat_39485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 14), os_39484, 'stat')
        # Calling stat(args, kwargs) (line 89)
        stat_call_result_39489 = invoke(stypy.reporting.localization.Localization(__file__, 89, 14), stat_39485, *[target_39487], **kwargs_39488)
        
        # Assigning a type to the variable 'st3' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'st3', stat_call_result_39489)
        
        # Call to assertTrue(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to samestat(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'st' (line 90)
        st_39495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 41), 'st', False)
        # Getting the type of 'st2' (line 90)
        st2_39496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 45), 'st2', False)
        # Processing the call keyword arguments (line 90)
        kwargs_39497 = {}
        # Getting the type of 'os' (line 90)
        os_39492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 90)
        path_39493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 24), os_39492, 'path')
        # Obtaining the member 'samestat' of a type (line 90)
        samestat_39494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 24), path_39493, 'samestat')
        # Calling samestat(args, kwargs) (line 90)
        samestat_call_result_39498 = invoke(stypy.reporting.localization.Localization(__file__, 90, 24), samestat_39494, *[st_39495, st2_39496], **kwargs_39497)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 90)
        tuple_39499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 90)
        # Adding element type (line 90)
        # Getting the type of 'st' (line 90)
        st_39500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 52), 'st', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 52), tuple_39499, st_39500)
        # Adding element type (line 90)
        # Getting the type of 'st2' (line 90)
        st2_39501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 56), 'st2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 52), tuple_39499, st2_39501)
        
        # Processing the call keyword arguments (line 90)
        kwargs_39502 = {}
        # Getting the type of 'self' (line 90)
        self_39490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 90)
        assertTrue_39491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_39490, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 90)
        assertTrue_call_result_39503 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), assertTrue_39491, *[samestat_call_result_39498, tuple_39499], **kwargs_39502)
        
        
        # Call to assertTrue(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to samestat(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'st2' (line 91)
        st2_39509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 41), 'st2', False)
        # Getting the type of 'st3' (line 91)
        st3_39510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 46), 'st3', False)
        # Processing the call keyword arguments (line 91)
        kwargs_39511 = {}
        # Getting the type of 'os' (line 91)
        os_39506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 91)
        path_39507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 24), os_39506, 'path')
        # Obtaining the member 'samestat' of a type (line 91)
        samestat_39508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 24), path_39507, 'samestat')
        # Calling samestat(args, kwargs) (line 91)
        samestat_call_result_39512 = invoke(stypy.reporting.localization.Localization(__file__, 91, 24), samestat_39508, *[st2_39509, st3_39510], **kwargs_39511)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 91)
        tuple_39513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 91)
        # Adding element type (line 91)
        # Getting the type of 'st2' (line 91)
        st2_39514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 53), 'st2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 53), tuple_39513, st2_39514)
        # Adding element type (line 91)
        # Getting the type of 'st3' (line 91)
        st3_39515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 58), 'st3', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 53), tuple_39513, st3_39515)
        
        # Processing the call keyword arguments (line 91)
        kwargs_39516 = {}
        # Getting the type of 'self' (line 91)
        self_39504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 91)
        assertTrue_39505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_39504, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 91)
        assertTrue_call_result_39517 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), assertTrue_39505, *[samestat_call_result_39512, tuple_39513], **kwargs_39516)
        
        
        # Call to open(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'self' (line 92)
        self_39519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 18), 'self', False)
        # Obtaining the member 'source' of a type (line 92)
        source_39520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 18), self_39519, 'source')
        str_39521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 31), 'str', 'r')
        # Processing the call keyword arguments (line 92)
        kwargs_39522 = {}
        # Getting the type of 'open' (line 92)
        open_39518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'open', False)
        # Calling open(args, kwargs) (line 92)
        open_call_result_39523 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), open_39518, *[source_39520, str_39521], **kwargs_39522)
        
        with_39524 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 92, 13), open_call_result_39523, 'with parameter', '__enter__', '__exit__')

        if with_39524:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 92)
            enter___39525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), open_call_result_39523, '__enter__')
            with_enter_39526 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), enter___39525)
            # Assigning a type to the variable 'f' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'f', with_enter_39526)
            
            # Call to assertEqual(...): (line 93)
            # Processing the call arguments (line 93)
            
            # Call to read(...): (line 93)
            # Processing the call keyword arguments (line 93)
            kwargs_39531 = {}
            # Getting the type of 'f' (line 93)
            f_39529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 29), 'f', False)
            # Obtaining the member 'read' of a type (line 93)
            read_39530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 29), f_39529, 'read')
            # Calling read(args, kwargs) (line 93)
            read_call_result_39532 = invoke(stypy.reporting.localization.Localization(__file__, 93, 29), read_39530, *[], **kwargs_39531)
            
            str_39533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 39), 'str', 'some content')
            # Processing the call keyword arguments (line 93)
            kwargs_39534 = {}
            # Getting the type of 'self' (line 93)
            self_39527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'self', False)
            # Obtaining the member 'assertEqual' of a type (line 93)
            assertEqual_39528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), self_39527, 'assertEqual')
            # Calling assertEqual(args, kwargs) (line 93)
            assertEqual_call_result_39535 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), assertEqual_39528, *[read_call_result_39532, str_39533], **kwargs_39534)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 92)
            exit___39536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), open_call_result_39523, '__exit__')
            with_exit_39537 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), exit___39536, None, None, None)

        
        # ################# End of 'test_copy_file_hard_link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_copy_file_hard_link' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_39538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39538)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_copy_file_hard_link'
        return stypy_return_type_39538


    @norecursion
    def test_copy_file_hard_link_failure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_copy_file_hard_link_failure'
        module_type_store = module_type_store.open_function_context('test_copy_file_hard_link_failure', 95, 4, False)
        # Assigning a type to the variable 'self' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileUtilTestCase.test_copy_file_hard_link_failure.__dict__.__setitem__('stypy_localization', localization)
        FileUtilTestCase.test_copy_file_hard_link_failure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileUtilTestCase.test_copy_file_hard_link_failure.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileUtilTestCase.test_copy_file_hard_link_failure.__dict__.__setitem__('stypy_function_name', 'FileUtilTestCase.test_copy_file_hard_link_failure')
        FileUtilTestCase.test_copy_file_hard_link_failure.__dict__.__setitem__('stypy_param_names_list', [])
        FileUtilTestCase.test_copy_file_hard_link_failure.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileUtilTestCase.test_copy_file_hard_link_failure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileUtilTestCase.test_copy_file_hard_link_failure.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileUtilTestCase.test_copy_file_hard_link_failure.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileUtilTestCase.test_copy_file_hard_link_failure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileUtilTestCase.test_copy_file_hard_link_failure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileUtilTestCase.test_copy_file_hard_link_failure', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_copy_file_hard_link_failure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_copy_file_hard_link_failure(...)' code ##################

        
        # Call to open(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'self' (line 100)
        self_39540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'self', False)
        # Obtaining the member 'source' of a type (line 100)
        source_39541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 18), self_39540, 'source')
        str_39542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 31), 'str', 'w')
        # Processing the call keyword arguments (line 100)
        kwargs_39543 = {}
        # Getting the type of 'open' (line 100)
        open_39539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'open', False)
        # Calling open(args, kwargs) (line 100)
        open_call_result_39544 = invoke(stypy.reporting.localization.Localization(__file__, 100, 13), open_39539, *[source_39541, str_39542], **kwargs_39543)
        
        with_39545 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 100, 13), open_call_result_39544, 'with parameter', '__enter__', '__exit__')

        if with_39545:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 100)
            enter___39546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 13), open_call_result_39544, '__enter__')
            with_enter_39547 = invoke(stypy.reporting.localization.Localization(__file__, 100, 13), enter___39546)
            # Assigning a type to the variable 'f' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'f', with_enter_39547)
            
            # Call to write(...): (line 101)
            # Processing the call arguments (line 101)
            str_39550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'str', 'some content')
            # Processing the call keyword arguments (line 101)
            kwargs_39551 = {}
            # Getting the type of 'f' (line 101)
            f_39548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'f', False)
            # Obtaining the member 'write' of a type (line 101)
            write_39549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), f_39548, 'write')
            # Calling write(args, kwargs) (line 101)
            write_call_result_39552 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), write_39549, *[str_39550], **kwargs_39551)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 100)
            exit___39553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 13), open_call_result_39544, '__exit__')
            with_exit_39554 = invoke(stypy.reporting.localization.Localization(__file__, 100, 13), exit___39553, None, None, None)

        
        # Assigning a Call to a Name (line 102):
        
        # Call to stat(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'self' (line 102)
        self_39557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 21), 'self', False)
        # Obtaining the member 'source' of a type (line 102)
        source_39558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 21), self_39557, 'source')
        # Processing the call keyword arguments (line 102)
        kwargs_39559 = {}
        # Getting the type of 'os' (line 102)
        os_39555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'os', False)
        # Obtaining the member 'stat' of a type (line 102)
        stat_39556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 13), os_39555, 'stat')
        # Calling stat(args, kwargs) (line 102)
        stat_call_result_39560 = invoke(stypy.reporting.localization.Localization(__file__, 102, 13), stat_39556, *[source_39558], **kwargs_39559)
        
        # Assigning a type to the variable 'st' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'st', stat_call_result_39560)

        @norecursion
        def _os_link(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_os_link'
            module_type_store = module_type_store.open_function_context('_os_link', 103, 8, False)
            
            # Passed parameters checking function
            _os_link.stypy_localization = localization
            _os_link.stypy_type_of_self = None
            _os_link.stypy_type_store = module_type_store
            _os_link.stypy_function_name = '_os_link'
            _os_link.stypy_param_names_list = []
            _os_link.stypy_varargs_param_name = 'args'
            _os_link.stypy_kwargs_param_name = None
            _os_link.stypy_call_defaults = defaults
            _os_link.stypy_call_varargs = varargs
            _os_link.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_os_link', [], 'args', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_os_link', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_os_link(...)' code ##################

            
            # Call to OSError(...): (line 104)
            # Processing the call arguments (line 104)
            int_39562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 26), 'int')
            str_39563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 29), 'str', 'linking unsupported')
            # Processing the call keyword arguments (line 104)
            kwargs_39564 = {}
            # Getting the type of 'OSError' (line 104)
            OSError_39561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'OSError', False)
            # Calling OSError(args, kwargs) (line 104)
            OSError_call_result_39565 = invoke(stypy.reporting.localization.Localization(__file__, 104, 18), OSError_39561, *[int_39562, str_39563], **kwargs_39564)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 104, 12), OSError_call_result_39565, 'raise parameter', BaseException)
            
            # ################# End of '_os_link(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_os_link' in the type store
            # Getting the type of 'stypy_return_type' (line 103)
            stypy_return_type_39566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_39566)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_os_link'
            return stypy_return_type_39566

        # Assigning a type to the variable '_os_link' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), '_os_link', _os_link)
        
        # Assigning a Attribute to a Name (line 105):
        # Getting the type of 'os' (line 105)
        os_39567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'os')
        # Obtaining the member 'link' of a type (line 105)
        link_39568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), os_39567, 'link')
        # Assigning a type to the variable 'old_link' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'old_link', link_39568)
        
        # Assigning a Name to a Attribute (line 106):
        # Getting the type of '_os_link' (line 106)
        _os_link_39569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), '_os_link')
        # Getting the type of 'os' (line 106)
        os_39570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'os')
        # Setting the type of the member 'link' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), os_39570, 'link', _os_link_39569)
        
        # Try-finally block (line 107)
        
        # Call to copy_file(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'self' (line 108)
        self_39572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 22), 'self', False)
        # Obtaining the member 'source' of a type (line 108)
        source_39573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 22), self_39572, 'source')
        # Getting the type of 'self' (line 108)
        self_39574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 35), 'self', False)
        # Obtaining the member 'target' of a type (line 108)
        target_39575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 35), self_39574, 'target')
        # Processing the call keyword arguments (line 108)
        str_39576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 53), 'str', 'hard')
        keyword_39577 = str_39576
        kwargs_39578 = {'link': keyword_39577}
        # Getting the type of 'copy_file' (line 108)
        copy_file_39571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'copy_file', False)
        # Calling copy_file(args, kwargs) (line 108)
        copy_file_call_result_39579 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), copy_file_39571, *[source_39573, target_39575], **kwargs_39578)
        
        
        # finally branch of the try-finally block (line 107)
        
        # Assigning a Name to a Attribute (line 110):
        # Getting the type of 'old_link' (line 110)
        old_link_39580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'old_link')
        # Getting the type of 'os' (line 110)
        os_39581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'os')
        # Setting the type of the member 'link' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), os_39581, 'link', old_link_39580)
        
        
        # Assigning a Call to a Name (line 111):
        
        # Call to stat(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'self' (line 111)
        self_39584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'self', False)
        # Obtaining the member 'source' of a type (line 111)
        source_39585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 22), self_39584, 'source')
        # Processing the call keyword arguments (line 111)
        kwargs_39586 = {}
        # Getting the type of 'os' (line 111)
        os_39582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 14), 'os', False)
        # Obtaining the member 'stat' of a type (line 111)
        stat_39583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 14), os_39582, 'stat')
        # Calling stat(args, kwargs) (line 111)
        stat_call_result_39587 = invoke(stypy.reporting.localization.Localization(__file__, 111, 14), stat_39583, *[source_39585], **kwargs_39586)
        
        # Assigning a type to the variable 'st2' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'st2', stat_call_result_39587)
        
        # Assigning a Call to a Name (line 112):
        
        # Call to stat(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'self' (line 112)
        self_39590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'self', False)
        # Obtaining the member 'target' of a type (line 112)
        target_39591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 22), self_39590, 'target')
        # Processing the call keyword arguments (line 112)
        kwargs_39592 = {}
        # Getting the type of 'os' (line 112)
        os_39588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 14), 'os', False)
        # Obtaining the member 'stat' of a type (line 112)
        stat_39589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 14), os_39588, 'stat')
        # Calling stat(args, kwargs) (line 112)
        stat_call_result_39593 = invoke(stypy.reporting.localization.Localization(__file__, 112, 14), stat_39589, *[target_39591], **kwargs_39592)
        
        # Assigning a type to the variable 'st3' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'st3', stat_call_result_39593)
        
        # Call to assertTrue(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to samestat(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'st' (line 113)
        st_39599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 41), 'st', False)
        # Getting the type of 'st2' (line 113)
        st2_39600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 45), 'st2', False)
        # Processing the call keyword arguments (line 113)
        kwargs_39601 = {}
        # Getting the type of 'os' (line 113)
        os_39596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 113)
        path_39597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 24), os_39596, 'path')
        # Obtaining the member 'samestat' of a type (line 113)
        samestat_39598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 24), path_39597, 'samestat')
        # Calling samestat(args, kwargs) (line 113)
        samestat_call_result_39602 = invoke(stypy.reporting.localization.Localization(__file__, 113, 24), samestat_39598, *[st_39599, st2_39600], **kwargs_39601)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 113)
        tuple_39603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 113)
        # Adding element type (line 113)
        # Getting the type of 'st' (line 113)
        st_39604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 52), 'st', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 52), tuple_39603, st_39604)
        # Adding element type (line 113)
        # Getting the type of 'st2' (line 113)
        st2_39605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 56), 'st2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 52), tuple_39603, st2_39605)
        
        # Processing the call keyword arguments (line 113)
        kwargs_39606 = {}
        # Getting the type of 'self' (line 113)
        self_39594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 113)
        assertTrue_39595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_39594, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 113)
        assertTrue_call_result_39607 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), assertTrue_39595, *[samestat_call_result_39602, tuple_39603], **kwargs_39606)
        
        
        # Call to assertFalse(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to samestat(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'st2' (line 114)
        st2_39613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 42), 'st2', False)
        # Getting the type of 'st3' (line 114)
        st3_39614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 47), 'st3', False)
        # Processing the call keyword arguments (line 114)
        kwargs_39615 = {}
        # Getting the type of 'os' (line 114)
        os_39610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 114)
        path_39611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 25), os_39610, 'path')
        # Obtaining the member 'samestat' of a type (line 114)
        samestat_39612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 25), path_39611, 'samestat')
        # Calling samestat(args, kwargs) (line 114)
        samestat_call_result_39616 = invoke(stypy.reporting.localization.Localization(__file__, 114, 25), samestat_39612, *[st2_39613, st3_39614], **kwargs_39615)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 114)
        tuple_39617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 114)
        # Adding element type (line 114)
        # Getting the type of 'st2' (line 114)
        st2_39618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 54), 'st2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 54), tuple_39617, st2_39618)
        # Adding element type (line 114)
        # Getting the type of 'st3' (line 114)
        st3_39619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 59), 'st3', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 54), tuple_39617, st3_39619)
        
        # Processing the call keyword arguments (line 114)
        kwargs_39620 = {}
        # Getting the type of 'self' (line 114)
        self_39608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 114)
        assertFalse_39609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_39608, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 114)
        assertFalse_call_result_39621 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), assertFalse_39609, *[samestat_call_result_39616, tuple_39617], **kwargs_39620)
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 115)
        tuple_39622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 115)
        # Adding element type (line 115)
        # Getting the type of 'self' (line 115)
        self_39623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'self')
        # Obtaining the member 'source' of a type (line 115)
        source_39624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 19), self_39623, 'source')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), tuple_39622, source_39624)
        # Adding element type (line 115)
        # Getting the type of 'self' (line 115)
        self_39625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'self')
        # Obtaining the member 'target' of a type (line 115)
        target_39626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 32), self_39625, 'target')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), tuple_39622, target_39626)
        
        # Testing the type of a for loop iterable (line 115)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 8), tuple_39622)
        # Getting the type of the for loop variable (line 115)
        for_loop_var_39627 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 8), tuple_39622)
        # Assigning a type to the variable 'fn' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'fn', for_loop_var_39627)
        # SSA begins for a for statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to open(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'fn' (line 116)
        fn_39629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'fn', False)
        str_39630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 26), 'str', 'r')
        # Processing the call keyword arguments (line 116)
        kwargs_39631 = {}
        # Getting the type of 'open' (line 116)
        open_39628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'open', False)
        # Calling open(args, kwargs) (line 116)
        open_call_result_39632 = invoke(stypy.reporting.localization.Localization(__file__, 116, 17), open_39628, *[fn_39629, str_39630], **kwargs_39631)
        
        with_39633 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 116, 17), open_call_result_39632, 'with parameter', '__enter__', '__exit__')

        if with_39633:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 116)
            enter___39634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 17), open_call_result_39632, '__enter__')
            with_enter_39635 = invoke(stypy.reporting.localization.Localization(__file__, 116, 17), enter___39634)
            # Assigning a type to the variable 'f' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'f', with_enter_39635)
            
            # Call to assertEqual(...): (line 117)
            # Processing the call arguments (line 117)
            
            # Call to read(...): (line 117)
            # Processing the call keyword arguments (line 117)
            kwargs_39640 = {}
            # Getting the type of 'f' (line 117)
            f_39638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 33), 'f', False)
            # Obtaining the member 'read' of a type (line 117)
            read_39639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 33), f_39638, 'read')
            # Calling read(args, kwargs) (line 117)
            read_call_result_39641 = invoke(stypy.reporting.localization.Localization(__file__, 117, 33), read_39639, *[], **kwargs_39640)
            
            str_39642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 43), 'str', 'some content')
            # Processing the call keyword arguments (line 117)
            kwargs_39643 = {}
            # Getting the type of 'self' (line 117)
            self_39636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'self', False)
            # Obtaining the member 'assertEqual' of a type (line 117)
            assertEqual_39637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), self_39636, 'assertEqual')
            # Calling assertEqual(args, kwargs) (line 117)
            assertEqual_call_result_39644 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), assertEqual_39637, *[read_call_result_39641, str_39642], **kwargs_39643)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 116)
            exit___39645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 17), open_call_result_39632, '__exit__')
            with_exit_39646 = invoke(stypy.reporting.localization.Localization(__file__, 116, 17), exit___39645, None, None, None)

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_copy_file_hard_link_failure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_copy_file_hard_link_failure' in the type store
        # Getting the type of 'stypy_return_type' (line 95)
        stypy_return_type_39647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39647)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_copy_file_hard_link_failure'
        return stypy_return_type_39647


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 16, 0, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileUtilTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FileUtilTestCase' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'FileUtilTestCase', FileUtilTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 120, 0, False)
    
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

    
    # Call to makeSuite(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'FileUtilTestCase' (line 121)
    FileUtilTestCase_39650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 30), 'FileUtilTestCase', False)
    # Processing the call keyword arguments (line 121)
    kwargs_39651 = {}
    # Getting the type of 'unittest' (line 121)
    unittest_39648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 121)
    makeSuite_39649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 11), unittest_39648, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 121)
    makeSuite_call_result_39652 = invoke(stypy.reporting.localization.Localization(__file__, 121, 11), makeSuite_39649, *[FileUtilTestCase_39650], **kwargs_39651)
    
    # Assigning a type to the variable 'stypy_return_type' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type', makeSuite_call_result_39652)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 120)
    stypy_return_type_39653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39653)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_39653

# Assigning a type to the variable 'test_suite' (line 120)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 124)
    # Processing the call arguments (line 124)
    
    # Call to test_suite(...): (line 124)
    # Processing the call keyword arguments (line 124)
    kwargs_39656 = {}
    # Getting the type of 'test_suite' (line 124)
    test_suite_39655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 124)
    test_suite_call_result_39657 = invoke(stypy.reporting.localization.Localization(__file__, 124, 17), test_suite_39655, *[], **kwargs_39656)
    
    # Processing the call keyword arguments (line 124)
    kwargs_39658 = {}
    # Getting the type of 'run_unittest' (line 124)
    run_unittest_39654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 124)
    run_unittest_call_result_39659 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), run_unittest_39654, *[test_suite_call_result_39657], **kwargs_39658)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
