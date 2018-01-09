
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- encoding: utf8 -*-
2: '''Tests for distutils.command.check.'''
3: import textwrap
4: import unittest
5: from test.test_support import run_unittest
6: 
7: from distutils.command.check import check, HAS_DOCUTILS
8: from distutils.tests import support
9: from distutils.errors import DistutilsSetupError
10: 
11: class CheckTestCase(support.LoggingSilencer,
12:                     support.TempdirManager,
13:                     unittest.TestCase):
14: 
15:     def _run(self, metadata=None, **options):
16:         if metadata is None:
17:             metadata = {}
18:         pkg_info, dist = self.create_dist(**metadata)
19:         cmd = check(dist)
20:         cmd.initialize_options()
21:         for name, value in options.items():
22:             setattr(cmd, name, value)
23:         cmd.ensure_finalized()
24:         cmd.run()
25:         return cmd
26: 
27:     def test_check_metadata(self):
28:         # let's run the command with no metadata at all
29:         # by default, check is checking the metadata
30:         # should have some warnings
31:         cmd = self._run()
32:         self.assertEqual(cmd._warnings, 2)
33: 
34:         # now let's add the required fields
35:         # and run it again, to make sure we don't get
36:         # any warning anymore
37:         metadata = {'url': 'xxx', 'author': 'xxx',
38:                     'author_email': 'xxx',
39:                     'name': 'xxx', 'version': 'xxx'}
40:         cmd = self._run(metadata)
41:         self.assertEqual(cmd._warnings, 0)
42: 
43:         # now with the strict mode, we should
44:         # get an error if there are missing metadata
45:         self.assertRaises(DistutilsSetupError, self._run, {}, **{'strict': 1})
46: 
47:         # and of course, no error when all metadata are present
48:         cmd = self._run(metadata, strict=1)
49:         self.assertEqual(cmd._warnings, 0)
50: 
51:         # now a test with Unicode entries
52:         metadata = {'url': u'xxx', 'author': u'\u00c9ric',
53:                     'author_email': u'xxx', u'name': 'xxx',
54:                     'version': u'xxx',
55:                     'description': u'Something about esszet \u00df',
56:                     'long_description': u'More things about esszet \u00df'}
57:         cmd = self._run(metadata)
58:         self.assertEqual(cmd._warnings, 0)
59: 
60:     @unittest.skipUnless(HAS_DOCUTILS, "won't test without docutils")
61:     def test_check_document(self):
62:         pkg_info, dist = self.create_dist()
63:         cmd = check(dist)
64: 
65:         # let's see if it detects broken rest
66:         broken_rest = 'title\n===\n\ntest'
67:         msgs = cmd._check_rst_data(broken_rest)
68:         self.assertEqual(len(msgs), 1)
69: 
70:         # and non-broken rest
71:         rest = 'title\n=====\n\ntest'
72:         msgs = cmd._check_rst_data(rest)
73:         self.assertEqual(len(msgs), 0)
74: 
75:     @unittest.skipUnless(HAS_DOCUTILS, "won't test without docutils")
76:     def test_check_restructuredtext(self):
77:         # let's see if it detects broken rest in long_description
78:         broken_rest = 'title\n===\n\ntest'
79:         pkg_info, dist = self.create_dist(long_description=broken_rest)
80:         cmd = check(dist)
81:         cmd.check_restructuredtext()
82:         self.assertEqual(cmd._warnings, 1)
83: 
84:         # let's see if we have an error with strict=1
85:         metadata = {'url': 'xxx', 'author': 'xxx',
86:                     'author_email': 'xxx',
87:                     'name': 'xxx', 'version': 'xxx',
88:                     'long_description': broken_rest}
89:         self.assertRaises(DistutilsSetupError, self._run, metadata,
90:                           **{'strict': 1, 'restructuredtext': 1})
91: 
92:         # and non-broken rest, including a non-ASCII character to test #12114
93:         metadata['long_description'] = u'title\n=====\n\ntest \u00df'
94:         cmd = self._run(metadata, strict=1, restructuredtext=1)
95:         self.assertEqual(cmd._warnings, 0)
96: 
97:     @unittest.skipUnless(HAS_DOCUTILS, "won't test without docutils")
98:     def test_check_restructuredtext_with_syntax_highlight(self):
99:         # Don't fail if there is a `code` or `code-block` directive
100: 
101:         example_rst_docs = []
102:         example_rst_docs.append(textwrap.dedent('''\
103:             Here's some code:
104: 
105:             .. code:: python
106: 
107:                 def foo():
108:                     pass
109:             '''))
110:         example_rst_docs.append(textwrap.dedent('''\
111:             Here's some code:
112: 
113:             .. code-block:: python
114: 
115:                 def foo():
116:                     pass
117:             '''))
118: 
119:         for rest_with_code in example_rst_docs:
120:             pkg_info, dist = self.create_dist(long_description=rest_with_code)
121:             cmd = check(dist)
122:             cmd.check_restructuredtext()
123:             self.assertEqual(cmd._warnings, 0)
124:             msgs = cmd._check_rst_data(rest_with_code)
125:             self.assertEqual(len(msgs), 0)
126: 
127:     def test_check_all(self):
128: 
129:         metadata = {'url': 'xxx', 'author': 'xxx'}
130:         self.assertRaises(DistutilsSetupError, self._run,
131:                           {}, **{'strict': 1,
132:                                  'restructuredtext': 1})
133: 
134: def test_suite():
135:     return unittest.makeSuite(CheckTestCase)
136: 
137: if __name__ == "__main__":
138:     run_unittest(test_suite())
139: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_34220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 0), 'str', 'Tests for distutils.command.check.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import textwrap' statement (line 3)
import textwrap

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'textwrap', textwrap, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import unittest' statement (line 4)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from test.test_support import run_unittest' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34221 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support')

if (type(import_34221) is not StypyTypeError):

    if (import_34221 != 'pyd_module'):
        __import__(import_34221)
        sys_modules_34222 = sys.modules[import_34221]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', sys_modules_34222.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_34222, sys_modules_34222.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', import_34221)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.command.check import check, HAS_DOCUTILS' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34223 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.check')

if (type(import_34223) is not StypyTypeError):

    if (import_34223 != 'pyd_module'):
        __import__(import_34223)
        sys_modules_34224 = sys.modules[import_34223]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.check', sys_modules_34224.module_type_store, module_type_store, ['check', 'HAS_DOCUTILS'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_34224, sys_modules_34224.module_type_store, module_type_store)
    else:
        from distutils.command.check import check, HAS_DOCUTILS

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.check', None, module_type_store, ['check', 'HAS_DOCUTILS'], [check, HAS_DOCUTILS])

else:
    # Assigning a type to the variable 'distutils.command.check' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.check', import_34223)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.tests import support' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34225 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests')

if (type(import_34225) is not StypyTypeError):

    if (import_34225 != 'pyd_module'):
        __import__(import_34225)
        sys_modules_34226 = sys.modules[import_34225]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', sys_modules_34226.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_34226, sys_modules_34226.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', import_34225)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.errors import DistutilsSetupError' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34227 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors')

if (type(import_34227) is not StypyTypeError):

    if (import_34227 != 'pyd_module'):
        __import__(import_34227)
        sys_modules_34228 = sys.modules[import_34227]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', sys_modules_34228.module_type_store, module_type_store, ['DistutilsSetupError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_34228, sys_modules_34228.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsSetupError

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', None, module_type_store, ['DistutilsSetupError'], [DistutilsSetupError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', import_34227)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'CheckTestCase' class
# Getting the type of 'support' (line 11)
support_34229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 20), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 11)
LoggingSilencer_34230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 20), support_34229, 'LoggingSilencer')
# Getting the type of 'support' (line 12)
support_34231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'support')
# Obtaining the member 'TempdirManager' of a type (line 12)
TempdirManager_34232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 20), support_34231, 'TempdirManager')
# Getting the type of 'unittest' (line 13)
unittest_34233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'unittest')
# Obtaining the member 'TestCase' of a type (line 13)
TestCase_34234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 20), unittest_34233, 'TestCase')

class CheckTestCase(LoggingSilencer_34230, TempdirManager_34232, TestCase_34234, ):

    @norecursion
    def _run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 15)
        None_34235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 28), 'None')
        defaults = [None_34235]
        # Create a new context for function '_run'
        module_type_store = module_type_store.open_function_context('_run', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CheckTestCase._run.__dict__.__setitem__('stypy_localization', localization)
        CheckTestCase._run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CheckTestCase._run.__dict__.__setitem__('stypy_type_store', module_type_store)
        CheckTestCase._run.__dict__.__setitem__('stypy_function_name', 'CheckTestCase._run')
        CheckTestCase._run.__dict__.__setitem__('stypy_param_names_list', ['metadata'])
        CheckTestCase._run.__dict__.__setitem__('stypy_varargs_param_name', None)
        CheckTestCase._run.__dict__.__setitem__('stypy_kwargs_param_name', 'options')
        CheckTestCase._run.__dict__.__setitem__('stypy_call_defaults', defaults)
        CheckTestCase._run.__dict__.__setitem__('stypy_call_varargs', varargs)
        CheckTestCase._run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CheckTestCase._run.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CheckTestCase._run', ['metadata'], None, 'options', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_run', localization, ['metadata'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_run(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 16)
        # Getting the type of 'metadata' (line 16)
        metadata_34236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'metadata')
        # Getting the type of 'None' (line 16)
        None_34237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'None')
        
        (may_be_34238, more_types_in_union_34239) = may_be_none(metadata_34236, None_34237)

        if may_be_34238:

            if more_types_in_union_34239:
                # Runtime conditional SSA (line 16)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Dict to a Name (line 17):
            
            # Assigning a Dict to a Name (line 17):
            
            # Obtaining an instance of the builtin type 'dict' (line 17)
            dict_34240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 17)
            
            # Assigning a type to the variable 'metadata' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'metadata', dict_34240)

            if more_types_in_union_34239:
                # SSA join for if statement (line 16)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 18):
        
        # Assigning a Subscript to a Name (line 18):
        
        # Obtaining the type of the subscript
        int_34241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'int')
        
        # Call to create_dist(...): (line 18)
        # Processing the call keyword arguments (line 18)
        # Getting the type of 'metadata' (line 18)
        metadata_34244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 44), 'metadata', False)
        kwargs_34245 = {'metadata_34244': metadata_34244}
        # Getting the type of 'self' (line 18)
        self_34242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 18)
        create_dist_34243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 25), self_34242, 'create_dist')
        # Calling create_dist(args, kwargs) (line 18)
        create_dist_call_result_34246 = invoke(stypy.reporting.localization.Localization(__file__, 18, 25), create_dist_34243, *[], **kwargs_34245)
        
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___34247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), create_dist_call_result_34246, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_34248 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), getitem___34247, int_34241)
        
        # Assigning a type to the variable 'tuple_var_assignment_34212' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_34212', subscript_call_result_34248)
        
        # Assigning a Subscript to a Name (line 18):
        
        # Obtaining the type of the subscript
        int_34249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'int')
        
        # Call to create_dist(...): (line 18)
        # Processing the call keyword arguments (line 18)
        # Getting the type of 'metadata' (line 18)
        metadata_34252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 44), 'metadata', False)
        kwargs_34253 = {'metadata_34252': metadata_34252}
        # Getting the type of 'self' (line 18)
        self_34250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 18)
        create_dist_34251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 25), self_34250, 'create_dist')
        # Calling create_dist(args, kwargs) (line 18)
        create_dist_call_result_34254 = invoke(stypy.reporting.localization.Localization(__file__, 18, 25), create_dist_34251, *[], **kwargs_34253)
        
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___34255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), create_dist_call_result_34254, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_34256 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), getitem___34255, int_34249)
        
        # Assigning a type to the variable 'tuple_var_assignment_34213' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_34213', subscript_call_result_34256)
        
        # Assigning a Name to a Name (line 18):
        # Getting the type of 'tuple_var_assignment_34212' (line 18)
        tuple_var_assignment_34212_34257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_34212')
        # Assigning a type to the variable 'pkg_info' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'pkg_info', tuple_var_assignment_34212_34257)
        
        # Assigning a Name to a Name (line 18):
        # Getting the type of 'tuple_var_assignment_34213' (line 18)
        tuple_var_assignment_34213_34258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'tuple_var_assignment_34213')
        # Assigning a type to the variable 'dist' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 18), 'dist', tuple_var_assignment_34213_34258)
        
        # Assigning a Call to a Name (line 19):
        
        # Assigning a Call to a Name (line 19):
        
        # Call to check(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'dist' (line 19)
        dist_34260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'dist', False)
        # Processing the call keyword arguments (line 19)
        kwargs_34261 = {}
        # Getting the type of 'check' (line 19)
        check_34259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'check', False)
        # Calling check(args, kwargs) (line 19)
        check_call_result_34262 = invoke(stypy.reporting.localization.Localization(__file__, 19, 14), check_34259, *[dist_34260], **kwargs_34261)
        
        # Assigning a type to the variable 'cmd' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'cmd', check_call_result_34262)
        
        # Call to initialize_options(...): (line 20)
        # Processing the call keyword arguments (line 20)
        kwargs_34265 = {}
        # Getting the type of 'cmd' (line 20)
        cmd_34263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'cmd', False)
        # Obtaining the member 'initialize_options' of a type (line 20)
        initialize_options_34264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), cmd_34263, 'initialize_options')
        # Calling initialize_options(args, kwargs) (line 20)
        initialize_options_call_result_34266 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), initialize_options_34264, *[], **kwargs_34265)
        
        
        
        # Call to items(...): (line 21)
        # Processing the call keyword arguments (line 21)
        kwargs_34269 = {}
        # Getting the type of 'options' (line 21)
        options_34267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 27), 'options', False)
        # Obtaining the member 'items' of a type (line 21)
        items_34268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 27), options_34267, 'items')
        # Calling items(args, kwargs) (line 21)
        items_call_result_34270 = invoke(stypy.reporting.localization.Localization(__file__, 21, 27), items_34268, *[], **kwargs_34269)
        
        # Testing the type of a for loop iterable (line 21)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 21, 8), items_call_result_34270)
        # Getting the type of the for loop variable (line 21)
        for_loop_var_34271 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 21, 8), items_call_result_34270)
        # Assigning a type to the variable 'name' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 8), for_loop_var_34271))
        # Assigning a type to the variable 'value' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 8), for_loop_var_34271))
        # SSA begins for a for statement (line 21)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'cmd' (line 22)
        cmd_34273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'cmd', False)
        # Getting the type of 'name' (line 22)
        name_34274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'name', False)
        # Getting the type of 'value' (line 22)
        value_34275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 31), 'value', False)
        # Processing the call keyword arguments (line 22)
        kwargs_34276 = {}
        # Getting the type of 'setattr' (line 22)
        setattr_34272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 22)
        setattr_call_result_34277 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), setattr_34272, *[cmd_34273, name_34274, value_34275], **kwargs_34276)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to ensure_finalized(...): (line 23)
        # Processing the call keyword arguments (line 23)
        kwargs_34280 = {}
        # Getting the type of 'cmd' (line 23)
        cmd_34278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 23)
        ensure_finalized_34279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), cmd_34278, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 23)
        ensure_finalized_call_result_34281 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), ensure_finalized_34279, *[], **kwargs_34280)
        
        
        # Call to run(...): (line 24)
        # Processing the call keyword arguments (line 24)
        kwargs_34284 = {}
        # Getting the type of 'cmd' (line 24)
        cmd_34282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 24)
        run_34283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), cmd_34282, 'run')
        # Calling run(args, kwargs) (line 24)
        run_call_result_34285 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), run_34283, *[], **kwargs_34284)
        
        # Getting the type of 'cmd' (line 25)
        cmd_34286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'cmd')
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', cmd_34286)
        
        # ################# End of '_run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_run' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_34287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34287)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_run'
        return stypy_return_type_34287


    @norecursion
    def test_check_metadata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_check_metadata'
        module_type_store = module_type_store.open_function_context('test_check_metadata', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CheckTestCase.test_check_metadata.__dict__.__setitem__('stypy_localization', localization)
        CheckTestCase.test_check_metadata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CheckTestCase.test_check_metadata.__dict__.__setitem__('stypy_type_store', module_type_store)
        CheckTestCase.test_check_metadata.__dict__.__setitem__('stypy_function_name', 'CheckTestCase.test_check_metadata')
        CheckTestCase.test_check_metadata.__dict__.__setitem__('stypy_param_names_list', [])
        CheckTestCase.test_check_metadata.__dict__.__setitem__('stypy_varargs_param_name', None)
        CheckTestCase.test_check_metadata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CheckTestCase.test_check_metadata.__dict__.__setitem__('stypy_call_defaults', defaults)
        CheckTestCase.test_check_metadata.__dict__.__setitem__('stypy_call_varargs', varargs)
        CheckTestCase.test_check_metadata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CheckTestCase.test_check_metadata.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CheckTestCase.test_check_metadata', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_check_metadata', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_check_metadata(...)' code ##################

        
        # Assigning a Call to a Name (line 31):
        
        # Assigning a Call to a Name (line 31):
        
        # Call to _run(...): (line 31)
        # Processing the call keyword arguments (line 31)
        kwargs_34290 = {}
        # Getting the type of 'self' (line 31)
        self_34288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'self', False)
        # Obtaining the member '_run' of a type (line 31)
        _run_34289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 14), self_34288, '_run')
        # Calling _run(args, kwargs) (line 31)
        _run_call_result_34291 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), _run_34289, *[], **kwargs_34290)
        
        # Assigning a type to the variable 'cmd' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'cmd', _run_call_result_34291)
        
        # Call to assertEqual(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'cmd' (line 32)
        cmd_34294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'cmd', False)
        # Obtaining the member '_warnings' of a type (line 32)
        _warnings_34295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 25), cmd_34294, '_warnings')
        int_34296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 40), 'int')
        # Processing the call keyword arguments (line 32)
        kwargs_34297 = {}
        # Getting the type of 'self' (line 32)
        self_34292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 32)
        assertEqual_34293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_34292, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 32)
        assertEqual_call_result_34298 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), assertEqual_34293, *[_warnings_34295, int_34296], **kwargs_34297)
        
        
        # Assigning a Dict to a Name (line 37):
        
        # Assigning a Dict to a Name (line 37):
        
        # Obtaining an instance of the builtin type 'dict' (line 37)
        dict_34299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 37)
        # Adding element type (key, value) (line 37)
        str_34300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'str', 'url')
        str_34301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 27), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 19), dict_34299, (str_34300, str_34301))
        # Adding element type (key, value) (line 37)
        str_34302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 34), 'str', 'author')
        str_34303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 44), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 19), dict_34299, (str_34302, str_34303))
        # Adding element type (key, value) (line 37)
        str_34304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'str', 'author_email')
        str_34305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 19), dict_34299, (str_34304, str_34305))
        # Adding element type (key, value) (line 37)
        str_34306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'str', 'name')
        str_34307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 28), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 19), dict_34299, (str_34306, str_34307))
        # Adding element type (key, value) (line 37)
        str_34308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 35), 'str', 'version')
        str_34309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 46), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 19), dict_34299, (str_34308, str_34309))
        
        # Assigning a type to the variable 'metadata' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'metadata', dict_34299)
        
        # Assigning a Call to a Name (line 40):
        
        # Assigning a Call to a Name (line 40):
        
        # Call to _run(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'metadata' (line 40)
        metadata_34312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'metadata', False)
        # Processing the call keyword arguments (line 40)
        kwargs_34313 = {}
        # Getting the type of 'self' (line 40)
        self_34310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'self', False)
        # Obtaining the member '_run' of a type (line 40)
        _run_34311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 14), self_34310, '_run')
        # Calling _run(args, kwargs) (line 40)
        _run_call_result_34314 = invoke(stypy.reporting.localization.Localization(__file__, 40, 14), _run_34311, *[metadata_34312], **kwargs_34313)
        
        # Assigning a type to the variable 'cmd' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'cmd', _run_call_result_34314)
        
        # Call to assertEqual(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'cmd' (line 41)
        cmd_34317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'cmd', False)
        # Obtaining the member '_warnings' of a type (line 41)
        _warnings_34318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 25), cmd_34317, '_warnings')
        int_34319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 40), 'int')
        # Processing the call keyword arguments (line 41)
        kwargs_34320 = {}
        # Getting the type of 'self' (line 41)
        self_34315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 41)
        assertEqual_34316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_34315, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 41)
        assertEqual_call_result_34321 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), assertEqual_34316, *[_warnings_34318, int_34319], **kwargs_34320)
        
        
        # Call to assertRaises(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'DistutilsSetupError' (line 45)
        DistutilsSetupError_34324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'DistutilsSetupError', False)
        # Getting the type of 'self' (line 45)
        self_34325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 47), 'self', False)
        # Obtaining the member '_run' of a type (line 45)
        _run_34326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 47), self_34325, '_run')
        
        # Obtaining an instance of the builtin type 'dict' (line 45)
        dict_34327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 58), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 45)
        
        # Processing the call keyword arguments (line 45)
        
        # Obtaining an instance of the builtin type 'dict' (line 45)
        dict_34328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 64), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 45)
        # Adding element type (key, value) (line 45)
        str_34329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 65), 'str', 'strict')
        int_34330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 75), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 64), dict_34328, (str_34329, int_34330))
        
        kwargs_34331 = {'dict_34328': dict_34328}
        # Getting the type of 'self' (line 45)
        self_34322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 45)
        assertRaises_34323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_34322, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 45)
        assertRaises_call_result_34332 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), assertRaises_34323, *[DistutilsSetupError_34324, _run_34326, dict_34327], **kwargs_34331)
        
        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to _run(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'metadata' (line 48)
        metadata_34335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'metadata', False)
        # Processing the call keyword arguments (line 48)
        int_34336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 41), 'int')
        keyword_34337 = int_34336
        kwargs_34338 = {'strict': keyword_34337}
        # Getting the type of 'self' (line 48)
        self_34333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'self', False)
        # Obtaining the member '_run' of a type (line 48)
        _run_34334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 14), self_34333, '_run')
        # Calling _run(args, kwargs) (line 48)
        _run_call_result_34339 = invoke(stypy.reporting.localization.Localization(__file__, 48, 14), _run_34334, *[metadata_34335], **kwargs_34338)
        
        # Assigning a type to the variable 'cmd' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'cmd', _run_call_result_34339)
        
        # Call to assertEqual(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'cmd' (line 49)
        cmd_34342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'cmd', False)
        # Obtaining the member '_warnings' of a type (line 49)
        _warnings_34343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 25), cmd_34342, '_warnings')
        int_34344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 40), 'int')
        # Processing the call keyword arguments (line 49)
        kwargs_34345 = {}
        # Getting the type of 'self' (line 49)
        self_34340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 49)
        assertEqual_34341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_34340, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 49)
        assertEqual_call_result_34346 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), assertEqual_34341, *[_warnings_34343, int_34344], **kwargs_34345)
        
        
        # Assigning a Dict to a Name (line 52):
        
        # Assigning a Dict to a Name (line 52):
        
        # Obtaining an instance of the builtin type 'dict' (line 52)
        dict_34347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 52)
        # Adding element type (key, value) (line 52)
        str_34348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 20), 'str', 'url')
        unicode_34349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 27), 'unicode', u'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 19), dict_34347, (str_34348, unicode_34349))
        # Adding element type (key, value) (line 52)
        str_34350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 35), 'str', 'author')
        unicode_34351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 45), 'unicode', u'\xc9ric')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 19), dict_34347, (str_34350, unicode_34351))
        # Adding element type (key, value) (line 52)
        str_34352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 20), 'str', 'author_email')
        unicode_34353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 36), 'unicode', u'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 19), dict_34347, (str_34352, unicode_34353))
        # Adding element type (key, value) (line 52)
        unicode_34354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 44), 'unicode', u'name')
        str_34355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 53), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 19), dict_34347, (unicode_34354, str_34355))
        # Adding element type (key, value) (line 52)
        str_34356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 20), 'str', 'version')
        unicode_34357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 31), 'unicode', u'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 19), dict_34347, (str_34356, unicode_34357))
        # Adding element type (key, value) (line 52)
        str_34358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'str', 'description')
        unicode_34359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 35), 'unicode', u'Something about esszet \xdf')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 19), dict_34347, (str_34358, unicode_34359))
        # Adding element type (key, value) (line 52)
        str_34360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 20), 'str', 'long_description')
        unicode_34361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 40), 'unicode', u'More things about esszet \xdf')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 19), dict_34347, (str_34360, unicode_34361))
        
        # Assigning a type to the variable 'metadata' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'metadata', dict_34347)
        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to _run(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'metadata' (line 57)
        metadata_34364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'metadata', False)
        # Processing the call keyword arguments (line 57)
        kwargs_34365 = {}
        # Getting the type of 'self' (line 57)
        self_34362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), 'self', False)
        # Obtaining the member '_run' of a type (line 57)
        _run_34363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 14), self_34362, '_run')
        # Calling _run(args, kwargs) (line 57)
        _run_call_result_34366 = invoke(stypy.reporting.localization.Localization(__file__, 57, 14), _run_34363, *[metadata_34364], **kwargs_34365)
        
        # Assigning a type to the variable 'cmd' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'cmd', _run_call_result_34366)
        
        # Call to assertEqual(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'cmd' (line 58)
        cmd_34369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 25), 'cmd', False)
        # Obtaining the member '_warnings' of a type (line 58)
        _warnings_34370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 25), cmd_34369, '_warnings')
        int_34371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 40), 'int')
        # Processing the call keyword arguments (line 58)
        kwargs_34372 = {}
        # Getting the type of 'self' (line 58)
        self_34367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 58)
        assertEqual_34368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_34367, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 58)
        assertEqual_call_result_34373 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), assertEqual_34368, *[_warnings_34370, int_34371], **kwargs_34372)
        
        
        # ################# End of 'test_check_metadata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_check_metadata' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_34374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34374)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_check_metadata'
        return stypy_return_type_34374


    @norecursion
    def test_check_document(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_check_document'
        module_type_store = module_type_store.open_function_context('test_check_document', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CheckTestCase.test_check_document.__dict__.__setitem__('stypy_localization', localization)
        CheckTestCase.test_check_document.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CheckTestCase.test_check_document.__dict__.__setitem__('stypy_type_store', module_type_store)
        CheckTestCase.test_check_document.__dict__.__setitem__('stypy_function_name', 'CheckTestCase.test_check_document')
        CheckTestCase.test_check_document.__dict__.__setitem__('stypy_param_names_list', [])
        CheckTestCase.test_check_document.__dict__.__setitem__('stypy_varargs_param_name', None)
        CheckTestCase.test_check_document.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CheckTestCase.test_check_document.__dict__.__setitem__('stypy_call_defaults', defaults)
        CheckTestCase.test_check_document.__dict__.__setitem__('stypy_call_varargs', varargs)
        CheckTestCase.test_check_document.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CheckTestCase.test_check_document.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CheckTestCase.test_check_document', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_check_document', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_check_document(...)' code ##################

        
        # Assigning a Call to a Tuple (line 62):
        
        # Assigning a Subscript to a Name (line 62):
        
        # Obtaining the type of the subscript
        int_34375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'int')
        
        # Call to create_dist(...): (line 62)
        # Processing the call keyword arguments (line 62)
        kwargs_34378 = {}
        # Getting the type of 'self' (line 62)
        self_34376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 62)
        create_dist_34377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 25), self_34376, 'create_dist')
        # Calling create_dist(args, kwargs) (line 62)
        create_dist_call_result_34379 = invoke(stypy.reporting.localization.Localization(__file__, 62, 25), create_dist_34377, *[], **kwargs_34378)
        
        # Obtaining the member '__getitem__' of a type (line 62)
        getitem___34380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), create_dist_call_result_34379, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 62)
        subscript_call_result_34381 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), getitem___34380, int_34375)
        
        # Assigning a type to the variable 'tuple_var_assignment_34214' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_34214', subscript_call_result_34381)
        
        # Assigning a Subscript to a Name (line 62):
        
        # Obtaining the type of the subscript
        int_34382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'int')
        
        # Call to create_dist(...): (line 62)
        # Processing the call keyword arguments (line 62)
        kwargs_34385 = {}
        # Getting the type of 'self' (line 62)
        self_34383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 62)
        create_dist_34384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 25), self_34383, 'create_dist')
        # Calling create_dist(args, kwargs) (line 62)
        create_dist_call_result_34386 = invoke(stypy.reporting.localization.Localization(__file__, 62, 25), create_dist_34384, *[], **kwargs_34385)
        
        # Obtaining the member '__getitem__' of a type (line 62)
        getitem___34387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), create_dist_call_result_34386, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 62)
        subscript_call_result_34388 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), getitem___34387, int_34382)
        
        # Assigning a type to the variable 'tuple_var_assignment_34215' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_34215', subscript_call_result_34388)
        
        # Assigning a Name to a Name (line 62):
        # Getting the type of 'tuple_var_assignment_34214' (line 62)
        tuple_var_assignment_34214_34389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_34214')
        # Assigning a type to the variable 'pkg_info' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'pkg_info', tuple_var_assignment_34214_34389)
        
        # Assigning a Name to a Name (line 62):
        # Getting the type of 'tuple_var_assignment_34215' (line 62)
        tuple_var_assignment_34215_34390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_34215')
        # Assigning a type to the variable 'dist' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'dist', tuple_var_assignment_34215_34390)
        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to check(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'dist' (line 63)
        dist_34392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'dist', False)
        # Processing the call keyword arguments (line 63)
        kwargs_34393 = {}
        # Getting the type of 'check' (line 63)
        check_34391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 14), 'check', False)
        # Calling check(args, kwargs) (line 63)
        check_call_result_34394 = invoke(stypy.reporting.localization.Localization(__file__, 63, 14), check_34391, *[dist_34392], **kwargs_34393)
        
        # Assigning a type to the variable 'cmd' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'cmd', check_call_result_34394)
        
        # Assigning a Str to a Name (line 66):
        
        # Assigning a Str to a Name (line 66):
        str_34395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 22), 'str', 'title\n===\n\ntest')
        # Assigning a type to the variable 'broken_rest' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'broken_rest', str_34395)
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to _check_rst_data(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'broken_rest' (line 67)
        broken_rest_34398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 35), 'broken_rest', False)
        # Processing the call keyword arguments (line 67)
        kwargs_34399 = {}
        # Getting the type of 'cmd' (line 67)
        cmd_34396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'cmd', False)
        # Obtaining the member '_check_rst_data' of a type (line 67)
        _check_rst_data_34397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 15), cmd_34396, '_check_rst_data')
        # Calling _check_rst_data(args, kwargs) (line 67)
        _check_rst_data_call_result_34400 = invoke(stypy.reporting.localization.Localization(__file__, 67, 15), _check_rst_data_34397, *[broken_rest_34398], **kwargs_34399)
        
        # Assigning a type to the variable 'msgs' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'msgs', _check_rst_data_call_result_34400)
        
        # Call to assertEqual(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Call to len(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'msgs' (line 68)
        msgs_34404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 29), 'msgs', False)
        # Processing the call keyword arguments (line 68)
        kwargs_34405 = {}
        # Getting the type of 'len' (line 68)
        len_34403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'len', False)
        # Calling len(args, kwargs) (line 68)
        len_call_result_34406 = invoke(stypy.reporting.localization.Localization(__file__, 68, 25), len_34403, *[msgs_34404], **kwargs_34405)
        
        int_34407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 36), 'int')
        # Processing the call keyword arguments (line 68)
        kwargs_34408 = {}
        # Getting the type of 'self' (line 68)
        self_34401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 68)
        assertEqual_34402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_34401, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 68)
        assertEqual_call_result_34409 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), assertEqual_34402, *[len_call_result_34406, int_34407], **kwargs_34408)
        
        
        # Assigning a Str to a Name (line 71):
        
        # Assigning a Str to a Name (line 71):
        str_34410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 15), 'str', 'title\n=====\n\ntest')
        # Assigning a type to the variable 'rest' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'rest', str_34410)
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to _check_rst_data(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'rest' (line 72)
        rest_34413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 35), 'rest', False)
        # Processing the call keyword arguments (line 72)
        kwargs_34414 = {}
        # Getting the type of 'cmd' (line 72)
        cmd_34411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'cmd', False)
        # Obtaining the member '_check_rst_data' of a type (line 72)
        _check_rst_data_34412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 15), cmd_34411, '_check_rst_data')
        # Calling _check_rst_data(args, kwargs) (line 72)
        _check_rst_data_call_result_34415 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), _check_rst_data_34412, *[rest_34413], **kwargs_34414)
        
        # Assigning a type to the variable 'msgs' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'msgs', _check_rst_data_call_result_34415)
        
        # Call to assertEqual(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to len(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'msgs' (line 73)
        msgs_34419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 29), 'msgs', False)
        # Processing the call keyword arguments (line 73)
        kwargs_34420 = {}
        # Getting the type of 'len' (line 73)
        len_34418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 25), 'len', False)
        # Calling len(args, kwargs) (line 73)
        len_call_result_34421 = invoke(stypy.reporting.localization.Localization(__file__, 73, 25), len_34418, *[msgs_34419], **kwargs_34420)
        
        int_34422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 36), 'int')
        # Processing the call keyword arguments (line 73)
        kwargs_34423 = {}
        # Getting the type of 'self' (line 73)
        self_34416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 73)
        assertEqual_34417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_34416, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 73)
        assertEqual_call_result_34424 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), assertEqual_34417, *[len_call_result_34421, int_34422], **kwargs_34423)
        
        
        # ################# End of 'test_check_document(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_check_document' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_34425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34425)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_check_document'
        return stypy_return_type_34425


    @norecursion
    def test_check_restructuredtext(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_check_restructuredtext'
        module_type_store = module_type_store.open_function_context('test_check_restructuredtext', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CheckTestCase.test_check_restructuredtext.__dict__.__setitem__('stypy_localization', localization)
        CheckTestCase.test_check_restructuredtext.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CheckTestCase.test_check_restructuredtext.__dict__.__setitem__('stypy_type_store', module_type_store)
        CheckTestCase.test_check_restructuredtext.__dict__.__setitem__('stypy_function_name', 'CheckTestCase.test_check_restructuredtext')
        CheckTestCase.test_check_restructuredtext.__dict__.__setitem__('stypy_param_names_list', [])
        CheckTestCase.test_check_restructuredtext.__dict__.__setitem__('stypy_varargs_param_name', None)
        CheckTestCase.test_check_restructuredtext.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CheckTestCase.test_check_restructuredtext.__dict__.__setitem__('stypy_call_defaults', defaults)
        CheckTestCase.test_check_restructuredtext.__dict__.__setitem__('stypy_call_varargs', varargs)
        CheckTestCase.test_check_restructuredtext.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CheckTestCase.test_check_restructuredtext.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CheckTestCase.test_check_restructuredtext', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_check_restructuredtext', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_check_restructuredtext(...)' code ##################

        
        # Assigning a Str to a Name (line 78):
        
        # Assigning a Str to a Name (line 78):
        str_34426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 22), 'str', 'title\n===\n\ntest')
        # Assigning a type to the variable 'broken_rest' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'broken_rest', str_34426)
        
        # Assigning a Call to a Tuple (line 79):
        
        # Assigning a Subscript to a Name (line 79):
        
        # Obtaining the type of the subscript
        int_34427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 8), 'int')
        
        # Call to create_dist(...): (line 79)
        # Processing the call keyword arguments (line 79)
        # Getting the type of 'broken_rest' (line 79)
        broken_rest_34430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 59), 'broken_rest', False)
        keyword_34431 = broken_rest_34430
        kwargs_34432 = {'long_description': keyword_34431}
        # Getting the type of 'self' (line 79)
        self_34428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 79)
        create_dist_34429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 25), self_34428, 'create_dist')
        # Calling create_dist(args, kwargs) (line 79)
        create_dist_call_result_34433 = invoke(stypy.reporting.localization.Localization(__file__, 79, 25), create_dist_34429, *[], **kwargs_34432)
        
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___34434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), create_dist_call_result_34433, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_34435 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), getitem___34434, int_34427)
        
        # Assigning a type to the variable 'tuple_var_assignment_34216' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'tuple_var_assignment_34216', subscript_call_result_34435)
        
        # Assigning a Subscript to a Name (line 79):
        
        # Obtaining the type of the subscript
        int_34436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 8), 'int')
        
        # Call to create_dist(...): (line 79)
        # Processing the call keyword arguments (line 79)
        # Getting the type of 'broken_rest' (line 79)
        broken_rest_34439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 59), 'broken_rest', False)
        keyword_34440 = broken_rest_34439
        kwargs_34441 = {'long_description': keyword_34440}
        # Getting the type of 'self' (line 79)
        self_34437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 79)
        create_dist_34438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 25), self_34437, 'create_dist')
        # Calling create_dist(args, kwargs) (line 79)
        create_dist_call_result_34442 = invoke(stypy.reporting.localization.Localization(__file__, 79, 25), create_dist_34438, *[], **kwargs_34441)
        
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___34443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), create_dist_call_result_34442, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_34444 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), getitem___34443, int_34436)
        
        # Assigning a type to the variable 'tuple_var_assignment_34217' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'tuple_var_assignment_34217', subscript_call_result_34444)
        
        # Assigning a Name to a Name (line 79):
        # Getting the type of 'tuple_var_assignment_34216' (line 79)
        tuple_var_assignment_34216_34445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'tuple_var_assignment_34216')
        # Assigning a type to the variable 'pkg_info' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'pkg_info', tuple_var_assignment_34216_34445)
        
        # Assigning a Name to a Name (line 79):
        # Getting the type of 'tuple_var_assignment_34217' (line 79)
        tuple_var_assignment_34217_34446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'tuple_var_assignment_34217')
        # Assigning a type to the variable 'dist' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'dist', tuple_var_assignment_34217_34446)
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to check(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'dist' (line 80)
        dist_34448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'dist', False)
        # Processing the call keyword arguments (line 80)
        kwargs_34449 = {}
        # Getting the type of 'check' (line 80)
        check_34447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 14), 'check', False)
        # Calling check(args, kwargs) (line 80)
        check_call_result_34450 = invoke(stypy.reporting.localization.Localization(__file__, 80, 14), check_34447, *[dist_34448], **kwargs_34449)
        
        # Assigning a type to the variable 'cmd' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'cmd', check_call_result_34450)
        
        # Call to check_restructuredtext(...): (line 81)
        # Processing the call keyword arguments (line 81)
        kwargs_34453 = {}
        # Getting the type of 'cmd' (line 81)
        cmd_34451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'cmd', False)
        # Obtaining the member 'check_restructuredtext' of a type (line 81)
        check_restructuredtext_34452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), cmd_34451, 'check_restructuredtext')
        # Calling check_restructuredtext(args, kwargs) (line 81)
        check_restructuredtext_call_result_34454 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), check_restructuredtext_34452, *[], **kwargs_34453)
        
        
        # Call to assertEqual(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'cmd' (line 82)
        cmd_34457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'cmd', False)
        # Obtaining the member '_warnings' of a type (line 82)
        _warnings_34458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 25), cmd_34457, '_warnings')
        int_34459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 40), 'int')
        # Processing the call keyword arguments (line 82)
        kwargs_34460 = {}
        # Getting the type of 'self' (line 82)
        self_34455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 82)
        assertEqual_34456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_34455, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 82)
        assertEqual_call_result_34461 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), assertEqual_34456, *[_warnings_34458, int_34459], **kwargs_34460)
        
        
        # Assigning a Dict to a Name (line 85):
        
        # Assigning a Dict to a Name (line 85):
        
        # Obtaining an instance of the builtin type 'dict' (line 85)
        dict_34462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 85)
        # Adding element type (key, value) (line 85)
        str_34463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 20), 'str', 'url')
        str_34464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 27), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 19), dict_34462, (str_34463, str_34464))
        # Adding element type (key, value) (line 85)
        str_34465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 34), 'str', 'author')
        str_34466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 44), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 19), dict_34462, (str_34465, str_34466))
        # Adding element type (key, value) (line 85)
        str_34467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 20), 'str', 'author_email')
        str_34468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 36), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 19), dict_34462, (str_34467, str_34468))
        # Adding element type (key, value) (line 85)
        str_34469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'str', 'name')
        str_34470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 28), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 19), dict_34462, (str_34469, str_34470))
        # Adding element type (key, value) (line 85)
        str_34471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 35), 'str', 'version')
        str_34472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 46), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 19), dict_34462, (str_34471, str_34472))
        # Adding element type (key, value) (line 85)
        str_34473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'str', 'long_description')
        # Getting the type of 'broken_rest' (line 88)
        broken_rest_34474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 40), 'broken_rest')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 19), dict_34462, (str_34473, broken_rest_34474))
        
        # Assigning a type to the variable 'metadata' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'metadata', dict_34462)
        
        # Call to assertRaises(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'DistutilsSetupError' (line 89)
        DistutilsSetupError_34477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'DistutilsSetupError', False)
        # Getting the type of 'self' (line 89)
        self_34478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 47), 'self', False)
        # Obtaining the member '_run' of a type (line 89)
        _run_34479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 47), self_34478, '_run')
        # Getting the type of 'metadata' (line 89)
        metadata_34480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 58), 'metadata', False)
        # Processing the call keyword arguments (line 89)
        
        # Obtaining an instance of the builtin type 'dict' (line 90)
        dict_34481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 90)
        # Adding element type (key, value) (line 90)
        str_34482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 29), 'str', 'strict')
        int_34483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 39), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 28), dict_34481, (str_34482, int_34483))
        # Adding element type (key, value) (line 90)
        str_34484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 42), 'str', 'restructuredtext')
        int_34485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 62), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 28), dict_34481, (str_34484, int_34485))
        
        kwargs_34486 = {'dict_34481': dict_34481}
        # Getting the type of 'self' (line 89)
        self_34475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 89)
        assertRaises_34476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_34475, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 89)
        assertRaises_call_result_34487 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assertRaises_34476, *[DistutilsSetupError_34477, _run_34479, metadata_34480], **kwargs_34486)
        
        
        # Assigning a Str to a Subscript (line 93):
        
        # Assigning a Str to a Subscript (line 93):
        unicode_34488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 39), 'unicode', u'title\n=====\n\ntest \xdf')
        # Getting the type of 'metadata' (line 93)
        metadata_34489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'metadata')
        str_34490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 17), 'str', 'long_description')
        # Storing an element on a container (line 93)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 8), metadata_34489, (str_34490, unicode_34488))
        
        # Assigning a Call to a Name (line 94):
        
        # Assigning a Call to a Name (line 94):
        
        # Call to _run(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'metadata' (line 94)
        metadata_34493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'metadata', False)
        # Processing the call keyword arguments (line 94)
        int_34494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 41), 'int')
        keyword_34495 = int_34494
        int_34496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 61), 'int')
        keyword_34497 = int_34496
        kwargs_34498 = {'strict': keyword_34495, 'restructuredtext': keyword_34497}
        # Getting the type of 'self' (line 94)
        self_34491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'self', False)
        # Obtaining the member '_run' of a type (line 94)
        _run_34492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 14), self_34491, '_run')
        # Calling _run(args, kwargs) (line 94)
        _run_call_result_34499 = invoke(stypy.reporting.localization.Localization(__file__, 94, 14), _run_34492, *[metadata_34493], **kwargs_34498)
        
        # Assigning a type to the variable 'cmd' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'cmd', _run_call_result_34499)
        
        # Call to assertEqual(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'cmd' (line 95)
        cmd_34502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'cmd', False)
        # Obtaining the member '_warnings' of a type (line 95)
        _warnings_34503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 25), cmd_34502, '_warnings')
        int_34504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 40), 'int')
        # Processing the call keyword arguments (line 95)
        kwargs_34505 = {}
        # Getting the type of 'self' (line 95)
        self_34500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 95)
        assertEqual_34501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_34500, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 95)
        assertEqual_call_result_34506 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), assertEqual_34501, *[_warnings_34503, int_34504], **kwargs_34505)
        
        
        # ################# End of 'test_check_restructuredtext(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_check_restructuredtext' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_34507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34507)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_check_restructuredtext'
        return stypy_return_type_34507


    @norecursion
    def test_check_restructuredtext_with_syntax_highlight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_check_restructuredtext_with_syntax_highlight'
        module_type_store = module_type_store.open_function_context('test_check_restructuredtext_with_syntax_highlight', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CheckTestCase.test_check_restructuredtext_with_syntax_highlight.__dict__.__setitem__('stypy_localization', localization)
        CheckTestCase.test_check_restructuredtext_with_syntax_highlight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CheckTestCase.test_check_restructuredtext_with_syntax_highlight.__dict__.__setitem__('stypy_type_store', module_type_store)
        CheckTestCase.test_check_restructuredtext_with_syntax_highlight.__dict__.__setitem__('stypy_function_name', 'CheckTestCase.test_check_restructuredtext_with_syntax_highlight')
        CheckTestCase.test_check_restructuredtext_with_syntax_highlight.__dict__.__setitem__('stypy_param_names_list', [])
        CheckTestCase.test_check_restructuredtext_with_syntax_highlight.__dict__.__setitem__('stypy_varargs_param_name', None)
        CheckTestCase.test_check_restructuredtext_with_syntax_highlight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CheckTestCase.test_check_restructuredtext_with_syntax_highlight.__dict__.__setitem__('stypy_call_defaults', defaults)
        CheckTestCase.test_check_restructuredtext_with_syntax_highlight.__dict__.__setitem__('stypy_call_varargs', varargs)
        CheckTestCase.test_check_restructuredtext_with_syntax_highlight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CheckTestCase.test_check_restructuredtext_with_syntax_highlight.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CheckTestCase.test_check_restructuredtext_with_syntax_highlight', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_check_restructuredtext_with_syntax_highlight', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_check_restructuredtext_with_syntax_highlight(...)' code ##################

        
        # Assigning a List to a Name (line 101):
        
        # Assigning a List to a Name (line 101):
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_34508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        
        # Assigning a type to the variable 'example_rst_docs' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'example_rst_docs', list_34508)
        
        # Call to append(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to dedent(...): (line 102)
        # Processing the call arguments (line 102)
        str_34513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', "            Here's some code:\n\n            .. code:: python\n\n                def foo():\n                    pass\n            ")
        # Processing the call keyword arguments (line 102)
        kwargs_34514 = {}
        # Getting the type of 'textwrap' (line 102)
        textwrap_34511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 32), 'textwrap', False)
        # Obtaining the member 'dedent' of a type (line 102)
        dedent_34512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 32), textwrap_34511, 'dedent')
        # Calling dedent(args, kwargs) (line 102)
        dedent_call_result_34515 = invoke(stypy.reporting.localization.Localization(__file__, 102, 32), dedent_34512, *[str_34513], **kwargs_34514)
        
        # Processing the call keyword arguments (line 102)
        kwargs_34516 = {}
        # Getting the type of 'example_rst_docs' (line 102)
        example_rst_docs_34509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'example_rst_docs', False)
        # Obtaining the member 'append' of a type (line 102)
        append_34510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), example_rst_docs_34509, 'append')
        # Calling append(args, kwargs) (line 102)
        append_call_result_34517 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), append_34510, *[dedent_call_result_34515], **kwargs_34516)
        
        
        # Call to append(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Call to dedent(...): (line 110)
        # Processing the call arguments (line 110)
        str_34522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', "            Here's some code:\n\n            .. code-block:: python\n\n                def foo():\n                    pass\n            ")
        # Processing the call keyword arguments (line 110)
        kwargs_34523 = {}
        # Getting the type of 'textwrap' (line 110)
        textwrap_34520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 32), 'textwrap', False)
        # Obtaining the member 'dedent' of a type (line 110)
        dedent_34521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 32), textwrap_34520, 'dedent')
        # Calling dedent(args, kwargs) (line 110)
        dedent_call_result_34524 = invoke(stypy.reporting.localization.Localization(__file__, 110, 32), dedent_34521, *[str_34522], **kwargs_34523)
        
        # Processing the call keyword arguments (line 110)
        kwargs_34525 = {}
        # Getting the type of 'example_rst_docs' (line 110)
        example_rst_docs_34518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'example_rst_docs', False)
        # Obtaining the member 'append' of a type (line 110)
        append_34519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), example_rst_docs_34518, 'append')
        # Calling append(args, kwargs) (line 110)
        append_call_result_34526 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), append_34519, *[dedent_call_result_34524], **kwargs_34525)
        
        
        # Getting the type of 'example_rst_docs' (line 119)
        example_rst_docs_34527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 30), 'example_rst_docs')
        # Testing the type of a for loop iterable (line 119)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 119, 8), example_rst_docs_34527)
        # Getting the type of the for loop variable (line 119)
        for_loop_var_34528 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 119, 8), example_rst_docs_34527)
        # Assigning a type to the variable 'rest_with_code' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'rest_with_code', for_loop_var_34528)
        # SSA begins for a for statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 120):
        
        # Assigning a Subscript to a Name (line 120):
        
        # Obtaining the type of the subscript
        int_34529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 12), 'int')
        
        # Call to create_dist(...): (line 120)
        # Processing the call keyword arguments (line 120)
        # Getting the type of 'rest_with_code' (line 120)
        rest_with_code_34532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 63), 'rest_with_code', False)
        keyword_34533 = rest_with_code_34532
        kwargs_34534 = {'long_description': keyword_34533}
        # Getting the type of 'self' (line 120)
        self_34530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 120)
        create_dist_34531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 29), self_34530, 'create_dist')
        # Calling create_dist(args, kwargs) (line 120)
        create_dist_call_result_34535 = invoke(stypy.reporting.localization.Localization(__file__, 120, 29), create_dist_34531, *[], **kwargs_34534)
        
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___34536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), create_dist_call_result_34535, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_34537 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), getitem___34536, int_34529)
        
        # Assigning a type to the variable 'tuple_var_assignment_34218' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'tuple_var_assignment_34218', subscript_call_result_34537)
        
        # Assigning a Subscript to a Name (line 120):
        
        # Obtaining the type of the subscript
        int_34538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 12), 'int')
        
        # Call to create_dist(...): (line 120)
        # Processing the call keyword arguments (line 120)
        # Getting the type of 'rest_with_code' (line 120)
        rest_with_code_34541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 63), 'rest_with_code', False)
        keyword_34542 = rest_with_code_34541
        kwargs_34543 = {'long_description': keyword_34542}
        # Getting the type of 'self' (line 120)
        self_34539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 120)
        create_dist_34540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 29), self_34539, 'create_dist')
        # Calling create_dist(args, kwargs) (line 120)
        create_dist_call_result_34544 = invoke(stypy.reporting.localization.Localization(__file__, 120, 29), create_dist_34540, *[], **kwargs_34543)
        
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___34545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), create_dist_call_result_34544, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_34546 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), getitem___34545, int_34538)
        
        # Assigning a type to the variable 'tuple_var_assignment_34219' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'tuple_var_assignment_34219', subscript_call_result_34546)
        
        # Assigning a Name to a Name (line 120):
        # Getting the type of 'tuple_var_assignment_34218' (line 120)
        tuple_var_assignment_34218_34547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'tuple_var_assignment_34218')
        # Assigning a type to the variable 'pkg_info' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'pkg_info', tuple_var_assignment_34218_34547)
        
        # Assigning a Name to a Name (line 120):
        # Getting the type of 'tuple_var_assignment_34219' (line 120)
        tuple_var_assignment_34219_34548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'tuple_var_assignment_34219')
        # Assigning a type to the variable 'dist' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'dist', tuple_var_assignment_34219_34548)
        
        # Assigning a Call to a Name (line 121):
        
        # Assigning a Call to a Name (line 121):
        
        # Call to check(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'dist' (line 121)
        dist_34550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), 'dist', False)
        # Processing the call keyword arguments (line 121)
        kwargs_34551 = {}
        # Getting the type of 'check' (line 121)
        check_34549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 18), 'check', False)
        # Calling check(args, kwargs) (line 121)
        check_call_result_34552 = invoke(stypy.reporting.localization.Localization(__file__, 121, 18), check_34549, *[dist_34550], **kwargs_34551)
        
        # Assigning a type to the variable 'cmd' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'cmd', check_call_result_34552)
        
        # Call to check_restructuredtext(...): (line 122)
        # Processing the call keyword arguments (line 122)
        kwargs_34555 = {}
        # Getting the type of 'cmd' (line 122)
        cmd_34553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'cmd', False)
        # Obtaining the member 'check_restructuredtext' of a type (line 122)
        check_restructuredtext_34554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), cmd_34553, 'check_restructuredtext')
        # Calling check_restructuredtext(args, kwargs) (line 122)
        check_restructuredtext_call_result_34556 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), check_restructuredtext_34554, *[], **kwargs_34555)
        
        
        # Call to assertEqual(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'cmd' (line 123)
        cmd_34559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 29), 'cmd', False)
        # Obtaining the member '_warnings' of a type (line 123)
        _warnings_34560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 29), cmd_34559, '_warnings')
        int_34561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 44), 'int')
        # Processing the call keyword arguments (line 123)
        kwargs_34562 = {}
        # Getting the type of 'self' (line 123)
        self_34557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 123)
        assertEqual_34558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), self_34557, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 123)
        assertEqual_call_result_34563 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), assertEqual_34558, *[_warnings_34560, int_34561], **kwargs_34562)
        
        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to _check_rst_data(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'rest_with_code' (line 124)
        rest_with_code_34566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 39), 'rest_with_code', False)
        # Processing the call keyword arguments (line 124)
        kwargs_34567 = {}
        # Getting the type of 'cmd' (line 124)
        cmd_34564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'cmd', False)
        # Obtaining the member '_check_rst_data' of a type (line 124)
        _check_rst_data_34565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 19), cmd_34564, '_check_rst_data')
        # Calling _check_rst_data(args, kwargs) (line 124)
        _check_rst_data_call_result_34568 = invoke(stypy.reporting.localization.Localization(__file__, 124, 19), _check_rst_data_34565, *[rest_with_code_34566], **kwargs_34567)
        
        # Assigning a type to the variable 'msgs' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'msgs', _check_rst_data_call_result_34568)
        
        # Call to assertEqual(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Call to len(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'msgs' (line 125)
        msgs_34572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 33), 'msgs', False)
        # Processing the call keyword arguments (line 125)
        kwargs_34573 = {}
        # Getting the type of 'len' (line 125)
        len_34571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 29), 'len', False)
        # Calling len(args, kwargs) (line 125)
        len_call_result_34574 = invoke(stypy.reporting.localization.Localization(__file__, 125, 29), len_34571, *[msgs_34572], **kwargs_34573)
        
        int_34575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 40), 'int')
        # Processing the call keyword arguments (line 125)
        kwargs_34576 = {}
        # Getting the type of 'self' (line 125)
        self_34569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 125)
        assertEqual_34570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), self_34569, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 125)
        assertEqual_call_result_34577 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), assertEqual_34570, *[len_call_result_34574, int_34575], **kwargs_34576)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_check_restructuredtext_with_syntax_highlight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_check_restructuredtext_with_syntax_highlight' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_34578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34578)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_check_restructuredtext_with_syntax_highlight'
        return stypy_return_type_34578


    @norecursion
    def test_check_all(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_check_all'
        module_type_store = module_type_store.open_function_context('test_check_all', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CheckTestCase.test_check_all.__dict__.__setitem__('stypy_localization', localization)
        CheckTestCase.test_check_all.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CheckTestCase.test_check_all.__dict__.__setitem__('stypy_type_store', module_type_store)
        CheckTestCase.test_check_all.__dict__.__setitem__('stypy_function_name', 'CheckTestCase.test_check_all')
        CheckTestCase.test_check_all.__dict__.__setitem__('stypy_param_names_list', [])
        CheckTestCase.test_check_all.__dict__.__setitem__('stypy_varargs_param_name', None)
        CheckTestCase.test_check_all.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CheckTestCase.test_check_all.__dict__.__setitem__('stypy_call_defaults', defaults)
        CheckTestCase.test_check_all.__dict__.__setitem__('stypy_call_varargs', varargs)
        CheckTestCase.test_check_all.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CheckTestCase.test_check_all.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CheckTestCase.test_check_all', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_check_all', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_check_all(...)' code ##################

        
        # Assigning a Dict to a Name (line 129):
        
        # Assigning a Dict to a Name (line 129):
        
        # Obtaining an instance of the builtin type 'dict' (line 129)
        dict_34579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 129)
        # Adding element type (key, value) (line 129)
        str_34580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 20), 'str', 'url')
        str_34581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 27), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 19), dict_34579, (str_34580, str_34581))
        # Adding element type (key, value) (line 129)
        str_34582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 34), 'str', 'author')
        str_34583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 44), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 19), dict_34579, (str_34582, str_34583))
        
        # Assigning a type to the variable 'metadata' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'metadata', dict_34579)
        
        # Call to assertRaises(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'DistutilsSetupError' (line 130)
        DistutilsSetupError_34586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 26), 'DistutilsSetupError', False)
        # Getting the type of 'self' (line 130)
        self_34587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'self', False)
        # Obtaining the member '_run' of a type (line 130)
        _run_34588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 47), self_34587, '_run')
        
        # Obtaining an instance of the builtin type 'dict' (line 131)
        dict_34589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 26), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 131)
        
        # Processing the call keyword arguments (line 130)
        
        # Obtaining an instance of the builtin type 'dict' (line 131)
        dict_34590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 32), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 131)
        # Adding element type (key, value) (line 131)
        str_34591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 33), 'str', 'strict')
        int_34592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 43), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 32), dict_34590, (str_34591, int_34592))
        # Adding element type (key, value) (line 131)
        str_34593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 33), 'str', 'restructuredtext')
        int_34594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 53), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 32), dict_34590, (str_34593, int_34594))
        
        kwargs_34595 = {'dict_34590': dict_34590}
        # Getting the type of 'self' (line 130)
        self_34584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 130)
        assertRaises_34585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_34584, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 130)
        assertRaises_call_result_34596 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), assertRaises_34585, *[DistutilsSetupError_34586, _run_34588, dict_34589], **kwargs_34595)
        
        
        # ################# End of 'test_check_all(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_check_all' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_34597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34597)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_check_all'
        return stypy_return_type_34597


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CheckTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CheckTestCase' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'CheckTestCase', CheckTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 134, 0, False)
    
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

    
    # Call to makeSuite(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'CheckTestCase' (line 135)
    CheckTestCase_34600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 30), 'CheckTestCase', False)
    # Processing the call keyword arguments (line 135)
    kwargs_34601 = {}
    # Getting the type of 'unittest' (line 135)
    unittest_34598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 135)
    makeSuite_34599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), unittest_34598, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 135)
    makeSuite_call_result_34602 = invoke(stypy.reporting.localization.Localization(__file__, 135, 11), makeSuite_34599, *[CheckTestCase_34600], **kwargs_34601)
    
    # Assigning a type to the variable 'stypy_return_type' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type', makeSuite_call_result_34602)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 134)
    stypy_return_type_34603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34603)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_34603

# Assigning a type to the variable 'test_suite' (line 134)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 138)
    # Processing the call arguments (line 138)
    
    # Call to test_suite(...): (line 138)
    # Processing the call keyword arguments (line 138)
    kwargs_34606 = {}
    # Getting the type of 'test_suite' (line 138)
    test_suite_34605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 138)
    test_suite_call_result_34607 = invoke(stypy.reporting.localization.Localization(__file__, 138, 17), test_suite_34605, *[], **kwargs_34606)
    
    # Processing the call keyword arguments (line 138)
    kwargs_34608 = {}
    # Getting the type of 'run_unittest' (line 138)
    run_unittest_34604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 138)
    run_unittest_call_result_34609 = invoke(stypy.reporting.localization.Localization(__file__, 138, 4), run_unittest_34604, *[test_suite_call_result_34607], **kwargs_34608)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
