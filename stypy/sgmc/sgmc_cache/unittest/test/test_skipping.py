
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import unittest
2: 
3: from unittest.test.support import LoggingResult
4: 
5: 
6: class Test_TestSkipping(unittest.TestCase):
7: 
8:     def test_skipping(self):
9:         class Foo(unittest.TestCase):
10:             def test_skip_me(self):
11:                 self.skipTest("skip")
12:         events = []
13:         result = LoggingResult(events)
14:         test = Foo("test_skip_me")
15:         test.run(result)
16:         self.assertEqual(events, ['startTest', 'addSkip', 'stopTest'])
17:         self.assertEqual(result.skipped, [(test, "skip")])
18: 
19:         # Try letting setUp skip the test now.
20:         class Foo(unittest.TestCase):
21:             def setUp(self):
22:                 self.skipTest("testing")
23:             def test_nothing(self): pass
24:         events = []
25:         result = LoggingResult(events)
26:         test = Foo("test_nothing")
27:         test.run(result)
28:         self.assertEqual(events, ['startTest', 'addSkip', 'stopTest'])
29:         self.assertEqual(result.skipped, [(test, "testing")])
30:         self.assertEqual(result.testsRun, 1)
31: 
32:     def test_skipping_decorators(self):
33:         op_table = ((unittest.skipUnless, False, True),
34:                     (unittest.skipIf, True, False))
35:         for deco, do_skip, dont_skip in op_table:
36:             class Foo(unittest.TestCase):
37:                 @deco(do_skip, "testing")
38:                 def test_skip(self): pass
39: 
40:                 @deco(dont_skip, "testing")
41:                 def test_dont_skip(self): pass
42:             test_do_skip = Foo("test_skip")
43:             test_dont_skip = Foo("test_dont_skip")
44:             suite = unittest.TestSuite([test_do_skip, test_dont_skip])
45:             events = []
46:             result = LoggingResult(events)
47:             suite.run(result)
48:             self.assertEqual(len(result.skipped), 1)
49:             expected = ['startTest', 'addSkip', 'stopTest',
50:                         'startTest', 'addSuccess', 'stopTest']
51:             self.assertEqual(events, expected)
52:             self.assertEqual(result.testsRun, 2)
53:             self.assertEqual(result.skipped, [(test_do_skip, "testing")])
54:             self.assertTrue(result.wasSuccessful())
55: 
56:     def test_skip_class(self):
57:         @unittest.skip("testing")
58:         class Foo(unittest.TestCase):
59:             def test_1(self):
60:                 record.append(1)
61:         record = []
62:         result = unittest.TestResult()
63:         test = Foo("test_1")
64:         suite = unittest.TestSuite([test])
65:         suite.run(result)
66:         self.assertEqual(result.skipped, [(test, "testing")])
67:         self.assertEqual(record, [])
68: 
69:     def test_skip_non_unittest_class_old_style(self):
70:         @unittest.skip("testing")
71:         class Mixin:
72:             def test_1(self):
73:                 record.append(1)
74:         class Foo(Mixin, unittest.TestCase):
75:             pass
76:         record = []
77:         result = unittest.TestResult()
78:         test = Foo("test_1")
79:         suite = unittest.TestSuite([test])
80:         suite.run(result)
81:         self.assertEqual(result.skipped, [(test, "testing")])
82:         self.assertEqual(record, [])
83: 
84:     def test_skip_non_unittest_class_new_style(self):
85:         @unittest.skip("testing")
86:         class Mixin(object):
87:             def test_1(self):
88:                 record.append(1)
89:         class Foo(Mixin, unittest.TestCase):
90:             pass
91:         record = []
92:         result = unittest.TestResult()
93:         test = Foo("test_1")
94:         suite = unittest.TestSuite([test])
95:         suite.run(result)
96:         self.assertEqual(result.skipped, [(test, "testing")])
97:         self.assertEqual(record, [])
98: 
99:     def test_expected_failure(self):
100:         class Foo(unittest.TestCase):
101:             @unittest.expectedFailure
102:             def test_die(self):
103:                 self.fail("help me!")
104:         events = []
105:         result = LoggingResult(events)
106:         test = Foo("test_die")
107:         test.run(result)
108:         self.assertEqual(events,
109:                          ['startTest', 'addExpectedFailure', 'stopTest'])
110:         self.assertEqual(result.expectedFailures[0][0], test)
111:         self.assertTrue(result.wasSuccessful())
112: 
113:     def test_unexpected_success(self):
114:         class Foo(unittest.TestCase):
115:             @unittest.expectedFailure
116:             def test_die(self):
117:                 pass
118:         events = []
119:         result = LoggingResult(events)
120:         test = Foo("test_die")
121:         test.run(result)
122:         self.assertEqual(events,
123:                          ['startTest', 'addUnexpectedSuccess', 'stopTest'])
124:         self.assertFalse(result.failures)
125:         self.assertEqual(result.unexpectedSuccesses, [test])
126:         self.assertTrue(result.wasSuccessful())
127: 
128:     def test_skip_doesnt_run_setup(self):
129:         class Foo(unittest.TestCase):
130:             wasSetUp = False
131:             wasTornDown = False
132:             def setUp(self):
133:                 Foo.wasSetUp = True
134:             def tornDown(self):
135:                 Foo.wasTornDown = True
136:             @unittest.skip('testing')
137:             def test_1(self):
138:                 pass
139: 
140:         result = unittest.TestResult()
141:         test = Foo("test_1")
142:         suite = unittest.TestSuite([test])
143:         suite.run(result)
144:         self.assertEqual(result.skipped, [(test, "testing")])
145:         self.assertFalse(Foo.wasSetUp)
146:         self.assertFalse(Foo.wasTornDown)
147: 
148:     def test_decorated_skip(self):
149:         def decorator(func):
150:             def inner(*a):
151:                 return func(*a)
152:             return inner
153: 
154:         class Foo(unittest.TestCase):
155:             @decorator
156:             @unittest.skip('testing')
157:             def test_1(self):
158:                 pass
159: 
160:         result = unittest.TestResult()
161:         test = Foo("test_1")
162:         suite = unittest.TestSuite([test])
163:         suite.run(result)
164:         self.assertEqual(result.skipped, [(test, "testing")])
165: 
166: 
167: if __name__ == '__main__':
168:     unittest.main()
169: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import unittest' statement (line 1)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from unittest.test.support import LoggingResult' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/unittest/test/')
import_207614 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest.test.support')

if (type(import_207614) is not StypyTypeError):

    if (import_207614 != 'pyd_module'):
        __import__(import_207614)
        sys_modules_207615 = sys.modules[import_207614]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest.test.support', sys_modules_207615.module_type_store, module_type_store, ['LoggingResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_207615, sys_modules_207615.module_type_store, module_type_store)
    else:
        from unittest.test.support import LoggingResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest.test.support', None, module_type_store, ['LoggingResult'], [LoggingResult])

else:
    # Assigning a type to the variable 'unittest.test.support' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest.test.support', import_207614)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/test/')

# Declaration of the 'Test_TestSkipping' class
# Getting the type of 'unittest' (line 6)
unittest_207616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 24), 'unittest')
# Obtaining the member 'TestCase' of a type (line 6)
TestCase_207617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 24), unittest_207616, 'TestCase')

class Test_TestSkipping(TestCase_207617, ):

    @norecursion
    def test_skipping(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_skipping'
        module_type_store = module_type_store.open_function_context('test_skipping', 8, 4, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSkipping.test_skipping.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSkipping.test_skipping.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSkipping.test_skipping.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSkipping.test_skipping.__dict__.__setitem__('stypy_function_name', 'Test_TestSkipping.test_skipping')
        Test_TestSkipping.test_skipping.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSkipping.test_skipping.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSkipping.test_skipping.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSkipping.test_skipping.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSkipping.test_skipping.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSkipping.test_skipping.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSkipping.test_skipping.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSkipping.test_skipping', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_skipping', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_skipping(...)' code ##################

        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 9)
        unittest_207618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 9)
        TestCase_207619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 18), unittest_207618, 'TestCase')

        class Foo(TestCase_207619, ):

            @norecursion
            def test_skip_me(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_skip_me'
                module_type_store = module_type_store.open_function_context('test_skip_me', 10, 12, False)
                # Assigning a type to the variable 'self' (line 11)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_skip_me.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_skip_me.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_skip_me.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_skip_me.__dict__.__setitem__('stypy_function_name', 'Foo.test_skip_me')
                Foo.test_skip_me.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_skip_me.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_skip_me.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_skip_me.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_skip_me.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_skip_me.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_skip_me.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_skip_me', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_skip_me', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_skip_me(...)' code ##################

                
                # Call to skipTest(...): (line 11)
                # Processing the call arguments (line 11)
                str_207622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 30), 'str', 'skip')
                # Processing the call keyword arguments (line 11)
                kwargs_207623 = {}
                # Getting the type of 'self' (line 11)
                self_207620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'self', False)
                # Obtaining the member 'skipTest' of a type (line 11)
                skipTest_207621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 16), self_207620, 'skipTest')
                # Calling skipTest(args, kwargs) (line 11)
                skipTest_call_result_207624 = invoke(stypy.reporting.localization.Localization(__file__, 11, 16), skipTest_207621, *[str_207622], **kwargs_207623)
                
                
                # ################# End of 'test_skip_me(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_skip_me' in the type store
                # Getting the type of 'stypy_return_type' (line 10)
                stypy_return_type_207625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207625)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_skip_me'
                return stypy_return_type_207625

        
        # Assigning a type to the variable 'Foo' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'Foo', Foo)
        
        # Assigning a List to a Name (line 12):
        
        # Obtaining an instance of the builtin type 'list' (line 12)
        list_207626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 12)
        
        # Assigning a type to the variable 'events' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'events', list_207626)
        
        # Assigning a Call to a Name (line 13):
        
        # Call to LoggingResult(...): (line 13)
        # Processing the call arguments (line 13)
        # Getting the type of 'events' (line 13)
        events_207628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 31), 'events', False)
        # Processing the call keyword arguments (line 13)
        kwargs_207629 = {}
        # Getting the type of 'LoggingResult' (line 13)
        LoggingResult_207627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 17), 'LoggingResult', False)
        # Calling LoggingResult(args, kwargs) (line 13)
        LoggingResult_call_result_207630 = invoke(stypy.reporting.localization.Localization(__file__, 13, 17), LoggingResult_207627, *[events_207628], **kwargs_207629)
        
        # Assigning a type to the variable 'result' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'result', LoggingResult_call_result_207630)
        
        # Assigning a Call to a Name (line 14):
        
        # Call to Foo(...): (line 14)
        # Processing the call arguments (line 14)
        str_207632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 19), 'str', 'test_skip_me')
        # Processing the call keyword arguments (line 14)
        kwargs_207633 = {}
        # Getting the type of 'Foo' (line 14)
        Foo_207631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 14)
        Foo_call_result_207634 = invoke(stypy.reporting.localization.Localization(__file__, 14, 15), Foo_207631, *[str_207632], **kwargs_207633)
        
        # Assigning a type to the variable 'test' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'test', Foo_call_result_207634)
        
        # Call to run(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'result' (line 15)
        result_207637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 17), 'result', False)
        # Processing the call keyword arguments (line 15)
        kwargs_207638 = {}
        # Getting the type of 'test' (line 15)
        test_207635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'test', False)
        # Obtaining the member 'run' of a type (line 15)
        run_207636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), test_207635, 'run')
        # Calling run(args, kwargs) (line 15)
        run_call_result_207639 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), run_207636, *[result_207637], **kwargs_207638)
        
        
        # Call to assertEqual(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'events' (line 16)
        events_207642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 25), 'events', False)
        
        # Obtaining an instance of the builtin type 'list' (line 16)
        list_207643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 16)
        # Adding element type (line 16)
        str_207644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 34), 'str', 'startTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 33), list_207643, str_207644)
        # Adding element type (line 16)
        str_207645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 47), 'str', 'addSkip')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 33), list_207643, str_207645)
        # Adding element type (line 16)
        str_207646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 58), 'str', 'stopTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 33), list_207643, str_207646)
        
        # Processing the call keyword arguments (line 16)
        kwargs_207647 = {}
        # Getting the type of 'self' (line 16)
        self_207640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 16)
        assertEqual_207641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_207640, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 16)
        assertEqual_call_result_207648 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), assertEqual_207641, *[events_207642, list_207643], **kwargs_207647)
        
        
        # Call to assertEqual(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'result' (line 17)
        result_207651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), 'result', False)
        # Obtaining the member 'skipped' of a type (line 17)
        skipped_207652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 25), result_207651, 'skipped')
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_207653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        
        # Obtaining an instance of the builtin type 'tuple' (line 17)
        tuple_207654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 17)
        # Adding element type (line 17)
        # Getting the type of 'test' (line 17)
        test_207655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 43), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 43), tuple_207654, test_207655)
        # Adding element type (line 17)
        str_207656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 49), 'str', 'skip')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 43), tuple_207654, str_207656)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 41), list_207653, tuple_207654)
        
        # Processing the call keyword arguments (line 17)
        kwargs_207657 = {}
        # Getting the type of 'self' (line 17)
        self_207649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 17)
        assertEqual_207650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), self_207649, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 17)
        assertEqual_call_result_207658 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), assertEqual_207650, *[skipped_207652, list_207653], **kwargs_207657)
        
        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 20)
        unittest_207659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 20)
        TestCase_207660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 18), unittest_207659, 'TestCase')

        class Foo(TestCase_207660, ):

            @norecursion
            def setUp(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUp'
                module_type_store = module_type_store.open_function_context('setUp', 21, 12, False)
                # Assigning a type to the variable 'self' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.setUp.__dict__.__setitem__('stypy_localization', localization)
                Foo.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.setUp.__dict__.__setitem__('stypy_function_name', 'Foo.setUp')
                Foo.setUp.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.setUp', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to skipTest(...): (line 22)
                # Processing the call arguments (line 22)
                str_207663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'str', 'testing')
                # Processing the call keyword arguments (line 22)
                kwargs_207664 = {}
                # Getting the type of 'self' (line 22)
                self_207661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'self', False)
                # Obtaining the member 'skipTest' of a type (line 22)
                skipTest_207662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 16), self_207661, 'skipTest')
                # Calling skipTest(args, kwargs) (line 22)
                skipTest_call_result_207665 = invoke(stypy.reporting.localization.Localization(__file__, 22, 16), skipTest_207662, *[str_207663], **kwargs_207664)
                
                
                # ################# End of 'setUp(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUp' in the type store
                # Getting the type of 'stypy_return_type' (line 21)
                stypy_return_type_207666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207666)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUp'
                return stypy_return_type_207666


            @norecursion
            def test_nothing(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_nothing'
                module_type_store = module_type_store.open_function_context('test_nothing', 23, 12, False)
                # Assigning a type to the variable 'self' (line 24)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_nothing.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_nothing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_nothing.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_nothing.__dict__.__setitem__('stypy_function_name', 'Foo.test_nothing')
                Foo.test_nothing.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_nothing.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_nothing.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_nothing.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_nothing.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_nothing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_nothing.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_nothing', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_nothing', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_nothing(...)' code ##################

                pass
                
                # ################# End of 'test_nothing(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_nothing' in the type store
                # Getting the type of 'stypy_return_type' (line 23)
                stypy_return_type_207667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207667)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_nothing'
                return stypy_return_type_207667

        
        # Assigning a type to the variable 'Foo' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'Foo', Foo)
        
        # Assigning a List to a Name (line 24):
        
        # Obtaining an instance of the builtin type 'list' (line 24)
        list_207668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 24)
        
        # Assigning a type to the variable 'events' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'events', list_207668)
        
        # Assigning a Call to a Name (line 25):
        
        # Call to LoggingResult(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'events' (line 25)
        events_207670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'events', False)
        # Processing the call keyword arguments (line 25)
        kwargs_207671 = {}
        # Getting the type of 'LoggingResult' (line 25)
        LoggingResult_207669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'LoggingResult', False)
        # Calling LoggingResult(args, kwargs) (line 25)
        LoggingResult_call_result_207672 = invoke(stypy.reporting.localization.Localization(__file__, 25, 17), LoggingResult_207669, *[events_207670], **kwargs_207671)
        
        # Assigning a type to the variable 'result' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'result', LoggingResult_call_result_207672)
        
        # Assigning a Call to a Name (line 26):
        
        # Call to Foo(...): (line 26)
        # Processing the call arguments (line 26)
        str_207674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'str', 'test_nothing')
        # Processing the call keyword arguments (line 26)
        kwargs_207675 = {}
        # Getting the type of 'Foo' (line 26)
        Foo_207673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 26)
        Foo_call_result_207676 = invoke(stypy.reporting.localization.Localization(__file__, 26, 15), Foo_207673, *[str_207674], **kwargs_207675)
        
        # Assigning a type to the variable 'test' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'test', Foo_call_result_207676)
        
        # Call to run(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'result' (line 27)
        result_207679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 17), 'result', False)
        # Processing the call keyword arguments (line 27)
        kwargs_207680 = {}
        # Getting the type of 'test' (line 27)
        test_207677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'test', False)
        # Obtaining the member 'run' of a type (line 27)
        run_207678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), test_207677, 'run')
        # Calling run(args, kwargs) (line 27)
        run_call_result_207681 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), run_207678, *[result_207679], **kwargs_207680)
        
        
        # Call to assertEqual(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'events' (line 28)
        events_207684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 25), 'events', False)
        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_207685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        # Adding element type (line 28)
        str_207686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'str', 'startTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 33), list_207685, str_207686)
        # Adding element type (line 28)
        str_207687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 47), 'str', 'addSkip')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 33), list_207685, str_207687)
        # Adding element type (line 28)
        str_207688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 58), 'str', 'stopTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 33), list_207685, str_207688)
        
        # Processing the call keyword arguments (line 28)
        kwargs_207689 = {}
        # Getting the type of 'self' (line 28)
        self_207682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 28)
        assertEqual_207683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_207682, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 28)
        assertEqual_call_result_207690 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), assertEqual_207683, *[events_207684, list_207685], **kwargs_207689)
        
        
        # Call to assertEqual(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'result' (line 29)
        result_207693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 25), 'result', False)
        # Obtaining the member 'skipped' of a type (line 29)
        skipped_207694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 25), result_207693, 'skipped')
        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_207695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        # Adding element type (line 29)
        
        # Obtaining an instance of the builtin type 'tuple' (line 29)
        tuple_207696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 29)
        # Adding element type (line 29)
        # Getting the type of 'test' (line 29)
        test_207697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 43), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 43), tuple_207696, test_207697)
        # Adding element type (line 29)
        str_207698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 49), 'str', 'testing')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 43), tuple_207696, str_207698)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 41), list_207695, tuple_207696)
        
        # Processing the call keyword arguments (line 29)
        kwargs_207699 = {}
        # Getting the type of 'self' (line 29)
        self_207691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 29)
        assertEqual_207692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_207691, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 29)
        assertEqual_call_result_207700 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), assertEqual_207692, *[skipped_207694, list_207695], **kwargs_207699)
        
        
        # Call to assertEqual(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'result' (line 30)
        result_207703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 30)
        testsRun_207704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 25), result_207703, 'testsRun')
        int_207705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 42), 'int')
        # Processing the call keyword arguments (line 30)
        kwargs_207706 = {}
        # Getting the type of 'self' (line 30)
        self_207701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 30)
        assertEqual_207702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_207701, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 30)
        assertEqual_call_result_207707 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assertEqual_207702, *[testsRun_207704, int_207705], **kwargs_207706)
        
        
        # ################# End of 'test_skipping(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_skipping' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_207708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207708)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_skipping'
        return stypy_return_type_207708


    @norecursion
    def test_skipping_decorators(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_skipping_decorators'
        module_type_store = module_type_store.open_function_context('test_skipping_decorators', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSkipping.test_skipping_decorators.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSkipping.test_skipping_decorators.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSkipping.test_skipping_decorators.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSkipping.test_skipping_decorators.__dict__.__setitem__('stypy_function_name', 'Test_TestSkipping.test_skipping_decorators')
        Test_TestSkipping.test_skipping_decorators.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSkipping.test_skipping_decorators.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSkipping.test_skipping_decorators.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSkipping.test_skipping_decorators.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSkipping.test_skipping_decorators.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSkipping.test_skipping_decorators.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSkipping.test_skipping_decorators.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSkipping.test_skipping_decorators', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_skipping_decorators', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_skipping_decorators(...)' code ##################

        
        # Assigning a Tuple to a Name (line 33):
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_207709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_207710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        # Getting the type of 'unittest' (line 33)
        unittest_207711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 21), 'unittest')
        # Obtaining the member 'skipUnless' of a type (line 33)
        skipUnless_207712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 21), unittest_207711, 'skipUnless')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), tuple_207710, skipUnless_207712)
        # Adding element type (line 33)
        # Getting the type of 'False' (line 33)
        False_207713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 42), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), tuple_207710, False_207713)
        # Adding element type (line 33)
        # Getting the type of 'True' (line 33)
        True_207714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 49), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), tuple_207710, True_207714)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), tuple_207709, tuple_207710)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'tuple' (line 34)
        tuple_207715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 34)
        # Adding element type (line 34)
        # Getting the type of 'unittest' (line 34)
        unittest_207716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'unittest')
        # Obtaining the member 'skipIf' of a type (line 34)
        skipIf_207717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), unittest_207716, 'skipIf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 21), tuple_207715, skipIf_207717)
        # Adding element type (line 34)
        # Getting the type of 'True' (line 34)
        True_207718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 38), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 21), tuple_207715, True_207718)
        # Adding element type (line 34)
        # Getting the type of 'False' (line 34)
        False_207719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 44), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 21), tuple_207715, False_207719)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), tuple_207709, tuple_207715)
        
        # Assigning a type to the variable 'op_table' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'op_table', tuple_207709)
        
        # Getting the type of 'op_table' (line 35)
        op_table_207720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 40), 'op_table')
        # Testing the type of a for loop iterable (line 35)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 8), op_table_207720)
        # Getting the type of the for loop variable (line 35)
        for_loop_var_207721 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 8), op_table_207720)
        # Assigning a type to the variable 'deco' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'deco', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 8), for_loop_var_207721))
        # Assigning a type to the variable 'do_skip' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'do_skip', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 8), for_loop_var_207721))
        # Assigning a type to the variable 'dont_skip' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'dont_skip', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 8), for_loop_var_207721))
        # SSA begins for a for statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 36)
        unittest_207722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 36)
        TestCase_207723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 22), unittest_207722, 'TestCase')

        class Foo(TestCase_207723, ):

            @norecursion
            def test_skip(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_skip'
                module_type_store = module_type_store.open_function_context('test_skip', 37, 16, False)
                # Assigning a type to the variable 'self' (line 38)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_skip.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_skip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_skip.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_skip.__dict__.__setitem__('stypy_function_name', 'Foo.test_skip')
                Foo.test_skip.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_skip.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_skip.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_skip.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_skip.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_skip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_skip.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_skip', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_skip', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_skip(...)' code ##################

                pass
                
                # ################# End of 'test_skip(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_skip' in the type store
                # Getting the type of 'stypy_return_type' (line 37)
                stypy_return_type_207724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207724)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_skip'
                return stypy_return_type_207724


            @norecursion
            def test_dont_skip(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_dont_skip'
                module_type_store = module_type_store.open_function_context('test_dont_skip', 40, 16, False)
                # Assigning a type to the variable 'self' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_dont_skip.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_dont_skip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_dont_skip.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_dont_skip.__dict__.__setitem__('stypy_function_name', 'Foo.test_dont_skip')
                Foo.test_dont_skip.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_dont_skip.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_dont_skip.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_dont_skip.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_dont_skip.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_dont_skip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_dont_skip.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_dont_skip', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_dont_skip', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_dont_skip(...)' code ##################

                pass
                
                # ################# End of 'test_dont_skip(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_dont_skip' in the type store
                # Getting the type of 'stypy_return_type' (line 40)
                stypy_return_type_207725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207725)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_dont_skip'
                return stypy_return_type_207725

        
        # Assigning a type to the variable 'Foo' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'Foo', Foo)
        
        # Assigning a Call to a Name (line 42):
        
        # Call to Foo(...): (line 42)
        # Processing the call arguments (line 42)
        str_207727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 31), 'str', 'test_skip')
        # Processing the call keyword arguments (line 42)
        kwargs_207728 = {}
        # Getting the type of 'Foo' (line 42)
        Foo_207726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'Foo', False)
        # Calling Foo(args, kwargs) (line 42)
        Foo_call_result_207729 = invoke(stypy.reporting.localization.Localization(__file__, 42, 27), Foo_207726, *[str_207727], **kwargs_207728)
        
        # Assigning a type to the variable 'test_do_skip' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'test_do_skip', Foo_call_result_207729)
        
        # Assigning a Call to a Name (line 43):
        
        # Call to Foo(...): (line 43)
        # Processing the call arguments (line 43)
        str_207731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 33), 'str', 'test_dont_skip')
        # Processing the call keyword arguments (line 43)
        kwargs_207732 = {}
        # Getting the type of 'Foo' (line 43)
        Foo_207730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 29), 'Foo', False)
        # Calling Foo(args, kwargs) (line 43)
        Foo_call_result_207733 = invoke(stypy.reporting.localization.Localization(__file__, 43, 29), Foo_207730, *[str_207731], **kwargs_207732)
        
        # Assigning a type to the variable 'test_dont_skip' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'test_dont_skip', Foo_call_result_207733)
        
        # Assigning a Call to a Name (line 44):
        
        # Call to TestSuite(...): (line 44)
        # Processing the call arguments (line 44)
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_207736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        # Adding element type (line 44)
        # Getting the type of 'test_do_skip' (line 44)
        test_do_skip_207737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 40), 'test_do_skip', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 39), list_207736, test_do_skip_207737)
        # Adding element type (line 44)
        # Getting the type of 'test_dont_skip' (line 44)
        test_dont_skip_207738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 54), 'test_dont_skip', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 39), list_207736, test_dont_skip_207738)
        
        # Processing the call keyword arguments (line 44)
        kwargs_207739 = {}
        # Getting the type of 'unittest' (line 44)
        unittest_207734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 44)
        TestSuite_207735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 20), unittest_207734, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 44)
        TestSuite_call_result_207740 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), TestSuite_207735, *[list_207736], **kwargs_207739)
        
        # Assigning a type to the variable 'suite' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'suite', TestSuite_call_result_207740)
        
        # Assigning a List to a Name (line 45):
        
        # Obtaining an instance of the builtin type 'list' (line 45)
        list_207741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 45)
        
        # Assigning a type to the variable 'events' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'events', list_207741)
        
        # Assigning a Call to a Name (line 46):
        
        # Call to LoggingResult(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'events' (line 46)
        events_207743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 35), 'events', False)
        # Processing the call keyword arguments (line 46)
        kwargs_207744 = {}
        # Getting the type of 'LoggingResult' (line 46)
        LoggingResult_207742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'LoggingResult', False)
        # Calling LoggingResult(args, kwargs) (line 46)
        LoggingResult_call_result_207745 = invoke(stypy.reporting.localization.Localization(__file__, 46, 21), LoggingResult_207742, *[events_207743], **kwargs_207744)
        
        # Assigning a type to the variable 'result' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'result', LoggingResult_call_result_207745)
        
        # Call to run(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'result' (line 47)
        result_207748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'result', False)
        # Processing the call keyword arguments (line 47)
        kwargs_207749 = {}
        # Getting the type of 'suite' (line 47)
        suite_207746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'suite', False)
        # Obtaining the member 'run' of a type (line 47)
        run_207747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), suite_207746, 'run')
        # Calling run(args, kwargs) (line 47)
        run_call_result_207750 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), run_207747, *[result_207748], **kwargs_207749)
        
        
        # Call to assertEqual(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to len(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'result' (line 48)
        result_207754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 33), 'result', False)
        # Obtaining the member 'skipped' of a type (line 48)
        skipped_207755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 33), result_207754, 'skipped')
        # Processing the call keyword arguments (line 48)
        kwargs_207756 = {}
        # Getting the type of 'len' (line 48)
        len_207753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 29), 'len', False)
        # Calling len(args, kwargs) (line 48)
        len_call_result_207757 = invoke(stypy.reporting.localization.Localization(__file__, 48, 29), len_207753, *[skipped_207755], **kwargs_207756)
        
        int_207758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 50), 'int')
        # Processing the call keyword arguments (line 48)
        kwargs_207759 = {}
        # Getting the type of 'self' (line 48)
        self_207751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 48)
        assertEqual_207752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), self_207751, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 48)
        assertEqual_call_result_207760 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), assertEqual_207752, *[len_call_result_207757, int_207758], **kwargs_207759)
        
        
        # Assigning a List to a Name (line 49):
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_207761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        # Adding element type (line 49)
        str_207762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 24), 'str', 'startTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 23), list_207761, str_207762)
        # Adding element type (line 49)
        str_207763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'str', 'addSkip')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 23), list_207761, str_207763)
        # Adding element type (line 49)
        str_207764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 48), 'str', 'stopTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 23), list_207761, str_207764)
        # Adding element type (line 49)
        str_207765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 24), 'str', 'startTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 23), list_207761, str_207765)
        # Adding element type (line 49)
        str_207766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 37), 'str', 'addSuccess')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 23), list_207761, str_207766)
        # Adding element type (line 49)
        str_207767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 51), 'str', 'stopTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 23), list_207761, str_207767)
        
        # Assigning a type to the variable 'expected' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'expected', list_207761)
        
        # Call to assertEqual(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'events' (line 51)
        events_207770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'events', False)
        # Getting the type of 'expected' (line 51)
        expected_207771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 37), 'expected', False)
        # Processing the call keyword arguments (line 51)
        kwargs_207772 = {}
        # Getting the type of 'self' (line 51)
        self_207768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 51)
        assertEqual_207769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_207768, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 51)
        assertEqual_call_result_207773 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), assertEqual_207769, *[events_207770, expected_207771], **kwargs_207772)
        
        
        # Call to assertEqual(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'result' (line 52)
        result_207776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 29), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 52)
        testsRun_207777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 29), result_207776, 'testsRun')
        int_207778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 46), 'int')
        # Processing the call keyword arguments (line 52)
        kwargs_207779 = {}
        # Getting the type of 'self' (line 52)
        self_207774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 52)
        assertEqual_207775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), self_207774, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 52)
        assertEqual_call_result_207780 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), assertEqual_207775, *[testsRun_207777, int_207778], **kwargs_207779)
        
        
        # Call to assertEqual(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'result' (line 53)
        result_207783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 29), 'result', False)
        # Obtaining the member 'skipped' of a type (line 53)
        skipped_207784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 29), result_207783, 'skipped')
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_207785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_207786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        # Getting the type of 'test_do_skip' (line 53)
        test_do_skip_207787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 47), 'test_do_skip', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 47), tuple_207786, test_do_skip_207787)
        # Adding element type (line 53)
        str_207788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 61), 'str', 'testing')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 47), tuple_207786, str_207788)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 45), list_207785, tuple_207786)
        
        # Processing the call keyword arguments (line 53)
        kwargs_207789 = {}
        # Getting the type of 'self' (line 53)
        self_207781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 53)
        assertEqual_207782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 12), self_207781, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 53)
        assertEqual_call_result_207790 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), assertEqual_207782, *[skipped_207784, list_207785], **kwargs_207789)
        
        
        # Call to assertTrue(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Call to wasSuccessful(...): (line 54)
        # Processing the call keyword arguments (line 54)
        kwargs_207795 = {}
        # Getting the type of 'result' (line 54)
        result_207793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'result', False)
        # Obtaining the member 'wasSuccessful' of a type (line 54)
        wasSuccessful_207794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 28), result_207793, 'wasSuccessful')
        # Calling wasSuccessful(args, kwargs) (line 54)
        wasSuccessful_call_result_207796 = invoke(stypy.reporting.localization.Localization(__file__, 54, 28), wasSuccessful_207794, *[], **kwargs_207795)
        
        # Processing the call keyword arguments (line 54)
        kwargs_207797 = {}
        # Getting the type of 'self' (line 54)
        self_207791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 54)
        assertTrue_207792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), self_207791, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 54)
        assertTrue_call_result_207798 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), assertTrue_207792, *[wasSuccessful_call_result_207796], **kwargs_207797)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_skipping_decorators(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_skipping_decorators' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_207799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207799)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_skipping_decorators'
        return stypy_return_type_207799


    @norecursion
    def test_skip_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_skip_class'
        module_type_store = module_type_store.open_function_context('test_skip_class', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSkipping.test_skip_class.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSkipping.test_skip_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSkipping.test_skip_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSkipping.test_skip_class.__dict__.__setitem__('stypy_function_name', 'Test_TestSkipping.test_skip_class')
        Test_TestSkipping.test_skip_class.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSkipping.test_skip_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSkipping.test_skip_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSkipping.test_skip_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSkipping.test_skip_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSkipping.test_skip_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSkipping.test_skip_class.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSkipping.test_skip_class', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_skip_class', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_skip_class(...)' code ##################

        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 58)
        unittest_207800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 58)
        TestCase_207801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 18), unittest_207800, 'TestCase')

        class Foo(TestCase_207801, ):

            @norecursion
            def test_1(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_1'
                module_type_store = module_type_store.open_function_context('test_1', 59, 12, False)
                # Assigning a type to the variable 'self' (line 60)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_1.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_1.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_1.__dict__.__setitem__('stypy_function_name', 'Foo.test_1')
                Foo.test_1.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_1.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_1.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_1', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_1', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_1(...)' code ##################

                
                # Call to append(...): (line 60)
                # Processing the call arguments (line 60)
                int_207804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 30), 'int')
                # Processing the call keyword arguments (line 60)
                kwargs_207805 = {}
                # Getting the type of 'record' (line 60)
                record_207802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'record', False)
                # Obtaining the member 'append' of a type (line 60)
                append_207803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), record_207802, 'append')
                # Calling append(args, kwargs) (line 60)
                append_call_result_207806 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), append_207803, *[int_207804], **kwargs_207805)
                
                
                # ################# End of 'test_1(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_1' in the type store
                # Getting the type of 'stypy_return_type' (line 59)
                stypy_return_type_207807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207807)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_1'
                return stypy_return_type_207807

        
        # Assigning a type to the variable 'Foo' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'Foo', Foo)
        
        # Assigning a List to a Name (line 61):
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_207808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        
        # Assigning a type to the variable 'record' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'record', list_207808)
        
        # Assigning a Call to a Name (line 62):
        
        # Call to TestResult(...): (line 62)
        # Processing the call keyword arguments (line 62)
        kwargs_207811 = {}
        # Getting the type of 'unittest' (line 62)
        unittest_207809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 62)
        TestResult_207810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 17), unittest_207809, 'TestResult')
        # Calling TestResult(args, kwargs) (line 62)
        TestResult_call_result_207812 = invoke(stypy.reporting.localization.Localization(__file__, 62, 17), TestResult_207810, *[], **kwargs_207811)
        
        # Assigning a type to the variable 'result' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'result', TestResult_call_result_207812)
        
        # Assigning a Call to a Name (line 63):
        
        # Call to Foo(...): (line 63)
        # Processing the call arguments (line 63)
        str_207814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 19), 'str', 'test_1')
        # Processing the call keyword arguments (line 63)
        kwargs_207815 = {}
        # Getting the type of 'Foo' (line 63)
        Foo_207813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 63)
        Foo_call_result_207816 = invoke(stypy.reporting.localization.Localization(__file__, 63, 15), Foo_207813, *[str_207814], **kwargs_207815)
        
        # Assigning a type to the variable 'test' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'test', Foo_call_result_207816)
        
        # Assigning a Call to a Name (line 64):
        
        # Call to TestSuite(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_207819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        # Getting the type of 'test' (line 64)
        test_207820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 36), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 35), list_207819, test_207820)
        
        # Processing the call keyword arguments (line 64)
        kwargs_207821 = {}
        # Getting the type of 'unittest' (line 64)
        unittest_207817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 64)
        TestSuite_207818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 16), unittest_207817, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 64)
        TestSuite_call_result_207822 = invoke(stypy.reporting.localization.Localization(__file__, 64, 16), TestSuite_207818, *[list_207819], **kwargs_207821)
        
        # Assigning a type to the variable 'suite' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'suite', TestSuite_call_result_207822)
        
        # Call to run(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'result' (line 65)
        result_207825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 18), 'result', False)
        # Processing the call keyword arguments (line 65)
        kwargs_207826 = {}
        # Getting the type of 'suite' (line 65)
        suite_207823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'suite', False)
        # Obtaining the member 'run' of a type (line 65)
        run_207824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), suite_207823, 'run')
        # Calling run(args, kwargs) (line 65)
        run_call_result_207827 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), run_207824, *[result_207825], **kwargs_207826)
        
        
        # Call to assertEqual(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'result' (line 66)
        result_207830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'result', False)
        # Obtaining the member 'skipped' of a type (line 66)
        skipped_207831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 25), result_207830, 'skipped')
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_207832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_207833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        # Adding element type (line 66)
        # Getting the type of 'test' (line 66)
        test_207834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 43), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 43), tuple_207833, test_207834)
        # Adding element type (line 66)
        str_207835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 49), 'str', 'testing')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 43), tuple_207833, str_207835)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 41), list_207832, tuple_207833)
        
        # Processing the call keyword arguments (line 66)
        kwargs_207836 = {}
        # Getting the type of 'self' (line 66)
        self_207828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 66)
        assertEqual_207829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_207828, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 66)
        assertEqual_call_result_207837 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assertEqual_207829, *[skipped_207831, list_207832], **kwargs_207836)
        
        
        # Call to assertEqual(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'record' (line 67)
        record_207840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'record', False)
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_207841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        
        # Processing the call keyword arguments (line 67)
        kwargs_207842 = {}
        # Getting the type of 'self' (line 67)
        self_207838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 67)
        assertEqual_207839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_207838, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 67)
        assertEqual_call_result_207843 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), assertEqual_207839, *[record_207840, list_207841], **kwargs_207842)
        
        
        # ################# End of 'test_skip_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_skip_class' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_207844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207844)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_skip_class'
        return stypy_return_type_207844


    @norecursion
    def test_skip_non_unittest_class_old_style(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_skip_non_unittest_class_old_style'
        module_type_store = module_type_store.open_function_context('test_skip_non_unittest_class_old_style', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSkipping.test_skip_non_unittest_class_old_style.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSkipping.test_skip_non_unittest_class_old_style.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSkipping.test_skip_non_unittest_class_old_style.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSkipping.test_skip_non_unittest_class_old_style.__dict__.__setitem__('stypy_function_name', 'Test_TestSkipping.test_skip_non_unittest_class_old_style')
        Test_TestSkipping.test_skip_non_unittest_class_old_style.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSkipping.test_skip_non_unittest_class_old_style.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSkipping.test_skip_non_unittest_class_old_style.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSkipping.test_skip_non_unittest_class_old_style.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSkipping.test_skip_non_unittest_class_old_style.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSkipping.test_skip_non_unittest_class_old_style.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSkipping.test_skip_non_unittest_class_old_style.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSkipping.test_skip_non_unittest_class_old_style', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_skip_non_unittest_class_old_style', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_skip_non_unittest_class_old_style(...)' code ##################

        # Declaration of the 'Mixin' class

        class Mixin:

            @norecursion
            def test_1(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_1'
                module_type_store = module_type_store.open_function_context('test_1', 72, 12, False)
                # Assigning a type to the variable 'self' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Mixin.test_1.__dict__.__setitem__('stypy_localization', localization)
                Mixin.test_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Mixin.test_1.__dict__.__setitem__('stypy_type_store', module_type_store)
                Mixin.test_1.__dict__.__setitem__('stypy_function_name', 'Mixin.test_1')
                Mixin.test_1.__dict__.__setitem__('stypy_param_names_list', [])
                Mixin.test_1.__dict__.__setitem__('stypy_varargs_param_name', None)
                Mixin.test_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Mixin.test_1.__dict__.__setitem__('stypy_call_defaults', defaults)
                Mixin.test_1.__dict__.__setitem__('stypy_call_varargs', varargs)
                Mixin.test_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Mixin.test_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Mixin.test_1', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_1', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_1(...)' code ##################

                
                # Call to append(...): (line 73)
                # Processing the call arguments (line 73)
                int_207847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 30), 'int')
                # Processing the call keyword arguments (line 73)
                kwargs_207848 = {}
                # Getting the type of 'record' (line 73)
                record_207845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'record', False)
                # Obtaining the member 'append' of a type (line 73)
                append_207846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), record_207845, 'append')
                # Calling append(args, kwargs) (line 73)
                append_call_result_207849 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), append_207846, *[int_207847], **kwargs_207848)
                
                
                # ################# End of 'test_1(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_1' in the type store
                # Getting the type of 'stypy_return_type' (line 72)
                stypy_return_type_207850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207850)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_1'
                return stypy_return_type_207850

        
        # Assigning a type to the variable 'Mixin' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'Mixin', Mixin)
        # Declaration of the 'Foo' class
        # Getting the type of 'Mixin' (line 74)
        Mixin_207851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'Mixin')
        # Getting the type of 'unittest' (line 74)
        unittest_207852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 74)
        TestCase_207853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 25), unittest_207852, 'TestCase')

        class Foo(Mixin_207851, TestCase_207853, ):
            pass
        
        # Assigning a type to the variable 'Foo' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'Foo', Foo)
        
        # Assigning a List to a Name (line 76):
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_207854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        
        # Assigning a type to the variable 'record' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'record', list_207854)
        
        # Assigning a Call to a Name (line 77):
        
        # Call to TestResult(...): (line 77)
        # Processing the call keyword arguments (line 77)
        kwargs_207857 = {}
        # Getting the type of 'unittest' (line 77)
        unittest_207855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 77)
        TestResult_207856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 17), unittest_207855, 'TestResult')
        # Calling TestResult(args, kwargs) (line 77)
        TestResult_call_result_207858 = invoke(stypy.reporting.localization.Localization(__file__, 77, 17), TestResult_207856, *[], **kwargs_207857)
        
        # Assigning a type to the variable 'result' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'result', TestResult_call_result_207858)
        
        # Assigning a Call to a Name (line 78):
        
        # Call to Foo(...): (line 78)
        # Processing the call arguments (line 78)
        str_207860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 19), 'str', 'test_1')
        # Processing the call keyword arguments (line 78)
        kwargs_207861 = {}
        # Getting the type of 'Foo' (line 78)
        Foo_207859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 78)
        Foo_call_result_207862 = invoke(stypy.reporting.localization.Localization(__file__, 78, 15), Foo_207859, *[str_207860], **kwargs_207861)
        
        # Assigning a type to the variable 'test' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'test', Foo_call_result_207862)
        
        # Assigning a Call to a Name (line 79):
        
        # Call to TestSuite(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_207865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        # Getting the type of 'test' (line 79)
        test_207866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 36), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 35), list_207865, test_207866)
        
        # Processing the call keyword arguments (line 79)
        kwargs_207867 = {}
        # Getting the type of 'unittest' (line 79)
        unittest_207863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 79)
        TestSuite_207864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 16), unittest_207863, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 79)
        TestSuite_call_result_207868 = invoke(stypy.reporting.localization.Localization(__file__, 79, 16), TestSuite_207864, *[list_207865], **kwargs_207867)
        
        # Assigning a type to the variable 'suite' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'suite', TestSuite_call_result_207868)
        
        # Call to run(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'result' (line 80)
        result_207871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'result', False)
        # Processing the call keyword arguments (line 80)
        kwargs_207872 = {}
        # Getting the type of 'suite' (line 80)
        suite_207869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'suite', False)
        # Obtaining the member 'run' of a type (line 80)
        run_207870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), suite_207869, 'run')
        # Calling run(args, kwargs) (line 80)
        run_call_result_207873 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), run_207870, *[result_207871], **kwargs_207872)
        
        
        # Call to assertEqual(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'result' (line 81)
        result_207876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 25), 'result', False)
        # Obtaining the member 'skipped' of a type (line 81)
        skipped_207877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 25), result_207876, 'skipped')
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_207878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        
        # Obtaining an instance of the builtin type 'tuple' (line 81)
        tuple_207879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 81)
        # Adding element type (line 81)
        # Getting the type of 'test' (line 81)
        test_207880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 43), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 43), tuple_207879, test_207880)
        # Adding element type (line 81)
        str_207881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 49), 'str', 'testing')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 43), tuple_207879, str_207881)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 41), list_207878, tuple_207879)
        
        # Processing the call keyword arguments (line 81)
        kwargs_207882 = {}
        # Getting the type of 'self' (line 81)
        self_207874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 81)
        assertEqual_207875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_207874, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 81)
        assertEqual_call_result_207883 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), assertEqual_207875, *[skipped_207877, list_207878], **kwargs_207882)
        
        
        # Call to assertEqual(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'record' (line 82)
        record_207886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'record', False)
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_207887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        
        # Processing the call keyword arguments (line 82)
        kwargs_207888 = {}
        # Getting the type of 'self' (line 82)
        self_207884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 82)
        assertEqual_207885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_207884, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 82)
        assertEqual_call_result_207889 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), assertEqual_207885, *[record_207886, list_207887], **kwargs_207888)
        
        
        # ################# End of 'test_skip_non_unittest_class_old_style(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_skip_non_unittest_class_old_style' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_207890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207890)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_skip_non_unittest_class_old_style'
        return stypy_return_type_207890


    @norecursion
    def test_skip_non_unittest_class_new_style(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_skip_non_unittest_class_new_style'
        module_type_store = module_type_store.open_function_context('test_skip_non_unittest_class_new_style', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSkipping.test_skip_non_unittest_class_new_style.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSkipping.test_skip_non_unittest_class_new_style.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSkipping.test_skip_non_unittest_class_new_style.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSkipping.test_skip_non_unittest_class_new_style.__dict__.__setitem__('stypy_function_name', 'Test_TestSkipping.test_skip_non_unittest_class_new_style')
        Test_TestSkipping.test_skip_non_unittest_class_new_style.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSkipping.test_skip_non_unittest_class_new_style.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSkipping.test_skip_non_unittest_class_new_style.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSkipping.test_skip_non_unittest_class_new_style.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSkipping.test_skip_non_unittest_class_new_style.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSkipping.test_skip_non_unittest_class_new_style.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSkipping.test_skip_non_unittest_class_new_style.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSkipping.test_skip_non_unittest_class_new_style', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_skip_non_unittest_class_new_style', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_skip_non_unittest_class_new_style(...)' code ##################

        # Declaration of the 'Mixin' class

        class Mixin(object, ):

            @norecursion
            def test_1(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_1'
                module_type_store = module_type_store.open_function_context('test_1', 87, 12, False)
                # Assigning a type to the variable 'self' (line 88)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Mixin.test_1.__dict__.__setitem__('stypy_localization', localization)
                Mixin.test_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Mixin.test_1.__dict__.__setitem__('stypy_type_store', module_type_store)
                Mixin.test_1.__dict__.__setitem__('stypy_function_name', 'Mixin.test_1')
                Mixin.test_1.__dict__.__setitem__('stypy_param_names_list', [])
                Mixin.test_1.__dict__.__setitem__('stypy_varargs_param_name', None)
                Mixin.test_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Mixin.test_1.__dict__.__setitem__('stypy_call_defaults', defaults)
                Mixin.test_1.__dict__.__setitem__('stypy_call_varargs', varargs)
                Mixin.test_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Mixin.test_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Mixin.test_1', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_1', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_1(...)' code ##################

                
                # Call to append(...): (line 88)
                # Processing the call arguments (line 88)
                int_207893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 30), 'int')
                # Processing the call keyword arguments (line 88)
                kwargs_207894 = {}
                # Getting the type of 'record' (line 88)
                record_207891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'record', False)
                # Obtaining the member 'append' of a type (line 88)
                append_207892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 16), record_207891, 'append')
                # Calling append(args, kwargs) (line 88)
                append_call_result_207895 = invoke(stypy.reporting.localization.Localization(__file__, 88, 16), append_207892, *[int_207893], **kwargs_207894)
                
                
                # ################# End of 'test_1(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_1' in the type store
                # Getting the type of 'stypy_return_type' (line 87)
                stypy_return_type_207896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207896)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_1'
                return stypy_return_type_207896

        
        # Assigning a type to the variable 'Mixin' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'Mixin', Mixin)
        # Declaration of the 'Foo' class
        # Getting the type of 'Mixin' (line 89)
        Mixin_207897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'Mixin')
        # Getting the type of 'unittest' (line 89)
        unittest_207898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 89)
        TestCase_207899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 25), unittest_207898, 'TestCase')

        class Foo(Mixin_207897, TestCase_207899, ):
            pass
        
        # Assigning a type to the variable 'Foo' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'Foo', Foo)
        
        # Assigning a List to a Name (line 91):
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_207900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        
        # Assigning a type to the variable 'record' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'record', list_207900)
        
        # Assigning a Call to a Name (line 92):
        
        # Call to TestResult(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_207903 = {}
        # Getting the type of 'unittest' (line 92)
        unittest_207901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 92)
        TestResult_207902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 17), unittest_207901, 'TestResult')
        # Calling TestResult(args, kwargs) (line 92)
        TestResult_call_result_207904 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), TestResult_207902, *[], **kwargs_207903)
        
        # Assigning a type to the variable 'result' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'result', TestResult_call_result_207904)
        
        # Assigning a Call to a Name (line 93):
        
        # Call to Foo(...): (line 93)
        # Processing the call arguments (line 93)
        str_207906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 19), 'str', 'test_1')
        # Processing the call keyword arguments (line 93)
        kwargs_207907 = {}
        # Getting the type of 'Foo' (line 93)
        Foo_207905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 93)
        Foo_call_result_207908 = invoke(stypy.reporting.localization.Localization(__file__, 93, 15), Foo_207905, *[str_207906], **kwargs_207907)
        
        # Assigning a type to the variable 'test' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'test', Foo_call_result_207908)
        
        # Assigning a Call to a Name (line 94):
        
        # Call to TestSuite(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_207911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        # Adding element type (line 94)
        # Getting the type of 'test' (line 94)
        test_207912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 36), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 35), list_207911, test_207912)
        
        # Processing the call keyword arguments (line 94)
        kwargs_207913 = {}
        # Getting the type of 'unittest' (line 94)
        unittest_207909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 94)
        TestSuite_207910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 16), unittest_207909, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 94)
        TestSuite_call_result_207914 = invoke(stypy.reporting.localization.Localization(__file__, 94, 16), TestSuite_207910, *[list_207911], **kwargs_207913)
        
        # Assigning a type to the variable 'suite' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'suite', TestSuite_call_result_207914)
        
        # Call to run(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'result' (line 95)
        result_207917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'result', False)
        # Processing the call keyword arguments (line 95)
        kwargs_207918 = {}
        # Getting the type of 'suite' (line 95)
        suite_207915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'suite', False)
        # Obtaining the member 'run' of a type (line 95)
        run_207916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), suite_207915, 'run')
        # Calling run(args, kwargs) (line 95)
        run_call_result_207919 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), run_207916, *[result_207917], **kwargs_207918)
        
        
        # Call to assertEqual(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'result' (line 96)
        result_207922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'result', False)
        # Obtaining the member 'skipped' of a type (line 96)
        skipped_207923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 25), result_207922, 'skipped')
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_207924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        
        # Obtaining an instance of the builtin type 'tuple' (line 96)
        tuple_207925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 96)
        # Adding element type (line 96)
        # Getting the type of 'test' (line 96)
        test_207926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 43), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 43), tuple_207925, test_207926)
        # Adding element type (line 96)
        str_207927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 49), 'str', 'testing')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 43), tuple_207925, str_207927)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 41), list_207924, tuple_207925)
        
        # Processing the call keyword arguments (line 96)
        kwargs_207928 = {}
        # Getting the type of 'self' (line 96)
        self_207920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 96)
        assertEqual_207921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_207920, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 96)
        assertEqual_call_result_207929 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), assertEqual_207921, *[skipped_207923, list_207924], **kwargs_207928)
        
        
        # Call to assertEqual(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'record' (line 97)
        record_207932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 25), 'record', False)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_207933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        
        # Processing the call keyword arguments (line 97)
        kwargs_207934 = {}
        # Getting the type of 'self' (line 97)
        self_207930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 97)
        assertEqual_207931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_207930, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 97)
        assertEqual_call_result_207935 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assertEqual_207931, *[record_207932, list_207933], **kwargs_207934)
        
        
        # ################# End of 'test_skip_non_unittest_class_new_style(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_skip_non_unittest_class_new_style' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_207936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207936)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_skip_non_unittest_class_new_style'
        return stypy_return_type_207936


    @norecursion
    def test_expected_failure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_expected_failure'
        module_type_store = module_type_store.open_function_context('test_expected_failure', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSkipping.test_expected_failure.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSkipping.test_expected_failure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSkipping.test_expected_failure.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSkipping.test_expected_failure.__dict__.__setitem__('stypy_function_name', 'Test_TestSkipping.test_expected_failure')
        Test_TestSkipping.test_expected_failure.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSkipping.test_expected_failure.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSkipping.test_expected_failure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSkipping.test_expected_failure.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSkipping.test_expected_failure.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSkipping.test_expected_failure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSkipping.test_expected_failure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSkipping.test_expected_failure', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_expected_failure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_expected_failure(...)' code ##################

        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 100)
        unittest_207937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 100)
        TestCase_207938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 18), unittest_207937, 'TestCase')

        class Foo(TestCase_207938, ):

            @norecursion
            def test_die(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_die'
                module_type_store = module_type_store.open_function_context('test_die', 101, 12, False)
                # Assigning a type to the variable 'self' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_die.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_die.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_die.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_die.__dict__.__setitem__('stypy_function_name', 'Foo.test_die')
                Foo.test_die.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_die.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_die.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_die.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_die.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_die.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_die.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_die', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_die', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_die(...)' code ##################

                
                # Call to fail(...): (line 103)
                # Processing the call arguments (line 103)
                str_207941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 26), 'str', 'help me!')
                # Processing the call keyword arguments (line 103)
                kwargs_207942 = {}
                # Getting the type of 'self' (line 103)
                self_207939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'self', False)
                # Obtaining the member 'fail' of a type (line 103)
                fail_207940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), self_207939, 'fail')
                # Calling fail(args, kwargs) (line 103)
                fail_call_result_207943 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), fail_207940, *[str_207941], **kwargs_207942)
                
                
                # ################# End of 'test_die(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_die' in the type store
                # Getting the type of 'stypy_return_type' (line 101)
                stypy_return_type_207944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207944)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_die'
                return stypy_return_type_207944

        
        # Assigning a type to the variable 'Foo' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'Foo', Foo)
        
        # Assigning a List to a Name (line 104):
        
        # Obtaining an instance of the builtin type 'list' (line 104)
        list_207945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 104)
        
        # Assigning a type to the variable 'events' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'events', list_207945)
        
        # Assigning a Call to a Name (line 105):
        
        # Call to LoggingResult(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'events' (line 105)
        events_207947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 31), 'events', False)
        # Processing the call keyword arguments (line 105)
        kwargs_207948 = {}
        # Getting the type of 'LoggingResult' (line 105)
        LoggingResult_207946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 17), 'LoggingResult', False)
        # Calling LoggingResult(args, kwargs) (line 105)
        LoggingResult_call_result_207949 = invoke(stypy.reporting.localization.Localization(__file__, 105, 17), LoggingResult_207946, *[events_207947], **kwargs_207948)
        
        # Assigning a type to the variable 'result' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'result', LoggingResult_call_result_207949)
        
        # Assigning a Call to a Name (line 106):
        
        # Call to Foo(...): (line 106)
        # Processing the call arguments (line 106)
        str_207951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 19), 'str', 'test_die')
        # Processing the call keyword arguments (line 106)
        kwargs_207952 = {}
        # Getting the type of 'Foo' (line 106)
        Foo_207950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 106)
        Foo_call_result_207953 = invoke(stypy.reporting.localization.Localization(__file__, 106, 15), Foo_207950, *[str_207951], **kwargs_207952)
        
        # Assigning a type to the variable 'test' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'test', Foo_call_result_207953)
        
        # Call to run(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'result' (line 107)
        result_207956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'result', False)
        # Processing the call keyword arguments (line 107)
        kwargs_207957 = {}
        # Getting the type of 'test' (line 107)
        test_207954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'test', False)
        # Obtaining the member 'run' of a type (line 107)
        run_207955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), test_207954, 'run')
        # Calling run(args, kwargs) (line 107)
        run_call_result_207958 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), run_207955, *[result_207956], **kwargs_207957)
        
        
        # Call to assertEqual(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'events' (line 108)
        events_207961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'events', False)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_207962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        str_207963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 26), 'str', 'startTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 25), list_207962, str_207963)
        # Adding element type (line 109)
        str_207964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 39), 'str', 'addExpectedFailure')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 25), list_207962, str_207964)
        # Adding element type (line 109)
        str_207965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 61), 'str', 'stopTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 25), list_207962, str_207965)
        
        # Processing the call keyword arguments (line 108)
        kwargs_207966 = {}
        # Getting the type of 'self' (line 108)
        self_207959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 108)
        assertEqual_207960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_207959, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 108)
        assertEqual_call_result_207967 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), assertEqual_207960, *[events_207961, list_207962], **kwargs_207966)
        
        
        # Call to assertEqual(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Obtaining the type of the subscript
        int_207970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 52), 'int')
        
        # Obtaining the type of the subscript
        int_207971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 49), 'int')
        # Getting the type of 'result' (line 110)
        result_207972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'result', False)
        # Obtaining the member 'expectedFailures' of a type (line 110)
        expectedFailures_207973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 25), result_207972, 'expectedFailures')
        # Obtaining the member '__getitem__' of a type (line 110)
        getitem___207974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 25), expectedFailures_207973, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 110)
        subscript_call_result_207975 = invoke(stypy.reporting.localization.Localization(__file__, 110, 25), getitem___207974, int_207971)
        
        # Obtaining the member '__getitem__' of a type (line 110)
        getitem___207976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 25), subscript_call_result_207975, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 110)
        subscript_call_result_207977 = invoke(stypy.reporting.localization.Localization(__file__, 110, 25), getitem___207976, int_207970)
        
        # Getting the type of 'test' (line 110)
        test_207978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 56), 'test', False)
        # Processing the call keyword arguments (line 110)
        kwargs_207979 = {}
        # Getting the type of 'self' (line 110)
        self_207968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 110)
        assertEqual_207969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_207968, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 110)
        assertEqual_call_result_207980 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), assertEqual_207969, *[subscript_call_result_207977, test_207978], **kwargs_207979)
        
        
        # Call to assertTrue(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Call to wasSuccessful(...): (line 111)
        # Processing the call keyword arguments (line 111)
        kwargs_207985 = {}
        # Getting the type of 'result' (line 111)
        result_207983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'result', False)
        # Obtaining the member 'wasSuccessful' of a type (line 111)
        wasSuccessful_207984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), result_207983, 'wasSuccessful')
        # Calling wasSuccessful(args, kwargs) (line 111)
        wasSuccessful_call_result_207986 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), wasSuccessful_207984, *[], **kwargs_207985)
        
        # Processing the call keyword arguments (line 111)
        kwargs_207987 = {}
        # Getting the type of 'self' (line 111)
        self_207981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 111)
        assertTrue_207982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_207981, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 111)
        assertTrue_call_result_207988 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), assertTrue_207982, *[wasSuccessful_call_result_207986], **kwargs_207987)
        
        
        # ################# End of 'test_expected_failure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_expected_failure' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_207989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207989)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_expected_failure'
        return stypy_return_type_207989


    @norecursion
    def test_unexpected_success(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_unexpected_success'
        module_type_store = module_type_store.open_function_context('test_unexpected_success', 113, 4, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSkipping.test_unexpected_success.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSkipping.test_unexpected_success.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSkipping.test_unexpected_success.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSkipping.test_unexpected_success.__dict__.__setitem__('stypy_function_name', 'Test_TestSkipping.test_unexpected_success')
        Test_TestSkipping.test_unexpected_success.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSkipping.test_unexpected_success.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSkipping.test_unexpected_success.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSkipping.test_unexpected_success.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSkipping.test_unexpected_success.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSkipping.test_unexpected_success.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSkipping.test_unexpected_success.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSkipping.test_unexpected_success', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_unexpected_success', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_unexpected_success(...)' code ##################

        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 114)
        unittest_207990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 114)
        TestCase_207991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 18), unittest_207990, 'TestCase')

        class Foo(TestCase_207991, ):

            @norecursion
            def test_die(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_die'
                module_type_store = module_type_store.open_function_context('test_die', 115, 12, False)
                # Assigning a type to the variable 'self' (line 116)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_die.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_die.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_die.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_die.__dict__.__setitem__('stypy_function_name', 'Foo.test_die')
                Foo.test_die.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_die.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_die.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_die.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_die.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_die.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_die.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_die', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_die', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_die(...)' code ##################

                pass
                
                # ################# End of 'test_die(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_die' in the type store
                # Getting the type of 'stypy_return_type' (line 115)
                stypy_return_type_207992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207992)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_die'
                return stypy_return_type_207992

        
        # Assigning a type to the variable 'Foo' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'Foo', Foo)
        
        # Assigning a List to a Name (line 118):
        
        # Obtaining an instance of the builtin type 'list' (line 118)
        list_207993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 118)
        
        # Assigning a type to the variable 'events' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'events', list_207993)
        
        # Assigning a Call to a Name (line 119):
        
        # Call to LoggingResult(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'events' (line 119)
        events_207995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 31), 'events', False)
        # Processing the call keyword arguments (line 119)
        kwargs_207996 = {}
        # Getting the type of 'LoggingResult' (line 119)
        LoggingResult_207994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 17), 'LoggingResult', False)
        # Calling LoggingResult(args, kwargs) (line 119)
        LoggingResult_call_result_207997 = invoke(stypy.reporting.localization.Localization(__file__, 119, 17), LoggingResult_207994, *[events_207995], **kwargs_207996)
        
        # Assigning a type to the variable 'result' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'result', LoggingResult_call_result_207997)
        
        # Assigning a Call to a Name (line 120):
        
        # Call to Foo(...): (line 120)
        # Processing the call arguments (line 120)
        str_207999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 19), 'str', 'test_die')
        # Processing the call keyword arguments (line 120)
        kwargs_208000 = {}
        # Getting the type of 'Foo' (line 120)
        Foo_207998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 120)
        Foo_call_result_208001 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), Foo_207998, *[str_207999], **kwargs_208000)
        
        # Assigning a type to the variable 'test' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'test', Foo_call_result_208001)
        
        # Call to run(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'result' (line 121)
        result_208004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'result', False)
        # Processing the call keyword arguments (line 121)
        kwargs_208005 = {}
        # Getting the type of 'test' (line 121)
        test_208002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'test', False)
        # Obtaining the member 'run' of a type (line 121)
        run_208003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), test_208002, 'run')
        # Calling run(args, kwargs) (line 121)
        run_call_result_208006 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), run_208003, *[result_208004], **kwargs_208005)
        
        
        # Call to assertEqual(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'events' (line 122)
        events_208009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'events', False)
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_208010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        str_208011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 26), 'str', 'startTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 25), list_208010, str_208011)
        # Adding element type (line 123)
        str_208012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 39), 'str', 'addUnexpectedSuccess')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 25), list_208010, str_208012)
        # Adding element type (line 123)
        str_208013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 63), 'str', 'stopTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 25), list_208010, str_208013)
        
        # Processing the call keyword arguments (line 122)
        kwargs_208014 = {}
        # Getting the type of 'self' (line 122)
        self_208007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 122)
        assertEqual_208008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_208007, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 122)
        assertEqual_call_result_208015 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), assertEqual_208008, *[events_208009, list_208010], **kwargs_208014)
        
        
        # Call to assertFalse(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'result' (line 124)
        result_208018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'result', False)
        # Obtaining the member 'failures' of a type (line 124)
        failures_208019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 25), result_208018, 'failures')
        # Processing the call keyword arguments (line 124)
        kwargs_208020 = {}
        # Getting the type of 'self' (line 124)
        self_208016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 124)
        assertFalse_208017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), self_208016, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 124)
        assertFalse_call_result_208021 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), assertFalse_208017, *[failures_208019], **kwargs_208020)
        
        
        # Call to assertEqual(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'result' (line 125)
        result_208024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'result', False)
        # Obtaining the member 'unexpectedSuccesses' of a type (line 125)
        unexpectedSuccesses_208025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 25), result_208024, 'unexpectedSuccesses')
        
        # Obtaining an instance of the builtin type 'list' (line 125)
        list_208026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 125)
        # Adding element type (line 125)
        # Getting the type of 'test' (line 125)
        test_208027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 54), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 53), list_208026, test_208027)
        
        # Processing the call keyword arguments (line 125)
        kwargs_208028 = {}
        # Getting the type of 'self' (line 125)
        self_208022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 125)
        assertEqual_208023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_208022, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 125)
        assertEqual_call_result_208029 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), assertEqual_208023, *[unexpectedSuccesses_208025, list_208026], **kwargs_208028)
        
        
        # Call to assertTrue(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Call to wasSuccessful(...): (line 126)
        # Processing the call keyword arguments (line 126)
        kwargs_208034 = {}
        # Getting the type of 'result' (line 126)
        result_208032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'result', False)
        # Obtaining the member 'wasSuccessful' of a type (line 126)
        wasSuccessful_208033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 24), result_208032, 'wasSuccessful')
        # Calling wasSuccessful(args, kwargs) (line 126)
        wasSuccessful_call_result_208035 = invoke(stypy.reporting.localization.Localization(__file__, 126, 24), wasSuccessful_208033, *[], **kwargs_208034)
        
        # Processing the call keyword arguments (line 126)
        kwargs_208036 = {}
        # Getting the type of 'self' (line 126)
        self_208030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 126)
        assertTrue_208031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), self_208030, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 126)
        assertTrue_call_result_208037 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), assertTrue_208031, *[wasSuccessful_call_result_208035], **kwargs_208036)
        
        
        # ################# End of 'test_unexpected_success(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_unexpected_success' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_208038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208038)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_unexpected_success'
        return stypy_return_type_208038


    @norecursion
    def test_skip_doesnt_run_setup(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_skip_doesnt_run_setup'
        module_type_store = module_type_store.open_function_context('test_skip_doesnt_run_setup', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSkipping.test_skip_doesnt_run_setup.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSkipping.test_skip_doesnt_run_setup.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSkipping.test_skip_doesnt_run_setup.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSkipping.test_skip_doesnt_run_setup.__dict__.__setitem__('stypy_function_name', 'Test_TestSkipping.test_skip_doesnt_run_setup')
        Test_TestSkipping.test_skip_doesnt_run_setup.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSkipping.test_skip_doesnt_run_setup.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSkipping.test_skip_doesnt_run_setup.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSkipping.test_skip_doesnt_run_setup.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSkipping.test_skip_doesnt_run_setup.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSkipping.test_skip_doesnt_run_setup.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSkipping.test_skip_doesnt_run_setup.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSkipping.test_skip_doesnt_run_setup', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_skip_doesnt_run_setup', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_skip_doesnt_run_setup(...)' code ##################

        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 129)
        unittest_208039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 129)
        TestCase_208040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 18), unittest_208039, 'TestCase')

        class Foo(TestCase_208040, ):
            
            # Assigning a Name to a Name (line 130):
            # Getting the type of 'False' (line 130)
            False_208041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 23), 'False')
            # Assigning a type to the variable 'wasSetUp' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'wasSetUp', False_208041)
            
            # Assigning a Name to a Name (line 131):
            # Getting the type of 'False' (line 131)
            False_208042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 26), 'False')
            # Assigning a type to the variable 'wasTornDown' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'wasTornDown', False_208042)

            @norecursion
            def setUp(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUp'
                module_type_store = module_type_store.open_function_context('setUp', 132, 12, False)
                # Assigning a type to the variable 'self' (line 133)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.setUp.__dict__.__setitem__('stypy_localization', localization)
                Foo.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.setUp.__dict__.__setitem__('stypy_function_name', 'Foo.setUp')
                Foo.setUp.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.setUp', [], None, None, defaults, varargs, kwargs)

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

                
                # Assigning a Name to a Attribute (line 133):
                # Getting the type of 'True' (line 133)
                True_208043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 31), 'True')
                # Getting the type of 'Foo' (line 133)
                Foo_208044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'Foo')
                # Setting the type of the member 'wasSetUp' of a type (line 133)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), Foo_208044, 'wasSetUp', True_208043)
                
                # ################# End of 'setUp(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUp' in the type store
                # Getting the type of 'stypy_return_type' (line 132)
                stypy_return_type_208045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208045)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUp'
                return stypy_return_type_208045


            @norecursion
            def tornDown(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tornDown'
                module_type_store = module_type_store.open_function_context('tornDown', 134, 12, False)
                # Assigning a type to the variable 'self' (line 135)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.tornDown.__dict__.__setitem__('stypy_localization', localization)
                Foo.tornDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.tornDown.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.tornDown.__dict__.__setitem__('stypy_function_name', 'Foo.tornDown')
                Foo.tornDown.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.tornDown.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.tornDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.tornDown.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.tornDown.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.tornDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.tornDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.tornDown', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'tornDown', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'tornDown(...)' code ##################

                
                # Assigning a Name to a Attribute (line 135):
                # Getting the type of 'True' (line 135)
                True_208046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 34), 'True')
                # Getting the type of 'Foo' (line 135)
                Foo_208047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'Foo')
                # Setting the type of the member 'wasTornDown' of a type (line 135)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), Foo_208047, 'wasTornDown', True_208046)
                
                # ################# End of 'tornDown(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tornDown' in the type store
                # Getting the type of 'stypy_return_type' (line 134)
                stypy_return_type_208048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208048)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tornDown'
                return stypy_return_type_208048


            @norecursion
            def test_1(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_1'
                module_type_store = module_type_store.open_function_context('test_1', 136, 12, False)
                # Assigning a type to the variable 'self' (line 137)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_1.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_1.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_1.__dict__.__setitem__('stypy_function_name', 'Foo.test_1')
                Foo.test_1.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_1.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_1.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_1', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_1', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_1(...)' code ##################

                pass
                
                # ################# End of 'test_1(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_1' in the type store
                # Getting the type of 'stypy_return_type' (line 136)
                stypy_return_type_208049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208049)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_1'
                return stypy_return_type_208049

        
        # Assigning a type to the variable 'Foo' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'Foo', Foo)
        
        # Assigning a Call to a Name (line 140):
        
        # Call to TestResult(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_208052 = {}
        # Getting the type of 'unittest' (line 140)
        unittest_208050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 140)
        TestResult_208051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 17), unittest_208050, 'TestResult')
        # Calling TestResult(args, kwargs) (line 140)
        TestResult_call_result_208053 = invoke(stypy.reporting.localization.Localization(__file__, 140, 17), TestResult_208051, *[], **kwargs_208052)
        
        # Assigning a type to the variable 'result' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'result', TestResult_call_result_208053)
        
        # Assigning a Call to a Name (line 141):
        
        # Call to Foo(...): (line 141)
        # Processing the call arguments (line 141)
        str_208055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 19), 'str', 'test_1')
        # Processing the call keyword arguments (line 141)
        kwargs_208056 = {}
        # Getting the type of 'Foo' (line 141)
        Foo_208054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 141)
        Foo_call_result_208057 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), Foo_208054, *[str_208055], **kwargs_208056)
        
        # Assigning a type to the variable 'test' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'test', Foo_call_result_208057)
        
        # Assigning a Call to a Name (line 142):
        
        # Call to TestSuite(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Obtaining an instance of the builtin type 'list' (line 142)
        list_208060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 142)
        # Adding element type (line 142)
        # Getting the type of 'test' (line 142)
        test_208061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 36), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 35), list_208060, test_208061)
        
        # Processing the call keyword arguments (line 142)
        kwargs_208062 = {}
        # Getting the type of 'unittest' (line 142)
        unittest_208058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 142)
        TestSuite_208059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 16), unittest_208058, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 142)
        TestSuite_call_result_208063 = invoke(stypy.reporting.localization.Localization(__file__, 142, 16), TestSuite_208059, *[list_208060], **kwargs_208062)
        
        # Assigning a type to the variable 'suite' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'suite', TestSuite_call_result_208063)
        
        # Call to run(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'result' (line 143)
        result_208066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 18), 'result', False)
        # Processing the call keyword arguments (line 143)
        kwargs_208067 = {}
        # Getting the type of 'suite' (line 143)
        suite_208064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'suite', False)
        # Obtaining the member 'run' of a type (line 143)
        run_208065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), suite_208064, 'run')
        # Calling run(args, kwargs) (line 143)
        run_call_result_208068 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), run_208065, *[result_208066], **kwargs_208067)
        
        
        # Call to assertEqual(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'result' (line 144)
        result_208071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'result', False)
        # Obtaining the member 'skipped' of a type (line 144)
        skipped_208072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 25), result_208071, 'skipped')
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_208073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        
        # Obtaining an instance of the builtin type 'tuple' (line 144)
        tuple_208074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 144)
        # Adding element type (line 144)
        # Getting the type of 'test' (line 144)
        test_208075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 43), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 43), tuple_208074, test_208075)
        # Adding element type (line 144)
        str_208076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 49), 'str', 'testing')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 43), tuple_208074, str_208076)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 41), list_208073, tuple_208074)
        
        # Processing the call keyword arguments (line 144)
        kwargs_208077 = {}
        # Getting the type of 'self' (line 144)
        self_208069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 144)
        assertEqual_208070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_208069, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 144)
        assertEqual_call_result_208078 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), assertEqual_208070, *[skipped_208072, list_208073], **kwargs_208077)
        
        
        # Call to assertFalse(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'Foo' (line 145)
        Foo_208081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'Foo', False)
        # Obtaining the member 'wasSetUp' of a type (line 145)
        wasSetUp_208082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 25), Foo_208081, 'wasSetUp')
        # Processing the call keyword arguments (line 145)
        kwargs_208083 = {}
        # Getting the type of 'self' (line 145)
        self_208079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 145)
        assertFalse_208080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_208079, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 145)
        assertFalse_call_result_208084 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), assertFalse_208080, *[wasSetUp_208082], **kwargs_208083)
        
        
        # Call to assertFalse(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'Foo' (line 146)
        Foo_208087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'Foo', False)
        # Obtaining the member 'wasTornDown' of a type (line 146)
        wasTornDown_208088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 25), Foo_208087, 'wasTornDown')
        # Processing the call keyword arguments (line 146)
        kwargs_208089 = {}
        # Getting the type of 'self' (line 146)
        self_208085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 146)
        assertFalse_208086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_208085, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 146)
        assertFalse_call_result_208090 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), assertFalse_208086, *[wasTornDown_208088], **kwargs_208089)
        
        
        # ################# End of 'test_skip_doesnt_run_setup(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_skip_doesnt_run_setup' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_208091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208091)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_skip_doesnt_run_setup'
        return stypy_return_type_208091


    @norecursion
    def test_decorated_skip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_decorated_skip'
        module_type_store = module_type_store.open_function_context('test_decorated_skip', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSkipping.test_decorated_skip.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSkipping.test_decorated_skip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSkipping.test_decorated_skip.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSkipping.test_decorated_skip.__dict__.__setitem__('stypy_function_name', 'Test_TestSkipping.test_decorated_skip')
        Test_TestSkipping.test_decorated_skip.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSkipping.test_decorated_skip.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSkipping.test_decorated_skip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSkipping.test_decorated_skip.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSkipping.test_decorated_skip.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSkipping.test_decorated_skip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSkipping.test_decorated_skip.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSkipping.test_decorated_skip', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_decorated_skip', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_decorated_skip(...)' code ##################


        @norecursion
        def decorator(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'decorator'
            module_type_store = module_type_store.open_function_context('decorator', 149, 8, False)
            
            # Passed parameters checking function
            decorator.stypy_localization = localization
            decorator.stypy_type_of_self = None
            decorator.stypy_type_store = module_type_store
            decorator.stypy_function_name = 'decorator'
            decorator.stypy_param_names_list = ['func']
            decorator.stypy_varargs_param_name = None
            decorator.stypy_kwargs_param_name = None
            decorator.stypy_call_defaults = defaults
            decorator.stypy_call_varargs = varargs
            decorator.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'decorator', ['func'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'decorator', localization, ['func'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'decorator(...)' code ##################


            @norecursion
            def inner(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'inner'
                module_type_store = module_type_store.open_function_context('inner', 150, 12, False)
                
                # Passed parameters checking function
                inner.stypy_localization = localization
                inner.stypy_type_of_self = None
                inner.stypy_type_store = module_type_store
                inner.stypy_function_name = 'inner'
                inner.stypy_param_names_list = []
                inner.stypy_varargs_param_name = 'a'
                inner.stypy_kwargs_param_name = None
                inner.stypy_call_defaults = defaults
                inner.stypy_call_varargs = varargs
                inner.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'inner', [], 'a', None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'inner', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'inner(...)' code ##################

                
                # Call to func(...): (line 151)
                # Getting the type of 'a' (line 151)
                a_208093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 29), 'a', False)
                # Processing the call keyword arguments (line 151)
                kwargs_208094 = {}
                # Getting the type of 'func' (line 151)
                func_208092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'func', False)
                # Calling func(args, kwargs) (line 151)
                func_call_result_208095 = invoke(stypy.reporting.localization.Localization(__file__, 151, 23), func_208092, *[a_208093], **kwargs_208094)
                
                # Assigning a type to the variable 'stypy_return_type' (line 151)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'stypy_return_type', func_call_result_208095)
                
                # ################# End of 'inner(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'inner' in the type store
                # Getting the type of 'stypy_return_type' (line 150)
                stypy_return_type_208096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208096)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'inner'
                return stypy_return_type_208096

            # Assigning a type to the variable 'inner' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'inner', inner)
            # Getting the type of 'inner' (line 152)
            inner_208097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'inner')
            # Assigning a type to the variable 'stypy_return_type' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'stypy_return_type', inner_208097)
            
            # ################# End of 'decorator(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'decorator' in the type store
            # Getting the type of 'stypy_return_type' (line 149)
            stypy_return_type_208098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208098)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'decorator'
            return stypy_return_type_208098

        # Assigning a type to the variable 'decorator' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'decorator', decorator)
        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 154)
        unittest_208099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 154)
        TestCase_208100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 18), unittest_208099, 'TestCase')

        class Foo(TestCase_208100, ):

            @norecursion
            def test_1(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_1'
                module_type_store = module_type_store.open_function_context('test_1', 155, 12, False)
                # Assigning a type to the variable 'self' (line 156)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_1.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_1.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_1.__dict__.__setitem__('stypy_function_name', 'Foo.test_1')
                Foo.test_1.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_1.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_1.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_1', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_1', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_1(...)' code ##################

                pass
                
                # ################# End of 'test_1(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_1' in the type store
                # Getting the type of 'stypy_return_type' (line 155)
                stypy_return_type_208101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208101)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_1'
                return stypy_return_type_208101

        
        # Assigning a type to the variable 'Foo' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'Foo', Foo)
        
        # Assigning a Call to a Name (line 160):
        
        # Call to TestResult(...): (line 160)
        # Processing the call keyword arguments (line 160)
        kwargs_208104 = {}
        # Getting the type of 'unittest' (line 160)
        unittest_208102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 160)
        TestResult_208103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 17), unittest_208102, 'TestResult')
        # Calling TestResult(args, kwargs) (line 160)
        TestResult_call_result_208105 = invoke(stypy.reporting.localization.Localization(__file__, 160, 17), TestResult_208103, *[], **kwargs_208104)
        
        # Assigning a type to the variable 'result' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'result', TestResult_call_result_208105)
        
        # Assigning a Call to a Name (line 161):
        
        # Call to Foo(...): (line 161)
        # Processing the call arguments (line 161)
        str_208107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 19), 'str', 'test_1')
        # Processing the call keyword arguments (line 161)
        kwargs_208108 = {}
        # Getting the type of 'Foo' (line 161)
        Foo_208106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 161)
        Foo_call_result_208109 = invoke(stypy.reporting.localization.Localization(__file__, 161, 15), Foo_208106, *[str_208107], **kwargs_208108)
        
        # Assigning a type to the variable 'test' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'test', Foo_call_result_208109)
        
        # Assigning a Call to a Name (line 162):
        
        # Call to TestSuite(...): (line 162)
        # Processing the call arguments (line 162)
        
        # Obtaining an instance of the builtin type 'list' (line 162)
        list_208112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 162)
        # Adding element type (line 162)
        # Getting the type of 'test' (line 162)
        test_208113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 36), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 35), list_208112, test_208113)
        
        # Processing the call keyword arguments (line 162)
        kwargs_208114 = {}
        # Getting the type of 'unittest' (line 162)
        unittest_208110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 162)
        TestSuite_208111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 16), unittest_208110, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 162)
        TestSuite_call_result_208115 = invoke(stypy.reporting.localization.Localization(__file__, 162, 16), TestSuite_208111, *[list_208112], **kwargs_208114)
        
        # Assigning a type to the variable 'suite' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'suite', TestSuite_call_result_208115)
        
        # Call to run(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'result' (line 163)
        result_208118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 18), 'result', False)
        # Processing the call keyword arguments (line 163)
        kwargs_208119 = {}
        # Getting the type of 'suite' (line 163)
        suite_208116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'suite', False)
        # Obtaining the member 'run' of a type (line 163)
        run_208117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), suite_208116, 'run')
        # Calling run(args, kwargs) (line 163)
        run_call_result_208120 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), run_208117, *[result_208118], **kwargs_208119)
        
        
        # Call to assertEqual(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'result' (line 164)
        result_208123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 25), 'result', False)
        # Obtaining the member 'skipped' of a type (line 164)
        skipped_208124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 25), result_208123, 'skipped')
        
        # Obtaining an instance of the builtin type 'list' (line 164)
        list_208125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 164)
        # Adding element type (line 164)
        
        # Obtaining an instance of the builtin type 'tuple' (line 164)
        tuple_208126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 164)
        # Adding element type (line 164)
        # Getting the type of 'test' (line 164)
        test_208127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 43), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 43), tuple_208126, test_208127)
        # Adding element type (line 164)
        str_208128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 49), 'str', 'testing')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 43), tuple_208126, str_208128)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 41), list_208125, tuple_208126)
        
        # Processing the call keyword arguments (line 164)
        kwargs_208129 = {}
        # Getting the type of 'self' (line 164)
        self_208121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 164)
        assertEqual_208122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), self_208121, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 164)
        assertEqual_call_result_208130 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), assertEqual_208122, *[skipped_208124, list_208125], **kwargs_208129)
        
        
        # ################# End of 'test_decorated_skip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_decorated_skip' in the type store
        # Getting the type of 'stypy_return_type' (line 148)
        stypy_return_type_208131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208131)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_decorated_skip'
        return stypy_return_type_208131


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 6, 0, False)
        # Assigning a type to the variable 'self' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSkipping.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_TestSkipping' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Test_TestSkipping', Test_TestSkipping)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 168)
    # Processing the call keyword arguments (line 168)
    kwargs_208134 = {}
    # Getting the type of 'unittest' (line 168)
    unittest_208132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'unittest', False)
    # Obtaining the member 'main' of a type (line 168)
    main_208133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 4), unittest_208132, 'main')
    # Calling main(args, kwargs) (line 168)
    main_call_result_208135 = invoke(stypy.reporting.localization.Localization(__file__, 168, 4), main_208133, *[], **kwargs_208134)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
