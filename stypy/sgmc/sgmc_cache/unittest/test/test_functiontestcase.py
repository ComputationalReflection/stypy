
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import unittest
2: 
3: from unittest.test.support import LoggingResult
4: 
5: 
6: class Test_FunctionTestCase(unittest.TestCase):
7: 
8:     # "Return the number of tests represented by the this test object. For
9:     # TestCase instances, this will always be 1"
10:     def test_countTestCases(self):
11:         test = unittest.FunctionTestCase(lambda: None)
12: 
13:         self.assertEqual(test.countTestCases(), 1)
14: 
15:     # "When a setUp() method is defined, the test runner will run that method
16:     # prior to each test. Likewise, if a tearDown() method is defined, the
17:     # test runner will invoke that method after each test. In the example,
18:     # setUp() was used to create a fresh sequence for each test."
19:     #
20:     # Make sure the proper call order is maintained, even if setUp() raises
21:     # an exception.
22:     def test_run_call_order__error_in_setUp(self):
23:         events = []
24:         result = LoggingResult(events)
25: 
26:         def setUp():
27:             events.append('setUp')
28:             raise RuntimeError('raised by setUp')
29: 
30:         def test():
31:             events.append('test')
32: 
33:         def tearDown():
34:             events.append('tearDown')
35: 
36:         expected = ['startTest', 'setUp', 'addError', 'stopTest']
37:         unittest.FunctionTestCase(test, setUp, tearDown).run(result)
38:         self.assertEqual(events, expected)
39: 
40:     # "When a setUp() method is defined, the test runner will run that method
41:     # prior to each test. Likewise, if a tearDown() method is defined, the
42:     # test runner will invoke that method after each test. In the example,
43:     # setUp() was used to create a fresh sequence for each test."
44:     #
45:     # Make sure the proper call order is maintained, even if the test raises
46:     # an error (as opposed to a failure).
47:     def test_run_call_order__error_in_test(self):
48:         events = []
49:         result = LoggingResult(events)
50: 
51:         def setUp():
52:             events.append('setUp')
53: 
54:         def test():
55:             events.append('test')
56:             raise RuntimeError('raised by test')
57: 
58:         def tearDown():
59:             events.append('tearDown')
60: 
61:         expected = ['startTest', 'setUp', 'test', 'addError', 'tearDown',
62:                     'stopTest']
63:         unittest.FunctionTestCase(test, setUp, tearDown).run(result)
64:         self.assertEqual(events, expected)
65: 
66:     # "When a setUp() method is defined, the test runner will run that method
67:     # prior to each test. Likewise, if a tearDown() method is defined, the
68:     # test runner will invoke that method after each test. In the example,
69:     # setUp() was used to create a fresh sequence for each test."
70:     #
71:     # Make sure the proper call order is maintained, even if the test signals
72:     # a failure (as opposed to an error).
73:     def test_run_call_order__failure_in_test(self):
74:         events = []
75:         result = LoggingResult(events)
76: 
77:         def setUp():
78:             events.append('setUp')
79: 
80:         def test():
81:             events.append('test')
82:             self.fail('raised by test')
83: 
84:         def tearDown():
85:             events.append('tearDown')
86: 
87:         expected = ['startTest', 'setUp', 'test', 'addFailure', 'tearDown',
88:                     'stopTest']
89:         unittest.FunctionTestCase(test, setUp, tearDown).run(result)
90:         self.assertEqual(events, expected)
91: 
92:     # "When a setUp() method is defined, the test runner will run that method
93:     # prior to each test. Likewise, if a tearDown() method is defined, the
94:     # test runner will invoke that method after each test. In the example,
95:     # setUp() was used to create a fresh sequence for each test."
96:     #
97:     # Make sure the proper call order is maintained, even if tearDown() raises
98:     # an exception.
99:     def test_run_call_order__error_in_tearDown(self):
100:         events = []
101:         result = LoggingResult(events)
102: 
103:         def setUp():
104:             events.append('setUp')
105: 
106:         def test():
107:             events.append('test')
108: 
109:         def tearDown():
110:             events.append('tearDown')
111:             raise RuntimeError('raised by tearDown')
112: 
113:         expected = ['startTest', 'setUp', 'test', 'tearDown', 'addError',
114:                     'stopTest']
115:         unittest.FunctionTestCase(test, setUp, tearDown).run(result)
116:         self.assertEqual(events, expected)
117: 
118:     # "Return a string identifying the specific test case."
119:     #
120:     # Because of the vague nature of the docs, I'm not going to lock this
121:     # test down too much. Really all that can be asserted is that the id()
122:     # will be a string (either 8-byte or unicode -- again, because the docs
123:     # just say "string")
124:     def test_id(self):
125:         test = unittest.FunctionTestCase(lambda: None)
126: 
127:         self.assertIsInstance(test.id(), basestring)
128: 
129:     # "Returns a one-line description of the test, or None if no description
130:     # has been provided. The default implementation of this method returns
131:     # the first line of the test method's docstring, if available, or None."
132:     def test_shortDescription__no_docstring(self):
133:         test = unittest.FunctionTestCase(lambda: None)
134: 
135:         self.assertEqual(test.shortDescription(), None)
136: 
137:     # "Returns a one-line description of the test, or None if no description
138:     # has been provided. The default implementation of this method returns
139:     # the first line of the test method's docstring, if available, or None."
140:     def test_shortDescription__singleline_docstring(self):
141:         desc = "this tests foo"
142:         test = unittest.FunctionTestCase(lambda: None, description=desc)
143: 
144:         self.assertEqual(test.shortDescription(), "this tests foo")
145: 
146: 
147: if __name__ == '__main__':
148:     unittest.main()
149: 

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
import_200469 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest.test.support')

if (type(import_200469) is not StypyTypeError):

    if (import_200469 != 'pyd_module'):
        __import__(import_200469)
        sys_modules_200470 = sys.modules[import_200469]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest.test.support', sys_modules_200470.module_type_store, module_type_store, ['LoggingResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_200470, sys_modules_200470.module_type_store, module_type_store)
    else:
        from unittest.test.support import LoggingResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest.test.support', None, module_type_store, ['LoggingResult'], [LoggingResult])

else:
    # Assigning a type to the variable 'unittest.test.support' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest.test.support', import_200469)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/test/')

# Declaration of the 'Test_FunctionTestCase' class
# Getting the type of 'unittest' (line 6)
unittest_200471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 28), 'unittest')
# Obtaining the member 'TestCase' of a type (line 6)
TestCase_200472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 28), unittest_200471, 'TestCase')

class Test_FunctionTestCase(TestCase_200472, ):

    @norecursion
    def test_countTestCases(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_countTestCases'
        module_type_store = module_type_store.open_function_context('test_countTestCases', 10, 4, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_FunctionTestCase.test_countTestCases.__dict__.__setitem__('stypy_localization', localization)
        Test_FunctionTestCase.test_countTestCases.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_FunctionTestCase.test_countTestCases.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_FunctionTestCase.test_countTestCases.__dict__.__setitem__('stypy_function_name', 'Test_FunctionTestCase.test_countTestCases')
        Test_FunctionTestCase.test_countTestCases.__dict__.__setitem__('stypy_param_names_list', [])
        Test_FunctionTestCase.test_countTestCases.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_FunctionTestCase.test_countTestCases.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_FunctionTestCase.test_countTestCases.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_FunctionTestCase.test_countTestCases.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_FunctionTestCase.test_countTestCases.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_FunctionTestCase.test_countTestCases.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_FunctionTestCase.test_countTestCases', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_countTestCases', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_countTestCases(...)' code ##################

        
        # Assigning a Call to a Name (line 11):
        
        # Call to FunctionTestCase(...): (line 11)
        # Processing the call arguments (line 11)

        @norecursion
        def _stypy_temp_lambda_77(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_77'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_77', 11, 41, True)
            # Passed parameters checking function
            _stypy_temp_lambda_77.stypy_localization = localization
            _stypy_temp_lambda_77.stypy_type_of_self = None
            _stypy_temp_lambda_77.stypy_type_store = module_type_store
            _stypy_temp_lambda_77.stypy_function_name = '_stypy_temp_lambda_77'
            _stypy_temp_lambda_77.stypy_param_names_list = []
            _stypy_temp_lambda_77.stypy_varargs_param_name = None
            _stypy_temp_lambda_77.stypy_kwargs_param_name = None
            _stypy_temp_lambda_77.stypy_call_defaults = defaults
            _stypy_temp_lambda_77.stypy_call_varargs = varargs
            _stypy_temp_lambda_77.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_77', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_77', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 11)
            None_200475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 49), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 41), 'stypy_return_type', None_200475)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_77' in the type store
            # Getting the type of 'stypy_return_type' (line 11)
            stypy_return_type_200476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 41), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200476)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_77'
            return stypy_return_type_200476

        # Assigning a type to the variable '_stypy_temp_lambda_77' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 41), '_stypy_temp_lambda_77', _stypy_temp_lambda_77)
        # Getting the type of '_stypy_temp_lambda_77' (line 11)
        _stypy_temp_lambda_77_200477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 41), '_stypy_temp_lambda_77')
        # Processing the call keyword arguments (line 11)
        kwargs_200478 = {}
        # Getting the type of 'unittest' (line 11)
        unittest_200473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 11)
        FunctionTestCase_200474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 15), unittest_200473, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 11)
        FunctionTestCase_call_result_200479 = invoke(stypy.reporting.localization.Localization(__file__, 11, 15), FunctionTestCase_200474, *[_stypy_temp_lambda_77_200477], **kwargs_200478)
        
        # Assigning a type to the variable 'test' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'test', FunctionTestCase_call_result_200479)
        
        # Call to assertEqual(...): (line 13)
        # Processing the call arguments (line 13)
        
        # Call to countTestCases(...): (line 13)
        # Processing the call keyword arguments (line 13)
        kwargs_200484 = {}
        # Getting the type of 'test' (line 13)
        test_200482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 25), 'test', False)
        # Obtaining the member 'countTestCases' of a type (line 13)
        countTestCases_200483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 25), test_200482, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 13)
        countTestCases_call_result_200485 = invoke(stypy.reporting.localization.Localization(__file__, 13, 25), countTestCases_200483, *[], **kwargs_200484)
        
        int_200486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 48), 'int')
        # Processing the call keyword arguments (line 13)
        kwargs_200487 = {}
        # Getting the type of 'self' (line 13)
        self_200480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 13)
        assertEqual_200481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), self_200480, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 13)
        assertEqual_call_result_200488 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), assertEqual_200481, *[countTestCases_call_result_200485, int_200486], **kwargs_200487)
        
        
        # ################# End of 'test_countTestCases(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_countTestCases' in the type store
        # Getting the type of 'stypy_return_type' (line 10)
        stypy_return_type_200489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200489)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_countTestCases'
        return stypy_return_type_200489


    @norecursion
    def test_run_call_order__error_in_setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_run_call_order__error_in_setUp'
        module_type_store = module_type_store.open_function_context('test_run_call_order__error_in_setUp', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_FunctionTestCase.test_run_call_order__error_in_setUp.__dict__.__setitem__('stypy_localization', localization)
        Test_FunctionTestCase.test_run_call_order__error_in_setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_FunctionTestCase.test_run_call_order__error_in_setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_FunctionTestCase.test_run_call_order__error_in_setUp.__dict__.__setitem__('stypy_function_name', 'Test_FunctionTestCase.test_run_call_order__error_in_setUp')
        Test_FunctionTestCase.test_run_call_order__error_in_setUp.__dict__.__setitem__('stypy_param_names_list', [])
        Test_FunctionTestCase.test_run_call_order__error_in_setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_FunctionTestCase.test_run_call_order__error_in_setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_FunctionTestCase.test_run_call_order__error_in_setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_FunctionTestCase.test_run_call_order__error_in_setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_FunctionTestCase.test_run_call_order__error_in_setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_FunctionTestCase.test_run_call_order__error_in_setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_FunctionTestCase.test_run_call_order__error_in_setUp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_run_call_order__error_in_setUp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_run_call_order__error_in_setUp(...)' code ##################

        
        # Assigning a List to a Name (line 23):
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_200490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        
        # Assigning a type to the variable 'events' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'events', list_200490)
        
        # Assigning a Call to a Name (line 24):
        
        # Call to LoggingResult(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'events' (line 24)
        events_200492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 31), 'events', False)
        # Processing the call keyword arguments (line 24)
        kwargs_200493 = {}
        # Getting the type of 'LoggingResult' (line 24)
        LoggingResult_200491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'LoggingResult', False)
        # Calling LoggingResult(args, kwargs) (line 24)
        LoggingResult_call_result_200494 = invoke(stypy.reporting.localization.Localization(__file__, 24, 17), LoggingResult_200491, *[events_200492], **kwargs_200493)
        
        # Assigning a type to the variable 'result' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'result', LoggingResult_call_result_200494)

        @norecursion
        def setUp(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'setUp'
            module_type_store = module_type_store.open_function_context('setUp', 26, 8, False)
            
            # Passed parameters checking function
            setUp.stypy_localization = localization
            setUp.stypy_type_of_self = None
            setUp.stypy_type_store = module_type_store
            setUp.stypy_function_name = 'setUp'
            setUp.stypy_param_names_list = []
            setUp.stypy_varargs_param_name = None
            setUp.stypy_kwargs_param_name = None
            setUp.stypy_call_defaults = defaults
            setUp.stypy_call_varargs = varargs
            setUp.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'setUp', [], None, None, defaults, varargs, kwargs)

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

            
            # Call to append(...): (line 27)
            # Processing the call arguments (line 27)
            str_200497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'str', 'setUp')
            # Processing the call keyword arguments (line 27)
            kwargs_200498 = {}
            # Getting the type of 'events' (line 27)
            events_200495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'events', False)
            # Obtaining the member 'append' of a type (line 27)
            append_200496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), events_200495, 'append')
            # Calling append(args, kwargs) (line 27)
            append_call_result_200499 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), append_200496, *[str_200497], **kwargs_200498)
            
            
            # Call to RuntimeError(...): (line 28)
            # Processing the call arguments (line 28)
            str_200501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 31), 'str', 'raised by setUp')
            # Processing the call keyword arguments (line 28)
            kwargs_200502 = {}
            # Getting the type of 'RuntimeError' (line 28)
            RuntimeError_200500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 28)
            RuntimeError_call_result_200503 = invoke(stypy.reporting.localization.Localization(__file__, 28, 18), RuntimeError_200500, *[str_200501], **kwargs_200502)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 28, 12), RuntimeError_call_result_200503, 'raise parameter', BaseException)
            
            # ################# End of 'setUp(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'setUp' in the type store
            # Getting the type of 'stypy_return_type' (line 26)
            stypy_return_type_200504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200504)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'setUp'
            return stypy_return_type_200504

        # Assigning a type to the variable 'setUp' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'setUp', setUp)

        @norecursion
        def test(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'test'
            module_type_store = module_type_store.open_function_context('test', 30, 8, False)
            
            # Passed parameters checking function
            test.stypy_localization = localization
            test.stypy_type_of_self = None
            test.stypy_type_store = module_type_store
            test.stypy_function_name = 'test'
            test.stypy_param_names_list = []
            test.stypy_varargs_param_name = None
            test.stypy_kwargs_param_name = None
            test.stypy_call_defaults = defaults
            test.stypy_call_varargs = varargs
            test.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'test', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'test', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'test(...)' code ##################

            
            # Call to append(...): (line 31)
            # Processing the call arguments (line 31)
            str_200507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 26), 'str', 'test')
            # Processing the call keyword arguments (line 31)
            kwargs_200508 = {}
            # Getting the type of 'events' (line 31)
            events_200505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'events', False)
            # Obtaining the member 'append' of a type (line 31)
            append_200506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), events_200505, 'append')
            # Calling append(args, kwargs) (line 31)
            append_call_result_200509 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), append_200506, *[str_200507], **kwargs_200508)
            
            
            # ################# End of 'test(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'test' in the type store
            # Getting the type of 'stypy_return_type' (line 30)
            stypy_return_type_200510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200510)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'test'
            return stypy_return_type_200510

        # Assigning a type to the variable 'test' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'test', test)

        @norecursion
        def tearDown(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'tearDown'
            module_type_store = module_type_store.open_function_context('tearDown', 33, 8, False)
            
            # Passed parameters checking function
            tearDown.stypy_localization = localization
            tearDown.stypy_type_of_self = None
            tearDown.stypy_type_store = module_type_store
            tearDown.stypy_function_name = 'tearDown'
            tearDown.stypy_param_names_list = []
            tearDown.stypy_varargs_param_name = None
            tearDown.stypy_kwargs_param_name = None
            tearDown.stypy_call_defaults = defaults
            tearDown.stypy_call_varargs = varargs
            tearDown.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'tearDown', [], None, None, defaults, varargs, kwargs)

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

            
            # Call to append(...): (line 34)
            # Processing the call arguments (line 34)
            str_200513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 26), 'str', 'tearDown')
            # Processing the call keyword arguments (line 34)
            kwargs_200514 = {}
            # Getting the type of 'events' (line 34)
            events_200511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'events', False)
            # Obtaining the member 'append' of a type (line 34)
            append_200512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), events_200511, 'append')
            # Calling append(args, kwargs) (line 34)
            append_call_result_200515 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), append_200512, *[str_200513], **kwargs_200514)
            
            
            # ################# End of 'tearDown(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'tearDown' in the type store
            # Getting the type of 'stypy_return_type' (line 33)
            stypy_return_type_200516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200516)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'tearDown'
            return stypy_return_type_200516

        # Assigning a type to the variable 'tearDown' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'tearDown', tearDown)
        
        # Assigning a List to a Name (line 36):
        
        # Obtaining an instance of the builtin type 'list' (line 36)
        list_200517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 36)
        # Adding element type (line 36)
        str_200518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 20), 'str', 'startTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 19), list_200517, str_200518)
        # Adding element type (line 36)
        str_200519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 33), 'str', 'setUp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 19), list_200517, str_200519)
        # Adding element type (line 36)
        str_200520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 42), 'str', 'addError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 19), list_200517, str_200520)
        # Adding element type (line 36)
        str_200521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 54), 'str', 'stopTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 19), list_200517, str_200521)
        
        # Assigning a type to the variable 'expected' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'expected', list_200517)
        
        # Call to run(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'result' (line 37)
        result_200530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 61), 'result', False)
        # Processing the call keyword arguments (line 37)
        kwargs_200531 = {}
        
        # Call to FunctionTestCase(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'test' (line 37)
        test_200524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 34), 'test', False)
        # Getting the type of 'setUp' (line 37)
        setUp_200525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 40), 'setUp', False)
        # Getting the type of 'tearDown' (line 37)
        tearDown_200526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 47), 'tearDown', False)
        # Processing the call keyword arguments (line 37)
        kwargs_200527 = {}
        # Getting the type of 'unittest' (line 37)
        unittest_200522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 37)
        FunctionTestCase_200523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), unittest_200522, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 37)
        FunctionTestCase_call_result_200528 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), FunctionTestCase_200523, *[test_200524, setUp_200525, tearDown_200526], **kwargs_200527)
        
        # Obtaining the member 'run' of a type (line 37)
        run_200529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), FunctionTestCase_call_result_200528, 'run')
        # Calling run(args, kwargs) (line 37)
        run_call_result_200532 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), run_200529, *[result_200530], **kwargs_200531)
        
        
        # Call to assertEqual(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'events' (line 38)
        events_200535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'events', False)
        # Getting the type of 'expected' (line 38)
        expected_200536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'expected', False)
        # Processing the call keyword arguments (line 38)
        kwargs_200537 = {}
        # Getting the type of 'self' (line 38)
        self_200533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 38)
        assertEqual_200534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_200533, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 38)
        assertEqual_call_result_200538 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), assertEqual_200534, *[events_200535, expected_200536], **kwargs_200537)
        
        
        # ################# End of 'test_run_call_order__error_in_setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_run_call_order__error_in_setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_200539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200539)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_run_call_order__error_in_setUp'
        return stypy_return_type_200539


    @norecursion
    def test_run_call_order__error_in_test(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_run_call_order__error_in_test'
        module_type_store = module_type_store.open_function_context('test_run_call_order__error_in_test', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_FunctionTestCase.test_run_call_order__error_in_test.__dict__.__setitem__('stypy_localization', localization)
        Test_FunctionTestCase.test_run_call_order__error_in_test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_FunctionTestCase.test_run_call_order__error_in_test.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_FunctionTestCase.test_run_call_order__error_in_test.__dict__.__setitem__('stypy_function_name', 'Test_FunctionTestCase.test_run_call_order__error_in_test')
        Test_FunctionTestCase.test_run_call_order__error_in_test.__dict__.__setitem__('stypy_param_names_list', [])
        Test_FunctionTestCase.test_run_call_order__error_in_test.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_FunctionTestCase.test_run_call_order__error_in_test.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_FunctionTestCase.test_run_call_order__error_in_test.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_FunctionTestCase.test_run_call_order__error_in_test.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_FunctionTestCase.test_run_call_order__error_in_test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_FunctionTestCase.test_run_call_order__error_in_test.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_FunctionTestCase.test_run_call_order__error_in_test', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_run_call_order__error_in_test', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_run_call_order__error_in_test(...)' code ##################

        
        # Assigning a List to a Name (line 48):
        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_200540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        
        # Assigning a type to the variable 'events' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'events', list_200540)
        
        # Assigning a Call to a Name (line 49):
        
        # Call to LoggingResult(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'events' (line 49)
        events_200542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 31), 'events', False)
        # Processing the call keyword arguments (line 49)
        kwargs_200543 = {}
        # Getting the type of 'LoggingResult' (line 49)
        LoggingResult_200541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'LoggingResult', False)
        # Calling LoggingResult(args, kwargs) (line 49)
        LoggingResult_call_result_200544 = invoke(stypy.reporting.localization.Localization(__file__, 49, 17), LoggingResult_200541, *[events_200542], **kwargs_200543)
        
        # Assigning a type to the variable 'result' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'result', LoggingResult_call_result_200544)

        @norecursion
        def setUp(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'setUp'
            module_type_store = module_type_store.open_function_context('setUp', 51, 8, False)
            
            # Passed parameters checking function
            setUp.stypy_localization = localization
            setUp.stypy_type_of_self = None
            setUp.stypy_type_store = module_type_store
            setUp.stypy_function_name = 'setUp'
            setUp.stypy_param_names_list = []
            setUp.stypy_varargs_param_name = None
            setUp.stypy_kwargs_param_name = None
            setUp.stypy_call_defaults = defaults
            setUp.stypy_call_varargs = varargs
            setUp.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'setUp', [], None, None, defaults, varargs, kwargs)

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

            
            # Call to append(...): (line 52)
            # Processing the call arguments (line 52)
            str_200547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 26), 'str', 'setUp')
            # Processing the call keyword arguments (line 52)
            kwargs_200548 = {}
            # Getting the type of 'events' (line 52)
            events_200545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'events', False)
            # Obtaining the member 'append' of a type (line 52)
            append_200546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), events_200545, 'append')
            # Calling append(args, kwargs) (line 52)
            append_call_result_200549 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), append_200546, *[str_200547], **kwargs_200548)
            
            
            # ################# End of 'setUp(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'setUp' in the type store
            # Getting the type of 'stypy_return_type' (line 51)
            stypy_return_type_200550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200550)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'setUp'
            return stypy_return_type_200550

        # Assigning a type to the variable 'setUp' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'setUp', setUp)

        @norecursion
        def test(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'test'
            module_type_store = module_type_store.open_function_context('test', 54, 8, False)
            
            # Passed parameters checking function
            test.stypy_localization = localization
            test.stypy_type_of_self = None
            test.stypy_type_store = module_type_store
            test.stypy_function_name = 'test'
            test.stypy_param_names_list = []
            test.stypy_varargs_param_name = None
            test.stypy_kwargs_param_name = None
            test.stypy_call_defaults = defaults
            test.stypy_call_varargs = varargs
            test.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'test', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'test', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'test(...)' code ##################

            
            # Call to append(...): (line 55)
            # Processing the call arguments (line 55)
            str_200553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'str', 'test')
            # Processing the call keyword arguments (line 55)
            kwargs_200554 = {}
            # Getting the type of 'events' (line 55)
            events_200551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'events', False)
            # Obtaining the member 'append' of a type (line 55)
            append_200552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 12), events_200551, 'append')
            # Calling append(args, kwargs) (line 55)
            append_call_result_200555 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), append_200552, *[str_200553], **kwargs_200554)
            
            
            # Call to RuntimeError(...): (line 56)
            # Processing the call arguments (line 56)
            str_200557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 31), 'str', 'raised by test')
            # Processing the call keyword arguments (line 56)
            kwargs_200558 = {}
            # Getting the type of 'RuntimeError' (line 56)
            RuntimeError_200556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 56)
            RuntimeError_call_result_200559 = invoke(stypy.reporting.localization.Localization(__file__, 56, 18), RuntimeError_200556, *[str_200557], **kwargs_200558)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 56, 12), RuntimeError_call_result_200559, 'raise parameter', BaseException)
            
            # ################# End of 'test(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'test' in the type store
            # Getting the type of 'stypy_return_type' (line 54)
            stypy_return_type_200560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200560)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'test'
            return stypy_return_type_200560

        # Assigning a type to the variable 'test' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'test', test)

        @norecursion
        def tearDown(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'tearDown'
            module_type_store = module_type_store.open_function_context('tearDown', 58, 8, False)
            
            # Passed parameters checking function
            tearDown.stypy_localization = localization
            tearDown.stypy_type_of_self = None
            tearDown.stypy_type_store = module_type_store
            tearDown.stypy_function_name = 'tearDown'
            tearDown.stypy_param_names_list = []
            tearDown.stypy_varargs_param_name = None
            tearDown.stypy_kwargs_param_name = None
            tearDown.stypy_call_defaults = defaults
            tearDown.stypy_call_varargs = varargs
            tearDown.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'tearDown', [], None, None, defaults, varargs, kwargs)

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

            
            # Call to append(...): (line 59)
            # Processing the call arguments (line 59)
            str_200563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 26), 'str', 'tearDown')
            # Processing the call keyword arguments (line 59)
            kwargs_200564 = {}
            # Getting the type of 'events' (line 59)
            events_200561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'events', False)
            # Obtaining the member 'append' of a type (line 59)
            append_200562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 12), events_200561, 'append')
            # Calling append(args, kwargs) (line 59)
            append_call_result_200565 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), append_200562, *[str_200563], **kwargs_200564)
            
            
            # ################# End of 'tearDown(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'tearDown' in the type store
            # Getting the type of 'stypy_return_type' (line 58)
            stypy_return_type_200566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200566)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'tearDown'
            return stypy_return_type_200566

        # Assigning a type to the variable 'tearDown' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'tearDown', tearDown)
        
        # Assigning a List to a Name (line 61):
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_200567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        str_200568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 20), 'str', 'startTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 19), list_200567, str_200568)
        # Adding element type (line 61)
        str_200569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 33), 'str', 'setUp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 19), list_200567, str_200569)
        # Adding element type (line 61)
        str_200570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 42), 'str', 'test')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 19), list_200567, str_200570)
        # Adding element type (line 61)
        str_200571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 50), 'str', 'addError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 19), list_200567, str_200571)
        # Adding element type (line 61)
        str_200572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 62), 'str', 'tearDown')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 19), list_200567, str_200572)
        # Adding element type (line 61)
        str_200573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'str', 'stopTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 19), list_200567, str_200573)
        
        # Assigning a type to the variable 'expected' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'expected', list_200567)
        
        # Call to run(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'result' (line 63)
        result_200582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 61), 'result', False)
        # Processing the call keyword arguments (line 63)
        kwargs_200583 = {}
        
        # Call to FunctionTestCase(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'test' (line 63)
        test_200576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 34), 'test', False)
        # Getting the type of 'setUp' (line 63)
        setUp_200577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 40), 'setUp', False)
        # Getting the type of 'tearDown' (line 63)
        tearDown_200578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 47), 'tearDown', False)
        # Processing the call keyword arguments (line 63)
        kwargs_200579 = {}
        # Getting the type of 'unittest' (line 63)
        unittest_200574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 63)
        FunctionTestCase_200575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), unittest_200574, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 63)
        FunctionTestCase_call_result_200580 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), FunctionTestCase_200575, *[test_200576, setUp_200577, tearDown_200578], **kwargs_200579)
        
        # Obtaining the member 'run' of a type (line 63)
        run_200581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), FunctionTestCase_call_result_200580, 'run')
        # Calling run(args, kwargs) (line 63)
        run_call_result_200584 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), run_200581, *[result_200582], **kwargs_200583)
        
        
        # Call to assertEqual(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'events' (line 64)
        events_200587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'events', False)
        # Getting the type of 'expected' (line 64)
        expected_200588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'expected', False)
        # Processing the call keyword arguments (line 64)
        kwargs_200589 = {}
        # Getting the type of 'self' (line 64)
        self_200585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 64)
        assertEqual_200586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_200585, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 64)
        assertEqual_call_result_200590 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assertEqual_200586, *[events_200587, expected_200588], **kwargs_200589)
        
        
        # ################# End of 'test_run_call_order__error_in_test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_run_call_order__error_in_test' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_200591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200591)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_run_call_order__error_in_test'
        return stypy_return_type_200591


    @norecursion
    def test_run_call_order__failure_in_test(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_run_call_order__failure_in_test'
        module_type_store = module_type_store.open_function_context('test_run_call_order__failure_in_test', 73, 4, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_FunctionTestCase.test_run_call_order__failure_in_test.__dict__.__setitem__('stypy_localization', localization)
        Test_FunctionTestCase.test_run_call_order__failure_in_test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_FunctionTestCase.test_run_call_order__failure_in_test.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_FunctionTestCase.test_run_call_order__failure_in_test.__dict__.__setitem__('stypy_function_name', 'Test_FunctionTestCase.test_run_call_order__failure_in_test')
        Test_FunctionTestCase.test_run_call_order__failure_in_test.__dict__.__setitem__('stypy_param_names_list', [])
        Test_FunctionTestCase.test_run_call_order__failure_in_test.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_FunctionTestCase.test_run_call_order__failure_in_test.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_FunctionTestCase.test_run_call_order__failure_in_test.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_FunctionTestCase.test_run_call_order__failure_in_test.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_FunctionTestCase.test_run_call_order__failure_in_test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_FunctionTestCase.test_run_call_order__failure_in_test.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_FunctionTestCase.test_run_call_order__failure_in_test', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_run_call_order__failure_in_test', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_run_call_order__failure_in_test(...)' code ##################

        
        # Assigning a List to a Name (line 74):
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_200592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        
        # Assigning a type to the variable 'events' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'events', list_200592)
        
        # Assigning a Call to a Name (line 75):
        
        # Call to LoggingResult(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'events' (line 75)
        events_200594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 31), 'events', False)
        # Processing the call keyword arguments (line 75)
        kwargs_200595 = {}
        # Getting the type of 'LoggingResult' (line 75)
        LoggingResult_200593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'LoggingResult', False)
        # Calling LoggingResult(args, kwargs) (line 75)
        LoggingResult_call_result_200596 = invoke(stypy.reporting.localization.Localization(__file__, 75, 17), LoggingResult_200593, *[events_200594], **kwargs_200595)
        
        # Assigning a type to the variable 'result' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'result', LoggingResult_call_result_200596)

        @norecursion
        def setUp(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'setUp'
            module_type_store = module_type_store.open_function_context('setUp', 77, 8, False)
            
            # Passed parameters checking function
            setUp.stypy_localization = localization
            setUp.stypy_type_of_self = None
            setUp.stypy_type_store = module_type_store
            setUp.stypy_function_name = 'setUp'
            setUp.stypy_param_names_list = []
            setUp.stypy_varargs_param_name = None
            setUp.stypy_kwargs_param_name = None
            setUp.stypy_call_defaults = defaults
            setUp.stypy_call_varargs = varargs
            setUp.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'setUp', [], None, None, defaults, varargs, kwargs)

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

            
            # Call to append(...): (line 78)
            # Processing the call arguments (line 78)
            str_200599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'str', 'setUp')
            # Processing the call keyword arguments (line 78)
            kwargs_200600 = {}
            # Getting the type of 'events' (line 78)
            events_200597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'events', False)
            # Obtaining the member 'append' of a type (line 78)
            append_200598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), events_200597, 'append')
            # Calling append(args, kwargs) (line 78)
            append_call_result_200601 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), append_200598, *[str_200599], **kwargs_200600)
            
            
            # ################# End of 'setUp(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'setUp' in the type store
            # Getting the type of 'stypy_return_type' (line 77)
            stypy_return_type_200602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200602)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'setUp'
            return stypy_return_type_200602

        # Assigning a type to the variable 'setUp' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'setUp', setUp)

        @norecursion
        def test(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'test'
            module_type_store = module_type_store.open_function_context('test', 80, 8, False)
            
            # Passed parameters checking function
            test.stypy_localization = localization
            test.stypy_type_of_self = None
            test.stypy_type_store = module_type_store
            test.stypy_function_name = 'test'
            test.stypy_param_names_list = []
            test.stypy_varargs_param_name = None
            test.stypy_kwargs_param_name = None
            test.stypy_call_defaults = defaults
            test.stypy_call_varargs = varargs
            test.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'test', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'test', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'test(...)' code ##################

            
            # Call to append(...): (line 81)
            # Processing the call arguments (line 81)
            str_200605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 26), 'str', 'test')
            # Processing the call keyword arguments (line 81)
            kwargs_200606 = {}
            # Getting the type of 'events' (line 81)
            events_200603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'events', False)
            # Obtaining the member 'append' of a type (line 81)
            append_200604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), events_200603, 'append')
            # Calling append(args, kwargs) (line 81)
            append_call_result_200607 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), append_200604, *[str_200605], **kwargs_200606)
            
            
            # Call to fail(...): (line 82)
            # Processing the call arguments (line 82)
            str_200610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'str', 'raised by test')
            # Processing the call keyword arguments (line 82)
            kwargs_200611 = {}
            # Getting the type of 'self' (line 82)
            self_200608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'self', False)
            # Obtaining the member 'fail' of a type (line 82)
            fail_200609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), self_200608, 'fail')
            # Calling fail(args, kwargs) (line 82)
            fail_call_result_200612 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), fail_200609, *[str_200610], **kwargs_200611)
            
            
            # ################# End of 'test(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'test' in the type store
            # Getting the type of 'stypy_return_type' (line 80)
            stypy_return_type_200613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200613)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'test'
            return stypy_return_type_200613

        # Assigning a type to the variable 'test' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'test', test)

        @norecursion
        def tearDown(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'tearDown'
            module_type_store = module_type_store.open_function_context('tearDown', 84, 8, False)
            
            # Passed parameters checking function
            tearDown.stypy_localization = localization
            tearDown.stypy_type_of_self = None
            tearDown.stypy_type_store = module_type_store
            tearDown.stypy_function_name = 'tearDown'
            tearDown.stypy_param_names_list = []
            tearDown.stypy_varargs_param_name = None
            tearDown.stypy_kwargs_param_name = None
            tearDown.stypy_call_defaults = defaults
            tearDown.stypy_call_varargs = varargs
            tearDown.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'tearDown', [], None, None, defaults, varargs, kwargs)

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

            
            # Call to append(...): (line 85)
            # Processing the call arguments (line 85)
            str_200616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 26), 'str', 'tearDown')
            # Processing the call keyword arguments (line 85)
            kwargs_200617 = {}
            # Getting the type of 'events' (line 85)
            events_200614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'events', False)
            # Obtaining the member 'append' of a type (line 85)
            append_200615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), events_200614, 'append')
            # Calling append(args, kwargs) (line 85)
            append_call_result_200618 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), append_200615, *[str_200616], **kwargs_200617)
            
            
            # ################# End of 'tearDown(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'tearDown' in the type store
            # Getting the type of 'stypy_return_type' (line 84)
            stypy_return_type_200619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200619)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'tearDown'
            return stypy_return_type_200619

        # Assigning a type to the variable 'tearDown' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tearDown', tearDown)
        
        # Assigning a List to a Name (line 87):
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_200620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        str_200621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'str', 'startTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 19), list_200620, str_200621)
        # Adding element type (line 87)
        str_200622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 33), 'str', 'setUp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 19), list_200620, str_200622)
        # Adding element type (line 87)
        str_200623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 42), 'str', 'test')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 19), list_200620, str_200623)
        # Adding element type (line 87)
        str_200624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 50), 'str', 'addFailure')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 19), list_200620, str_200624)
        # Adding element type (line 87)
        str_200625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 64), 'str', 'tearDown')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 19), list_200620, str_200625)
        # Adding element type (line 87)
        str_200626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'str', 'stopTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 19), list_200620, str_200626)
        
        # Assigning a type to the variable 'expected' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'expected', list_200620)
        
        # Call to run(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'result' (line 89)
        result_200635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 61), 'result', False)
        # Processing the call keyword arguments (line 89)
        kwargs_200636 = {}
        
        # Call to FunctionTestCase(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'test' (line 89)
        test_200629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 34), 'test', False)
        # Getting the type of 'setUp' (line 89)
        setUp_200630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 40), 'setUp', False)
        # Getting the type of 'tearDown' (line 89)
        tearDown_200631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 47), 'tearDown', False)
        # Processing the call keyword arguments (line 89)
        kwargs_200632 = {}
        # Getting the type of 'unittest' (line 89)
        unittest_200627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 89)
        FunctionTestCase_200628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), unittest_200627, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 89)
        FunctionTestCase_call_result_200633 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), FunctionTestCase_200628, *[test_200629, setUp_200630, tearDown_200631], **kwargs_200632)
        
        # Obtaining the member 'run' of a type (line 89)
        run_200634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), FunctionTestCase_call_result_200633, 'run')
        # Calling run(args, kwargs) (line 89)
        run_call_result_200637 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), run_200634, *[result_200635], **kwargs_200636)
        
        
        # Call to assertEqual(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'events' (line 90)
        events_200640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'events', False)
        # Getting the type of 'expected' (line 90)
        expected_200641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 33), 'expected', False)
        # Processing the call keyword arguments (line 90)
        kwargs_200642 = {}
        # Getting the type of 'self' (line 90)
        self_200638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 90)
        assertEqual_200639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_200638, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 90)
        assertEqual_call_result_200643 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), assertEqual_200639, *[events_200640, expected_200641], **kwargs_200642)
        
        
        # ################# End of 'test_run_call_order__failure_in_test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_run_call_order__failure_in_test' in the type store
        # Getting the type of 'stypy_return_type' (line 73)
        stypy_return_type_200644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200644)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_run_call_order__failure_in_test'
        return stypy_return_type_200644


    @norecursion
    def test_run_call_order__error_in_tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_run_call_order__error_in_tearDown'
        module_type_store = module_type_store.open_function_context('test_run_call_order__error_in_tearDown', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_FunctionTestCase.test_run_call_order__error_in_tearDown.__dict__.__setitem__('stypy_localization', localization)
        Test_FunctionTestCase.test_run_call_order__error_in_tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_FunctionTestCase.test_run_call_order__error_in_tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_FunctionTestCase.test_run_call_order__error_in_tearDown.__dict__.__setitem__('stypy_function_name', 'Test_FunctionTestCase.test_run_call_order__error_in_tearDown')
        Test_FunctionTestCase.test_run_call_order__error_in_tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        Test_FunctionTestCase.test_run_call_order__error_in_tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_FunctionTestCase.test_run_call_order__error_in_tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_FunctionTestCase.test_run_call_order__error_in_tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_FunctionTestCase.test_run_call_order__error_in_tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_FunctionTestCase.test_run_call_order__error_in_tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_FunctionTestCase.test_run_call_order__error_in_tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_FunctionTestCase.test_run_call_order__error_in_tearDown', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_run_call_order__error_in_tearDown', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_run_call_order__error_in_tearDown(...)' code ##################

        
        # Assigning a List to a Name (line 100):
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_200645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        
        # Assigning a type to the variable 'events' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'events', list_200645)
        
        # Assigning a Call to a Name (line 101):
        
        # Call to LoggingResult(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'events' (line 101)
        events_200647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'events', False)
        # Processing the call keyword arguments (line 101)
        kwargs_200648 = {}
        # Getting the type of 'LoggingResult' (line 101)
        LoggingResult_200646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'LoggingResult', False)
        # Calling LoggingResult(args, kwargs) (line 101)
        LoggingResult_call_result_200649 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), LoggingResult_200646, *[events_200647], **kwargs_200648)
        
        # Assigning a type to the variable 'result' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'result', LoggingResult_call_result_200649)

        @norecursion
        def setUp(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'setUp'
            module_type_store = module_type_store.open_function_context('setUp', 103, 8, False)
            
            # Passed parameters checking function
            setUp.stypy_localization = localization
            setUp.stypy_type_of_self = None
            setUp.stypy_type_store = module_type_store
            setUp.stypy_function_name = 'setUp'
            setUp.stypy_param_names_list = []
            setUp.stypy_varargs_param_name = None
            setUp.stypy_kwargs_param_name = None
            setUp.stypy_call_defaults = defaults
            setUp.stypy_call_varargs = varargs
            setUp.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'setUp', [], None, None, defaults, varargs, kwargs)

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

            
            # Call to append(...): (line 104)
            # Processing the call arguments (line 104)
            str_200652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 26), 'str', 'setUp')
            # Processing the call keyword arguments (line 104)
            kwargs_200653 = {}
            # Getting the type of 'events' (line 104)
            events_200650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'events', False)
            # Obtaining the member 'append' of a type (line 104)
            append_200651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), events_200650, 'append')
            # Calling append(args, kwargs) (line 104)
            append_call_result_200654 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), append_200651, *[str_200652], **kwargs_200653)
            
            
            # ################# End of 'setUp(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'setUp' in the type store
            # Getting the type of 'stypy_return_type' (line 103)
            stypy_return_type_200655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200655)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'setUp'
            return stypy_return_type_200655

        # Assigning a type to the variable 'setUp' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'setUp', setUp)

        @norecursion
        def test(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'test'
            module_type_store = module_type_store.open_function_context('test', 106, 8, False)
            
            # Passed parameters checking function
            test.stypy_localization = localization
            test.stypy_type_of_self = None
            test.stypy_type_store = module_type_store
            test.stypy_function_name = 'test'
            test.stypy_param_names_list = []
            test.stypy_varargs_param_name = None
            test.stypy_kwargs_param_name = None
            test.stypy_call_defaults = defaults
            test.stypy_call_varargs = varargs
            test.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'test', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'test', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'test(...)' code ##################

            
            # Call to append(...): (line 107)
            # Processing the call arguments (line 107)
            str_200658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 26), 'str', 'test')
            # Processing the call keyword arguments (line 107)
            kwargs_200659 = {}
            # Getting the type of 'events' (line 107)
            events_200656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'events', False)
            # Obtaining the member 'append' of a type (line 107)
            append_200657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), events_200656, 'append')
            # Calling append(args, kwargs) (line 107)
            append_call_result_200660 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), append_200657, *[str_200658], **kwargs_200659)
            
            
            # ################# End of 'test(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'test' in the type store
            # Getting the type of 'stypy_return_type' (line 106)
            stypy_return_type_200661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200661)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'test'
            return stypy_return_type_200661

        # Assigning a type to the variable 'test' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'test', test)

        @norecursion
        def tearDown(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'tearDown'
            module_type_store = module_type_store.open_function_context('tearDown', 109, 8, False)
            
            # Passed parameters checking function
            tearDown.stypy_localization = localization
            tearDown.stypy_type_of_self = None
            tearDown.stypy_type_store = module_type_store
            tearDown.stypy_function_name = 'tearDown'
            tearDown.stypy_param_names_list = []
            tearDown.stypy_varargs_param_name = None
            tearDown.stypy_kwargs_param_name = None
            tearDown.stypy_call_defaults = defaults
            tearDown.stypy_call_varargs = varargs
            tearDown.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'tearDown', [], None, None, defaults, varargs, kwargs)

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

            
            # Call to append(...): (line 110)
            # Processing the call arguments (line 110)
            str_200664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 26), 'str', 'tearDown')
            # Processing the call keyword arguments (line 110)
            kwargs_200665 = {}
            # Getting the type of 'events' (line 110)
            events_200662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'events', False)
            # Obtaining the member 'append' of a type (line 110)
            append_200663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), events_200662, 'append')
            # Calling append(args, kwargs) (line 110)
            append_call_result_200666 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), append_200663, *[str_200664], **kwargs_200665)
            
            
            # Call to RuntimeError(...): (line 111)
            # Processing the call arguments (line 111)
            str_200668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 31), 'str', 'raised by tearDown')
            # Processing the call keyword arguments (line 111)
            kwargs_200669 = {}
            # Getting the type of 'RuntimeError' (line 111)
            RuntimeError_200667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 18), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 111)
            RuntimeError_call_result_200670 = invoke(stypy.reporting.localization.Localization(__file__, 111, 18), RuntimeError_200667, *[str_200668], **kwargs_200669)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 111, 12), RuntimeError_call_result_200670, 'raise parameter', BaseException)
            
            # ################# End of 'tearDown(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'tearDown' in the type store
            # Getting the type of 'stypy_return_type' (line 109)
            stypy_return_type_200671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200671)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'tearDown'
            return stypy_return_type_200671

        # Assigning a type to the variable 'tearDown' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tearDown', tearDown)
        
        # Assigning a List to a Name (line 113):
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_200672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        str_200673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 20), 'str', 'startTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 19), list_200672, str_200673)
        # Adding element type (line 113)
        str_200674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 33), 'str', 'setUp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 19), list_200672, str_200674)
        # Adding element type (line 113)
        str_200675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 42), 'str', 'test')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 19), list_200672, str_200675)
        # Adding element type (line 113)
        str_200676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 50), 'str', 'tearDown')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 19), list_200672, str_200676)
        # Adding element type (line 113)
        str_200677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 62), 'str', 'addError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 19), list_200672, str_200677)
        # Adding element type (line 113)
        str_200678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 20), 'str', 'stopTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 19), list_200672, str_200678)
        
        # Assigning a type to the variable 'expected' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'expected', list_200672)
        
        # Call to run(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'result' (line 115)
        result_200687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 61), 'result', False)
        # Processing the call keyword arguments (line 115)
        kwargs_200688 = {}
        
        # Call to FunctionTestCase(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'test' (line 115)
        test_200681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 34), 'test', False)
        # Getting the type of 'setUp' (line 115)
        setUp_200682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 40), 'setUp', False)
        # Getting the type of 'tearDown' (line 115)
        tearDown_200683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 47), 'tearDown', False)
        # Processing the call keyword arguments (line 115)
        kwargs_200684 = {}
        # Getting the type of 'unittest' (line 115)
        unittest_200679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 115)
        FunctionTestCase_200680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), unittest_200679, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 115)
        FunctionTestCase_call_result_200685 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), FunctionTestCase_200680, *[test_200681, setUp_200682, tearDown_200683], **kwargs_200684)
        
        # Obtaining the member 'run' of a type (line 115)
        run_200686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), FunctionTestCase_call_result_200685, 'run')
        # Calling run(args, kwargs) (line 115)
        run_call_result_200689 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), run_200686, *[result_200687], **kwargs_200688)
        
        
        # Call to assertEqual(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'events' (line 116)
        events_200692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 25), 'events', False)
        # Getting the type of 'expected' (line 116)
        expected_200693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 33), 'expected', False)
        # Processing the call keyword arguments (line 116)
        kwargs_200694 = {}
        # Getting the type of 'self' (line 116)
        self_200690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 116)
        assertEqual_200691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_200690, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 116)
        assertEqual_call_result_200695 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), assertEqual_200691, *[events_200692, expected_200693], **kwargs_200694)
        
        
        # ################# End of 'test_run_call_order__error_in_tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_run_call_order__error_in_tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_200696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200696)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_run_call_order__error_in_tearDown'
        return stypy_return_type_200696


    @norecursion
    def test_id(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_id'
        module_type_store = module_type_store.open_function_context('test_id', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_FunctionTestCase.test_id.__dict__.__setitem__('stypy_localization', localization)
        Test_FunctionTestCase.test_id.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_FunctionTestCase.test_id.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_FunctionTestCase.test_id.__dict__.__setitem__('stypy_function_name', 'Test_FunctionTestCase.test_id')
        Test_FunctionTestCase.test_id.__dict__.__setitem__('stypy_param_names_list', [])
        Test_FunctionTestCase.test_id.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_FunctionTestCase.test_id.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_FunctionTestCase.test_id.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_FunctionTestCase.test_id.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_FunctionTestCase.test_id.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_FunctionTestCase.test_id.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_FunctionTestCase.test_id', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_id', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_id(...)' code ##################

        
        # Assigning a Call to a Name (line 125):
        
        # Call to FunctionTestCase(...): (line 125)
        # Processing the call arguments (line 125)

        @norecursion
        def _stypy_temp_lambda_78(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_78'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_78', 125, 41, True)
            # Passed parameters checking function
            _stypy_temp_lambda_78.stypy_localization = localization
            _stypy_temp_lambda_78.stypy_type_of_self = None
            _stypy_temp_lambda_78.stypy_type_store = module_type_store
            _stypy_temp_lambda_78.stypy_function_name = '_stypy_temp_lambda_78'
            _stypy_temp_lambda_78.stypy_param_names_list = []
            _stypy_temp_lambda_78.stypy_varargs_param_name = None
            _stypy_temp_lambda_78.stypy_kwargs_param_name = None
            _stypy_temp_lambda_78.stypy_call_defaults = defaults
            _stypy_temp_lambda_78.stypy_call_varargs = varargs
            _stypy_temp_lambda_78.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_78', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_78', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 125)
            None_200699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 49), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 41), 'stypy_return_type', None_200699)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_78' in the type store
            # Getting the type of 'stypy_return_type' (line 125)
            stypy_return_type_200700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 41), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200700)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_78'
            return stypy_return_type_200700

        # Assigning a type to the variable '_stypy_temp_lambda_78' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 41), '_stypy_temp_lambda_78', _stypy_temp_lambda_78)
        # Getting the type of '_stypy_temp_lambda_78' (line 125)
        _stypy_temp_lambda_78_200701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 41), '_stypy_temp_lambda_78')
        # Processing the call keyword arguments (line 125)
        kwargs_200702 = {}
        # Getting the type of 'unittest' (line 125)
        unittest_200697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 125)
        FunctionTestCase_200698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 15), unittest_200697, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 125)
        FunctionTestCase_call_result_200703 = invoke(stypy.reporting.localization.Localization(__file__, 125, 15), FunctionTestCase_200698, *[_stypy_temp_lambda_78_200701], **kwargs_200702)
        
        # Assigning a type to the variable 'test' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'test', FunctionTestCase_call_result_200703)
        
        # Call to assertIsInstance(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Call to id(...): (line 127)
        # Processing the call keyword arguments (line 127)
        kwargs_200708 = {}
        # Getting the type of 'test' (line 127)
        test_200706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 30), 'test', False)
        # Obtaining the member 'id' of a type (line 127)
        id_200707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), test_200706, 'id')
        # Calling id(args, kwargs) (line 127)
        id_call_result_200709 = invoke(stypy.reporting.localization.Localization(__file__, 127, 30), id_200707, *[], **kwargs_200708)
        
        # Getting the type of 'basestring' (line 127)
        basestring_200710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 41), 'basestring', False)
        # Processing the call keyword arguments (line 127)
        kwargs_200711 = {}
        # Getting the type of 'self' (line 127)
        self_200704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'self', False)
        # Obtaining the member 'assertIsInstance' of a type (line 127)
        assertIsInstance_200705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), self_200704, 'assertIsInstance')
        # Calling assertIsInstance(args, kwargs) (line 127)
        assertIsInstance_call_result_200712 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), assertIsInstance_200705, *[id_call_result_200709, basestring_200710], **kwargs_200711)
        
        
        # ################# End of 'test_id(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_id' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_200713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200713)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_id'
        return stypy_return_type_200713


    @norecursion
    def test_shortDescription__no_docstring(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_shortDescription__no_docstring'
        module_type_store = module_type_store.open_function_context('test_shortDescription__no_docstring', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_FunctionTestCase.test_shortDescription__no_docstring.__dict__.__setitem__('stypy_localization', localization)
        Test_FunctionTestCase.test_shortDescription__no_docstring.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_FunctionTestCase.test_shortDescription__no_docstring.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_FunctionTestCase.test_shortDescription__no_docstring.__dict__.__setitem__('stypy_function_name', 'Test_FunctionTestCase.test_shortDescription__no_docstring')
        Test_FunctionTestCase.test_shortDescription__no_docstring.__dict__.__setitem__('stypy_param_names_list', [])
        Test_FunctionTestCase.test_shortDescription__no_docstring.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_FunctionTestCase.test_shortDescription__no_docstring.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_FunctionTestCase.test_shortDescription__no_docstring.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_FunctionTestCase.test_shortDescription__no_docstring.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_FunctionTestCase.test_shortDescription__no_docstring.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_FunctionTestCase.test_shortDescription__no_docstring.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_FunctionTestCase.test_shortDescription__no_docstring', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_shortDescription__no_docstring', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_shortDescription__no_docstring(...)' code ##################

        
        # Assigning a Call to a Name (line 133):
        
        # Call to FunctionTestCase(...): (line 133)
        # Processing the call arguments (line 133)

        @norecursion
        def _stypy_temp_lambda_79(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_79'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_79', 133, 41, True)
            # Passed parameters checking function
            _stypy_temp_lambda_79.stypy_localization = localization
            _stypy_temp_lambda_79.stypy_type_of_self = None
            _stypy_temp_lambda_79.stypy_type_store = module_type_store
            _stypy_temp_lambda_79.stypy_function_name = '_stypy_temp_lambda_79'
            _stypy_temp_lambda_79.stypy_param_names_list = []
            _stypy_temp_lambda_79.stypy_varargs_param_name = None
            _stypy_temp_lambda_79.stypy_kwargs_param_name = None
            _stypy_temp_lambda_79.stypy_call_defaults = defaults
            _stypy_temp_lambda_79.stypy_call_varargs = varargs
            _stypy_temp_lambda_79.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_79', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_79', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 133)
            None_200716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 49), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 41), 'stypy_return_type', None_200716)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_79' in the type store
            # Getting the type of 'stypy_return_type' (line 133)
            stypy_return_type_200717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 41), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200717)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_79'
            return stypy_return_type_200717

        # Assigning a type to the variable '_stypy_temp_lambda_79' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 41), '_stypy_temp_lambda_79', _stypy_temp_lambda_79)
        # Getting the type of '_stypy_temp_lambda_79' (line 133)
        _stypy_temp_lambda_79_200718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 41), '_stypy_temp_lambda_79')
        # Processing the call keyword arguments (line 133)
        kwargs_200719 = {}
        # Getting the type of 'unittest' (line 133)
        unittest_200714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 133)
        FunctionTestCase_200715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 15), unittest_200714, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 133)
        FunctionTestCase_call_result_200720 = invoke(stypy.reporting.localization.Localization(__file__, 133, 15), FunctionTestCase_200715, *[_stypy_temp_lambda_79_200718], **kwargs_200719)
        
        # Assigning a type to the variable 'test' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'test', FunctionTestCase_call_result_200720)
        
        # Call to assertEqual(...): (line 135)
        # Processing the call arguments (line 135)
        
        # Call to shortDescription(...): (line 135)
        # Processing the call keyword arguments (line 135)
        kwargs_200725 = {}
        # Getting the type of 'test' (line 135)
        test_200723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 25), 'test', False)
        # Obtaining the member 'shortDescription' of a type (line 135)
        shortDescription_200724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 25), test_200723, 'shortDescription')
        # Calling shortDescription(args, kwargs) (line 135)
        shortDescription_call_result_200726 = invoke(stypy.reporting.localization.Localization(__file__, 135, 25), shortDescription_200724, *[], **kwargs_200725)
        
        # Getting the type of 'None' (line 135)
        None_200727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 50), 'None', False)
        # Processing the call keyword arguments (line 135)
        kwargs_200728 = {}
        # Getting the type of 'self' (line 135)
        self_200721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 135)
        assertEqual_200722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), self_200721, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 135)
        assertEqual_call_result_200729 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), assertEqual_200722, *[shortDescription_call_result_200726, None_200727], **kwargs_200728)
        
        
        # ################# End of 'test_shortDescription__no_docstring(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_shortDescription__no_docstring' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_200730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200730)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_shortDescription__no_docstring'
        return stypy_return_type_200730


    @norecursion
    def test_shortDescription__singleline_docstring(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_shortDescription__singleline_docstring'
        module_type_store = module_type_store.open_function_context('test_shortDescription__singleline_docstring', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_FunctionTestCase.test_shortDescription__singleline_docstring.__dict__.__setitem__('stypy_localization', localization)
        Test_FunctionTestCase.test_shortDescription__singleline_docstring.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_FunctionTestCase.test_shortDescription__singleline_docstring.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_FunctionTestCase.test_shortDescription__singleline_docstring.__dict__.__setitem__('stypy_function_name', 'Test_FunctionTestCase.test_shortDescription__singleline_docstring')
        Test_FunctionTestCase.test_shortDescription__singleline_docstring.__dict__.__setitem__('stypy_param_names_list', [])
        Test_FunctionTestCase.test_shortDescription__singleline_docstring.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_FunctionTestCase.test_shortDescription__singleline_docstring.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_FunctionTestCase.test_shortDescription__singleline_docstring.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_FunctionTestCase.test_shortDescription__singleline_docstring.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_FunctionTestCase.test_shortDescription__singleline_docstring.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_FunctionTestCase.test_shortDescription__singleline_docstring.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_FunctionTestCase.test_shortDescription__singleline_docstring', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_shortDescription__singleline_docstring', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_shortDescription__singleline_docstring(...)' code ##################

        
        # Assigning a Str to a Name (line 141):
        str_200731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 15), 'str', 'this tests foo')
        # Assigning a type to the variable 'desc' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'desc', str_200731)
        
        # Assigning a Call to a Name (line 142):
        
        # Call to FunctionTestCase(...): (line 142)
        # Processing the call arguments (line 142)

        @norecursion
        def _stypy_temp_lambda_80(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_80'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_80', 142, 41, True)
            # Passed parameters checking function
            _stypy_temp_lambda_80.stypy_localization = localization
            _stypy_temp_lambda_80.stypy_type_of_self = None
            _stypy_temp_lambda_80.stypy_type_store = module_type_store
            _stypy_temp_lambda_80.stypy_function_name = '_stypy_temp_lambda_80'
            _stypy_temp_lambda_80.stypy_param_names_list = []
            _stypy_temp_lambda_80.stypy_varargs_param_name = None
            _stypy_temp_lambda_80.stypy_kwargs_param_name = None
            _stypy_temp_lambda_80.stypy_call_defaults = defaults
            _stypy_temp_lambda_80.stypy_call_varargs = varargs
            _stypy_temp_lambda_80.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_80', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_80', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 142)
            None_200734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 49), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 41), 'stypy_return_type', None_200734)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_80' in the type store
            # Getting the type of 'stypy_return_type' (line 142)
            stypy_return_type_200735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 41), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200735)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_80'
            return stypy_return_type_200735

        # Assigning a type to the variable '_stypy_temp_lambda_80' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 41), '_stypy_temp_lambda_80', _stypy_temp_lambda_80)
        # Getting the type of '_stypy_temp_lambda_80' (line 142)
        _stypy_temp_lambda_80_200736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 41), '_stypy_temp_lambda_80')
        # Processing the call keyword arguments (line 142)
        # Getting the type of 'desc' (line 142)
        desc_200737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 67), 'desc', False)
        keyword_200738 = desc_200737
        kwargs_200739 = {'description': keyword_200738}
        # Getting the type of 'unittest' (line 142)
        unittest_200732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 142)
        FunctionTestCase_200733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 15), unittest_200732, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 142)
        FunctionTestCase_call_result_200740 = invoke(stypy.reporting.localization.Localization(__file__, 142, 15), FunctionTestCase_200733, *[_stypy_temp_lambda_80_200736], **kwargs_200739)
        
        # Assigning a type to the variable 'test' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'test', FunctionTestCase_call_result_200740)
        
        # Call to assertEqual(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Call to shortDescription(...): (line 144)
        # Processing the call keyword arguments (line 144)
        kwargs_200745 = {}
        # Getting the type of 'test' (line 144)
        test_200743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'test', False)
        # Obtaining the member 'shortDescription' of a type (line 144)
        shortDescription_200744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 25), test_200743, 'shortDescription')
        # Calling shortDescription(args, kwargs) (line 144)
        shortDescription_call_result_200746 = invoke(stypy.reporting.localization.Localization(__file__, 144, 25), shortDescription_200744, *[], **kwargs_200745)
        
        str_200747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 50), 'str', 'this tests foo')
        # Processing the call keyword arguments (line 144)
        kwargs_200748 = {}
        # Getting the type of 'self' (line 144)
        self_200741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 144)
        assertEqual_200742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_200741, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 144)
        assertEqual_call_result_200749 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), assertEqual_200742, *[shortDescription_call_result_200746, str_200747], **kwargs_200748)
        
        
        # ################# End of 'test_shortDescription__singleline_docstring(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_shortDescription__singleline_docstring' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_200750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200750)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_shortDescription__singleline_docstring'
        return stypy_return_type_200750


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_FunctionTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_FunctionTestCase' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Test_FunctionTestCase', Test_FunctionTestCase)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 148)
    # Processing the call keyword arguments (line 148)
    kwargs_200753 = {}
    # Getting the type of 'unittest' (line 148)
    unittest_200751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'unittest', False)
    # Obtaining the member 'main' of a type (line 148)
    main_200752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 4), unittest_200751, 'main')
    # Calling main(args, kwargs) (line 148)
    main_call_result_200754 = invoke(stypy.reporting.localization.Localization(__file__, 148, 4), main_200752, *[], **kwargs_200753)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
