
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import unittest
2: 
3: 
4: class TestHashing(object):
5:     '''Used as a mixin for TestCase'''
6: 
7:     # Check for a valid __hash__ implementation
8:     def test_hash(self):
9:         for obj_1, obj_2 in self.eq_pairs:
10:             try:
11:                 if not hash(obj_1) == hash(obj_2):
12:                     self.fail("%r and %r do not hash equal" % (obj_1, obj_2))
13:             except KeyboardInterrupt:
14:                 raise
15:             except Exception, e:
16:                 self.fail("Problem hashing %r and %r: %s" % (obj_1, obj_2, e))
17: 
18:         for obj_1, obj_2 in self.ne_pairs:
19:             try:
20:                 if hash(obj_1) == hash(obj_2):
21:                     self.fail("%s and %s hash equal, but shouldn't" %
22:                               (obj_1, obj_2))
23:             except KeyboardInterrupt:
24:                 raise
25:             except Exception, e:
26:                 self.fail("Problem hashing %s and %s: %s" % (obj_1, obj_2, e))
27: 
28: 
29: class TestEquality(object):
30:     '''Used as a mixin for TestCase'''
31: 
32:     # Check for a valid __eq__ implementation
33:     def test_eq(self):
34:         for obj_1, obj_2 in self.eq_pairs:
35:             self.assertEqual(obj_1, obj_2)
36:             self.assertEqual(obj_2, obj_1)
37: 
38:     # Check for a valid __ne__ implementation
39:     def test_ne(self):
40:         for obj_1, obj_2 in self.ne_pairs:
41:             self.assertNotEqual(obj_1, obj_2)
42:             self.assertNotEqual(obj_2, obj_1)
43: 
44: 
45: class LoggingResult(unittest.TestResult):
46:     def __init__(self, log):
47:         self._events = log
48:         super(LoggingResult, self).__init__()
49: 
50:     def startTest(self, test):
51:         self._events.append('startTest')
52:         super(LoggingResult, self).startTest(test)
53: 
54:     def startTestRun(self):
55:         self._events.append('startTestRun')
56:         super(LoggingResult, self).startTestRun()
57: 
58:     def stopTest(self, test):
59:         self._events.append('stopTest')
60:         super(LoggingResult, self).stopTest(test)
61: 
62:     def stopTestRun(self):
63:         self._events.append('stopTestRun')
64:         super(LoggingResult, self).stopTestRun()
65: 
66:     def addFailure(self, *args):
67:         self._events.append('addFailure')
68:         super(LoggingResult, self).addFailure(*args)
69: 
70:     def addSuccess(self, *args):
71:         self._events.append('addSuccess')
72:         super(LoggingResult, self).addSuccess(*args)
73: 
74:     def addError(self, *args):
75:         self._events.append('addError')
76:         super(LoggingResult, self).addError(*args)
77: 
78:     def addSkip(self, *args):
79:         self._events.append('addSkip')
80:         super(LoggingResult, self).addSkip(*args)
81: 
82:     def addExpectedFailure(self, *args):
83:         self._events.append('addExpectedFailure')
84:         super(LoggingResult, self).addExpectedFailure(*args)
85: 
86:     def addUnexpectedSuccess(self, *args):
87:         self._events.append('addUnexpectedSuccess')
88:         super(LoggingResult, self).addUnexpectedSuccess(*args)
89: 
90: 
91: class ResultWithNoStartTestRunStopTestRun(object):
92:     '''An object honouring TestResult before startTestRun/stopTestRun.'''
93: 
94:     def __init__(self):
95:         self.failures = []
96:         self.errors = []
97:         self.testsRun = 0
98:         self.skipped = []
99:         self.expectedFailures = []
100:         self.unexpectedSuccesses = []
101:         self.shouldStop = False
102: 
103:     def startTest(self, test):
104:         pass
105: 
106:     def stopTest(self, test):
107:         pass
108: 
109:     def addError(self, test):
110:         pass
111: 
112:     def addFailure(self, test):
113:         pass
114: 
115:     def addSuccess(self, test):
116:         pass
117: 
118:     def wasSuccessful(self):
119:         return True
120: 

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

# Declaration of the 'TestHashing' class

class TestHashing(object, ):
    str_193115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 4), 'str', 'Used as a mixin for TestCase')

    @norecursion
    def test_hash(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hash'
        module_type_store = module_type_store.open_function_context('test_hash', 8, 4, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHashing.test_hash.__dict__.__setitem__('stypy_localization', localization)
        TestHashing.test_hash.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHashing.test_hash.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHashing.test_hash.__dict__.__setitem__('stypy_function_name', 'TestHashing.test_hash')
        TestHashing.test_hash.__dict__.__setitem__('stypy_param_names_list', [])
        TestHashing.test_hash.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHashing.test_hash.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHashing.test_hash.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHashing.test_hash.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHashing.test_hash.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHashing.test_hash.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHashing.test_hash', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hash', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hash(...)' code ##################

        
        # Getting the type of 'self' (line 9)
        self_193116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 28), 'self')
        # Obtaining the member 'eq_pairs' of a type (line 9)
        eq_pairs_193117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 28), self_193116, 'eq_pairs')
        # Testing the type of a for loop iterable (line 9)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 9, 8), eq_pairs_193117)
        # Getting the type of the for loop variable (line 9)
        for_loop_var_193118 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 9, 8), eq_pairs_193117)
        # Assigning a type to the variable 'obj_1' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'obj_1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 8), for_loop_var_193118))
        # Assigning a type to the variable 'obj_2' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'obj_2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 8), for_loop_var_193118))
        # SSA begins for a for statement (line 9)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 10)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        
        
        # Call to hash(...): (line 11)
        # Processing the call arguments (line 11)
        # Getting the type of 'obj_1' (line 11)
        obj_1_193120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 28), 'obj_1', False)
        # Processing the call keyword arguments (line 11)
        kwargs_193121 = {}
        # Getting the type of 'hash' (line 11)
        hash_193119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 23), 'hash', False)
        # Calling hash(args, kwargs) (line 11)
        hash_call_result_193122 = invoke(stypy.reporting.localization.Localization(__file__, 11, 23), hash_193119, *[obj_1_193120], **kwargs_193121)
        
        
        # Call to hash(...): (line 11)
        # Processing the call arguments (line 11)
        # Getting the type of 'obj_2' (line 11)
        obj_2_193124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 43), 'obj_2', False)
        # Processing the call keyword arguments (line 11)
        kwargs_193125 = {}
        # Getting the type of 'hash' (line 11)
        hash_193123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 38), 'hash', False)
        # Calling hash(args, kwargs) (line 11)
        hash_call_result_193126 = invoke(stypy.reporting.localization.Localization(__file__, 11, 38), hash_193123, *[obj_2_193124], **kwargs_193125)
        
        # Applying the binary operator '==' (line 11)
        result_eq_193127 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 23), '==', hash_call_result_193122, hash_call_result_193126)
        
        # Applying the 'not' unary operator (line 11)
        result_not__193128 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 19), 'not', result_eq_193127)
        
        # Testing the type of an if condition (line 11)
        if_condition_193129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 11, 16), result_not__193128)
        # Assigning a type to the variable 'if_condition_193129' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'if_condition_193129', if_condition_193129)
        # SSA begins for if statement (line 11)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to fail(...): (line 12)
        # Processing the call arguments (line 12)
        str_193132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 30), 'str', '%r and %r do not hash equal')
        
        # Obtaining an instance of the builtin type 'tuple' (line 12)
        tuple_193133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 12)
        # Adding element type (line 12)
        # Getting the type of 'obj_1' (line 12)
        obj_1_193134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 63), 'obj_1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 63), tuple_193133, obj_1_193134)
        # Adding element type (line 12)
        # Getting the type of 'obj_2' (line 12)
        obj_2_193135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 70), 'obj_2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 63), tuple_193133, obj_2_193135)
        
        # Applying the binary operator '%' (line 12)
        result_mod_193136 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 30), '%', str_193132, tuple_193133)
        
        # Processing the call keyword arguments (line 12)
        kwargs_193137 = {}
        # Getting the type of 'self' (line 12)
        self_193130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'self', False)
        # Obtaining the member 'fail' of a type (line 12)
        fail_193131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 20), self_193130, 'fail')
        # Calling fail(args, kwargs) (line 12)
        fail_call_result_193138 = invoke(stypy.reporting.localization.Localization(__file__, 12, 20), fail_193131, *[result_mod_193136], **kwargs_193137)
        
        # SSA join for if statement (line 11)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 10)
        # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 10)
        module_type_store.open_ssa_branch('except')
        # SSA branch for the except 'Exception' branch of a try statement (line 10)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 15)
        Exception_193139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'Exception')
        # Assigning a type to the variable 'e' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'e', Exception_193139)
        
        # Call to fail(...): (line 16)
        # Processing the call arguments (line 16)
        str_193142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', 'Problem hashing %r and %r: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 16)
        tuple_193143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 61), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 16)
        # Adding element type (line 16)
        # Getting the type of 'obj_1' (line 16)
        obj_1_193144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 61), 'obj_1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 61), tuple_193143, obj_1_193144)
        # Adding element type (line 16)
        # Getting the type of 'obj_2' (line 16)
        obj_2_193145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 68), 'obj_2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 61), tuple_193143, obj_2_193145)
        # Adding element type (line 16)
        # Getting the type of 'e' (line 16)
        e_193146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 75), 'e', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 61), tuple_193143, e_193146)
        
        # Applying the binary operator '%' (line 16)
        result_mod_193147 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 26), '%', str_193142, tuple_193143)
        
        # Processing the call keyword arguments (line 16)
        kwargs_193148 = {}
        # Getting the type of 'self' (line 16)
        self_193140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'self', False)
        # Obtaining the member 'fail' of a type (line 16)
        fail_193141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 16), self_193140, 'fail')
        # Calling fail(args, kwargs) (line 16)
        fail_call_result_193149 = invoke(stypy.reporting.localization.Localization(__file__, 16, 16), fail_193141, *[result_mod_193147], **kwargs_193148)
        
        # SSA join for try-except statement (line 10)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 18)
        self_193150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 28), 'self')
        # Obtaining the member 'ne_pairs' of a type (line 18)
        ne_pairs_193151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 28), self_193150, 'ne_pairs')
        # Testing the type of a for loop iterable (line 18)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 18, 8), ne_pairs_193151)
        # Getting the type of the for loop variable (line 18)
        for_loop_var_193152 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 18, 8), ne_pairs_193151)
        # Assigning a type to the variable 'obj_1' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'obj_1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 8), for_loop_var_193152))
        # Assigning a type to the variable 'obj_2' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'obj_2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 8), for_loop_var_193152))
        # SSA begins for a for statement (line 18)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 19)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        
        # Call to hash(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'obj_1' (line 20)
        obj_1_193154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 24), 'obj_1', False)
        # Processing the call keyword arguments (line 20)
        kwargs_193155 = {}
        # Getting the type of 'hash' (line 20)
        hash_193153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'hash', False)
        # Calling hash(args, kwargs) (line 20)
        hash_call_result_193156 = invoke(stypy.reporting.localization.Localization(__file__, 20, 19), hash_193153, *[obj_1_193154], **kwargs_193155)
        
        
        # Call to hash(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'obj_2' (line 20)
        obj_2_193158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 39), 'obj_2', False)
        # Processing the call keyword arguments (line 20)
        kwargs_193159 = {}
        # Getting the type of 'hash' (line 20)
        hash_193157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 34), 'hash', False)
        # Calling hash(args, kwargs) (line 20)
        hash_call_result_193160 = invoke(stypy.reporting.localization.Localization(__file__, 20, 34), hash_193157, *[obj_2_193158], **kwargs_193159)
        
        # Applying the binary operator '==' (line 20)
        result_eq_193161 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 19), '==', hash_call_result_193156, hash_call_result_193160)
        
        # Testing the type of an if condition (line 20)
        if_condition_193162 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 16), result_eq_193161)
        # Assigning a type to the variable 'if_condition_193162' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'if_condition_193162', if_condition_193162)
        # SSA begins for if statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to fail(...): (line 21)
        # Processing the call arguments (line 21)
        str_193165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'str', "%s and %s hash equal, but shouldn't")
        
        # Obtaining an instance of the builtin type 'tuple' (line 22)
        tuple_193166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 22)
        # Adding element type (line 22)
        # Getting the type of 'obj_1' (line 22)
        obj_1_193167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 31), 'obj_1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 31), tuple_193166, obj_1_193167)
        # Adding element type (line 22)
        # Getting the type of 'obj_2' (line 22)
        obj_2_193168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 38), 'obj_2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 31), tuple_193166, obj_2_193168)
        
        # Applying the binary operator '%' (line 21)
        result_mod_193169 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 30), '%', str_193165, tuple_193166)
        
        # Processing the call keyword arguments (line 21)
        kwargs_193170 = {}
        # Getting the type of 'self' (line 21)
        self_193163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'self', False)
        # Obtaining the member 'fail' of a type (line 21)
        fail_193164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 20), self_193163, 'fail')
        # Calling fail(args, kwargs) (line 21)
        fail_call_result_193171 = invoke(stypy.reporting.localization.Localization(__file__, 21, 20), fail_193164, *[result_mod_193169], **kwargs_193170)
        
        # SSA join for if statement (line 20)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 19)
        # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 19)
        module_type_store.open_ssa_branch('except')
        # SSA branch for the except 'Exception' branch of a try statement (line 19)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 25)
        Exception_193172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'Exception')
        # Assigning a type to the variable 'e' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'e', Exception_193172)
        
        # Call to fail(...): (line 26)
        # Processing the call arguments (line 26)
        str_193175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'str', 'Problem hashing %s and %s: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 26)
        tuple_193176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 61), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 26)
        # Adding element type (line 26)
        # Getting the type of 'obj_1' (line 26)
        obj_1_193177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 61), 'obj_1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 61), tuple_193176, obj_1_193177)
        # Adding element type (line 26)
        # Getting the type of 'obj_2' (line 26)
        obj_2_193178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 68), 'obj_2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 61), tuple_193176, obj_2_193178)
        # Adding element type (line 26)
        # Getting the type of 'e' (line 26)
        e_193179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 75), 'e', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 61), tuple_193176, e_193179)
        
        # Applying the binary operator '%' (line 26)
        result_mod_193180 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 26), '%', str_193175, tuple_193176)
        
        # Processing the call keyword arguments (line 26)
        kwargs_193181 = {}
        # Getting the type of 'self' (line 26)
        self_193173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'self', False)
        # Obtaining the member 'fail' of a type (line 26)
        fail_193174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), self_193173, 'fail')
        # Calling fail(args, kwargs) (line 26)
        fail_call_result_193182 = invoke(stypy.reporting.localization.Localization(__file__, 26, 16), fail_193174, *[result_mod_193180], **kwargs_193181)
        
        # SSA join for try-except statement (line 19)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_hash(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hash' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_193183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193183)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hash'
        return stypy_return_type_193183


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 4, 0, False)
        # Assigning a type to the variable 'self' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHashing.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestHashing' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'TestHashing', TestHashing)
# Declaration of the 'TestEquality' class

class TestEquality(object, ):
    str_193184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'str', 'Used as a mixin for TestCase')

    @norecursion
    def test_eq(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_eq'
        module_type_store = module_type_store.open_function_context('test_eq', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestEquality.test_eq.__dict__.__setitem__('stypy_localization', localization)
        TestEquality.test_eq.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestEquality.test_eq.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestEquality.test_eq.__dict__.__setitem__('stypy_function_name', 'TestEquality.test_eq')
        TestEquality.test_eq.__dict__.__setitem__('stypy_param_names_list', [])
        TestEquality.test_eq.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestEquality.test_eq.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestEquality.test_eq.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestEquality.test_eq.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestEquality.test_eq.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestEquality.test_eq.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestEquality.test_eq', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_eq', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_eq(...)' code ##################

        
        # Getting the type of 'self' (line 34)
        self_193185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 28), 'self')
        # Obtaining the member 'eq_pairs' of a type (line 34)
        eq_pairs_193186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 28), self_193185, 'eq_pairs')
        # Testing the type of a for loop iterable (line 34)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 8), eq_pairs_193186)
        # Getting the type of the for loop variable (line 34)
        for_loop_var_193187 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 8), eq_pairs_193186)
        # Assigning a type to the variable 'obj_1' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'obj_1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 8), for_loop_var_193187))
        # Assigning a type to the variable 'obj_2' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'obj_2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 8), for_loop_var_193187))
        # SSA begins for a for statement (line 34)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertEqual(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'obj_1' (line 35)
        obj_1_193190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 29), 'obj_1', False)
        # Getting the type of 'obj_2' (line 35)
        obj_2_193191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 36), 'obj_2', False)
        # Processing the call keyword arguments (line 35)
        kwargs_193192 = {}
        # Getting the type of 'self' (line 35)
        self_193188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 35)
        assertEqual_193189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), self_193188, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 35)
        assertEqual_call_result_193193 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), assertEqual_193189, *[obj_1_193190, obj_2_193191], **kwargs_193192)
        
        
        # Call to assertEqual(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'obj_2' (line 36)
        obj_2_193196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 29), 'obj_2', False)
        # Getting the type of 'obj_1' (line 36)
        obj_1_193197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 36), 'obj_1', False)
        # Processing the call keyword arguments (line 36)
        kwargs_193198 = {}
        # Getting the type of 'self' (line 36)
        self_193194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 36)
        assertEqual_193195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), self_193194, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 36)
        assertEqual_call_result_193199 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), assertEqual_193195, *[obj_2_193196, obj_1_193197], **kwargs_193198)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_eq(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_eq' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_193200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193200)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_eq'
        return stypy_return_type_193200


    @norecursion
    def test_ne(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ne'
        module_type_store = module_type_store.open_function_context('test_ne', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestEquality.test_ne.__dict__.__setitem__('stypy_localization', localization)
        TestEquality.test_ne.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestEquality.test_ne.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestEquality.test_ne.__dict__.__setitem__('stypy_function_name', 'TestEquality.test_ne')
        TestEquality.test_ne.__dict__.__setitem__('stypy_param_names_list', [])
        TestEquality.test_ne.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestEquality.test_ne.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestEquality.test_ne.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestEquality.test_ne.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestEquality.test_ne.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestEquality.test_ne.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestEquality.test_ne', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ne', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ne(...)' code ##################

        
        # Getting the type of 'self' (line 40)
        self_193201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'self')
        # Obtaining the member 'ne_pairs' of a type (line 40)
        ne_pairs_193202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 28), self_193201, 'ne_pairs')
        # Testing the type of a for loop iterable (line 40)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 8), ne_pairs_193202)
        # Getting the type of the for loop variable (line 40)
        for_loop_var_193203 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 8), ne_pairs_193202)
        # Assigning a type to the variable 'obj_1' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'obj_1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 8), for_loop_var_193203))
        # Assigning a type to the variable 'obj_2' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'obj_2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 8), for_loop_var_193203))
        # SSA begins for a for statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertNotEqual(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'obj_1' (line 41)
        obj_1_193206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 32), 'obj_1', False)
        # Getting the type of 'obj_2' (line 41)
        obj_2_193207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 39), 'obj_2', False)
        # Processing the call keyword arguments (line 41)
        kwargs_193208 = {}
        # Getting the type of 'self' (line 41)
        self_193204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'self', False)
        # Obtaining the member 'assertNotEqual' of a type (line 41)
        assertNotEqual_193205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), self_193204, 'assertNotEqual')
        # Calling assertNotEqual(args, kwargs) (line 41)
        assertNotEqual_call_result_193209 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), assertNotEqual_193205, *[obj_1_193206, obj_2_193207], **kwargs_193208)
        
        
        # Call to assertNotEqual(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'obj_2' (line 42)
        obj_2_193212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 32), 'obj_2', False)
        # Getting the type of 'obj_1' (line 42)
        obj_1_193213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 'obj_1', False)
        # Processing the call keyword arguments (line 42)
        kwargs_193214 = {}
        # Getting the type of 'self' (line 42)
        self_193210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'self', False)
        # Obtaining the member 'assertNotEqual' of a type (line 42)
        assertNotEqual_193211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), self_193210, 'assertNotEqual')
        # Calling assertNotEqual(args, kwargs) (line 42)
        assertNotEqual_call_result_193215 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), assertNotEqual_193211, *[obj_2_193212, obj_1_193213], **kwargs_193214)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_ne(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ne' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_193216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193216)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ne'
        return stypy_return_type_193216


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 29, 0, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestEquality.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestEquality' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'TestEquality', TestEquality)
# Declaration of the 'LoggingResult' class
# Getting the type of 'unittest' (line 45)
unittest_193217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'unittest')
# Obtaining the member 'TestResult' of a type (line 45)
TestResult_193218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 20), unittest_193217, 'TestResult')

class LoggingResult(TestResult_193218, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingResult.__init__', ['log'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['log'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'log' (line 47)
        log_193219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'log')
        # Getting the type of 'self' (line 47)
        self_193220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member '_events' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_193220, '_events', log_193219)
        
        # Call to __init__(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_193227 = {}
        
        # Call to super(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'LoggingResult' (line 48)
        LoggingResult_193222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'LoggingResult', False)
        # Getting the type of 'self' (line 48)
        self_193223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 29), 'self', False)
        # Processing the call keyword arguments (line 48)
        kwargs_193224 = {}
        # Getting the type of 'super' (line 48)
        super_193221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'super', False)
        # Calling super(args, kwargs) (line 48)
        super_call_result_193225 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), super_193221, *[LoggingResult_193222, self_193223], **kwargs_193224)
        
        # Obtaining the member '__init__' of a type (line 48)
        init___193226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), super_call_result_193225, '__init__')
        # Calling __init__(args, kwargs) (line 48)
        init___call_result_193228 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), init___193226, *[], **kwargs_193227)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def startTest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'startTest'
        module_type_store = module_type_store.open_function_context('startTest', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingResult.startTest.__dict__.__setitem__('stypy_localization', localization)
        LoggingResult.startTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingResult.startTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingResult.startTest.__dict__.__setitem__('stypy_function_name', 'LoggingResult.startTest')
        LoggingResult.startTest.__dict__.__setitem__('stypy_param_names_list', ['test'])
        LoggingResult.startTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        LoggingResult.startTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingResult.startTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingResult.startTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingResult.startTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingResult.startTest.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingResult.startTest', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'startTest', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'startTest(...)' code ##################

        
        # Call to append(...): (line 51)
        # Processing the call arguments (line 51)
        str_193232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 28), 'str', 'startTest')
        # Processing the call keyword arguments (line 51)
        kwargs_193233 = {}
        # Getting the type of 'self' (line 51)
        self_193229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self', False)
        # Obtaining the member '_events' of a type (line 51)
        _events_193230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_193229, '_events')
        # Obtaining the member 'append' of a type (line 51)
        append_193231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), _events_193230, 'append')
        # Calling append(args, kwargs) (line 51)
        append_call_result_193234 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), append_193231, *[str_193232], **kwargs_193233)
        
        
        # Call to startTest(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'test' (line 52)
        test_193241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 45), 'test', False)
        # Processing the call keyword arguments (line 52)
        kwargs_193242 = {}
        
        # Call to super(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'LoggingResult' (line 52)
        LoggingResult_193236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 14), 'LoggingResult', False)
        # Getting the type of 'self' (line 52)
        self_193237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 29), 'self', False)
        # Processing the call keyword arguments (line 52)
        kwargs_193238 = {}
        # Getting the type of 'super' (line 52)
        super_193235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'super', False)
        # Calling super(args, kwargs) (line 52)
        super_call_result_193239 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), super_193235, *[LoggingResult_193236, self_193237], **kwargs_193238)
        
        # Obtaining the member 'startTest' of a type (line 52)
        startTest_193240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), super_call_result_193239, 'startTest')
        # Calling startTest(args, kwargs) (line 52)
        startTest_call_result_193243 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), startTest_193240, *[test_193241], **kwargs_193242)
        
        
        # ################# End of 'startTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'startTest' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_193244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193244)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'startTest'
        return stypy_return_type_193244


    @norecursion
    def startTestRun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'startTestRun'
        module_type_store = module_type_store.open_function_context('startTestRun', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingResult.startTestRun.__dict__.__setitem__('stypy_localization', localization)
        LoggingResult.startTestRun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingResult.startTestRun.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingResult.startTestRun.__dict__.__setitem__('stypy_function_name', 'LoggingResult.startTestRun')
        LoggingResult.startTestRun.__dict__.__setitem__('stypy_param_names_list', [])
        LoggingResult.startTestRun.__dict__.__setitem__('stypy_varargs_param_name', None)
        LoggingResult.startTestRun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingResult.startTestRun.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingResult.startTestRun.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingResult.startTestRun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingResult.startTestRun.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingResult.startTestRun', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'startTestRun', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'startTestRun(...)' code ##################

        
        # Call to append(...): (line 55)
        # Processing the call arguments (line 55)
        str_193248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 28), 'str', 'startTestRun')
        # Processing the call keyword arguments (line 55)
        kwargs_193249 = {}
        # Getting the type of 'self' (line 55)
        self_193245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self', False)
        # Obtaining the member '_events' of a type (line 55)
        _events_193246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_193245, '_events')
        # Obtaining the member 'append' of a type (line 55)
        append_193247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), _events_193246, 'append')
        # Calling append(args, kwargs) (line 55)
        append_call_result_193250 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), append_193247, *[str_193248], **kwargs_193249)
        
        
        # Call to startTestRun(...): (line 56)
        # Processing the call keyword arguments (line 56)
        kwargs_193257 = {}
        
        # Call to super(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'LoggingResult' (line 56)
        LoggingResult_193252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'LoggingResult', False)
        # Getting the type of 'self' (line 56)
        self_193253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'self', False)
        # Processing the call keyword arguments (line 56)
        kwargs_193254 = {}
        # Getting the type of 'super' (line 56)
        super_193251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'super', False)
        # Calling super(args, kwargs) (line 56)
        super_call_result_193255 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), super_193251, *[LoggingResult_193252, self_193253], **kwargs_193254)
        
        # Obtaining the member 'startTestRun' of a type (line 56)
        startTestRun_193256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), super_call_result_193255, 'startTestRun')
        # Calling startTestRun(args, kwargs) (line 56)
        startTestRun_call_result_193258 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), startTestRun_193256, *[], **kwargs_193257)
        
        
        # ################# End of 'startTestRun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'startTestRun' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_193259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193259)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'startTestRun'
        return stypy_return_type_193259


    @norecursion
    def stopTest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'stopTest'
        module_type_store = module_type_store.open_function_context('stopTest', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingResult.stopTest.__dict__.__setitem__('stypy_localization', localization)
        LoggingResult.stopTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingResult.stopTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingResult.stopTest.__dict__.__setitem__('stypy_function_name', 'LoggingResult.stopTest')
        LoggingResult.stopTest.__dict__.__setitem__('stypy_param_names_list', ['test'])
        LoggingResult.stopTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        LoggingResult.stopTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingResult.stopTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingResult.stopTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingResult.stopTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingResult.stopTest.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingResult.stopTest', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'stopTest', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'stopTest(...)' code ##################

        
        # Call to append(...): (line 59)
        # Processing the call arguments (line 59)
        str_193263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 28), 'str', 'stopTest')
        # Processing the call keyword arguments (line 59)
        kwargs_193264 = {}
        # Getting the type of 'self' (line 59)
        self_193260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self', False)
        # Obtaining the member '_events' of a type (line 59)
        _events_193261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_193260, '_events')
        # Obtaining the member 'append' of a type (line 59)
        append_193262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), _events_193261, 'append')
        # Calling append(args, kwargs) (line 59)
        append_call_result_193265 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), append_193262, *[str_193263], **kwargs_193264)
        
        
        # Call to stopTest(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'test' (line 60)
        test_193272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), 'test', False)
        # Processing the call keyword arguments (line 60)
        kwargs_193273 = {}
        
        # Call to super(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'LoggingResult' (line 60)
        LoggingResult_193267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'LoggingResult', False)
        # Getting the type of 'self' (line 60)
        self_193268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'self', False)
        # Processing the call keyword arguments (line 60)
        kwargs_193269 = {}
        # Getting the type of 'super' (line 60)
        super_193266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'super', False)
        # Calling super(args, kwargs) (line 60)
        super_call_result_193270 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), super_193266, *[LoggingResult_193267, self_193268], **kwargs_193269)
        
        # Obtaining the member 'stopTest' of a type (line 60)
        stopTest_193271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), super_call_result_193270, 'stopTest')
        # Calling stopTest(args, kwargs) (line 60)
        stopTest_call_result_193274 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), stopTest_193271, *[test_193272], **kwargs_193273)
        
        
        # ################# End of 'stopTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'stopTest' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_193275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193275)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'stopTest'
        return stypy_return_type_193275


    @norecursion
    def stopTestRun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'stopTestRun'
        module_type_store = module_type_store.open_function_context('stopTestRun', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingResult.stopTestRun.__dict__.__setitem__('stypy_localization', localization)
        LoggingResult.stopTestRun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingResult.stopTestRun.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingResult.stopTestRun.__dict__.__setitem__('stypy_function_name', 'LoggingResult.stopTestRun')
        LoggingResult.stopTestRun.__dict__.__setitem__('stypy_param_names_list', [])
        LoggingResult.stopTestRun.__dict__.__setitem__('stypy_varargs_param_name', None)
        LoggingResult.stopTestRun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingResult.stopTestRun.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingResult.stopTestRun.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingResult.stopTestRun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingResult.stopTestRun.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingResult.stopTestRun', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'stopTestRun', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'stopTestRun(...)' code ##################

        
        # Call to append(...): (line 63)
        # Processing the call arguments (line 63)
        str_193279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 28), 'str', 'stopTestRun')
        # Processing the call keyword arguments (line 63)
        kwargs_193280 = {}
        # Getting the type of 'self' (line 63)
        self_193276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self', False)
        # Obtaining the member '_events' of a type (line 63)
        _events_193277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_193276, '_events')
        # Obtaining the member 'append' of a type (line 63)
        append_193278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), _events_193277, 'append')
        # Calling append(args, kwargs) (line 63)
        append_call_result_193281 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), append_193278, *[str_193279], **kwargs_193280)
        
        
        # Call to stopTestRun(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_193288 = {}
        
        # Call to super(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'LoggingResult' (line 64)
        LoggingResult_193283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 14), 'LoggingResult', False)
        # Getting the type of 'self' (line 64)
        self_193284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'self', False)
        # Processing the call keyword arguments (line 64)
        kwargs_193285 = {}
        # Getting the type of 'super' (line 64)
        super_193282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'super', False)
        # Calling super(args, kwargs) (line 64)
        super_call_result_193286 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), super_193282, *[LoggingResult_193283, self_193284], **kwargs_193285)
        
        # Obtaining the member 'stopTestRun' of a type (line 64)
        stopTestRun_193287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), super_call_result_193286, 'stopTestRun')
        # Calling stopTestRun(args, kwargs) (line 64)
        stopTestRun_call_result_193289 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), stopTestRun_193287, *[], **kwargs_193288)
        
        
        # ################# End of 'stopTestRun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'stopTestRun' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_193290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193290)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'stopTestRun'
        return stypy_return_type_193290


    @norecursion
    def addFailure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addFailure'
        module_type_store = module_type_store.open_function_context('addFailure', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingResult.addFailure.__dict__.__setitem__('stypy_localization', localization)
        LoggingResult.addFailure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingResult.addFailure.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingResult.addFailure.__dict__.__setitem__('stypy_function_name', 'LoggingResult.addFailure')
        LoggingResult.addFailure.__dict__.__setitem__('stypy_param_names_list', [])
        LoggingResult.addFailure.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        LoggingResult.addFailure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingResult.addFailure.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingResult.addFailure.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingResult.addFailure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingResult.addFailure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingResult.addFailure', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addFailure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addFailure(...)' code ##################

        
        # Call to append(...): (line 67)
        # Processing the call arguments (line 67)
        str_193294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 28), 'str', 'addFailure')
        # Processing the call keyword arguments (line 67)
        kwargs_193295 = {}
        # Getting the type of 'self' (line 67)
        self_193291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self', False)
        # Obtaining the member '_events' of a type (line 67)
        _events_193292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_193291, '_events')
        # Obtaining the member 'append' of a type (line 67)
        append_193293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), _events_193292, 'append')
        # Calling append(args, kwargs) (line 67)
        append_call_result_193296 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), append_193293, *[str_193294], **kwargs_193295)
        
        
        # Call to addFailure(...): (line 68)
        # Getting the type of 'args' (line 68)
        args_193303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 47), 'args', False)
        # Processing the call keyword arguments (line 68)
        kwargs_193304 = {}
        
        # Call to super(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'LoggingResult' (line 68)
        LoggingResult_193298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), 'LoggingResult', False)
        # Getting the type of 'self' (line 68)
        self_193299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 29), 'self', False)
        # Processing the call keyword arguments (line 68)
        kwargs_193300 = {}
        # Getting the type of 'super' (line 68)
        super_193297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'super', False)
        # Calling super(args, kwargs) (line 68)
        super_call_result_193301 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), super_193297, *[LoggingResult_193298, self_193299], **kwargs_193300)
        
        # Obtaining the member 'addFailure' of a type (line 68)
        addFailure_193302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), super_call_result_193301, 'addFailure')
        # Calling addFailure(args, kwargs) (line 68)
        addFailure_call_result_193305 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), addFailure_193302, *[args_193303], **kwargs_193304)
        
        
        # ################# End of 'addFailure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addFailure' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_193306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193306)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addFailure'
        return stypy_return_type_193306


    @norecursion
    def addSuccess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addSuccess'
        module_type_store = module_type_store.open_function_context('addSuccess', 70, 4, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingResult.addSuccess.__dict__.__setitem__('stypy_localization', localization)
        LoggingResult.addSuccess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingResult.addSuccess.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingResult.addSuccess.__dict__.__setitem__('stypy_function_name', 'LoggingResult.addSuccess')
        LoggingResult.addSuccess.__dict__.__setitem__('stypy_param_names_list', [])
        LoggingResult.addSuccess.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        LoggingResult.addSuccess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingResult.addSuccess.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingResult.addSuccess.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingResult.addSuccess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingResult.addSuccess.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingResult.addSuccess', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addSuccess', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addSuccess(...)' code ##################

        
        # Call to append(...): (line 71)
        # Processing the call arguments (line 71)
        str_193310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 28), 'str', 'addSuccess')
        # Processing the call keyword arguments (line 71)
        kwargs_193311 = {}
        # Getting the type of 'self' (line 71)
        self_193307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self', False)
        # Obtaining the member '_events' of a type (line 71)
        _events_193308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_193307, '_events')
        # Obtaining the member 'append' of a type (line 71)
        append_193309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), _events_193308, 'append')
        # Calling append(args, kwargs) (line 71)
        append_call_result_193312 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), append_193309, *[str_193310], **kwargs_193311)
        
        
        # Call to addSuccess(...): (line 72)
        # Getting the type of 'args' (line 72)
        args_193319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 47), 'args', False)
        # Processing the call keyword arguments (line 72)
        kwargs_193320 = {}
        
        # Call to super(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'LoggingResult' (line 72)
        LoggingResult_193314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 14), 'LoggingResult', False)
        # Getting the type of 'self' (line 72)
        self_193315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'self', False)
        # Processing the call keyword arguments (line 72)
        kwargs_193316 = {}
        # Getting the type of 'super' (line 72)
        super_193313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'super', False)
        # Calling super(args, kwargs) (line 72)
        super_call_result_193317 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), super_193313, *[LoggingResult_193314, self_193315], **kwargs_193316)
        
        # Obtaining the member 'addSuccess' of a type (line 72)
        addSuccess_193318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), super_call_result_193317, 'addSuccess')
        # Calling addSuccess(args, kwargs) (line 72)
        addSuccess_call_result_193321 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), addSuccess_193318, *[args_193319], **kwargs_193320)
        
        
        # ################# End of 'addSuccess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addSuccess' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_193322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193322)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addSuccess'
        return stypy_return_type_193322


    @norecursion
    def addError(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addError'
        module_type_store = module_type_store.open_function_context('addError', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingResult.addError.__dict__.__setitem__('stypy_localization', localization)
        LoggingResult.addError.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingResult.addError.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingResult.addError.__dict__.__setitem__('stypy_function_name', 'LoggingResult.addError')
        LoggingResult.addError.__dict__.__setitem__('stypy_param_names_list', [])
        LoggingResult.addError.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        LoggingResult.addError.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingResult.addError.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingResult.addError.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingResult.addError.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingResult.addError.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingResult.addError', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addError', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addError(...)' code ##################

        
        # Call to append(...): (line 75)
        # Processing the call arguments (line 75)
        str_193326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 28), 'str', 'addError')
        # Processing the call keyword arguments (line 75)
        kwargs_193327 = {}
        # Getting the type of 'self' (line 75)
        self_193323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self', False)
        # Obtaining the member '_events' of a type (line 75)
        _events_193324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_193323, '_events')
        # Obtaining the member 'append' of a type (line 75)
        append_193325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), _events_193324, 'append')
        # Calling append(args, kwargs) (line 75)
        append_call_result_193328 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), append_193325, *[str_193326], **kwargs_193327)
        
        
        # Call to addError(...): (line 76)
        # Getting the type of 'args' (line 76)
        args_193335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 45), 'args', False)
        # Processing the call keyword arguments (line 76)
        kwargs_193336 = {}
        
        # Call to super(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'LoggingResult' (line 76)
        LoggingResult_193330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'LoggingResult', False)
        # Getting the type of 'self' (line 76)
        self_193331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 29), 'self', False)
        # Processing the call keyword arguments (line 76)
        kwargs_193332 = {}
        # Getting the type of 'super' (line 76)
        super_193329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'super', False)
        # Calling super(args, kwargs) (line 76)
        super_call_result_193333 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), super_193329, *[LoggingResult_193330, self_193331], **kwargs_193332)
        
        # Obtaining the member 'addError' of a type (line 76)
        addError_193334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), super_call_result_193333, 'addError')
        # Calling addError(args, kwargs) (line 76)
        addError_call_result_193337 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), addError_193334, *[args_193335], **kwargs_193336)
        
        
        # ################# End of 'addError(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addError' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_193338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193338)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addError'
        return stypy_return_type_193338


    @norecursion
    def addSkip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addSkip'
        module_type_store = module_type_store.open_function_context('addSkip', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingResult.addSkip.__dict__.__setitem__('stypy_localization', localization)
        LoggingResult.addSkip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingResult.addSkip.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingResult.addSkip.__dict__.__setitem__('stypy_function_name', 'LoggingResult.addSkip')
        LoggingResult.addSkip.__dict__.__setitem__('stypy_param_names_list', [])
        LoggingResult.addSkip.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        LoggingResult.addSkip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingResult.addSkip.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingResult.addSkip.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingResult.addSkip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingResult.addSkip.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingResult.addSkip', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addSkip', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addSkip(...)' code ##################

        
        # Call to append(...): (line 79)
        # Processing the call arguments (line 79)
        str_193342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'str', 'addSkip')
        # Processing the call keyword arguments (line 79)
        kwargs_193343 = {}
        # Getting the type of 'self' (line 79)
        self_193339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self', False)
        # Obtaining the member '_events' of a type (line 79)
        _events_193340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_193339, '_events')
        # Obtaining the member 'append' of a type (line 79)
        append_193341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), _events_193340, 'append')
        # Calling append(args, kwargs) (line 79)
        append_call_result_193344 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), append_193341, *[str_193342], **kwargs_193343)
        
        
        # Call to addSkip(...): (line 80)
        # Getting the type of 'args' (line 80)
        args_193351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'args', False)
        # Processing the call keyword arguments (line 80)
        kwargs_193352 = {}
        
        # Call to super(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'LoggingResult' (line 80)
        LoggingResult_193346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 14), 'LoggingResult', False)
        # Getting the type of 'self' (line 80)
        self_193347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'self', False)
        # Processing the call keyword arguments (line 80)
        kwargs_193348 = {}
        # Getting the type of 'super' (line 80)
        super_193345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'super', False)
        # Calling super(args, kwargs) (line 80)
        super_call_result_193349 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), super_193345, *[LoggingResult_193346, self_193347], **kwargs_193348)
        
        # Obtaining the member 'addSkip' of a type (line 80)
        addSkip_193350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), super_call_result_193349, 'addSkip')
        # Calling addSkip(args, kwargs) (line 80)
        addSkip_call_result_193353 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), addSkip_193350, *[args_193351], **kwargs_193352)
        
        
        # ################# End of 'addSkip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addSkip' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_193354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193354)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addSkip'
        return stypy_return_type_193354


    @norecursion
    def addExpectedFailure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addExpectedFailure'
        module_type_store = module_type_store.open_function_context('addExpectedFailure', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingResult.addExpectedFailure.__dict__.__setitem__('stypy_localization', localization)
        LoggingResult.addExpectedFailure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingResult.addExpectedFailure.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingResult.addExpectedFailure.__dict__.__setitem__('stypy_function_name', 'LoggingResult.addExpectedFailure')
        LoggingResult.addExpectedFailure.__dict__.__setitem__('stypy_param_names_list', [])
        LoggingResult.addExpectedFailure.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        LoggingResult.addExpectedFailure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingResult.addExpectedFailure.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingResult.addExpectedFailure.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingResult.addExpectedFailure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingResult.addExpectedFailure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingResult.addExpectedFailure', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addExpectedFailure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addExpectedFailure(...)' code ##################

        
        # Call to append(...): (line 83)
        # Processing the call arguments (line 83)
        str_193358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 28), 'str', 'addExpectedFailure')
        # Processing the call keyword arguments (line 83)
        kwargs_193359 = {}
        # Getting the type of 'self' (line 83)
        self_193355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self', False)
        # Obtaining the member '_events' of a type (line 83)
        _events_193356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_193355, '_events')
        # Obtaining the member 'append' of a type (line 83)
        append_193357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), _events_193356, 'append')
        # Calling append(args, kwargs) (line 83)
        append_call_result_193360 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), append_193357, *[str_193358], **kwargs_193359)
        
        
        # Call to addExpectedFailure(...): (line 84)
        # Getting the type of 'args' (line 84)
        args_193367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 55), 'args', False)
        # Processing the call keyword arguments (line 84)
        kwargs_193368 = {}
        
        # Call to super(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'LoggingResult' (line 84)
        LoggingResult_193362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'LoggingResult', False)
        # Getting the type of 'self' (line 84)
        self_193363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 29), 'self', False)
        # Processing the call keyword arguments (line 84)
        kwargs_193364 = {}
        # Getting the type of 'super' (line 84)
        super_193361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'super', False)
        # Calling super(args, kwargs) (line 84)
        super_call_result_193365 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), super_193361, *[LoggingResult_193362, self_193363], **kwargs_193364)
        
        # Obtaining the member 'addExpectedFailure' of a type (line 84)
        addExpectedFailure_193366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), super_call_result_193365, 'addExpectedFailure')
        # Calling addExpectedFailure(args, kwargs) (line 84)
        addExpectedFailure_call_result_193369 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), addExpectedFailure_193366, *[args_193367], **kwargs_193368)
        
        
        # ################# End of 'addExpectedFailure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addExpectedFailure' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_193370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193370)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addExpectedFailure'
        return stypy_return_type_193370


    @norecursion
    def addUnexpectedSuccess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addUnexpectedSuccess'
        module_type_store = module_type_store.open_function_context('addUnexpectedSuccess', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_localization', localization)
        LoggingResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_function_name', 'LoggingResult.addUnexpectedSuccess')
        LoggingResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_param_names_list', [])
        LoggingResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        LoggingResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingResult.addUnexpectedSuccess', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addUnexpectedSuccess', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addUnexpectedSuccess(...)' code ##################

        
        # Call to append(...): (line 87)
        # Processing the call arguments (line 87)
        str_193374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 28), 'str', 'addUnexpectedSuccess')
        # Processing the call keyword arguments (line 87)
        kwargs_193375 = {}
        # Getting the type of 'self' (line 87)
        self_193371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self', False)
        # Obtaining the member '_events' of a type (line 87)
        _events_193372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_193371, '_events')
        # Obtaining the member 'append' of a type (line 87)
        append_193373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), _events_193372, 'append')
        # Calling append(args, kwargs) (line 87)
        append_call_result_193376 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), append_193373, *[str_193374], **kwargs_193375)
        
        
        # Call to addUnexpectedSuccess(...): (line 88)
        # Getting the type of 'args' (line 88)
        args_193383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 57), 'args', False)
        # Processing the call keyword arguments (line 88)
        kwargs_193384 = {}
        
        # Call to super(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'LoggingResult' (line 88)
        LoggingResult_193378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'LoggingResult', False)
        # Getting the type of 'self' (line 88)
        self_193379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 29), 'self', False)
        # Processing the call keyword arguments (line 88)
        kwargs_193380 = {}
        # Getting the type of 'super' (line 88)
        super_193377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'super', False)
        # Calling super(args, kwargs) (line 88)
        super_call_result_193381 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), super_193377, *[LoggingResult_193378, self_193379], **kwargs_193380)
        
        # Obtaining the member 'addUnexpectedSuccess' of a type (line 88)
        addUnexpectedSuccess_193382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), super_call_result_193381, 'addUnexpectedSuccess')
        # Calling addUnexpectedSuccess(args, kwargs) (line 88)
        addUnexpectedSuccess_call_result_193385 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), addUnexpectedSuccess_193382, *[args_193383], **kwargs_193384)
        
        
        # ################# End of 'addUnexpectedSuccess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addUnexpectedSuccess' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_193386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193386)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addUnexpectedSuccess'
        return stypy_return_type_193386


# Assigning a type to the variable 'LoggingResult' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'LoggingResult', LoggingResult)
# Declaration of the 'ResultWithNoStartTestRunStopTestRun' class

class ResultWithNoStartTestRunStopTestRun(object, ):
    str_193387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 4), 'str', 'An object honouring TestResult before startTestRun/stopTestRun.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ResultWithNoStartTestRunStopTestRun.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 95):
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_193388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        
        # Getting the type of 'self' (line 95)
        self_193389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Setting the type of the member 'failures' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_193389, 'failures', list_193388)
        
        # Assigning a List to a Attribute (line 96):
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_193390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        
        # Getting the type of 'self' (line 96)
        self_193391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self')
        # Setting the type of the member 'errors' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_193391, 'errors', list_193390)
        
        # Assigning a Num to a Attribute (line 97):
        int_193392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 24), 'int')
        # Getting the type of 'self' (line 97)
        self_193393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self')
        # Setting the type of the member 'testsRun' of a type (line 97)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_193393, 'testsRun', int_193392)
        
        # Assigning a List to a Attribute (line 98):
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_193394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        
        # Getting the type of 'self' (line 98)
        self_193395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self')
        # Setting the type of the member 'skipped' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_193395, 'skipped', list_193394)
        
        # Assigning a List to a Attribute (line 99):
        
        # Obtaining an instance of the builtin type 'list' (line 99)
        list_193396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 99)
        
        # Getting the type of 'self' (line 99)
        self_193397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self')
        # Setting the type of the member 'expectedFailures' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_193397, 'expectedFailures', list_193396)
        
        # Assigning a List to a Attribute (line 100):
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_193398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        
        # Getting the type of 'self' (line 100)
        self_193399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self')
        # Setting the type of the member 'unexpectedSuccesses' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_193399, 'unexpectedSuccesses', list_193398)
        
        # Assigning a Name to a Attribute (line 101):
        # Getting the type of 'False' (line 101)
        False_193400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'False')
        # Getting the type of 'self' (line 101)
        self_193401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'self')
        # Setting the type of the member 'shouldStop' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), self_193401, 'shouldStop', False_193400)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def startTest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'startTest'
        module_type_store = module_type_store.open_function_context('startTest', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ResultWithNoStartTestRunStopTestRun.startTest.__dict__.__setitem__('stypy_localization', localization)
        ResultWithNoStartTestRunStopTestRun.startTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ResultWithNoStartTestRunStopTestRun.startTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        ResultWithNoStartTestRunStopTestRun.startTest.__dict__.__setitem__('stypy_function_name', 'ResultWithNoStartTestRunStopTestRun.startTest')
        ResultWithNoStartTestRunStopTestRun.startTest.__dict__.__setitem__('stypy_param_names_list', ['test'])
        ResultWithNoStartTestRunStopTestRun.startTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        ResultWithNoStartTestRunStopTestRun.startTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ResultWithNoStartTestRunStopTestRun.startTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        ResultWithNoStartTestRunStopTestRun.startTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        ResultWithNoStartTestRunStopTestRun.startTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ResultWithNoStartTestRunStopTestRun.startTest.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ResultWithNoStartTestRunStopTestRun.startTest', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'startTest', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'startTest(...)' code ##################

        pass
        
        # ################# End of 'startTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'startTest' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_193402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193402)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'startTest'
        return stypy_return_type_193402


    @norecursion
    def stopTest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'stopTest'
        module_type_store = module_type_store.open_function_context('stopTest', 106, 4, False)
        # Assigning a type to the variable 'self' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ResultWithNoStartTestRunStopTestRun.stopTest.__dict__.__setitem__('stypy_localization', localization)
        ResultWithNoStartTestRunStopTestRun.stopTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ResultWithNoStartTestRunStopTestRun.stopTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        ResultWithNoStartTestRunStopTestRun.stopTest.__dict__.__setitem__('stypy_function_name', 'ResultWithNoStartTestRunStopTestRun.stopTest')
        ResultWithNoStartTestRunStopTestRun.stopTest.__dict__.__setitem__('stypy_param_names_list', ['test'])
        ResultWithNoStartTestRunStopTestRun.stopTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        ResultWithNoStartTestRunStopTestRun.stopTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ResultWithNoStartTestRunStopTestRun.stopTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        ResultWithNoStartTestRunStopTestRun.stopTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        ResultWithNoStartTestRunStopTestRun.stopTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ResultWithNoStartTestRunStopTestRun.stopTest.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ResultWithNoStartTestRunStopTestRun.stopTest', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'stopTest', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'stopTest(...)' code ##################

        pass
        
        # ################# End of 'stopTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'stopTest' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_193403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193403)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'stopTest'
        return stypy_return_type_193403


    @norecursion
    def addError(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addError'
        module_type_store = module_type_store.open_function_context('addError', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ResultWithNoStartTestRunStopTestRun.addError.__dict__.__setitem__('stypy_localization', localization)
        ResultWithNoStartTestRunStopTestRun.addError.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ResultWithNoStartTestRunStopTestRun.addError.__dict__.__setitem__('stypy_type_store', module_type_store)
        ResultWithNoStartTestRunStopTestRun.addError.__dict__.__setitem__('stypy_function_name', 'ResultWithNoStartTestRunStopTestRun.addError')
        ResultWithNoStartTestRunStopTestRun.addError.__dict__.__setitem__('stypy_param_names_list', ['test'])
        ResultWithNoStartTestRunStopTestRun.addError.__dict__.__setitem__('stypy_varargs_param_name', None)
        ResultWithNoStartTestRunStopTestRun.addError.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ResultWithNoStartTestRunStopTestRun.addError.__dict__.__setitem__('stypy_call_defaults', defaults)
        ResultWithNoStartTestRunStopTestRun.addError.__dict__.__setitem__('stypy_call_varargs', varargs)
        ResultWithNoStartTestRunStopTestRun.addError.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ResultWithNoStartTestRunStopTestRun.addError.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ResultWithNoStartTestRunStopTestRun.addError', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addError', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addError(...)' code ##################

        pass
        
        # ################# End of 'addError(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addError' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_193404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193404)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addError'
        return stypy_return_type_193404


    @norecursion
    def addFailure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addFailure'
        module_type_store = module_type_store.open_function_context('addFailure', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ResultWithNoStartTestRunStopTestRun.addFailure.__dict__.__setitem__('stypy_localization', localization)
        ResultWithNoStartTestRunStopTestRun.addFailure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ResultWithNoStartTestRunStopTestRun.addFailure.__dict__.__setitem__('stypy_type_store', module_type_store)
        ResultWithNoStartTestRunStopTestRun.addFailure.__dict__.__setitem__('stypy_function_name', 'ResultWithNoStartTestRunStopTestRun.addFailure')
        ResultWithNoStartTestRunStopTestRun.addFailure.__dict__.__setitem__('stypy_param_names_list', ['test'])
        ResultWithNoStartTestRunStopTestRun.addFailure.__dict__.__setitem__('stypy_varargs_param_name', None)
        ResultWithNoStartTestRunStopTestRun.addFailure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ResultWithNoStartTestRunStopTestRun.addFailure.__dict__.__setitem__('stypy_call_defaults', defaults)
        ResultWithNoStartTestRunStopTestRun.addFailure.__dict__.__setitem__('stypy_call_varargs', varargs)
        ResultWithNoStartTestRunStopTestRun.addFailure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ResultWithNoStartTestRunStopTestRun.addFailure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ResultWithNoStartTestRunStopTestRun.addFailure', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addFailure', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addFailure(...)' code ##################

        pass
        
        # ################# End of 'addFailure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addFailure' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_193405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193405)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addFailure'
        return stypy_return_type_193405


    @norecursion
    def addSuccess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addSuccess'
        module_type_store = module_type_store.open_function_context('addSuccess', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ResultWithNoStartTestRunStopTestRun.addSuccess.__dict__.__setitem__('stypy_localization', localization)
        ResultWithNoStartTestRunStopTestRun.addSuccess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ResultWithNoStartTestRunStopTestRun.addSuccess.__dict__.__setitem__('stypy_type_store', module_type_store)
        ResultWithNoStartTestRunStopTestRun.addSuccess.__dict__.__setitem__('stypy_function_name', 'ResultWithNoStartTestRunStopTestRun.addSuccess')
        ResultWithNoStartTestRunStopTestRun.addSuccess.__dict__.__setitem__('stypy_param_names_list', ['test'])
        ResultWithNoStartTestRunStopTestRun.addSuccess.__dict__.__setitem__('stypy_varargs_param_name', None)
        ResultWithNoStartTestRunStopTestRun.addSuccess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ResultWithNoStartTestRunStopTestRun.addSuccess.__dict__.__setitem__('stypy_call_defaults', defaults)
        ResultWithNoStartTestRunStopTestRun.addSuccess.__dict__.__setitem__('stypy_call_varargs', varargs)
        ResultWithNoStartTestRunStopTestRun.addSuccess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ResultWithNoStartTestRunStopTestRun.addSuccess.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ResultWithNoStartTestRunStopTestRun.addSuccess', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addSuccess', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addSuccess(...)' code ##################

        pass
        
        # ################# End of 'addSuccess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addSuccess' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_193406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193406)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addSuccess'
        return stypy_return_type_193406


    @norecursion
    def wasSuccessful(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wasSuccessful'
        module_type_store = module_type_store.open_function_context('wasSuccessful', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ResultWithNoStartTestRunStopTestRun.wasSuccessful.__dict__.__setitem__('stypy_localization', localization)
        ResultWithNoStartTestRunStopTestRun.wasSuccessful.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ResultWithNoStartTestRunStopTestRun.wasSuccessful.__dict__.__setitem__('stypy_type_store', module_type_store)
        ResultWithNoStartTestRunStopTestRun.wasSuccessful.__dict__.__setitem__('stypy_function_name', 'ResultWithNoStartTestRunStopTestRun.wasSuccessful')
        ResultWithNoStartTestRunStopTestRun.wasSuccessful.__dict__.__setitem__('stypy_param_names_list', [])
        ResultWithNoStartTestRunStopTestRun.wasSuccessful.__dict__.__setitem__('stypy_varargs_param_name', None)
        ResultWithNoStartTestRunStopTestRun.wasSuccessful.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ResultWithNoStartTestRunStopTestRun.wasSuccessful.__dict__.__setitem__('stypy_call_defaults', defaults)
        ResultWithNoStartTestRunStopTestRun.wasSuccessful.__dict__.__setitem__('stypy_call_varargs', varargs)
        ResultWithNoStartTestRunStopTestRun.wasSuccessful.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ResultWithNoStartTestRunStopTestRun.wasSuccessful.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ResultWithNoStartTestRunStopTestRun.wasSuccessful', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wasSuccessful', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wasSuccessful(...)' code ##################

        # Getting the type of 'True' (line 119)
        True_193407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'stypy_return_type', True_193407)
        
        # ################# End of 'wasSuccessful(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wasSuccessful' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_193408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193408)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wasSuccessful'
        return stypy_return_type_193408


# Assigning a type to the variable 'ResultWithNoStartTestRunStopTestRun' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'ResultWithNoStartTestRunStopTestRun', ResultWithNoStartTestRunStopTestRun)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
