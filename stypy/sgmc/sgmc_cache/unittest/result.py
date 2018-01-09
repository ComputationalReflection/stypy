
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Test result object'''
2: 
3: import os
4: import sys
5: import traceback
6: 
7: from StringIO import StringIO
8: 
9: from . import util
10: from functools import wraps
11: 
12: __unittest = True
13: 
14: def failfast(method):
15:     @wraps(method)
16:     def inner(self, *args, **kw):
17:         if getattr(self, 'failfast', False):
18:             self.stop()
19:         return method(self, *args, **kw)
20:     return inner
21: 
22: STDOUT_LINE = '\nStdout:\n%s'
23: STDERR_LINE = '\nStderr:\n%s'
24: 
25: 
26: class TestResult(object):
27:     '''Holder for test result information.
28: 
29:     Test results are automatically managed by the TestCase and TestSuite
30:     classes, and do not need to be explicitly manipulated by writers of tests.
31: 
32:     Each instance holds the total number of tests run, and collections of
33:     failures and errors that occurred among those test runs. The collections
34:     contain tuples of (testcase, exceptioninfo), where exceptioninfo is the
35:     formatted traceback of the error that occurred.
36:     '''
37:     _previousTestClass = None
38:     _testRunEntered = False
39:     _moduleSetUpFailed = False
40:     def __init__(self, stream=None, descriptions=None, verbosity=None):
41:         self.failfast = False
42:         self.failures = []
43:         self.errors = []
44:         self.testsRun = 0
45:         self.skipped = []
46:         self.expectedFailures = []
47:         self.unexpectedSuccesses = []
48:         self.shouldStop = False
49:         self.buffer = False
50:         self._stdout_buffer = None
51:         self._stderr_buffer = None
52:         self._original_stdout = sys.stdout
53:         self._original_stderr = sys.stderr
54:         self._mirrorOutput = False
55: 
56:     def printErrors(self):
57:         "Called by TestRunner after test run"
58: 
59:     def startTest(self, test):
60:         "Called when the given test is about to be run"
61:         self.testsRun += 1
62:         self._mirrorOutput = False
63:         self._setupStdout()
64: 
65:     def _setupStdout(self):
66:         if self.buffer:
67:             if self._stderr_buffer is None:
68:                 self._stderr_buffer = StringIO()
69:                 self._stdout_buffer = StringIO()
70:             sys.stdout = self._stdout_buffer
71:             sys.stderr = self._stderr_buffer
72: 
73:     def startTestRun(self):
74:         '''Called once before any tests are executed.
75: 
76:         See startTest for a method called before each test.
77:         '''
78: 
79:     def stopTest(self, test):
80:         '''Called when the given test has been run'''
81:         self._restoreStdout()
82:         self._mirrorOutput = False
83: 
84:     def _restoreStdout(self):
85:         if self.buffer:
86:             if self._mirrorOutput:
87:                 output = sys.stdout.getvalue()
88:                 error = sys.stderr.getvalue()
89:                 if output:
90:                     if not output.endswith('\n'):
91:                         output += '\n'
92:                     self._original_stdout.write(STDOUT_LINE % output)
93:                 if error:
94:                     if not error.endswith('\n'):
95:                         error += '\n'
96:                     self._original_stderr.write(STDERR_LINE % error)
97: 
98:             sys.stdout = self._original_stdout
99:             sys.stderr = self._original_stderr
100:             self._stdout_buffer.seek(0)
101:             self._stdout_buffer.truncate()
102:             self._stderr_buffer.seek(0)
103:             self._stderr_buffer.truncate()
104: 
105:     def stopTestRun(self):
106:         '''Called once after all tests are executed.
107: 
108:         See stopTest for a method called after each test.
109:         '''
110: 
111:     @failfast
112:     def addError(self, test, err):
113:         '''Called when an error has occurred. 'err' is a tuple of values as
114:         returned by sys.exc_info().
115:         '''
116:         self.errors.append((test, self._exc_info_to_string(err, test)))
117:         self._mirrorOutput = True
118: 
119:     @failfast
120:     def addFailure(self, test, err):
121:         '''Called when an error has occurred. 'err' is a tuple of values as
122:         returned by sys.exc_info().'''
123:         self.failures.append((test, self._exc_info_to_string(err, test)))
124:         self._mirrorOutput = True
125: 
126:     def addSuccess(self, test):
127:         "Called when a test has completed successfully"
128:         pass
129: 
130:     def addSkip(self, test, reason):
131:         '''Called when a test is skipped.'''
132:         self.skipped.append((test, reason))
133: 
134:     def addExpectedFailure(self, test, err):
135:         '''Called when an expected failure/error occured.'''
136:         self.expectedFailures.append(
137:             (test, self._exc_info_to_string(err, test)))
138: 
139:     @failfast
140:     def addUnexpectedSuccess(self, test):
141:         '''Called when a test was expected to fail, but succeed.'''
142:         self.unexpectedSuccesses.append(test)
143: 
144:     def wasSuccessful(self):
145:         "Tells whether or not this result was a success"
146:         return len(self.failures) == len(self.errors) == 0
147: 
148:     def stop(self):
149:         "Indicates that the tests should be aborted"
150:         self.shouldStop = True
151: 
152:     def _exc_info_to_string(self, err, test):
153:         '''Converts a sys.exc_info()-style tuple of values into a string.'''
154:         exctype, value, tb = err
155:         # Skip test runner traceback levels
156:         while tb and self._is_relevant_tb_level(tb):
157:             tb = tb.tb_next
158: 
159:         if exctype is test.failureException:
160:             # Skip assert*() traceback levels
161:             length = self._count_relevant_tb_levels(tb)
162:             msgLines = traceback.format_exception(exctype, value, tb, length)
163:         else:
164:             msgLines = traceback.format_exception(exctype, value, tb)
165: 
166:         if self.buffer:
167:             output = sys.stdout.getvalue()
168:             error = sys.stderr.getvalue()
169:             if output:
170:                 if not output.endswith('\n'):
171:                     output += '\n'
172:                 msgLines.append(STDOUT_LINE % output)
173:             if error:
174:                 if not error.endswith('\n'):
175:                     error += '\n'
176:                 msgLines.append(STDERR_LINE % error)
177:         return ''.join(msgLines)
178: 
179: 
180:     def _is_relevant_tb_level(self, tb):
181:         return '__unittest' in tb.tb_frame.f_globals
182: 
183:     def _count_relevant_tb_levels(self, tb):
184:         length = 0
185:         while tb and not self._is_relevant_tb_level(tb):
186:             length += 1
187:             tb = tb.tb_next
188:         return length
189: 
190:     def __repr__(self):
191:         return ("<%s run=%i errors=%i failures=%i>" %
192:                (util.strclass(self.__class__), self.testsRun, len(self.errors),
193:                 len(self.failures)))
194: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_190728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Test result object')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import traceback' statement (line 5)
import traceback

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'traceback', traceback, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from StringIO import StringIO' statement (line 7)
from StringIO import StringIO

import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'StringIO', None, module_type_store, ['StringIO'], [StringIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from unittest import util' statement (line 9)
from unittest import util

import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'unittest', None, module_type_store, ['util'], [util])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from functools import wraps' statement (line 10)
from functools import wraps

import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'functools', None, module_type_store, ['wraps'], [wraps])


# Assigning a Name to a Name (line 12):

# Assigning a Name to a Name (line 12):
# Getting the type of 'True' (line 12)
True_190729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'True')
# Assigning a type to the variable '__unittest' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__unittest', True_190729)

@norecursion
def failfast(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'failfast'
    module_type_store = module_type_store.open_function_context('failfast', 14, 0, False)
    
    # Passed parameters checking function
    failfast.stypy_localization = localization
    failfast.stypy_type_of_self = None
    failfast.stypy_type_store = module_type_store
    failfast.stypy_function_name = 'failfast'
    failfast.stypy_param_names_list = ['method']
    failfast.stypy_varargs_param_name = None
    failfast.stypy_kwargs_param_name = None
    failfast.stypy_call_defaults = defaults
    failfast.stypy_call_varargs = varargs
    failfast.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'failfast', ['method'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'failfast', localization, ['method'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'failfast(...)' code ##################


    @norecursion
    def inner(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inner'
        module_type_store = module_type_store.open_function_context('inner', 15, 4, False)
        
        # Passed parameters checking function
        inner.stypy_localization = localization
        inner.stypy_type_of_self = None
        inner.stypy_type_store = module_type_store
        inner.stypy_function_name = 'inner'
        inner.stypy_param_names_list = ['self']
        inner.stypy_varargs_param_name = 'args'
        inner.stypy_kwargs_param_name = 'kw'
        inner.stypy_call_defaults = defaults
        inner.stypy_call_varargs = varargs
        inner.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'inner', ['self'], 'args', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'inner', localization, ['self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'inner(...)' code ##################

        
        
        # Call to getattr(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'self' (line 17)
        self_190731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'self', False)
        str_190732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'str', 'failfast')
        # Getting the type of 'False' (line 17)
        False_190733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 37), 'False', False)
        # Processing the call keyword arguments (line 17)
        kwargs_190734 = {}
        # Getting the type of 'getattr' (line 17)
        getattr_190730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'getattr', False)
        # Calling getattr(args, kwargs) (line 17)
        getattr_call_result_190735 = invoke(stypy.reporting.localization.Localization(__file__, 17, 11), getattr_190730, *[self_190731, str_190732, False_190733], **kwargs_190734)
        
        # Testing the type of an if condition (line 17)
        if_condition_190736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 8), getattr_call_result_190735)
        # Assigning a type to the variable 'if_condition_190736' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'if_condition_190736', if_condition_190736)
        # SSA begins for if statement (line 17)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to stop(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_190739 = {}
        # Getting the type of 'self' (line 18)
        self_190737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'self', False)
        # Obtaining the member 'stop' of a type (line 18)
        stop_190738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 12), self_190737, 'stop')
        # Calling stop(args, kwargs) (line 18)
        stop_call_result_190740 = invoke(stypy.reporting.localization.Localization(__file__, 18, 12), stop_190738, *[], **kwargs_190739)
        
        # SSA join for if statement (line 17)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to method(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'self' (line 19)
        self_190742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'self', False)
        # Getting the type of 'args' (line 19)
        args_190743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 29), 'args', False)
        # Processing the call keyword arguments (line 19)
        # Getting the type of 'kw' (line 19)
        kw_190744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 37), 'kw', False)
        kwargs_190745 = {'kw_190744': kw_190744}
        # Getting the type of 'method' (line 19)
        method_190741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'method', False)
        # Calling method(args, kwargs) (line 19)
        method_call_result_190746 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), method_190741, *[self_190742, args_190743], **kwargs_190745)
        
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type', method_call_result_190746)
        
        # ################# End of 'inner(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inner' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_190747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190747)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inner'
        return stypy_return_type_190747

    # Assigning a type to the variable 'inner' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'inner', inner)
    # Getting the type of 'inner' (line 20)
    inner_190748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'inner')
    # Assigning a type to the variable 'stypy_return_type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type', inner_190748)
    
    # ################# End of 'failfast(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'failfast' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_190749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190749)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'failfast'
    return stypy_return_type_190749

# Assigning a type to the variable 'failfast' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'failfast', failfast)

# Assigning a Str to a Name (line 22):

# Assigning a Str to a Name (line 22):
str_190750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'str', '\nStdout:\n%s')
# Assigning a type to the variable 'STDOUT_LINE' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'STDOUT_LINE', str_190750)

# Assigning a Str to a Name (line 23):

# Assigning a Str to a Name (line 23):
str_190751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'str', '\nStderr:\n%s')
# Assigning a type to the variable 'STDERR_LINE' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'STDERR_LINE', str_190751)
# Declaration of the 'TestResult' class

class TestResult(object, ):
    str_190752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', 'Holder for test result information.\n\n    Test results are automatically managed by the TestCase and TestSuite\n    classes, and do not need to be explicitly manipulated by writers of tests.\n\n    Each instance holds the total number of tests run, and collections of\n    failures and errors that occurred among those test runs. The collections\n    contain tuples of (testcase, exceptioninfo), where exceptioninfo is the\n    formatted traceback of the error that occurred.\n    ')
    
    # Assigning a Name to a Name (line 37):
    
    # Assigning a Name to a Name (line 38):
    
    # Assigning a Name to a Name (line 39):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 40)
        None_190753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 30), 'None')
        # Getting the type of 'None' (line 40)
        None_190754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 49), 'None')
        # Getting the type of 'None' (line 40)
        None_190755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 65), 'None')
        defaults = [None_190753, None_190754, None_190755]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.__init__', ['stream', 'descriptions', 'verbosity'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['stream', 'descriptions', 'verbosity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 41):
        
        # Assigning a Name to a Attribute (line 41):
        # Getting the type of 'False' (line 41)
        False_190756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'False')
        # Getting the type of 'self' (line 41)
        self_190757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'failfast' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_190757, 'failfast', False_190756)
        
        # Assigning a List to a Attribute (line 42):
        
        # Assigning a List to a Attribute (line 42):
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_190758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        
        # Getting the type of 'self' (line 42)
        self_190759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member 'failures' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_190759, 'failures', list_190758)
        
        # Assigning a List to a Attribute (line 43):
        
        # Assigning a List to a Attribute (line 43):
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_190760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        
        # Getting the type of 'self' (line 43)
        self_190761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'errors' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_190761, 'errors', list_190760)
        
        # Assigning a Num to a Attribute (line 44):
        
        # Assigning a Num to a Attribute (line 44):
        int_190762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'int')
        # Getting the type of 'self' (line 44)
        self_190763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'self')
        # Setting the type of the member 'testsRun' of a type (line 44)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), self_190763, 'testsRun', int_190762)
        
        # Assigning a List to a Attribute (line 45):
        
        # Assigning a List to a Attribute (line 45):
        
        # Obtaining an instance of the builtin type 'list' (line 45)
        list_190764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 45)
        
        # Getting the type of 'self' (line 45)
        self_190765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self')
        # Setting the type of the member 'skipped' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_190765, 'skipped', list_190764)
        
        # Assigning a List to a Attribute (line 46):
        
        # Assigning a List to a Attribute (line 46):
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_190766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        
        # Getting the type of 'self' (line 46)
        self_190767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'expectedFailures' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_190767, 'expectedFailures', list_190766)
        
        # Assigning a List to a Attribute (line 47):
        
        # Assigning a List to a Attribute (line 47):
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_190768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        
        # Getting the type of 'self' (line 47)
        self_190769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member 'unexpectedSuccesses' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_190769, 'unexpectedSuccesses', list_190768)
        
        # Assigning a Name to a Attribute (line 48):
        
        # Assigning a Name to a Attribute (line 48):
        # Getting the type of 'False' (line 48)
        False_190770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'False')
        # Getting the type of 'self' (line 48)
        self_190771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member 'shouldStop' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_190771, 'shouldStop', False_190770)
        
        # Assigning a Name to a Attribute (line 49):
        
        # Assigning a Name to a Attribute (line 49):
        # Getting the type of 'False' (line 49)
        False_190772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'False')
        # Getting the type of 'self' (line 49)
        self_190773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self')
        # Setting the type of the member 'buffer' of a type (line 49)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_190773, 'buffer', False_190772)
        
        # Assigning a Name to a Attribute (line 50):
        
        # Assigning a Name to a Attribute (line 50):
        # Getting the type of 'None' (line 50)
        None_190774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 30), 'None')
        # Getting the type of 'self' (line 50)
        self_190775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'self')
        # Setting the type of the member '_stdout_buffer' of a type (line 50)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), self_190775, '_stdout_buffer', None_190774)
        
        # Assigning a Name to a Attribute (line 51):
        
        # Assigning a Name to a Attribute (line 51):
        # Getting the type of 'None' (line 51)
        None_190776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'None')
        # Getting the type of 'self' (line 51)
        self_190777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self')
        # Setting the type of the member '_stderr_buffer' of a type (line 51)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_190777, '_stderr_buffer', None_190776)
        
        # Assigning a Attribute to a Attribute (line 52):
        
        # Assigning a Attribute to a Attribute (line 52):
        # Getting the type of 'sys' (line 52)
        sys_190778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 32), 'sys')
        # Obtaining the member 'stdout' of a type (line 52)
        stdout_190779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 32), sys_190778, 'stdout')
        # Getting the type of 'self' (line 52)
        self_190780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self')
        # Setting the type of the member '_original_stdout' of a type (line 52)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_190780, '_original_stdout', stdout_190779)
        
        # Assigning a Attribute to a Attribute (line 53):
        
        # Assigning a Attribute to a Attribute (line 53):
        # Getting the type of 'sys' (line 53)
        sys_190781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 32), 'sys')
        # Obtaining the member 'stderr' of a type (line 53)
        stderr_190782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 32), sys_190781, 'stderr')
        # Getting the type of 'self' (line 53)
        self_190783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self')
        # Setting the type of the member '_original_stderr' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_190783, '_original_stderr', stderr_190782)
        
        # Assigning a Name to a Attribute (line 54):
        
        # Assigning a Name to a Attribute (line 54):
        # Getting the type of 'False' (line 54)
        False_190784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'False')
        # Getting the type of 'self' (line 54)
        self_190785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self')
        # Setting the type of the member '_mirrorOutput' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_190785, '_mirrorOutput', False_190784)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def printErrors(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'printErrors'
        module_type_store = module_type_store.open_function_context('printErrors', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.printErrors.__dict__.__setitem__('stypy_localization', localization)
        TestResult.printErrors.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.printErrors.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.printErrors.__dict__.__setitem__('stypy_function_name', 'TestResult.printErrors')
        TestResult.printErrors.__dict__.__setitem__('stypy_param_names_list', [])
        TestResult.printErrors.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.printErrors.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.printErrors.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.printErrors.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.printErrors.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.printErrors.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.printErrors', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'printErrors', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'printErrors(...)' code ##################

        str_190786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 8), 'str', 'Called by TestRunner after test run')
        
        # ################# End of 'printErrors(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'printErrors' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_190787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190787)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'printErrors'
        return stypy_return_type_190787


    @norecursion
    def startTest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'startTest'
        module_type_store = module_type_store.open_function_context('startTest', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.startTest.__dict__.__setitem__('stypy_localization', localization)
        TestResult.startTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.startTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.startTest.__dict__.__setitem__('stypy_function_name', 'TestResult.startTest')
        TestResult.startTest.__dict__.__setitem__('stypy_param_names_list', ['test'])
        TestResult.startTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.startTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.startTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.startTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.startTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.startTest.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.startTest', ['test'], None, None, defaults, varargs, kwargs)

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

        str_190788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'str', 'Called when the given test is about to be run')
        
        # Getting the type of 'self' (line 61)
        self_190789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Obtaining the member 'testsRun' of a type (line 61)
        testsRun_190790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_190789, 'testsRun')
        int_190791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 25), 'int')
        # Applying the binary operator '+=' (line 61)
        result_iadd_190792 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 8), '+=', testsRun_190790, int_190791)
        # Getting the type of 'self' (line 61)
        self_190793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member 'testsRun' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_190793, 'testsRun', result_iadd_190792)
        
        
        # Assigning a Name to a Attribute (line 62):
        
        # Assigning a Name to a Attribute (line 62):
        # Getting the type of 'False' (line 62)
        False_190794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 29), 'False')
        # Getting the type of 'self' (line 62)
        self_190795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member '_mirrorOutput' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_190795, '_mirrorOutput', False_190794)
        
        # Call to _setupStdout(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_190798 = {}
        # Getting the type of 'self' (line 63)
        self_190796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self', False)
        # Obtaining the member '_setupStdout' of a type (line 63)
        _setupStdout_190797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_190796, '_setupStdout')
        # Calling _setupStdout(args, kwargs) (line 63)
        _setupStdout_call_result_190799 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), _setupStdout_190797, *[], **kwargs_190798)
        
        
        # ################# End of 'startTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'startTest' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_190800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190800)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'startTest'
        return stypy_return_type_190800


    @norecursion
    def _setupStdout(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_setupStdout'
        module_type_store = module_type_store.open_function_context('_setupStdout', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult._setupStdout.__dict__.__setitem__('stypy_localization', localization)
        TestResult._setupStdout.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult._setupStdout.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult._setupStdout.__dict__.__setitem__('stypy_function_name', 'TestResult._setupStdout')
        TestResult._setupStdout.__dict__.__setitem__('stypy_param_names_list', [])
        TestResult._setupStdout.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult._setupStdout.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult._setupStdout.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult._setupStdout.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult._setupStdout.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult._setupStdout.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult._setupStdout', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_setupStdout', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_setupStdout(...)' code ##################

        
        # Getting the type of 'self' (line 66)
        self_190801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'self')
        # Obtaining the member 'buffer' of a type (line 66)
        buffer_190802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 11), self_190801, 'buffer')
        # Testing the type of an if condition (line 66)
        if_condition_190803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 8), buffer_190802)
        # Assigning a type to the variable 'if_condition_190803' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'if_condition_190803', if_condition_190803)
        # SSA begins for if statement (line 66)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 67)
        # Getting the type of 'self' (line 67)
        self_190804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'self')
        # Obtaining the member '_stderr_buffer' of a type (line 67)
        _stderr_buffer_190805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 15), self_190804, '_stderr_buffer')
        # Getting the type of 'None' (line 67)
        None_190806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 38), 'None')
        
        (may_be_190807, more_types_in_union_190808) = may_be_none(_stderr_buffer_190805, None_190806)

        if may_be_190807:

            if more_types_in_union_190808:
                # Runtime conditional SSA (line 67)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 68):
            
            # Assigning a Call to a Attribute (line 68):
            
            # Call to StringIO(...): (line 68)
            # Processing the call keyword arguments (line 68)
            kwargs_190810 = {}
            # Getting the type of 'StringIO' (line 68)
            StringIO_190809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 38), 'StringIO', False)
            # Calling StringIO(args, kwargs) (line 68)
            StringIO_call_result_190811 = invoke(stypy.reporting.localization.Localization(__file__, 68, 38), StringIO_190809, *[], **kwargs_190810)
            
            # Getting the type of 'self' (line 68)
            self_190812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'self')
            # Setting the type of the member '_stderr_buffer' of a type (line 68)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), self_190812, '_stderr_buffer', StringIO_call_result_190811)
            
            # Assigning a Call to a Attribute (line 69):
            
            # Assigning a Call to a Attribute (line 69):
            
            # Call to StringIO(...): (line 69)
            # Processing the call keyword arguments (line 69)
            kwargs_190814 = {}
            # Getting the type of 'StringIO' (line 69)
            StringIO_190813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 38), 'StringIO', False)
            # Calling StringIO(args, kwargs) (line 69)
            StringIO_call_result_190815 = invoke(stypy.reporting.localization.Localization(__file__, 69, 38), StringIO_190813, *[], **kwargs_190814)
            
            # Getting the type of 'self' (line 69)
            self_190816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'self')
            # Setting the type of the member '_stdout_buffer' of a type (line 69)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), self_190816, '_stdout_buffer', StringIO_call_result_190815)

            if more_types_in_union_190808:
                # SSA join for if statement (line 67)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Attribute (line 70):
        
        # Assigning a Attribute to a Attribute (line 70):
        # Getting the type of 'self' (line 70)
        self_190817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'self')
        # Obtaining the member '_stdout_buffer' of a type (line 70)
        _stdout_buffer_190818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 25), self_190817, '_stdout_buffer')
        # Getting the type of 'sys' (line 70)
        sys_190819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'sys')
        # Setting the type of the member 'stdout' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), sys_190819, 'stdout', _stdout_buffer_190818)
        
        # Assigning a Attribute to a Attribute (line 71):
        
        # Assigning a Attribute to a Attribute (line 71):
        # Getting the type of 'self' (line 71)
        self_190820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'self')
        # Obtaining the member '_stderr_buffer' of a type (line 71)
        _stderr_buffer_190821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 25), self_190820, '_stderr_buffer')
        # Getting the type of 'sys' (line 71)
        sys_190822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'sys')
        # Setting the type of the member 'stderr' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), sys_190822, 'stderr', _stderr_buffer_190821)
        # SSA join for if statement (line 66)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_setupStdout(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_setupStdout' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_190823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190823)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_setupStdout'
        return stypy_return_type_190823


    @norecursion
    def startTestRun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'startTestRun'
        module_type_store = module_type_store.open_function_context('startTestRun', 73, 4, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.startTestRun.__dict__.__setitem__('stypy_localization', localization)
        TestResult.startTestRun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.startTestRun.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.startTestRun.__dict__.__setitem__('stypy_function_name', 'TestResult.startTestRun')
        TestResult.startTestRun.__dict__.__setitem__('stypy_param_names_list', [])
        TestResult.startTestRun.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.startTestRun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.startTestRun.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.startTestRun.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.startTestRun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.startTestRun.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.startTestRun', [], None, None, defaults, varargs, kwargs)

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

        str_190824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, (-1)), 'str', 'Called once before any tests are executed.\n\n        See startTest for a method called before each test.\n        ')
        
        # ################# End of 'startTestRun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'startTestRun' in the type store
        # Getting the type of 'stypy_return_type' (line 73)
        stypy_return_type_190825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190825)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'startTestRun'
        return stypy_return_type_190825


    @norecursion
    def stopTest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'stopTest'
        module_type_store = module_type_store.open_function_context('stopTest', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.stopTest.__dict__.__setitem__('stypy_localization', localization)
        TestResult.stopTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.stopTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.stopTest.__dict__.__setitem__('stypy_function_name', 'TestResult.stopTest')
        TestResult.stopTest.__dict__.__setitem__('stypy_param_names_list', ['test'])
        TestResult.stopTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.stopTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.stopTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.stopTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.stopTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.stopTest.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.stopTest', ['test'], None, None, defaults, varargs, kwargs)

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

        str_190826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 8), 'str', 'Called when the given test has been run')
        
        # Call to _restoreStdout(...): (line 81)
        # Processing the call keyword arguments (line 81)
        kwargs_190829 = {}
        # Getting the type of 'self' (line 81)
        self_190827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self', False)
        # Obtaining the member '_restoreStdout' of a type (line 81)
        _restoreStdout_190828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_190827, '_restoreStdout')
        # Calling _restoreStdout(args, kwargs) (line 81)
        _restoreStdout_call_result_190830 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), _restoreStdout_190828, *[], **kwargs_190829)
        
        
        # Assigning a Name to a Attribute (line 82):
        
        # Assigning a Name to a Attribute (line 82):
        # Getting the type of 'False' (line 82)
        False_190831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'False')
        # Getting the type of 'self' (line 82)
        self_190832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self')
        # Setting the type of the member '_mirrorOutput' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_190832, '_mirrorOutput', False_190831)
        
        # ################# End of 'stopTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'stopTest' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_190833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190833)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'stopTest'
        return stypy_return_type_190833


    @norecursion
    def _restoreStdout(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_restoreStdout'
        module_type_store = module_type_store.open_function_context('_restoreStdout', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult._restoreStdout.__dict__.__setitem__('stypy_localization', localization)
        TestResult._restoreStdout.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult._restoreStdout.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult._restoreStdout.__dict__.__setitem__('stypy_function_name', 'TestResult._restoreStdout')
        TestResult._restoreStdout.__dict__.__setitem__('stypy_param_names_list', [])
        TestResult._restoreStdout.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult._restoreStdout.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult._restoreStdout.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult._restoreStdout.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult._restoreStdout.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult._restoreStdout.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult._restoreStdout', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_restoreStdout', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_restoreStdout(...)' code ##################

        
        # Getting the type of 'self' (line 85)
        self_190834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'self')
        # Obtaining the member 'buffer' of a type (line 85)
        buffer_190835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 11), self_190834, 'buffer')
        # Testing the type of an if condition (line 85)
        if_condition_190836 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 8), buffer_190835)
        # Assigning a type to the variable 'if_condition_190836' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'if_condition_190836', if_condition_190836)
        # SSA begins for if statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 86)
        self_190837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'self')
        # Obtaining the member '_mirrorOutput' of a type (line 86)
        _mirrorOutput_190838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), self_190837, '_mirrorOutput')
        # Testing the type of an if condition (line 86)
        if_condition_190839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 12), _mirrorOutput_190838)
        # Assigning a type to the variable 'if_condition_190839' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'if_condition_190839', if_condition_190839)
        # SSA begins for if statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 87):
        
        # Assigning a Call to a Name (line 87):
        
        # Call to getvalue(...): (line 87)
        # Processing the call keyword arguments (line 87)
        kwargs_190843 = {}
        # Getting the type of 'sys' (line 87)
        sys_190840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 87)
        stdout_190841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 25), sys_190840, 'stdout')
        # Obtaining the member 'getvalue' of a type (line 87)
        getvalue_190842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 25), stdout_190841, 'getvalue')
        # Calling getvalue(args, kwargs) (line 87)
        getvalue_call_result_190844 = invoke(stypy.reporting.localization.Localization(__file__, 87, 25), getvalue_190842, *[], **kwargs_190843)
        
        # Assigning a type to the variable 'output' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'output', getvalue_call_result_190844)
        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Call to getvalue(...): (line 88)
        # Processing the call keyword arguments (line 88)
        kwargs_190848 = {}
        # Getting the type of 'sys' (line 88)
        sys_190845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 88)
        stderr_190846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), sys_190845, 'stderr')
        # Obtaining the member 'getvalue' of a type (line 88)
        getvalue_190847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), stderr_190846, 'getvalue')
        # Calling getvalue(args, kwargs) (line 88)
        getvalue_call_result_190849 = invoke(stypy.reporting.localization.Localization(__file__, 88, 24), getvalue_190847, *[], **kwargs_190848)
        
        # Assigning a type to the variable 'error' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'error', getvalue_call_result_190849)
        
        # Getting the type of 'output' (line 89)
        output_190850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'output')
        # Testing the type of an if condition (line 89)
        if_condition_190851 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 16), output_190850)
        # Assigning a type to the variable 'if_condition_190851' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'if_condition_190851', if_condition_190851)
        # SSA begins for if statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to endswith(...): (line 90)
        # Processing the call arguments (line 90)
        str_190854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 43), 'str', '\n')
        # Processing the call keyword arguments (line 90)
        kwargs_190855 = {}
        # Getting the type of 'output' (line 90)
        output_190852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 27), 'output', False)
        # Obtaining the member 'endswith' of a type (line 90)
        endswith_190853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 27), output_190852, 'endswith')
        # Calling endswith(args, kwargs) (line 90)
        endswith_call_result_190856 = invoke(stypy.reporting.localization.Localization(__file__, 90, 27), endswith_190853, *[str_190854], **kwargs_190855)
        
        # Applying the 'not' unary operator (line 90)
        result_not__190857 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), 'not', endswith_call_result_190856)
        
        # Testing the type of an if condition (line 90)
        if_condition_190858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 20), result_not__190857)
        # Assigning a type to the variable 'if_condition_190858' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'if_condition_190858', if_condition_190858)
        # SSA begins for if statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'output' (line 91)
        output_190859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'output')
        str_190860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 34), 'str', '\n')
        # Applying the binary operator '+=' (line 91)
        result_iadd_190861 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 24), '+=', output_190859, str_190860)
        # Assigning a type to the variable 'output' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'output', result_iadd_190861)
        
        # SSA join for if statement (line 90)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'STDOUT_LINE' (line 92)
        STDOUT_LINE_190865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 48), 'STDOUT_LINE', False)
        # Getting the type of 'output' (line 92)
        output_190866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 62), 'output', False)
        # Applying the binary operator '%' (line 92)
        result_mod_190867 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 48), '%', STDOUT_LINE_190865, output_190866)
        
        # Processing the call keyword arguments (line 92)
        kwargs_190868 = {}
        # Getting the type of 'self' (line 92)
        self_190862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'self', False)
        # Obtaining the member '_original_stdout' of a type (line 92)
        _original_stdout_190863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 20), self_190862, '_original_stdout')
        # Obtaining the member 'write' of a type (line 92)
        write_190864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 20), _original_stdout_190863, 'write')
        # Calling write(args, kwargs) (line 92)
        write_call_result_190869 = invoke(stypy.reporting.localization.Localization(__file__, 92, 20), write_190864, *[result_mod_190867], **kwargs_190868)
        
        # SSA join for if statement (line 89)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'error' (line 93)
        error_190870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'error')
        # Testing the type of an if condition (line 93)
        if_condition_190871 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 16), error_190870)
        # Assigning a type to the variable 'if_condition_190871' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'if_condition_190871', if_condition_190871)
        # SSA begins for if statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to endswith(...): (line 94)
        # Processing the call arguments (line 94)
        str_190874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 42), 'str', '\n')
        # Processing the call keyword arguments (line 94)
        kwargs_190875 = {}
        # Getting the type of 'error' (line 94)
        error_190872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'error', False)
        # Obtaining the member 'endswith' of a type (line 94)
        endswith_190873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 27), error_190872, 'endswith')
        # Calling endswith(args, kwargs) (line 94)
        endswith_call_result_190876 = invoke(stypy.reporting.localization.Localization(__file__, 94, 27), endswith_190873, *[str_190874], **kwargs_190875)
        
        # Applying the 'not' unary operator (line 94)
        result_not__190877 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 23), 'not', endswith_call_result_190876)
        
        # Testing the type of an if condition (line 94)
        if_condition_190878 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 20), result_not__190877)
        # Assigning a type to the variable 'if_condition_190878' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'if_condition_190878', if_condition_190878)
        # SSA begins for if statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'error' (line 95)
        error_190879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'error')
        str_190880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 33), 'str', '\n')
        # Applying the binary operator '+=' (line 95)
        result_iadd_190881 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 24), '+=', error_190879, str_190880)
        # Assigning a type to the variable 'error' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'error', result_iadd_190881)
        
        # SSA join for if statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'STDERR_LINE' (line 96)
        STDERR_LINE_190885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 48), 'STDERR_LINE', False)
        # Getting the type of 'error' (line 96)
        error_190886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 62), 'error', False)
        # Applying the binary operator '%' (line 96)
        result_mod_190887 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 48), '%', STDERR_LINE_190885, error_190886)
        
        # Processing the call keyword arguments (line 96)
        kwargs_190888 = {}
        # Getting the type of 'self' (line 96)
        self_190882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 20), 'self', False)
        # Obtaining the member '_original_stderr' of a type (line 96)
        _original_stderr_190883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 20), self_190882, '_original_stderr')
        # Obtaining the member 'write' of a type (line 96)
        write_190884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 20), _original_stderr_190883, 'write')
        # Calling write(args, kwargs) (line 96)
        write_call_result_190889 = invoke(stypy.reporting.localization.Localization(__file__, 96, 20), write_190884, *[result_mod_190887], **kwargs_190888)
        
        # SSA join for if statement (line 93)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 86)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 98):
        
        # Assigning a Attribute to a Attribute (line 98):
        # Getting the type of 'self' (line 98)
        self_190890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'self')
        # Obtaining the member '_original_stdout' of a type (line 98)
        _original_stdout_190891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 25), self_190890, '_original_stdout')
        # Getting the type of 'sys' (line 98)
        sys_190892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'sys')
        # Setting the type of the member 'stdout' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), sys_190892, 'stdout', _original_stdout_190891)
        
        # Assigning a Attribute to a Attribute (line 99):
        
        # Assigning a Attribute to a Attribute (line 99):
        # Getting the type of 'self' (line 99)
        self_190893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'self')
        # Obtaining the member '_original_stderr' of a type (line 99)
        _original_stderr_190894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 25), self_190893, '_original_stderr')
        # Getting the type of 'sys' (line 99)
        sys_190895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'sys')
        # Setting the type of the member 'stderr' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), sys_190895, 'stderr', _original_stderr_190894)
        
        # Call to seek(...): (line 100)
        # Processing the call arguments (line 100)
        int_190899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 37), 'int')
        # Processing the call keyword arguments (line 100)
        kwargs_190900 = {}
        # Getting the type of 'self' (line 100)
        self_190896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self', False)
        # Obtaining the member '_stdout_buffer' of a type (line 100)
        _stdout_buffer_190897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_190896, '_stdout_buffer')
        # Obtaining the member 'seek' of a type (line 100)
        seek_190898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), _stdout_buffer_190897, 'seek')
        # Calling seek(args, kwargs) (line 100)
        seek_call_result_190901 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), seek_190898, *[int_190899], **kwargs_190900)
        
        
        # Call to truncate(...): (line 101)
        # Processing the call keyword arguments (line 101)
        kwargs_190905 = {}
        # Getting the type of 'self' (line 101)
        self_190902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'self', False)
        # Obtaining the member '_stdout_buffer' of a type (line 101)
        _stdout_buffer_190903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), self_190902, '_stdout_buffer')
        # Obtaining the member 'truncate' of a type (line 101)
        truncate_190904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), _stdout_buffer_190903, 'truncate')
        # Calling truncate(args, kwargs) (line 101)
        truncate_call_result_190906 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), truncate_190904, *[], **kwargs_190905)
        
        
        # Call to seek(...): (line 102)
        # Processing the call arguments (line 102)
        int_190910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 37), 'int')
        # Processing the call keyword arguments (line 102)
        kwargs_190911 = {}
        # Getting the type of 'self' (line 102)
        self_190907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self', False)
        # Obtaining the member '_stderr_buffer' of a type (line 102)
        _stderr_buffer_190908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), self_190907, '_stderr_buffer')
        # Obtaining the member 'seek' of a type (line 102)
        seek_190909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), _stderr_buffer_190908, 'seek')
        # Calling seek(args, kwargs) (line 102)
        seek_call_result_190912 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), seek_190909, *[int_190910], **kwargs_190911)
        
        
        # Call to truncate(...): (line 103)
        # Processing the call keyword arguments (line 103)
        kwargs_190916 = {}
        # Getting the type of 'self' (line 103)
        self_190913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self', False)
        # Obtaining the member '_stderr_buffer' of a type (line 103)
        _stderr_buffer_190914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_190913, '_stderr_buffer')
        # Obtaining the member 'truncate' of a type (line 103)
        truncate_190915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), _stderr_buffer_190914, 'truncate')
        # Calling truncate(args, kwargs) (line 103)
        truncate_call_result_190917 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), truncate_190915, *[], **kwargs_190916)
        
        # SSA join for if statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_restoreStdout(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_restoreStdout' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_190918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190918)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_restoreStdout'
        return stypy_return_type_190918


    @norecursion
    def stopTestRun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'stopTestRun'
        module_type_store = module_type_store.open_function_context('stopTestRun', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.stopTestRun.__dict__.__setitem__('stypy_localization', localization)
        TestResult.stopTestRun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.stopTestRun.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.stopTestRun.__dict__.__setitem__('stypy_function_name', 'TestResult.stopTestRun')
        TestResult.stopTestRun.__dict__.__setitem__('stypy_param_names_list', [])
        TestResult.stopTestRun.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.stopTestRun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.stopTestRun.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.stopTestRun.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.stopTestRun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.stopTestRun.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.stopTestRun', [], None, None, defaults, varargs, kwargs)

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

        str_190919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', 'Called once after all tests are executed.\n\n        See stopTest for a method called after each test.\n        ')
        
        # ################# End of 'stopTestRun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'stopTestRun' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_190920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190920)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'stopTestRun'
        return stypy_return_type_190920


    @norecursion
    def addError(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addError'
        module_type_store = module_type_store.open_function_context('addError', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.addError.__dict__.__setitem__('stypy_localization', localization)
        TestResult.addError.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.addError.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.addError.__dict__.__setitem__('stypy_function_name', 'TestResult.addError')
        TestResult.addError.__dict__.__setitem__('stypy_param_names_list', ['test', 'err'])
        TestResult.addError.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.addError.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.addError.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.addError.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.addError.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.addError.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.addError', ['test', 'err'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addError', localization, ['test', 'err'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addError(...)' code ##################

        str_190921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, (-1)), 'str', "Called when an error has occurred. 'err' is a tuple of values as\n        returned by sys.exc_info().\n        ")
        
        # Call to append(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Obtaining an instance of the builtin type 'tuple' (line 116)
        tuple_190925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 116)
        # Adding element type (line 116)
        # Getting the type of 'test' (line 116)
        test_190926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 28), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 28), tuple_190925, test_190926)
        # Adding element type (line 116)
        
        # Call to _exc_info_to_string(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'err' (line 116)
        err_190929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 59), 'err', False)
        # Getting the type of 'test' (line 116)
        test_190930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 64), 'test', False)
        # Processing the call keyword arguments (line 116)
        kwargs_190931 = {}
        # Getting the type of 'self' (line 116)
        self_190927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 34), 'self', False)
        # Obtaining the member '_exc_info_to_string' of a type (line 116)
        _exc_info_to_string_190928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 34), self_190927, '_exc_info_to_string')
        # Calling _exc_info_to_string(args, kwargs) (line 116)
        _exc_info_to_string_call_result_190932 = invoke(stypy.reporting.localization.Localization(__file__, 116, 34), _exc_info_to_string_190928, *[err_190929, test_190930], **kwargs_190931)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 28), tuple_190925, _exc_info_to_string_call_result_190932)
        
        # Processing the call keyword arguments (line 116)
        kwargs_190933 = {}
        # Getting the type of 'self' (line 116)
        self_190922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self', False)
        # Obtaining the member 'errors' of a type (line 116)
        errors_190923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_190922, 'errors')
        # Obtaining the member 'append' of a type (line 116)
        append_190924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), errors_190923, 'append')
        # Calling append(args, kwargs) (line 116)
        append_call_result_190934 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), append_190924, *[tuple_190925], **kwargs_190933)
        
        
        # Assigning a Name to a Attribute (line 117):
        
        # Assigning a Name to a Attribute (line 117):
        # Getting the type of 'True' (line 117)
        True_190935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'True')
        # Getting the type of 'self' (line 117)
        self_190936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self')
        # Setting the type of the member '_mirrorOutput' of a type (line 117)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_190936, '_mirrorOutput', True_190935)
        
        # ################# End of 'addError(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addError' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_190937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190937)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addError'
        return stypy_return_type_190937


    @norecursion
    def addFailure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addFailure'
        module_type_store = module_type_store.open_function_context('addFailure', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.addFailure.__dict__.__setitem__('stypy_localization', localization)
        TestResult.addFailure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.addFailure.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.addFailure.__dict__.__setitem__('stypy_function_name', 'TestResult.addFailure')
        TestResult.addFailure.__dict__.__setitem__('stypy_param_names_list', ['test', 'err'])
        TestResult.addFailure.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.addFailure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.addFailure.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.addFailure.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.addFailure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.addFailure.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.addFailure', ['test', 'err'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addFailure', localization, ['test', 'err'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addFailure(...)' code ##################

        str_190938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, (-1)), 'str', "Called when an error has occurred. 'err' is a tuple of values as\n        returned by sys.exc_info().")
        
        # Call to append(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_190942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        # Getting the type of 'test' (line 123)
        test_190943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 30), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 30), tuple_190942, test_190943)
        # Adding element type (line 123)
        
        # Call to _exc_info_to_string(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'err' (line 123)
        err_190946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 61), 'err', False)
        # Getting the type of 'test' (line 123)
        test_190947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 66), 'test', False)
        # Processing the call keyword arguments (line 123)
        kwargs_190948 = {}
        # Getting the type of 'self' (line 123)
        self_190944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 36), 'self', False)
        # Obtaining the member '_exc_info_to_string' of a type (line 123)
        _exc_info_to_string_190945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 36), self_190944, '_exc_info_to_string')
        # Calling _exc_info_to_string(args, kwargs) (line 123)
        _exc_info_to_string_call_result_190949 = invoke(stypy.reporting.localization.Localization(__file__, 123, 36), _exc_info_to_string_190945, *[err_190946, test_190947], **kwargs_190948)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 30), tuple_190942, _exc_info_to_string_call_result_190949)
        
        # Processing the call keyword arguments (line 123)
        kwargs_190950 = {}
        # Getting the type of 'self' (line 123)
        self_190939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self', False)
        # Obtaining the member 'failures' of a type (line 123)
        failures_190940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_190939, 'failures')
        # Obtaining the member 'append' of a type (line 123)
        append_190941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), failures_190940, 'append')
        # Calling append(args, kwargs) (line 123)
        append_call_result_190951 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), append_190941, *[tuple_190942], **kwargs_190950)
        
        
        # Assigning a Name to a Attribute (line 124):
        
        # Assigning a Name to a Attribute (line 124):
        # Getting the type of 'True' (line 124)
        True_190952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 29), 'True')
        # Getting the type of 'self' (line 124)
        self_190953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'self')
        # Setting the type of the member '_mirrorOutput' of a type (line 124)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), self_190953, '_mirrorOutput', True_190952)
        
        # ################# End of 'addFailure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addFailure' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_190954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190954)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addFailure'
        return stypy_return_type_190954


    @norecursion
    def addSuccess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addSuccess'
        module_type_store = module_type_store.open_function_context('addSuccess', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.addSuccess.__dict__.__setitem__('stypy_localization', localization)
        TestResult.addSuccess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.addSuccess.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.addSuccess.__dict__.__setitem__('stypy_function_name', 'TestResult.addSuccess')
        TestResult.addSuccess.__dict__.__setitem__('stypy_param_names_list', ['test'])
        TestResult.addSuccess.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.addSuccess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.addSuccess.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.addSuccess.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.addSuccess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.addSuccess.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.addSuccess', ['test'], None, None, defaults, varargs, kwargs)

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

        str_190955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 8), 'str', 'Called when a test has completed successfully')
        pass
        
        # ################# End of 'addSuccess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addSuccess' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_190956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190956)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addSuccess'
        return stypy_return_type_190956


    @norecursion
    def addSkip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addSkip'
        module_type_store = module_type_store.open_function_context('addSkip', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.addSkip.__dict__.__setitem__('stypy_localization', localization)
        TestResult.addSkip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.addSkip.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.addSkip.__dict__.__setitem__('stypy_function_name', 'TestResult.addSkip')
        TestResult.addSkip.__dict__.__setitem__('stypy_param_names_list', ['test', 'reason'])
        TestResult.addSkip.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.addSkip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.addSkip.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.addSkip.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.addSkip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.addSkip.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.addSkip', ['test', 'reason'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addSkip', localization, ['test', 'reason'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addSkip(...)' code ##################

        str_190957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 8), 'str', 'Called when a test is skipped.')
        
        # Call to append(...): (line 132)
        # Processing the call arguments (line 132)
        
        # Obtaining an instance of the builtin type 'tuple' (line 132)
        tuple_190961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 132)
        # Adding element type (line 132)
        # Getting the type of 'test' (line 132)
        test_190962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 29), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 29), tuple_190961, test_190962)
        # Adding element type (line 132)
        # Getting the type of 'reason' (line 132)
        reason_190963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 35), 'reason', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 29), tuple_190961, reason_190963)
        
        # Processing the call keyword arguments (line 132)
        kwargs_190964 = {}
        # Getting the type of 'self' (line 132)
        self_190958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'self', False)
        # Obtaining the member 'skipped' of a type (line 132)
        skipped_190959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), self_190958, 'skipped')
        # Obtaining the member 'append' of a type (line 132)
        append_190960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), skipped_190959, 'append')
        # Calling append(args, kwargs) (line 132)
        append_call_result_190965 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), append_190960, *[tuple_190961], **kwargs_190964)
        
        
        # ################# End of 'addSkip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addSkip' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_190966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190966)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addSkip'
        return stypy_return_type_190966


    @norecursion
    def addExpectedFailure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addExpectedFailure'
        module_type_store = module_type_store.open_function_context('addExpectedFailure', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.addExpectedFailure.__dict__.__setitem__('stypy_localization', localization)
        TestResult.addExpectedFailure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.addExpectedFailure.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.addExpectedFailure.__dict__.__setitem__('stypy_function_name', 'TestResult.addExpectedFailure')
        TestResult.addExpectedFailure.__dict__.__setitem__('stypy_param_names_list', ['test', 'err'])
        TestResult.addExpectedFailure.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.addExpectedFailure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.addExpectedFailure.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.addExpectedFailure.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.addExpectedFailure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.addExpectedFailure.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.addExpectedFailure', ['test', 'err'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addExpectedFailure', localization, ['test', 'err'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addExpectedFailure(...)' code ##################

        str_190967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'str', 'Called when an expected failure/error occured.')
        
        # Call to append(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Obtaining an instance of the builtin type 'tuple' (line 137)
        tuple_190971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 137)
        # Adding element type (line 137)
        # Getting the type of 'test' (line 137)
        test_190972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 13), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 13), tuple_190971, test_190972)
        # Adding element type (line 137)
        
        # Call to _exc_info_to_string(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'err' (line 137)
        err_190975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 44), 'err', False)
        # Getting the type of 'test' (line 137)
        test_190976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 49), 'test', False)
        # Processing the call keyword arguments (line 137)
        kwargs_190977 = {}
        # Getting the type of 'self' (line 137)
        self_190973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 19), 'self', False)
        # Obtaining the member '_exc_info_to_string' of a type (line 137)
        _exc_info_to_string_190974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 19), self_190973, '_exc_info_to_string')
        # Calling _exc_info_to_string(args, kwargs) (line 137)
        _exc_info_to_string_call_result_190978 = invoke(stypy.reporting.localization.Localization(__file__, 137, 19), _exc_info_to_string_190974, *[err_190975, test_190976], **kwargs_190977)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 13), tuple_190971, _exc_info_to_string_call_result_190978)
        
        # Processing the call keyword arguments (line 136)
        kwargs_190979 = {}
        # Getting the type of 'self' (line 136)
        self_190968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'self', False)
        # Obtaining the member 'expectedFailures' of a type (line 136)
        expectedFailures_190969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), self_190968, 'expectedFailures')
        # Obtaining the member 'append' of a type (line 136)
        append_190970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), expectedFailures_190969, 'append')
        # Calling append(args, kwargs) (line 136)
        append_call_result_190980 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), append_190970, *[tuple_190971], **kwargs_190979)
        
        
        # ################# End of 'addExpectedFailure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addExpectedFailure' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_190981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190981)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addExpectedFailure'
        return stypy_return_type_190981


    @norecursion
    def addUnexpectedSuccess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addUnexpectedSuccess'
        module_type_store = module_type_store.open_function_context('addUnexpectedSuccess', 139, 4, False)
        # Assigning a type to the variable 'self' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_localization', localization)
        TestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_function_name', 'TestResult.addUnexpectedSuccess')
        TestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_param_names_list', ['test'])
        TestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.addUnexpectedSuccess', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addUnexpectedSuccess', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addUnexpectedSuccess(...)' code ##################

        str_190982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'str', 'Called when a test was expected to fail, but succeed.')
        
        # Call to append(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'test' (line 142)
        test_190986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 40), 'test', False)
        # Processing the call keyword arguments (line 142)
        kwargs_190987 = {}
        # Getting the type of 'self' (line 142)
        self_190983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self', False)
        # Obtaining the member 'unexpectedSuccesses' of a type (line 142)
        unexpectedSuccesses_190984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_190983, 'unexpectedSuccesses')
        # Obtaining the member 'append' of a type (line 142)
        append_190985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), unexpectedSuccesses_190984, 'append')
        # Calling append(args, kwargs) (line 142)
        append_call_result_190988 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), append_190985, *[test_190986], **kwargs_190987)
        
        
        # ################# End of 'addUnexpectedSuccess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addUnexpectedSuccess' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_190989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190989)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addUnexpectedSuccess'
        return stypy_return_type_190989


    @norecursion
    def wasSuccessful(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wasSuccessful'
        module_type_store = module_type_store.open_function_context('wasSuccessful', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.wasSuccessful.__dict__.__setitem__('stypy_localization', localization)
        TestResult.wasSuccessful.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.wasSuccessful.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.wasSuccessful.__dict__.__setitem__('stypy_function_name', 'TestResult.wasSuccessful')
        TestResult.wasSuccessful.__dict__.__setitem__('stypy_param_names_list', [])
        TestResult.wasSuccessful.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.wasSuccessful.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.wasSuccessful.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.wasSuccessful.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.wasSuccessful.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.wasSuccessful.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.wasSuccessful', [], None, None, defaults, varargs, kwargs)

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

        str_190990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 8), 'str', 'Tells whether or not this result was a success')
        
        
        # Call to len(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'self' (line 146)
        self_190992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'self', False)
        # Obtaining the member 'failures' of a type (line 146)
        failures_190993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 19), self_190992, 'failures')
        # Processing the call keyword arguments (line 146)
        kwargs_190994 = {}
        # Getting the type of 'len' (line 146)
        len_190991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'len', False)
        # Calling len(args, kwargs) (line 146)
        len_call_result_190995 = invoke(stypy.reporting.localization.Localization(__file__, 146, 15), len_190991, *[failures_190993], **kwargs_190994)
        
        
        # Call to len(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'self' (line 146)
        self_190997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 41), 'self', False)
        # Obtaining the member 'errors' of a type (line 146)
        errors_190998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 41), self_190997, 'errors')
        # Processing the call keyword arguments (line 146)
        kwargs_190999 = {}
        # Getting the type of 'len' (line 146)
        len_190996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 37), 'len', False)
        # Calling len(args, kwargs) (line 146)
        len_call_result_191000 = invoke(stypy.reporting.localization.Localization(__file__, 146, 37), len_190996, *[errors_190998], **kwargs_190999)
        
        # Applying the binary operator '==' (line 146)
        result_eq_191001 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 15), '==', len_call_result_190995, len_call_result_191000)
        int_191002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 57), 'int')
        # Applying the binary operator '==' (line 146)
        result_eq_191003 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 15), '==', len_call_result_191000, int_191002)
        # Applying the binary operator '&' (line 146)
        result_and__191004 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 15), '&', result_eq_191001, result_eq_191003)
        
        # Assigning a type to the variable 'stypy_return_type' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'stypy_return_type', result_and__191004)
        
        # ################# End of 'wasSuccessful(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wasSuccessful' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_191005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191005)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wasSuccessful'
        return stypy_return_type_191005


    @norecursion
    def stop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'stop'
        module_type_store = module_type_store.open_function_context('stop', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.stop.__dict__.__setitem__('stypy_localization', localization)
        TestResult.stop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.stop.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.stop.__dict__.__setitem__('stypy_function_name', 'TestResult.stop')
        TestResult.stop.__dict__.__setitem__('stypy_param_names_list', [])
        TestResult.stop.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.stop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.stop.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.stop.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.stop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.stop.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.stop', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'stop', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'stop(...)' code ##################

        str_191006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 8), 'str', 'Indicates that the tests should be aborted')
        
        # Assigning a Name to a Attribute (line 150):
        
        # Assigning a Name to a Attribute (line 150):
        # Getting the type of 'True' (line 150)
        True_191007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 26), 'True')
        # Getting the type of 'self' (line 150)
        self_191008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self')
        # Setting the type of the member 'shouldStop' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_191008, 'shouldStop', True_191007)
        
        # ################# End of 'stop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'stop' in the type store
        # Getting the type of 'stypy_return_type' (line 148)
        stypy_return_type_191009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191009)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'stop'
        return stypy_return_type_191009


    @norecursion
    def _exc_info_to_string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_exc_info_to_string'
        module_type_store = module_type_store.open_function_context('_exc_info_to_string', 152, 4, False)
        # Assigning a type to the variable 'self' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult._exc_info_to_string.__dict__.__setitem__('stypy_localization', localization)
        TestResult._exc_info_to_string.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult._exc_info_to_string.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult._exc_info_to_string.__dict__.__setitem__('stypy_function_name', 'TestResult._exc_info_to_string')
        TestResult._exc_info_to_string.__dict__.__setitem__('stypy_param_names_list', ['err', 'test'])
        TestResult._exc_info_to_string.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult._exc_info_to_string.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult._exc_info_to_string.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult._exc_info_to_string.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult._exc_info_to_string.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult._exc_info_to_string.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult._exc_info_to_string', ['err', 'test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_exc_info_to_string', localization, ['err', 'test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_exc_info_to_string(...)' code ##################

        str_191010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 8), 'str', 'Converts a sys.exc_info()-style tuple of values into a string.')
        
        # Assigning a Name to a Tuple (line 154):
        
        # Assigning a Subscript to a Name (line 154):
        
        # Obtaining the type of the subscript
        int_191011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 8), 'int')
        # Getting the type of 'err' (line 154)
        err_191012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'err')
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___191013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), err_191012, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_191014 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), getitem___191013, int_191011)
        
        # Assigning a type to the variable 'tuple_var_assignment_190725' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_var_assignment_190725', subscript_call_result_191014)
        
        # Assigning a Subscript to a Name (line 154):
        
        # Obtaining the type of the subscript
        int_191015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 8), 'int')
        # Getting the type of 'err' (line 154)
        err_191016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'err')
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___191017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), err_191016, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_191018 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), getitem___191017, int_191015)
        
        # Assigning a type to the variable 'tuple_var_assignment_190726' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_var_assignment_190726', subscript_call_result_191018)
        
        # Assigning a Subscript to a Name (line 154):
        
        # Obtaining the type of the subscript
        int_191019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 8), 'int')
        # Getting the type of 'err' (line 154)
        err_191020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'err')
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___191021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), err_191020, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_191022 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), getitem___191021, int_191019)
        
        # Assigning a type to the variable 'tuple_var_assignment_190727' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_var_assignment_190727', subscript_call_result_191022)
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'tuple_var_assignment_190725' (line 154)
        tuple_var_assignment_190725_191023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_var_assignment_190725')
        # Assigning a type to the variable 'exctype' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'exctype', tuple_var_assignment_190725_191023)
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'tuple_var_assignment_190726' (line 154)
        tuple_var_assignment_190726_191024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_var_assignment_190726')
        # Assigning a type to the variable 'value' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 17), 'value', tuple_var_assignment_190726_191024)
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'tuple_var_assignment_190727' (line 154)
        tuple_var_assignment_190727_191025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_var_assignment_190727')
        # Assigning a type to the variable 'tb' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 24), 'tb', tuple_var_assignment_190727_191025)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'tb' (line 156)
        tb_191026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 14), 'tb')
        
        # Call to _is_relevant_tb_level(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'tb' (line 156)
        tb_191029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 48), 'tb', False)
        # Processing the call keyword arguments (line 156)
        kwargs_191030 = {}
        # Getting the type of 'self' (line 156)
        self_191027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'self', False)
        # Obtaining the member '_is_relevant_tb_level' of a type (line 156)
        _is_relevant_tb_level_191028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 21), self_191027, '_is_relevant_tb_level')
        # Calling _is_relevant_tb_level(args, kwargs) (line 156)
        _is_relevant_tb_level_call_result_191031 = invoke(stypy.reporting.localization.Localization(__file__, 156, 21), _is_relevant_tb_level_191028, *[tb_191029], **kwargs_191030)
        
        # Applying the binary operator 'and' (line 156)
        result_and_keyword_191032 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 14), 'and', tb_191026, _is_relevant_tb_level_call_result_191031)
        
        # Testing the type of an if condition (line 156)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), result_and_keyword_191032)
        # SSA begins for while statement (line 156)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Attribute to a Name (line 157):
        
        # Assigning a Attribute to a Name (line 157):
        # Getting the type of 'tb' (line 157)
        tb_191033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 17), 'tb')
        # Obtaining the member 'tb_next' of a type (line 157)
        tb_next_191034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 17), tb_191033, 'tb_next')
        # Assigning a type to the variable 'tb' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'tb', tb_next_191034)
        # SSA join for while statement (line 156)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'exctype' (line 159)
        exctype_191035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'exctype')
        # Getting the type of 'test' (line 159)
        test_191036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'test')
        # Obtaining the member 'failureException' of a type (line 159)
        failureException_191037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 22), test_191036, 'failureException')
        # Applying the binary operator 'is' (line 159)
        result_is__191038 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 11), 'is', exctype_191035, failureException_191037)
        
        # Testing the type of an if condition (line 159)
        if_condition_191039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 8), result_is__191038)
        # Assigning a type to the variable 'if_condition_191039' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'if_condition_191039', if_condition_191039)
        # SSA begins for if statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to _count_relevant_tb_levels(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'tb' (line 161)
        tb_191042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 52), 'tb', False)
        # Processing the call keyword arguments (line 161)
        kwargs_191043 = {}
        # Getting the type of 'self' (line 161)
        self_191040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 21), 'self', False)
        # Obtaining the member '_count_relevant_tb_levels' of a type (line 161)
        _count_relevant_tb_levels_191041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 21), self_191040, '_count_relevant_tb_levels')
        # Calling _count_relevant_tb_levels(args, kwargs) (line 161)
        _count_relevant_tb_levels_call_result_191044 = invoke(stypy.reporting.localization.Localization(__file__, 161, 21), _count_relevant_tb_levels_191041, *[tb_191042], **kwargs_191043)
        
        # Assigning a type to the variable 'length' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'length', _count_relevant_tb_levels_call_result_191044)
        
        # Assigning a Call to a Name (line 162):
        
        # Assigning a Call to a Name (line 162):
        
        # Call to format_exception(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'exctype' (line 162)
        exctype_191047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 50), 'exctype', False)
        # Getting the type of 'value' (line 162)
        value_191048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 59), 'value', False)
        # Getting the type of 'tb' (line 162)
        tb_191049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 66), 'tb', False)
        # Getting the type of 'length' (line 162)
        length_191050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 70), 'length', False)
        # Processing the call keyword arguments (line 162)
        kwargs_191051 = {}
        # Getting the type of 'traceback' (line 162)
        traceback_191045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 23), 'traceback', False)
        # Obtaining the member 'format_exception' of a type (line 162)
        format_exception_191046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 23), traceback_191045, 'format_exception')
        # Calling format_exception(args, kwargs) (line 162)
        format_exception_call_result_191052 = invoke(stypy.reporting.localization.Localization(__file__, 162, 23), format_exception_191046, *[exctype_191047, value_191048, tb_191049, length_191050], **kwargs_191051)
        
        # Assigning a type to the variable 'msgLines' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'msgLines', format_exception_call_result_191052)
        # SSA branch for the else part of an if statement (line 159)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to format_exception(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'exctype' (line 164)
        exctype_191055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 50), 'exctype', False)
        # Getting the type of 'value' (line 164)
        value_191056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 59), 'value', False)
        # Getting the type of 'tb' (line 164)
        tb_191057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 66), 'tb', False)
        # Processing the call keyword arguments (line 164)
        kwargs_191058 = {}
        # Getting the type of 'traceback' (line 164)
        traceback_191053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'traceback', False)
        # Obtaining the member 'format_exception' of a type (line 164)
        format_exception_191054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 23), traceback_191053, 'format_exception')
        # Calling format_exception(args, kwargs) (line 164)
        format_exception_call_result_191059 = invoke(stypy.reporting.localization.Localization(__file__, 164, 23), format_exception_191054, *[exctype_191055, value_191056, tb_191057], **kwargs_191058)
        
        # Assigning a type to the variable 'msgLines' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'msgLines', format_exception_call_result_191059)
        # SSA join for if statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 166)
        self_191060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), 'self')
        # Obtaining the member 'buffer' of a type (line 166)
        buffer_191061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 11), self_191060, 'buffer')
        # Testing the type of an if condition (line 166)
        if_condition_191062 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 8), buffer_191061)
        # Assigning a type to the variable 'if_condition_191062' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'if_condition_191062', if_condition_191062)
        # SSA begins for if statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to getvalue(...): (line 167)
        # Processing the call keyword arguments (line 167)
        kwargs_191066 = {}
        # Getting the type of 'sys' (line 167)
        sys_191063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 167)
        stdout_191064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 21), sys_191063, 'stdout')
        # Obtaining the member 'getvalue' of a type (line 167)
        getvalue_191065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 21), stdout_191064, 'getvalue')
        # Calling getvalue(args, kwargs) (line 167)
        getvalue_call_result_191067 = invoke(stypy.reporting.localization.Localization(__file__, 167, 21), getvalue_191065, *[], **kwargs_191066)
        
        # Assigning a type to the variable 'output' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'output', getvalue_call_result_191067)
        
        # Assigning a Call to a Name (line 168):
        
        # Assigning a Call to a Name (line 168):
        
        # Call to getvalue(...): (line 168)
        # Processing the call keyword arguments (line 168)
        kwargs_191071 = {}
        # Getting the type of 'sys' (line 168)
        sys_191068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 168)
        stderr_191069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 20), sys_191068, 'stderr')
        # Obtaining the member 'getvalue' of a type (line 168)
        getvalue_191070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 20), stderr_191069, 'getvalue')
        # Calling getvalue(args, kwargs) (line 168)
        getvalue_call_result_191072 = invoke(stypy.reporting.localization.Localization(__file__, 168, 20), getvalue_191070, *[], **kwargs_191071)
        
        # Assigning a type to the variable 'error' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'error', getvalue_call_result_191072)
        
        # Getting the type of 'output' (line 169)
        output_191073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'output')
        # Testing the type of an if condition (line 169)
        if_condition_191074 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 12), output_191073)
        # Assigning a type to the variable 'if_condition_191074' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'if_condition_191074', if_condition_191074)
        # SSA begins for if statement (line 169)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to endswith(...): (line 170)
        # Processing the call arguments (line 170)
        str_191077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 39), 'str', '\n')
        # Processing the call keyword arguments (line 170)
        kwargs_191078 = {}
        # Getting the type of 'output' (line 170)
        output_191075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 23), 'output', False)
        # Obtaining the member 'endswith' of a type (line 170)
        endswith_191076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 23), output_191075, 'endswith')
        # Calling endswith(args, kwargs) (line 170)
        endswith_call_result_191079 = invoke(stypy.reporting.localization.Localization(__file__, 170, 23), endswith_191076, *[str_191077], **kwargs_191078)
        
        # Applying the 'not' unary operator (line 170)
        result_not__191080 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 19), 'not', endswith_call_result_191079)
        
        # Testing the type of an if condition (line 170)
        if_condition_191081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 16), result_not__191080)
        # Assigning a type to the variable 'if_condition_191081' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'if_condition_191081', if_condition_191081)
        # SSA begins for if statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'output' (line 171)
        output_191082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'output')
        str_191083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'str', '\n')
        # Applying the binary operator '+=' (line 171)
        result_iadd_191084 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 20), '+=', output_191082, str_191083)
        # Assigning a type to the variable 'output' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'output', result_iadd_191084)
        
        # SSA join for if statement (line 170)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'STDOUT_LINE' (line 172)
        STDOUT_LINE_191087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'STDOUT_LINE', False)
        # Getting the type of 'output' (line 172)
        output_191088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 46), 'output', False)
        # Applying the binary operator '%' (line 172)
        result_mod_191089 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 32), '%', STDOUT_LINE_191087, output_191088)
        
        # Processing the call keyword arguments (line 172)
        kwargs_191090 = {}
        # Getting the type of 'msgLines' (line 172)
        msgLines_191085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'msgLines', False)
        # Obtaining the member 'append' of a type (line 172)
        append_191086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), msgLines_191085, 'append')
        # Calling append(args, kwargs) (line 172)
        append_call_result_191091 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), append_191086, *[result_mod_191089], **kwargs_191090)
        
        # SSA join for if statement (line 169)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'error' (line 173)
        error_191092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'error')
        # Testing the type of an if condition (line 173)
        if_condition_191093 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 12), error_191092)
        # Assigning a type to the variable 'if_condition_191093' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'if_condition_191093', if_condition_191093)
        # SSA begins for if statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to endswith(...): (line 174)
        # Processing the call arguments (line 174)
        str_191096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 38), 'str', '\n')
        # Processing the call keyword arguments (line 174)
        kwargs_191097 = {}
        # Getting the type of 'error' (line 174)
        error_191094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 23), 'error', False)
        # Obtaining the member 'endswith' of a type (line 174)
        endswith_191095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 23), error_191094, 'endswith')
        # Calling endswith(args, kwargs) (line 174)
        endswith_call_result_191098 = invoke(stypy.reporting.localization.Localization(__file__, 174, 23), endswith_191095, *[str_191096], **kwargs_191097)
        
        # Applying the 'not' unary operator (line 174)
        result_not__191099 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 19), 'not', endswith_call_result_191098)
        
        # Testing the type of an if condition (line 174)
        if_condition_191100 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 16), result_not__191099)
        # Assigning a type to the variable 'if_condition_191100' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'if_condition_191100', if_condition_191100)
        # SSA begins for if statement (line 174)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'error' (line 175)
        error_191101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 20), 'error')
        str_191102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 29), 'str', '\n')
        # Applying the binary operator '+=' (line 175)
        result_iadd_191103 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 20), '+=', error_191101, str_191102)
        # Assigning a type to the variable 'error' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 20), 'error', result_iadd_191103)
        
        # SSA join for if statement (line 174)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'STDERR_LINE' (line 176)
        STDERR_LINE_191106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'STDERR_LINE', False)
        # Getting the type of 'error' (line 176)
        error_191107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 46), 'error', False)
        # Applying the binary operator '%' (line 176)
        result_mod_191108 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 32), '%', STDERR_LINE_191106, error_191107)
        
        # Processing the call keyword arguments (line 176)
        kwargs_191109 = {}
        # Getting the type of 'msgLines' (line 176)
        msgLines_191104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'msgLines', False)
        # Obtaining the member 'append' of a type (line 176)
        append_191105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 16), msgLines_191104, 'append')
        # Calling append(args, kwargs) (line 176)
        append_call_result_191110 = invoke(stypy.reporting.localization.Localization(__file__, 176, 16), append_191105, *[result_mod_191108], **kwargs_191109)
        
        # SSA join for if statement (line 173)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 166)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to join(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'msgLines' (line 177)
        msgLines_191113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 23), 'msgLines', False)
        # Processing the call keyword arguments (line 177)
        kwargs_191114 = {}
        str_191111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 15), 'str', '')
        # Obtaining the member 'join' of a type (line 177)
        join_191112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 15), str_191111, 'join')
        # Calling join(args, kwargs) (line 177)
        join_call_result_191115 = invoke(stypy.reporting.localization.Localization(__file__, 177, 15), join_191112, *[msgLines_191113], **kwargs_191114)
        
        # Assigning a type to the variable 'stypy_return_type' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'stypy_return_type', join_call_result_191115)
        
        # ################# End of '_exc_info_to_string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_exc_info_to_string' in the type store
        # Getting the type of 'stypy_return_type' (line 152)
        stypy_return_type_191116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191116)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_exc_info_to_string'
        return stypy_return_type_191116


    @norecursion
    def _is_relevant_tb_level(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_is_relevant_tb_level'
        module_type_store = module_type_store.open_function_context('_is_relevant_tb_level', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult._is_relevant_tb_level.__dict__.__setitem__('stypy_localization', localization)
        TestResult._is_relevant_tb_level.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult._is_relevant_tb_level.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult._is_relevant_tb_level.__dict__.__setitem__('stypy_function_name', 'TestResult._is_relevant_tb_level')
        TestResult._is_relevant_tb_level.__dict__.__setitem__('stypy_param_names_list', ['tb'])
        TestResult._is_relevant_tb_level.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult._is_relevant_tb_level.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult._is_relevant_tb_level.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult._is_relevant_tb_level.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult._is_relevant_tb_level.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult._is_relevant_tb_level.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult._is_relevant_tb_level', ['tb'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_is_relevant_tb_level', localization, ['tb'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_is_relevant_tb_level(...)' code ##################

        
        str_191117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 15), 'str', '__unittest')
        # Getting the type of 'tb' (line 181)
        tb_191118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 31), 'tb')
        # Obtaining the member 'tb_frame' of a type (line 181)
        tb_frame_191119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 31), tb_191118, 'tb_frame')
        # Obtaining the member 'f_globals' of a type (line 181)
        f_globals_191120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 31), tb_frame_191119, 'f_globals')
        # Applying the binary operator 'in' (line 181)
        result_contains_191121 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 15), 'in', str_191117, f_globals_191120)
        
        # Assigning a type to the variable 'stypy_return_type' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'stypy_return_type', result_contains_191121)
        
        # ################# End of '_is_relevant_tb_level(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_is_relevant_tb_level' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_191122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191122)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_is_relevant_tb_level'
        return stypy_return_type_191122


    @norecursion
    def _count_relevant_tb_levels(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_count_relevant_tb_levels'
        module_type_store = module_type_store.open_function_context('_count_relevant_tb_levels', 183, 4, False)
        # Assigning a type to the variable 'self' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult._count_relevant_tb_levels.__dict__.__setitem__('stypy_localization', localization)
        TestResult._count_relevant_tb_levels.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult._count_relevant_tb_levels.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult._count_relevant_tb_levels.__dict__.__setitem__('stypy_function_name', 'TestResult._count_relevant_tb_levels')
        TestResult._count_relevant_tb_levels.__dict__.__setitem__('stypy_param_names_list', ['tb'])
        TestResult._count_relevant_tb_levels.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult._count_relevant_tb_levels.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult._count_relevant_tb_levels.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult._count_relevant_tb_levels.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult._count_relevant_tb_levels.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult._count_relevant_tb_levels.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult._count_relevant_tb_levels', ['tb'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_count_relevant_tb_levels', localization, ['tb'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_count_relevant_tb_levels(...)' code ##################

        
        # Assigning a Num to a Name (line 184):
        
        # Assigning a Num to a Name (line 184):
        int_191123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 17), 'int')
        # Assigning a type to the variable 'length' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'length', int_191123)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'tb' (line 185)
        tb_191124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 14), 'tb')
        
        
        # Call to _is_relevant_tb_level(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'tb' (line 185)
        tb_191127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 52), 'tb', False)
        # Processing the call keyword arguments (line 185)
        kwargs_191128 = {}
        # Getting the type of 'self' (line 185)
        self_191125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 25), 'self', False)
        # Obtaining the member '_is_relevant_tb_level' of a type (line 185)
        _is_relevant_tb_level_191126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 25), self_191125, '_is_relevant_tb_level')
        # Calling _is_relevant_tb_level(args, kwargs) (line 185)
        _is_relevant_tb_level_call_result_191129 = invoke(stypy.reporting.localization.Localization(__file__, 185, 25), _is_relevant_tb_level_191126, *[tb_191127], **kwargs_191128)
        
        # Applying the 'not' unary operator (line 185)
        result_not__191130 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 21), 'not', _is_relevant_tb_level_call_result_191129)
        
        # Applying the binary operator 'and' (line 185)
        result_and_keyword_191131 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 14), 'and', tb_191124, result_not__191130)
        
        # Testing the type of an if condition (line 185)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 8), result_and_keyword_191131)
        # SSA begins for while statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'length' (line 186)
        length_191132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'length')
        int_191133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 22), 'int')
        # Applying the binary operator '+=' (line 186)
        result_iadd_191134 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 12), '+=', length_191132, int_191133)
        # Assigning a type to the variable 'length' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'length', result_iadd_191134)
        
        
        # Assigning a Attribute to a Name (line 187):
        
        # Assigning a Attribute to a Name (line 187):
        # Getting the type of 'tb' (line 187)
        tb_191135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 17), 'tb')
        # Obtaining the member 'tb_next' of a type (line 187)
        tb_next_191136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 17), tb_191135, 'tb_next')
        # Assigning a type to the variable 'tb' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'tb', tb_next_191136)
        # SSA join for while statement (line 185)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'length' (line 188)
        length_191137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'length')
        # Assigning a type to the variable 'stypy_return_type' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'stypy_return_type', length_191137)
        
        # ################# End of '_count_relevant_tb_levels(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_count_relevant_tb_levels' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_191138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191138)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_count_relevant_tb_levels'
        return stypy_return_type_191138


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 190, 4, False)
        # Assigning a type to the variable 'self' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestResult.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        TestResult.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestResult.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestResult.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'TestResult.__repr__')
        TestResult.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        TestResult.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestResult.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestResult.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestResult.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestResult.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestResult.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestResult.__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        str_191139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 16), 'str', '<%s run=%i errors=%i failures=%i>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 192)
        tuple_191140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 192)
        # Adding element type (line 192)
        
        # Call to strclass(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'self' (line 192)
        self_191143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 30), 'self', False)
        # Obtaining the member '__class__' of a type (line 192)
        class___191144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 30), self_191143, '__class__')
        # Processing the call keyword arguments (line 192)
        kwargs_191145 = {}
        # Getting the type of 'util' (line 192)
        util_191141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'util', False)
        # Obtaining the member 'strclass' of a type (line 192)
        strclass_191142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 16), util_191141, 'strclass')
        # Calling strclass(args, kwargs) (line 192)
        strclass_call_result_191146 = invoke(stypy.reporting.localization.Localization(__file__, 192, 16), strclass_191142, *[class___191144], **kwargs_191145)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 16), tuple_191140, strclass_call_result_191146)
        # Adding element type (line 192)
        # Getting the type of 'self' (line 192)
        self_191147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 47), 'self')
        # Obtaining the member 'testsRun' of a type (line 192)
        testsRun_191148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 47), self_191147, 'testsRun')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 16), tuple_191140, testsRun_191148)
        # Adding element type (line 192)
        
        # Call to len(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'self' (line 192)
        self_191150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 66), 'self', False)
        # Obtaining the member 'errors' of a type (line 192)
        errors_191151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 66), self_191150, 'errors')
        # Processing the call keyword arguments (line 192)
        kwargs_191152 = {}
        # Getting the type of 'len' (line 192)
        len_191149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 62), 'len', False)
        # Calling len(args, kwargs) (line 192)
        len_call_result_191153 = invoke(stypy.reporting.localization.Localization(__file__, 192, 62), len_191149, *[errors_191151], **kwargs_191152)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 16), tuple_191140, len_call_result_191153)
        # Adding element type (line 192)
        
        # Call to len(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'self' (line 193)
        self_191155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 20), 'self', False)
        # Obtaining the member 'failures' of a type (line 193)
        failures_191156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 20), self_191155, 'failures')
        # Processing the call keyword arguments (line 193)
        kwargs_191157 = {}
        # Getting the type of 'len' (line 193)
        len_191154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'len', False)
        # Calling len(args, kwargs) (line 193)
        len_call_result_191158 = invoke(stypy.reporting.localization.Localization(__file__, 193, 16), len_191154, *[failures_191156], **kwargs_191157)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 16), tuple_191140, len_call_result_191158)
        
        # Applying the binary operator '%' (line 191)
        result_mod_191159 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 16), '%', str_191139, tuple_191140)
        
        # Assigning a type to the variable 'stypy_return_type' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'stypy_return_type', result_mod_191159)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 190)
        stypy_return_type_191160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191160)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_191160


# Assigning a type to the variable 'TestResult' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'TestResult', TestResult)

# Assigning a Name to a Name (line 37):
# Getting the type of 'None' (line 37)
None_191161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 25), 'None')
# Getting the type of 'TestResult'
TestResult_191162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestResult')
# Setting the type of the member '_previousTestClass' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestResult_191162, '_previousTestClass', None_191161)

# Assigning a Name to a Name (line 38):
# Getting the type of 'False' (line 38)
False_191163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'False')
# Getting the type of 'TestResult'
TestResult_191164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestResult')
# Setting the type of the member '_testRunEntered' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestResult_191164, '_testRunEntered', False_191163)

# Assigning a Name to a Name (line 39):
# Getting the type of 'False' (line 39)
False_191165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'False')
# Getting the type of 'TestResult'
TestResult_191166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestResult')
# Setting the type of the member '_moduleSetUpFailed' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestResult_191166, '_moduleSetUpFailed', False_191165)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
