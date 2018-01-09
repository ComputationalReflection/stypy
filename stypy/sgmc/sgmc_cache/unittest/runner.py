
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Running tests'''
2: 
3: import sys
4: import time
5: 
6: from . import result
7: from .signals import registerResult
8: 
9: __unittest = True
10: 
11: 
12: class _WritelnDecorator(object):
13:     '''Used to decorate file-like objects with a handy 'writeln' method'''
14:     def __init__(self,stream):
15:         self.stream = stream
16: 
17:     def __getattr__(self, attr):
18:         if attr in ('stream', '__getstate__'):
19:             raise AttributeError(attr)
20:         return getattr(self.stream,attr)
21: 
22:     def writeln(self, arg=None):
23:         if arg:
24:             self.write(arg)
25:         self.write('\n') # text-mode streams translate to \r\n if needed
26: 
27: 
28: class TextTestResult(result.TestResult):
29:     '''A test result class that can print formatted text results to a stream.
30: 
31:     Used by TextTestRunner.
32:     '''
33:     separator1 = '=' * 70
34:     separator2 = '-' * 70
35: 
36:     def __init__(self, stream, descriptions, verbosity):
37:         super(TextTestResult, self).__init__(stream, descriptions, verbosity)
38:         self.stream = stream
39:         self.showAll = verbosity > 1
40:         self.dots = verbosity == 1
41:         self.descriptions = descriptions
42: 
43:     def getDescription(self, test):
44:         doc_first_line = test.shortDescription()
45:         if self.descriptions and doc_first_line:
46:             return '\n'.join((str(test), doc_first_line))
47:         else:
48:             return str(test)
49: 
50:     def startTest(self, test):
51:         super(TextTestResult, self).startTest(test)
52:         if self.showAll:
53:             self.stream.write(self.getDescription(test))
54:             self.stream.write(" ... ")
55:             self.stream.flush()
56: 
57:     def addSuccess(self, test):
58:         super(TextTestResult, self).addSuccess(test)
59:         if self.showAll:
60:             self.stream.writeln("ok")
61:         elif self.dots:
62:             self.stream.write('.')
63:             self.stream.flush()
64: 
65:     def addError(self, test, err):
66:         super(TextTestResult, self).addError(test, err)
67:         if self.showAll:
68:             self.stream.writeln("ERROR")
69:         elif self.dots:
70:             self.stream.write('E')
71:             self.stream.flush()
72: 
73:     def addFailure(self, test, err):
74:         super(TextTestResult, self).addFailure(test, err)
75:         if self.showAll:
76:             self.stream.writeln("FAIL")
77:         elif self.dots:
78:             self.stream.write('F')
79:             self.stream.flush()
80: 
81:     def addSkip(self, test, reason):
82:         super(TextTestResult, self).addSkip(test, reason)
83:         if self.showAll:
84:             self.stream.writeln("skipped {0!r}".format(reason))
85:         elif self.dots:
86:             self.stream.write("s")
87:             self.stream.flush()
88: 
89:     def addExpectedFailure(self, test, err):
90:         super(TextTestResult, self).addExpectedFailure(test, err)
91:         if self.showAll:
92:             self.stream.writeln("expected failure")
93:         elif self.dots:
94:             self.stream.write("x")
95:             self.stream.flush()
96: 
97:     def addUnexpectedSuccess(self, test):
98:         super(TextTestResult, self).addUnexpectedSuccess(test)
99:         if self.showAll:
100:             self.stream.writeln("unexpected success")
101:         elif self.dots:
102:             self.stream.write("u")
103:             self.stream.flush()
104: 
105:     def printErrors(self):
106:         if self.dots or self.showAll:
107:             self.stream.writeln()
108:         self.printErrorList('ERROR', self.errors)
109:         self.printErrorList('FAIL', self.failures)
110: 
111:     def printErrorList(self, flavour, errors):
112:         for test, err in errors:
113:             self.stream.writeln(self.separator1)
114:             self.stream.writeln("%s: %s" % (flavour,self.getDescription(test)))
115:             self.stream.writeln(self.separator2)
116:             self.stream.writeln("%s" % err)
117: 
118: 
119: class TextTestRunner(object):
120:     '''A test runner class that displays results in textual form.
121: 
122:     It prints out the names of tests as they are run, errors as they
123:     occur, and a summary of the results at the end of the test run.
124:     '''
125:     resultclass = TextTestResult
126: 
127:     def __init__(self, stream=sys.stderr, descriptions=True, verbosity=1,
128:                  failfast=False, buffer=False, resultclass=None):
129:         self.stream = _WritelnDecorator(stream)
130:         self.descriptions = descriptions
131:         self.verbosity = verbosity
132:         self.failfast = failfast
133:         self.buffer = buffer
134:         if resultclass is not None:
135:             self.resultclass = resultclass
136: 
137:     def _makeResult(self):
138:         return self.resultclass(self.stream, self.descriptions, self.verbosity)
139: 
140:     def run(self, test):
141:         "Run the given test case or test suite."
142:         result = self._makeResult()
143:         registerResult(result)
144:         result.failfast = self.failfast
145:         result.buffer = self.buffer
146:         startTime = time.time()
147:         startTestRun = getattr(result, 'startTestRun', None)
148:         if startTestRun is not None:
149:             startTestRun()
150:         try:
151:             test(result)
152:         finally:
153:             stopTestRun = getattr(result, 'stopTestRun', None)
154:             if stopTestRun is not None:
155:                 stopTestRun()
156:         stopTime = time.time()
157:         timeTaken = stopTime - startTime
158:         result.printErrors()
159:         if hasattr(result, 'separator2'):
160:             self.stream.writeln(result.separator2)
161:         run = result.testsRun
162:         self.stream.writeln("Ran %d test%s in %.3fs" %
163:                             (run, run != 1 and "s" or "", timeTaken))
164:         self.stream.writeln()
165: 
166:         expectedFails = unexpectedSuccesses = skipped = 0
167:         try:
168:             results = map(len, (result.expectedFailures,
169:                                 result.unexpectedSuccesses,
170:                                 result.skipped))
171:         except AttributeError:
172:             pass
173:         else:
174:             expectedFails, unexpectedSuccesses, skipped = results
175: 
176:         infos = []
177:         if not result.wasSuccessful():
178:             self.stream.write("FAILED")
179:             failed, errored = map(len, (result.failures, result.errors))
180:             if failed:
181:                 infos.append("failures=%d" % failed)
182:             if errored:
183:                 infos.append("errors=%d" % errored)
184:         else:
185:             self.stream.write("OK")
186:         if skipped:
187:             infos.append("skipped=%d" % skipped)
188:         if expectedFails:
189:             infos.append("expected failures=%d" % expectedFails)
190:         if unexpectedSuccesses:
191:             infos.append("unexpected successes=%d" % unexpectedSuccesses)
192:         if infos:
193:             self.stream.writeln(" (%s)" % (", ".join(infos),))
194:         else:
195:             self.stream.write("\n")
196:         return result
197: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_191173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Running tests')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import time' statement (line 4)
import time

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from unittest import result' statement (line 6)
from unittest import result

import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest', None, module_type_store, ['result'], [result])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from unittest.signals import registerResult' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/unittest/')
import_191174 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'unittest.signals')

if (type(import_191174) is not StypyTypeError):

    if (import_191174 != 'pyd_module'):
        __import__(import_191174)
        sys_modules_191175 = sys.modules[import_191174]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'unittest.signals', sys_modules_191175.module_type_store, module_type_store, ['registerResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_191175, sys_modules_191175.module_type_store, module_type_store)
    else:
        from unittest.signals import registerResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'unittest.signals', None, module_type_store, ['registerResult'], [registerResult])

else:
    # Assigning a type to the variable 'unittest.signals' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'unittest.signals', import_191174)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/')


# Assigning a Name to a Name (line 9):

# Assigning a Name to a Name (line 9):
# Getting the type of 'True' (line 9)
True_191176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'True')
# Assigning a type to the variable '__unittest' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__unittest', True_191176)
# Declaration of the '_WritelnDecorator' class

class _WritelnDecorator(object, ):
    str_191177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'str', "Used to decorate file-like objects with a handy 'writeln' method")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_WritelnDecorator.__init__', ['stream'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['stream'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 15):
        
        # Assigning a Name to a Attribute (line 15):
        # Getting the type of 'stream' (line 15)
        stream_191178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'stream')
        # Getting the type of 'self' (line 15)
        self_191179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member 'stream' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_191179, 'stream', stream_191178)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __getattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattr__'
        module_type_store = module_type_store.open_function_context('__getattr__', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _WritelnDecorator.__getattr__.__dict__.__setitem__('stypy_localization', localization)
        _WritelnDecorator.__getattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _WritelnDecorator.__getattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _WritelnDecorator.__getattr__.__dict__.__setitem__('stypy_function_name', '_WritelnDecorator.__getattr__')
        _WritelnDecorator.__getattr__.__dict__.__setitem__('stypy_param_names_list', ['attr'])
        _WritelnDecorator.__getattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _WritelnDecorator.__getattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _WritelnDecorator.__getattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _WritelnDecorator.__getattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _WritelnDecorator.__getattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _WritelnDecorator.__getattr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_WritelnDecorator.__getattr__', ['attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattr__', localization, ['attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattr__(...)' code ##################

        
        
        # Getting the type of 'attr' (line 18)
        attr_191180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'attr')
        
        # Obtaining an instance of the builtin type 'tuple' (line 18)
        tuple_191181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 18)
        # Adding element type (line 18)
        str_191182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'str', 'stream')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 20), tuple_191181, str_191182)
        # Adding element type (line 18)
        str_191183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'str', '__getstate__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 20), tuple_191181, str_191183)
        
        # Applying the binary operator 'in' (line 18)
        result_contains_191184 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 11), 'in', attr_191180, tuple_191181)
        
        # Testing the type of an if condition (line 18)
        if_condition_191185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 8), result_contains_191184)
        # Assigning a type to the variable 'if_condition_191185' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'if_condition_191185', if_condition_191185)
        # SSA begins for if statement (line 18)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to AttributeError(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'attr' (line 19)
        attr_191187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 33), 'attr', False)
        # Processing the call keyword arguments (line 19)
        kwargs_191188 = {}
        # Getting the type of 'AttributeError' (line 19)
        AttributeError_191186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 19)
        AttributeError_call_result_191189 = invoke(stypy.reporting.localization.Localization(__file__, 19, 18), AttributeError_191186, *[attr_191187], **kwargs_191188)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 19, 12), AttributeError_call_result_191189, 'raise parameter', BaseException)
        # SSA join for if statement (line 18)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to getattr(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'self' (line 20)
        self_191191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 23), 'self', False)
        # Obtaining the member 'stream' of a type (line 20)
        stream_191192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 23), self_191191, 'stream')
        # Getting the type of 'attr' (line 20)
        attr_191193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 35), 'attr', False)
        # Processing the call keyword arguments (line 20)
        kwargs_191194 = {}
        # Getting the type of 'getattr' (line 20)
        getattr_191190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 20)
        getattr_call_result_191195 = invoke(stypy.reporting.localization.Localization(__file__, 20, 15), getattr_191190, *[stream_191192, attr_191193], **kwargs_191194)
        
        # Assigning a type to the variable 'stypy_return_type' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'stypy_return_type', getattr_call_result_191195)
        
        # ################# End of '__getattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_191196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattr__'
        return stypy_return_type_191196


    @norecursion
    def writeln(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 22)
        None_191197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 26), 'None')
        defaults = [None_191197]
        # Create a new context for function 'writeln'
        module_type_store = module_type_store.open_function_context('writeln', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _WritelnDecorator.writeln.__dict__.__setitem__('stypy_localization', localization)
        _WritelnDecorator.writeln.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _WritelnDecorator.writeln.__dict__.__setitem__('stypy_type_store', module_type_store)
        _WritelnDecorator.writeln.__dict__.__setitem__('stypy_function_name', '_WritelnDecorator.writeln')
        _WritelnDecorator.writeln.__dict__.__setitem__('stypy_param_names_list', ['arg'])
        _WritelnDecorator.writeln.__dict__.__setitem__('stypy_varargs_param_name', None)
        _WritelnDecorator.writeln.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _WritelnDecorator.writeln.__dict__.__setitem__('stypy_call_defaults', defaults)
        _WritelnDecorator.writeln.__dict__.__setitem__('stypy_call_varargs', varargs)
        _WritelnDecorator.writeln.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _WritelnDecorator.writeln.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_WritelnDecorator.writeln', ['arg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'writeln', localization, ['arg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'writeln(...)' code ##################

        
        # Getting the type of 'arg' (line 23)
        arg_191198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'arg')
        # Testing the type of an if condition (line 23)
        if_condition_191199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 8), arg_191198)
        # Assigning a type to the variable 'if_condition_191199' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'if_condition_191199', if_condition_191199)
        # SSA begins for if statement (line 23)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'arg' (line 24)
        arg_191202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 23), 'arg', False)
        # Processing the call keyword arguments (line 24)
        kwargs_191203 = {}
        # Getting the type of 'self' (line 24)
        self_191200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'self', False)
        # Obtaining the member 'write' of a type (line 24)
        write_191201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), self_191200, 'write')
        # Calling write(args, kwargs) (line 24)
        write_call_result_191204 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), write_191201, *[arg_191202], **kwargs_191203)
        
        # SSA join for if statement (line 23)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 25)
        # Processing the call arguments (line 25)
        str_191207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'str', '\n')
        # Processing the call keyword arguments (line 25)
        kwargs_191208 = {}
        # Getting the type of 'self' (line 25)
        self_191205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 25)
        write_191206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_191205, 'write')
        # Calling write(args, kwargs) (line 25)
        write_call_result_191209 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), write_191206, *[str_191207], **kwargs_191208)
        
        
        # ################# End of 'writeln(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'writeln' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_191210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191210)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'writeln'
        return stypy_return_type_191210


# Assigning a type to the variable '_WritelnDecorator' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '_WritelnDecorator', _WritelnDecorator)
# Declaration of the 'TextTestResult' class
# Getting the type of 'result' (line 28)
result_191211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 21), 'result')
# Obtaining the member 'TestResult' of a type (line 28)
TestResult_191212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 21), result_191211, 'TestResult')

class TextTestResult(TestResult_191212, ):
    str_191213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', 'A test result class that can print formatted text results to a stream.\n\n    Used by TextTestRunner.\n    ')
    
    # Assigning a BinOp to a Name (line 33):
    
    # Assigning a BinOp to a Name (line 34):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestResult.__init__', ['stream', 'descriptions', 'verbosity'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'stream' (line 37)
        stream_191220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 45), 'stream', False)
        # Getting the type of 'descriptions' (line 37)
        descriptions_191221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 53), 'descriptions', False)
        # Getting the type of 'verbosity' (line 37)
        verbosity_191222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 67), 'verbosity', False)
        # Processing the call keyword arguments (line 37)
        kwargs_191223 = {}
        
        # Call to super(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'TextTestResult' (line 37)
        TextTestResult_191215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'TextTestResult', False)
        # Getting the type of 'self' (line 37)
        self_191216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 30), 'self', False)
        # Processing the call keyword arguments (line 37)
        kwargs_191217 = {}
        # Getting the type of 'super' (line 37)
        super_191214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'super', False)
        # Calling super(args, kwargs) (line 37)
        super_call_result_191218 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), super_191214, *[TextTestResult_191215, self_191216], **kwargs_191217)
        
        # Obtaining the member '__init__' of a type (line 37)
        init___191219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), super_call_result_191218, '__init__')
        # Calling __init__(args, kwargs) (line 37)
        init___call_result_191224 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), init___191219, *[stream_191220, descriptions_191221, verbosity_191222], **kwargs_191223)
        
        
        # Assigning a Name to a Attribute (line 38):
        
        # Assigning a Name to a Attribute (line 38):
        # Getting the type of 'stream' (line 38)
        stream_191225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'stream')
        # Getting the type of 'self' (line 38)
        self_191226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'stream' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_191226, 'stream', stream_191225)
        
        # Assigning a Compare to a Attribute (line 39):
        
        # Assigning a Compare to a Attribute (line 39):
        
        # Getting the type of 'verbosity' (line 39)
        verbosity_191227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'verbosity')
        int_191228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 35), 'int')
        # Applying the binary operator '>' (line 39)
        result_gt_191229 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 23), '>', verbosity_191227, int_191228)
        
        # Getting the type of 'self' (line 39)
        self_191230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'showAll' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_191230, 'showAll', result_gt_191229)
        
        # Assigning a Compare to a Attribute (line 40):
        
        # Assigning a Compare to a Attribute (line 40):
        
        # Getting the type of 'verbosity' (line 40)
        verbosity_191231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'verbosity')
        int_191232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 33), 'int')
        # Applying the binary operator '==' (line 40)
        result_eq_191233 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 20), '==', verbosity_191231, int_191232)
        
        # Getting the type of 'self' (line 40)
        self_191234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'dots' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_191234, 'dots', result_eq_191233)
        
        # Assigning a Name to a Attribute (line 41):
        
        # Assigning a Name to a Attribute (line 41):
        # Getting the type of 'descriptions' (line 41)
        descriptions_191235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'descriptions')
        # Getting the type of 'self' (line 41)
        self_191236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'descriptions' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_191236, 'descriptions', descriptions_191235)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def getDescription(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getDescription'
        module_type_store = module_type_store.open_function_context('getDescription', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextTestResult.getDescription.__dict__.__setitem__('stypy_localization', localization)
        TextTestResult.getDescription.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextTestResult.getDescription.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextTestResult.getDescription.__dict__.__setitem__('stypy_function_name', 'TextTestResult.getDescription')
        TextTestResult.getDescription.__dict__.__setitem__('stypy_param_names_list', ['test'])
        TextTestResult.getDescription.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextTestResult.getDescription.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextTestResult.getDescription.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextTestResult.getDescription.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextTestResult.getDescription.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextTestResult.getDescription.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestResult.getDescription', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getDescription', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getDescription(...)' code ##################

        
        # Assigning a Call to a Name (line 44):
        
        # Assigning a Call to a Name (line 44):
        
        # Call to shortDescription(...): (line 44)
        # Processing the call keyword arguments (line 44)
        kwargs_191239 = {}
        # Getting the type of 'test' (line 44)
        test_191237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 25), 'test', False)
        # Obtaining the member 'shortDescription' of a type (line 44)
        shortDescription_191238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 25), test_191237, 'shortDescription')
        # Calling shortDescription(args, kwargs) (line 44)
        shortDescription_call_result_191240 = invoke(stypy.reporting.localization.Localization(__file__, 44, 25), shortDescription_191238, *[], **kwargs_191239)
        
        # Assigning a type to the variable 'doc_first_line' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'doc_first_line', shortDescription_call_result_191240)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 45)
        self_191241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'self')
        # Obtaining the member 'descriptions' of a type (line 45)
        descriptions_191242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 11), self_191241, 'descriptions')
        # Getting the type of 'doc_first_line' (line 45)
        doc_first_line_191243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 33), 'doc_first_line')
        # Applying the binary operator 'and' (line 45)
        result_and_keyword_191244 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), 'and', descriptions_191242, doc_first_line_191243)
        
        # Testing the type of an if condition (line 45)
        if_condition_191245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 8), result_and_keyword_191244)
        # Assigning a type to the variable 'if_condition_191245' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'if_condition_191245', if_condition_191245)
        # SSA begins for if statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to join(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Obtaining an instance of the builtin type 'tuple' (line 46)
        tuple_191248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 46)
        # Adding element type (line 46)
        
        # Call to str(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'test' (line 46)
        test_191250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'test', False)
        # Processing the call keyword arguments (line 46)
        kwargs_191251 = {}
        # Getting the type of 'str' (line 46)
        str_191249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 30), 'str', False)
        # Calling str(args, kwargs) (line 46)
        str_call_result_191252 = invoke(stypy.reporting.localization.Localization(__file__, 46, 30), str_191249, *[test_191250], **kwargs_191251)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 30), tuple_191248, str_call_result_191252)
        # Adding element type (line 46)
        # Getting the type of 'doc_first_line' (line 46)
        doc_first_line_191253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 41), 'doc_first_line', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 30), tuple_191248, doc_first_line_191253)
        
        # Processing the call keyword arguments (line 46)
        kwargs_191254 = {}
        str_191246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'str', '\n')
        # Obtaining the member 'join' of a type (line 46)
        join_191247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 19), str_191246, 'join')
        # Calling join(args, kwargs) (line 46)
        join_call_result_191255 = invoke(stypy.reporting.localization.Localization(__file__, 46, 19), join_191247, *[tuple_191248], **kwargs_191254)
        
        # Assigning a type to the variable 'stypy_return_type' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type', join_call_result_191255)
        # SSA branch for the else part of an if statement (line 45)
        module_type_store.open_ssa_branch('else')
        
        # Call to str(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'test' (line 48)
        test_191257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'test', False)
        # Processing the call keyword arguments (line 48)
        kwargs_191258 = {}
        # Getting the type of 'str' (line 48)
        str_191256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'str', False)
        # Calling str(args, kwargs) (line 48)
        str_call_result_191259 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), str_191256, *[test_191257], **kwargs_191258)
        
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'stypy_return_type', str_call_result_191259)
        # SSA join for if statement (line 45)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'getDescription(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getDescription' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_191260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191260)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getDescription'
        return stypy_return_type_191260


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
        TextTestResult.startTest.__dict__.__setitem__('stypy_localization', localization)
        TextTestResult.startTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextTestResult.startTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextTestResult.startTest.__dict__.__setitem__('stypy_function_name', 'TextTestResult.startTest')
        TextTestResult.startTest.__dict__.__setitem__('stypy_param_names_list', ['test'])
        TextTestResult.startTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextTestResult.startTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextTestResult.startTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextTestResult.startTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextTestResult.startTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextTestResult.startTest.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestResult.startTest', ['test'], None, None, defaults, varargs, kwargs)

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

        
        # Call to startTest(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'test' (line 51)
        test_191267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 46), 'test', False)
        # Processing the call keyword arguments (line 51)
        kwargs_191268 = {}
        
        # Call to super(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'TextTestResult' (line 51)
        TextTestResult_191262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'TextTestResult', False)
        # Getting the type of 'self' (line 51)
        self_191263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'self', False)
        # Processing the call keyword arguments (line 51)
        kwargs_191264 = {}
        # Getting the type of 'super' (line 51)
        super_191261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'super', False)
        # Calling super(args, kwargs) (line 51)
        super_call_result_191265 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), super_191261, *[TextTestResult_191262, self_191263], **kwargs_191264)
        
        # Obtaining the member 'startTest' of a type (line 51)
        startTest_191266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), super_call_result_191265, 'startTest')
        # Calling startTest(args, kwargs) (line 51)
        startTest_call_result_191269 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), startTest_191266, *[test_191267], **kwargs_191268)
        
        
        # Getting the type of 'self' (line 52)
        self_191270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'self')
        # Obtaining the member 'showAll' of a type (line 52)
        showAll_191271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 11), self_191270, 'showAll')
        # Testing the type of an if condition (line 52)
        if_condition_191272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 8), showAll_191271)
        # Assigning a type to the variable 'if_condition_191272' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'if_condition_191272', if_condition_191272)
        # SSA begins for if statement (line 52)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to getDescription(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'test' (line 53)
        test_191278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 50), 'test', False)
        # Processing the call keyword arguments (line 53)
        kwargs_191279 = {}
        # Getting the type of 'self' (line 53)
        self_191276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 30), 'self', False)
        # Obtaining the member 'getDescription' of a type (line 53)
        getDescription_191277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 30), self_191276, 'getDescription')
        # Calling getDescription(args, kwargs) (line 53)
        getDescription_call_result_191280 = invoke(stypy.reporting.localization.Localization(__file__, 53, 30), getDescription_191277, *[test_191278], **kwargs_191279)
        
        # Processing the call keyword arguments (line 53)
        kwargs_191281 = {}
        # Getting the type of 'self' (line 53)
        self_191273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 53)
        stream_191274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 12), self_191273, 'stream')
        # Obtaining the member 'write' of a type (line 53)
        write_191275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 12), stream_191274, 'write')
        # Calling write(args, kwargs) (line 53)
        write_call_result_191282 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), write_191275, *[getDescription_call_result_191280], **kwargs_191281)
        
        
        # Call to write(...): (line 54)
        # Processing the call arguments (line 54)
        str_191286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 30), 'str', ' ... ')
        # Processing the call keyword arguments (line 54)
        kwargs_191287 = {}
        # Getting the type of 'self' (line 54)
        self_191283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 54)
        stream_191284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), self_191283, 'stream')
        # Obtaining the member 'write' of a type (line 54)
        write_191285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), stream_191284, 'write')
        # Calling write(args, kwargs) (line 54)
        write_call_result_191288 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), write_191285, *[str_191286], **kwargs_191287)
        
        
        # Call to flush(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_191292 = {}
        # Getting the type of 'self' (line 55)
        self_191289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 55)
        stream_191290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 12), self_191289, 'stream')
        # Obtaining the member 'flush' of a type (line 55)
        flush_191291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 12), stream_191290, 'flush')
        # Calling flush(args, kwargs) (line 55)
        flush_call_result_191293 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), flush_191291, *[], **kwargs_191292)
        
        # SSA join for if statement (line 52)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'startTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'startTest' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_191294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191294)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'startTest'
        return stypy_return_type_191294


    @norecursion
    def addSuccess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addSuccess'
        module_type_store = module_type_store.open_function_context('addSuccess', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextTestResult.addSuccess.__dict__.__setitem__('stypy_localization', localization)
        TextTestResult.addSuccess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextTestResult.addSuccess.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextTestResult.addSuccess.__dict__.__setitem__('stypy_function_name', 'TextTestResult.addSuccess')
        TextTestResult.addSuccess.__dict__.__setitem__('stypy_param_names_list', ['test'])
        TextTestResult.addSuccess.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextTestResult.addSuccess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextTestResult.addSuccess.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextTestResult.addSuccess.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextTestResult.addSuccess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextTestResult.addSuccess.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestResult.addSuccess', ['test'], None, None, defaults, varargs, kwargs)

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

        
        # Call to addSuccess(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'test' (line 58)
        test_191301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 47), 'test', False)
        # Processing the call keyword arguments (line 58)
        kwargs_191302 = {}
        
        # Call to super(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'TextTestResult' (line 58)
        TextTestResult_191296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 14), 'TextTestResult', False)
        # Getting the type of 'self' (line 58)
        self_191297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 30), 'self', False)
        # Processing the call keyword arguments (line 58)
        kwargs_191298 = {}
        # Getting the type of 'super' (line 58)
        super_191295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'super', False)
        # Calling super(args, kwargs) (line 58)
        super_call_result_191299 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), super_191295, *[TextTestResult_191296, self_191297], **kwargs_191298)
        
        # Obtaining the member 'addSuccess' of a type (line 58)
        addSuccess_191300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), super_call_result_191299, 'addSuccess')
        # Calling addSuccess(args, kwargs) (line 58)
        addSuccess_call_result_191303 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), addSuccess_191300, *[test_191301], **kwargs_191302)
        
        
        # Getting the type of 'self' (line 59)
        self_191304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'self')
        # Obtaining the member 'showAll' of a type (line 59)
        showAll_191305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 11), self_191304, 'showAll')
        # Testing the type of an if condition (line 59)
        if_condition_191306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), showAll_191305)
        # Assigning a type to the variable 'if_condition_191306' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_191306', if_condition_191306)
        # SSA begins for if statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to writeln(...): (line 60)
        # Processing the call arguments (line 60)
        str_191310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'str', 'ok')
        # Processing the call keyword arguments (line 60)
        kwargs_191311 = {}
        # Getting the type of 'self' (line 60)
        self_191307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 60)
        stream_191308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), self_191307, 'stream')
        # Obtaining the member 'writeln' of a type (line 60)
        writeln_191309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), stream_191308, 'writeln')
        # Calling writeln(args, kwargs) (line 60)
        writeln_call_result_191312 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), writeln_191309, *[str_191310], **kwargs_191311)
        
        # SSA branch for the else part of an if statement (line 59)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 61)
        self_191313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 13), 'self')
        # Obtaining the member 'dots' of a type (line 61)
        dots_191314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 13), self_191313, 'dots')
        # Testing the type of an if condition (line 61)
        if_condition_191315 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 13), dots_191314)
        # Assigning a type to the variable 'if_condition_191315' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 13), 'if_condition_191315', if_condition_191315)
        # SSA begins for if statement (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 62)
        # Processing the call arguments (line 62)
        str_191319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 30), 'str', '.')
        # Processing the call keyword arguments (line 62)
        kwargs_191320 = {}
        # Getting the type of 'self' (line 62)
        self_191316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 62)
        stream_191317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), self_191316, 'stream')
        # Obtaining the member 'write' of a type (line 62)
        write_191318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), stream_191317, 'write')
        # Calling write(args, kwargs) (line 62)
        write_call_result_191321 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), write_191318, *[str_191319], **kwargs_191320)
        
        
        # Call to flush(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_191325 = {}
        # Getting the type of 'self' (line 63)
        self_191322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 63)
        stream_191323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), self_191322, 'stream')
        # Obtaining the member 'flush' of a type (line 63)
        flush_191324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), stream_191323, 'flush')
        # Calling flush(args, kwargs) (line 63)
        flush_call_result_191326 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), flush_191324, *[], **kwargs_191325)
        
        # SSA join for if statement (line 61)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 59)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'addSuccess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addSuccess' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_191327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191327)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addSuccess'
        return stypy_return_type_191327


    @norecursion
    def addError(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addError'
        module_type_store = module_type_store.open_function_context('addError', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextTestResult.addError.__dict__.__setitem__('stypy_localization', localization)
        TextTestResult.addError.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextTestResult.addError.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextTestResult.addError.__dict__.__setitem__('stypy_function_name', 'TextTestResult.addError')
        TextTestResult.addError.__dict__.__setitem__('stypy_param_names_list', ['test', 'err'])
        TextTestResult.addError.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextTestResult.addError.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextTestResult.addError.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextTestResult.addError.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextTestResult.addError.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextTestResult.addError.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestResult.addError', ['test', 'err'], None, None, defaults, varargs, kwargs)

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

        
        # Call to addError(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'test' (line 66)
        test_191334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 45), 'test', False)
        # Getting the type of 'err' (line 66)
        err_191335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 51), 'err', False)
        # Processing the call keyword arguments (line 66)
        kwargs_191336 = {}
        
        # Call to super(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'TextTestResult' (line 66)
        TextTestResult_191329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 14), 'TextTestResult', False)
        # Getting the type of 'self' (line 66)
        self_191330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), 'self', False)
        # Processing the call keyword arguments (line 66)
        kwargs_191331 = {}
        # Getting the type of 'super' (line 66)
        super_191328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'super', False)
        # Calling super(args, kwargs) (line 66)
        super_call_result_191332 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), super_191328, *[TextTestResult_191329, self_191330], **kwargs_191331)
        
        # Obtaining the member 'addError' of a type (line 66)
        addError_191333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), super_call_result_191332, 'addError')
        # Calling addError(args, kwargs) (line 66)
        addError_call_result_191337 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), addError_191333, *[test_191334, err_191335], **kwargs_191336)
        
        
        # Getting the type of 'self' (line 67)
        self_191338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'self')
        # Obtaining the member 'showAll' of a type (line 67)
        showAll_191339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 11), self_191338, 'showAll')
        # Testing the type of an if condition (line 67)
        if_condition_191340 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), showAll_191339)
        # Assigning a type to the variable 'if_condition_191340' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_191340', if_condition_191340)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to writeln(...): (line 68)
        # Processing the call arguments (line 68)
        str_191344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 32), 'str', 'ERROR')
        # Processing the call keyword arguments (line 68)
        kwargs_191345 = {}
        # Getting the type of 'self' (line 68)
        self_191341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 68)
        stream_191342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), self_191341, 'stream')
        # Obtaining the member 'writeln' of a type (line 68)
        writeln_191343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), stream_191342, 'writeln')
        # Calling writeln(args, kwargs) (line 68)
        writeln_call_result_191346 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), writeln_191343, *[str_191344], **kwargs_191345)
        
        # SSA branch for the else part of an if statement (line 67)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 69)
        self_191347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'self')
        # Obtaining the member 'dots' of a type (line 69)
        dots_191348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), self_191347, 'dots')
        # Testing the type of an if condition (line 69)
        if_condition_191349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 13), dots_191348)
        # Assigning a type to the variable 'if_condition_191349' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'if_condition_191349', if_condition_191349)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 70)
        # Processing the call arguments (line 70)
        str_191353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 30), 'str', 'E')
        # Processing the call keyword arguments (line 70)
        kwargs_191354 = {}
        # Getting the type of 'self' (line 70)
        self_191350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 70)
        stream_191351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), self_191350, 'stream')
        # Obtaining the member 'write' of a type (line 70)
        write_191352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), stream_191351, 'write')
        # Calling write(args, kwargs) (line 70)
        write_call_result_191355 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), write_191352, *[str_191353], **kwargs_191354)
        
        
        # Call to flush(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_191359 = {}
        # Getting the type of 'self' (line 71)
        self_191356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 71)
        stream_191357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), self_191356, 'stream')
        # Obtaining the member 'flush' of a type (line 71)
        flush_191358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), stream_191357, 'flush')
        # Calling flush(args, kwargs) (line 71)
        flush_call_result_191360 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), flush_191358, *[], **kwargs_191359)
        
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'addError(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addError' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_191361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191361)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addError'
        return stypy_return_type_191361


    @norecursion
    def addFailure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addFailure'
        module_type_store = module_type_store.open_function_context('addFailure', 73, 4, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextTestResult.addFailure.__dict__.__setitem__('stypy_localization', localization)
        TextTestResult.addFailure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextTestResult.addFailure.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextTestResult.addFailure.__dict__.__setitem__('stypy_function_name', 'TextTestResult.addFailure')
        TextTestResult.addFailure.__dict__.__setitem__('stypy_param_names_list', ['test', 'err'])
        TextTestResult.addFailure.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextTestResult.addFailure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextTestResult.addFailure.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextTestResult.addFailure.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextTestResult.addFailure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextTestResult.addFailure.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestResult.addFailure', ['test', 'err'], None, None, defaults, varargs, kwargs)

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

        
        # Call to addFailure(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'test' (line 74)
        test_191368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 47), 'test', False)
        # Getting the type of 'err' (line 74)
        err_191369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 53), 'err', False)
        # Processing the call keyword arguments (line 74)
        kwargs_191370 = {}
        
        # Call to super(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'TextTestResult' (line 74)
        TextTestResult_191363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 14), 'TextTestResult', False)
        # Getting the type of 'self' (line 74)
        self_191364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 30), 'self', False)
        # Processing the call keyword arguments (line 74)
        kwargs_191365 = {}
        # Getting the type of 'super' (line 74)
        super_191362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'super', False)
        # Calling super(args, kwargs) (line 74)
        super_call_result_191366 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), super_191362, *[TextTestResult_191363, self_191364], **kwargs_191365)
        
        # Obtaining the member 'addFailure' of a type (line 74)
        addFailure_191367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), super_call_result_191366, 'addFailure')
        # Calling addFailure(args, kwargs) (line 74)
        addFailure_call_result_191371 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), addFailure_191367, *[test_191368, err_191369], **kwargs_191370)
        
        
        # Getting the type of 'self' (line 75)
        self_191372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'self')
        # Obtaining the member 'showAll' of a type (line 75)
        showAll_191373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 11), self_191372, 'showAll')
        # Testing the type of an if condition (line 75)
        if_condition_191374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 8), showAll_191373)
        # Assigning a type to the variable 'if_condition_191374' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'if_condition_191374', if_condition_191374)
        # SSA begins for if statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to writeln(...): (line 76)
        # Processing the call arguments (line 76)
        str_191378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 32), 'str', 'FAIL')
        # Processing the call keyword arguments (line 76)
        kwargs_191379 = {}
        # Getting the type of 'self' (line 76)
        self_191375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 76)
        stream_191376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), self_191375, 'stream')
        # Obtaining the member 'writeln' of a type (line 76)
        writeln_191377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), stream_191376, 'writeln')
        # Calling writeln(args, kwargs) (line 76)
        writeln_call_result_191380 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), writeln_191377, *[str_191378], **kwargs_191379)
        
        # SSA branch for the else part of an if statement (line 75)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 77)
        self_191381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'self')
        # Obtaining the member 'dots' of a type (line 77)
        dots_191382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), self_191381, 'dots')
        # Testing the type of an if condition (line 77)
        if_condition_191383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 13), dots_191382)
        # Assigning a type to the variable 'if_condition_191383' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'if_condition_191383', if_condition_191383)
        # SSA begins for if statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 78)
        # Processing the call arguments (line 78)
        str_191387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 30), 'str', 'F')
        # Processing the call keyword arguments (line 78)
        kwargs_191388 = {}
        # Getting the type of 'self' (line 78)
        self_191384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 78)
        stream_191385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), self_191384, 'stream')
        # Obtaining the member 'write' of a type (line 78)
        write_191386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), stream_191385, 'write')
        # Calling write(args, kwargs) (line 78)
        write_call_result_191389 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), write_191386, *[str_191387], **kwargs_191388)
        
        
        # Call to flush(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_191393 = {}
        # Getting the type of 'self' (line 79)
        self_191390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 79)
        stream_191391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), self_191390, 'stream')
        # Obtaining the member 'flush' of a type (line 79)
        flush_191392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), stream_191391, 'flush')
        # Calling flush(args, kwargs) (line 79)
        flush_call_result_191394 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), flush_191392, *[], **kwargs_191393)
        
        # SSA join for if statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'addFailure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addFailure' in the type store
        # Getting the type of 'stypy_return_type' (line 73)
        stypy_return_type_191395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191395)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addFailure'
        return stypy_return_type_191395


    @norecursion
    def addSkip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addSkip'
        module_type_store = module_type_store.open_function_context('addSkip', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextTestResult.addSkip.__dict__.__setitem__('stypy_localization', localization)
        TextTestResult.addSkip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextTestResult.addSkip.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextTestResult.addSkip.__dict__.__setitem__('stypy_function_name', 'TextTestResult.addSkip')
        TextTestResult.addSkip.__dict__.__setitem__('stypy_param_names_list', ['test', 'reason'])
        TextTestResult.addSkip.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextTestResult.addSkip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextTestResult.addSkip.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextTestResult.addSkip.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextTestResult.addSkip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextTestResult.addSkip.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestResult.addSkip', ['test', 'reason'], None, None, defaults, varargs, kwargs)

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

        
        # Call to addSkip(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'test' (line 82)
        test_191402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 44), 'test', False)
        # Getting the type of 'reason' (line 82)
        reason_191403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 50), 'reason', False)
        # Processing the call keyword arguments (line 82)
        kwargs_191404 = {}
        
        # Call to super(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'TextTestResult' (line 82)
        TextTestResult_191397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 14), 'TextTestResult', False)
        # Getting the type of 'self' (line 82)
        self_191398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 30), 'self', False)
        # Processing the call keyword arguments (line 82)
        kwargs_191399 = {}
        # Getting the type of 'super' (line 82)
        super_191396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'super', False)
        # Calling super(args, kwargs) (line 82)
        super_call_result_191400 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), super_191396, *[TextTestResult_191397, self_191398], **kwargs_191399)
        
        # Obtaining the member 'addSkip' of a type (line 82)
        addSkip_191401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), super_call_result_191400, 'addSkip')
        # Calling addSkip(args, kwargs) (line 82)
        addSkip_call_result_191405 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), addSkip_191401, *[test_191402, reason_191403], **kwargs_191404)
        
        
        # Getting the type of 'self' (line 83)
        self_191406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'self')
        # Obtaining the member 'showAll' of a type (line 83)
        showAll_191407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 11), self_191406, 'showAll')
        # Testing the type of an if condition (line 83)
        if_condition_191408 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 8), showAll_191407)
        # Assigning a type to the variable 'if_condition_191408' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'if_condition_191408', if_condition_191408)
        # SSA begins for if statement (line 83)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to writeln(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Call to format(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'reason' (line 84)
        reason_191414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 55), 'reason', False)
        # Processing the call keyword arguments (line 84)
        kwargs_191415 = {}
        str_191412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 32), 'str', 'skipped {0!r}')
        # Obtaining the member 'format' of a type (line 84)
        format_191413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 32), str_191412, 'format')
        # Calling format(args, kwargs) (line 84)
        format_call_result_191416 = invoke(stypy.reporting.localization.Localization(__file__, 84, 32), format_191413, *[reason_191414], **kwargs_191415)
        
        # Processing the call keyword arguments (line 84)
        kwargs_191417 = {}
        # Getting the type of 'self' (line 84)
        self_191409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 84)
        stream_191410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), self_191409, 'stream')
        # Obtaining the member 'writeln' of a type (line 84)
        writeln_191411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), stream_191410, 'writeln')
        # Calling writeln(args, kwargs) (line 84)
        writeln_call_result_191418 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), writeln_191411, *[format_call_result_191416], **kwargs_191417)
        
        # SSA branch for the else part of an if statement (line 83)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 85)
        self_191419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'self')
        # Obtaining the member 'dots' of a type (line 85)
        dots_191420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 13), self_191419, 'dots')
        # Testing the type of an if condition (line 85)
        if_condition_191421 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 13), dots_191420)
        # Assigning a type to the variable 'if_condition_191421' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'if_condition_191421', if_condition_191421)
        # SSA begins for if statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 86)
        # Processing the call arguments (line 86)
        str_191425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 30), 'str', 's')
        # Processing the call keyword arguments (line 86)
        kwargs_191426 = {}
        # Getting the type of 'self' (line 86)
        self_191422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 86)
        stream_191423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), self_191422, 'stream')
        # Obtaining the member 'write' of a type (line 86)
        write_191424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), stream_191423, 'write')
        # Calling write(args, kwargs) (line 86)
        write_call_result_191427 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), write_191424, *[str_191425], **kwargs_191426)
        
        
        # Call to flush(...): (line 87)
        # Processing the call keyword arguments (line 87)
        kwargs_191431 = {}
        # Getting the type of 'self' (line 87)
        self_191428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 87)
        stream_191429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), self_191428, 'stream')
        # Obtaining the member 'flush' of a type (line 87)
        flush_191430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), stream_191429, 'flush')
        # Calling flush(args, kwargs) (line 87)
        flush_call_result_191432 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), flush_191430, *[], **kwargs_191431)
        
        # SSA join for if statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 83)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'addSkip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addSkip' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_191433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191433)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addSkip'
        return stypy_return_type_191433


    @norecursion
    def addExpectedFailure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addExpectedFailure'
        module_type_store = module_type_store.open_function_context('addExpectedFailure', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextTestResult.addExpectedFailure.__dict__.__setitem__('stypy_localization', localization)
        TextTestResult.addExpectedFailure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextTestResult.addExpectedFailure.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextTestResult.addExpectedFailure.__dict__.__setitem__('stypy_function_name', 'TextTestResult.addExpectedFailure')
        TextTestResult.addExpectedFailure.__dict__.__setitem__('stypy_param_names_list', ['test', 'err'])
        TextTestResult.addExpectedFailure.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextTestResult.addExpectedFailure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextTestResult.addExpectedFailure.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextTestResult.addExpectedFailure.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextTestResult.addExpectedFailure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextTestResult.addExpectedFailure.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestResult.addExpectedFailure', ['test', 'err'], None, None, defaults, varargs, kwargs)

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

        
        # Call to addExpectedFailure(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'test' (line 90)
        test_191440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 55), 'test', False)
        # Getting the type of 'err' (line 90)
        err_191441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 61), 'err', False)
        # Processing the call keyword arguments (line 90)
        kwargs_191442 = {}
        
        # Call to super(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'TextTestResult' (line 90)
        TextTestResult_191435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 14), 'TextTestResult', False)
        # Getting the type of 'self' (line 90)
        self_191436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'self', False)
        # Processing the call keyword arguments (line 90)
        kwargs_191437 = {}
        # Getting the type of 'super' (line 90)
        super_191434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'super', False)
        # Calling super(args, kwargs) (line 90)
        super_call_result_191438 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), super_191434, *[TextTestResult_191435, self_191436], **kwargs_191437)
        
        # Obtaining the member 'addExpectedFailure' of a type (line 90)
        addExpectedFailure_191439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), super_call_result_191438, 'addExpectedFailure')
        # Calling addExpectedFailure(args, kwargs) (line 90)
        addExpectedFailure_call_result_191443 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), addExpectedFailure_191439, *[test_191440, err_191441], **kwargs_191442)
        
        
        # Getting the type of 'self' (line 91)
        self_191444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'self')
        # Obtaining the member 'showAll' of a type (line 91)
        showAll_191445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 11), self_191444, 'showAll')
        # Testing the type of an if condition (line 91)
        if_condition_191446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 8), showAll_191445)
        # Assigning a type to the variable 'if_condition_191446' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'if_condition_191446', if_condition_191446)
        # SSA begins for if statement (line 91)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to writeln(...): (line 92)
        # Processing the call arguments (line 92)
        str_191450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 32), 'str', 'expected failure')
        # Processing the call keyword arguments (line 92)
        kwargs_191451 = {}
        # Getting the type of 'self' (line 92)
        self_191447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 92)
        stream_191448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), self_191447, 'stream')
        # Obtaining the member 'writeln' of a type (line 92)
        writeln_191449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), stream_191448, 'writeln')
        # Calling writeln(args, kwargs) (line 92)
        writeln_call_result_191452 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), writeln_191449, *[str_191450], **kwargs_191451)
        
        # SSA branch for the else part of an if statement (line 91)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 93)
        self_191453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 13), 'self')
        # Obtaining the member 'dots' of a type (line 93)
        dots_191454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 13), self_191453, 'dots')
        # Testing the type of an if condition (line 93)
        if_condition_191455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 13), dots_191454)
        # Assigning a type to the variable 'if_condition_191455' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 13), 'if_condition_191455', if_condition_191455)
        # SSA begins for if statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 94)
        # Processing the call arguments (line 94)
        str_191459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 30), 'str', 'x')
        # Processing the call keyword arguments (line 94)
        kwargs_191460 = {}
        # Getting the type of 'self' (line 94)
        self_191456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 94)
        stream_191457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), self_191456, 'stream')
        # Obtaining the member 'write' of a type (line 94)
        write_191458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), stream_191457, 'write')
        # Calling write(args, kwargs) (line 94)
        write_call_result_191461 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), write_191458, *[str_191459], **kwargs_191460)
        
        
        # Call to flush(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_191465 = {}
        # Getting the type of 'self' (line 95)
        self_191462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 95)
        stream_191463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), self_191462, 'stream')
        # Obtaining the member 'flush' of a type (line 95)
        flush_191464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), stream_191463, 'flush')
        # Calling flush(args, kwargs) (line 95)
        flush_call_result_191466 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), flush_191464, *[], **kwargs_191465)
        
        # SSA join for if statement (line 93)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 91)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'addExpectedFailure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addExpectedFailure' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_191467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191467)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addExpectedFailure'
        return stypy_return_type_191467


    @norecursion
    def addUnexpectedSuccess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addUnexpectedSuccess'
        module_type_store = module_type_store.open_function_context('addUnexpectedSuccess', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextTestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_localization', localization)
        TextTestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextTestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextTestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_function_name', 'TextTestResult.addUnexpectedSuccess')
        TextTestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_param_names_list', ['test'])
        TextTestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextTestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextTestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextTestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextTestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextTestResult.addUnexpectedSuccess.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestResult.addUnexpectedSuccess', ['test'], None, None, defaults, varargs, kwargs)

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

        
        # Call to addUnexpectedSuccess(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'test' (line 98)
        test_191474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 57), 'test', False)
        # Processing the call keyword arguments (line 98)
        kwargs_191475 = {}
        
        # Call to super(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'TextTestResult' (line 98)
        TextTestResult_191469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'TextTestResult', False)
        # Getting the type of 'self' (line 98)
        self_191470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), 'self', False)
        # Processing the call keyword arguments (line 98)
        kwargs_191471 = {}
        # Getting the type of 'super' (line 98)
        super_191468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'super', False)
        # Calling super(args, kwargs) (line 98)
        super_call_result_191472 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), super_191468, *[TextTestResult_191469, self_191470], **kwargs_191471)
        
        # Obtaining the member 'addUnexpectedSuccess' of a type (line 98)
        addUnexpectedSuccess_191473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), super_call_result_191472, 'addUnexpectedSuccess')
        # Calling addUnexpectedSuccess(args, kwargs) (line 98)
        addUnexpectedSuccess_call_result_191476 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), addUnexpectedSuccess_191473, *[test_191474], **kwargs_191475)
        
        
        # Getting the type of 'self' (line 99)
        self_191477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'self')
        # Obtaining the member 'showAll' of a type (line 99)
        showAll_191478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 11), self_191477, 'showAll')
        # Testing the type of an if condition (line 99)
        if_condition_191479 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 8), showAll_191478)
        # Assigning a type to the variable 'if_condition_191479' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'if_condition_191479', if_condition_191479)
        # SSA begins for if statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to writeln(...): (line 100)
        # Processing the call arguments (line 100)
        str_191483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 32), 'str', 'unexpected success')
        # Processing the call keyword arguments (line 100)
        kwargs_191484 = {}
        # Getting the type of 'self' (line 100)
        self_191480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 100)
        stream_191481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_191480, 'stream')
        # Obtaining the member 'writeln' of a type (line 100)
        writeln_191482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), stream_191481, 'writeln')
        # Calling writeln(args, kwargs) (line 100)
        writeln_call_result_191485 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), writeln_191482, *[str_191483], **kwargs_191484)
        
        # SSA branch for the else part of an if statement (line 99)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 101)
        self_191486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), 'self')
        # Obtaining the member 'dots' of a type (line 101)
        dots_191487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 13), self_191486, 'dots')
        # Testing the type of an if condition (line 101)
        if_condition_191488 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 13), dots_191487)
        # Assigning a type to the variable 'if_condition_191488' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), 'if_condition_191488', if_condition_191488)
        # SSA begins for if statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 102)
        # Processing the call arguments (line 102)
        str_191492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 30), 'str', 'u')
        # Processing the call keyword arguments (line 102)
        kwargs_191493 = {}
        # Getting the type of 'self' (line 102)
        self_191489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 102)
        stream_191490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), self_191489, 'stream')
        # Obtaining the member 'write' of a type (line 102)
        write_191491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), stream_191490, 'write')
        # Calling write(args, kwargs) (line 102)
        write_call_result_191494 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), write_191491, *[str_191492], **kwargs_191493)
        
        
        # Call to flush(...): (line 103)
        # Processing the call keyword arguments (line 103)
        kwargs_191498 = {}
        # Getting the type of 'self' (line 103)
        self_191495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 103)
        stream_191496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_191495, 'stream')
        # Obtaining the member 'flush' of a type (line 103)
        flush_191497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), stream_191496, 'flush')
        # Calling flush(args, kwargs) (line 103)
        flush_call_result_191499 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), flush_191497, *[], **kwargs_191498)
        
        # SSA join for if statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'addUnexpectedSuccess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addUnexpectedSuccess' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_191500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191500)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addUnexpectedSuccess'
        return stypy_return_type_191500


    @norecursion
    def printErrors(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'printErrors'
        module_type_store = module_type_store.open_function_context('printErrors', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextTestResult.printErrors.__dict__.__setitem__('stypy_localization', localization)
        TextTestResult.printErrors.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextTestResult.printErrors.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextTestResult.printErrors.__dict__.__setitem__('stypy_function_name', 'TextTestResult.printErrors')
        TextTestResult.printErrors.__dict__.__setitem__('stypy_param_names_list', [])
        TextTestResult.printErrors.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextTestResult.printErrors.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextTestResult.printErrors.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextTestResult.printErrors.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextTestResult.printErrors.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextTestResult.printErrors.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestResult.printErrors', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 106)
        self_191501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'self')
        # Obtaining the member 'dots' of a type (line 106)
        dots_191502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 11), self_191501, 'dots')
        # Getting the type of 'self' (line 106)
        self_191503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'self')
        # Obtaining the member 'showAll' of a type (line 106)
        showAll_191504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 24), self_191503, 'showAll')
        # Applying the binary operator 'or' (line 106)
        result_or_keyword_191505 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 11), 'or', dots_191502, showAll_191504)
        
        # Testing the type of an if condition (line 106)
        if_condition_191506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), result_or_keyword_191505)
        # Assigning a type to the variable 'if_condition_191506' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_191506', if_condition_191506)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to writeln(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_191510 = {}
        # Getting the type of 'self' (line 107)
        self_191507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 107)
        stream_191508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), self_191507, 'stream')
        # Obtaining the member 'writeln' of a type (line 107)
        writeln_191509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), stream_191508, 'writeln')
        # Calling writeln(args, kwargs) (line 107)
        writeln_call_result_191511 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), writeln_191509, *[], **kwargs_191510)
        
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to printErrorList(...): (line 108)
        # Processing the call arguments (line 108)
        str_191514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 28), 'str', 'ERROR')
        # Getting the type of 'self' (line 108)
        self_191515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 37), 'self', False)
        # Obtaining the member 'errors' of a type (line 108)
        errors_191516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 37), self_191515, 'errors')
        # Processing the call keyword arguments (line 108)
        kwargs_191517 = {}
        # Getting the type of 'self' (line 108)
        self_191512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self', False)
        # Obtaining the member 'printErrorList' of a type (line 108)
        printErrorList_191513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_191512, 'printErrorList')
        # Calling printErrorList(args, kwargs) (line 108)
        printErrorList_call_result_191518 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), printErrorList_191513, *[str_191514, errors_191516], **kwargs_191517)
        
        
        # Call to printErrorList(...): (line 109)
        # Processing the call arguments (line 109)
        str_191521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 28), 'str', 'FAIL')
        # Getting the type of 'self' (line 109)
        self_191522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 36), 'self', False)
        # Obtaining the member 'failures' of a type (line 109)
        failures_191523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 36), self_191522, 'failures')
        # Processing the call keyword arguments (line 109)
        kwargs_191524 = {}
        # Getting the type of 'self' (line 109)
        self_191519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self', False)
        # Obtaining the member 'printErrorList' of a type (line 109)
        printErrorList_191520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_191519, 'printErrorList')
        # Calling printErrorList(args, kwargs) (line 109)
        printErrorList_call_result_191525 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), printErrorList_191520, *[str_191521, failures_191523], **kwargs_191524)
        
        
        # ################# End of 'printErrors(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'printErrors' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_191526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191526)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'printErrors'
        return stypy_return_type_191526


    @norecursion
    def printErrorList(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'printErrorList'
        module_type_store = module_type_store.open_function_context('printErrorList', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextTestResult.printErrorList.__dict__.__setitem__('stypy_localization', localization)
        TextTestResult.printErrorList.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextTestResult.printErrorList.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextTestResult.printErrorList.__dict__.__setitem__('stypy_function_name', 'TextTestResult.printErrorList')
        TextTestResult.printErrorList.__dict__.__setitem__('stypy_param_names_list', ['flavour', 'errors'])
        TextTestResult.printErrorList.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextTestResult.printErrorList.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextTestResult.printErrorList.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextTestResult.printErrorList.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextTestResult.printErrorList.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextTestResult.printErrorList.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestResult.printErrorList', ['flavour', 'errors'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'printErrorList', localization, ['flavour', 'errors'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'printErrorList(...)' code ##################

        
        # Getting the type of 'errors' (line 112)
        errors_191527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 25), 'errors')
        # Testing the type of a for loop iterable (line 112)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 112, 8), errors_191527)
        # Getting the type of the for loop variable (line 112)
        for_loop_var_191528 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 112, 8), errors_191527)
        # Assigning a type to the variable 'test' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'test', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 8), for_loop_var_191528))
        # Assigning a type to the variable 'err' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'err', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 8), for_loop_var_191528))
        # SSA begins for a for statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to writeln(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'self' (line 113)
        self_191532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 32), 'self', False)
        # Obtaining the member 'separator1' of a type (line 113)
        separator1_191533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 32), self_191532, 'separator1')
        # Processing the call keyword arguments (line 113)
        kwargs_191534 = {}
        # Getting the type of 'self' (line 113)
        self_191529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 113)
        stream_191530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), self_191529, 'stream')
        # Obtaining the member 'writeln' of a type (line 113)
        writeln_191531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), stream_191530, 'writeln')
        # Calling writeln(args, kwargs) (line 113)
        writeln_call_result_191535 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), writeln_191531, *[separator1_191533], **kwargs_191534)
        
        
        # Call to writeln(...): (line 114)
        # Processing the call arguments (line 114)
        str_191539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 32), 'str', '%s: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 114)
        tuple_191540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 114)
        # Adding element type (line 114)
        # Getting the type of 'flavour' (line 114)
        flavour_191541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 44), 'flavour', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 44), tuple_191540, flavour_191541)
        # Adding element type (line 114)
        
        # Call to getDescription(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'test' (line 114)
        test_191544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 72), 'test', False)
        # Processing the call keyword arguments (line 114)
        kwargs_191545 = {}
        # Getting the type of 'self' (line 114)
        self_191542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 52), 'self', False)
        # Obtaining the member 'getDescription' of a type (line 114)
        getDescription_191543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 52), self_191542, 'getDescription')
        # Calling getDescription(args, kwargs) (line 114)
        getDescription_call_result_191546 = invoke(stypy.reporting.localization.Localization(__file__, 114, 52), getDescription_191543, *[test_191544], **kwargs_191545)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 44), tuple_191540, getDescription_call_result_191546)
        
        # Applying the binary operator '%' (line 114)
        result_mod_191547 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 32), '%', str_191539, tuple_191540)
        
        # Processing the call keyword arguments (line 114)
        kwargs_191548 = {}
        # Getting the type of 'self' (line 114)
        self_191536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 114)
        stream_191537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), self_191536, 'stream')
        # Obtaining the member 'writeln' of a type (line 114)
        writeln_191538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), stream_191537, 'writeln')
        # Calling writeln(args, kwargs) (line 114)
        writeln_call_result_191549 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), writeln_191538, *[result_mod_191547], **kwargs_191548)
        
        
        # Call to writeln(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'self' (line 115)
        self_191553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'self', False)
        # Obtaining the member 'separator2' of a type (line 115)
        separator2_191554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 32), self_191553, 'separator2')
        # Processing the call keyword arguments (line 115)
        kwargs_191555 = {}
        # Getting the type of 'self' (line 115)
        self_191550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 115)
        stream_191551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), self_191550, 'stream')
        # Obtaining the member 'writeln' of a type (line 115)
        writeln_191552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), stream_191551, 'writeln')
        # Calling writeln(args, kwargs) (line 115)
        writeln_call_result_191556 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), writeln_191552, *[separator2_191554], **kwargs_191555)
        
        
        # Call to writeln(...): (line 116)
        # Processing the call arguments (line 116)
        str_191560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 32), 'str', '%s')
        # Getting the type of 'err' (line 116)
        err_191561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 39), 'err', False)
        # Applying the binary operator '%' (line 116)
        result_mod_191562 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 32), '%', str_191560, err_191561)
        
        # Processing the call keyword arguments (line 116)
        kwargs_191563 = {}
        # Getting the type of 'self' (line 116)
        self_191557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 116)
        stream_191558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), self_191557, 'stream')
        # Obtaining the member 'writeln' of a type (line 116)
        writeln_191559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), stream_191558, 'writeln')
        # Calling writeln(args, kwargs) (line 116)
        writeln_call_result_191564 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), writeln_191559, *[result_mod_191562], **kwargs_191563)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'printErrorList(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'printErrorList' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_191565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191565)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'printErrorList'
        return stypy_return_type_191565


# Assigning a type to the variable 'TextTestResult' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'TextTestResult', TextTestResult)

# Assigning a BinOp to a Name (line 33):
str_191566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 17), 'str', '=')
int_191567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'int')
# Applying the binary operator '*' (line 33)
result_mul_191568 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 17), '*', str_191566, int_191567)

# Getting the type of 'TextTestResult'
TextTestResult_191569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TextTestResult')
# Setting the type of the member 'separator1' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TextTestResult_191569, 'separator1', result_mul_191568)

# Assigning a BinOp to a Name (line 34):
str_191570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 17), 'str', '-')
int_191571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'int')
# Applying the binary operator '*' (line 34)
result_mul_191572 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 17), '*', str_191570, int_191571)

# Getting the type of 'TextTestResult'
TextTestResult_191573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TextTestResult')
# Setting the type of the member 'separator2' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TextTestResult_191573, 'separator2', result_mul_191572)
# Declaration of the 'TextTestRunner' class

class TextTestRunner(object, ):
    str_191574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, (-1)), 'str', 'A test runner class that displays results in textual form.\n\n    It prints out the names of tests as they are run, errors as they\n    occur, and a summary of the results at the end of the test run.\n    ')
    
    # Assigning a Name to a Name (line 125):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'sys' (line 127)
        sys_191575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 30), 'sys')
        # Obtaining the member 'stderr' of a type (line 127)
        stderr_191576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), sys_191575, 'stderr')
        # Getting the type of 'True' (line 127)
        True_191577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 55), 'True')
        int_191578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 71), 'int')
        # Getting the type of 'False' (line 128)
        False_191579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 26), 'False')
        # Getting the type of 'False' (line 128)
        False_191580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 40), 'False')
        # Getting the type of 'None' (line 128)
        None_191581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 59), 'None')
        defaults = [stderr_191576, True_191577, int_191578, False_191579, False_191580, None_191581]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestRunner.__init__', ['stream', 'descriptions', 'verbosity', 'failfast', 'buffer', 'resultclass'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['stream', 'descriptions', 'verbosity', 'failfast', 'buffer', 'resultclass'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 129):
        
        # Assigning a Call to a Attribute (line 129):
        
        # Call to _WritelnDecorator(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'stream' (line 129)
        stream_191583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'stream', False)
        # Processing the call keyword arguments (line 129)
        kwargs_191584 = {}
        # Getting the type of '_WritelnDecorator' (line 129)
        _WritelnDecorator_191582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 22), '_WritelnDecorator', False)
        # Calling _WritelnDecorator(args, kwargs) (line 129)
        _WritelnDecorator_call_result_191585 = invoke(stypy.reporting.localization.Localization(__file__, 129, 22), _WritelnDecorator_191582, *[stream_191583], **kwargs_191584)
        
        # Getting the type of 'self' (line 129)
        self_191586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Setting the type of the member 'stream' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_191586, 'stream', _WritelnDecorator_call_result_191585)
        
        # Assigning a Name to a Attribute (line 130):
        
        # Assigning a Name to a Attribute (line 130):
        # Getting the type of 'descriptions' (line 130)
        descriptions_191587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'descriptions')
        # Getting the type of 'self' (line 130)
        self_191588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self')
        # Setting the type of the member 'descriptions' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_191588, 'descriptions', descriptions_191587)
        
        # Assigning a Name to a Attribute (line 131):
        
        # Assigning a Name to a Attribute (line 131):
        # Getting the type of 'verbosity' (line 131)
        verbosity_191589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'verbosity')
        # Getting the type of 'self' (line 131)
        self_191590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self')
        # Setting the type of the member 'verbosity' of a type (line 131)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_191590, 'verbosity', verbosity_191589)
        
        # Assigning a Name to a Attribute (line 132):
        
        # Assigning a Name to a Attribute (line 132):
        # Getting the type of 'failfast' (line 132)
        failfast_191591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'failfast')
        # Getting the type of 'self' (line 132)
        self_191592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'self')
        # Setting the type of the member 'failfast' of a type (line 132)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), self_191592, 'failfast', failfast_191591)
        
        # Assigning a Name to a Attribute (line 133):
        
        # Assigning a Name to a Attribute (line 133):
        # Getting the type of 'buffer' (line 133)
        buffer_191593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'buffer')
        # Getting the type of 'self' (line 133)
        self_191594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'self')
        # Setting the type of the member 'buffer' of a type (line 133)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), self_191594, 'buffer', buffer_191593)
        
        # Type idiom detected: calculating its left and rigth part (line 134)
        # Getting the type of 'resultclass' (line 134)
        resultclass_191595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'resultclass')
        # Getting the type of 'None' (line 134)
        None_191596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 30), 'None')
        
        (may_be_191597, more_types_in_union_191598) = may_not_be_none(resultclass_191595, None_191596)

        if may_be_191597:

            if more_types_in_union_191598:
                # Runtime conditional SSA (line 134)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 135):
            
            # Assigning a Name to a Attribute (line 135):
            # Getting the type of 'resultclass' (line 135)
            resultclass_191599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'resultclass')
            # Getting the type of 'self' (line 135)
            self_191600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'self')
            # Setting the type of the member 'resultclass' of a type (line 135)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), self_191600, 'resultclass', resultclass_191599)

            if more_types_in_union_191598:
                # SSA join for if statement (line 134)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _makeResult(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_makeResult'
        module_type_store = module_type_store.open_function_context('_makeResult', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextTestRunner._makeResult.__dict__.__setitem__('stypy_localization', localization)
        TextTestRunner._makeResult.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextTestRunner._makeResult.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextTestRunner._makeResult.__dict__.__setitem__('stypy_function_name', 'TextTestRunner._makeResult')
        TextTestRunner._makeResult.__dict__.__setitem__('stypy_param_names_list', [])
        TextTestRunner._makeResult.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextTestRunner._makeResult.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextTestRunner._makeResult.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextTestRunner._makeResult.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextTestRunner._makeResult.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextTestRunner._makeResult.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestRunner._makeResult', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_makeResult', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_makeResult(...)' code ##################

        
        # Call to resultclass(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'self' (line 138)
        self_191603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 32), 'self', False)
        # Obtaining the member 'stream' of a type (line 138)
        stream_191604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 32), self_191603, 'stream')
        # Getting the type of 'self' (line 138)
        self_191605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 45), 'self', False)
        # Obtaining the member 'descriptions' of a type (line 138)
        descriptions_191606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 45), self_191605, 'descriptions')
        # Getting the type of 'self' (line 138)
        self_191607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 64), 'self', False)
        # Obtaining the member 'verbosity' of a type (line 138)
        verbosity_191608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 64), self_191607, 'verbosity')
        # Processing the call keyword arguments (line 138)
        kwargs_191609 = {}
        # Getting the type of 'self' (line 138)
        self_191601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'self', False)
        # Obtaining the member 'resultclass' of a type (line 138)
        resultclass_191602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), self_191601, 'resultclass')
        # Calling resultclass(args, kwargs) (line 138)
        resultclass_call_result_191610 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), resultclass_191602, *[stream_191604, descriptions_191606, verbosity_191608], **kwargs_191609)
        
        # Assigning a type to the variable 'stypy_return_type' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type', resultclass_call_result_191610)
        
        # ################# End of '_makeResult(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_makeResult' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_191611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191611)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_makeResult'
        return stypy_return_type_191611


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextTestRunner.run.__dict__.__setitem__('stypy_localization', localization)
        TextTestRunner.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextTestRunner.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextTestRunner.run.__dict__.__setitem__('stypy_function_name', 'TextTestRunner.run')
        TextTestRunner.run.__dict__.__setitem__('stypy_param_names_list', ['test'])
        TextTestRunner.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextTestRunner.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextTestRunner.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextTestRunner.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextTestRunner.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextTestRunner.run.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextTestRunner.run', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        str_191612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'str', 'Run the given test case or test suite.')
        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to _makeResult(...): (line 142)
        # Processing the call keyword arguments (line 142)
        kwargs_191615 = {}
        # Getting the type of 'self' (line 142)
        self_191613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 17), 'self', False)
        # Obtaining the member '_makeResult' of a type (line 142)
        _makeResult_191614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 17), self_191613, '_makeResult')
        # Calling _makeResult(args, kwargs) (line 142)
        _makeResult_call_result_191616 = invoke(stypy.reporting.localization.Localization(__file__, 142, 17), _makeResult_191614, *[], **kwargs_191615)
        
        # Assigning a type to the variable 'result' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'result', _makeResult_call_result_191616)
        
        # Call to registerResult(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'result' (line 143)
        result_191618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'result', False)
        # Processing the call keyword arguments (line 143)
        kwargs_191619 = {}
        # Getting the type of 'registerResult' (line 143)
        registerResult_191617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'registerResult', False)
        # Calling registerResult(args, kwargs) (line 143)
        registerResult_call_result_191620 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), registerResult_191617, *[result_191618], **kwargs_191619)
        
        
        # Assigning a Attribute to a Attribute (line 144):
        
        # Assigning a Attribute to a Attribute (line 144):
        # Getting the type of 'self' (line 144)
        self_191621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 'self')
        # Obtaining the member 'failfast' of a type (line 144)
        failfast_191622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 26), self_191621, 'failfast')
        # Getting the type of 'result' (line 144)
        result_191623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'result')
        # Setting the type of the member 'failfast' of a type (line 144)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), result_191623, 'failfast', failfast_191622)
        
        # Assigning a Attribute to a Attribute (line 145):
        
        # Assigning a Attribute to a Attribute (line 145):
        # Getting the type of 'self' (line 145)
        self_191624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 24), 'self')
        # Obtaining the member 'buffer' of a type (line 145)
        buffer_191625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 24), self_191624, 'buffer')
        # Getting the type of 'result' (line 145)
        result_191626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'result')
        # Setting the type of the member 'buffer' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), result_191626, 'buffer', buffer_191625)
        
        # Assigning a Call to a Name (line 146):
        
        # Assigning a Call to a Name (line 146):
        
        # Call to time(...): (line 146)
        # Processing the call keyword arguments (line 146)
        kwargs_191629 = {}
        # Getting the type of 'time' (line 146)
        time_191627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'time', False)
        # Obtaining the member 'time' of a type (line 146)
        time_191628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 20), time_191627, 'time')
        # Calling time(args, kwargs) (line 146)
        time_call_result_191630 = invoke(stypy.reporting.localization.Localization(__file__, 146, 20), time_191628, *[], **kwargs_191629)
        
        # Assigning a type to the variable 'startTime' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'startTime', time_call_result_191630)
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to getattr(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'result' (line 147)
        result_191632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 31), 'result', False)
        str_191633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 39), 'str', 'startTestRun')
        # Getting the type of 'None' (line 147)
        None_191634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 55), 'None', False)
        # Processing the call keyword arguments (line 147)
        kwargs_191635 = {}
        # Getting the type of 'getattr' (line 147)
        getattr_191631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'getattr', False)
        # Calling getattr(args, kwargs) (line 147)
        getattr_call_result_191636 = invoke(stypy.reporting.localization.Localization(__file__, 147, 23), getattr_191631, *[result_191632, str_191633, None_191634], **kwargs_191635)
        
        # Assigning a type to the variable 'startTestRun' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'startTestRun', getattr_call_result_191636)
        
        # Type idiom detected: calculating its left and rigth part (line 148)
        # Getting the type of 'startTestRun' (line 148)
        startTestRun_191637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'startTestRun')
        # Getting the type of 'None' (line 148)
        None_191638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 31), 'None')
        
        (may_be_191639, more_types_in_union_191640) = may_not_be_none(startTestRun_191637, None_191638)

        if may_be_191639:

            if more_types_in_union_191640:
                # Runtime conditional SSA (line 148)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to startTestRun(...): (line 149)
            # Processing the call keyword arguments (line 149)
            kwargs_191642 = {}
            # Getting the type of 'startTestRun' (line 149)
            startTestRun_191641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'startTestRun', False)
            # Calling startTestRun(args, kwargs) (line 149)
            startTestRun_call_result_191643 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), startTestRun_191641, *[], **kwargs_191642)
            

            if more_types_in_union_191640:
                # SSA join for if statement (line 148)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Try-finally block (line 150)
        
        # Call to test(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'result' (line 151)
        result_191645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 17), 'result', False)
        # Processing the call keyword arguments (line 151)
        kwargs_191646 = {}
        # Getting the type of 'test' (line 151)
        test_191644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'test', False)
        # Calling test(args, kwargs) (line 151)
        test_call_result_191647 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), test_191644, *[result_191645], **kwargs_191646)
        
        
        # finally branch of the try-finally block (line 150)
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to getattr(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'result' (line 153)
        result_191649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), 'result', False)
        str_191650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 42), 'str', 'stopTestRun')
        # Getting the type of 'None' (line 153)
        None_191651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 57), 'None', False)
        # Processing the call keyword arguments (line 153)
        kwargs_191652 = {}
        # Getting the type of 'getattr' (line 153)
        getattr_191648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 26), 'getattr', False)
        # Calling getattr(args, kwargs) (line 153)
        getattr_call_result_191653 = invoke(stypy.reporting.localization.Localization(__file__, 153, 26), getattr_191648, *[result_191649, str_191650, None_191651], **kwargs_191652)
        
        # Assigning a type to the variable 'stopTestRun' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'stopTestRun', getattr_call_result_191653)
        
        # Type idiom detected: calculating its left and rigth part (line 154)
        # Getting the type of 'stopTestRun' (line 154)
        stopTestRun_191654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'stopTestRun')
        # Getting the type of 'None' (line 154)
        None_191655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), 'None')
        
        (may_be_191656, more_types_in_union_191657) = may_not_be_none(stopTestRun_191654, None_191655)

        if may_be_191656:

            if more_types_in_union_191657:
                # Runtime conditional SSA (line 154)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to stopTestRun(...): (line 155)
            # Processing the call keyword arguments (line 155)
            kwargs_191659 = {}
            # Getting the type of 'stopTestRun' (line 155)
            stopTestRun_191658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'stopTestRun', False)
            # Calling stopTestRun(args, kwargs) (line 155)
            stopTestRun_call_result_191660 = invoke(stypy.reporting.localization.Localization(__file__, 155, 16), stopTestRun_191658, *[], **kwargs_191659)
            

            if more_types_in_union_191657:
                # SSA join for if statement (line 154)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Assigning a Call to a Name (line 156):
        
        # Assigning a Call to a Name (line 156):
        
        # Call to time(...): (line 156)
        # Processing the call keyword arguments (line 156)
        kwargs_191663 = {}
        # Getting the type of 'time' (line 156)
        time_191661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'time', False)
        # Obtaining the member 'time' of a type (line 156)
        time_191662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 19), time_191661, 'time')
        # Calling time(args, kwargs) (line 156)
        time_call_result_191664 = invoke(stypy.reporting.localization.Localization(__file__, 156, 19), time_191662, *[], **kwargs_191663)
        
        # Assigning a type to the variable 'stopTime' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'stopTime', time_call_result_191664)
        
        # Assigning a BinOp to a Name (line 157):
        
        # Assigning a BinOp to a Name (line 157):
        # Getting the type of 'stopTime' (line 157)
        stopTime_191665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'stopTime')
        # Getting the type of 'startTime' (line 157)
        startTime_191666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 31), 'startTime')
        # Applying the binary operator '-' (line 157)
        result_sub_191667 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 20), '-', stopTime_191665, startTime_191666)
        
        # Assigning a type to the variable 'timeTaken' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'timeTaken', result_sub_191667)
        
        # Call to printErrors(...): (line 158)
        # Processing the call keyword arguments (line 158)
        kwargs_191670 = {}
        # Getting the type of 'result' (line 158)
        result_191668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'result', False)
        # Obtaining the member 'printErrors' of a type (line 158)
        printErrors_191669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), result_191668, 'printErrors')
        # Calling printErrors(args, kwargs) (line 158)
        printErrors_call_result_191671 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), printErrors_191669, *[], **kwargs_191670)
        
        
        # Type idiom detected: calculating its left and rigth part (line 159)
        str_191672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 27), 'str', 'separator2')
        # Getting the type of 'result' (line 159)
        result_191673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'result')
        
        (may_be_191674, more_types_in_union_191675) = may_provide_member(str_191672, result_191673)

        if may_be_191674:

            if more_types_in_union_191675:
                # Runtime conditional SSA (line 159)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'result' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'result', remove_not_member_provider_from_union(result_191673, 'separator2'))
            
            # Call to writeln(...): (line 160)
            # Processing the call arguments (line 160)
            # Getting the type of 'result' (line 160)
            result_191679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'result', False)
            # Obtaining the member 'separator2' of a type (line 160)
            separator2_191680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 32), result_191679, 'separator2')
            # Processing the call keyword arguments (line 160)
            kwargs_191681 = {}
            # Getting the type of 'self' (line 160)
            self_191676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'self', False)
            # Obtaining the member 'stream' of a type (line 160)
            stream_191677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), self_191676, 'stream')
            # Obtaining the member 'writeln' of a type (line 160)
            writeln_191678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), stream_191677, 'writeln')
            # Calling writeln(args, kwargs) (line 160)
            writeln_call_result_191682 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), writeln_191678, *[separator2_191680], **kwargs_191681)
            

            if more_types_in_union_191675:
                # SSA join for if statement (line 159)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Name (line 161):
        
        # Assigning a Attribute to a Name (line 161):
        # Getting the type of 'result' (line 161)
        result_191683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 14), 'result')
        # Obtaining the member 'testsRun' of a type (line 161)
        testsRun_191684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 14), result_191683, 'testsRun')
        # Assigning a type to the variable 'run' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'run', testsRun_191684)
        
        # Call to writeln(...): (line 162)
        # Processing the call arguments (line 162)
        str_191688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 28), 'str', 'Ran %d test%s in %.3fs')
        
        # Obtaining an instance of the builtin type 'tuple' (line 163)
        tuple_191689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 163)
        # Adding element type (line 163)
        # Getting the type of 'run' (line 163)
        run_191690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'run', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 29), tuple_191689, run_191690)
        # Adding element type (line 163)
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Getting the type of 'run' (line 163)
        run_191691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), 'run', False)
        int_191692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 41), 'int')
        # Applying the binary operator '!=' (line 163)
        result_ne_191693 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 34), '!=', run_191691, int_191692)
        
        str_191694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 47), 'str', 's')
        # Applying the binary operator 'and' (line 163)
        result_and_keyword_191695 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 34), 'and', result_ne_191693, str_191694)
        
        str_191696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 54), 'str', '')
        # Applying the binary operator 'or' (line 163)
        result_or_keyword_191697 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 34), 'or', result_and_keyword_191695, str_191696)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 29), tuple_191689, result_or_keyword_191697)
        # Adding element type (line 163)
        # Getting the type of 'timeTaken' (line 163)
        timeTaken_191698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 58), 'timeTaken', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 29), tuple_191689, timeTaken_191698)
        
        # Applying the binary operator '%' (line 162)
        result_mod_191699 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 28), '%', str_191688, tuple_191689)
        
        # Processing the call keyword arguments (line 162)
        kwargs_191700 = {}
        # Getting the type of 'self' (line 162)
        self_191685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self', False)
        # Obtaining the member 'stream' of a type (line 162)
        stream_191686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_191685, 'stream')
        # Obtaining the member 'writeln' of a type (line 162)
        writeln_191687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), stream_191686, 'writeln')
        # Calling writeln(args, kwargs) (line 162)
        writeln_call_result_191701 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), writeln_191687, *[result_mod_191699], **kwargs_191700)
        
        
        # Call to writeln(...): (line 164)
        # Processing the call keyword arguments (line 164)
        kwargs_191705 = {}
        # Getting the type of 'self' (line 164)
        self_191702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'self', False)
        # Obtaining the member 'stream' of a type (line 164)
        stream_191703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), self_191702, 'stream')
        # Obtaining the member 'writeln' of a type (line 164)
        writeln_191704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), stream_191703, 'writeln')
        # Calling writeln(args, kwargs) (line 164)
        writeln_call_result_191706 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), writeln_191704, *[], **kwargs_191705)
        
        
        # Multiple assignment of 3 elements.
        
        # Assigning a Num to a Name (line 166):
        int_191707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 56), 'int')
        # Assigning a type to the variable 'skipped' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 46), 'skipped', int_191707)
        
        # Assigning a Name to a Name (line 166):
        # Getting the type of 'skipped' (line 166)
        skipped_191708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 46), 'skipped')
        # Assigning a type to the variable 'unexpectedSuccesses' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'unexpectedSuccesses', skipped_191708)
        
        # Assigning a Name to a Name (line 166):
        # Getting the type of 'unexpectedSuccesses' (line 166)
        unexpectedSuccesses_191709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'unexpectedSuccesses')
        # Assigning a type to the variable 'expectedFails' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'expectedFails', unexpectedSuccesses_191709)
        
        
        # SSA begins for try-except statement (line 167)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 168):
        
        # Assigning a Call to a Name (line 168):
        
        # Call to map(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'len' (line 168)
        len_191711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 26), 'len', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 168)
        tuple_191712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 168)
        # Adding element type (line 168)
        # Getting the type of 'result' (line 168)
        result_191713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 32), 'result', False)
        # Obtaining the member 'expectedFailures' of a type (line 168)
        expectedFailures_191714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 32), result_191713, 'expectedFailures')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 32), tuple_191712, expectedFailures_191714)
        # Adding element type (line 168)
        # Getting the type of 'result' (line 169)
        result_191715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 32), 'result', False)
        # Obtaining the member 'unexpectedSuccesses' of a type (line 169)
        unexpectedSuccesses_191716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 32), result_191715, 'unexpectedSuccesses')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 32), tuple_191712, unexpectedSuccesses_191716)
        # Adding element type (line 168)
        # Getting the type of 'result' (line 170)
        result_191717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 32), 'result', False)
        # Obtaining the member 'skipped' of a type (line 170)
        skipped_191718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 32), result_191717, 'skipped')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 32), tuple_191712, skipped_191718)
        
        # Processing the call keyword arguments (line 168)
        kwargs_191719 = {}
        # Getting the type of 'map' (line 168)
        map_191710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'map', False)
        # Calling map(args, kwargs) (line 168)
        map_call_result_191720 = invoke(stypy.reporting.localization.Localization(__file__, 168, 22), map_191710, *[len_191711, tuple_191712], **kwargs_191719)
        
        # Assigning a type to the variable 'results' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'results', map_call_result_191720)
        # SSA branch for the except part of a try statement (line 167)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 167)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 167)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a Name to a Tuple (line 174):
        
        # Assigning a Subscript to a Name (line 174):
        
        # Obtaining the type of the subscript
        int_191721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 12), 'int')
        # Getting the type of 'results' (line 174)
        results_191722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 58), 'results')
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___191723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 12), results_191722, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_191724 = invoke(stypy.reporting.localization.Localization(__file__, 174, 12), getitem___191723, int_191721)
        
        # Assigning a type to the variable 'tuple_var_assignment_191167' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'tuple_var_assignment_191167', subscript_call_result_191724)
        
        # Assigning a Subscript to a Name (line 174):
        
        # Obtaining the type of the subscript
        int_191725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 12), 'int')
        # Getting the type of 'results' (line 174)
        results_191726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 58), 'results')
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___191727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 12), results_191726, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_191728 = invoke(stypy.reporting.localization.Localization(__file__, 174, 12), getitem___191727, int_191725)
        
        # Assigning a type to the variable 'tuple_var_assignment_191168' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'tuple_var_assignment_191168', subscript_call_result_191728)
        
        # Assigning a Subscript to a Name (line 174):
        
        # Obtaining the type of the subscript
        int_191729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 12), 'int')
        # Getting the type of 'results' (line 174)
        results_191730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 58), 'results')
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___191731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 12), results_191730, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_191732 = invoke(stypy.reporting.localization.Localization(__file__, 174, 12), getitem___191731, int_191729)
        
        # Assigning a type to the variable 'tuple_var_assignment_191169' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'tuple_var_assignment_191169', subscript_call_result_191732)
        
        # Assigning a Name to a Name (line 174):
        # Getting the type of 'tuple_var_assignment_191167' (line 174)
        tuple_var_assignment_191167_191733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'tuple_var_assignment_191167')
        # Assigning a type to the variable 'expectedFails' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'expectedFails', tuple_var_assignment_191167_191733)
        
        # Assigning a Name to a Name (line 174):
        # Getting the type of 'tuple_var_assignment_191168' (line 174)
        tuple_var_assignment_191168_191734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'tuple_var_assignment_191168')
        # Assigning a type to the variable 'unexpectedSuccesses' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'unexpectedSuccesses', tuple_var_assignment_191168_191734)
        
        # Assigning a Name to a Name (line 174):
        # Getting the type of 'tuple_var_assignment_191169' (line 174)
        tuple_var_assignment_191169_191735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'tuple_var_assignment_191169')
        # Assigning a type to the variable 'skipped' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 48), 'skipped', tuple_var_assignment_191169_191735)
        # SSA join for try-except statement (line 167)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 176):
        
        # Assigning a List to a Name (line 176):
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_191736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        
        # Assigning a type to the variable 'infos' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'infos', list_191736)
        
        
        
        # Call to wasSuccessful(...): (line 177)
        # Processing the call keyword arguments (line 177)
        kwargs_191739 = {}
        # Getting the type of 'result' (line 177)
        result_191737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 'result', False)
        # Obtaining the member 'wasSuccessful' of a type (line 177)
        wasSuccessful_191738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 15), result_191737, 'wasSuccessful')
        # Calling wasSuccessful(args, kwargs) (line 177)
        wasSuccessful_call_result_191740 = invoke(stypy.reporting.localization.Localization(__file__, 177, 15), wasSuccessful_191738, *[], **kwargs_191739)
        
        # Applying the 'not' unary operator (line 177)
        result_not__191741 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 11), 'not', wasSuccessful_call_result_191740)
        
        # Testing the type of an if condition (line 177)
        if_condition_191742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 8), result_not__191741)
        # Assigning a type to the variable 'if_condition_191742' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'if_condition_191742', if_condition_191742)
        # SSA begins for if statement (line 177)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 178)
        # Processing the call arguments (line 178)
        str_191746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 30), 'str', 'FAILED')
        # Processing the call keyword arguments (line 178)
        kwargs_191747 = {}
        # Getting the type of 'self' (line 178)
        self_191743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 178)
        stream_191744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), self_191743, 'stream')
        # Obtaining the member 'write' of a type (line 178)
        write_191745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), stream_191744, 'write')
        # Calling write(args, kwargs) (line 178)
        write_call_result_191748 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), write_191745, *[str_191746], **kwargs_191747)
        
        
        # Assigning a Call to a Tuple (line 179):
        
        # Assigning a Call to a Name:
        
        # Call to map(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'len' (line 179)
        len_191750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 34), 'len', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 179)
        tuple_191751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 179)
        # Adding element type (line 179)
        # Getting the type of 'result' (line 179)
        result_191752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 40), 'result', False)
        # Obtaining the member 'failures' of a type (line 179)
        failures_191753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 40), result_191752, 'failures')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 40), tuple_191751, failures_191753)
        # Adding element type (line 179)
        # Getting the type of 'result' (line 179)
        result_191754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 57), 'result', False)
        # Obtaining the member 'errors' of a type (line 179)
        errors_191755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 57), result_191754, 'errors')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 40), tuple_191751, errors_191755)
        
        # Processing the call keyword arguments (line 179)
        kwargs_191756 = {}
        # Getting the type of 'map' (line 179)
        map_191749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 30), 'map', False)
        # Calling map(args, kwargs) (line 179)
        map_call_result_191757 = invoke(stypy.reporting.localization.Localization(__file__, 179, 30), map_191749, *[len_191750, tuple_191751], **kwargs_191756)
        
        # Assigning a type to the variable 'call_assignment_191170' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'call_assignment_191170', map_call_result_191757)
        
        # Assigning a Call to a Name (line 179):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_191760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 12), 'int')
        # Processing the call keyword arguments
        kwargs_191761 = {}
        # Getting the type of 'call_assignment_191170' (line 179)
        call_assignment_191170_191758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'call_assignment_191170', False)
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___191759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), call_assignment_191170_191758, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_191762 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___191759, *[int_191760], **kwargs_191761)
        
        # Assigning a type to the variable 'call_assignment_191171' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'call_assignment_191171', getitem___call_result_191762)
        
        # Assigning a Name to a Name (line 179):
        # Getting the type of 'call_assignment_191171' (line 179)
        call_assignment_191171_191763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'call_assignment_191171')
        # Assigning a type to the variable 'failed' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'failed', call_assignment_191171_191763)
        
        # Assigning a Call to a Name (line 179):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_191766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 12), 'int')
        # Processing the call keyword arguments
        kwargs_191767 = {}
        # Getting the type of 'call_assignment_191170' (line 179)
        call_assignment_191170_191764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'call_assignment_191170', False)
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___191765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), call_assignment_191170_191764, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_191768 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___191765, *[int_191766], **kwargs_191767)
        
        # Assigning a type to the variable 'call_assignment_191172' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'call_assignment_191172', getitem___call_result_191768)
        
        # Assigning a Name to a Name (line 179):
        # Getting the type of 'call_assignment_191172' (line 179)
        call_assignment_191172_191769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'call_assignment_191172')
        # Assigning a type to the variable 'errored' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'errored', call_assignment_191172_191769)
        
        # Getting the type of 'failed' (line 180)
        failed_191770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'failed')
        # Testing the type of an if condition (line 180)
        if_condition_191771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 12), failed_191770)
        # Assigning a type to the variable 'if_condition_191771' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'if_condition_191771', if_condition_191771)
        # SSA begins for if statement (line 180)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 181)
        # Processing the call arguments (line 181)
        str_191774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 29), 'str', 'failures=%d')
        # Getting the type of 'failed' (line 181)
        failed_191775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 45), 'failed', False)
        # Applying the binary operator '%' (line 181)
        result_mod_191776 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 29), '%', str_191774, failed_191775)
        
        # Processing the call keyword arguments (line 181)
        kwargs_191777 = {}
        # Getting the type of 'infos' (line 181)
        infos_191772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'infos', False)
        # Obtaining the member 'append' of a type (line 181)
        append_191773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 16), infos_191772, 'append')
        # Calling append(args, kwargs) (line 181)
        append_call_result_191778 = invoke(stypy.reporting.localization.Localization(__file__, 181, 16), append_191773, *[result_mod_191776], **kwargs_191777)
        
        # SSA join for if statement (line 180)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'errored' (line 182)
        errored_191779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'errored')
        # Testing the type of an if condition (line 182)
        if_condition_191780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 12), errored_191779)
        # Assigning a type to the variable 'if_condition_191780' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'if_condition_191780', if_condition_191780)
        # SSA begins for if statement (line 182)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 183)
        # Processing the call arguments (line 183)
        str_191783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 29), 'str', 'errors=%d')
        # Getting the type of 'errored' (line 183)
        errored_191784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 43), 'errored', False)
        # Applying the binary operator '%' (line 183)
        result_mod_191785 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 29), '%', str_191783, errored_191784)
        
        # Processing the call keyword arguments (line 183)
        kwargs_191786 = {}
        # Getting the type of 'infos' (line 183)
        infos_191781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'infos', False)
        # Obtaining the member 'append' of a type (line 183)
        append_191782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 16), infos_191781, 'append')
        # Calling append(args, kwargs) (line 183)
        append_call_result_191787 = invoke(stypy.reporting.localization.Localization(__file__, 183, 16), append_191782, *[result_mod_191785], **kwargs_191786)
        
        # SSA join for if statement (line 182)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 177)
        module_type_store.open_ssa_branch('else')
        
        # Call to write(...): (line 185)
        # Processing the call arguments (line 185)
        str_191791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 30), 'str', 'OK')
        # Processing the call keyword arguments (line 185)
        kwargs_191792 = {}
        # Getting the type of 'self' (line 185)
        self_191788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 185)
        stream_191789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 12), self_191788, 'stream')
        # Obtaining the member 'write' of a type (line 185)
        write_191790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 12), stream_191789, 'write')
        # Calling write(args, kwargs) (line 185)
        write_call_result_191793 = invoke(stypy.reporting.localization.Localization(__file__, 185, 12), write_191790, *[str_191791], **kwargs_191792)
        
        # SSA join for if statement (line 177)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'skipped' (line 186)
        skipped_191794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 11), 'skipped')
        # Testing the type of an if condition (line 186)
        if_condition_191795 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 8), skipped_191794)
        # Assigning a type to the variable 'if_condition_191795' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'if_condition_191795', if_condition_191795)
        # SSA begins for if statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 187)
        # Processing the call arguments (line 187)
        str_191798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 25), 'str', 'skipped=%d')
        # Getting the type of 'skipped' (line 187)
        skipped_191799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 40), 'skipped', False)
        # Applying the binary operator '%' (line 187)
        result_mod_191800 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 25), '%', str_191798, skipped_191799)
        
        # Processing the call keyword arguments (line 187)
        kwargs_191801 = {}
        # Getting the type of 'infos' (line 187)
        infos_191796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'infos', False)
        # Obtaining the member 'append' of a type (line 187)
        append_191797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 12), infos_191796, 'append')
        # Calling append(args, kwargs) (line 187)
        append_call_result_191802 = invoke(stypy.reporting.localization.Localization(__file__, 187, 12), append_191797, *[result_mod_191800], **kwargs_191801)
        
        # SSA join for if statement (line 186)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'expectedFails' (line 188)
        expectedFails_191803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'expectedFails')
        # Testing the type of an if condition (line 188)
        if_condition_191804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 8), expectedFails_191803)
        # Assigning a type to the variable 'if_condition_191804' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'if_condition_191804', if_condition_191804)
        # SSA begins for if statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 189)
        # Processing the call arguments (line 189)
        str_191807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 25), 'str', 'expected failures=%d')
        # Getting the type of 'expectedFails' (line 189)
        expectedFails_191808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 50), 'expectedFails', False)
        # Applying the binary operator '%' (line 189)
        result_mod_191809 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 25), '%', str_191807, expectedFails_191808)
        
        # Processing the call keyword arguments (line 189)
        kwargs_191810 = {}
        # Getting the type of 'infos' (line 189)
        infos_191805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'infos', False)
        # Obtaining the member 'append' of a type (line 189)
        append_191806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), infos_191805, 'append')
        # Calling append(args, kwargs) (line 189)
        append_call_result_191811 = invoke(stypy.reporting.localization.Localization(__file__, 189, 12), append_191806, *[result_mod_191809], **kwargs_191810)
        
        # SSA join for if statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'unexpectedSuccesses' (line 190)
        unexpectedSuccesses_191812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'unexpectedSuccesses')
        # Testing the type of an if condition (line 190)
        if_condition_191813 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 8), unexpectedSuccesses_191812)
        # Assigning a type to the variable 'if_condition_191813' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'if_condition_191813', if_condition_191813)
        # SSA begins for if statement (line 190)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 191)
        # Processing the call arguments (line 191)
        str_191816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 25), 'str', 'unexpected successes=%d')
        # Getting the type of 'unexpectedSuccesses' (line 191)
        unexpectedSuccesses_191817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 53), 'unexpectedSuccesses', False)
        # Applying the binary operator '%' (line 191)
        result_mod_191818 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 25), '%', str_191816, unexpectedSuccesses_191817)
        
        # Processing the call keyword arguments (line 191)
        kwargs_191819 = {}
        # Getting the type of 'infos' (line 191)
        infos_191814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'infos', False)
        # Obtaining the member 'append' of a type (line 191)
        append_191815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), infos_191814, 'append')
        # Calling append(args, kwargs) (line 191)
        append_call_result_191820 = invoke(stypy.reporting.localization.Localization(__file__, 191, 12), append_191815, *[result_mod_191818], **kwargs_191819)
        
        # SSA join for if statement (line 190)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'infos' (line 192)
        infos_191821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), 'infos')
        # Testing the type of an if condition (line 192)
        if_condition_191822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 8), infos_191821)
        # Assigning a type to the variable 'if_condition_191822' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'if_condition_191822', if_condition_191822)
        # SSA begins for if statement (line 192)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to writeln(...): (line 193)
        # Processing the call arguments (line 193)
        str_191826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 32), 'str', ' (%s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 193)
        tuple_191827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 193)
        # Adding element type (line 193)
        
        # Call to join(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'infos' (line 193)
        infos_191830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 53), 'infos', False)
        # Processing the call keyword arguments (line 193)
        kwargs_191831 = {}
        str_191828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 43), 'str', ', ')
        # Obtaining the member 'join' of a type (line 193)
        join_191829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 43), str_191828, 'join')
        # Calling join(args, kwargs) (line 193)
        join_call_result_191832 = invoke(stypy.reporting.localization.Localization(__file__, 193, 43), join_191829, *[infos_191830], **kwargs_191831)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 43), tuple_191827, join_call_result_191832)
        
        # Applying the binary operator '%' (line 193)
        result_mod_191833 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 32), '%', str_191826, tuple_191827)
        
        # Processing the call keyword arguments (line 193)
        kwargs_191834 = {}
        # Getting the type of 'self' (line 193)
        self_191823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 193)
        stream_191824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), self_191823, 'stream')
        # Obtaining the member 'writeln' of a type (line 193)
        writeln_191825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), stream_191824, 'writeln')
        # Calling writeln(args, kwargs) (line 193)
        writeln_call_result_191835 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), writeln_191825, *[result_mod_191833], **kwargs_191834)
        
        # SSA branch for the else part of an if statement (line 192)
        module_type_store.open_ssa_branch('else')
        
        # Call to write(...): (line 195)
        # Processing the call arguments (line 195)
        str_191839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 30), 'str', '\n')
        # Processing the call keyword arguments (line 195)
        kwargs_191840 = {}
        # Getting the type of 'self' (line 195)
        self_191836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'self', False)
        # Obtaining the member 'stream' of a type (line 195)
        stream_191837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), self_191836, 'stream')
        # Obtaining the member 'write' of a type (line 195)
        write_191838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), stream_191837, 'write')
        # Calling write(args, kwargs) (line 195)
        write_call_result_191841 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), write_191838, *[str_191839], **kwargs_191840)
        
        # SSA join for if statement (line 192)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 196)
        result_191842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'stypy_return_type', result_191842)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_191843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191843)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_191843


# Assigning a type to the variable 'TextTestRunner' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'TextTestRunner', TextTestRunner)

# Assigning a Name to a Name (line 125):
# Getting the type of 'TextTestResult' (line 125)
TextTestResult_191844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 18), 'TextTestResult')
# Getting the type of 'TextTestRunner'
TextTestRunner_191845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TextTestRunner')
# Setting the type of the member 'resultclass' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TextTestRunner_191845, 'resultclass', TextTestResult_191844)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
