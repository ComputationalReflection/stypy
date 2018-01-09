
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Unittest main program'''
2: 
3: import sys
4: import os
5: import types
6: 
7: from . import loader, runner
8: from .signals import installHandler
9: 
10: __unittest = True
11: 
12: FAILFAST     = "  -f, --failfast   Stop on first failure\n"
13: CATCHBREAK   = "  -c, --catch      Catch control-C and display results\n"
14: BUFFEROUTPUT = "  -b, --buffer     Buffer stdout and stderr during test runs\n"
15: 
16: USAGE_AS_MAIN = '''\
17: Usage: %(progName)s [options] [tests]
18: 
19: Options:
20:   -h, --help       Show this message
21:   -v, --verbose    Verbose output
22:   -q, --quiet      Minimal output
23: %(failfast)s%(catchbreak)s%(buffer)s
24: Examples:
25:   %(progName)s test_module               - run tests from test_module
26:   %(progName)s module.TestClass          - run tests from module.TestClass
27:   %(progName)s module.Class.test_method  - run specified test method
28: 
29: [tests] can be a list of any number of test modules, classes and test
30: methods.
31: 
32: Alternative Usage: %(progName)s discover [options]
33: 
34: Options:
35:   -v, --verbose    Verbose output
36: %(failfast)s%(catchbreak)s%(buffer)s  -s directory     Directory to start discovery ('.' default)
37:   -p pattern       Pattern to match test files ('test*.py' default)
38:   -t directory     Top level directory of project (default to
39:                    start directory)
40: 
41: For test discovery all test modules must be importable from the top
42: level directory of the project.
43: '''
44: 
45: USAGE_FROM_MODULE = '''\
46: Usage: %(progName)s [options] [test] [...]
47: 
48: Options:
49:   -h, --help       Show this message
50:   -v, --verbose    Verbose output
51:   -q, --quiet      Minimal output
52: %(failfast)s%(catchbreak)s%(buffer)s
53: Examples:
54:   %(progName)s                               - run default set of tests
55:   %(progName)s MyTestSuite                   - run suite 'MyTestSuite'
56:   %(progName)s MyTestCase.testSomething      - run MyTestCase.testSomething
57:   %(progName)s MyTestCase                    - run all 'test*' test methods
58:                                                in MyTestCase
59: '''
60: 
61: 
62: 
63: class TestProgram(object):
64:     '''A command-line program that runs a set of tests; this is primarily
65:        for making test modules conveniently executable.
66:     '''
67:     USAGE = USAGE_FROM_MODULE
68: 
69:     # defaults for testing
70:     failfast = catchbreak = buffer = progName = None
71: 
72:     def __init__(self, module='__main__', defaultTest=None, argv=None,
73:                     testRunner=None, testLoader=loader.defaultTestLoader,
74:                     exit=True, verbosity=1, failfast=None, catchbreak=None,
75:                     buffer=None):
76:         if isinstance(module, basestring):
77:             self.module = __import__(module)
78:             for part in module.split('.')[1:]:
79:                 self.module = getattr(self.module, part)
80:         else:
81:             self.module = module
82:         if argv is None:
83:             argv = sys.argv
84: 
85:         self.exit = exit
86:         self.failfast = failfast
87:         self.catchbreak = catchbreak
88:         self.verbosity = verbosity
89:         self.buffer = buffer
90:         self.defaultTest = defaultTest
91:         self.testRunner = testRunner
92:         self.testLoader = testLoader
93:         self.progName = os.path.basename(argv[0])
94:         self.parseArgs(argv)
95:         self.runTests()
96: 
97:     def usageExit(self, msg=None):
98:         if msg:
99:             print msg
100:         usage = {'progName': self.progName, 'catchbreak': '', 'failfast': '',
101:                  'buffer': ''}
102:         if self.failfast != False:
103:             usage['failfast'] = FAILFAST
104:         if self.catchbreak != False:
105:             usage['catchbreak'] = CATCHBREAK
106:         if self.buffer != False:
107:             usage['buffer'] = BUFFEROUTPUT
108:         print self.USAGE % usage
109:         sys.exit(2)
110: 
111:     def parseArgs(self, argv):
112:         if len(argv) > 1 and argv[1].lower() == 'discover':
113:             self._do_discovery(argv[2:])
114:             return
115: 
116:         import getopt
117:         long_opts = ['help', 'verbose', 'quiet', 'failfast', 'catch', 'buffer']
118:         try:
119:             options, args = getopt.getopt(argv[1:], 'hHvqfcb', long_opts)
120:             for opt, value in options:
121:                 if opt in ('-h','-H','--help'):
122:                     self.usageExit()
123:                 if opt in ('-q','--quiet'):
124:                     self.verbosity = 0
125:                 if opt in ('-v','--verbose'):
126:                     self.verbosity = 2
127:                 if opt in ('-f','--failfast'):
128:                     if self.failfast is None:
129:                         self.failfast = True
130:                     # Should this raise an exception if -f is not valid?
131:                 if opt in ('-c','--catch'):
132:                     if self.catchbreak is None:
133:                         self.catchbreak = True
134:                     # Should this raise an exception if -c is not valid?
135:                 if opt in ('-b','--buffer'):
136:                     if self.buffer is None:
137:                         self.buffer = True
138:                     # Should this raise an exception if -b is not valid?
139:             if len(args) == 0 and self.defaultTest is None:
140:                 # createTests will load tests from self.module
141:                 self.testNames = None
142:             elif len(args) > 0:
143:                 self.testNames = args
144:                 if __name__ == '__main__':
145:                     # to support python -m unittest ...
146:                     self.module = None
147:             else:
148:                 self.testNames = (self.defaultTest,)
149:             self.createTests()
150:         except getopt.error, msg:
151:             self.usageExit(msg)
152: 
153:     def createTests(self):
154:         if self.testNames is None:
155:             self.test = self.testLoader.loadTestsFromModule(self.module)
156:         else:
157:             self.test = self.testLoader.loadTestsFromNames(self.testNames,
158:                                                            self.module)
159: 
160:     def _do_discovery(self, argv, Loader=None):
161:         if Loader is None:
162:             Loader = lambda: self.testLoader
163: 
164:         # handle command line args for test discovery
165:         self.progName = '%s discover' % self.progName
166:         import optparse
167:         parser = optparse.OptionParser()
168:         parser.prog = self.progName
169:         parser.add_option('-v', '--verbose', dest='verbose', default=False,
170:                           help='Verbose output', action='store_true')
171:         if self.failfast != False:
172:             parser.add_option('-f', '--failfast', dest='failfast', default=False,
173:                               help='Stop on first fail or error',
174:                               action='store_true')
175:         if self.catchbreak != False:
176:             parser.add_option('-c', '--catch', dest='catchbreak', default=False,
177:                               help='Catch ctrl-C and display results so far',
178:                               action='store_true')
179:         if self.buffer != False:
180:             parser.add_option('-b', '--buffer', dest='buffer', default=False,
181:                               help='Buffer stdout and stderr during tests',
182:                               action='store_true')
183:         parser.add_option('-s', '--start-directory', dest='start', default='.',
184:                           help="Directory to start discovery ('.' default)")
185:         parser.add_option('-p', '--pattern', dest='pattern', default='test*.py',
186:                           help="Pattern to match tests ('test*.py' default)")
187:         parser.add_option('-t', '--top-level-directory', dest='top', default=None,
188:                           help='Top level directory of project (defaults to start directory)')
189: 
190:         options, args = parser.parse_args(argv)
191:         if len(args) > 3:
192:             self.usageExit()
193: 
194:         for name, value in zip(('start', 'pattern', 'top'), args):
195:             setattr(options, name, value)
196: 
197:         # only set options from the parsing here
198:         # if they weren't set explicitly in the constructor
199:         if self.failfast is None:
200:             self.failfast = options.failfast
201:         if self.catchbreak is None:
202:             self.catchbreak = options.catchbreak
203:         if self.buffer is None:
204:             self.buffer = options.buffer
205: 
206:         if options.verbose:
207:             self.verbosity = 2
208: 
209:         start_dir = options.start
210:         pattern = options.pattern
211:         top_level_dir = options.top
212: 
213:         loader = Loader()
214:         self.test = loader.discover(start_dir, pattern, top_level_dir)
215: 
216:     def runTests(self):
217:         if self.catchbreak:
218:             installHandler()
219:         if self.testRunner is None:
220:             self.testRunner = runner.TextTestRunner
221:         if isinstance(self.testRunner, (type, types.ClassType)):
222:             try:
223:                 testRunner = self.testRunner(verbosity=self.verbosity,
224:                                              failfast=self.failfast,
225:                                              buffer=self.buffer)
226:             except TypeError:
227:                 # didn't accept the verbosity, buffer or failfast arguments
228:                 testRunner = self.testRunner()
229:         else:
230:             # it is assumed to be a TestRunner instance
231:             testRunner = self.testRunner
232:         self.result = testRunner.run(self.test)
233:         if self.exit:
234:             sys.exit(not self.result.wasSuccessful())
235: 
236: main = TestProgram
237: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_190102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Unittest main program')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import types' statement (line 5)
import types

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from unittest import loader, runner' statement (line 7)
from unittest import loader, runner

import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'unittest', None, module_type_store, ['loader', 'runner'], [loader, runner])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from unittest.signals import installHandler' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/unittest/')
import_190103 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'unittest.signals')

if (type(import_190103) is not StypyTypeError):

    if (import_190103 != 'pyd_module'):
        __import__(import_190103)
        sys_modules_190104 = sys.modules[import_190103]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'unittest.signals', sys_modules_190104.module_type_store, module_type_store, ['installHandler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_190104, sys_modules_190104.module_type_store, module_type_store)
    else:
        from unittest.signals import installHandler

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'unittest.signals', None, module_type_store, ['installHandler'], [installHandler])

else:
    # Assigning a type to the variable 'unittest.signals' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'unittest.signals', import_190103)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/')


# Assigning a Name to a Name (line 10):

# Assigning a Name to a Name (line 10):
# Getting the type of 'True' (line 10)
True_190105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 13), 'True')
# Assigning a type to the variable '__unittest' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '__unittest', True_190105)

# Assigning a Str to a Name (line 12):

# Assigning a Str to a Name (line 12):
str_190106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'str', '  -f, --failfast   Stop on first failure\n')
# Assigning a type to the variable 'FAILFAST' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'FAILFAST', str_190106)

# Assigning a Str to a Name (line 13):

# Assigning a Str to a Name (line 13):
str_190107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', '  -c, --catch      Catch control-C and display results\n')
# Assigning a type to the variable 'CATCHBREAK' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'CATCHBREAK', str_190107)

# Assigning a Str to a Name (line 14):

# Assigning a Str to a Name (line 14):
str_190108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'str', '  -b, --buffer     Buffer stdout and stderr during test runs\n')
# Assigning a type to the variable 'BUFFEROUTPUT' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'BUFFEROUTPUT', str_190108)

# Assigning a Str to a Name (line 16):

# Assigning a Str to a Name (line 16):
str_190109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', "Usage: %(progName)s [options] [tests]\n\nOptions:\n  -h, --help       Show this message\n  -v, --verbose    Verbose output\n  -q, --quiet      Minimal output\n%(failfast)s%(catchbreak)s%(buffer)s\nExamples:\n  %(progName)s test_module               - run tests from test_module\n  %(progName)s module.TestClass          - run tests from module.TestClass\n  %(progName)s module.Class.test_method  - run specified test method\n\n[tests] can be a list of any number of test modules, classes and test\nmethods.\n\nAlternative Usage: %(progName)s discover [options]\n\nOptions:\n  -v, --verbose    Verbose output\n%(failfast)s%(catchbreak)s%(buffer)s  -s directory     Directory to start discovery ('.' default)\n  -p pattern       Pattern to match test files ('test*.py' default)\n  -t directory     Top level directory of project (default to\n                   start directory)\n\nFor test discovery all test modules must be importable from the top\nlevel directory of the project.\n")
# Assigning a type to the variable 'USAGE_AS_MAIN' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'USAGE_AS_MAIN', str_190109)

# Assigning a Str to a Name (line 45):

# Assigning a Str to a Name (line 45):
str_190110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', "Usage: %(progName)s [options] [test] [...]\n\nOptions:\n  -h, --help       Show this message\n  -v, --verbose    Verbose output\n  -q, --quiet      Minimal output\n%(failfast)s%(catchbreak)s%(buffer)s\nExamples:\n  %(progName)s                               - run default set of tests\n  %(progName)s MyTestSuite                   - run suite 'MyTestSuite'\n  %(progName)s MyTestCase.testSomething      - run MyTestCase.testSomething\n  %(progName)s MyTestCase                    - run all 'test*' test methods\n                                               in MyTestCase\n")
# Assigning a type to the variable 'USAGE_FROM_MODULE' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'USAGE_FROM_MODULE', str_190110)
# Declaration of the 'TestProgram' class

class TestProgram(object, ):
    str_190111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, (-1)), 'str', 'A command-line program that runs a set of tests; this is primarily\n       for making test modules conveniently executable.\n    ')
    
    # Assigning a Name to a Name (line 67):
    
    # Multiple assignment of 4 elements.

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_190112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 30), 'str', '__main__')
        # Getting the type of 'None' (line 72)
        None_190113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 54), 'None')
        # Getting the type of 'None' (line 72)
        None_190114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 65), 'None')
        # Getting the type of 'None' (line 73)
        None_190115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 31), 'None')
        # Getting the type of 'loader' (line 73)
        loader_190116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 48), 'loader')
        # Obtaining the member 'defaultTestLoader' of a type (line 73)
        defaultTestLoader_190117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 48), loader_190116, 'defaultTestLoader')
        # Getting the type of 'True' (line 74)
        True_190118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'True')
        int_190119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 41), 'int')
        # Getting the type of 'None' (line 74)
        None_190120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 53), 'None')
        # Getting the type of 'None' (line 74)
        None_190121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 70), 'None')
        # Getting the type of 'None' (line 75)
        None_190122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 27), 'None')
        defaults = [str_190112, None_190113, None_190114, None_190115, defaultTestLoader_190117, True_190118, int_190119, None_190120, None_190121, None_190122]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProgram.__init__', ['module', 'defaultTest', 'argv', 'testRunner', 'testLoader', 'exit', 'verbosity', 'failfast', 'catchbreak', 'buffer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['module', 'defaultTest', 'argv', 'testRunner', 'testLoader', 'exit', 'verbosity', 'failfast', 'catchbreak', 'buffer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 76)
        # Getting the type of 'basestring' (line 76)
        basestring_190123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), 'basestring')
        # Getting the type of 'module' (line 76)
        module_190124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 22), 'module')
        
        (may_be_190125, more_types_in_union_190126) = may_be_subtype(basestring_190123, module_190124)

        if may_be_190125:

            if more_types_in_union_190126:
                # Runtime conditional SSA (line 76)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'module' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'module', remove_not_subtype_from_union(module_190124, basestring))
            
            # Assigning a Call to a Attribute (line 77):
            
            # Assigning a Call to a Attribute (line 77):
            
            # Call to __import__(...): (line 77)
            # Processing the call arguments (line 77)
            # Getting the type of 'module' (line 77)
            module_190128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 37), 'module', False)
            # Processing the call keyword arguments (line 77)
            kwargs_190129 = {}
            # Getting the type of '__import__' (line 77)
            import___190127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), '__import__', False)
            # Calling __import__(args, kwargs) (line 77)
            import___call_result_190130 = invoke(stypy.reporting.localization.Localization(__file__, 77, 26), import___190127, *[module_190128], **kwargs_190129)
            
            # Getting the type of 'self' (line 77)
            self_190131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'self')
            # Setting the type of the member 'module' of a type (line 77)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), self_190131, 'module', import___call_result_190130)
            
            
            # Obtaining the type of the subscript
            int_190132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 42), 'int')
            slice_190133 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 78, 24), int_190132, None, None)
            
            # Call to split(...): (line 78)
            # Processing the call arguments (line 78)
            str_190136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 37), 'str', '.')
            # Processing the call keyword arguments (line 78)
            kwargs_190137 = {}
            # Getting the type of 'module' (line 78)
            module_190134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'module', False)
            # Obtaining the member 'split' of a type (line 78)
            split_190135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 24), module_190134, 'split')
            # Calling split(args, kwargs) (line 78)
            split_call_result_190138 = invoke(stypy.reporting.localization.Localization(__file__, 78, 24), split_190135, *[str_190136], **kwargs_190137)
            
            # Obtaining the member '__getitem__' of a type (line 78)
            getitem___190139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 24), split_call_result_190138, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 78)
            subscript_call_result_190140 = invoke(stypy.reporting.localization.Localization(__file__, 78, 24), getitem___190139, slice_190133)
            
            # Testing the type of a for loop iterable (line 78)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 12), subscript_call_result_190140)
            # Getting the type of the for loop variable (line 78)
            for_loop_var_190141 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 12), subscript_call_result_190140)
            # Assigning a type to the variable 'part' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'part', for_loop_var_190141)
            # SSA begins for a for statement (line 78)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Attribute (line 79):
            
            # Assigning a Call to a Attribute (line 79):
            
            # Call to getattr(...): (line 79)
            # Processing the call arguments (line 79)
            # Getting the type of 'self' (line 79)
            self_190143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 38), 'self', False)
            # Obtaining the member 'module' of a type (line 79)
            module_190144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 38), self_190143, 'module')
            # Getting the type of 'part' (line 79)
            part_190145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 51), 'part', False)
            # Processing the call keyword arguments (line 79)
            kwargs_190146 = {}
            # Getting the type of 'getattr' (line 79)
            getattr_190142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'getattr', False)
            # Calling getattr(args, kwargs) (line 79)
            getattr_call_result_190147 = invoke(stypy.reporting.localization.Localization(__file__, 79, 30), getattr_190142, *[module_190144, part_190145], **kwargs_190146)
            
            # Getting the type of 'self' (line 79)
            self_190148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'self')
            # Setting the type of the member 'module' of a type (line 79)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 16), self_190148, 'module', getattr_call_result_190147)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_190126:
                # Runtime conditional SSA for else branch (line 76)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_190125) or more_types_in_union_190126):
            # Assigning a type to the variable 'module' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'module', remove_subtype_from_union(module_190124, basestring))
            
            # Assigning a Name to a Attribute (line 81):
            
            # Assigning a Name to a Attribute (line 81):
            # Getting the type of 'module' (line 81)
            module_190149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'module')
            # Getting the type of 'self' (line 81)
            self_190150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'self')
            # Setting the type of the member 'module' of a type (line 81)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), self_190150, 'module', module_190149)

            if (may_be_190125 and more_types_in_union_190126):
                # SSA join for if statement (line 76)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 82)
        # Getting the type of 'argv' (line 82)
        argv_190151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'argv')
        # Getting the type of 'None' (line 82)
        None_190152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'None')
        
        (may_be_190153, more_types_in_union_190154) = may_be_none(argv_190151, None_190152)

        if may_be_190153:

            if more_types_in_union_190154:
                # Runtime conditional SSA (line 82)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 83):
            
            # Assigning a Attribute to a Name (line 83):
            # Getting the type of 'sys' (line 83)
            sys_190155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'sys')
            # Obtaining the member 'argv' of a type (line 83)
            argv_190156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 19), sys_190155, 'argv')
            # Assigning a type to the variable 'argv' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'argv', argv_190156)

            if more_types_in_union_190154:
                # SSA join for if statement (line 82)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 85):
        
        # Assigning a Name to a Attribute (line 85):
        # Getting the type of 'exit' (line 85)
        exit_190157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'exit')
        # Getting the type of 'self' (line 85)
        self_190158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self')
        # Setting the type of the member 'exit' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_190158, 'exit', exit_190157)
        
        # Assigning a Name to a Attribute (line 86):
        
        # Assigning a Name to a Attribute (line 86):
        # Getting the type of 'failfast' (line 86)
        failfast_190159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'failfast')
        # Getting the type of 'self' (line 86)
        self_190160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'self')
        # Setting the type of the member 'failfast' of a type (line 86)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), self_190160, 'failfast', failfast_190159)
        
        # Assigning a Name to a Attribute (line 87):
        
        # Assigning a Name to a Attribute (line 87):
        # Getting the type of 'catchbreak' (line 87)
        catchbreak_190161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 26), 'catchbreak')
        # Getting the type of 'self' (line 87)
        self_190162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self')
        # Setting the type of the member 'catchbreak' of a type (line 87)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_190162, 'catchbreak', catchbreak_190161)
        
        # Assigning a Name to a Attribute (line 88):
        
        # Assigning a Name to a Attribute (line 88):
        # Getting the type of 'verbosity' (line 88)
        verbosity_190163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 25), 'verbosity')
        # Getting the type of 'self' (line 88)
        self_190164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self')
        # Setting the type of the member 'verbosity' of a type (line 88)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_190164, 'verbosity', verbosity_190163)
        
        # Assigning a Name to a Attribute (line 89):
        
        # Assigning a Name to a Attribute (line 89):
        # Getting the type of 'buffer' (line 89)
        buffer_190165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'buffer')
        # Getting the type of 'self' (line 89)
        self_190166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self')
        # Setting the type of the member 'buffer' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_190166, 'buffer', buffer_190165)
        
        # Assigning a Name to a Attribute (line 90):
        
        # Assigning a Name to a Attribute (line 90):
        # Getting the type of 'defaultTest' (line 90)
        defaultTest_190167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 27), 'defaultTest')
        # Getting the type of 'self' (line 90)
        self_190168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Setting the type of the member 'defaultTest' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_190168, 'defaultTest', defaultTest_190167)
        
        # Assigning a Name to a Attribute (line 91):
        
        # Assigning a Name to a Attribute (line 91):
        # Getting the type of 'testRunner' (line 91)
        testRunner_190169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'testRunner')
        # Getting the type of 'self' (line 91)
        self_190170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member 'testRunner' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_190170, 'testRunner', testRunner_190169)
        
        # Assigning a Name to a Attribute (line 92):
        
        # Assigning a Name to a Attribute (line 92):
        # Getting the type of 'testLoader' (line 92)
        testLoader_190171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 26), 'testLoader')
        # Getting the type of 'self' (line 92)
        self_190172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self')
        # Setting the type of the member 'testLoader' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_190172, 'testLoader', testLoader_190171)
        
        # Assigning a Call to a Attribute (line 93):
        
        # Assigning a Call to a Attribute (line 93):
        
        # Call to basename(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Obtaining the type of the subscript
        int_190176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 46), 'int')
        # Getting the type of 'argv' (line 93)
        argv_190177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 41), 'argv', False)
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___190178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 41), argv_190177, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_190179 = invoke(stypy.reporting.localization.Localization(__file__, 93, 41), getitem___190178, int_190176)
        
        # Processing the call keyword arguments (line 93)
        kwargs_190180 = {}
        # Getting the type of 'os' (line 93)
        os_190173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 93)
        path_190174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 24), os_190173, 'path')
        # Obtaining the member 'basename' of a type (line 93)
        basename_190175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 24), path_190174, 'basename')
        # Calling basename(args, kwargs) (line 93)
        basename_call_result_190181 = invoke(stypy.reporting.localization.Localization(__file__, 93, 24), basename_190175, *[subscript_call_result_190179], **kwargs_190180)
        
        # Getting the type of 'self' (line 93)
        self_190182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'self')
        # Setting the type of the member 'progName' of a type (line 93)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), self_190182, 'progName', basename_call_result_190181)
        
        # Call to parseArgs(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'argv' (line 94)
        argv_190185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'argv', False)
        # Processing the call keyword arguments (line 94)
        kwargs_190186 = {}
        # Getting the type of 'self' (line 94)
        self_190183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self', False)
        # Obtaining the member 'parseArgs' of a type (line 94)
        parseArgs_190184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_190183, 'parseArgs')
        # Calling parseArgs(args, kwargs) (line 94)
        parseArgs_call_result_190187 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), parseArgs_190184, *[argv_190185], **kwargs_190186)
        
        
        # Call to runTests(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_190190 = {}
        # Getting the type of 'self' (line 95)
        self_190188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self', False)
        # Obtaining the member 'runTests' of a type (line 95)
        runTests_190189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_190188, 'runTests')
        # Calling runTests(args, kwargs) (line 95)
        runTests_call_result_190191 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), runTests_190189, *[], **kwargs_190190)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def usageExit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 97)
        None_190192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'None')
        defaults = [None_190192]
        # Create a new context for function 'usageExit'
        module_type_store = module_type_store.open_function_context('usageExit', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProgram.usageExit.__dict__.__setitem__('stypy_localization', localization)
        TestProgram.usageExit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProgram.usageExit.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProgram.usageExit.__dict__.__setitem__('stypy_function_name', 'TestProgram.usageExit')
        TestProgram.usageExit.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        TestProgram.usageExit.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProgram.usageExit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProgram.usageExit.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProgram.usageExit.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProgram.usageExit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProgram.usageExit.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProgram.usageExit', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'usageExit', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'usageExit(...)' code ##################

        
        # Getting the type of 'msg' (line 98)
        msg_190193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'msg')
        # Testing the type of an if condition (line 98)
        if_condition_190194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 8), msg_190193)
        # Assigning a type to the variable 'if_condition_190194' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'if_condition_190194', if_condition_190194)
        # SSA begins for if statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'msg' (line 99)
        msg_190195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'msg')
        # SSA join for if statement (line 98)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Dict to a Name (line 100):
        
        # Assigning a Dict to a Name (line 100):
        
        # Obtaining an instance of the builtin type 'dict' (line 100)
        dict_190196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 100)
        # Adding element type (key, value) (line 100)
        str_190197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 17), 'str', 'progName')
        # Getting the type of 'self' (line 100)
        self_190198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 29), 'self')
        # Obtaining the member 'progName' of a type (line 100)
        progName_190199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 29), self_190198, 'progName')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 16), dict_190196, (str_190197, progName_190199))
        # Adding element type (key, value) (line 100)
        str_190200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 44), 'str', 'catchbreak')
        str_190201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 58), 'str', '')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 16), dict_190196, (str_190200, str_190201))
        # Adding element type (key, value) (line 100)
        str_190202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 62), 'str', 'failfast')
        str_190203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 74), 'str', '')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 16), dict_190196, (str_190202, str_190203))
        # Adding element type (key, value) (line 100)
        str_190204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'str', 'buffer')
        str_190205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 27), 'str', '')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 16), dict_190196, (str_190204, str_190205))
        
        # Assigning a type to the variable 'usage' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'usage', dict_190196)
        
        
        # Getting the type of 'self' (line 102)
        self_190206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'self')
        # Obtaining the member 'failfast' of a type (line 102)
        failfast_190207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 11), self_190206, 'failfast')
        # Getting the type of 'False' (line 102)
        False_190208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'False')
        # Applying the binary operator '!=' (line 102)
        result_ne_190209 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), '!=', failfast_190207, False_190208)
        
        # Testing the type of an if condition (line 102)
        if_condition_190210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), result_ne_190209)
        # Assigning a type to the variable 'if_condition_190210' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'if_condition_190210', if_condition_190210)
        # SSA begins for if statement (line 102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 103):
        
        # Assigning a Name to a Subscript (line 103):
        # Getting the type of 'FAILFAST' (line 103)
        FAILFAST_190211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 32), 'FAILFAST')
        # Getting the type of 'usage' (line 103)
        usage_190212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'usage')
        str_190213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 18), 'str', 'failfast')
        # Storing an element on a container (line 103)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 12), usage_190212, (str_190213, FAILFAST_190211))
        # SSA join for if statement (line 102)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 104)
        self_190214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'self')
        # Obtaining the member 'catchbreak' of a type (line 104)
        catchbreak_190215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), self_190214, 'catchbreak')
        # Getting the type of 'False' (line 104)
        False_190216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'False')
        # Applying the binary operator '!=' (line 104)
        result_ne_190217 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 11), '!=', catchbreak_190215, False_190216)
        
        # Testing the type of an if condition (line 104)
        if_condition_190218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 8), result_ne_190217)
        # Assigning a type to the variable 'if_condition_190218' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'if_condition_190218', if_condition_190218)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 105):
        
        # Assigning a Name to a Subscript (line 105):
        # Getting the type of 'CATCHBREAK' (line 105)
        CATCHBREAK_190219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'CATCHBREAK')
        # Getting the type of 'usage' (line 105)
        usage_190220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'usage')
        str_190221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 18), 'str', 'catchbreak')
        # Storing an element on a container (line 105)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), usage_190220, (str_190221, CATCHBREAK_190219))
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 106)
        self_190222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'self')
        # Obtaining the member 'buffer' of a type (line 106)
        buffer_190223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 11), self_190222, 'buffer')
        # Getting the type of 'False' (line 106)
        False_190224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'False')
        # Applying the binary operator '!=' (line 106)
        result_ne_190225 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 11), '!=', buffer_190223, False_190224)
        
        # Testing the type of an if condition (line 106)
        if_condition_190226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), result_ne_190225)
        # Assigning a type to the variable 'if_condition_190226' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_190226', if_condition_190226)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 107):
        
        # Assigning a Name to a Subscript (line 107):
        # Getting the type of 'BUFFEROUTPUT' (line 107)
        BUFFEROUTPUT_190227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), 'BUFFEROUTPUT')
        # Getting the type of 'usage' (line 107)
        usage_190228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'usage')
        str_190229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 18), 'str', 'buffer')
        # Storing an element on a container (line 107)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), usage_190228, (str_190229, BUFFEROUTPUT_190227))
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 108)
        self_190230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 'self')
        # Obtaining the member 'USAGE' of a type (line 108)
        USAGE_190231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), self_190230, 'USAGE')
        # Getting the type of 'usage' (line 108)
        usage_190232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'usage')
        # Applying the binary operator '%' (line 108)
        result_mod_190233 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), '%', USAGE_190231, usage_190232)
        
        
        # Call to exit(...): (line 109)
        # Processing the call arguments (line 109)
        int_190236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 17), 'int')
        # Processing the call keyword arguments (line 109)
        kwargs_190237 = {}
        # Getting the type of 'sys' (line 109)
        sys_190234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'sys', False)
        # Obtaining the member 'exit' of a type (line 109)
        exit_190235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), sys_190234, 'exit')
        # Calling exit(args, kwargs) (line 109)
        exit_call_result_190238 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), exit_190235, *[int_190236], **kwargs_190237)
        
        
        # ################# End of 'usageExit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'usageExit' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_190239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190239)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'usageExit'
        return stypy_return_type_190239


    @norecursion
    def parseArgs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'parseArgs'
        module_type_store = module_type_store.open_function_context('parseArgs', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProgram.parseArgs.__dict__.__setitem__('stypy_localization', localization)
        TestProgram.parseArgs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProgram.parseArgs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProgram.parseArgs.__dict__.__setitem__('stypy_function_name', 'TestProgram.parseArgs')
        TestProgram.parseArgs.__dict__.__setitem__('stypy_param_names_list', ['argv'])
        TestProgram.parseArgs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProgram.parseArgs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProgram.parseArgs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProgram.parseArgs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProgram.parseArgs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProgram.parseArgs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProgram.parseArgs', ['argv'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'parseArgs', localization, ['argv'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'parseArgs(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'argv' (line 112)
        argv_190241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'argv', False)
        # Processing the call keyword arguments (line 112)
        kwargs_190242 = {}
        # Getting the type of 'len' (line 112)
        len_190240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'len', False)
        # Calling len(args, kwargs) (line 112)
        len_call_result_190243 = invoke(stypy.reporting.localization.Localization(__file__, 112, 11), len_190240, *[argv_190241], **kwargs_190242)
        
        int_190244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 23), 'int')
        # Applying the binary operator '>' (line 112)
        result_gt_190245 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), '>', len_call_result_190243, int_190244)
        
        
        
        # Call to lower(...): (line 112)
        # Processing the call keyword arguments (line 112)
        kwargs_190251 = {}
        
        # Obtaining the type of the subscript
        int_190246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 34), 'int')
        # Getting the type of 'argv' (line 112)
        argv_190247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 29), 'argv', False)
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___190248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 29), argv_190247, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_190249 = invoke(stypy.reporting.localization.Localization(__file__, 112, 29), getitem___190248, int_190246)
        
        # Obtaining the member 'lower' of a type (line 112)
        lower_190250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 29), subscript_call_result_190249, 'lower')
        # Calling lower(args, kwargs) (line 112)
        lower_call_result_190252 = invoke(stypy.reporting.localization.Localization(__file__, 112, 29), lower_190250, *[], **kwargs_190251)
        
        str_190253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 48), 'str', 'discover')
        # Applying the binary operator '==' (line 112)
        result_eq_190254 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 29), '==', lower_call_result_190252, str_190253)
        
        # Applying the binary operator 'and' (line 112)
        result_and_keyword_190255 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), 'and', result_gt_190245, result_eq_190254)
        
        # Testing the type of an if condition (line 112)
        if_condition_190256 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 8), result_and_keyword_190255)
        # Assigning a type to the variable 'if_condition_190256' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'if_condition_190256', if_condition_190256)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _do_discovery(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Obtaining the type of the subscript
        int_190259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 36), 'int')
        slice_190260 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 113, 31), int_190259, None, None)
        # Getting the type of 'argv' (line 113)
        argv_190261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'argv', False)
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___190262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 31), argv_190261, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
        subscript_call_result_190263 = invoke(stypy.reporting.localization.Localization(__file__, 113, 31), getitem___190262, slice_190260)
        
        # Processing the call keyword arguments (line 113)
        kwargs_190264 = {}
        # Getting the type of 'self' (line 113)
        self_190257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'self', False)
        # Obtaining the member '_do_discovery' of a type (line 113)
        _do_discovery_190258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), self_190257, '_do_discovery')
        # Calling _do_discovery(args, kwargs) (line 113)
        _do_discovery_call_result_190265 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), _do_discovery_190258, *[subscript_call_result_190263], **kwargs_190264)
        
        # Assigning a type to the variable 'stypy_return_type' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 116, 8))
        
        # 'import getopt' statement (line 116)
        import getopt

        import_module(stypy.reporting.localization.Localization(__file__, 116, 8), 'getopt', getopt, module_type_store)
        
        
        # Assigning a List to a Name (line 117):
        
        # Assigning a List to a Name (line 117):
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_190266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        str_190267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 21), 'str', 'help')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 20), list_190266, str_190267)
        # Adding element type (line 117)
        str_190268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 29), 'str', 'verbose')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 20), list_190266, str_190268)
        # Adding element type (line 117)
        str_190269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 40), 'str', 'quiet')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 20), list_190266, str_190269)
        # Adding element type (line 117)
        str_190270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 49), 'str', 'failfast')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 20), list_190266, str_190270)
        # Adding element type (line 117)
        str_190271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 61), 'str', 'catch')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 20), list_190266, str_190271)
        # Adding element type (line 117)
        str_190272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 70), 'str', 'buffer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 20), list_190266, str_190272)
        
        # Assigning a type to the variable 'long_opts' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'long_opts', list_190266)
        
        
        # SSA begins for try-except statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Tuple (line 119):
        
        # Assigning a Call to a Name:
        
        # Call to getopt(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Obtaining the type of the subscript
        int_190275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 47), 'int')
        slice_190276 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 119, 42), int_190275, None, None)
        # Getting the type of 'argv' (line 119)
        argv_190277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 42), 'argv', False)
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___190278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 42), argv_190277, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_190279 = invoke(stypy.reporting.localization.Localization(__file__, 119, 42), getitem___190278, slice_190276)
        
        str_190280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 52), 'str', 'hHvqfcb')
        # Getting the type of 'long_opts' (line 119)
        long_opts_190281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 63), 'long_opts', False)
        # Processing the call keyword arguments (line 119)
        kwargs_190282 = {}
        # Getting the type of 'getopt' (line 119)
        getopt_190273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 28), 'getopt', False)
        # Obtaining the member 'getopt' of a type (line 119)
        getopt_190274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 28), getopt_190273, 'getopt')
        # Calling getopt(args, kwargs) (line 119)
        getopt_call_result_190283 = invoke(stypy.reporting.localization.Localization(__file__, 119, 28), getopt_190274, *[subscript_call_result_190279, str_190280, long_opts_190281], **kwargs_190282)
        
        # Assigning a type to the variable 'call_assignment_190096' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'call_assignment_190096', getopt_call_result_190283)
        
        # Assigning a Call to a Name (line 119):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_190286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 12), 'int')
        # Processing the call keyword arguments
        kwargs_190287 = {}
        # Getting the type of 'call_assignment_190096' (line 119)
        call_assignment_190096_190284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'call_assignment_190096', False)
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___190285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), call_assignment_190096_190284, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_190288 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___190285, *[int_190286], **kwargs_190287)
        
        # Assigning a type to the variable 'call_assignment_190097' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'call_assignment_190097', getitem___call_result_190288)
        
        # Assigning a Name to a Name (line 119):
        # Getting the type of 'call_assignment_190097' (line 119)
        call_assignment_190097_190289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'call_assignment_190097')
        # Assigning a type to the variable 'options' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'options', call_assignment_190097_190289)
        
        # Assigning a Call to a Name (line 119):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_190292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 12), 'int')
        # Processing the call keyword arguments
        kwargs_190293 = {}
        # Getting the type of 'call_assignment_190096' (line 119)
        call_assignment_190096_190290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'call_assignment_190096', False)
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___190291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), call_assignment_190096_190290, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_190294 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___190291, *[int_190292], **kwargs_190293)
        
        # Assigning a type to the variable 'call_assignment_190098' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'call_assignment_190098', getitem___call_result_190294)
        
        # Assigning a Name to a Name (line 119):
        # Getting the type of 'call_assignment_190098' (line 119)
        call_assignment_190098_190295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'call_assignment_190098')
        # Assigning a type to the variable 'args' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'args', call_assignment_190098_190295)
        
        # Getting the type of 'options' (line 120)
        options_190296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 30), 'options')
        # Testing the type of a for loop iterable (line 120)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 120, 12), options_190296)
        # Getting the type of the for loop variable (line 120)
        for_loop_var_190297 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 120, 12), options_190296)
        # Assigning a type to the variable 'opt' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'opt', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 12), for_loop_var_190297))
        # Assigning a type to the variable 'value' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 12), for_loop_var_190297))
        # SSA begins for a for statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'opt' (line 121)
        opt_190298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'opt')
        
        # Obtaining an instance of the builtin type 'tuple' (line 121)
        tuple_190299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 121)
        # Adding element type (line 121)
        str_190300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 27), 'str', '-h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 27), tuple_190299, str_190300)
        # Adding element type (line 121)
        str_190301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 32), 'str', '-H')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 27), tuple_190299, str_190301)
        # Adding element type (line 121)
        str_190302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 37), 'str', '--help')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 27), tuple_190299, str_190302)
        
        # Applying the binary operator 'in' (line 121)
        result_contains_190303 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 19), 'in', opt_190298, tuple_190299)
        
        # Testing the type of an if condition (line 121)
        if_condition_190304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 16), result_contains_190303)
        # Assigning a type to the variable 'if_condition_190304' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'if_condition_190304', if_condition_190304)
        # SSA begins for if statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to usageExit(...): (line 122)
        # Processing the call keyword arguments (line 122)
        kwargs_190307 = {}
        # Getting the type of 'self' (line 122)
        self_190305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'self', False)
        # Obtaining the member 'usageExit' of a type (line 122)
        usageExit_190306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 20), self_190305, 'usageExit')
        # Calling usageExit(args, kwargs) (line 122)
        usageExit_call_result_190308 = invoke(stypy.reporting.localization.Localization(__file__, 122, 20), usageExit_190306, *[], **kwargs_190307)
        
        # SSA join for if statement (line 121)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'opt' (line 123)
        opt_190309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 19), 'opt')
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_190310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        str_190311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 27), 'str', '-q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 27), tuple_190310, str_190311)
        # Adding element type (line 123)
        str_190312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 32), 'str', '--quiet')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 27), tuple_190310, str_190312)
        
        # Applying the binary operator 'in' (line 123)
        result_contains_190313 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 19), 'in', opt_190309, tuple_190310)
        
        # Testing the type of an if condition (line 123)
        if_condition_190314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 16), result_contains_190313)
        # Assigning a type to the variable 'if_condition_190314' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'if_condition_190314', if_condition_190314)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 124):
        
        # Assigning a Num to a Attribute (line 124):
        int_190315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 37), 'int')
        # Getting the type of 'self' (line 124)
        self_190316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'self')
        # Setting the type of the member 'verbosity' of a type (line 124)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 20), self_190316, 'verbosity', int_190315)
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'opt' (line 125)
        opt_190317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'opt')
        
        # Obtaining an instance of the builtin type 'tuple' (line 125)
        tuple_190318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 125)
        # Adding element type (line 125)
        str_190319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 27), 'str', '-v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 27), tuple_190318, str_190319)
        # Adding element type (line 125)
        str_190320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 32), 'str', '--verbose')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 27), tuple_190318, str_190320)
        
        # Applying the binary operator 'in' (line 125)
        result_contains_190321 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 19), 'in', opt_190317, tuple_190318)
        
        # Testing the type of an if condition (line 125)
        if_condition_190322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 16), result_contains_190321)
        # Assigning a type to the variable 'if_condition_190322' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'if_condition_190322', if_condition_190322)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 126):
        
        # Assigning a Num to a Attribute (line 126):
        int_190323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 37), 'int')
        # Getting the type of 'self' (line 126)
        self_190324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'self')
        # Setting the type of the member 'verbosity' of a type (line 126)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 20), self_190324, 'verbosity', int_190323)
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'opt' (line 127)
        opt_190325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'opt')
        
        # Obtaining an instance of the builtin type 'tuple' (line 127)
        tuple_190326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 127)
        # Adding element type (line 127)
        str_190327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 27), 'str', '-f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 27), tuple_190326, str_190327)
        # Adding element type (line 127)
        str_190328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 32), 'str', '--failfast')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 27), tuple_190326, str_190328)
        
        # Applying the binary operator 'in' (line 127)
        result_contains_190329 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 19), 'in', opt_190325, tuple_190326)
        
        # Testing the type of an if condition (line 127)
        if_condition_190330 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 16), result_contains_190329)
        # Assigning a type to the variable 'if_condition_190330' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'if_condition_190330', if_condition_190330)
        # SSA begins for if statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 128)
        # Getting the type of 'self' (line 128)
        self_190331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'self')
        # Obtaining the member 'failfast' of a type (line 128)
        failfast_190332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 23), self_190331, 'failfast')
        # Getting the type of 'None' (line 128)
        None_190333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 40), 'None')
        
        (may_be_190334, more_types_in_union_190335) = may_be_none(failfast_190332, None_190333)

        if may_be_190334:

            if more_types_in_union_190335:
                # Runtime conditional SSA (line 128)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 129):
            
            # Assigning a Name to a Attribute (line 129):
            # Getting the type of 'True' (line 129)
            True_190336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'True')
            # Getting the type of 'self' (line 129)
            self_190337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 24), 'self')
            # Setting the type of the member 'failfast' of a type (line 129)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 24), self_190337, 'failfast', True_190336)

            if more_types_in_union_190335:
                # SSA join for if statement (line 128)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 127)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'opt' (line 131)
        opt_190338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'opt')
        
        # Obtaining an instance of the builtin type 'tuple' (line 131)
        tuple_190339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 131)
        # Adding element type (line 131)
        str_190340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 27), 'str', '-c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 27), tuple_190339, str_190340)
        # Adding element type (line 131)
        str_190341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 32), 'str', '--catch')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 27), tuple_190339, str_190341)
        
        # Applying the binary operator 'in' (line 131)
        result_contains_190342 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 19), 'in', opt_190338, tuple_190339)
        
        # Testing the type of an if condition (line 131)
        if_condition_190343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 16), result_contains_190342)
        # Assigning a type to the variable 'if_condition_190343' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'if_condition_190343', if_condition_190343)
        # SSA begins for if statement (line 131)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 132)
        # Getting the type of 'self' (line 132)
        self_190344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 23), 'self')
        # Obtaining the member 'catchbreak' of a type (line 132)
        catchbreak_190345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 23), self_190344, 'catchbreak')
        # Getting the type of 'None' (line 132)
        None_190346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 42), 'None')
        
        (may_be_190347, more_types_in_union_190348) = may_be_none(catchbreak_190345, None_190346)

        if may_be_190347:

            if more_types_in_union_190348:
                # Runtime conditional SSA (line 132)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 133):
            
            # Assigning a Name to a Attribute (line 133):
            # Getting the type of 'True' (line 133)
            True_190349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 42), 'True')
            # Getting the type of 'self' (line 133)
            self_190350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'self')
            # Setting the type of the member 'catchbreak' of a type (line 133)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 24), self_190350, 'catchbreak', True_190349)

            if more_types_in_union_190348:
                # SSA join for if statement (line 132)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 131)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'opt' (line 135)
        opt_190351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'opt')
        
        # Obtaining an instance of the builtin type 'tuple' (line 135)
        tuple_190352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 135)
        # Adding element type (line 135)
        str_190353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 27), 'str', '-b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 27), tuple_190352, str_190353)
        # Adding element type (line 135)
        str_190354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 32), 'str', '--buffer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 27), tuple_190352, str_190354)
        
        # Applying the binary operator 'in' (line 135)
        result_contains_190355 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 19), 'in', opt_190351, tuple_190352)
        
        # Testing the type of an if condition (line 135)
        if_condition_190356 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 16), result_contains_190355)
        # Assigning a type to the variable 'if_condition_190356' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'if_condition_190356', if_condition_190356)
        # SSA begins for if statement (line 135)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 136)
        # Getting the type of 'self' (line 136)
        self_190357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'self')
        # Obtaining the member 'buffer' of a type (line 136)
        buffer_190358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 23), self_190357, 'buffer')
        # Getting the type of 'None' (line 136)
        None_190359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 38), 'None')
        
        (may_be_190360, more_types_in_union_190361) = may_be_none(buffer_190358, None_190359)

        if may_be_190360:

            if more_types_in_union_190361:
                # Runtime conditional SSA (line 136)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 137):
            
            # Assigning a Name to a Attribute (line 137):
            # Getting the type of 'True' (line 137)
            True_190362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 38), 'True')
            # Getting the type of 'self' (line 137)
            self_190363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 24), 'self')
            # Setting the type of the member 'buffer' of a type (line 137)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 24), self_190363, 'buffer', True_190362)

            if more_types_in_union_190361:
                # SSA join for if statement (line 136)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 135)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'args' (line 139)
        args_190365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'args', False)
        # Processing the call keyword arguments (line 139)
        kwargs_190366 = {}
        # Getting the type of 'len' (line 139)
        len_190364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'len', False)
        # Calling len(args, kwargs) (line 139)
        len_call_result_190367 = invoke(stypy.reporting.localization.Localization(__file__, 139, 15), len_190364, *[args_190365], **kwargs_190366)
        
        int_190368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 28), 'int')
        # Applying the binary operator '==' (line 139)
        result_eq_190369 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 15), '==', len_call_result_190367, int_190368)
        
        
        # Getting the type of 'self' (line 139)
        self_190370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 34), 'self')
        # Obtaining the member 'defaultTest' of a type (line 139)
        defaultTest_190371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 34), self_190370, 'defaultTest')
        # Getting the type of 'None' (line 139)
        None_190372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 54), 'None')
        # Applying the binary operator 'is' (line 139)
        result_is__190373 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 34), 'is', defaultTest_190371, None_190372)
        
        # Applying the binary operator 'and' (line 139)
        result_and_keyword_190374 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 15), 'and', result_eq_190369, result_is__190373)
        
        # Testing the type of an if condition (line 139)
        if_condition_190375 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 12), result_and_keyword_190374)
        # Assigning a type to the variable 'if_condition_190375' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'if_condition_190375', if_condition_190375)
        # SSA begins for if statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 141):
        
        # Assigning a Name to a Attribute (line 141):
        # Getting the type of 'None' (line 141)
        None_190376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), 'None')
        # Getting the type of 'self' (line 141)
        self_190377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'self')
        # Setting the type of the member 'testNames' of a type (line 141)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 16), self_190377, 'testNames', None_190376)
        # SSA branch for the else part of an if statement (line 139)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'args' (line 142)
        args_190379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'args', False)
        # Processing the call keyword arguments (line 142)
        kwargs_190380 = {}
        # Getting the type of 'len' (line 142)
        len_190378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 17), 'len', False)
        # Calling len(args, kwargs) (line 142)
        len_call_result_190381 = invoke(stypy.reporting.localization.Localization(__file__, 142, 17), len_190378, *[args_190379], **kwargs_190380)
        
        int_190382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 29), 'int')
        # Applying the binary operator '>' (line 142)
        result_gt_190383 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 17), '>', len_call_result_190381, int_190382)
        
        # Testing the type of an if condition (line 142)
        if_condition_190384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 17), result_gt_190383)
        # Assigning a type to the variable 'if_condition_190384' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 17), 'if_condition_190384', if_condition_190384)
        # SSA begins for if statement (line 142)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 143):
        
        # Assigning a Name to a Attribute (line 143):
        # Getting the type of 'args' (line 143)
        args_190385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 33), 'args')
        # Getting the type of 'self' (line 143)
        self_190386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'self')
        # Setting the type of the member 'testNames' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), self_190386, 'testNames', args_190385)

        if (__name__ == '__main__'):
            
            # Assigning a Name to a Attribute (line 146):
            
            # Assigning a Name to a Attribute (line 146):
            # Getting the type of 'None' (line 146)
            None_190387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'None')
            # Getting the type of 'self' (line 146)
            self_190388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'self')
            # Setting the type of the member 'module' of a type (line 146)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 20), self_190388, 'module', None_190387)

        # SSA branch for the else part of an if statement (line 142)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Tuple to a Attribute (line 148):
        
        # Assigning a Tuple to a Attribute (line 148):
        
        # Obtaining an instance of the builtin type 'tuple' (line 148)
        tuple_190389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 148)
        # Adding element type (line 148)
        # Getting the type of 'self' (line 148)
        self_190390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 34), 'self')
        # Obtaining the member 'defaultTest' of a type (line 148)
        defaultTest_190391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 34), self_190390, 'defaultTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 34), tuple_190389, defaultTest_190391)
        
        # Getting the type of 'self' (line 148)
        self_190392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'self')
        # Setting the type of the member 'testNames' of a type (line 148)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 16), self_190392, 'testNames', tuple_190389)
        # SSA join for if statement (line 142)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 139)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to createTests(...): (line 149)
        # Processing the call keyword arguments (line 149)
        kwargs_190395 = {}
        # Getting the type of 'self' (line 149)
        self_190393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'self', False)
        # Obtaining the member 'createTests' of a type (line 149)
        createTests_190394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), self_190393, 'createTests')
        # Calling createTests(args, kwargs) (line 149)
        createTests_call_result_190396 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), createTests_190394, *[], **kwargs_190395)
        
        # SSA branch for the except part of a try statement (line 118)
        # SSA branch for the except 'Attribute' branch of a try statement (line 118)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'getopt' (line 150)
        getopt_190397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'getopt')
        # Obtaining the member 'error' of a type (line 150)
        error_190398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 15), getopt_190397, 'error')
        # Assigning a type to the variable 'msg' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'msg', error_190398)
        
        # Call to usageExit(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'msg' (line 151)
        msg_190401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 27), 'msg', False)
        # Processing the call keyword arguments (line 151)
        kwargs_190402 = {}
        # Getting the type of 'self' (line 151)
        self_190399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'self', False)
        # Obtaining the member 'usageExit' of a type (line 151)
        usageExit_190400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), self_190399, 'usageExit')
        # Calling usageExit(args, kwargs) (line 151)
        usageExit_call_result_190403 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), usageExit_190400, *[msg_190401], **kwargs_190402)
        
        # SSA join for try-except statement (line 118)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'parseArgs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'parseArgs' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_190404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190404)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'parseArgs'
        return stypy_return_type_190404


    @norecursion
    def createTests(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'createTests'
        module_type_store = module_type_store.open_function_context('createTests', 153, 4, False)
        # Assigning a type to the variable 'self' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProgram.createTests.__dict__.__setitem__('stypy_localization', localization)
        TestProgram.createTests.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProgram.createTests.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProgram.createTests.__dict__.__setitem__('stypy_function_name', 'TestProgram.createTests')
        TestProgram.createTests.__dict__.__setitem__('stypy_param_names_list', [])
        TestProgram.createTests.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProgram.createTests.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProgram.createTests.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProgram.createTests.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProgram.createTests.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProgram.createTests.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProgram.createTests', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'createTests', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'createTests(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 154)
        # Getting the type of 'self' (line 154)
        self_190405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), 'self')
        # Obtaining the member 'testNames' of a type (line 154)
        testNames_190406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 11), self_190405, 'testNames')
        # Getting the type of 'None' (line 154)
        None_190407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'None')
        
        (may_be_190408, more_types_in_union_190409) = may_be_none(testNames_190406, None_190407)

        if may_be_190408:

            if more_types_in_union_190409:
                # Runtime conditional SSA (line 154)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 155):
            
            # Assigning a Call to a Attribute (line 155):
            
            # Call to loadTestsFromModule(...): (line 155)
            # Processing the call arguments (line 155)
            # Getting the type of 'self' (line 155)
            self_190413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 60), 'self', False)
            # Obtaining the member 'module' of a type (line 155)
            module_190414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 60), self_190413, 'module')
            # Processing the call keyword arguments (line 155)
            kwargs_190415 = {}
            # Getting the type of 'self' (line 155)
            self_190410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'self', False)
            # Obtaining the member 'testLoader' of a type (line 155)
            testLoader_190411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 24), self_190410, 'testLoader')
            # Obtaining the member 'loadTestsFromModule' of a type (line 155)
            loadTestsFromModule_190412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 24), testLoader_190411, 'loadTestsFromModule')
            # Calling loadTestsFromModule(args, kwargs) (line 155)
            loadTestsFromModule_call_result_190416 = invoke(stypy.reporting.localization.Localization(__file__, 155, 24), loadTestsFromModule_190412, *[module_190414], **kwargs_190415)
            
            # Getting the type of 'self' (line 155)
            self_190417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'self')
            # Setting the type of the member 'test' of a type (line 155)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), self_190417, 'test', loadTestsFromModule_call_result_190416)

            if more_types_in_union_190409:
                # Runtime conditional SSA for else branch (line 154)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_190408) or more_types_in_union_190409):
            
            # Assigning a Call to a Attribute (line 157):
            
            # Assigning a Call to a Attribute (line 157):
            
            # Call to loadTestsFromNames(...): (line 157)
            # Processing the call arguments (line 157)
            # Getting the type of 'self' (line 157)
            self_190421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 59), 'self', False)
            # Obtaining the member 'testNames' of a type (line 157)
            testNames_190422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 59), self_190421, 'testNames')
            # Getting the type of 'self' (line 158)
            self_190423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 59), 'self', False)
            # Obtaining the member 'module' of a type (line 158)
            module_190424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 59), self_190423, 'module')
            # Processing the call keyword arguments (line 157)
            kwargs_190425 = {}
            # Getting the type of 'self' (line 157)
            self_190418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'self', False)
            # Obtaining the member 'testLoader' of a type (line 157)
            testLoader_190419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 24), self_190418, 'testLoader')
            # Obtaining the member 'loadTestsFromNames' of a type (line 157)
            loadTestsFromNames_190420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 24), testLoader_190419, 'loadTestsFromNames')
            # Calling loadTestsFromNames(args, kwargs) (line 157)
            loadTestsFromNames_call_result_190426 = invoke(stypy.reporting.localization.Localization(__file__, 157, 24), loadTestsFromNames_190420, *[testNames_190422, module_190424], **kwargs_190425)
            
            # Getting the type of 'self' (line 157)
            self_190427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'self')
            # Setting the type of the member 'test' of a type (line 157)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), self_190427, 'test', loadTestsFromNames_call_result_190426)

            if (may_be_190408 and more_types_in_union_190409):
                # SSA join for if statement (line 154)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'createTests(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'createTests' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_190428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190428)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'createTests'
        return stypy_return_type_190428


    @norecursion
    def _do_discovery(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 160)
        None_190429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 41), 'None')
        defaults = [None_190429]
        # Create a new context for function '_do_discovery'
        module_type_store = module_type_store.open_function_context('_do_discovery', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProgram._do_discovery.__dict__.__setitem__('stypy_localization', localization)
        TestProgram._do_discovery.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProgram._do_discovery.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProgram._do_discovery.__dict__.__setitem__('stypy_function_name', 'TestProgram._do_discovery')
        TestProgram._do_discovery.__dict__.__setitem__('stypy_param_names_list', ['argv', 'Loader'])
        TestProgram._do_discovery.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProgram._do_discovery.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProgram._do_discovery.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProgram._do_discovery.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProgram._do_discovery.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProgram._do_discovery.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProgram._do_discovery', ['argv', 'Loader'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_do_discovery', localization, ['argv', 'Loader'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_do_discovery(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 161)
        # Getting the type of 'Loader' (line 161)
        Loader_190430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'Loader')
        # Getting the type of 'None' (line 161)
        None_190431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 21), 'None')
        
        (may_be_190432, more_types_in_union_190433) = may_be_none(Loader_190430, None_190431)

        if may_be_190432:

            if more_types_in_union_190433:
                # Runtime conditional SSA (line 161)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Lambda to a Name (line 162):
            
            # Assigning a Lambda to a Name (line 162):

            @norecursion
            def _stypy_temp_lambda_52(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_52'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_52', 162, 21, True)
                # Passed parameters checking function
                _stypy_temp_lambda_52.stypy_localization = localization
                _stypy_temp_lambda_52.stypy_type_of_self = None
                _stypy_temp_lambda_52.stypy_type_store = module_type_store
                _stypy_temp_lambda_52.stypy_function_name = '_stypy_temp_lambda_52'
                _stypy_temp_lambda_52.stypy_param_names_list = []
                _stypy_temp_lambda_52.stypy_varargs_param_name = None
                _stypy_temp_lambda_52.stypy_kwargs_param_name = None
                _stypy_temp_lambda_52.stypy_call_defaults = defaults
                _stypy_temp_lambda_52.stypy_call_varargs = varargs
                _stypy_temp_lambda_52.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_52', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_52', [], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                # Getting the type of 'self' (line 162)
                self_190434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'self')
                # Obtaining the member 'testLoader' of a type (line 162)
                testLoader_190435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 29), self_190434, 'testLoader')
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 162)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), 'stypy_return_type', testLoader_190435)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_52' in the type store
                # Getting the type of 'stypy_return_type' (line 162)
                stypy_return_type_190436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_190436)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_52'
                return stypy_return_type_190436

            # Assigning a type to the variable '_stypy_temp_lambda_52' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), '_stypy_temp_lambda_52', _stypy_temp_lambda_52)
            # Getting the type of '_stypy_temp_lambda_52' (line 162)
            _stypy_temp_lambda_52_190437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), '_stypy_temp_lambda_52')
            # Assigning a type to the variable 'Loader' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'Loader', _stypy_temp_lambda_52_190437)

            if more_types_in_union_190433:
                # SSA join for if statement (line 161)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Attribute (line 165):
        
        # Assigning a BinOp to a Attribute (line 165):
        str_190438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 24), 'str', '%s discover')
        # Getting the type of 'self' (line 165)
        self_190439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 40), 'self')
        # Obtaining the member 'progName' of a type (line 165)
        progName_190440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 40), self_190439, 'progName')
        # Applying the binary operator '%' (line 165)
        result_mod_190441 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 24), '%', str_190438, progName_190440)
        
        # Getting the type of 'self' (line 165)
        self_190442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'self')
        # Setting the type of the member 'progName' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), self_190442, 'progName', result_mod_190441)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 166, 8))
        
        # 'import optparse' statement (line 166)
        import optparse

        import_module(stypy.reporting.localization.Localization(__file__, 166, 8), 'optparse', optparse, module_type_store)
        
        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to OptionParser(...): (line 167)
        # Processing the call keyword arguments (line 167)
        kwargs_190445 = {}
        # Getting the type of 'optparse' (line 167)
        optparse_190443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 17), 'optparse', False)
        # Obtaining the member 'OptionParser' of a type (line 167)
        OptionParser_190444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 17), optparse_190443, 'OptionParser')
        # Calling OptionParser(args, kwargs) (line 167)
        OptionParser_call_result_190446 = invoke(stypy.reporting.localization.Localization(__file__, 167, 17), OptionParser_190444, *[], **kwargs_190445)
        
        # Assigning a type to the variable 'parser' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'parser', OptionParser_call_result_190446)
        
        # Assigning a Attribute to a Attribute (line 168):
        
        # Assigning a Attribute to a Attribute (line 168):
        # Getting the type of 'self' (line 168)
        self_190447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'self')
        # Obtaining the member 'progName' of a type (line 168)
        progName_190448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 22), self_190447, 'progName')
        # Getting the type of 'parser' (line 168)
        parser_190449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'parser')
        # Setting the type of the member 'prog' of a type (line 168)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), parser_190449, 'prog', progName_190448)
        
        # Call to add_option(...): (line 169)
        # Processing the call arguments (line 169)
        str_190452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 26), 'str', '-v')
        str_190453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 32), 'str', '--verbose')
        # Processing the call keyword arguments (line 169)
        str_190454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 50), 'str', 'verbose')
        keyword_190455 = str_190454
        # Getting the type of 'False' (line 169)
        False_190456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 69), 'False', False)
        keyword_190457 = False_190456
        str_190458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 31), 'str', 'Verbose output')
        keyword_190459 = str_190458
        str_190460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 56), 'str', 'store_true')
        keyword_190461 = str_190460
        kwargs_190462 = {'dest': keyword_190455, 'default': keyword_190457, 'help': keyword_190459, 'action': keyword_190461}
        # Getting the type of 'parser' (line 169)
        parser_190450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'parser', False)
        # Obtaining the member 'add_option' of a type (line 169)
        add_option_190451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), parser_190450, 'add_option')
        # Calling add_option(args, kwargs) (line 169)
        add_option_call_result_190463 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), add_option_190451, *[str_190452, str_190453], **kwargs_190462)
        
        
        
        # Getting the type of 'self' (line 171)
        self_190464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'self')
        # Obtaining the member 'failfast' of a type (line 171)
        failfast_190465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 11), self_190464, 'failfast')
        # Getting the type of 'False' (line 171)
        False_190466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 28), 'False')
        # Applying the binary operator '!=' (line 171)
        result_ne_190467 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 11), '!=', failfast_190465, False_190466)
        
        # Testing the type of an if condition (line 171)
        if_condition_190468 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 8), result_ne_190467)
        # Assigning a type to the variable 'if_condition_190468' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'if_condition_190468', if_condition_190468)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add_option(...): (line 172)
        # Processing the call arguments (line 172)
        str_190471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 30), 'str', '-f')
        str_190472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 36), 'str', '--failfast')
        # Processing the call keyword arguments (line 172)
        str_190473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 55), 'str', 'failfast')
        keyword_190474 = str_190473
        # Getting the type of 'False' (line 172)
        False_190475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 75), 'False', False)
        keyword_190476 = False_190475
        str_190477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 35), 'str', 'Stop on first fail or error')
        keyword_190478 = str_190477
        str_190479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 37), 'str', 'store_true')
        keyword_190480 = str_190479
        kwargs_190481 = {'dest': keyword_190474, 'default': keyword_190476, 'help': keyword_190478, 'action': keyword_190480}
        # Getting the type of 'parser' (line 172)
        parser_190469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'parser', False)
        # Obtaining the member 'add_option' of a type (line 172)
        add_option_190470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), parser_190469, 'add_option')
        # Calling add_option(args, kwargs) (line 172)
        add_option_call_result_190482 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), add_option_190470, *[str_190471, str_190472], **kwargs_190481)
        
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 175)
        self_190483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 11), 'self')
        # Obtaining the member 'catchbreak' of a type (line 175)
        catchbreak_190484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 11), self_190483, 'catchbreak')
        # Getting the type of 'False' (line 175)
        False_190485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 30), 'False')
        # Applying the binary operator '!=' (line 175)
        result_ne_190486 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 11), '!=', catchbreak_190484, False_190485)
        
        # Testing the type of an if condition (line 175)
        if_condition_190487 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 8), result_ne_190486)
        # Assigning a type to the variable 'if_condition_190487' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'if_condition_190487', if_condition_190487)
        # SSA begins for if statement (line 175)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add_option(...): (line 176)
        # Processing the call arguments (line 176)
        str_190490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 30), 'str', '-c')
        str_190491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 36), 'str', '--catch')
        # Processing the call keyword arguments (line 176)
        str_190492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 52), 'str', 'catchbreak')
        keyword_190493 = str_190492
        # Getting the type of 'False' (line 176)
        False_190494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 74), 'False', False)
        keyword_190495 = False_190494
        str_190496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 35), 'str', 'Catch ctrl-C and display results so far')
        keyword_190497 = str_190496
        str_190498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 37), 'str', 'store_true')
        keyword_190499 = str_190498
        kwargs_190500 = {'dest': keyword_190493, 'default': keyword_190495, 'help': keyword_190497, 'action': keyword_190499}
        # Getting the type of 'parser' (line 176)
        parser_190488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'parser', False)
        # Obtaining the member 'add_option' of a type (line 176)
        add_option_190489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), parser_190488, 'add_option')
        # Calling add_option(args, kwargs) (line 176)
        add_option_call_result_190501 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), add_option_190489, *[str_190490, str_190491], **kwargs_190500)
        
        # SSA join for if statement (line 175)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 179)
        self_190502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 11), 'self')
        # Obtaining the member 'buffer' of a type (line 179)
        buffer_190503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 11), self_190502, 'buffer')
        # Getting the type of 'False' (line 179)
        False_190504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 26), 'False')
        # Applying the binary operator '!=' (line 179)
        result_ne_190505 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 11), '!=', buffer_190503, False_190504)
        
        # Testing the type of an if condition (line 179)
        if_condition_190506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 8), result_ne_190505)
        # Assigning a type to the variable 'if_condition_190506' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'if_condition_190506', if_condition_190506)
        # SSA begins for if statement (line 179)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add_option(...): (line 180)
        # Processing the call arguments (line 180)
        str_190509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 30), 'str', '-b')
        str_190510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 36), 'str', '--buffer')
        # Processing the call keyword arguments (line 180)
        str_190511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 53), 'str', 'buffer')
        keyword_190512 = str_190511
        # Getting the type of 'False' (line 180)
        False_190513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 71), 'False', False)
        keyword_190514 = False_190513
        str_190515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 35), 'str', 'Buffer stdout and stderr during tests')
        keyword_190516 = str_190515
        str_190517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 37), 'str', 'store_true')
        keyword_190518 = str_190517
        kwargs_190519 = {'dest': keyword_190512, 'default': keyword_190514, 'help': keyword_190516, 'action': keyword_190518}
        # Getting the type of 'parser' (line 180)
        parser_190507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'parser', False)
        # Obtaining the member 'add_option' of a type (line 180)
        add_option_190508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 12), parser_190507, 'add_option')
        # Calling add_option(args, kwargs) (line 180)
        add_option_call_result_190520 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), add_option_190508, *[str_190509, str_190510], **kwargs_190519)
        
        # SSA join for if statement (line 179)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to add_option(...): (line 183)
        # Processing the call arguments (line 183)
        str_190523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 26), 'str', '-s')
        str_190524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 32), 'str', '--start-directory')
        # Processing the call keyword arguments (line 183)
        str_190525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 58), 'str', 'start')
        keyword_190526 = str_190525
        str_190527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 75), 'str', '.')
        keyword_190528 = str_190527
        str_190529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 31), 'str', "Directory to start discovery ('.' default)")
        keyword_190530 = str_190529
        kwargs_190531 = {'dest': keyword_190526, 'default': keyword_190528, 'help': keyword_190530}
        # Getting the type of 'parser' (line 183)
        parser_190521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'parser', False)
        # Obtaining the member 'add_option' of a type (line 183)
        add_option_190522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), parser_190521, 'add_option')
        # Calling add_option(args, kwargs) (line 183)
        add_option_call_result_190532 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), add_option_190522, *[str_190523, str_190524], **kwargs_190531)
        
        
        # Call to add_option(...): (line 185)
        # Processing the call arguments (line 185)
        str_190535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 26), 'str', '-p')
        str_190536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 32), 'str', '--pattern')
        # Processing the call keyword arguments (line 185)
        str_190537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 50), 'str', 'pattern')
        keyword_190538 = str_190537
        str_190539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 69), 'str', 'test*.py')
        keyword_190540 = str_190539
        str_190541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 31), 'str', "Pattern to match tests ('test*.py' default)")
        keyword_190542 = str_190541
        kwargs_190543 = {'dest': keyword_190538, 'default': keyword_190540, 'help': keyword_190542}
        # Getting the type of 'parser' (line 185)
        parser_190533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'parser', False)
        # Obtaining the member 'add_option' of a type (line 185)
        add_option_190534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), parser_190533, 'add_option')
        # Calling add_option(args, kwargs) (line 185)
        add_option_call_result_190544 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), add_option_190534, *[str_190535, str_190536], **kwargs_190543)
        
        
        # Call to add_option(...): (line 187)
        # Processing the call arguments (line 187)
        str_190547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 26), 'str', '-t')
        str_190548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 32), 'str', '--top-level-directory')
        # Processing the call keyword arguments (line 187)
        str_190549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 62), 'str', 'top')
        keyword_190550 = str_190549
        # Getting the type of 'None' (line 187)
        None_190551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 77), 'None', False)
        keyword_190552 = None_190551
        str_190553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 31), 'str', 'Top level directory of project (defaults to start directory)')
        keyword_190554 = str_190553
        kwargs_190555 = {'dest': keyword_190550, 'default': keyword_190552, 'help': keyword_190554}
        # Getting the type of 'parser' (line 187)
        parser_190545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'parser', False)
        # Obtaining the member 'add_option' of a type (line 187)
        add_option_190546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), parser_190545, 'add_option')
        # Calling add_option(args, kwargs) (line 187)
        add_option_call_result_190556 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), add_option_190546, *[str_190547, str_190548], **kwargs_190555)
        
        
        # Assigning a Call to a Tuple (line 190):
        
        # Assigning a Call to a Name:
        
        # Call to parse_args(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'argv' (line 190)
        argv_190559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 42), 'argv', False)
        # Processing the call keyword arguments (line 190)
        kwargs_190560 = {}
        # Getting the type of 'parser' (line 190)
        parser_190557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 24), 'parser', False)
        # Obtaining the member 'parse_args' of a type (line 190)
        parse_args_190558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 24), parser_190557, 'parse_args')
        # Calling parse_args(args, kwargs) (line 190)
        parse_args_call_result_190561 = invoke(stypy.reporting.localization.Localization(__file__, 190, 24), parse_args_190558, *[argv_190559], **kwargs_190560)
        
        # Assigning a type to the variable 'call_assignment_190099' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_190099', parse_args_call_result_190561)
        
        # Assigning a Call to a Name (line 190):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_190564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 8), 'int')
        # Processing the call keyword arguments
        kwargs_190565 = {}
        # Getting the type of 'call_assignment_190099' (line 190)
        call_assignment_190099_190562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_190099', False)
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___190563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), call_assignment_190099_190562, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_190566 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___190563, *[int_190564], **kwargs_190565)
        
        # Assigning a type to the variable 'call_assignment_190100' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_190100', getitem___call_result_190566)
        
        # Assigning a Name to a Name (line 190):
        # Getting the type of 'call_assignment_190100' (line 190)
        call_assignment_190100_190567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_190100')
        # Assigning a type to the variable 'options' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'options', call_assignment_190100_190567)
        
        # Assigning a Call to a Name (line 190):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_190570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 8), 'int')
        # Processing the call keyword arguments
        kwargs_190571 = {}
        # Getting the type of 'call_assignment_190099' (line 190)
        call_assignment_190099_190568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_190099', False)
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___190569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), call_assignment_190099_190568, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_190572 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___190569, *[int_190570], **kwargs_190571)
        
        # Assigning a type to the variable 'call_assignment_190101' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_190101', getitem___call_result_190572)
        
        # Assigning a Name to a Name (line 190):
        # Getting the type of 'call_assignment_190101' (line 190)
        call_assignment_190101_190573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_190101')
        # Assigning a type to the variable 'args' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'args', call_assignment_190101_190573)
        
        
        
        # Call to len(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'args' (line 191)
        args_190575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'args', False)
        # Processing the call keyword arguments (line 191)
        kwargs_190576 = {}
        # Getting the type of 'len' (line 191)
        len_190574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'len', False)
        # Calling len(args, kwargs) (line 191)
        len_call_result_190577 = invoke(stypy.reporting.localization.Localization(__file__, 191, 11), len_190574, *[args_190575], **kwargs_190576)
        
        int_190578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 23), 'int')
        # Applying the binary operator '>' (line 191)
        result_gt_190579 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 11), '>', len_call_result_190577, int_190578)
        
        # Testing the type of an if condition (line 191)
        if_condition_190580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 8), result_gt_190579)
        # Assigning a type to the variable 'if_condition_190580' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'if_condition_190580', if_condition_190580)
        # SSA begins for if statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to usageExit(...): (line 192)
        # Processing the call keyword arguments (line 192)
        kwargs_190583 = {}
        # Getting the type of 'self' (line 192)
        self_190581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'self', False)
        # Obtaining the member 'usageExit' of a type (line 192)
        usageExit_190582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), self_190581, 'usageExit')
        # Calling usageExit(args, kwargs) (line 192)
        usageExit_call_result_190584 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), usageExit_190582, *[], **kwargs_190583)
        
        # SSA join for if statement (line 191)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to zip(...): (line 194)
        # Processing the call arguments (line 194)
        
        # Obtaining an instance of the builtin type 'tuple' (line 194)
        tuple_190586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 194)
        # Adding element type (line 194)
        str_190587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 32), 'str', 'start')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 32), tuple_190586, str_190587)
        # Adding element type (line 194)
        str_190588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 41), 'str', 'pattern')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 32), tuple_190586, str_190588)
        # Adding element type (line 194)
        str_190589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 52), 'str', 'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 32), tuple_190586, str_190589)
        
        # Getting the type of 'args' (line 194)
        args_190590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 60), 'args', False)
        # Processing the call keyword arguments (line 194)
        kwargs_190591 = {}
        # Getting the type of 'zip' (line 194)
        zip_190585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 27), 'zip', False)
        # Calling zip(args, kwargs) (line 194)
        zip_call_result_190592 = invoke(stypy.reporting.localization.Localization(__file__, 194, 27), zip_190585, *[tuple_190586, args_190590], **kwargs_190591)
        
        # Testing the type of a for loop iterable (line 194)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 194, 8), zip_call_result_190592)
        # Getting the type of the for loop variable (line 194)
        for_loop_var_190593 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 194, 8), zip_call_result_190592)
        # Assigning a type to the variable 'name' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 8), for_loop_var_190593))
        # Assigning a type to the variable 'value' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 8), for_loop_var_190593))
        # SSA begins for a for statement (line 194)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'options' (line 195)
        options_190595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'options', False)
        # Getting the type of 'name' (line 195)
        name_190596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 29), 'name', False)
        # Getting the type of 'value' (line 195)
        value_190597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 35), 'value', False)
        # Processing the call keyword arguments (line 195)
        kwargs_190598 = {}
        # Getting the type of 'setattr' (line 195)
        setattr_190594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 195)
        setattr_call_result_190599 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), setattr_190594, *[options_190595, name_190596, value_190597], **kwargs_190598)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 199)
        # Getting the type of 'self' (line 199)
        self_190600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'self')
        # Obtaining the member 'failfast' of a type (line 199)
        failfast_190601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 11), self_190600, 'failfast')
        # Getting the type of 'None' (line 199)
        None_190602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'None')
        
        (may_be_190603, more_types_in_union_190604) = may_be_none(failfast_190601, None_190602)

        if may_be_190603:

            if more_types_in_union_190604:
                # Runtime conditional SSA (line 199)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 200):
            
            # Assigning a Attribute to a Attribute (line 200):
            # Getting the type of 'options' (line 200)
            options_190605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 28), 'options')
            # Obtaining the member 'failfast' of a type (line 200)
            failfast_190606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 28), options_190605, 'failfast')
            # Getting the type of 'self' (line 200)
            self_190607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'self')
            # Setting the type of the member 'failfast' of a type (line 200)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 12), self_190607, 'failfast', failfast_190606)

            if more_types_in_union_190604:
                # SSA join for if statement (line 199)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 201)
        # Getting the type of 'self' (line 201)
        self_190608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'self')
        # Obtaining the member 'catchbreak' of a type (line 201)
        catchbreak_190609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 11), self_190608, 'catchbreak')
        # Getting the type of 'None' (line 201)
        None_190610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 30), 'None')
        
        (may_be_190611, more_types_in_union_190612) = may_be_none(catchbreak_190609, None_190610)

        if may_be_190611:

            if more_types_in_union_190612:
                # Runtime conditional SSA (line 201)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 202):
            
            # Assigning a Attribute to a Attribute (line 202):
            # Getting the type of 'options' (line 202)
            options_190613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 30), 'options')
            # Obtaining the member 'catchbreak' of a type (line 202)
            catchbreak_190614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 30), options_190613, 'catchbreak')
            # Getting the type of 'self' (line 202)
            self_190615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'self')
            # Setting the type of the member 'catchbreak' of a type (line 202)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 12), self_190615, 'catchbreak', catchbreak_190614)

            if more_types_in_union_190612:
                # SSA join for if statement (line 201)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 203)
        # Getting the type of 'self' (line 203)
        self_190616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'self')
        # Obtaining the member 'buffer' of a type (line 203)
        buffer_190617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 11), self_190616, 'buffer')
        # Getting the type of 'None' (line 203)
        None_190618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 26), 'None')
        
        (may_be_190619, more_types_in_union_190620) = may_be_none(buffer_190617, None_190618)

        if may_be_190619:

            if more_types_in_union_190620:
                # Runtime conditional SSA (line 203)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 204):
            
            # Assigning a Attribute to a Attribute (line 204):
            # Getting the type of 'options' (line 204)
            options_190621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 26), 'options')
            # Obtaining the member 'buffer' of a type (line 204)
            buffer_190622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 26), options_190621, 'buffer')
            # Getting the type of 'self' (line 204)
            self_190623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'self')
            # Setting the type of the member 'buffer' of a type (line 204)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), self_190623, 'buffer', buffer_190622)

            if more_types_in_union_190620:
                # SSA join for if statement (line 203)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'options' (line 206)
        options_190624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'options')
        # Obtaining the member 'verbose' of a type (line 206)
        verbose_190625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 11), options_190624, 'verbose')
        # Testing the type of an if condition (line 206)
        if_condition_190626 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 8), verbose_190625)
        # Assigning a type to the variable 'if_condition_190626' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'if_condition_190626', if_condition_190626)
        # SSA begins for if statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 207):
        
        # Assigning a Num to a Attribute (line 207):
        int_190627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 29), 'int')
        # Getting the type of 'self' (line 207)
        self_190628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'self')
        # Setting the type of the member 'verbosity' of a type (line 207)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), self_190628, 'verbosity', int_190627)
        # SSA join for if statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 209):
        
        # Assigning a Attribute to a Name (line 209):
        # Getting the type of 'options' (line 209)
        options_190629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'options')
        # Obtaining the member 'start' of a type (line 209)
        start_190630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 20), options_190629, 'start')
        # Assigning a type to the variable 'start_dir' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'start_dir', start_190630)
        
        # Assigning a Attribute to a Name (line 210):
        
        # Assigning a Attribute to a Name (line 210):
        # Getting the type of 'options' (line 210)
        options_190631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 18), 'options')
        # Obtaining the member 'pattern' of a type (line 210)
        pattern_190632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 18), options_190631, 'pattern')
        # Assigning a type to the variable 'pattern' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'pattern', pattern_190632)
        
        # Assigning a Attribute to a Name (line 211):
        
        # Assigning a Attribute to a Name (line 211):
        # Getting the type of 'options' (line 211)
        options_190633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 24), 'options')
        # Obtaining the member 'top' of a type (line 211)
        top_190634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 24), options_190633, 'top')
        # Assigning a type to the variable 'top_level_dir' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'top_level_dir', top_190634)
        
        # Assigning a Call to a Name (line 213):
        
        # Assigning a Call to a Name (line 213):
        
        # Call to Loader(...): (line 213)
        # Processing the call keyword arguments (line 213)
        kwargs_190636 = {}
        # Getting the type of 'Loader' (line 213)
        Loader_190635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 17), 'Loader', False)
        # Calling Loader(args, kwargs) (line 213)
        Loader_call_result_190637 = invoke(stypy.reporting.localization.Localization(__file__, 213, 17), Loader_190635, *[], **kwargs_190636)
        
        # Assigning a type to the variable 'loader' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'loader', Loader_call_result_190637)
        
        # Assigning a Call to a Attribute (line 214):
        
        # Assigning a Call to a Attribute (line 214):
        
        # Call to discover(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'start_dir' (line 214)
        start_dir_190640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 36), 'start_dir', False)
        # Getting the type of 'pattern' (line 214)
        pattern_190641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 47), 'pattern', False)
        # Getting the type of 'top_level_dir' (line 214)
        top_level_dir_190642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 56), 'top_level_dir', False)
        # Processing the call keyword arguments (line 214)
        kwargs_190643 = {}
        # Getting the type of 'loader' (line 214)
        loader_190638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'loader', False)
        # Obtaining the member 'discover' of a type (line 214)
        discover_190639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), loader_190638, 'discover')
        # Calling discover(args, kwargs) (line 214)
        discover_call_result_190644 = invoke(stypy.reporting.localization.Localization(__file__, 214, 20), discover_190639, *[start_dir_190640, pattern_190641, top_level_dir_190642], **kwargs_190643)
        
        # Getting the type of 'self' (line 214)
        self_190645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'self')
        # Setting the type of the member 'test' of a type (line 214)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), self_190645, 'test', discover_call_result_190644)
        
        # ################# End of '_do_discovery(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_do_discovery' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_190646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190646)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_do_discovery'
        return stypy_return_type_190646


    @norecursion
    def runTests(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runTests'
        module_type_store = module_type_store.open_function_context('runTests', 216, 4, False)
        # Assigning a type to the variable 'self' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestProgram.runTests.__dict__.__setitem__('stypy_localization', localization)
        TestProgram.runTests.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestProgram.runTests.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestProgram.runTests.__dict__.__setitem__('stypy_function_name', 'TestProgram.runTests')
        TestProgram.runTests.__dict__.__setitem__('stypy_param_names_list', [])
        TestProgram.runTests.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestProgram.runTests.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestProgram.runTests.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestProgram.runTests.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestProgram.runTests.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestProgram.runTests.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestProgram.runTests', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'runTests', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'runTests(...)' code ##################

        
        # Getting the type of 'self' (line 217)
        self_190647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'self')
        # Obtaining the member 'catchbreak' of a type (line 217)
        catchbreak_190648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 11), self_190647, 'catchbreak')
        # Testing the type of an if condition (line 217)
        if_condition_190649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 8), catchbreak_190648)
        # Assigning a type to the variable 'if_condition_190649' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'if_condition_190649', if_condition_190649)
        # SSA begins for if statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to installHandler(...): (line 218)
        # Processing the call keyword arguments (line 218)
        kwargs_190651 = {}
        # Getting the type of 'installHandler' (line 218)
        installHandler_190650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'installHandler', False)
        # Calling installHandler(args, kwargs) (line 218)
        installHandler_call_result_190652 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), installHandler_190650, *[], **kwargs_190651)
        
        # SSA join for if statement (line 217)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 219)
        # Getting the type of 'self' (line 219)
        self_190653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 11), 'self')
        # Obtaining the member 'testRunner' of a type (line 219)
        testRunner_190654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 11), self_190653, 'testRunner')
        # Getting the type of 'None' (line 219)
        None_190655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'None')
        
        (may_be_190656, more_types_in_union_190657) = may_be_none(testRunner_190654, None_190655)

        if may_be_190656:

            if more_types_in_union_190657:
                # Runtime conditional SSA (line 219)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 220):
            
            # Assigning a Attribute to a Attribute (line 220):
            # Getting the type of 'runner' (line 220)
            runner_190658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 30), 'runner')
            # Obtaining the member 'TextTestRunner' of a type (line 220)
            TextTestRunner_190659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 30), runner_190658, 'TextTestRunner')
            # Getting the type of 'self' (line 220)
            self_190660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'self')
            # Setting the type of the member 'testRunner' of a type (line 220)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), self_190660, 'testRunner', TextTestRunner_190659)

            if more_types_in_union_190657:
                # SSA join for if statement (line 219)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to isinstance(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'self' (line 221)
        self_190662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 22), 'self', False)
        # Obtaining the member 'testRunner' of a type (line 221)
        testRunner_190663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 22), self_190662, 'testRunner')
        
        # Obtaining an instance of the builtin type 'tuple' (line 221)
        tuple_190664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 221)
        # Adding element type (line 221)
        # Getting the type of 'type' (line 221)
        type_190665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 40), 'type', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 40), tuple_190664, type_190665)
        # Adding element type (line 221)
        # Getting the type of 'types' (line 221)
        types_190666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 46), 'types', False)
        # Obtaining the member 'ClassType' of a type (line 221)
        ClassType_190667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 46), types_190666, 'ClassType')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 40), tuple_190664, ClassType_190667)
        
        # Processing the call keyword arguments (line 221)
        kwargs_190668 = {}
        # Getting the type of 'isinstance' (line 221)
        isinstance_190661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 221)
        isinstance_call_result_190669 = invoke(stypy.reporting.localization.Localization(__file__, 221, 11), isinstance_190661, *[testRunner_190663, tuple_190664], **kwargs_190668)
        
        # Testing the type of an if condition (line 221)
        if_condition_190670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 8), isinstance_call_result_190669)
        # Assigning a type to the variable 'if_condition_190670' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'if_condition_190670', if_condition_190670)
        # SSA begins for if statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 222)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to testRunner(...): (line 223)
        # Processing the call keyword arguments (line 223)
        # Getting the type of 'self' (line 223)
        self_190673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 55), 'self', False)
        # Obtaining the member 'verbosity' of a type (line 223)
        verbosity_190674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 55), self_190673, 'verbosity')
        keyword_190675 = verbosity_190674
        # Getting the type of 'self' (line 224)
        self_190676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 54), 'self', False)
        # Obtaining the member 'failfast' of a type (line 224)
        failfast_190677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 54), self_190676, 'failfast')
        keyword_190678 = failfast_190677
        # Getting the type of 'self' (line 225)
        self_190679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 52), 'self', False)
        # Obtaining the member 'buffer' of a type (line 225)
        buffer_190680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 52), self_190679, 'buffer')
        keyword_190681 = buffer_190680
        kwargs_190682 = {'buffer': keyword_190681, 'verbosity': keyword_190675, 'failfast': keyword_190678}
        # Getting the type of 'self' (line 223)
        self_190671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), 'self', False)
        # Obtaining the member 'testRunner' of a type (line 223)
        testRunner_190672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 29), self_190671, 'testRunner')
        # Calling testRunner(args, kwargs) (line 223)
        testRunner_call_result_190683 = invoke(stypy.reporting.localization.Localization(__file__, 223, 29), testRunner_190672, *[], **kwargs_190682)
        
        # Assigning a type to the variable 'testRunner' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'testRunner', testRunner_call_result_190683)
        # SSA branch for the except part of a try statement (line 222)
        # SSA branch for the except 'TypeError' branch of a try statement (line 222)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 228):
        
        # Assigning a Call to a Name (line 228):
        
        # Call to testRunner(...): (line 228)
        # Processing the call keyword arguments (line 228)
        kwargs_190686 = {}
        # Getting the type of 'self' (line 228)
        self_190684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 29), 'self', False)
        # Obtaining the member 'testRunner' of a type (line 228)
        testRunner_190685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 29), self_190684, 'testRunner')
        # Calling testRunner(args, kwargs) (line 228)
        testRunner_call_result_190687 = invoke(stypy.reporting.localization.Localization(__file__, 228, 29), testRunner_190685, *[], **kwargs_190686)
        
        # Assigning a type to the variable 'testRunner' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'testRunner', testRunner_call_result_190687)
        # SSA join for try-except statement (line 222)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 221)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 231):
        
        # Assigning a Attribute to a Name (line 231):
        # Getting the type of 'self' (line 231)
        self_190688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 25), 'self')
        # Obtaining the member 'testRunner' of a type (line 231)
        testRunner_190689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 25), self_190688, 'testRunner')
        # Assigning a type to the variable 'testRunner' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'testRunner', testRunner_190689)
        # SSA join for if statement (line 221)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 232):
        
        # Assigning a Call to a Attribute (line 232):
        
        # Call to run(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'self' (line 232)
        self_190692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 37), 'self', False)
        # Obtaining the member 'test' of a type (line 232)
        test_190693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 37), self_190692, 'test')
        # Processing the call keyword arguments (line 232)
        kwargs_190694 = {}
        # Getting the type of 'testRunner' (line 232)
        testRunner_190690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 22), 'testRunner', False)
        # Obtaining the member 'run' of a type (line 232)
        run_190691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 22), testRunner_190690, 'run')
        # Calling run(args, kwargs) (line 232)
        run_call_result_190695 = invoke(stypy.reporting.localization.Localization(__file__, 232, 22), run_190691, *[test_190693], **kwargs_190694)
        
        # Getting the type of 'self' (line 232)
        self_190696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'self')
        # Setting the type of the member 'result' of a type (line 232)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), self_190696, 'result', run_call_result_190695)
        
        # Getting the type of 'self' (line 233)
        self_190697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), 'self')
        # Obtaining the member 'exit' of a type (line 233)
        exit_190698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 11), self_190697, 'exit')
        # Testing the type of an if condition (line 233)
        if_condition_190699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 8), exit_190698)
        # Assigning a type to the variable 'if_condition_190699' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'if_condition_190699', if_condition_190699)
        # SSA begins for if statement (line 233)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to exit(...): (line 234)
        # Processing the call arguments (line 234)
        
        
        # Call to wasSuccessful(...): (line 234)
        # Processing the call keyword arguments (line 234)
        kwargs_190705 = {}
        # Getting the type of 'self' (line 234)
        self_190702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 25), 'self', False)
        # Obtaining the member 'result' of a type (line 234)
        result_190703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 25), self_190702, 'result')
        # Obtaining the member 'wasSuccessful' of a type (line 234)
        wasSuccessful_190704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 25), result_190703, 'wasSuccessful')
        # Calling wasSuccessful(args, kwargs) (line 234)
        wasSuccessful_call_result_190706 = invoke(stypy.reporting.localization.Localization(__file__, 234, 25), wasSuccessful_190704, *[], **kwargs_190705)
        
        # Applying the 'not' unary operator (line 234)
        result_not__190707 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 21), 'not', wasSuccessful_call_result_190706)
        
        # Processing the call keyword arguments (line 234)
        kwargs_190708 = {}
        # Getting the type of 'sys' (line 234)
        sys_190700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'sys', False)
        # Obtaining the member 'exit' of a type (line 234)
        exit_190701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), sys_190700, 'exit')
        # Calling exit(args, kwargs) (line 234)
        exit_call_result_190709 = invoke(stypy.reporting.localization.Localization(__file__, 234, 12), exit_190701, *[result_not__190707], **kwargs_190708)
        
        # SSA join for if statement (line 233)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'runTests(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runTests' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_190710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190710)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runTests'
        return stypy_return_type_190710


# Assigning a type to the variable 'TestProgram' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'TestProgram', TestProgram)

# Assigning a Name to a Name (line 67):
# Getting the type of 'USAGE_FROM_MODULE' (line 67)
USAGE_FROM_MODULE_190711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'USAGE_FROM_MODULE')
# Getting the type of 'TestProgram'
TestProgram_190712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestProgram')
# Setting the type of the member 'USAGE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestProgram_190712, 'USAGE', USAGE_FROM_MODULE_190711)

# Assigning a Name to a Name (line 70):
# Getting the type of 'None' (line 70)
None_190713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 48), 'None')
# Getting the type of 'TestProgram'
TestProgram_190714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestProgram')
# Setting the type of the member 'progName' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestProgram_190714, 'progName', None_190713)

# Assigning a Name to a Name (line 70):
# Getting the type of 'TestProgram'
TestProgram_190715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestProgram')
# Obtaining the member 'progName' of a type
progName_190716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestProgram_190715, 'progName')
# Getting the type of 'TestProgram'
TestProgram_190717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestProgram')
# Setting the type of the member 'buffer' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestProgram_190717, 'buffer', progName_190716)

# Assigning a Name to a Name (line 70):
# Getting the type of 'TestProgram'
TestProgram_190718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestProgram')
# Obtaining the member 'buffer' of a type
buffer_190719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestProgram_190718, 'buffer')
# Getting the type of 'TestProgram'
TestProgram_190720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestProgram')
# Setting the type of the member 'catchbreak' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestProgram_190720, 'catchbreak', buffer_190719)

# Assigning a Name to a Name (line 70):
# Getting the type of 'TestProgram'
TestProgram_190721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestProgram')
# Obtaining the member 'catchbreak' of a type
catchbreak_190722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestProgram_190721, 'catchbreak')
# Getting the type of 'TestProgram'
TestProgram_190723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestProgram')
# Setting the type of the member 'failfast' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestProgram_190723, 'failfast', catchbreak_190722)

# Assigning a Name to a Name (line 236):

# Assigning a Name to a Name (line 236):
# Getting the type of 'TestProgram' (line 236)
TestProgram_190724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 7), 'TestProgram')
# Assigning a type to the variable 'main' (line 236)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), 'main', TestProgram_190724)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
