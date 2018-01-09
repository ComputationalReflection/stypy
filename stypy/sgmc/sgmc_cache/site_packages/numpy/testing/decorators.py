
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Decorators for labeling and modifying behavior of test objects.
3: 
4: Decorators that merely return a modified version of the original
5: function object are straightforward. Decorators that return a new
6: function object need to use
7: ::
8: 
9:   nose.tools.make_decorator(original_function)(decorator)
10: 
11: in returning the decorator, in order to preserve meta-data such as
12: function name, setup and teardown functions and so on - see
13: ``nose.tools`` for more information.
14: 
15: '''
16: from __future__ import division, absolute_import, print_function
17: 
18: import warnings
19: import collections
20: 
21: from .utils import SkipTest
22: 
23: def slow(t):
24:     '''
25:     Label a test as 'slow'.
26: 
27:     The exact definition of a slow test is obviously both subjective and
28:     hardware-dependent, but in general any individual test that requires more
29:     than a second or two should be labeled as slow (the whole suite consits of
30:     thousands of tests, so even a second is significant).
31: 
32:     Parameters
33:     ----------
34:     t : callable
35:         The test to label as slow.
36: 
37:     Returns
38:     -------
39:     t : callable
40:         The decorated test `t`.
41: 
42:     Examples
43:     --------
44:     The `numpy.testing` module includes ``import decorators as dec``.
45:     A test can be decorated as slow like this::
46: 
47:       from numpy.testing import *
48: 
49:       @dec.slow
50:       def test_big(self):
51:           print('Big, slow test')
52: 
53:     '''
54: 
55:     t.slow = True
56:     return t
57: 
58: def setastest(tf=True):
59:     '''
60:     Signals to nose that this function is or is not a test.
61: 
62:     Parameters
63:     ----------
64:     tf : bool
65:         If True, specifies that the decorated callable is a test.
66:         If False, specifies that the decorated callable is not a test.
67:         Default is True.
68: 
69:     Notes
70:     -----
71:     This decorator can't use the nose namespace, because it can be
72:     called from a non-test module. See also ``istest`` and ``nottest`` in
73:     ``nose.tools``.
74: 
75:     Examples
76:     --------
77:     `setastest` can be used in the following way::
78: 
79:       from numpy.testing.decorators import setastest
80: 
81:       @setastest(False)
82:       def func_with_test_in_name(arg1, arg2):
83:           pass
84: 
85:     '''
86:     def set_test(t):
87:         t.__test__ = tf
88:         return t
89:     return set_test
90: 
91: def skipif(skip_condition, msg=None):
92:     '''
93:     Make function raise SkipTest exception if a given condition is true.
94: 
95:     If the condition is a callable, it is used at runtime to dynamically
96:     make the decision. This is useful for tests that may require costly
97:     imports, to delay the cost until the test suite is actually executed.
98: 
99:     Parameters
100:     ----------
101:     skip_condition : bool or callable
102:         Flag to determine whether to skip the decorated test.
103:     msg : str, optional
104:         Message to give on raising a SkipTest exception. Default is None.
105: 
106:     Returns
107:     -------
108:     decorator : function
109:         Decorator which, when applied to a function, causes SkipTest
110:         to be raised when `skip_condition` is True, and the function
111:         to be called normally otherwise.
112: 
113:     Notes
114:     -----
115:     The decorator itself is decorated with the ``nose.tools.make_decorator``
116:     function in order to transmit function name, and various other metadata.
117: 
118:     '''
119: 
120:     def skip_decorator(f):
121:         # Local import to avoid a hard nose dependency and only incur the
122:         # import time overhead at actual test-time.
123:         import nose
124: 
125:         # Allow for both boolean or callable skip conditions.
126:         if isinstance(skip_condition, collections.Callable):
127:             skip_val = lambda: skip_condition()
128:         else:
129:             skip_val = lambda: skip_condition
130: 
131:         def get_msg(func,msg=None):
132:             '''Skip message with information about function being skipped.'''
133:             if msg is None:
134:                 out = 'Test skipped due to test condition'
135:             else:
136:                 out = msg
137: 
138:             return "Skipping test: %s: %s" % (func.__name__, out)
139: 
140:         # We need to define *two* skippers because Python doesn't allow both
141:         # return with value and yield inside the same function.
142:         def skipper_func(*args, **kwargs):
143:             '''Skipper for normal test functions.'''
144:             if skip_val():
145:                 raise SkipTest(get_msg(f, msg))
146:             else:
147:                 return f(*args, **kwargs)
148: 
149:         def skipper_gen(*args, **kwargs):
150:             '''Skipper for test generators.'''
151:             if skip_val():
152:                 raise SkipTest(get_msg(f, msg))
153:             else:
154:                 for x in f(*args, **kwargs):
155:                     yield x
156: 
157:         # Choose the right skipper to use when building the actual decorator.
158:         if nose.util.isgenerator(f):
159:             skipper = skipper_gen
160:         else:
161:             skipper = skipper_func
162: 
163:         return nose.tools.make_decorator(f)(skipper)
164: 
165:     return skip_decorator
166: 
167: 
168: def knownfailureif(fail_condition, msg=None):
169:     '''
170:     Make function raise KnownFailureException exception if given condition is true.
171: 
172:     If the condition is a callable, it is used at runtime to dynamically
173:     make the decision. This is useful for tests that may require costly
174:     imports, to delay the cost until the test suite is actually executed.
175: 
176:     Parameters
177:     ----------
178:     fail_condition : bool or callable
179:         Flag to determine whether to mark the decorated test as a known
180:         failure (if True) or not (if False).
181:     msg : str, optional
182:         Message to give on raising a KnownFailureException exception.
183:         Default is None.
184: 
185:     Returns
186:     -------
187:     decorator : function
188:         Decorator, which, when applied to a function, causes
189:         KnownFailureException to be raised when `fail_condition` is True,
190:         and the function to be called normally otherwise.
191: 
192:     Notes
193:     -----
194:     The decorator itself is decorated with the ``nose.tools.make_decorator``
195:     function in order to transmit function name, and various other metadata.
196: 
197:     '''
198:     if msg is None:
199:         msg = 'Test skipped due to known failure'
200: 
201:     # Allow for both boolean or callable known failure conditions.
202:     if isinstance(fail_condition, collections.Callable):
203:         fail_val = lambda: fail_condition()
204:     else:
205:         fail_val = lambda: fail_condition
206: 
207:     def knownfail_decorator(f):
208:         # Local import to avoid a hard nose dependency and only incur the
209:         # import time overhead at actual test-time.
210:         import nose
211:         from .noseclasses import KnownFailureException
212: 
213:         def knownfailer(*args, **kwargs):
214:             if fail_val():
215:                 raise KnownFailureException(msg)
216:             else:
217:                 return f(*args, **kwargs)
218:         return nose.tools.make_decorator(f)(knownfailer)
219: 
220:     return knownfail_decorator
221: 
222: def deprecated(conditional=True):
223:     '''
224:     Filter deprecation warnings while running the test suite.
225: 
226:     This decorator can be used to filter DeprecationWarning's, to avoid
227:     printing them during the test suite run, while checking that the test
228:     actually raises a DeprecationWarning.
229: 
230:     Parameters
231:     ----------
232:     conditional : bool or callable, optional
233:         Flag to determine whether to mark test as deprecated or not. If the
234:         condition is a callable, it is used at runtime to dynamically make the
235:         decision. Default is True.
236: 
237:     Returns
238:     -------
239:     decorator : function
240:         The `deprecated` decorator itself.
241: 
242:     Notes
243:     -----
244:     .. versionadded:: 1.4.0
245: 
246:     '''
247:     def deprecate_decorator(f):
248:         # Local import to avoid a hard nose dependency and only incur the
249:         # import time overhead at actual test-time.
250:         import nose
251: 
252:         def _deprecated_imp(*args, **kwargs):
253:             # Poor man's replacement for the with statement
254:             with warnings.catch_warnings(record=True) as l:
255:                 warnings.simplefilter('always')
256:                 f(*args, **kwargs)
257:                 if not len(l) > 0:
258:                     raise AssertionError("No warning raised when calling %s"
259:                             % f.__name__)
260:                 if not l[0].category is DeprecationWarning:
261:                     raise AssertionError("First warning for %s is not a "
262:                             "DeprecationWarning( is %s)" % (f.__name__, l[0]))
263: 
264:         if isinstance(conditional, collections.Callable):
265:             cond = conditional()
266:         else:
267:             cond = conditional
268:         if cond:
269:             return nose.tools.make_decorator(f)(_deprecated_imp)
270:         else:
271:             return f
272:     return deprecate_decorator
273: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_180816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'str', '\nDecorators for labeling and modifying behavior of test objects.\n\nDecorators that merely return a modified version of the original\nfunction object are straightforward. Decorators that return a new\nfunction object need to use\n::\n\n  nose.tools.make_decorator(original_function)(decorator)\n\nin returning the decorator, in order to preserve meta-data such as\nfunction name, setup and teardown functions and so on - see\n``nose.tools`` for more information.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import warnings' statement (line 18)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import collections' statement (line 19)
import collections

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'collections', collections, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from numpy.testing.utils import SkipTest' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_180817 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.testing.utils')

if (type(import_180817) is not StypyTypeError):

    if (import_180817 != 'pyd_module'):
        __import__(import_180817)
        sys_modules_180818 = sys.modules[import_180817]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.testing.utils', sys_modules_180818.module_type_store, module_type_store, ['SkipTest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_180818, sys_modules_180818.module_type_store, module_type_store)
    else:
        from numpy.testing.utils import SkipTest

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.testing.utils', None, module_type_store, ['SkipTest'], [SkipTest])

else:
    # Assigning a type to the variable 'numpy.testing.utils' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.testing.utils', import_180817)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')


@norecursion
def slow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'slow'
    module_type_store = module_type_store.open_function_context('slow', 23, 0, False)
    
    # Passed parameters checking function
    slow.stypy_localization = localization
    slow.stypy_type_of_self = None
    slow.stypy_type_store = module_type_store
    slow.stypy_function_name = 'slow'
    slow.stypy_param_names_list = ['t']
    slow.stypy_varargs_param_name = None
    slow.stypy_kwargs_param_name = None
    slow.stypy_call_defaults = defaults
    slow.stypy_call_varargs = varargs
    slow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'slow', ['t'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'slow', localization, ['t'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'slow(...)' code ##################

    str_180819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, (-1)), 'str', "\n    Label a test as 'slow'.\n\n    The exact definition of a slow test is obviously both subjective and\n    hardware-dependent, but in general any individual test that requires more\n    than a second or two should be labeled as slow (the whole suite consits of\n    thousands of tests, so even a second is significant).\n\n    Parameters\n    ----------\n    t : callable\n        The test to label as slow.\n\n    Returns\n    -------\n    t : callable\n        The decorated test `t`.\n\n    Examples\n    --------\n    The `numpy.testing` module includes ``import decorators as dec``.\n    A test can be decorated as slow like this::\n\n      from numpy.testing import *\n\n      @dec.slow\n      def test_big(self):\n          print('Big, slow test')\n\n    ")
    
    # Assigning a Name to a Attribute (line 55):
    # Getting the type of 'True' (line 55)
    True_180820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 13), 'True')
    # Getting the type of 't' (line 55)
    t_180821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 't')
    # Setting the type of the member 'slow' of a type (line 55)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 4), t_180821, 'slow', True_180820)
    # Getting the type of 't' (line 56)
    t_180822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 't')
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type', t_180822)
    
    # ################# End of 'slow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'slow' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_180823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_180823)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'slow'
    return stypy_return_type_180823

# Assigning a type to the variable 'slow' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'slow', slow)

@norecursion
def setastest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 58)
    True_180824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'True')
    defaults = [True_180824]
    # Create a new context for function 'setastest'
    module_type_store = module_type_store.open_function_context('setastest', 58, 0, False)
    
    # Passed parameters checking function
    setastest.stypy_localization = localization
    setastest.stypy_type_of_self = None
    setastest.stypy_type_store = module_type_store
    setastest.stypy_function_name = 'setastest'
    setastest.stypy_param_names_list = ['tf']
    setastest.stypy_varargs_param_name = None
    setastest.stypy_kwargs_param_name = None
    setastest.stypy_call_defaults = defaults
    setastest.stypy_call_varargs = varargs
    setastest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setastest', ['tf'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setastest', localization, ['tf'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setastest(...)' code ##################

    str_180825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'str', "\n    Signals to nose that this function is or is not a test.\n\n    Parameters\n    ----------\n    tf : bool\n        If True, specifies that the decorated callable is a test.\n        If False, specifies that the decorated callable is not a test.\n        Default is True.\n\n    Notes\n    -----\n    This decorator can't use the nose namespace, because it can be\n    called from a non-test module. See also ``istest`` and ``nottest`` in\n    ``nose.tools``.\n\n    Examples\n    --------\n    `setastest` can be used in the following way::\n\n      from numpy.testing.decorators import setastest\n\n      @setastest(False)\n      def func_with_test_in_name(arg1, arg2):\n          pass\n\n    ")

    @norecursion
    def set_test(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_test'
        module_type_store = module_type_store.open_function_context('set_test', 86, 4, False)
        
        # Passed parameters checking function
        set_test.stypy_localization = localization
        set_test.stypy_type_of_self = None
        set_test.stypy_type_store = module_type_store
        set_test.stypy_function_name = 'set_test'
        set_test.stypy_param_names_list = ['t']
        set_test.stypy_varargs_param_name = None
        set_test.stypy_kwargs_param_name = None
        set_test.stypy_call_defaults = defaults
        set_test.stypy_call_varargs = varargs
        set_test.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'set_test', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_test', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_test(...)' code ##################

        
        # Assigning a Name to a Attribute (line 87):
        # Getting the type of 'tf' (line 87)
        tf_180826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 21), 'tf')
        # Getting the type of 't' (line 87)
        t_180827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 't')
        # Setting the type of the member '__test__' of a type (line 87)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), t_180827, '__test__', tf_180826)
        # Getting the type of 't' (line 88)
        t_180828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 't')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', t_180828)
        
        # ################# End of 'set_test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_test' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_180829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180829)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_test'
        return stypy_return_type_180829

    # Assigning a type to the variable 'set_test' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'set_test', set_test)
    # Getting the type of 'set_test' (line 89)
    set_test_180830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'set_test')
    # Assigning a type to the variable 'stypy_return_type' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type', set_test_180830)
    
    # ################# End of 'setastest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setastest' in the type store
    # Getting the type of 'stypy_return_type' (line 58)
    stypy_return_type_180831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_180831)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setastest'
    return stypy_return_type_180831

# Assigning a type to the variable 'setastest' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'setastest', setastest)

@norecursion
def skipif(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 91)
    None_180832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 31), 'None')
    defaults = [None_180832]
    # Create a new context for function 'skipif'
    module_type_store = module_type_store.open_function_context('skipif', 91, 0, False)
    
    # Passed parameters checking function
    skipif.stypy_localization = localization
    skipif.stypy_type_of_self = None
    skipif.stypy_type_store = module_type_store
    skipif.stypy_function_name = 'skipif'
    skipif.stypy_param_names_list = ['skip_condition', 'msg']
    skipif.stypy_varargs_param_name = None
    skipif.stypy_kwargs_param_name = None
    skipif.stypy_call_defaults = defaults
    skipif.stypy_call_varargs = varargs
    skipif.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'skipif', ['skip_condition', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'skipif', localization, ['skip_condition', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'skipif(...)' code ##################

    str_180833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, (-1)), 'str', '\n    Make function raise SkipTest exception if a given condition is true.\n\n    If the condition is a callable, it is used at runtime to dynamically\n    make the decision. This is useful for tests that may require costly\n    imports, to delay the cost until the test suite is actually executed.\n\n    Parameters\n    ----------\n    skip_condition : bool or callable\n        Flag to determine whether to skip the decorated test.\n    msg : str, optional\n        Message to give on raising a SkipTest exception. Default is None.\n\n    Returns\n    -------\n    decorator : function\n        Decorator which, when applied to a function, causes SkipTest\n        to be raised when `skip_condition` is True, and the function\n        to be called normally otherwise.\n\n    Notes\n    -----\n    The decorator itself is decorated with the ``nose.tools.make_decorator``\n    function in order to transmit function name, and various other metadata.\n\n    ')

    @norecursion
    def skip_decorator(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'skip_decorator'
        module_type_store = module_type_store.open_function_context('skip_decorator', 120, 4, False)
        
        # Passed parameters checking function
        skip_decorator.stypy_localization = localization
        skip_decorator.stypy_type_of_self = None
        skip_decorator.stypy_type_store = module_type_store
        skip_decorator.stypy_function_name = 'skip_decorator'
        skip_decorator.stypy_param_names_list = ['f']
        skip_decorator.stypy_varargs_param_name = None
        skip_decorator.stypy_kwargs_param_name = None
        skip_decorator.stypy_call_defaults = defaults
        skip_decorator.stypy_call_varargs = varargs
        skip_decorator.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'skip_decorator', ['f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'skip_decorator', localization, ['f'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'skip_decorator(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 123, 8))
        
        # 'import nose' statement (line 123)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
        import_180834 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 123, 8), 'nose')

        if (type(import_180834) is not StypyTypeError):

            if (import_180834 != 'pyd_module'):
                __import__(import_180834)
                sys_modules_180835 = sys.modules[import_180834]
                import_module(stypy.reporting.localization.Localization(__file__, 123, 8), 'nose', sys_modules_180835.module_type_store, module_type_store)
            else:
                import nose

                import_module(stypy.reporting.localization.Localization(__file__, 123, 8), 'nose', nose, module_type_store)

        else:
            # Assigning a type to the variable 'nose' (line 123)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'nose', import_180834)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
        
        
        
        # Call to isinstance(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'skip_condition' (line 126)
        skip_condition_180837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'skip_condition', False)
        # Getting the type of 'collections' (line 126)
        collections_180838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 38), 'collections', False)
        # Obtaining the member 'Callable' of a type (line 126)
        Callable_180839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 38), collections_180838, 'Callable')
        # Processing the call keyword arguments (line 126)
        kwargs_180840 = {}
        # Getting the type of 'isinstance' (line 126)
        isinstance_180836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 126)
        isinstance_call_result_180841 = invoke(stypy.reporting.localization.Localization(__file__, 126, 11), isinstance_180836, *[skip_condition_180837, Callable_180839], **kwargs_180840)
        
        # Testing the type of an if condition (line 126)
        if_condition_180842 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 8), isinstance_call_result_180841)
        # Assigning a type to the variable 'if_condition_180842' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'if_condition_180842', if_condition_180842)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Lambda to a Name (line 127):

        @norecursion
        def _stypy_temp_lambda_44(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_44'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_44', 127, 23, True)
            # Passed parameters checking function
            _stypy_temp_lambda_44.stypy_localization = localization
            _stypy_temp_lambda_44.stypy_type_of_self = None
            _stypy_temp_lambda_44.stypy_type_store = module_type_store
            _stypy_temp_lambda_44.stypy_function_name = '_stypy_temp_lambda_44'
            _stypy_temp_lambda_44.stypy_param_names_list = []
            _stypy_temp_lambda_44.stypy_varargs_param_name = None
            _stypy_temp_lambda_44.stypy_kwargs_param_name = None
            _stypy_temp_lambda_44.stypy_call_defaults = defaults
            _stypy_temp_lambda_44.stypy_call_varargs = varargs
            _stypy_temp_lambda_44.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_44', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_44', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to skip_condition(...): (line 127)
            # Processing the call keyword arguments (line 127)
            kwargs_180844 = {}
            # Getting the type of 'skip_condition' (line 127)
            skip_condition_180843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 31), 'skip_condition', False)
            # Calling skip_condition(args, kwargs) (line 127)
            skip_condition_call_result_180845 = invoke(stypy.reporting.localization.Localization(__file__, 127, 31), skip_condition_180843, *[], **kwargs_180844)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'stypy_return_type', skip_condition_call_result_180845)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_44' in the type store
            # Getting the type of 'stypy_return_type' (line 127)
            stypy_return_type_180846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_180846)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_44'
            return stypy_return_type_180846

        # Assigning a type to the variable '_stypy_temp_lambda_44' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), '_stypy_temp_lambda_44', _stypy_temp_lambda_44)
        # Getting the type of '_stypy_temp_lambda_44' (line 127)
        _stypy_temp_lambda_44_180847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), '_stypy_temp_lambda_44')
        # Assigning a type to the variable 'skip_val' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'skip_val', _stypy_temp_lambda_44_180847)
        # SSA branch for the else part of an if statement (line 126)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Lambda to a Name (line 129):

        @norecursion
        def _stypy_temp_lambda_45(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_45'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_45', 129, 23, True)
            # Passed parameters checking function
            _stypy_temp_lambda_45.stypy_localization = localization
            _stypy_temp_lambda_45.stypy_type_of_self = None
            _stypy_temp_lambda_45.stypy_type_store = module_type_store
            _stypy_temp_lambda_45.stypy_function_name = '_stypy_temp_lambda_45'
            _stypy_temp_lambda_45.stypy_param_names_list = []
            _stypy_temp_lambda_45.stypy_varargs_param_name = None
            _stypy_temp_lambda_45.stypy_kwargs_param_name = None
            _stypy_temp_lambda_45.stypy_call_defaults = defaults
            _stypy_temp_lambda_45.stypy_call_varargs = varargs
            _stypy_temp_lambda_45.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_45', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_45', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'skip_condition' (line 129)
            skip_condition_180848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 31), 'skip_condition')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'stypy_return_type', skip_condition_180848)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_45' in the type store
            # Getting the type of 'stypy_return_type' (line 129)
            stypy_return_type_180849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_180849)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_45'
            return stypy_return_type_180849

        # Assigning a type to the variable '_stypy_temp_lambda_45' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), '_stypy_temp_lambda_45', _stypy_temp_lambda_45)
        # Getting the type of '_stypy_temp_lambda_45' (line 129)
        _stypy_temp_lambda_45_180850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), '_stypy_temp_lambda_45')
        # Assigning a type to the variable 'skip_val' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'skip_val', _stypy_temp_lambda_45_180850)
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def get_msg(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'None' (line 131)
            None_180851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 29), 'None')
            defaults = [None_180851]
            # Create a new context for function 'get_msg'
            module_type_store = module_type_store.open_function_context('get_msg', 131, 8, False)
            
            # Passed parameters checking function
            get_msg.stypy_localization = localization
            get_msg.stypy_type_of_self = None
            get_msg.stypy_type_store = module_type_store
            get_msg.stypy_function_name = 'get_msg'
            get_msg.stypy_param_names_list = ['func', 'msg']
            get_msg.stypy_varargs_param_name = None
            get_msg.stypy_kwargs_param_name = None
            get_msg.stypy_call_defaults = defaults
            get_msg.stypy_call_varargs = varargs
            get_msg.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'get_msg', ['func', 'msg'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'get_msg', localization, ['func', 'msg'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'get_msg(...)' code ##################

            str_180852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 12), 'str', 'Skip message with information about function being skipped.')
            
            # Type idiom detected: calculating its left and rigth part (line 133)
            # Getting the type of 'msg' (line 133)
            msg_180853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'msg')
            # Getting the type of 'None' (line 133)
            None_180854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'None')
            
            (may_be_180855, more_types_in_union_180856) = may_be_none(msg_180853, None_180854)

            if may_be_180855:

                if more_types_in_union_180856:
                    # Runtime conditional SSA (line 133)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Str to a Name (line 134):
                str_180857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 22), 'str', 'Test skipped due to test condition')
                # Assigning a type to the variable 'out' (line 134)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'out', str_180857)

                if more_types_in_union_180856:
                    # Runtime conditional SSA for else branch (line 133)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_180855) or more_types_in_union_180856):
                
                # Assigning a Name to a Name (line 136):
                # Getting the type of 'msg' (line 136)
                msg_180858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 22), 'msg')
                # Assigning a type to the variable 'out' (line 136)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'out', msg_180858)

                if (may_be_180855 and more_types_in_union_180856):
                    # SSA join for if statement (line 133)
                    module_type_store = module_type_store.join_ssa_context()


            
            str_180859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 19), 'str', 'Skipping test: %s: %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 138)
            tuple_180860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 46), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 138)
            # Adding element type (line 138)
            # Getting the type of 'func' (line 138)
            func_180861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 46), 'func')
            # Obtaining the member '__name__' of a type (line 138)
            name___180862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 46), func_180861, '__name__')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 46), tuple_180860, name___180862)
            # Adding element type (line 138)
            # Getting the type of 'out' (line 138)
            out_180863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 61), 'out')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 46), tuple_180860, out_180863)
            
            # Applying the binary operator '%' (line 138)
            result_mod_180864 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 19), '%', str_180859, tuple_180860)
            
            # Assigning a type to the variable 'stypy_return_type' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'stypy_return_type', result_mod_180864)
            
            # ################# End of 'get_msg(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'get_msg' in the type store
            # Getting the type of 'stypy_return_type' (line 131)
            stypy_return_type_180865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_180865)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'get_msg'
            return stypy_return_type_180865

        # Assigning a type to the variable 'get_msg' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'get_msg', get_msg)

        @norecursion
        def skipper_func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'skipper_func'
            module_type_store = module_type_store.open_function_context('skipper_func', 142, 8, False)
            
            # Passed parameters checking function
            skipper_func.stypy_localization = localization
            skipper_func.stypy_type_of_self = None
            skipper_func.stypy_type_store = module_type_store
            skipper_func.stypy_function_name = 'skipper_func'
            skipper_func.stypy_param_names_list = []
            skipper_func.stypy_varargs_param_name = 'args'
            skipper_func.stypy_kwargs_param_name = 'kwargs'
            skipper_func.stypy_call_defaults = defaults
            skipper_func.stypy_call_varargs = varargs
            skipper_func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'skipper_func', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'skipper_func', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'skipper_func(...)' code ##################

            str_180866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 12), 'str', 'Skipper for normal test functions.')
            
            
            # Call to skip_val(...): (line 144)
            # Processing the call keyword arguments (line 144)
            kwargs_180868 = {}
            # Getting the type of 'skip_val' (line 144)
            skip_val_180867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'skip_val', False)
            # Calling skip_val(args, kwargs) (line 144)
            skip_val_call_result_180869 = invoke(stypy.reporting.localization.Localization(__file__, 144, 15), skip_val_180867, *[], **kwargs_180868)
            
            # Testing the type of an if condition (line 144)
            if_condition_180870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 12), skip_val_call_result_180869)
            # Assigning a type to the variable 'if_condition_180870' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'if_condition_180870', if_condition_180870)
            # SSA begins for if statement (line 144)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to SkipTest(...): (line 145)
            # Processing the call arguments (line 145)
            
            # Call to get_msg(...): (line 145)
            # Processing the call arguments (line 145)
            # Getting the type of 'f' (line 145)
            f_180873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 39), 'f', False)
            # Getting the type of 'msg' (line 145)
            msg_180874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 42), 'msg', False)
            # Processing the call keyword arguments (line 145)
            kwargs_180875 = {}
            # Getting the type of 'get_msg' (line 145)
            get_msg_180872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 31), 'get_msg', False)
            # Calling get_msg(args, kwargs) (line 145)
            get_msg_call_result_180876 = invoke(stypy.reporting.localization.Localization(__file__, 145, 31), get_msg_180872, *[f_180873, msg_180874], **kwargs_180875)
            
            # Processing the call keyword arguments (line 145)
            kwargs_180877 = {}
            # Getting the type of 'SkipTest' (line 145)
            SkipTest_180871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 22), 'SkipTest', False)
            # Calling SkipTest(args, kwargs) (line 145)
            SkipTest_call_result_180878 = invoke(stypy.reporting.localization.Localization(__file__, 145, 22), SkipTest_180871, *[get_msg_call_result_180876], **kwargs_180877)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 145, 16), SkipTest_call_result_180878, 'raise parameter', BaseException)
            # SSA branch for the else part of an if statement (line 144)
            module_type_store.open_ssa_branch('else')
            
            # Call to f(...): (line 147)
            # Getting the type of 'args' (line 147)
            args_180880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 26), 'args', False)
            # Processing the call keyword arguments (line 147)
            # Getting the type of 'kwargs' (line 147)
            kwargs_180881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 34), 'kwargs', False)
            kwargs_180882 = {'kwargs_180881': kwargs_180881}
            # Getting the type of 'f' (line 147)
            f_180879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'f', False)
            # Calling f(args, kwargs) (line 147)
            f_call_result_180883 = invoke(stypy.reporting.localization.Localization(__file__, 147, 23), f_180879, *[args_180880], **kwargs_180882)
            
            # Assigning a type to the variable 'stypy_return_type' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'stypy_return_type', f_call_result_180883)
            # SSA join for if statement (line 144)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'skipper_func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'skipper_func' in the type store
            # Getting the type of 'stypy_return_type' (line 142)
            stypy_return_type_180884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_180884)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'skipper_func'
            return stypy_return_type_180884

        # Assigning a type to the variable 'skipper_func' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'skipper_func', skipper_func)

        @norecursion
        def skipper_gen(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'skipper_gen'
            module_type_store = module_type_store.open_function_context('skipper_gen', 149, 8, False)
            
            # Passed parameters checking function
            skipper_gen.stypy_localization = localization
            skipper_gen.stypy_type_of_self = None
            skipper_gen.stypy_type_store = module_type_store
            skipper_gen.stypy_function_name = 'skipper_gen'
            skipper_gen.stypy_param_names_list = []
            skipper_gen.stypy_varargs_param_name = 'args'
            skipper_gen.stypy_kwargs_param_name = 'kwargs'
            skipper_gen.stypy_call_defaults = defaults
            skipper_gen.stypy_call_varargs = varargs
            skipper_gen.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'skipper_gen', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'skipper_gen', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'skipper_gen(...)' code ##################

            str_180885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 12), 'str', 'Skipper for test generators.')
            
            
            # Call to skip_val(...): (line 151)
            # Processing the call keyword arguments (line 151)
            kwargs_180887 = {}
            # Getting the type of 'skip_val' (line 151)
            skip_val_180886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'skip_val', False)
            # Calling skip_val(args, kwargs) (line 151)
            skip_val_call_result_180888 = invoke(stypy.reporting.localization.Localization(__file__, 151, 15), skip_val_180886, *[], **kwargs_180887)
            
            # Testing the type of an if condition (line 151)
            if_condition_180889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 12), skip_val_call_result_180888)
            # Assigning a type to the variable 'if_condition_180889' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'if_condition_180889', if_condition_180889)
            # SSA begins for if statement (line 151)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to SkipTest(...): (line 152)
            # Processing the call arguments (line 152)
            
            # Call to get_msg(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'f' (line 152)
            f_180892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 39), 'f', False)
            # Getting the type of 'msg' (line 152)
            msg_180893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 42), 'msg', False)
            # Processing the call keyword arguments (line 152)
            kwargs_180894 = {}
            # Getting the type of 'get_msg' (line 152)
            get_msg_180891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'get_msg', False)
            # Calling get_msg(args, kwargs) (line 152)
            get_msg_call_result_180895 = invoke(stypy.reporting.localization.Localization(__file__, 152, 31), get_msg_180891, *[f_180892, msg_180893], **kwargs_180894)
            
            # Processing the call keyword arguments (line 152)
            kwargs_180896 = {}
            # Getting the type of 'SkipTest' (line 152)
            SkipTest_180890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'SkipTest', False)
            # Calling SkipTest(args, kwargs) (line 152)
            SkipTest_call_result_180897 = invoke(stypy.reporting.localization.Localization(__file__, 152, 22), SkipTest_180890, *[get_msg_call_result_180895], **kwargs_180896)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 152, 16), SkipTest_call_result_180897, 'raise parameter', BaseException)
            # SSA branch for the else part of an if statement (line 151)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to f(...): (line 154)
            # Getting the type of 'args' (line 154)
            args_180899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 28), 'args', False)
            # Processing the call keyword arguments (line 154)
            # Getting the type of 'kwargs' (line 154)
            kwargs_180900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 36), 'kwargs', False)
            kwargs_180901 = {'kwargs_180900': kwargs_180900}
            # Getting the type of 'f' (line 154)
            f_180898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 25), 'f', False)
            # Calling f(args, kwargs) (line 154)
            f_call_result_180902 = invoke(stypy.reporting.localization.Localization(__file__, 154, 25), f_180898, *[args_180899], **kwargs_180901)
            
            # Testing the type of a for loop iterable (line 154)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 154, 16), f_call_result_180902)
            # Getting the type of the for loop variable (line 154)
            for_loop_var_180903 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 154, 16), f_call_result_180902)
            # Assigning a type to the variable 'x' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'x', for_loop_var_180903)
            # SSA begins for a for statement (line 154)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Creating a generator
            # Getting the type of 'x' (line 155)
            x_180904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 26), 'x')
            GeneratorType_180905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 20), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 20), GeneratorType_180905, x_180904)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'stypy_return_type', GeneratorType_180905)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 151)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'skipper_gen(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'skipper_gen' in the type store
            # Getting the type of 'stypy_return_type' (line 149)
            stypy_return_type_180906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_180906)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'skipper_gen'
            return stypy_return_type_180906

        # Assigning a type to the variable 'skipper_gen' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'skipper_gen', skipper_gen)
        
        
        # Call to isgenerator(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'f' (line 158)
        f_180910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 33), 'f', False)
        # Processing the call keyword arguments (line 158)
        kwargs_180911 = {}
        # Getting the type of 'nose' (line 158)
        nose_180907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'nose', False)
        # Obtaining the member 'util' of a type (line 158)
        util_180908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 11), nose_180907, 'util')
        # Obtaining the member 'isgenerator' of a type (line 158)
        isgenerator_180909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 11), util_180908, 'isgenerator')
        # Calling isgenerator(args, kwargs) (line 158)
        isgenerator_call_result_180912 = invoke(stypy.reporting.localization.Localization(__file__, 158, 11), isgenerator_180909, *[f_180910], **kwargs_180911)
        
        # Testing the type of an if condition (line 158)
        if_condition_180913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 8), isgenerator_call_result_180912)
        # Assigning a type to the variable 'if_condition_180913' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'if_condition_180913', if_condition_180913)
        # SSA begins for if statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'skipper_gen' (line 159)
        skipper_gen_180914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'skipper_gen')
        # Assigning a type to the variable 'skipper' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'skipper', skipper_gen_180914)
        # SSA branch for the else part of an if statement (line 158)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 161):
        # Getting the type of 'skipper_func' (line 161)
        skipper_func_180915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 22), 'skipper_func')
        # Assigning a type to the variable 'skipper' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'skipper', skipper_func_180915)
        # SSA join for if statement (line 158)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to (...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'skipper' (line 163)
        skipper_180922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 44), 'skipper', False)
        # Processing the call keyword arguments (line 163)
        kwargs_180923 = {}
        
        # Call to make_decorator(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'f' (line 163)
        f_180919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 41), 'f', False)
        # Processing the call keyword arguments (line 163)
        kwargs_180920 = {}
        # Getting the type of 'nose' (line 163)
        nose_180916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'nose', False)
        # Obtaining the member 'tools' of a type (line 163)
        tools_180917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 15), nose_180916, 'tools')
        # Obtaining the member 'make_decorator' of a type (line 163)
        make_decorator_180918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 15), tools_180917, 'make_decorator')
        # Calling make_decorator(args, kwargs) (line 163)
        make_decorator_call_result_180921 = invoke(stypy.reporting.localization.Localization(__file__, 163, 15), make_decorator_180918, *[f_180919], **kwargs_180920)
        
        # Calling (args, kwargs) (line 163)
        _call_result_180924 = invoke(stypy.reporting.localization.Localization(__file__, 163, 15), make_decorator_call_result_180921, *[skipper_180922], **kwargs_180923)
        
        # Assigning a type to the variable 'stypy_return_type' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'stypy_return_type', _call_result_180924)
        
        # ################# End of 'skip_decorator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'skip_decorator' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_180925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180925)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'skip_decorator'
        return stypy_return_type_180925

    # Assigning a type to the variable 'skip_decorator' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'skip_decorator', skip_decorator)
    # Getting the type of 'skip_decorator' (line 165)
    skip_decorator_180926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'skip_decorator')
    # Assigning a type to the variable 'stypy_return_type' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type', skip_decorator_180926)
    
    # ################# End of 'skipif(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'skipif' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_180927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_180927)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'skipif'
    return stypy_return_type_180927

# Assigning a type to the variable 'skipif' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'skipif', skipif)

@norecursion
def knownfailureif(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 168)
    None_180928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 39), 'None')
    defaults = [None_180928]
    # Create a new context for function 'knownfailureif'
    module_type_store = module_type_store.open_function_context('knownfailureif', 168, 0, False)
    
    # Passed parameters checking function
    knownfailureif.stypy_localization = localization
    knownfailureif.stypy_type_of_self = None
    knownfailureif.stypy_type_store = module_type_store
    knownfailureif.stypy_function_name = 'knownfailureif'
    knownfailureif.stypy_param_names_list = ['fail_condition', 'msg']
    knownfailureif.stypy_varargs_param_name = None
    knownfailureif.stypy_kwargs_param_name = None
    knownfailureif.stypy_call_defaults = defaults
    knownfailureif.stypy_call_varargs = varargs
    knownfailureif.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'knownfailureif', ['fail_condition', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'knownfailureif', localization, ['fail_condition', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'knownfailureif(...)' code ##################

    str_180929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, (-1)), 'str', '\n    Make function raise KnownFailureException exception if given condition is true.\n\n    If the condition is a callable, it is used at runtime to dynamically\n    make the decision. This is useful for tests that may require costly\n    imports, to delay the cost until the test suite is actually executed.\n\n    Parameters\n    ----------\n    fail_condition : bool or callable\n        Flag to determine whether to mark the decorated test as a known\n        failure (if True) or not (if False).\n    msg : str, optional\n        Message to give on raising a KnownFailureException exception.\n        Default is None.\n\n    Returns\n    -------\n    decorator : function\n        Decorator, which, when applied to a function, causes\n        KnownFailureException to be raised when `fail_condition` is True,\n        and the function to be called normally otherwise.\n\n    Notes\n    -----\n    The decorator itself is decorated with the ``nose.tools.make_decorator``\n    function in order to transmit function name, and various other metadata.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 198)
    # Getting the type of 'msg' (line 198)
    msg_180930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 7), 'msg')
    # Getting the type of 'None' (line 198)
    None_180931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 14), 'None')
    
    (may_be_180932, more_types_in_union_180933) = may_be_none(msg_180930, None_180931)

    if may_be_180932:

        if more_types_in_union_180933:
            # Runtime conditional SSA (line 198)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Name (line 199):
        str_180934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 14), 'str', 'Test skipped due to known failure')
        # Assigning a type to the variable 'msg' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'msg', str_180934)

        if more_types_in_union_180933:
            # SSA join for if statement (line 198)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to isinstance(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'fail_condition' (line 202)
    fail_condition_180936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'fail_condition', False)
    # Getting the type of 'collections' (line 202)
    collections_180937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 34), 'collections', False)
    # Obtaining the member 'Callable' of a type (line 202)
    Callable_180938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 34), collections_180937, 'Callable')
    # Processing the call keyword arguments (line 202)
    kwargs_180939 = {}
    # Getting the type of 'isinstance' (line 202)
    isinstance_180935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 202)
    isinstance_call_result_180940 = invoke(stypy.reporting.localization.Localization(__file__, 202, 7), isinstance_180935, *[fail_condition_180936, Callable_180938], **kwargs_180939)
    
    # Testing the type of an if condition (line 202)
    if_condition_180941 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 4), isinstance_call_result_180940)
    # Assigning a type to the variable 'if_condition_180941' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'if_condition_180941', if_condition_180941)
    # SSA begins for if statement (line 202)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Lambda to a Name (line 203):

    @norecursion
    def _stypy_temp_lambda_46(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_46'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_46', 203, 19, True)
        # Passed parameters checking function
        _stypy_temp_lambda_46.stypy_localization = localization
        _stypy_temp_lambda_46.stypy_type_of_self = None
        _stypy_temp_lambda_46.stypy_type_store = module_type_store
        _stypy_temp_lambda_46.stypy_function_name = '_stypy_temp_lambda_46'
        _stypy_temp_lambda_46.stypy_param_names_list = []
        _stypy_temp_lambda_46.stypy_varargs_param_name = None
        _stypy_temp_lambda_46.stypy_kwargs_param_name = None
        _stypy_temp_lambda_46.stypy_call_defaults = defaults
        _stypy_temp_lambda_46.stypy_call_varargs = varargs
        _stypy_temp_lambda_46.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_46', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_46', [], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to fail_condition(...): (line 203)
        # Processing the call keyword arguments (line 203)
        kwargs_180943 = {}
        # Getting the type of 'fail_condition' (line 203)
        fail_condition_180942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 27), 'fail_condition', False)
        # Calling fail_condition(args, kwargs) (line 203)
        fail_condition_call_result_180944 = invoke(stypy.reporting.localization.Localization(__file__, 203, 27), fail_condition_180942, *[], **kwargs_180943)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 19), 'stypy_return_type', fail_condition_call_result_180944)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_46' in the type store
        # Getting the type of 'stypy_return_type' (line 203)
        stypy_return_type_180945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 19), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180945)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_46'
        return stypy_return_type_180945

    # Assigning a type to the variable '_stypy_temp_lambda_46' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 19), '_stypy_temp_lambda_46', _stypy_temp_lambda_46)
    # Getting the type of '_stypy_temp_lambda_46' (line 203)
    _stypy_temp_lambda_46_180946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 19), '_stypy_temp_lambda_46')
    # Assigning a type to the variable 'fail_val' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'fail_val', _stypy_temp_lambda_46_180946)
    # SSA branch for the else part of an if statement (line 202)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Lambda to a Name (line 205):

    @norecursion
    def _stypy_temp_lambda_47(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_47'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_47', 205, 19, True)
        # Passed parameters checking function
        _stypy_temp_lambda_47.stypy_localization = localization
        _stypy_temp_lambda_47.stypy_type_of_self = None
        _stypy_temp_lambda_47.stypy_type_store = module_type_store
        _stypy_temp_lambda_47.stypy_function_name = '_stypy_temp_lambda_47'
        _stypy_temp_lambda_47.stypy_param_names_list = []
        _stypy_temp_lambda_47.stypy_varargs_param_name = None
        _stypy_temp_lambda_47.stypy_kwargs_param_name = None
        _stypy_temp_lambda_47.stypy_call_defaults = defaults
        _stypy_temp_lambda_47.stypy_call_varargs = varargs
        _stypy_temp_lambda_47.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_47', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_47', [], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'fail_condition' (line 205)
        fail_condition_180947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'fail_condition')
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'stypy_return_type', fail_condition_180947)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_47' in the type store
        # Getting the type of 'stypy_return_type' (line 205)
        stypy_return_type_180948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180948)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_47'
        return stypy_return_type_180948

    # Assigning a type to the variable '_stypy_temp_lambda_47' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), '_stypy_temp_lambda_47', _stypy_temp_lambda_47)
    # Getting the type of '_stypy_temp_lambda_47' (line 205)
    _stypy_temp_lambda_47_180949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), '_stypy_temp_lambda_47')
    # Assigning a type to the variable 'fail_val' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'fail_val', _stypy_temp_lambda_47_180949)
    # SSA join for if statement (line 202)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def knownfail_decorator(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'knownfail_decorator'
        module_type_store = module_type_store.open_function_context('knownfail_decorator', 207, 4, False)
        
        # Passed parameters checking function
        knownfail_decorator.stypy_localization = localization
        knownfail_decorator.stypy_type_of_self = None
        knownfail_decorator.stypy_type_store = module_type_store
        knownfail_decorator.stypy_function_name = 'knownfail_decorator'
        knownfail_decorator.stypy_param_names_list = ['f']
        knownfail_decorator.stypy_varargs_param_name = None
        knownfail_decorator.stypy_kwargs_param_name = None
        knownfail_decorator.stypy_call_defaults = defaults
        knownfail_decorator.stypy_call_varargs = varargs
        knownfail_decorator.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'knownfail_decorator', ['f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'knownfail_decorator', localization, ['f'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'knownfail_decorator(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 210, 8))
        
        # 'import nose' statement (line 210)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
        import_180950 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 210, 8), 'nose')

        if (type(import_180950) is not StypyTypeError):

            if (import_180950 != 'pyd_module'):
                __import__(import_180950)
                sys_modules_180951 = sys.modules[import_180950]
                import_module(stypy.reporting.localization.Localization(__file__, 210, 8), 'nose', sys_modules_180951.module_type_store, module_type_store)
            else:
                import nose

                import_module(stypy.reporting.localization.Localization(__file__, 210, 8), 'nose', nose, module_type_store)

        else:
            # Assigning a type to the variable 'nose' (line 210)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'nose', import_180950)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 211, 8))
        
        # 'from numpy.testing.noseclasses import KnownFailureException' statement (line 211)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
        import_180952 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 211, 8), 'numpy.testing.noseclasses')

        if (type(import_180952) is not StypyTypeError):

            if (import_180952 != 'pyd_module'):
                __import__(import_180952)
                sys_modules_180953 = sys.modules[import_180952]
                import_from_module(stypy.reporting.localization.Localization(__file__, 211, 8), 'numpy.testing.noseclasses', sys_modules_180953.module_type_store, module_type_store, ['KnownFailureException'])
                nest_module(stypy.reporting.localization.Localization(__file__, 211, 8), __file__, sys_modules_180953, sys_modules_180953.module_type_store, module_type_store)
            else:
                from numpy.testing.noseclasses import KnownFailureException

                import_from_module(stypy.reporting.localization.Localization(__file__, 211, 8), 'numpy.testing.noseclasses', None, module_type_store, ['KnownFailureException'], [KnownFailureException])

        else:
            # Assigning a type to the variable 'numpy.testing.noseclasses' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'numpy.testing.noseclasses', import_180952)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
        

        @norecursion
        def knownfailer(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'knownfailer'
            module_type_store = module_type_store.open_function_context('knownfailer', 213, 8, False)
            
            # Passed parameters checking function
            knownfailer.stypy_localization = localization
            knownfailer.stypy_type_of_self = None
            knownfailer.stypy_type_store = module_type_store
            knownfailer.stypy_function_name = 'knownfailer'
            knownfailer.stypy_param_names_list = []
            knownfailer.stypy_varargs_param_name = 'args'
            knownfailer.stypy_kwargs_param_name = 'kwargs'
            knownfailer.stypy_call_defaults = defaults
            knownfailer.stypy_call_varargs = varargs
            knownfailer.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'knownfailer', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'knownfailer', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'knownfailer(...)' code ##################

            
            
            # Call to fail_val(...): (line 214)
            # Processing the call keyword arguments (line 214)
            kwargs_180955 = {}
            # Getting the type of 'fail_val' (line 214)
            fail_val_180954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'fail_val', False)
            # Calling fail_val(args, kwargs) (line 214)
            fail_val_call_result_180956 = invoke(stypy.reporting.localization.Localization(__file__, 214, 15), fail_val_180954, *[], **kwargs_180955)
            
            # Testing the type of an if condition (line 214)
            if_condition_180957 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 12), fail_val_call_result_180956)
            # Assigning a type to the variable 'if_condition_180957' (line 214)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'if_condition_180957', if_condition_180957)
            # SSA begins for if statement (line 214)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to KnownFailureException(...): (line 215)
            # Processing the call arguments (line 215)
            # Getting the type of 'msg' (line 215)
            msg_180959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 44), 'msg', False)
            # Processing the call keyword arguments (line 215)
            kwargs_180960 = {}
            # Getting the type of 'KnownFailureException' (line 215)
            KnownFailureException_180958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 22), 'KnownFailureException', False)
            # Calling KnownFailureException(args, kwargs) (line 215)
            KnownFailureException_call_result_180961 = invoke(stypy.reporting.localization.Localization(__file__, 215, 22), KnownFailureException_180958, *[msg_180959], **kwargs_180960)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 215, 16), KnownFailureException_call_result_180961, 'raise parameter', BaseException)
            # SSA branch for the else part of an if statement (line 214)
            module_type_store.open_ssa_branch('else')
            
            # Call to f(...): (line 217)
            # Getting the type of 'args' (line 217)
            args_180963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 26), 'args', False)
            # Processing the call keyword arguments (line 217)
            # Getting the type of 'kwargs' (line 217)
            kwargs_180964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 34), 'kwargs', False)
            kwargs_180965 = {'kwargs_180964': kwargs_180964}
            # Getting the type of 'f' (line 217)
            f_180962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 23), 'f', False)
            # Calling f(args, kwargs) (line 217)
            f_call_result_180966 = invoke(stypy.reporting.localization.Localization(__file__, 217, 23), f_180962, *[args_180963], **kwargs_180965)
            
            # Assigning a type to the variable 'stypy_return_type' (line 217)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'stypy_return_type', f_call_result_180966)
            # SSA join for if statement (line 214)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'knownfailer(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'knownfailer' in the type store
            # Getting the type of 'stypy_return_type' (line 213)
            stypy_return_type_180967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_180967)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'knownfailer'
            return stypy_return_type_180967

        # Assigning a type to the variable 'knownfailer' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'knownfailer', knownfailer)
        
        # Call to (...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'knownfailer' (line 218)
        knownfailer_180974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 44), 'knownfailer', False)
        # Processing the call keyword arguments (line 218)
        kwargs_180975 = {}
        
        # Call to make_decorator(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'f' (line 218)
        f_180971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 41), 'f', False)
        # Processing the call keyword arguments (line 218)
        kwargs_180972 = {}
        # Getting the type of 'nose' (line 218)
        nose_180968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'nose', False)
        # Obtaining the member 'tools' of a type (line 218)
        tools_180969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 15), nose_180968, 'tools')
        # Obtaining the member 'make_decorator' of a type (line 218)
        make_decorator_180970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 15), tools_180969, 'make_decorator')
        # Calling make_decorator(args, kwargs) (line 218)
        make_decorator_call_result_180973 = invoke(stypy.reporting.localization.Localization(__file__, 218, 15), make_decorator_180970, *[f_180971], **kwargs_180972)
        
        # Calling (args, kwargs) (line 218)
        _call_result_180976 = invoke(stypy.reporting.localization.Localization(__file__, 218, 15), make_decorator_call_result_180973, *[knownfailer_180974], **kwargs_180975)
        
        # Assigning a type to the variable 'stypy_return_type' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'stypy_return_type', _call_result_180976)
        
        # ################# End of 'knownfail_decorator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'knownfail_decorator' in the type store
        # Getting the type of 'stypy_return_type' (line 207)
        stypy_return_type_180977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_180977)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'knownfail_decorator'
        return stypy_return_type_180977

    # Assigning a type to the variable 'knownfail_decorator' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'knownfail_decorator', knownfail_decorator)
    # Getting the type of 'knownfail_decorator' (line 220)
    knownfail_decorator_180978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 11), 'knownfail_decorator')
    # Assigning a type to the variable 'stypy_return_type' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'stypy_return_type', knownfail_decorator_180978)
    
    # ################# End of 'knownfailureif(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'knownfailureif' in the type store
    # Getting the type of 'stypy_return_type' (line 168)
    stypy_return_type_180979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_180979)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'knownfailureif'
    return stypy_return_type_180979

# Assigning a type to the variable 'knownfailureif' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'knownfailureif', knownfailureif)

@norecursion
def deprecated(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 222)
    True_180980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 27), 'True')
    defaults = [True_180980]
    # Create a new context for function 'deprecated'
    module_type_store = module_type_store.open_function_context('deprecated', 222, 0, False)
    
    # Passed parameters checking function
    deprecated.stypy_localization = localization
    deprecated.stypy_type_of_self = None
    deprecated.stypy_type_store = module_type_store
    deprecated.stypy_function_name = 'deprecated'
    deprecated.stypy_param_names_list = ['conditional']
    deprecated.stypy_varargs_param_name = None
    deprecated.stypy_kwargs_param_name = None
    deprecated.stypy_call_defaults = defaults
    deprecated.stypy_call_varargs = varargs
    deprecated.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'deprecated', ['conditional'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'deprecated', localization, ['conditional'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'deprecated(...)' code ##################

    str_180981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, (-1)), 'str', "\n    Filter deprecation warnings while running the test suite.\n\n    This decorator can be used to filter DeprecationWarning's, to avoid\n    printing them during the test suite run, while checking that the test\n    actually raises a DeprecationWarning.\n\n    Parameters\n    ----------\n    conditional : bool or callable, optional\n        Flag to determine whether to mark test as deprecated or not. If the\n        condition is a callable, it is used at runtime to dynamically make the\n        decision. Default is True.\n\n    Returns\n    -------\n    decorator : function\n        The `deprecated` decorator itself.\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    ")

    @norecursion
    def deprecate_decorator(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'deprecate_decorator'
        module_type_store = module_type_store.open_function_context('deprecate_decorator', 247, 4, False)
        
        # Passed parameters checking function
        deprecate_decorator.stypy_localization = localization
        deprecate_decorator.stypy_type_of_self = None
        deprecate_decorator.stypy_type_store = module_type_store
        deprecate_decorator.stypy_function_name = 'deprecate_decorator'
        deprecate_decorator.stypy_param_names_list = ['f']
        deprecate_decorator.stypy_varargs_param_name = None
        deprecate_decorator.stypy_kwargs_param_name = None
        deprecate_decorator.stypy_call_defaults = defaults
        deprecate_decorator.stypy_call_varargs = varargs
        deprecate_decorator.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'deprecate_decorator', ['f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'deprecate_decorator', localization, ['f'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'deprecate_decorator(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 250, 8))
        
        # 'import nose' statement (line 250)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
        import_180982 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 250, 8), 'nose')

        if (type(import_180982) is not StypyTypeError):

            if (import_180982 != 'pyd_module'):
                __import__(import_180982)
                sys_modules_180983 = sys.modules[import_180982]
                import_module(stypy.reporting.localization.Localization(__file__, 250, 8), 'nose', sys_modules_180983.module_type_store, module_type_store)
            else:
                import nose

                import_module(stypy.reporting.localization.Localization(__file__, 250, 8), 'nose', nose, module_type_store)

        else:
            # Assigning a type to the variable 'nose' (line 250)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'nose', import_180982)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
        

        @norecursion
        def _deprecated_imp(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_deprecated_imp'
            module_type_store = module_type_store.open_function_context('_deprecated_imp', 252, 8, False)
            
            # Passed parameters checking function
            _deprecated_imp.stypy_localization = localization
            _deprecated_imp.stypy_type_of_self = None
            _deprecated_imp.stypy_type_store = module_type_store
            _deprecated_imp.stypy_function_name = '_deprecated_imp'
            _deprecated_imp.stypy_param_names_list = []
            _deprecated_imp.stypy_varargs_param_name = 'args'
            _deprecated_imp.stypy_kwargs_param_name = 'kwargs'
            _deprecated_imp.stypy_call_defaults = defaults
            _deprecated_imp.stypy_call_varargs = varargs
            _deprecated_imp.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_deprecated_imp', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_deprecated_imp', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_deprecated_imp(...)' code ##################

            
            # Call to catch_warnings(...): (line 254)
            # Processing the call keyword arguments (line 254)
            # Getting the type of 'True' (line 254)
            True_180986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 48), 'True', False)
            keyword_180987 = True_180986
            kwargs_180988 = {'record': keyword_180987}
            # Getting the type of 'warnings' (line 254)
            warnings_180984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 17), 'warnings', False)
            # Obtaining the member 'catch_warnings' of a type (line 254)
            catch_warnings_180985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 17), warnings_180984, 'catch_warnings')
            # Calling catch_warnings(args, kwargs) (line 254)
            catch_warnings_call_result_180989 = invoke(stypy.reporting.localization.Localization(__file__, 254, 17), catch_warnings_180985, *[], **kwargs_180988)
            
            with_180990 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 254, 17), catch_warnings_call_result_180989, 'with parameter', '__enter__', '__exit__')

            if with_180990:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 254)
                enter___180991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 17), catch_warnings_call_result_180989, '__enter__')
                with_enter_180992 = invoke(stypy.reporting.localization.Localization(__file__, 254, 17), enter___180991)
                # Assigning a type to the variable 'l' (line 254)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 17), 'l', with_enter_180992)
                
                # Call to simplefilter(...): (line 255)
                # Processing the call arguments (line 255)
                str_180995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 38), 'str', 'always')
                # Processing the call keyword arguments (line 255)
                kwargs_180996 = {}
                # Getting the type of 'warnings' (line 255)
                warnings_180993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'warnings', False)
                # Obtaining the member 'simplefilter' of a type (line 255)
                simplefilter_180994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 16), warnings_180993, 'simplefilter')
                # Calling simplefilter(args, kwargs) (line 255)
                simplefilter_call_result_180997 = invoke(stypy.reporting.localization.Localization(__file__, 255, 16), simplefilter_180994, *[str_180995], **kwargs_180996)
                
                
                # Call to f(...): (line 256)
                # Getting the type of 'args' (line 256)
                args_180999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 19), 'args', False)
                # Processing the call keyword arguments (line 256)
                # Getting the type of 'kwargs' (line 256)
                kwargs_181000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 27), 'kwargs', False)
                kwargs_181001 = {'kwargs_181000': kwargs_181000}
                # Getting the type of 'f' (line 256)
                f_180998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'f', False)
                # Calling f(args, kwargs) (line 256)
                f_call_result_181002 = invoke(stypy.reporting.localization.Localization(__file__, 256, 16), f_180998, *[args_180999], **kwargs_181001)
                
                
                
                
                
                # Call to len(...): (line 257)
                # Processing the call arguments (line 257)
                # Getting the type of 'l' (line 257)
                l_181004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 27), 'l', False)
                # Processing the call keyword arguments (line 257)
                kwargs_181005 = {}
                # Getting the type of 'len' (line 257)
                len_181003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 23), 'len', False)
                # Calling len(args, kwargs) (line 257)
                len_call_result_181006 = invoke(stypy.reporting.localization.Localization(__file__, 257, 23), len_181003, *[l_181004], **kwargs_181005)
                
                int_181007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 32), 'int')
                # Applying the binary operator '>' (line 257)
                result_gt_181008 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 23), '>', len_call_result_181006, int_181007)
                
                # Applying the 'not' unary operator (line 257)
                result_not__181009 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 19), 'not', result_gt_181008)
                
                # Testing the type of an if condition (line 257)
                if_condition_181010 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 16), result_not__181009)
                # Assigning a type to the variable 'if_condition_181010' (line 257)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'if_condition_181010', if_condition_181010)
                # SSA begins for if statement (line 257)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to AssertionError(...): (line 258)
                # Processing the call arguments (line 258)
                str_181012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 41), 'str', 'No warning raised when calling %s')
                # Getting the type of 'f' (line 259)
                f_181013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 30), 'f', False)
                # Obtaining the member '__name__' of a type (line 259)
                name___181014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 30), f_181013, '__name__')
                # Applying the binary operator '%' (line 258)
                result_mod_181015 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 41), '%', str_181012, name___181014)
                
                # Processing the call keyword arguments (line 258)
                kwargs_181016 = {}
                # Getting the type of 'AssertionError' (line 258)
                AssertionError_181011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 26), 'AssertionError', False)
                # Calling AssertionError(args, kwargs) (line 258)
                AssertionError_call_result_181017 = invoke(stypy.reporting.localization.Localization(__file__, 258, 26), AssertionError_181011, *[result_mod_181015], **kwargs_181016)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 258, 20), AssertionError_call_result_181017, 'raise parameter', BaseException)
                # SSA join for if statement (line 257)
                module_type_store = module_type_store.join_ssa_context()
                
                
                
                
                
                # Obtaining the type of the subscript
                int_181018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 25), 'int')
                # Getting the type of 'l' (line 260)
                l_181019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 23), 'l')
                # Obtaining the member '__getitem__' of a type (line 260)
                getitem___181020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 23), l_181019, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 260)
                subscript_call_result_181021 = invoke(stypy.reporting.localization.Localization(__file__, 260, 23), getitem___181020, int_181018)
                
                # Obtaining the member 'category' of a type (line 260)
                category_181022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 23), subscript_call_result_181021, 'category')
                # Getting the type of 'DeprecationWarning' (line 260)
                DeprecationWarning_181023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 40), 'DeprecationWarning')
                # Applying the binary operator 'is' (line 260)
                result_is__181024 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 23), 'is', category_181022, DeprecationWarning_181023)
                
                # Applying the 'not' unary operator (line 260)
                result_not__181025 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 19), 'not', result_is__181024)
                
                # Testing the type of an if condition (line 260)
                if_condition_181026 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 16), result_not__181025)
                # Assigning a type to the variable 'if_condition_181026' (line 260)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'if_condition_181026', if_condition_181026)
                # SSA begins for if statement (line 260)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to AssertionError(...): (line 261)
                # Processing the call arguments (line 261)
                str_181028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 41), 'str', 'First warning for %s is not a DeprecationWarning( is %s)')
                
                # Obtaining an instance of the builtin type 'tuple' (line 262)
                tuple_181029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 60), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 262)
                # Adding element type (line 262)
                # Getting the type of 'f' (line 262)
                f_181030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 60), 'f', False)
                # Obtaining the member '__name__' of a type (line 262)
                name___181031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 60), f_181030, '__name__')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 60), tuple_181029, name___181031)
                # Adding element type (line 262)
                
                # Obtaining the type of the subscript
                int_181032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 74), 'int')
                # Getting the type of 'l' (line 262)
                l_181033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 72), 'l', False)
                # Obtaining the member '__getitem__' of a type (line 262)
                getitem___181034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 72), l_181033, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 262)
                subscript_call_result_181035 = invoke(stypy.reporting.localization.Localization(__file__, 262, 72), getitem___181034, int_181032)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 60), tuple_181029, subscript_call_result_181035)
                
                # Applying the binary operator '%' (line 261)
                result_mod_181036 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 41), '%', str_181028, tuple_181029)
                
                # Processing the call keyword arguments (line 261)
                kwargs_181037 = {}
                # Getting the type of 'AssertionError' (line 261)
                AssertionError_181027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 26), 'AssertionError', False)
                # Calling AssertionError(args, kwargs) (line 261)
                AssertionError_call_result_181038 = invoke(stypy.reporting.localization.Localization(__file__, 261, 26), AssertionError_181027, *[result_mod_181036], **kwargs_181037)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 261, 20), AssertionError_call_result_181038, 'raise parameter', BaseException)
                # SSA join for if statement (line 260)
                module_type_store = module_type_store.join_ssa_context()
                
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 254)
                exit___181039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 17), catch_warnings_call_result_180989, '__exit__')
                with_exit_181040 = invoke(stypy.reporting.localization.Localization(__file__, 254, 17), exit___181039, None, None, None)

            
            # ################# End of '_deprecated_imp(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_deprecated_imp' in the type store
            # Getting the type of 'stypy_return_type' (line 252)
            stypy_return_type_181041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_181041)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_deprecated_imp'
            return stypy_return_type_181041

        # Assigning a type to the variable '_deprecated_imp' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), '_deprecated_imp', _deprecated_imp)
        
        
        # Call to isinstance(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'conditional' (line 264)
        conditional_181043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 22), 'conditional', False)
        # Getting the type of 'collections' (line 264)
        collections_181044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 35), 'collections', False)
        # Obtaining the member 'Callable' of a type (line 264)
        Callable_181045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 35), collections_181044, 'Callable')
        # Processing the call keyword arguments (line 264)
        kwargs_181046 = {}
        # Getting the type of 'isinstance' (line 264)
        isinstance_181042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 264)
        isinstance_call_result_181047 = invoke(stypy.reporting.localization.Localization(__file__, 264, 11), isinstance_181042, *[conditional_181043, Callable_181045], **kwargs_181046)
        
        # Testing the type of an if condition (line 264)
        if_condition_181048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 264, 8), isinstance_call_result_181047)
        # Assigning a type to the variable 'if_condition_181048' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'if_condition_181048', if_condition_181048)
        # SSA begins for if statement (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 265):
        
        # Call to conditional(...): (line 265)
        # Processing the call keyword arguments (line 265)
        kwargs_181050 = {}
        # Getting the type of 'conditional' (line 265)
        conditional_181049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 19), 'conditional', False)
        # Calling conditional(args, kwargs) (line 265)
        conditional_call_result_181051 = invoke(stypy.reporting.localization.Localization(__file__, 265, 19), conditional_181049, *[], **kwargs_181050)
        
        # Assigning a type to the variable 'cond' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'cond', conditional_call_result_181051)
        # SSA branch for the else part of an if statement (line 264)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 267):
        # Getting the type of 'conditional' (line 267)
        conditional_181052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 19), 'conditional')
        # Assigning a type to the variable 'cond' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'cond', conditional_181052)
        # SSA join for if statement (line 264)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'cond' (line 268)
        cond_181053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 11), 'cond')
        # Testing the type of an if condition (line 268)
        if_condition_181054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 8), cond_181053)
        # Assigning a type to the variable 'if_condition_181054' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'if_condition_181054', if_condition_181054)
        # SSA begins for if statement (line 268)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to (...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of '_deprecated_imp' (line 269)
        _deprecated_imp_181061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 48), '_deprecated_imp', False)
        # Processing the call keyword arguments (line 269)
        kwargs_181062 = {}
        
        # Call to make_decorator(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'f' (line 269)
        f_181058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 45), 'f', False)
        # Processing the call keyword arguments (line 269)
        kwargs_181059 = {}
        # Getting the type of 'nose' (line 269)
        nose_181055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 19), 'nose', False)
        # Obtaining the member 'tools' of a type (line 269)
        tools_181056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 19), nose_181055, 'tools')
        # Obtaining the member 'make_decorator' of a type (line 269)
        make_decorator_181057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 19), tools_181056, 'make_decorator')
        # Calling make_decorator(args, kwargs) (line 269)
        make_decorator_call_result_181060 = invoke(stypy.reporting.localization.Localization(__file__, 269, 19), make_decorator_181057, *[f_181058], **kwargs_181059)
        
        # Calling (args, kwargs) (line 269)
        _call_result_181063 = invoke(stypy.reporting.localization.Localization(__file__, 269, 19), make_decorator_call_result_181060, *[_deprecated_imp_181061], **kwargs_181062)
        
        # Assigning a type to the variable 'stypy_return_type' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'stypy_return_type', _call_result_181063)
        # SSA branch for the else part of an if statement (line 268)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'f' (line 271)
        f_181064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 19), 'f')
        # Assigning a type to the variable 'stypy_return_type' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'stypy_return_type', f_181064)
        # SSA join for if statement (line 268)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'deprecate_decorator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'deprecate_decorator' in the type store
        # Getting the type of 'stypy_return_type' (line 247)
        stypy_return_type_181065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181065)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'deprecate_decorator'
        return stypy_return_type_181065

    # Assigning a type to the variable 'deprecate_decorator' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'deprecate_decorator', deprecate_decorator)
    # Getting the type of 'deprecate_decorator' (line 272)
    deprecate_decorator_181066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'deprecate_decorator')
    # Assigning a type to the variable 'stypy_return_type' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type', deprecate_decorator_181066)
    
    # ################# End of 'deprecated(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'deprecated' in the type store
    # Getting the type of 'stypy_return_type' (line 222)
    stypy_return_type_181067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_181067)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'deprecated'
    return stypy_return_type_181067

# Assigning a type to the variable 'deprecated' (line 222)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 0), 'deprecated', deprecated)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
