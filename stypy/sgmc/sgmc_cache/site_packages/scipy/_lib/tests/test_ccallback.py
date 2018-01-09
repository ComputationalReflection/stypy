
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from numpy.testing import assert_equal, assert_
4: from pytest import raises as assert_raises
5: 
6: import time
7: import pytest
8: import ctypes
9: import threading
10: from scipy._lib import _ccallback_c as _test_ccallback_cython
11: from scipy._lib import _test_ccallback
12: from scipy._lib._ccallback import LowLevelCallable
13: 
14: try:
15:     import cffi
16:     HAVE_CFFI = True
17: except ImportError:
18:     HAVE_CFFI = False
19: 
20: 
21: ERROR_VALUE = 2.0
22: 
23: 
24: def callback_python(a, user_data=None):
25:     if a == ERROR_VALUE:
26:         raise ValueError("bad value")
27: 
28:     if user_data is None:
29:         return a + 1
30:     else:
31:         return a + user_data
32: 
33: def _get_cffi_func(base, signature):
34:     if not HAVE_CFFI:
35:         pytest.skip("cffi not installed")
36: 
37:     # Get function address
38:     voidp = ctypes.cast(base, ctypes.c_void_p)
39:     address = voidp.value
40: 
41:     # Create corresponding cffi handle
42:     ffi = cffi.FFI()
43:     func = ffi.cast(signature, address)
44:     return func
45: 
46: 
47: def _get_ctypes_data():
48:     value = ctypes.c_double(2.0)
49:     return ctypes.cast(ctypes.pointer(value), ctypes.c_voidp)
50: 
51: 
52: def _get_cffi_data():
53:     if not HAVE_CFFI:
54:         pytest.skip("cffi not installed")
55:     ffi = cffi.FFI()
56:     return ffi.new('double *', 2.0)
57: 
58: 
59: CALLERS = {
60:     'simple': _test_ccallback.test_call_simple,
61:     'nodata': _test_ccallback.test_call_nodata,
62:     'nonlocal': _test_ccallback.test_call_nonlocal,
63:     'cython': _test_ccallback_cython.test_call_cython,
64: }
65: 
66: # These functions have signatures known to the callers
67: FUNCS = {
68:     'python': lambda: callback_python,
69:     'capsule': lambda: _test_ccallback.test_get_plus1_capsule(),
70:     'cython': lambda: LowLevelCallable.from_cython(_test_ccallback_cython, "plus1_cython"),
71:     'ctypes': lambda: _test_ccallback_cython.plus1_ctypes,
72:     'cffi': lambda: _get_cffi_func(_test_ccallback_cython.plus1_ctypes,
73:                                    'double (*)(double, int *, void *)'),
74:     'capsule_b': lambda: _test_ccallback.test_get_plus1b_capsule(),
75:     'cython_b': lambda: LowLevelCallable.from_cython(_test_ccallback_cython, "plus1b_cython"),
76:     'ctypes_b': lambda: _test_ccallback_cython.plus1b_ctypes,
77:     'cffi_b': lambda: _get_cffi_func(_test_ccallback_cython.plus1b_ctypes,
78:                                      'double (*)(double, double, int *, void *)'),
79: }
80: 
81: # These functions have signatures the callers don't know
82: BAD_FUNCS = {
83:     'capsule_bc': lambda: _test_ccallback.test_get_plus1bc_capsule(),
84:     'cython_bc': lambda: LowLevelCallable.from_cython(_test_ccallback_cython, "plus1bc_cython"),
85:     'ctypes_bc': lambda: _test_ccallback_cython.plus1bc_ctypes,
86:     'cffi_bc': lambda: _get_cffi_func(_test_ccallback_cython.plus1bc_ctypes,
87:                                       'double (*)(double, double, double, int *, void *)'),
88: }
89: 
90: USER_DATAS = {
91:     'ctypes': _get_ctypes_data,
92:     'cffi': _get_cffi_data,
93:     'capsule': _test_ccallback.test_get_data_capsule,
94: }
95: 
96: 
97: def test_callbacks():
98:     def check(caller, func, user_data):
99:         caller = CALLERS[caller]
100:         func = FUNCS[func]()
101:         user_data = USER_DATAS[user_data]()
102: 
103:         if func is callback_python:
104:             func2 = lambda x: func(x, 2.0)
105:         else:
106:             func2 = LowLevelCallable(func, user_data)
107:             func = LowLevelCallable(func)
108: 
109:         # Test basic call
110:         assert_equal(caller(func, 1.0), 2.0)
111: 
112:         # Test 'bad' value resulting to an error
113:         assert_raises(ValueError, caller, func, ERROR_VALUE)
114: 
115:         # Test passing in user_data
116:         assert_equal(caller(func2, 1.0), 3.0)
117: 
118:     for caller in sorted(CALLERS.keys()):
119:         for func in sorted(FUNCS.keys()):
120:             for user_data in sorted(USER_DATAS.keys()):
121:                 check(caller, func, user_data)
122: 
123: 
124: def test_bad_callbacks():
125:     def check(caller, func, user_data):
126:         caller = CALLERS[caller]
127:         user_data = USER_DATAS[user_data]()
128:         func = BAD_FUNCS[func]()
129: 
130:         if func is callback_python:
131:             func2 = lambda x: func(x, 2.0)
132:         else:
133:             func2 = LowLevelCallable(func, user_data)
134:             func = LowLevelCallable(func)
135: 
136:         # Test that basic call fails
137:         assert_raises(ValueError, caller, LowLevelCallable(func), 1.0)
138: 
139:         # Test that passing in user_data also fails
140:         assert_raises(ValueError, caller, func2, 1.0)
141: 
142:         # Test error message
143:         llfunc = LowLevelCallable(func)
144:         try:
145:             caller(llfunc, 1.0)
146:         except ValueError as err:
147:             msg = str(err)
148:             assert_(llfunc.signature in msg, msg)
149:             assert_('double (double, double, int *, void *)' in msg, msg)
150: 
151:     for caller in sorted(CALLERS.keys()):
152:         for func in sorted(BAD_FUNCS.keys()):
153:             for user_data in sorted(USER_DATAS.keys()):
154:                 check(caller, func, user_data)
155: 
156: 
157: def test_signature_override():
158:     caller = _test_ccallback.test_call_simple
159:     func = _test_ccallback.test_get_plus1_capsule()
160: 
161:     llcallable = LowLevelCallable(func, signature="bad signature")
162:     assert_equal(llcallable.signature, "bad signature")
163:     assert_raises(ValueError, caller, llcallable, 3)
164: 
165:     llcallable = LowLevelCallable(func, signature="double (double, int *, void *)")
166:     assert_equal(llcallable.signature, "double (double, int *, void *)")
167:     assert_equal(caller(llcallable, 3), 4)
168: 
169: 
170: def test_threadsafety():
171:     def callback(a, caller):
172:         if a <= 0:
173:             return 1
174:         else:
175:             res = caller(lambda x: callback(x, caller), a - 1)
176:             return 2*res
177: 
178:     def check(caller):
179:         caller = CALLERS[caller]
180: 
181:         results = []
182: 
183:         count = 10
184: 
185:         def run():
186:             time.sleep(0.01)
187:             r = caller(lambda x: callback(x, caller), count)
188:             results.append(r)
189: 
190:         threads = [threading.Thread(target=run) for j in range(20)]
191:         for thread in threads:
192:             thread.start()
193:         for thread in threads:
194:             thread.join()
195: 
196:         assert_equal(results, [2.0**count]*len(threads))
197: 
198:     for caller in CALLERS.keys():
199:         check(caller)
200: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.testing import assert_equal, assert_' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_711098 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing')

if (type(import_711098) is not StypyTypeError):

    if (import_711098 != 'pyd_module'):
        __import__(import_711098)
        sys_modules_711099 = sys.modules[import_711098]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', sys_modules_711099.module_type_store, module_type_store, ['assert_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_711099, sys_modules_711099.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_'], [assert_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', import_711098)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from pytest import assert_raises' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_711100 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest')

if (type(import_711100) is not StypyTypeError):

    if (import_711100 != 'pyd_module'):
        __import__(import_711100)
        sys_modules_711101 = sys.modules[import_711100]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', sys_modules_711101.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_711101, sys_modules_711101.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', import_711100)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import time' statement (line 6)
import time

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import pytest' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_711102 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest')

if (type(import_711102) is not StypyTypeError):

    if (import_711102 != 'pyd_module'):
        __import__(import_711102)
        sys_modules_711103 = sys.modules[import_711102]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', sys_modules_711103.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', import_711102)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import ctypes' statement (line 8)
import ctypes

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'ctypes', ctypes, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import threading' statement (line 9)
import threading

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'threading', threading, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib import _test_ccallback_cython' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_711104 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib')

if (type(import_711104) is not StypyTypeError):

    if (import_711104 != 'pyd_module'):
        __import__(import_711104)
        sys_modules_711105 = sys.modules[import_711104]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib', sys_modules_711105.module_type_store, module_type_store, ['_ccallback_c'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_711105, sys_modules_711105.module_type_store, module_type_store)
    else:
        from scipy._lib import _ccallback_c as _test_ccallback_cython

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib', None, module_type_store, ['_ccallback_c'], [_test_ccallback_cython])

else:
    # Assigning a type to the variable 'scipy._lib' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib', import_711104)

# Adding an alias
module_type_store.add_alias('_test_ccallback_cython', '_ccallback_c')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib import _test_ccallback' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_711106 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib')

if (type(import_711106) is not StypyTypeError):

    if (import_711106 != 'pyd_module'):
        __import__(import_711106)
        sys_modules_711107 = sys.modules[import_711106]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib', sys_modules_711107.module_type_store, module_type_store, ['_test_ccallback'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_711107, sys_modules_711107.module_type_store, module_type_store)
    else:
        from scipy._lib import _test_ccallback

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib', None, module_type_store, ['_test_ccallback'], [_test_ccallback])

else:
    # Assigning a type to the variable 'scipy._lib' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib', import_711106)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy._lib._ccallback import LowLevelCallable' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_711108 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib._ccallback')

if (type(import_711108) is not StypyTypeError):

    if (import_711108 != 'pyd_module'):
        __import__(import_711108)
        sys_modules_711109 = sys.modules[import_711108]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib._ccallback', sys_modules_711109.module_type_store, module_type_store, ['LowLevelCallable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_711109, sys_modules_711109.module_type_store, module_type_store)
    else:
        from scipy._lib._ccallback import LowLevelCallable

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib._ccallback', None, module_type_store, ['LowLevelCallable'], [LowLevelCallable])

else:
    # Assigning a type to the variable 'scipy._lib._ccallback' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib._ccallback', import_711108)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')



# SSA begins for try-except statement (line 14)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 4))

# 'import cffi' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_711110 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'cffi')

if (type(import_711110) is not StypyTypeError):

    if (import_711110 != 'pyd_module'):
        __import__(import_711110)
        sys_modules_711111 = sys.modules[import_711110]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'cffi', sys_modules_711111.module_type_store, module_type_store)
    else:
        import cffi

        import_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'cffi', cffi, module_type_store)

else:
    # Assigning a type to the variable 'cffi' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'cffi', import_711110)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')


# Assigning a Name to a Name (line 16):
# Getting the type of 'True' (line 16)
True_711112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'True')
# Assigning a type to the variable 'HAVE_CFFI' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'HAVE_CFFI', True_711112)
# SSA branch for the except part of a try statement (line 14)
# SSA branch for the except 'ImportError' branch of a try statement (line 14)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 18):
# Getting the type of 'False' (line 18)
False_711113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'False')
# Assigning a type to the variable 'HAVE_CFFI' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'HAVE_CFFI', False_711113)
# SSA join for try-except statement (line 14)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Num to a Name (line 21):
float_711114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 14), 'float')
# Assigning a type to the variable 'ERROR_VALUE' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'ERROR_VALUE', float_711114)

@norecursion
def callback_python(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 24)
    None_711115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 33), 'None')
    defaults = [None_711115]
    # Create a new context for function 'callback_python'
    module_type_store = module_type_store.open_function_context('callback_python', 24, 0, False)
    
    # Passed parameters checking function
    callback_python.stypy_localization = localization
    callback_python.stypy_type_of_self = None
    callback_python.stypy_type_store = module_type_store
    callback_python.stypy_function_name = 'callback_python'
    callback_python.stypy_param_names_list = ['a', 'user_data']
    callback_python.stypy_varargs_param_name = None
    callback_python.stypy_kwargs_param_name = None
    callback_python.stypy_call_defaults = defaults
    callback_python.stypy_call_varargs = varargs
    callback_python.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'callback_python', ['a', 'user_data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'callback_python', localization, ['a', 'user_data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'callback_python(...)' code ##################

    
    
    # Getting the type of 'a' (line 25)
    a_711116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 7), 'a')
    # Getting the type of 'ERROR_VALUE' (line 25)
    ERROR_VALUE_711117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'ERROR_VALUE')
    # Applying the binary operator '==' (line 25)
    result_eq_711118 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 7), '==', a_711116, ERROR_VALUE_711117)
    
    # Testing the type of an if condition (line 25)
    if_condition_711119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 4), result_eq_711118)
    # Assigning a type to the variable 'if_condition_711119' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'if_condition_711119', if_condition_711119)
    # SSA begins for if statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 26)
    # Processing the call arguments (line 26)
    str_711121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'str', 'bad value')
    # Processing the call keyword arguments (line 26)
    kwargs_711122 = {}
    # Getting the type of 'ValueError' (line 26)
    ValueError_711120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 26)
    ValueError_call_result_711123 = invoke(stypy.reporting.localization.Localization(__file__, 26, 14), ValueError_711120, *[str_711121], **kwargs_711122)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 26, 8), ValueError_call_result_711123, 'raise parameter', BaseException)
    # SSA join for if statement (line 25)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 28)
    # Getting the type of 'user_data' (line 28)
    user_data_711124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 7), 'user_data')
    # Getting the type of 'None' (line 28)
    None_711125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'None')
    
    (may_be_711126, more_types_in_union_711127) = may_be_none(user_data_711124, None_711125)

    if may_be_711126:

        if more_types_in_union_711127:
            # Runtime conditional SSA (line 28)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 29)
        a_711128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'a')
        int_711129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'int')
        # Applying the binary operator '+' (line 29)
        result_add_711130 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 15), '+', a_711128, int_711129)
        
        # Assigning a type to the variable 'stypy_return_type' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', result_add_711130)

        if more_types_in_union_711127:
            # Runtime conditional SSA for else branch (line 28)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_711126) or more_types_in_union_711127):
        # Getting the type of 'a' (line 31)
        a_711131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'a')
        # Getting the type of 'user_data' (line 31)
        user_data_711132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'user_data')
        # Applying the binary operator '+' (line 31)
        result_add_711133 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), '+', a_711131, user_data_711132)
        
        # Assigning a type to the variable 'stypy_return_type' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', result_add_711133)

        if (may_be_711126 and more_types_in_union_711127):
            # SSA join for if statement (line 28)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'callback_python(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'callback_python' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_711134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711134)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'callback_python'
    return stypy_return_type_711134

# Assigning a type to the variable 'callback_python' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'callback_python', callback_python)

@norecursion
def _get_cffi_func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_cffi_func'
    module_type_store = module_type_store.open_function_context('_get_cffi_func', 33, 0, False)
    
    # Passed parameters checking function
    _get_cffi_func.stypy_localization = localization
    _get_cffi_func.stypy_type_of_self = None
    _get_cffi_func.stypy_type_store = module_type_store
    _get_cffi_func.stypy_function_name = '_get_cffi_func'
    _get_cffi_func.stypy_param_names_list = ['base', 'signature']
    _get_cffi_func.stypy_varargs_param_name = None
    _get_cffi_func.stypy_kwargs_param_name = None
    _get_cffi_func.stypy_call_defaults = defaults
    _get_cffi_func.stypy_call_varargs = varargs
    _get_cffi_func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_cffi_func', ['base', 'signature'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_cffi_func', localization, ['base', 'signature'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_cffi_func(...)' code ##################

    
    
    # Getting the type of 'HAVE_CFFI' (line 34)
    HAVE_CFFI_711135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'HAVE_CFFI')
    # Applying the 'not' unary operator (line 34)
    result_not__711136 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 7), 'not', HAVE_CFFI_711135)
    
    # Testing the type of an if condition (line 34)
    if_condition_711137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 4), result_not__711136)
    # Assigning a type to the variable 'if_condition_711137' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'if_condition_711137', if_condition_711137)
    # SSA begins for if statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to skip(...): (line 35)
    # Processing the call arguments (line 35)
    str_711140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 20), 'str', 'cffi not installed')
    # Processing the call keyword arguments (line 35)
    kwargs_711141 = {}
    # Getting the type of 'pytest' (line 35)
    pytest_711138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'pytest', False)
    # Obtaining the member 'skip' of a type (line 35)
    skip_711139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), pytest_711138, 'skip')
    # Calling skip(args, kwargs) (line 35)
    skip_call_result_711142 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), skip_711139, *[str_711140], **kwargs_711141)
    
    # SSA join for if statement (line 34)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 38):
    
    # Call to cast(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'base' (line 38)
    base_711145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'base', False)
    # Getting the type of 'ctypes' (line 38)
    ctypes_711146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 30), 'ctypes', False)
    # Obtaining the member 'c_void_p' of a type (line 38)
    c_void_p_711147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 30), ctypes_711146, 'c_void_p')
    # Processing the call keyword arguments (line 38)
    kwargs_711148 = {}
    # Getting the type of 'ctypes' (line 38)
    ctypes_711143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'ctypes', False)
    # Obtaining the member 'cast' of a type (line 38)
    cast_711144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), ctypes_711143, 'cast')
    # Calling cast(args, kwargs) (line 38)
    cast_call_result_711149 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), cast_711144, *[base_711145, c_void_p_711147], **kwargs_711148)
    
    # Assigning a type to the variable 'voidp' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'voidp', cast_call_result_711149)
    
    # Assigning a Attribute to a Name (line 39):
    # Getting the type of 'voidp' (line 39)
    voidp_711150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'voidp')
    # Obtaining the member 'value' of a type (line 39)
    value_711151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 14), voidp_711150, 'value')
    # Assigning a type to the variable 'address' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'address', value_711151)
    
    # Assigning a Call to a Name (line 42):
    
    # Call to FFI(...): (line 42)
    # Processing the call keyword arguments (line 42)
    kwargs_711154 = {}
    # Getting the type of 'cffi' (line 42)
    cffi_711152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 10), 'cffi', False)
    # Obtaining the member 'FFI' of a type (line 42)
    FFI_711153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 10), cffi_711152, 'FFI')
    # Calling FFI(args, kwargs) (line 42)
    FFI_call_result_711155 = invoke(stypy.reporting.localization.Localization(__file__, 42, 10), FFI_711153, *[], **kwargs_711154)
    
    # Assigning a type to the variable 'ffi' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'ffi', FFI_call_result_711155)
    
    # Assigning a Call to a Name (line 43):
    
    # Call to cast(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'signature' (line 43)
    signature_711158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'signature', False)
    # Getting the type of 'address' (line 43)
    address_711159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 31), 'address', False)
    # Processing the call keyword arguments (line 43)
    kwargs_711160 = {}
    # Getting the type of 'ffi' (line 43)
    ffi_711156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'ffi', False)
    # Obtaining the member 'cast' of a type (line 43)
    cast_711157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 11), ffi_711156, 'cast')
    # Calling cast(args, kwargs) (line 43)
    cast_call_result_711161 = invoke(stypy.reporting.localization.Localization(__file__, 43, 11), cast_711157, *[signature_711158, address_711159], **kwargs_711160)
    
    # Assigning a type to the variable 'func' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'func', cast_call_result_711161)
    # Getting the type of 'func' (line 44)
    func_711162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'func')
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type', func_711162)
    
    # ################# End of '_get_cffi_func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_cffi_func' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_711163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711163)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_cffi_func'
    return stypy_return_type_711163

# Assigning a type to the variable '_get_cffi_func' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), '_get_cffi_func', _get_cffi_func)

@norecursion
def _get_ctypes_data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_ctypes_data'
    module_type_store = module_type_store.open_function_context('_get_ctypes_data', 47, 0, False)
    
    # Passed parameters checking function
    _get_ctypes_data.stypy_localization = localization
    _get_ctypes_data.stypy_type_of_self = None
    _get_ctypes_data.stypy_type_store = module_type_store
    _get_ctypes_data.stypy_function_name = '_get_ctypes_data'
    _get_ctypes_data.stypy_param_names_list = []
    _get_ctypes_data.stypy_varargs_param_name = None
    _get_ctypes_data.stypy_kwargs_param_name = None
    _get_ctypes_data.stypy_call_defaults = defaults
    _get_ctypes_data.stypy_call_varargs = varargs
    _get_ctypes_data.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_ctypes_data', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_ctypes_data', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_ctypes_data(...)' code ##################

    
    # Assigning a Call to a Name (line 48):
    
    # Call to c_double(...): (line 48)
    # Processing the call arguments (line 48)
    float_711166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 28), 'float')
    # Processing the call keyword arguments (line 48)
    kwargs_711167 = {}
    # Getting the type of 'ctypes' (line 48)
    ctypes_711164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'ctypes', False)
    # Obtaining the member 'c_double' of a type (line 48)
    c_double_711165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), ctypes_711164, 'c_double')
    # Calling c_double(args, kwargs) (line 48)
    c_double_call_result_711168 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), c_double_711165, *[float_711166], **kwargs_711167)
    
    # Assigning a type to the variable 'value' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'value', c_double_call_result_711168)
    
    # Call to cast(...): (line 49)
    # Processing the call arguments (line 49)
    
    # Call to pointer(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'value' (line 49)
    value_711173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'value', False)
    # Processing the call keyword arguments (line 49)
    kwargs_711174 = {}
    # Getting the type of 'ctypes' (line 49)
    ctypes_711171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'ctypes', False)
    # Obtaining the member 'pointer' of a type (line 49)
    pointer_711172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 23), ctypes_711171, 'pointer')
    # Calling pointer(args, kwargs) (line 49)
    pointer_call_result_711175 = invoke(stypy.reporting.localization.Localization(__file__, 49, 23), pointer_711172, *[value_711173], **kwargs_711174)
    
    # Getting the type of 'ctypes' (line 49)
    ctypes_711176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 46), 'ctypes', False)
    # Obtaining the member 'c_voidp' of a type (line 49)
    c_voidp_711177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 46), ctypes_711176, 'c_voidp')
    # Processing the call keyword arguments (line 49)
    kwargs_711178 = {}
    # Getting the type of 'ctypes' (line 49)
    ctypes_711169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'ctypes', False)
    # Obtaining the member 'cast' of a type (line 49)
    cast_711170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 11), ctypes_711169, 'cast')
    # Calling cast(args, kwargs) (line 49)
    cast_call_result_711179 = invoke(stypy.reporting.localization.Localization(__file__, 49, 11), cast_711170, *[pointer_call_result_711175, c_voidp_711177], **kwargs_711178)
    
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type', cast_call_result_711179)
    
    # ################# End of '_get_ctypes_data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_ctypes_data' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_711180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711180)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_ctypes_data'
    return stypy_return_type_711180

# Assigning a type to the variable '_get_ctypes_data' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), '_get_ctypes_data', _get_ctypes_data)

@norecursion
def _get_cffi_data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_cffi_data'
    module_type_store = module_type_store.open_function_context('_get_cffi_data', 52, 0, False)
    
    # Passed parameters checking function
    _get_cffi_data.stypy_localization = localization
    _get_cffi_data.stypy_type_of_self = None
    _get_cffi_data.stypy_type_store = module_type_store
    _get_cffi_data.stypy_function_name = '_get_cffi_data'
    _get_cffi_data.stypy_param_names_list = []
    _get_cffi_data.stypy_varargs_param_name = None
    _get_cffi_data.stypy_kwargs_param_name = None
    _get_cffi_data.stypy_call_defaults = defaults
    _get_cffi_data.stypy_call_varargs = varargs
    _get_cffi_data.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_cffi_data', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_cffi_data', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_cffi_data(...)' code ##################

    
    
    # Getting the type of 'HAVE_CFFI' (line 53)
    HAVE_CFFI_711181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'HAVE_CFFI')
    # Applying the 'not' unary operator (line 53)
    result_not__711182 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 7), 'not', HAVE_CFFI_711181)
    
    # Testing the type of an if condition (line 53)
    if_condition_711183 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 4), result_not__711182)
    # Assigning a type to the variable 'if_condition_711183' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'if_condition_711183', if_condition_711183)
    # SSA begins for if statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to skip(...): (line 54)
    # Processing the call arguments (line 54)
    str_711186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 20), 'str', 'cffi not installed')
    # Processing the call keyword arguments (line 54)
    kwargs_711187 = {}
    # Getting the type of 'pytest' (line 54)
    pytest_711184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'pytest', False)
    # Obtaining the member 'skip' of a type (line 54)
    skip_711185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), pytest_711184, 'skip')
    # Calling skip(args, kwargs) (line 54)
    skip_call_result_711188 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), skip_711185, *[str_711186], **kwargs_711187)
    
    # SSA join for if statement (line 53)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 55):
    
    # Call to FFI(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_711191 = {}
    # Getting the type of 'cffi' (line 55)
    cffi_711189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 10), 'cffi', False)
    # Obtaining the member 'FFI' of a type (line 55)
    FFI_711190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 10), cffi_711189, 'FFI')
    # Calling FFI(args, kwargs) (line 55)
    FFI_call_result_711192 = invoke(stypy.reporting.localization.Localization(__file__, 55, 10), FFI_711190, *[], **kwargs_711191)
    
    # Assigning a type to the variable 'ffi' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'ffi', FFI_call_result_711192)
    
    # Call to new(...): (line 56)
    # Processing the call arguments (line 56)
    str_711195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 19), 'str', 'double *')
    float_711196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 31), 'float')
    # Processing the call keyword arguments (line 56)
    kwargs_711197 = {}
    # Getting the type of 'ffi' (line 56)
    ffi_711193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'ffi', False)
    # Obtaining the member 'new' of a type (line 56)
    new_711194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), ffi_711193, 'new')
    # Calling new(args, kwargs) (line 56)
    new_call_result_711198 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), new_711194, *[str_711195, float_711196], **kwargs_711197)
    
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type', new_call_result_711198)
    
    # ################# End of '_get_cffi_data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_cffi_data' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_711199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711199)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_cffi_data'
    return stypy_return_type_711199

# Assigning a type to the variable '_get_cffi_data' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), '_get_cffi_data', _get_cffi_data)

# Assigning a Dict to a Name (line 59):

# Obtaining an instance of the builtin type 'dict' (line 59)
dict_711200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 59)
# Adding element type (key, value) (line 59)
str_711201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 4), 'str', 'simple')
# Getting the type of '_test_ccallback' (line 60)
_test_ccallback_711202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), '_test_ccallback')
# Obtaining the member 'test_call_simple' of a type (line 60)
test_call_simple_711203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 14), _test_ccallback_711202, 'test_call_simple')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), dict_711200, (str_711201, test_call_simple_711203))
# Adding element type (key, value) (line 59)
str_711204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'str', 'nodata')
# Getting the type of '_test_ccallback' (line 61)
_test_ccallback_711205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), '_test_ccallback')
# Obtaining the member 'test_call_nodata' of a type (line 61)
test_call_nodata_711206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 14), _test_ccallback_711205, 'test_call_nodata')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), dict_711200, (str_711204, test_call_nodata_711206))
# Adding element type (key, value) (line 59)
str_711207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 4), 'str', 'nonlocal')
# Getting the type of '_test_ccallback' (line 62)
_test_ccallback_711208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), '_test_ccallback')
# Obtaining the member 'test_call_nonlocal' of a type (line 62)
test_call_nonlocal_711209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), _test_ccallback_711208, 'test_call_nonlocal')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), dict_711200, (str_711207, test_call_nonlocal_711209))
# Adding element type (key, value) (line 59)
str_711210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 4), 'str', 'cython')
# Getting the type of '_test_ccallback_cython' (line 63)
_test_ccallback_cython_711211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 14), '_test_ccallback_cython')
# Obtaining the member 'test_call_cython' of a type (line 63)
test_call_cython_711212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 14), _test_ccallback_cython_711211, 'test_call_cython')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), dict_711200, (str_711210, test_call_cython_711212))

# Assigning a type to the variable 'CALLERS' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'CALLERS', dict_711200)

# Assigning a Dict to a Name (line 67):

# Obtaining an instance of the builtin type 'dict' (line 67)
dict_711213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 8), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 67)
# Adding element type (key, value) (line 67)
str_711214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'str', 'python')

@norecursion
def _stypy_temp_lambda_577(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_577'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_577', 68, 14, True)
    # Passed parameters checking function
    _stypy_temp_lambda_577.stypy_localization = localization
    _stypy_temp_lambda_577.stypy_type_of_self = None
    _stypy_temp_lambda_577.stypy_type_store = module_type_store
    _stypy_temp_lambda_577.stypy_function_name = '_stypy_temp_lambda_577'
    _stypy_temp_lambda_577.stypy_param_names_list = []
    _stypy_temp_lambda_577.stypy_varargs_param_name = None
    _stypy_temp_lambda_577.stypy_kwargs_param_name = None
    _stypy_temp_lambda_577.stypy_call_defaults = defaults
    _stypy_temp_lambda_577.stypy_call_varargs = varargs
    _stypy_temp_lambda_577.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_577', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_577', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'callback_python' (line 68)
    callback_python_711215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 22), 'callback_python')
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), 'stypy_return_type', callback_python_711215)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_577' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_711216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711216)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_577'
    return stypy_return_type_711216

# Assigning a type to the variable '_stypy_temp_lambda_577' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), '_stypy_temp_lambda_577', _stypy_temp_lambda_577)
# Getting the type of '_stypy_temp_lambda_577' (line 68)
_stypy_temp_lambda_577_711217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), '_stypy_temp_lambda_577')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), dict_711213, (str_711214, _stypy_temp_lambda_577_711217))
# Adding element type (key, value) (line 67)
str_711218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 4), 'str', 'capsule')

@norecursion
def _stypy_temp_lambda_578(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_578'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_578', 69, 15, True)
    # Passed parameters checking function
    _stypy_temp_lambda_578.stypy_localization = localization
    _stypy_temp_lambda_578.stypy_type_of_self = None
    _stypy_temp_lambda_578.stypy_type_store = module_type_store
    _stypy_temp_lambda_578.stypy_function_name = '_stypy_temp_lambda_578'
    _stypy_temp_lambda_578.stypy_param_names_list = []
    _stypy_temp_lambda_578.stypy_varargs_param_name = None
    _stypy_temp_lambda_578.stypy_kwargs_param_name = None
    _stypy_temp_lambda_578.stypy_call_defaults = defaults
    _stypy_temp_lambda_578.stypy_call_varargs = varargs
    _stypy_temp_lambda_578.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_578', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_578', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to test_get_plus1_capsule(...): (line 69)
    # Processing the call keyword arguments (line 69)
    kwargs_711221 = {}
    # Getting the type of '_test_ccallback' (line 69)
    _test_ccallback_711219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), '_test_ccallback', False)
    # Obtaining the member 'test_get_plus1_capsule' of a type (line 69)
    test_get_plus1_capsule_711220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 23), _test_ccallback_711219, 'test_get_plus1_capsule')
    # Calling test_get_plus1_capsule(args, kwargs) (line 69)
    test_get_plus1_capsule_call_result_711222 = invoke(stypy.reporting.localization.Localization(__file__, 69, 23), test_get_plus1_capsule_711220, *[], **kwargs_711221)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'stypy_return_type', test_get_plus1_capsule_call_result_711222)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_578' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_711223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711223)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_578'
    return stypy_return_type_711223

# Assigning a type to the variable '_stypy_temp_lambda_578' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), '_stypy_temp_lambda_578', _stypy_temp_lambda_578)
# Getting the type of '_stypy_temp_lambda_578' (line 69)
_stypy_temp_lambda_578_711224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), '_stypy_temp_lambda_578')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), dict_711213, (str_711218, _stypy_temp_lambda_578_711224))
# Adding element type (key, value) (line 67)
str_711225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 4), 'str', 'cython')

@norecursion
def _stypy_temp_lambda_579(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_579'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_579', 70, 14, True)
    # Passed parameters checking function
    _stypy_temp_lambda_579.stypy_localization = localization
    _stypy_temp_lambda_579.stypy_type_of_self = None
    _stypy_temp_lambda_579.stypy_type_store = module_type_store
    _stypy_temp_lambda_579.stypy_function_name = '_stypy_temp_lambda_579'
    _stypy_temp_lambda_579.stypy_param_names_list = []
    _stypy_temp_lambda_579.stypy_varargs_param_name = None
    _stypy_temp_lambda_579.stypy_kwargs_param_name = None
    _stypy_temp_lambda_579.stypy_call_defaults = defaults
    _stypy_temp_lambda_579.stypy_call_varargs = varargs
    _stypy_temp_lambda_579.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_579', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_579', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to from_cython(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of '_test_ccallback_cython' (line 70)
    _test_ccallback_cython_711228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 51), '_test_ccallback_cython', False)
    str_711229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 75), 'str', 'plus1_cython')
    # Processing the call keyword arguments (line 70)
    kwargs_711230 = {}
    # Getting the type of 'LowLevelCallable' (line 70)
    LowLevelCallable_711226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 22), 'LowLevelCallable', False)
    # Obtaining the member 'from_cython' of a type (line 70)
    from_cython_711227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 22), LowLevelCallable_711226, 'from_cython')
    # Calling from_cython(args, kwargs) (line 70)
    from_cython_call_result_711231 = invoke(stypy.reporting.localization.Localization(__file__, 70, 22), from_cython_711227, *[_test_ccallback_cython_711228, str_711229], **kwargs_711230)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), 'stypy_return_type', from_cython_call_result_711231)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_579' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_711232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711232)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_579'
    return stypy_return_type_711232

# Assigning a type to the variable '_stypy_temp_lambda_579' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), '_stypy_temp_lambda_579', _stypy_temp_lambda_579)
# Getting the type of '_stypy_temp_lambda_579' (line 70)
_stypy_temp_lambda_579_711233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), '_stypy_temp_lambda_579')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), dict_711213, (str_711225, _stypy_temp_lambda_579_711233))
# Adding element type (key, value) (line 67)
str_711234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 4), 'str', 'ctypes')

@norecursion
def _stypy_temp_lambda_580(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_580'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_580', 71, 14, True)
    # Passed parameters checking function
    _stypy_temp_lambda_580.stypy_localization = localization
    _stypy_temp_lambda_580.stypy_type_of_self = None
    _stypy_temp_lambda_580.stypy_type_store = module_type_store
    _stypy_temp_lambda_580.stypy_function_name = '_stypy_temp_lambda_580'
    _stypy_temp_lambda_580.stypy_param_names_list = []
    _stypy_temp_lambda_580.stypy_varargs_param_name = None
    _stypy_temp_lambda_580.stypy_kwargs_param_name = None
    _stypy_temp_lambda_580.stypy_call_defaults = defaults
    _stypy_temp_lambda_580.stypy_call_varargs = varargs
    _stypy_temp_lambda_580.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_580', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_580', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of '_test_ccallback_cython' (line 71)
    _test_ccallback_cython_711235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), '_test_ccallback_cython')
    # Obtaining the member 'plus1_ctypes' of a type (line 71)
    plus1_ctypes_711236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 22), _test_ccallback_cython_711235, 'plus1_ctypes')
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'stypy_return_type', plus1_ctypes_711236)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_580' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_711237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711237)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_580'
    return stypy_return_type_711237

# Assigning a type to the variable '_stypy_temp_lambda_580' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), '_stypy_temp_lambda_580', _stypy_temp_lambda_580)
# Getting the type of '_stypy_temp_lambda_580' (line 71)
_stypy_temp_lambda_580_711238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), '_stypy_temp_lambda_580')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), dict_711213, (str_711234, _stypy_temp_lambda_580_711238))
# Adding element type (key, value) (line 67)
str_711239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'str', 'cffi')

@norecursion
def _stypy_temp_lambda_581(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_581'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_581', 72, 12, True)
    # Passed parameters checking function
    _stypy_temp_lambda_581.stypy_localization = localization
    _stypy_temp_lambda_581.stypy_type_of_self = None
    _stypy_temp_lambda_581.stypy_type_store = module_type_store
    _stypy_temp_lambda_581.stypy_function_name = '_stypy_temp_lambda_581'
    _stypy_temp_lambda_581.stypy_param_names_list = []
    _stypy_temp_lambda_581.stypy_varargs_param_name = None
    _stypy_temp_lambda_581.stypy_kwargs_param_name = None
    _stypy_temp_lambda_581.stypy_call_defaults = defaults
    _stypy_temp_lambda_581.stypy_call_varargs = varargs
    _stypy_temp_lambda_581.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_581', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_581', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to _get_cffi_func(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of '_test_ccallback_cython' (line 72)
    _test_ccallback_cython_711241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 35), '_test_ccallback_cython', False)
    # Obtaining the member 'plus1_ctypes' of a type (line 72)
    plus1_ctypes_711242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 35), _test_ccallback_cython_711241, 'plus1_ctypes')
    str_711243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 35), 'str', 'double (*)(double, int *, void *)')
    # Processing the call keyword arguments (line 72)
    kwargs_711244 = {}
    # Getting the type of '_get_cffi_func' (line 72)
    _get_cffi_func_711240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), '_get_cffi_func', False)
    # Calling _get_cffi_func(args, kwargs) (line 72)
    _get_cffi_func_call_result_711245 = invoke(stypy.reporting.localization.Localization(__file__, 72, 20), _get_cffi_func_711240, *[plus1_ctypes_711242, str_711243], **kwargs_711244)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'stypy_return_type', _get_cffi_func_call_result_711245)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_581' in the type store
    # Getting the type of 'stypy_return_type' (line 72)
    stypy_return_type_711246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711246)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_581'
    return stypy_return_type_711246

# Assigning a type to the variable '_stypy_temp_lambda_581' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), '_stypy_temp_lambda_581', _stypy_temp_lambda_581)
# Getting the type of '_stypy_temp_lambda_581' (line 72)
_stypy_temp_lambda_581_711247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), '_stypy_temp_lambda_581')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), dict_711213, (str_711239, _stypy_temp_lambda_581_711247))
# Adding element type (key, value) (line 67)
str_711248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'str', 'capsule_b')

@norecursion
def _stypy_temp_lambda_582(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_582'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_582', 74, 17, True)
    # Passed parameters checking function
    _stypy_temp_lambda_582.stypy_localization = localization
    _stypy_temp_lambda_582.stypy_type_of_self = None
    _stypy_temp_lambda_582.stypy_type_store = module_type_store
    _stypy_temp_lambda_582.stypy_function_name = '_stypy_temp_lambda_582'
    _stypy_temp_lambda_582.stypy_param_names_list = []
    _stypy_temp_lambda_582.stypy_varargs_param_name = None
    _stypy_temp_lambda_582.stypy_kwargs_param_name = None
    _stypy_temp_lambda_582.stypy_call_defaults = defaults
    _stypy_temp_lambda_582.stypy_call_varargs = varargs
    _stypy_temp_lambda_582.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_582', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_582', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to test_get_plus1b_capsule(...): (line 74)
    # Processing the call keyword arguments (line 74)
    kwargs_711251 = {}
    # Getting the type of '_test_ccallback' (line 74)
    _test_ccallback_711249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), '_test_ccallback', False)
    # Obtaining the member 'test_get_plus1b_capsule' of a type (line 74)
    test_get_plus1b_capsule_711250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 25), _test_ccallback_711249, 'test_get_plus1b_capsule')
    # Calling test_get_plus1b_capsule(args, kwargs) (line 74)
    test_get_plus1b_capsule_call_result_711252 = invoke(stypy.reporting.localization.Localization(__file__, 74, 25), test_get_plus1b_capsule_711250, *[], **kwargs_711251)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'stypy_return_type', test_get_plus1b_capsule_call_result_711252)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_582' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_711253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711253)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_582'
    return stypy_return_type_711253

# Assigning a type to the variable '_stypy_temp_lambda_582' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), '_stypy_temp_lambda_582', _stypy_temp_lambda_582)
# Getting the type of '_stypy_temp_lambda_582' (line 74)
_stypy_temp_lambda_582_711254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), '_stypy_temp_lambda_582')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), dict_711213, (str_711248, _stypy_temp_lambda_582_711254))
# Adding element type (key, value) (line 67)
str_711255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'str', 'cython_b')

@norecursion
def _stypy_temp_lambda_583(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_583'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_583', 75, 16, True)
    # Passed parameters checking function
    _stypy_temp_lambda_583.stypy_localization = localization
    _stypy_temp_lambda_583.stypy_type_of_self = None
    _stypy_temp_lambda_583.stypy_type_store = module_type_store
    _stypy_temp_lambda_583.stypy_function_name = '_stypy_temp_lambda_583'
    _stypy_temp_lambda_583.stypy_param_names_list = []
    _stypy_temp_lambda_583.stypy_varargs_param_name = None
    _stypy_temp_lambda_583.stypy_kwargs_param_name = None
    _stypy_temp_lambda_583.stypy_call_defaults = defaults
    _stypy_temp_lambda_583.stypy_call_varargs = varargs
    _stypy_temp_lambda_583.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_583', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_583', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to from_cython(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of '_test_ccallback_cython' (line 75)
    _test_ccallback_cython_711258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 53), '_test_ccallback_cython', False)
    str_711259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 77), 'str', 'plus1b_cython')
    # Processing the call keyword arguments (line 75)
    kwargs_711260 = {}
    # Getting the type of 'LowLevelCallable' (line 75)
    LowLevelCallable_711256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'LowLevelCallable', False)
    # Obtaining the member 'from_cython' of a type (line 75)
    from_cython_711257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 24), LowLevelCallable_711256, 'from_cython')
    # Calling from_cython(args, kwargs) (line 75)
    from_cython_call_result_711261 = invoke(stypy.reporting.localization.Localization(__file__, 75, 24), from_cython_711257, *[_test_ccallback_cython_711258, str_711259], **kwargs_711260)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'stypy_return_type', from_cython_call_result_711261)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_583' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_711262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711262)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_583'
    return stypy_return_type_711262

# Assigning a type to the variable '_stypy_temp_lambda_583' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), '_stypy_temp_lambda_583', _stypy_temp_lambda_583)
# Getting the type of '_stypy_temp_lambda_583' (line 75)
_stypy_temp_lambda_583_711263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), '_stypy_temp_lambda_583')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), dict_711213, (str_711255, _stypy_temp_lambda_583_711263))
# Adding element type (key, value) (line 67)
str_711264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'str', 'ctypes_b')

@norecursion
def _stypy_temp_lambda_584(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_584'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_584', 76, 16, True)
    # Passed parameters checking function
    _stypy_temp_lambda_584.stypy_localization = localization
    _stypy_temp_lambda_584.stypy_type_of_self = None
    _stypy_temp_lambda_584.stypy_type_store = module_type_store
    _stypy_temp_lambda_584.stypy_function_name = '_stypy_temp_lambda_584'
    _stypy_temp_lambda_584.stypy_param_names_list = []
    _stypy_temp_lambda_584.stypy_varargs_param_name = None
    _stypy_temp_lambda_584.stypy_kwargs_param_name = None
    _stypy_temp_lambda_584.stypy_call_defaults = defaults
    _stypy_temp_lambda_584.stypy_call_varargs = varargs
    _stypy_temp_lambda_584.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_584', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_584', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of '_test_ccallback_cython' (line 76)
    _test_ccallback_cython_711265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), '_test_ccallback_cython')
    # Obtaining the member 'plus1b_ctypes' of a type (line 76)
    plus1b_ctypes_711266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 24), _test_ccallback_cython_711265, 'plus1b_ctypes')
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', plus1b_ctypes_711266)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_584' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_711267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711267)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_584'
    return stypy_return_type_711267

# Assigning a type to the variable '_stypy_temp_lambda_584' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), '_stypy_temp_lambda_584', _stypy_temp_lambda_584)
# Getting the type of '_stypy_temp_lambda_584' (line 76)
_stypy_temp_lambda_584_711268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), '_stypy_temp_lambda_584')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), dict_711213, (str_711264, _stypy_temp_lambda_584_711268))
# Adding element type (key, value) (line 67)
str_711269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'str', 'cffi_b')

@norecursion
def _stypy_temp_lambda_585(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_585'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_585', 77, 14, True)
    # Passed parameters checking function
    _stypy_temp_lambda_585.stypy_localization = localization
    _stypy_temp_lambda_585.stypy_type_of_self = None
    _stypy_temp_lambda_585.stypy_type_store = module_type_store
    _stypy_temp_lambda_585.stypy_function_name = '_stypy_temp_lambda_585'
    _stypy_temp_lambda_585.stypy_param_names_list = []
    _stypy_temp_lambda_585.stypy_varargs_param_name = None
    _stypy_temp_lambda_585.stypy_kwargs_param_name = None
    _stypy_temp_lambda_585.stypy_call_defaults = defaults
    _stypy_temp_lambda_585.stypy_call_varargs = varargs
    _stypy_temp_lambda_585.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_585', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_585', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to _get_cffi_func(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of '_test_ccallback_cython' (line 77)
    _test_ccallback_cython_711271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 37), '_test_ccallback_cython', False)
    # Obtaining the member 'plus1b_ctypes' of a type (line 77)
    plus1b_ctypes_711272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 37), _test_ccallback_cython_711271, 'plus1b_ctypes')
    str_711273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 37), 'str', 'double (*)(double, double, int *, void *)')
    # Processing the call keyword arguments (line 77)
    kwargs_711274 = {}
    # Getting the type of '_get_cffi_func' (line 77)
    _get_cffi_func_711270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), '_get_cffi_func', False)
    # Calling _get_cffi_func(args, kwargs) (line 77)
    _get_cffi_func_call_result_711275 = invoke(stypy.reporting.localization.Localization(__file__, 77, 22), _get_cffi_func_711270, *[plus1b_ctypes_711272, str_711273], **kwargs_711274)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), 'stypy_return_type', _get_cffi_func_call_result_711275)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_585' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_711276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711276)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_585'
    return stypy_return_type_711276

# Assigning a type to the variable '_stypy_temp_lambda_585' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), '_stypy_temp_lambda_585', _stypy_temp_lambda_585)
# Getting the type of '_stypy_temp_lambda_585' (line 77)
_stypy_temp_lambda_585_711277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), '_stypy_temp_lambda_585')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), dict_711213, (str_711269, _stypy_temp_lambda_585_711277))

# Assigning a type to the variable 'FUNCS' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'FUNCS', dict_711213)

# Assigning a Dict to a Name (line 82):

# Obtaining an instance of the builtin type 'dict' (line 82)
dict_711278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 82)
# Adding element type (key, value) (line 82)
str_711279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 4), 'str', 'capsule_bc')

@norecursion
def _stypy_temp_lambda_586(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_586'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_586', 83, 18, True)
    # Passed parameters checking function
    _stypy_temp_lambda_586.stypy_localization = localization
    _stypy_temp_lambda_586.stypy_type_of_self = None
    _stypy_temp_lambda_586.stypy_type_store = module_type_store
    _stypy_temp_lambda_586.stypy_function_name = '_stypy_temp_lambda_586'
    _stypy_temp_lambda_586.stypy_param_names_list = []
    _stypy_temp_lambda_586.stypy_varargs_param_name = None
    _stypy_temp_lambda_586.stypy_kwargs_param_name = None
    _stypy_temp_lambda_586.stypy_call_defaults = defaults
    _stypy_temp_lambda_586.stypy_call_varargs = varargs
    _stypy_temp_lambda_586.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_586', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_586', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to test_get_plus1bc_capsule(...): (line 83)
    # Processing the call keyword arguments (line 83)
    kwargs_711282 = {}
    # Getting the type of '_test_ccallback' (line 83)
    _test_ccallback_711280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 26), '_test_ccallback', False)
    # Obtaining the member 'test_get_plus1bc_capsule' of a type (line 83)
    test_get_plus1bc_capsule_711281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 26), _test_ccallback_711280, 'test_get_plus1bc_capsule')
    # Calling test_get_plus1bc_capsule(args, kwargs) (line 83)
    test_get_plus1bc_capsule_call_result_711283 = invoke(stypy.reporting.localization.Localization(__file__, 83, 26), test_get_plus1bc_capsule_711281, *[], **kwargs_711282)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'stypy_return_type', test_get_plus1bc_capsule_call_result_711283)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_586' in the type store
    # Getting the type of 'stypy_return_type' (line 83)
    stypy_return_type_711284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711284)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_586'
    return stypy_return_type_711284

# Assigning a type to the variable '_stypy_temp_lambda_586' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), '_stypy_temp_lambda_586', _stypy_temp_lambda_586)
# Getting the type of '_stypy_temp_lambda_586' (line 83)
_stypy_temp_lambda_586_711285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), '_stypy_temp_lambda_586')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 12), dict_711278, (str_711279, _stypy_temp_lambda_586_711285))
# Adding element type (key, value) (line 82)
str_711286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 4), 'str', 'cython_bc')

@norecursion
def _stypy_temp_lambda_587(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_587'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_587', 84, 17, True)
    # Passed parameters checking function
    _stypy_temp_lambda_587.stypy_localization = localization
    _stypy_temp_lambda_587.stypy_type_of_self = None
    _stypy_temp_lambda_587.stypy_type_store = module_type_store
    _stypy_temp_lambda_587.stypy_function_name = '_stypy_temp_lambda_587'
    _stypy_temp_lambda_587.stypy_param_names_list = []
    _stypy_temp_lambda_587.stypy_varargs_param_name = None
    _stypy_temp_lambda_587.stypy_kwargs_param_name = None
    _stypy_temp_lambda_587.stypy_call_defaults = defaults
    _stypy_temp_lambda_587.stypy_call_varargs = varargs
    _stypy_temp_lambda_587.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_587', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_587', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to from_cython(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of '_test_ccallback_cython' (line 84)
    _test_ccallback_cython_711289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 54), '_test_ccallback_cython', False)
    str_711290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 78), 'str', 'plus1bc_cython')
    # Processing the call keyword arguments (line 84)
    kwargs_711291 = {}
    # Getting the type of 'LowLevelCallable' (line 84)
    LowLevelCallable_711287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 'LowLevelCallable', False)
    # Obtaining the member 'from_cython' of a type (line 84)
    from_cython_711288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 25), LowLevelCallable_711287, 'from_cython')
    # Calling from_cython(args, kwargs) (line 84)
    from_cython_call_result_711292 = invoke(stypy.reporting.localization.Localization(__file__, 84, 25), from_cython_711288, *[_test_ccallback_cython_711289, str_711290], **kwargs_711291)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'stypy_return_type', from_cython_call_result_711292)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_587' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_711293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711293)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_587'
    return stypy_return_type_711293

# Assigning a type to the variable '_stypy_temp_lambda_587' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), '_stypy_temp_lambda_587', _stypy_temp_lambda_587)
# Getting the type of '_stypy_temp_lambda_587' (line 84)
_stypy_temp_lambda_587_711294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), '_stypy_temp_lambda_587')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 12), dict_711278, (str_711286, _stypy_temp_lambda_587_711294))
# Adding element type (key, value) (line 82)
str_711295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'str', 'ctypes_bc')

@norecursion
def _stypy_temp_lambda_588(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_588'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_588', 85, 17, True)
    # Passed parameters checking function
    _stypy_temp_lambda_588.stypy_localization = localization
    _stypy_temp_lambda_588.stypy_type_of_self = None
    _stypy_temp_lambda_588.stypy_type_store = module_type_store
    _stypy_temp_lambda_588.stypy_function_name = '_stypy_temp_lambda_588'
    _stypy_temp_lambda_588.stypy_param_names_list = []
    _stypy_temp_lambda_588.stypy_varargs_param_name = None
    _stypy_temp_lambda_588.stypy_kwargs_param_name = None
    _stypy_temp_lambda_588.stypy_call_defaults = defaults
    _stypy_temp_lambda_588.stypy_call_varargs = varargs
    _stypy_temp_lambda_588.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_588', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_588', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of '_test_ccallback_cython' (line 85)
    _test_ccallback_cython_711296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), '_test_ccallback_cython')
    # Obtaining the member 'plus1bc_ctypes' of a type (line 85)
    plus1bc_ctypes_711297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 25), _test_ccallback_cython_711296, 'plus1bc_ctypes')
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'stypy_return_type', plus1bc_ctypes_711297)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_588' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_711298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711298)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_588'
    return stypy_return_type_711298

# Assigning a type to the variable '_stypy_temp_lambda_588' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), '_stypy_temp_lambda_588', _stypy_temp_lambda_588)
# Getting the type of '_stypy_temp_lambda_588' (line 85)
_stypy_temp_lambda_588_711299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), '_stypy_temp_lambda_588')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 12), dict_711278, (str_711295, _stypy_temp_lambda_588_711299))
# Adding element type (key, value) (line 82)
str_711300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 4), 'str', 'cffi_bc')

@norecursion
def _stypy_temp_lambda_589(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_589'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_589', 86, 15, True)
    # Passed parameters checking function
    _stypy_temp_lambda_589.stypy_localization = localization
    _stypy_temp_lambda_589.stypy_type_of_self = None
    _stypy_temp_lambda_589.stypy_type_store = module_type_store
    _stypy_temp_lambda_589.stypy_function_name = '_stypy_temp_lambda_589'
    _stypy_temp_lambda_589.stypy_param_names_list = []
    _stypy_temp_lambda_589.stypy_varargs_param_name = None
    _stypy_temp_lambda_589.stypy_kwargs_param_name = None
    _stypy_temp_lambda_589.stypy_call_defaults = defaults
    _stypy_temp_lambda_589.stypy_call_varargs = varargs
    _stypy_temp_lambda_589.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_589', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_589', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to _get_cffi_func(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of '_test_ccallback_cython' (line 86)
    _test_ccallback_cython_711302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 38), '_test_ccallback_cython', False)
    # Obtaining the member 'plus1bc_ctypes' of a type (line 86)
    plus1bc_ctypes_711303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 38), _test_ccallback_cython_711302, 'plus1bc_ctypes')
    str_711304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 38), 'str', 'double (*)(double, double, double, int *, void *)')
    # Processing the call keyword arguments (line 86)
    kwargs_711305 = {}
    # Getting the type of '_get_cffi_func' (line 86)
    _get_cffi_func_711301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), '_get_cffi_func', False)
    # Calling _get_cffi_func(args, kwargs) (line 86)
    _get_cffi_func_call_result_711306 = invoke(stypy.reporting.localization.Localization(__file__, 86, 23), _get_cffi_func_711301, *[plus1bc_ctypes_711303, str_711304], **kwargs_711305)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'stypy_return_type', _get_cffi_func_call_result_711306)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_589' in the type store
    # Getting the type of 'stypy_return_type' (line 86)
    stypy_return_type_711307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711307)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_589'
    return stypy_return_type_711307

# Assigning a type to the variable '_stypy_temp_lambda_589' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), '_stypy_temp_lambda_589', _stypy_temp_lambda_589)
# Getting the type of '_stypy_temp_lambda_589' (line 86)
_stypy_temp_lambda_589_711308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), '_stypy_temp_lambda_589')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 12), dict_711278, (str_711300, _stypy_temp_lambda_589_711308))

# Assigning a type to the variable 'BAD_FUNCS' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'BAD_FUNCS', dict_711278)

# Assigning a Dict to a Name (line 90):

# Obtaining an instance of the builtin type 'dict' (line 90)
dict_711309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 90)
# Adding element type (key, value) (line 90)
str_711310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 4), 'str', 'ctypes')
# Getting the type of '_get_ctypes_data' (line 91)
_get_ctypes_data_711311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 14), '_get_ctypes_data')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 13), dict_711309, (str_711310, _get_ctypes_data_711311))
# Adding element type (key, value) (line 90)
str_711312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 4), 'str', 'cffi')
# Getting the type of '_get_cffi_data' (line 92)
_get_cffi_data_711313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), '_get_cffi_data')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 13), dict_711309, (str_711312, _get_cffi_data_711313))
# Adding element type (key, value) (line 90)
str_711314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 4), 'str', 'capsule')
# Getting the type of '_test_ccallback' (line 93)
_test_ccallback_711315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), '_test_ccallback')
# Obtaining the member 'test_get_data_capsule' of a type (line 93)
test_get_data_capsule_711316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 15), _test_ccallback_711315, 'test_get_data_capsule')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 13), dict_711309, (str_711314, test_get_data_capsule_711316))

# Assigning a type to the variable 'USER_DATAS' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'USER_DATAS', dict_711309)

@norecursion
def test_callbacks(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_callbacks'
    module_type_store = module_type_store.open_function_context('test_callbacks', 97, 0, False)
    
    # Passed parameters checking function
    test_callbacks.stypy_localization = localization
    test_callbacks.stypy_type_of_self = None
    test_callbacks.stypy_type_store = module_type_store
    test_callbacks.stypy_function_name = 'test_callbacks'
    test_callbacks.stypy_param_names_list = []
    test_callbacks.stypy_varargs_param_name = None
    test_callbacks.stypy_kwargs_param_name = None
    test_callbacks.stypy_call_defaults = defaults
    test_callbacks.stypy_call_varargs = varargs
    test_callbacks.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_callbacks', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_callbacks', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_callbacks(...)' code ##################


    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 98, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['caller', 'func', 'user_data']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['caller', 'func', 'user_data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['caller', 'func', 'user_data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Subscript to a Name (line 99):
        
        # Obtaining the type of the subscript
        # Getting the type of 'caller' (line 99)
        caller_711317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'caller')
        # Getting the type of 'CALLERS' (line 99)
        CALLERS_711318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'CALLERS')
        # Obtaining the member '__getitem__' of a type (line 99)
        getitem___711319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 17), CALLERS_711318, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 99)
        subscript_call_result_711320 = invoke(stypy.reporting.localization.Localization(__file__, 99, 17), getitem___711319, caller_711317)
        
        # Assigning a type to the variable 'caller' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'caller', subscript_call_result_711320)
        
        # Assigning a Call to a Name (line 100):
        
        # Call to (...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_711325 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'func' (line 100)
        func_711321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 21), 'func', False)
        # Getting the type of 'FUNCS' (line 100)
        FUNCS_711322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'FUNCS', False)
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___711323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), FUNCS_711322, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 100)
        subscript_call_result_711324 = invoke(stypy.reporting.localization.Localization(__file__, 100, 15), getitem___711323, func_711321)
        
        # Calling (args, kwargs) (line 100)
        _call_result_711326 = invoke(stypy.reporting.localization.Localization(__file__, 100, 15), subscript_call_result_711324, *[], **kwargs_711325)
        
        # Assigning a type to the variable 'func' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'func', _call_result_711326)
        
        # Assigning a Call to a Name (line 101):
        
        # Call to (...): (line 101)
        # Processing the call keyword arguments (line 101)
        kwargs_711331 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'user_data' (line 101)
        user_data_711327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'user_data', False)
        # Getting the type of 'USER_DATAS' (line 101)
        USER_DATAS_711328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'USER_DATAS', False)
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___711329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 20), USER_DATAS_711328, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_711330 = invoke(stypy.reporting.localization.Localization(__file__, 101, 20), getitem___711329, user_data_711327)
        
        # Calling (args, kwargs) (line 101)
        _call_result_711332 = invoke(stypy.reporting.localization.Localization(__file__, 101, 20), subscript_call_result_711330, *[], **kwargs_711331)
        
        # Assigning a type to the variable 'user_data' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'user_data', _call_result_711332)
        
        
        # Getting the type of 'func' (line 103)
        func_711333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'func')
        # Getting the type of 'callback_python' (line 103)
        callback_python_711334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'callback_python')
        # Applying the binary operator 'is' (line 103)
        result_is__711335 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 11), 'is', func_711333, callback_python_711334)
        
        # Testing the type of an if condition (line 103)
        if_condition_711336 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), result_is__711335)
        # Assigning a type to the variable 'if_condition_711336' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_711336', if_condition_711336)
        # SSA begins for if statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Lambda to a Name (line 104):

        @norecursion
        def _stypy_temp_lambda_590(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_590'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_590', 104, 20, True)
            # Passed parameters checking function
            _stypy_temp_lambda_590.stypy_localization = localization
            _stypy_temp_lambda_590.stypy_type_of_self = None
            _stypy_temp_lambda_590.stypy_type_store = module_type_store
            _stypy_temp_lambda_590.stypy_function_name = '_stypy_temp_lambda_590'
            _stypy_temp_lambda_590.stypy_param_names_list = ['x']
            _stypy_temp_lambda_590.stypy_varargs_param_name = None
            _stypy_temp_lambda_590.stypy_kwargs_param_name = None
            _stypy_temp_lambda_590.stypy_call_defaults = defaults
            _stypy_temp_lambda_590.stypy_call_varargs = varargs
            _stypy_temp_lambda_590.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_590', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_590', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to func(...): (line 104)
            # Processing the call arguments (line 104)
            # Getting the type of 'x' (line 104)
            x_711338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 35), 'x', False)
            float_711339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 38), 'float')
            # Processing the call keyword arguments (line 104)
            kwargs_711340 = {}
            # Getting the type of 'func' (line 104)
            func_711337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'func', False)
            # Calling func(args, kwargs) (line 104)
            func_call_result_711341 = invoke(stypy.reporting.localization.Localization(__file__, 104, 30), func_711337, *[x_711338, float_711339], **kwargs_711340)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'stypy_return_type', func_call_result_711341)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_590' in the type store
            # Getting the type of 'stypy_return_type' (line 104)
            stypy_return_type_711342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_711342)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_590'
            return stypy_return_type_711342

        # Assigning a type to the variable '_stypy_temp_lambda_590' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), '_stypy_temp_lambda_590', _stypy_temp_lambda_590)
        # Getting the type of '_stypy_temp_lambda_590' (line 104)
        _stypy_temp_lambda_590_711343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), '_stypy_temp_lambda_590')
        # Assigning a type to the variable 'func2' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'func2', _stypy_temp_lambda_590_711343)
        # SSA branch for the else part of an if statement (line 103)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 106):
        
        # Call to LowLevelCallable(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'func' (line 106)
        func_711345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 37), 'func', False)
        # Getting the type of 'user_data' (line 106)
        user_data_711346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 43), 'user_data', False)
        # Processing the call keyword arguments (line 106)
        kwargs_711347 = {}
        # Getting the type of 'LowLevelCallable' (line 106)
        LowLevelCallable_711344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'LowLevelCallable', False)
        # Calling LowLevelCallable(args, kwargs) (line 106)
        LowLevelCallable_call_result_711348 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), LowLevelCallable_711344, *[func_711345, user_data_711346], **kwargs_711347)
        
        # Assigning a type to the variable 'func2' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'func2', LowLevelCallable_call_result_711348)
        
        # Assigning a Call to a Name (line 107):
        
        # Call to LowLevelCallable(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'func' (line 107)
        func_711350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 36), 'func', False)
        # Processing the call keyword arguments (line 107)
        kwargs_711351 = {}
        # Getting the type of 'LowLevelCallable' (line 107)
        LowLevelCallable_711349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'LowLevelCallable', False)
        # Calling LowLevelCallable(args, kwargs) (line 107)
        LowLevelCallable_call_result_711352 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), LowLevelCallable_711349, *[func_711350], **kwargs_711351)
        
        # Assigning a type to the variable 'func' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'func', LowLevelCallable_call_result_711352)
        # SSA join for if statement (line 103)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Call to caller(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'func' (line 110)
        func_711355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 28), 'func', False)
        float_711356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 34), 'float')
        # Processing the call keyword arguments (line 110)
        kwargs_711357 = {}
        # Getting the type of 'caller' (line 110)
        caller_711354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'caller', False)
        # Calling caller(args, kwargs) (line 110)
        caller_call_result_711358 = invoke(stypy.reporting.localization.Localization(__file__, 110, 21), caller_711354, *[func_711355, float_711356], **kwargs_711357)
        
        float_711359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 40), 'float')
        # Processing the call keyword arguments (line 110)
        kwargs_711360 = {}
        # Getting the type of 'assert_equal' (line 110)
        assert_equal_711353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 110)
        assert_equal_call_result_711361 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), assert_equal_711353, *[caller_call_result_711358, float_711359], **kwargs_711360)
        
        
        # Call to assert_raises(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'ValueError' (line 113)
        ValueError_711363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 22), 'ValueError', False)
        # Getting the type of 'caller' (line 113)
        caller_711364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 34), 'caller', False)
        # Getting the type of 'func' (line 113)
        func_711365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 42), 'func', False)
        # Getting the type of 'ERROR_VALUE' (line 113)
        ERROR_VALUE_711366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 48), 'ERROR_VALUE', False)
        # Processing the call keyword arguments (line 113)
        kwargs_711367 = {}
        # Getting the type of 'assert_raises' (line 113)
        assert_raises_711362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 113)
        assert_raises_call_result_711368 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), assert_raises_711362, *[ValueError_711363, caller_711364, func_711365, ERROR_VALUE_711366], **kwargs_711367)
        
        
        # Call to assert_equal(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Call to caller(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'func2' (line 116)
        func2_711371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 28), 'func2', False)
        float_711372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 35), 'float')
        # Processing the call keyword arguments (line 116)
        kwargs_711373 = {}
        # Getting the type of 'caller' (line 116)
        caller_711370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 21), 'caller', False)
        # Calling caller(args, kwargs) (line 116)
        caller_call_result_711374 = invoke(stypy.reporting.localization.Localization(__file__, 116, 21), caller_711370, *[func2_711371, float_711372], **kwargs_711373)
        
        float_711375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 41), 'float')
        # Processing the call keyword arguments (line 116)
        kwargs_711376 = {}
        # Getting the type of 'assert_equal' (line 116)
        assert_equal_711369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 116)
        assert_equal_call_result_711377 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), assert_equal_711369, *[caller_call_result_711374, float_711375], **kwargs_711376)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 98)
        stypy_return_type_711378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711378)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_711378

    # Assigning a type to the variable 'check' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'check', check)
    
    
    # Call to sorted(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Call to keys(...): (line 118)
    # Processing the call keyword arguments (line 118)
    kwargs_711382 = {}
    # Getting the type of 'CALLERS' (line 118)
    CALLERS_711380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 25), 'CALLERS', False)
    # Obtaining the member 'keys' of a type (line 118)
    keys_711381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 25), CALLERS_711380, 'keys')
    # Calling keys(args, kwargs) (line 118)
    keys_call_result_711383 = invoke(stypy.reporting.localization.Localization(__file__, 118, 25), keys_711381, *[], **kwargs_711382)
    
    # Processing the call keyword arguments (line 118)
    kwargs_711384 = {}
    # Getting the type of 'sorted' (line 118)
    sorted_711379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 18), 'sorted', False)
    # Calling sorted(args, kwargs) (line 118)
    sorted_call_result_711385 = invoke(stypy.reporting.localization.Localization(__file__, 118, 18), sorted_711379, *[keys_call_result_711383], **kwargs_711384)
    
    # Testing the type of a for loop iterable (line 118)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 4), sorted_call_result_711385)
    # Getting the type of the for loop variable (line 118)
    for_loop_var_711386 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 4), sorted_call_result_711385)
    # Assigning a type to the variable 'caller' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'caller', for_loop_var_711386)
    # SSA begins for a for statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to sorted(...): (line 119)
    # Processing the call arguments (line 119)
    
    # Call to keys(...): (line 119)
    # Processing the call keyword arguments (line 119)
    kwargs_711390 = {}
    # Getting the type of 'FUNCS' (line 119)
    FUNCS_711388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'FUNCS', False)
    # Obtaining the member 'keys' of a type (line 119)
    keys_711389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 27), FUNCS_711388, 'keys')
    # Calling keys(args, kwargs) (line 119)
    keys_call_result_711391 = invoke(stypy.reporting.localization.Localization(__file__, 119, 27), keys_711389, *[], **kwargs_711390)
    
    # Processing the call keyword arguments (line 119)
    kwargs_711392 = {}
    # Getting the type of 'sorted' (line 119)
    sorted_711387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'sorted', False)
    # Calling sorted(args, kwargs) (line 119)
    sorted_call_result_711393 = invoke(stypy.reporting.localization.Localization(__file__, 119, 20), sorted_711387, *[keys_call_result_711391], **kwargs_711392)
    
    # Testing the type of a for loop iterable (line 119)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 119, 8), sorted_call_result_711393)
    # Getting the type of the for loop variable (line 119)
    for_loop_var_711394 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 119, 8), sorted_call_result_711393)
    # Assigning a type to the variable 'func' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'func', for_loop_var_711394)
    # SSA begins for a for statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to sorted(...): (line 120)
    # Processing the call arguments (line 120)
    
    # Call to keys(...): (line 120)
    # Processing the call keyword arguments (line 120)
    kwargs_711398 = {}
    # Getting the type of 'USER_DATAS' (line 120)
    USER_DATAS_711396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 36), 'USER_DATAS', False)
    # Obtaining the member 'keys' of a type (line 120)
    keys_711397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 36), USER_DATAS_711396, 'keys')
    # Calling keys(args, kwargs) (line 120)
    keys_call_result_711399 = invoke(stypy.reporting.localization.Localization(__file__, 120, 36), keys_711397, *[], **kwargs_711398)
    
    # Processing the call keyword arguments (line 120)
    kwargs_711400 = {}
    # Getting the type of 'sorted' (line 120)
    sorted_711395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'sorted', False)
    # Calling sorted(args, kwargs) (line 120)
    sorted_call_result_711401 = invoke(stypy.reporting.localization.Localization(__file__, 120, 29), sorted_711395, *[keys_call_result_711399], **kwargs_711400)
    
    # Testing the type of a for loop iterable (line 120)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 120, 12), sorted_call_result_711401)
    # Getting the type of the for loop variable (line 120)
    for_loop_var_711402 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 120, 12), sorted_call_result_711401)
    # Assigning a type to the variable 'user_data' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'user_data', for_loop_var_711402)
    # SSA begins for a for statement (line 120)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'caller' (line 121)
    caller_711404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 22), 'caller', False)
    # Getting the type of 'func' (line 121)
    func_711405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 30), 'func', False)
    # Getting the type of 'user_data' (line 121)
    user_data_711406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 36), 'user_data', False)
    # Processing the call keyword arguments (line 121)
    kwargs_711407 = {}
    # Getting the type of 'check' (line 121)
    check_711403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'check', False)
    # Calling check(args, kwargs) (line 121)
    check_call_result_711408 = invoke(stypy.reporting.localization.Localization(__file__, 121, 16), check_711403, *[caller_711404, func_711405, user_data_711406], **kwargs_711407)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_callbacks(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_callbacks' in the type store
    # Getting the type of 'stypy_return_type' (line 97)
    stypy_return_type_711409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711409)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_callbacks'
    return stypy_return_type_711409

# Assigning a type to the variable 'test_callbacks' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'test_callbacks', test_callbacks)

@norecursion
def test_bad_callbacks(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_bad_callbacks'
    module_type_store = module_type_store.open_function_context('test_bad_callbacks', 124, 0, False)
    
    # Passed parameters checking function
    test_bad_callbacks.stypy_localization = localization
    test_bad_callbacks.stypy_type_of_self = None
    test_bad_callbacks.stypy_type_store = module_type_store
    test_bad_callbacks.stypy_function_name = 'test_bad_callbacks'
    test_bad_callbacks.stypy_param_names_list = []
    test_bad_callbacks.stypy_varargs_param_name = None
    test_bad_callbacks.stypy_kwargs_param_name = None
    test_bad_callbacks.stypy_call_defaults = defaults
    test_bad_callbacks.stypy_call_varargs = varargs
    test_bad_callbacks.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_bad_callbacks', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_bad_callbacks', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_bad_callbacks(...)' code ##################


    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 125, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['caller', 'func', 'user_data']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['caller', 'func', 'user_data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['caller', 'func', 'user_data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Subscript to a Name (line 126):
        
        # Obtaining the type of the subscript
        # Getting the type of 'caller' (line 126)
        caller_711410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 25), 'caller')
        # Getting the type of 'CALLERS' (line 126)
        CALLERS_711411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'CALLERS')
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___711412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 17), CALLERS_711411, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_711413 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), getitem___711412, caller_711410)
        
        # Assigning a type to the variable 'caller' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'caller', subscript_call_result_711413)
        
        # Assigning a Call to a Name (line 127):
        
        # Call to (...): (line 127)
        # Processing the call keyword arguments (line 127)
        kwargs_711418 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'user_data' (line 127)
        user_data_711414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 31), 'user_data', False)
        # Getting the type of 'USER_DATAS' (line 127)
        USER_DATAS_711415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'USER_DATAS', False)
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___711416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 20), USER_DATAS_711415, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_711417 = invoke(stypy.reporting.localization.Localization(__file__, 127, 20), getitem___711416, user_data_711414)
        
        # Calling (args, kwargs) (line 127)
        _call_result_711419 = invoke(stypy.reporting.localization.Localization(__file__, 127, 20), subscript_call_result_711417, *[], **kwargs_711418)
        
        # Assigning a type to the variable 'user_data' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'user_data', _call_result_711419)
        
        # Assigning a Call to a Name (line 128):
        
        # Call to (...): (line 128)
        # Processing the call keyword arguments (line 128)
        kwargs_711424 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'func' (line 128)
        func_711420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'func', False)
        # Getting the type of 'BAD_FUNCS' (line 128)
        BAD_FUNCS_711421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'BAD_FUNCS', False)
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___711422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 15), BAD_FUNCS_711421, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_711423 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), getitem___711422, func_711420)
        
        # Calling (args, kwargs) (line 128)
        _call_result_711425 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), subscript_call_result_711423, *[], **kwargs_711424)
        
        # Assigning a type to the variable 'func' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'func', _call_result_711425)
        
        
        # Getting the type of 'func' (line 130)
        func_711426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'func')
        # Getting the type of 'callback_python' (line 130)
        callback_python_711427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 19), 'callback_python')
        # Applying the binary operator 'is' (line 130)
        result_is__711428 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), 'is', func_711426, callback_python_711427)
        
        # Testing the type of an if condition (line 130)
        if_condition_711429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), result_is__711428)
        # Assigning a type to the variable 'if_condition_711429' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_711429', if_condition_711429)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Lambda to a Name (line 131):

        @norecursion
        def _stypy_temp_lambda_591(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_591'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_591', 131, 20, True)
            # Passed parameters checking function
            _stypy_temp_lambda_591.stypy_localization = localization
            _stypy_temp_lambda_591.stypy_type_of_self = None
            _stypy_temp_lambda_591.stypy_type_store = module_type_store
            _stypy_temp_lambda_591.stypy_function_name = '_stypy_temp_lambda_591'
            _stypy_temp_lambda_591.stypy_param_names_list = ['x']
            _stypy_temp_lambda_591.stypy_varargs_param_name = None
            _stypy_temp_lambda_591.stypy_kwargs_param_name = None
            _stypy_temp_lambda_591.stypy_call_defaults = defaults
            _stypy_temp_lambda_591.stypy_call_varargs = varargs
            _stypy_temp_lambda_591.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_591', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_591', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to func(...): (line 131)
            # Processing the call arguments (line 131)
            # Getting the type of 'x' (line 131)
            x_711431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'x', False)
            float_711432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 38), 'float')
            # Processing the call keyword arguments (line 131)
            kwargs_711433 = {}
            # Getting the type of 'func' (line 131)
            func_711430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'func', False)
            # Calling func(args, kwargs) (line 131)
            func_call_result_711434 = invoke(stypy.reporting.localization.Localization(__file__, 131, 30), func_711430, *[x_711431, float_711432], **kwargs_711433)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 20), 'stypy_return_type', func_call_result_711434)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_591' in the type store
            # Getting the type of 'stypy_return_type' (line 131)
            stypy_return_type_711435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 20), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_711435)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_591'
            return stypy_return_type_711435

        # Assigning a type to the variable '_stypy_temp_lambda_591' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 20), '_stypy_temp_lambda_591', _stypy_temp_lambda_591)
        # Getting the type of '_stypy_temp_lambda_591' (line 131)
        _stypy_temp_lambda_591_711436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 20), '_stypy_temp_lambda_591')
        # Assigning a type to the variable 'func2' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'func2', _stypy_temp_lambda_591_711436)
        # SSA branch for the else part of an if statement (line 130)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 133):
        
        # Call to LowLevelCallable(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'func' (line 133)
        func_711438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 37), 'func', False)
        # Getting the type of 'user_data' (line 133)
        user_data_711439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 43), 'user_data', False)
        # Processing the call keyword arguments (line 133)
        kwargs_711440 = {}
        # Getting the type of 'LowLevelCallable' (line 133)
        LowLevelCallable_711437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 20), 'LowLevelCallable', False)
        # Calling LowLevelCallable(args, kwargs) (line 133)
        LowLevelCallable_call_result_711441 = invoke(stypy.reporting.localization.Localization(__file__, 133, 20), LowLevelCallable_711437, *[func_711438, user_data_711439], **kwargs_711440)
        
        # Assigning a type to the variable 'func2' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'func2', LowLevelCallable_call_result_711441)
        
        # Assigning a Call to a Name (line 134):
        
        # Call to LowLevelCallable(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'func' (line 134)
        func_711443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 36), 'func', False)
        # Processing the call keyword arguments (line 134)
        kwargs_711444 = {}
        # Getting the type of 'LowLevelCallable' (line 134)
        LowLevelCallable_711442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 19), 'LowLevelCallable', False)
        # Calling LowLevelCallable(args, kwargs) (line 134)
        LowLevelCallable_call_result_711445 = invoke(stypy.reporting.localization.Localization(__file__, 134, 19), LowLevelCallable_711442, *[func_711443], **kwargs_711444)
        
        # Assigning a type to the variable 'func' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'func', LowLevelCallable_call_result_711445)
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_raises(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'ValueError' (line 137)
        ValueError_711447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 22), 'ValueError', False)
        # Getting the type of 'caller' (line 137)
        caller_711448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 34), 'caller', False)
        
        # Call to LowLevelCallable(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'func' (line 137)
        func_711450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 59), 'func', False)
        # Processing the call keyword arguments (line 137)
        kwargs_711451 = {}
        # Getting the type of 'LowLevelCallable' (line 137)
        LowLevelCallable_711449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), 'LowLevelCallable', False)
        # Calling LowLevelCallable(args, kwargs) (line 137)
        LowLevelCallable_call_result_711452 = invoke(stypy.reporting.localization.Localization(__file__, 137, 42), LowLevelCallable_711449, *[func_711450], **kwargs_711451)
        
        float_711453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 66), 'float')
        # Processing the call keyword arguments (line 137)
        kwargs_711454 = {}
        # Getting the type of 'assert_raises' (line 137)
        assert_raises_711446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 137)
        assert_raises_call_result_711455 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), assert_raises_711446, *[ValueError_711447, caller_711448, LowLevelCallable_call_result_711452, float_711453], **kwargs_711454)
        
        
        # Call to assert_raises(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'ValueError' (line 140)
        ValueError_711457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 22), 'ValueError', False)
        # Getting the type of 'caller' (line 140)
        caller_711458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 34), 'caller', False)
        # Getting the type of 'func2' (line 140)
        func2_711459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 42), 'func2', False)
        float_711460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 49), 'float')
        # Processing the call keyword arguments (line 140)
        kwargs_711461 = {}
        # Getting the type of 'assert_raises' (line 140)
        assert_raises_711456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 140)
        assert_raises_call_result_711462 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), assert_raises_711456, *[ValueError_711457, caller_711458, func2_711459, float_711460], **kwargs_711461)
        
        
        # Assigning a Call to a Name (line 143):
        
        # Call to LowLevelCallable(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'func' (line 143)
        func_711464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 34), 'func', False)
        # Processing the call keyword arguments (line 143)
        kwargs_711465 = {}
        # Getting the type of 'LowLevelCallable' (line 143)
        LowLevelCallable_711463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 17), 'LowLevelCallable', False)
        # Calling LowLevelCallable(args, kwargs) (line 143)
        LowLevelCallable_call_result_711466 = invoke(stypy.reporting.localization.Localization(__file__, 143, 17), LowLevelCallable_711463, *[func_711464], **kwargs_711465)
        
        # Assigning a type to the variable 'llfunc' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'llfunc', LowLevelCallable_call_result_711466)
        
        
        # SSA begins for try-except statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to caller(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'llfunc' (line 145)
        llfunc_711468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'llfunc', False)
        float_711469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 27), 'float')
        # Processing the call keyword arguments (line 145)
        kwargs_711470 = {}
        # Getting the type of 'caller' (line 145)
        caller_711467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'caller', False)
        # Calling caller(args, kwargs) (line 145)
        caller_call_result_711471 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), caller_711467, *[llfunc_711468, float_711469], **kwargs_711470)
        
        # SSA branch for the except part of a try statement (line 144)
        # SSA branch for the except 'ValueError' branch of a try statement (line 144)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'ValueError' (line 146)
        ValueError_711472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'ValueError')
        # Assigning a type to the variable 'err' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'err', ValueError_711472)
        
        # Assigning a Call to a Name (line 147):
        
        # Call to str(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'err' (line 147)
        err_711474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'err', False)
        # Processing the call keyword arguments (line 147)
        kwargs_711475 = {}
        # Getting the type of 'str' (line 147)
        str_711473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 18), 'str', False)
        # Calling str(args, kwargs) (line 147)
        str_call_result_711476 = invoke(stypy.reporting.localization.Localization(__file__, 147, 18), str_711473, *[err_711474], **kwargs_711475)
        
        # Assigning a type to the variable 'msg' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'msg', str_call_result_711476)
        
        # Call to assert_(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Getting the type of 'llfunc' (line 148)
        llfunc_711478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'llfunc', False)
        # Obtaining the member 'signature' of a type (line 148)
        signature_711479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 20), llfunc_711478, 'signature')
        # Getting the type of 'msg' (line 148)
        msg_711480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 40), 'msg', False)
        # Applying the binary operator 'in' (line 148)
        result_contains_711481 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 20), 'in', signature_711479, msg_711480)
        
        # Getting the type of 'msg' (line 148)
        msg_711482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 45), 'msg', False)
        # Processing the call keyword arguments (line 148)
        kwargs_711483 = {}
        # Getting the type of 'assert_' (line 148)
        assert__711477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 148)
        assert__call_result_711484 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), assert__711477, *[result_contains_711481, msg_711482], **kwargs_711483)
        
        
        # Call to assert_(...): (line 149)
        # Processing the call arguments (line 149)
        
        str_711486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 20), 'str', 'double (double, double, int *, void *)')
        # Getting the type of 'msg' (line 149)
        msg_711487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 64), 'msg', False)
        # Applying the binary operator 'in' (line 149)
        result_contains_711488 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 20), 'in', str_711486, msg_711487)
        
        # Getting the type of 'msg' (line 149)
        msg_711489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 69), 'msg', False)
        # Processing the call keyword arguments (line 149)
        kwargs_711490 = {}
        # Getting the type of 'assert_' (line 149)
        assert__711485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 149)
        assert__call_result_711491 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), assert__711485, *[result_contains_711488, msg_711489], **kwargs_711490)
        
        # SSA join for try-except statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_711492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711492)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_711492

    # Assigning a type to the variable 'check' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'check', check)
    
    
    # Call to sorted(...): (line 151)
    # Processing the call arguments (line 151)
    
    # Call to keys(...): (line 151)
    # Processing the call keyword arguments (line 151)
    kwargs_711496 = {}
    # Getting the type of 'CALLERS' (line 151)
    CALLERS_711494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'CALLERS', False)
    # Obtaining the member 'keys' of a type (line 151)
    keys_711495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 25), CALLERS_711494, 'keys')
    # Calling keys(args, kwargs) (line 151)
    keys_call_result_711497 = invoke(stypy.reporting.localization.Localization(__file__, 151, 25), keys_711495, *[], **kwargs_711496)
    
    # Processing the call keyword arguments (line 151)
    kwargs_711498 = {}
    # Getting the type of 'sorted' (line 151)
    sorted_711493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 18), 'sorted', False)
    # Calling sorted(args, kwargs) (line 151)
    sorted_call_result_711499 = invoke(stypy.reporting.localization.Localization(__file__, 151, 18), sorted_711493, *[keys_call_result_711497], **kwargs_711498)
    
    # Testing the type of a for loop iterable (line 151)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 151, 4), sorted_call_result_711499)
    # Getting the type of the for loop variable (line 151)
    for_loop_var_711500 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 151, 4), sorted_call_result_711499)
    # Assigning a type to the variable 'caller' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'caller', for_loop_var_711500)
    # SSA begins for a for statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to sorted(...): (line 152)
    # Processing the call arguments (line 152)
    
    # Call to keys(...): (line 152)
    # Processing the call keyword arguments (line 152)
    kwargs_711504 = {}
    # Getting the type of 'BAD_FUNCS' (line 152)
    BAD_FUNCS_711502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 27), 'BAD_FUNCS', False)
    # Obtaining the member 'keys' of a type (line 152)
    keys_711503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 27), BAD_FUNCS_711502, 'keys')
    # Calling keys(args, kwargs) (line 152)
    keys_call_result_711505 = invoke(stypy.reporting.localization.Localization(__file__, 152, 27), keys_711503, *[], **kwargs_711504)
    
    # Processing the call keyword arguments (line 152)
    kwargs_711506 = {}
    # Getting the type of 'sorted' (line 152)
    sorted_711501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'sorted', False)
    # Calling sorted(args, kwargs) (line 152)
    sorted_call_result_711507 = invoke(stypy.reporting.localization.Localization(__file__, 152, 20), sorted_711501, *[keys_call_result_711505], **kwargs_711506)
    
    # Testing the type of a for loop iterable (line 152)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 152, 8), sorted_call_result_711507)
    # Getting the type of the for loop variable (line 152)
    for_loop_var_711508 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 152, 8), sorted_call_result_711507)
    # Assigning a type to the variable 'func' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'func', for_loop_var_711508)
    # SSA begins for a for statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to sorted(...): (line 153)
    # Processing the call arguments (line 153)
    
    # Call to keys(...): (line 153)
    # Processing the call keyword arguments (line 153)
    kwargs_711512 = {}
    # Getting the type of 'USER_DATAS' (line 153)
    USER_DATAS_711510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 36), 'USER_DATAS', False)
    # Obtaining the member 'keys' of a type (line 153)
    keys_711511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 36), USER_DATAS_711510, 'keys')
    # Calling keys(args, kwargs) (line 153)
    keys_call_result_711513 = invoke(stypy.reporting.localization.Localization(__file__, 153, 36), keys_711511, *[], **kwargs_711512)
    
    # Processing the call keyword arguments (line 153)
    kwargs_711514 = {}
    # Getting the type of 'sorted' (line 153)
    sorted_711509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 29), 'sorted', False)
    # Calling sorted(args, kwargs) (line 153)
    sorted_call_result_711515 = invoke(stypy.reporting.localization.Localization(__file__, 153, 29), sorted_711509, *[keys_call_result_711513], **kwargs_711514)
    
    # Testing the type of a for loop iterable (line 153)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 153, 12), sorted_call_result_711515)
    # Getting the type of the for loop variable (line 153)
    for_loop_var_711516 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 153, 12), sorted_call_result_711515)
    # Assigning a type to the variable 'user_data' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'user_data', for_loop_var_711516)
    # SSA begins for a for statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'caller' (line 154)
    caller_711518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'caller', False)
    # Getting the type of 'func' (line 154)
    func_711519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), 'func', False)
    # Getting the type of 'user_data' (line 154)
    user_data_711520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 36), 'user_data', False)
    # Processing the call keyword arguments (line 154)
    kwargs_711521 = {}
    # Getting the type of 'check' (line 154)
    check_711517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'check', False)
    # Calling check(args, kwargs) (line 154)
    check_call_result_711522 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), check_711517, *[caller_711518, func_711519, user_data_711520], **kwargs_711521)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_bad_callbacks(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_bad_callbacks' in the type store
    # Getting the type of 'stypy_return_type' (line 124)
    stypy_return_type_711523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711523)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_bad_callbacks'
    return stypy_return_type_711523

# Assigning a type to the variable 'test_bad_callbacks' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'test_bad_callbacks', test_bad_callbacks)

@norecursion
def test_signature_override(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_signature_override'
    module_type_store = module_type_store.open_function_context('test_signature_override', 157, 0, False)
    
    # Passed parameters checking function
    test_signature_override.stypy_localization = localization
    test_signature_override.stypy_type_of_self = None
    test_signature_override.stypy_type_store = module_type_store
    test_signature_override.stypy_function_name = 'test_signature_override'
    test_signature_override.stypy_param_names_list = []
    test_signature_override.stypy_varargs_param_name = None
    test_signature_override.stypy_kwargs_param_name = None
    test_signature_override.stypy_call_defaults = defaults
    test_signature_override.stypy_call_varargs = varargs
    test_signature_override.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_signature_override', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_signature_override', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_signature_override(...)' code ##################

    
    # Assigning a Attribute to a Name (line 158):
    # Getting the type of '_test_ccallback' (line 158)
    _test_ccallback_711524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 13), '_test_ccallback')
    # Obtaining the member 'test_call_simple' of a type (line 158)
    test_call_simple_711525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 13), _test_ccallback_711524, 'test_call_simple')
    # Assigning a type to the variable 'caller' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'caller', test_call_simple_711525)
    
    # Assigning a Call to a Name (line 159):
    
    # Call to test_get_plus1_capsule(...): (line 159)
    # Processing the call keyword arguments (line 159)
    kwargs_711528 = {}
    # Getting the type of '_test_ccallback' (line 159)
    _test_ccallback_711526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), '_test_ccallback', False)
    # Obtaining the member 'test_get_plus1_capsule' of a type (line 159)
    test_get_plus1_capsule_711527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), _test_ccallback_711526, 'test_get_plus1_capsule')
    # Calling test_get_plus1_capsule(args, kwargs) (line 159)
    test_get_plus1_capsule_call_result_711529 = invoke(stypy.reporting.localization.Localization(__file__, 159, 11), test_get_plus1_capsule_711527, *[], **kwargs_711528)
    
    # Assigning a type to the variable 'func' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'func', test_get_plus1_capsule_call_result_711529)
    
    # Assigning a Call to a Name (line 161):
    
    # Call to LowLevelCallable(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'func' (line 161)
    func_711531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 34), 'func', False)
    # Processing the call keyword arguments (line 161)
    str_711532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 50), 'str', 'bad signature')
    keyword_711533 = str_711532
    kwargs_711534 = {'signature': keyword_711533}
    # Getting the type of 'LowLevelCallable' (line 161)
    LowLevelCallable_711530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 17), 'LowLevelCallable', False)
    # Calling LowLevelCallable(args, kwargs) (line 161)
    LowLevelCallable_call_result_711535 = invoke(stypy.reporting.localization.Localization(__file__, 161, 17), LowLevelCallable_711530, *[func_711531], **kwargs_711534)
    
    # Assigning a type to the variable 'llcallable' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'llcallable', LowLevelCallable_call_result_711535)
    
    # Call to assert_equal(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'llcallable' (line 162)
    llcallable_711537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 17), 'llcallable', False)
    # Obtaining the member 'signature' of a type (line 162)
    signature_711538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 17), llcallable_711537, 'signature')
    str_711539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 39), 'str', 'bad signature')
    # Processing the call keyword arguments (line 162)
    kwargs_711540 = {}
    # Getting the type of 'assert_equal' (line 162)
    assert_equal_711536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 162)
    assert_equal_call_result_711541 = invoke(stypy.reporting.localization.Localization(__file__, 162, 4), assert_equal_711536, *[signature_711538, str_711539], **kwargs_711540)
    
    
    # Call to assert_raises(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'ValueError' (line 163)
    ValueError_711543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 18), 'ValueError', False)
    # Getting the type of 'caller' (line 163)
    caller_711544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 30), 'caller', False)
    # Getting the type of 'llcallable' (line 163)
    llcallable_711545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 38), 'llcallable', False)
    int_711546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 50), 'int')
    # Processing the call keyword arguments (line 163)
    kwargs_711547 = {}
    # Getting the type of 'assert_raises' (line 163)
    assert_raises_711542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 163)
    assert_raises_call_result_711548 = invoke(stypy.reporting.localization.Localization(__file__, 163, 4), assert_raises_711542, *[ValueError_711543, caller_711544, llcallable_711545, int_711546], **kwargs_711547)
    
    
    # Assigning a Call to a Name (line 165):
    
    # Call to LowLevelCallable(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'func' (line 165)
    func_711550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 34), 'func', False)
    # Processing the call keyword arguments (line 165)
    str_711551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 50), 'str', 'double (double, int *, void *)')
    keyword_711552 = str_711551
    kwargs_711553 = {'signature': keyword_711552}
    # Getting the type of 'LowLevelCallable' (line 165)
    LowLevelCallable_711549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 17), 'LowLevelCallable', False)
    # Calling LowLevelCallable(args, kwargs) (line 165)
    LowLevelCallable_call_result_711554 = invoke(stypy.reporting.localization.Localization(__file__, 165, 17), LowLevelCallable_711549, *[func_711550], **kwargs_711553)
    
    # Assigning a type to the variable 'llcallable' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'llcallable', LowLevelCallable_call_result_711554)
    
    # Call to assert_equal(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'llcallable' (line 166)
    llcallable_711556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 17), 'llcallable', False)
    # Obtaining the member 'signature' of a type (line 166)
    signature_711557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 17), llcallable_711556, 'signature')
    str_711558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 39), 'str', 'double (double, int *, void *)')
    # Processing the call keyword arguments (line 166)
    kwargs_711559 = {}
    # Getting the type of 'assert_equal' (line 166)
    assert_equal_711555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 166)
    assert_equal_call_result_711560 = invoke(stypy.reporting.localization.Localization(__file__, 166, 4), assert_equal_711555, *[signature_711557, str_711558], **kwargs_711559)
    
    
    # Call to assert_equal(...): (line 167)
    # Processing the call arguments (line 167)
    
    # Call to caller(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'llcallable' (line 167)
    llcallable_711563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'llcallable', False)
    int_711564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 36), 'int')
    # Processing the call keyword arguments (line 167)
    kwargs_711565 = {}
    # Getting the type of 'caller' (line 167)
    caller_711562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 17), 'caller', False)
    # Calling caller(args, kwargs) (line 167)
    caller_call_result_711566 = invoke(stypy.reporting.localization.Localization(__file__, 167, 17), caller_711562, *[llcallable_711563, int_711564], **kwargs_711565)
    
    int_711567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 40), 'int')
    # Processing the call keyword arguments (line 167)
    kwargs_711568 = {}
    # Getting the type of 'assert_equal' (line 167)
    assert_equal_711561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 167)
    assert_equal_call_result_711569 = invoke(stypy.reporting.localization.Localization(__file__, 167, 4), assert_equal_711561, *[caller_call_result_711566, int_711567], **kwargs_711568)
    
    
    # ################# End of 'test_signature_override(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_signature_override' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_711570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711570)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_signature_override'
    return stypy_return_type_711570

# Assigning a type to the variable 'test_signature_override' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'test_signature_override', test_signature_override)

@norecursion
def test_threadsafety(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_threadsafety'
    module_type_store = module_type_store.open_function_context('test_threadsafety', 170, 0, False)
    
    # Passed parameters checking function
    test_threadsafety.stypy_localization = localization
    test_threadsafety.stypy_type_of_self = None
    test_threadsafety.stypy_type_store = module_type_store
    test_threadsafety.stypy_function_name = 'test_threadsafety'
    test_threadsafety.stypy_param_names_list = []
    test_threadsafety.stypy_varargs_param_name = None
    test_threadsafety.stypy_kwargs_param_name = None
    test_threadsafety.stypy_call_defaults = defaults
    test_threadsafety.stypy_call_varargs = varargs
    test_threadsafety.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_threadsafety', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_threadsafety', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_threadsafety(...)' code ##################


    @norecursion
    def callback(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'callback'
        module_type_store = module_type_store.open_function_context('callback', 171, 4, False)
        
        # Passed parameters checking function
        callback.stypy_localization = localization
        callback.stypy_type_of_self = None
        callback.stypy_type_store = module_type_store
        callback.stypy_function_name = 'callback'
        callback.stypy_param_names_list = ['a', 'caller']
        callback.stypy_varargs_param_name = None
        callback.stypy_kwargs_param_name = None
        callback.stypy_call_defaults = defaults
        callback.stypy_call_varargs = varargs
        callback.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'callback', ['a', 'caller'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'callback', localization, ['a', 'caller'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'callback(...)' code ##################

        
        
        # Getting the type of 'a' (line 172)
        a_711571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'a')
        int_711572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 16), 'int')
        # Applying the binary operator '<=' (line 172)
        result_le_711573 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 11), '<=', a_711571, int_711572)
        
        # Testing the type of an if condition (line 172)
        if_condition_711574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), result_le_711573)
        # Assigning a type to the variable 'if_condition_711574' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_711574', if_condition_711574)
        # SSA begins for if statement (line 172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_711575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'stypy_return_type', int_711575)
        # SSA branch for the else part of an if statement (line 172)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 175):
        
        # Call to caller(...): (line 175)
        # Processing the call arguments (line 175)

        @norecursion
        def _stypy_temp_lambda_592(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_592'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_592', 175, 25, True)
            # Passed parameters checking function
            _stypy_temp_lambda_592.stypy_localization = localization
            _stypy_temp_lambda_592.stypy_type_of_self = None
            _stypy_temp_lambda_592.stypy_type_store = module_type_store
            _stypy_temp_lambda_592.stypy_function_name = '_stypy_temp_lambda_592'
            _stypy_temp_lambda_592.stypy_param_names_list = ['x']
            _stypy_temp_lambda_592.stypy_varargs_param_name = None
            _stypy_temp_lambda_592.stypy_kwargs_param_name = None
            _stypy_temp_lambda_592.stypy_call_defaults = defaults
            _stypy_temp_lambda_592.stypy_call_varargs = varargs
            _stypy_temp_lambda_592.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_592', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_592', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to callback(...): (line 175)
            # Processing the call arguments (line 175)
            # Getting the type of 'x' (line 175)
            x_711578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 44), 'x', False)
            # Getting the type of 'caller' (line 175)
            caller_711579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 47), 'caller', False)
            # Processing the call keyword arguments (line 175)
            kwargs_711580 = {}
            # Getting the type of 'callback' (line 175)
            callback_711577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 35), 'callback', False)
            # Calling callback(args, kwargs) (line 175)
            callback_call_result_711581 = invoke(stypy.reporting.localization.Localization(__file__, 175, 35), callback_711577, *[x_711578, caller_711579], **kwargs_711580)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'stypy_return_type', callback_call_result_711581)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_592' in the type store
            # Getting the type of 'stypy_return_type' (line 175)
            stypy_return_type_711582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_711582)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_592'
            return stypy_return_type_711582

        # Assigning a type to the variable '_stypy_temp_lambda_592' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), '_stypy_temp_lambda_592', _stypy_temp_lambda_592)
        # Getting the type of '_stypy_temp_lambda_592' (line 175)
        _stypy_temp_lambda_592_711583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), '_stypy_temp_lambda_592')
        # Getting the type of 'a' (line 175)
        a_711584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 56), 'a', False)
        int_711585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 60), 'int')
        # Applying the binary operator '-' (line 175)
        result_sub_711586 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 56), '-', a_711584, int_711585)
        
        # Processing the call keyword arguments (line 175)
        kwargs_711587 = {}
        # Getting the type of 'caller' (line 175)
        caller_711576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 18), 'caller', False)
        # Calling caller(args, kwargs) (line 175)
        caller_call_result_711588 = invoke(stypy.reporting.localization.Localization(__file__, 175, 18), caller_711576, *[_stypy_temp_lambda_592_711583, result_sub_711586], **kwargs_711587)
        
        # Assigning a type to the variable 'res' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'res', caller_call_result_711588)
        int_711589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 19), 'int')
        # Getting the type of 'res' (line 176)
        res_711590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 21), 'res')
        # Applying the binary operator '*' (line 176)
        result_mul_711591 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 19), '*', int_711589, res_711590)
        
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'stypy_return_type', result_mul_711591)
        # SSA join for if statement (line 172)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'callback(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'callback' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_711592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711592)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'callback'
        return stypy_return_type_711592

    # Assigning a type to the variable 'callback' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'callback', callback)

    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 178, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['caller']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['caller'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['caller'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Subscript to a Name (line 179):
        
        # Obtaining the type of the subscript
        # Getting the type of 'caller' (line 179)
        caller_711593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 25), 'caller')
        # Getting the type of 'CALLERS' (line 179)
        CALLERS_711594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 17), 'CALLERS')
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___711595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 17), CALLERS_711594, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_711596 = invoke(stypy.reporting.localization.Localization(__file__, 179, 17), getitem___711595, caller_711593)
        
        # Assigning a type to the variable 'caller' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'caller', subscript_call_result_711596)
        
        # Assigning a List to a Name (line 181):
        
        # Obtaining an instance of the builtin type 'list' (line 181)
        list_711597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 181)
        
        # Assigning a type to the variable 'results' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'results', list_711597)
        
        # Assigning a Num to a Name (line 183):
        int_711598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 16), 'int')
        # Assigning a type to the variable 'count' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'count', int_711598)

        @norecursion
        def run(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'run'
            module_type_store = module_type_store.open_function_context('run', 185, 8, False)
            
            # Passed parameters checking function
            run.stypy_localization = localization
            run.stypy_type_of_self = None
            run.stypy_type_store = module_type_store
            run.stypy_function_name = 'run'
            run.stypy_param_names_list = []
            run.stypy_varargs_param_name = None
            run.stypy_kwargs_param_name = None
            run.stypy_call_defaults = defaults
            run.stypy_call_varargs = varargs
            run.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'run', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'run', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'run(...)' code ##################

            
            # Call to sleep(...): (line 186)
            # Processing the call arguments (line 186)
            float_711601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 23), 'float')
            # Processing the call keyword arguments (line 186)
            kwargs_711602 = {}
            # Getting the type of 'time' (line 186)
            time_711599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'time', False)
            # Obtaining the member 'sleep' of a type (line 186)
            sleep_711600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), time_711599, 'sleep')
            # Calling sleep(args, kwargs) (line 186)
            sleep_call_result_711603 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), sleep_711600, *[float_711601], **kwargs_711602)
            
            
            # Assigning a Call to a Name (line 187):
            
            # Call to caller(...): (line 187)
            # Processing the call arguments (line 187)

            @norecursion
            def _stypy_temp_lambda_593(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_593'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_593', 187, 23, True)
                # Passed parameters checking function
                _stypy_temp_lambda_593.stypy_localization = localization
                _stypy_temp_lambda_593.stypy_type_of_self = None
                _stypy_temp_lambda_593.stypy_type_store = module_type_store
                _stypy_temp_lambda_593.stypy_function_name = '_stypy_temp_lambda_593'
                _stypy_temp_lambda_593.stypy_param_names_list = ['x']
                _stypy_temp_lambda_593.stypy_varargs_param_name = None
                _stypy_temp_lambda_593.stypy_kwargs_param_name = None
                _stypy_temp_lambda_593.stypy_call_defaults = defaults
                _stypy_temp_lambda_593.stypy_call_varargs = varargs
                _stypy_temp_lambda_593.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_593', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_593', ['x'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to callback(...): (line 187)
                # Processing the call arguments (line 187)
                # Getting the type of 'x' (line 187)
                x_711606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 42), 'x', False)
                # Getting the type of 'caller' (line 187)
                caller_711607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 45), 'caller', False)
                # Processing the call keyword arguments (line 187)
                kwargs_711608 = {}
                # Getting the type of 'callback' (line 187)
                callback_711605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 33), 'callback', False)
                # Calling callback(args, kwargs) (line 187)
                callback_call_result_711609 = invoke(stypy.reporting.localization.Localization(__file__, 187, 33), callback_711605, *[x_711606, caller_711607], **kwargs_711608)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 187)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'stypy_return_type', callback_call_result_711609)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_593' in the type store
                # Getting the type of 'stypy_return_type' (line 187)
                stypy_return_type_711610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_711610)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_593'
                return stypy_return_type_711610

            # Assigning a type to the variable '_stypy_temp_lambda_593' (line 187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), '_stypy_temp_lambda_593', _stypy_temp_lambda_593)
            # Getting the type of '_stypy_temp_lambda_593' (line 187)
            _stypy_temp_lambda_593_711611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), '_stypy_temp_lambda_593')
            # Getting the type of 'count' (line 187)
            count_711612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 54), 'count', False)
            # Processing the call keyword arguments (line 187)
            kwargs_711613 = {}
            # Getting the type of 'caller' (line 187)
            caller_711604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'caller', False)
            # Calling caller(args, kwargs) (line 187)
            caller_call_result_711614 = invoke(stypy.reporting.localization.Localization(__file__, 187, 16), caller_711604, *[_stypy_temp_lambda_593_711611, count_711612], **kwargs_711613)
            
            # Assigning a type to the variable 'r' (line 187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'r', caller_call_result_711614)
            
            # Call to append(...): (line 188)
            # Processing the call arguments (line 188)
            # Getting the type of 'r' (line 188)
            r_711617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 27), 'r', False)
            # Processing the call keyword arguments (line 188)
            kwargs_711618 = {}
            # Getting the type of 'results' (line 188)
            results_711615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'results', False)
            # Obtaining the member 'append' of a type (line 188)
            append_711616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), results_711615, 'append')
            # Calling append(args, kwargs) (line 188)
            append_call_result_711619 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), append_711616, *[r_711617], **kwargs_711618)
            
            
            # ################# End of 'run(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'run' in the type store
            # Getting the type of 'stypy_return_type' (line 185)
            stypy_return_type_711620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_711620)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'run'
            return stypy_return_type_711620

        # Assigning a type to the variable 'run' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'run', run)
        
        # Assigning a ListComp to a Name (line 190):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 190)
        # Processing the call arguments (line 190)
        int_711628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 63), 'int')
        # Processing the call keyword arguments (line 190)
        kwargs_711629 = {}
        # Getting the type of 'range' (line 190)
        range_711627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 57), 'range', False)
        # Calling range(args, kwargs) (line 190)
        range_call_result_711630 = invoke(stypy.reporting.localization.Localization(__file__, 190, 57), range_711627, *[int_711628], **kwargs_711629)
        
        comprehension_711631 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 19), range_call_result_711630)
        # Assigning a type to the variable 'j' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'j', comprehension_711631)
        
        # Call to Thread(...): (line 190)
        # Processing the call keyword arguments (line 190)
        # Getting the type of 'run' (line 190)
        run_711623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 43), 'run', False)
        keyword_711624 = run_711623
        kwargs_711625 = {'target': keyword_711624}
        # Getting the type of 'threading' (line 190)
        threading_711621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'threading', False)
        # Obtaining the member 'Thread' of a type (line 190)
        Thread_711622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 19), threading_711621, 'Thread')
        # Calling Thread(args, kwargs) (line 190)
        Thread_call_result_711626 = invoke(stypy.reporting.localization.Localization(__file__, 190, 19), Thread_711622, *[], **kwargs_711625)
        
        list_711632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 19), list_711632, Thread_call_result_711626)
        # Assigning a type to the variable 'threads' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'threads', list_711632)
        
        # Getting the type of 'threads' (line 191)
        threads_711633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 22), 'threads')
        # Testing the type of a for loop iterable (line 191)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 191, 8), threads_711633)
        # Getting the type of the for loop variable (line 191)
        for_loop_var_711634 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 191, 8), threads_711633)
        # Assigning a type to the variable 'thread' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'thread', for_loop_var_711634)
        # SSA begins for a for statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to start(...): (line 192)
        # Processing the call keyword arguments (line 192)
        kwargs_711637 = {}
        # Getting the type of 'thread' (line 192)
        thread_711635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'thread', False)
        # Obtaining the member 'start' of a type (line 192)
        start_711636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), thread_711635, 'start')
        # Calling start(args, kwargs) (line 192)
        start_call_result_711638 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), start_711636, *[], **kwargs_711637)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'threads' (line 193)
        threads_711639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 22), 'threads')
        # Testing the type of a for loop iterable (line 193)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 193, 8), threads_711639)
        # Getting the type of the for loop variable (line 193)
        for_loop_var_711640 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 193, 8), threads_711639)
        # Assigning a type to the variable 'thread' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'thread', for_loop_var_711640)
        # SSA begins for a for statement (line 193)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to join(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_711643 = {}
        # Getting the type of 'thread' (line 194)
        thread_711641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'thread', False)
        # Obtaining the member 'join' of a type (line 194)
        join_711642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), thread_711641, 'join')
        # Calling join(args, kwargs) (line 194)
        join_call_result_711644 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), join_711642, *[], **kwargs_711643)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'results' (line 196)
        results_711646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 21), 'results', False)
        
        # Obtaining an instance of the builtin type 'list' (line 196)
        list_711647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 196)
        # Adding element type (line 196)
        float_711648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 31), 'float')
        # Getting the type of 'count' (line 196)
        count_711649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 36), 'count', False)
        # Applying the binary operator '**' (line 196)
        result_pow_711650 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 31), '**', float_711648, count_711649)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 30), list_711647, result_pow_711650)
        
        
        # Call to len(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'threads' (line 196)
        threads_711652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 47), 'threads', False)
        # Processing the call keyword arguments (line 196)
        kwargs_711653 = {}
        # Getting the type of 'len' (line 196)
        len_711651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 43), 'len', False)
        # Calling len(args, kwargs) (line 196)
        len_call_result_711654 = invoke(stypy.reporting.localization.Localization(__file__, 196, 43), len_711651, *[threads_711652], **kwargs_711653)
        
        # Applying the binary operator '*' (line 196)
        result_mul_711655 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 30), '*', list_711647, len_call_result_711654)
        
        # Processing the call keyword arguments (line 196)
        kwargs_711656 = {}
        # Getting the type of 'assert_equal' (line 196)
        assert_equal_711645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 196)
        assert_equal_call_result_711657 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), assert_equal_711645, *[results_711646, result_mul_711655], **kwargs_711656)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_711658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711658)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_711658

    # Assigning a type to the variable 'check' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'check', check)
    
    
    # Call to keys(...): (line 198)
    # Processing the call keyword arguments (line 198)
    kwargs_711661 = {}
    # Getting the type of 'CALLERS' (line 198)
    CALLERS_711659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 18), 'CALLERS', False)
    # Obtaining the member 'keys' of a type (line 198)
    keys_711660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 18), CALLERS_711659, 'keys')
    # Calling keys(args, kwargs) (line 198)
    keys_call_result_711662 = invoke(stypy.reporting.localization.Localization(__file__, 198, 18), keys_711660, *[], **kwargs_711661)
    
    # Testing the type of a for loop iterable (line 198)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 198, 4), keys_call_result_711662)
    # Getting the type of the for loop variable (line 198)
    for_loop_var_711663 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 198, 4), keys_call_result_711662)
    # Assigning a type to the variable 'caller' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'caller', for_loop_var_711663)
    # SSA begins for a for statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'caller' (line 199)
    caller_711665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 14), 'caller', False)
    # Processing the call keyword arguments (line 199)
    kwargs_711666 = {}
    # Getting the type of 'check' (line 199)
    check_711664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'check', False)
    # Calling check(args, kwargs) (line 199)
    check_call_result_711667 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), check_711664, *[caller_711665], **kwargs_711666)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_threadsafety(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_threadsafety' in the type store
    # Getting the type of 'stypy_return_type' (line 170)
    stypy_return_type_711668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711668)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_threadsafety'
    return stypy_return_type_711668

# Assigning a type to the variable 'test_threadsafety' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'test_threadsafety', test_threadsafety)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
