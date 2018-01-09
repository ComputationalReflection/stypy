
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: 
5: from numpy.testing import assert_array_equal
6: from pytest import raises as assert_raises
7: 
8: from scipy.signal._arraytools import (axis_slice, axis_reverse,
9:      odd_ext, even_ext, const_ext, zero_ext)
10: 
11: 
12: class TestArrayTools(object):
13: 
14:     def test_axis_slice(self):
15:         a = np.arange(12).reshape(3, 4)
16: 
17:         s = axis_slice(a, start=0, stop=1, axis=0)
18:         assert_array_equal(s, a[0:1, :])
19: 
20:         s = axis_slice(a, start=-1, axis=0)
21:         assert_array_equal(s, a[-1:, :])
22: 
23:         s = axis_slice(a, start=0, stop=1, axis=1)
24:         assert_array_equal(s, a[:, 0:1])
25: 
26:         s = axis_slice(a, start=-1, axis=1)
27:         assert_array_equal(s, a[:, -1:])
28: 
29:         s = axis_slice(a, start=0, step=2, axis=0)
30:         assert_array_equal(s, a[::2, :])
31: 
32:         s = axis_slice(a, start=0, step=2, axis=1)
33:         assert_array_equal(s, a[:, ::2])
34: 
35:     def test_axis_reverse(self):
36:         a = np.arange(12).reshape(3, 4)
37: 
38:         r = axis_reverse(a, axis=0)
39:         assert_array_equal(r, a[::-1, :])
40: 
41:         r = axis_reverse(a, axis=1)
42:         assert_array_equal(r, a[:, ::-1])
43: 
44:     def test_odd_ext(self):
45:         a = np.array([[1, 2, 3, 4, 5],
46:                       [9, 8, 7, 6, 5]])
47: 
48:         odd = odd_ext(a, 2, axis=1)
49:         expected = np.array([[-1, 0, 1, 2, 3, 4, 5, 6, 7],
50:                              [11, 10, 9, 8, 7, 6, 5, 4, 3]])
51:         assert_array_equal(odd, expected)
52: 
53:         odd = odd_ext(a, 1, axis=0)
54:         expected = np.array([[-7, -4, -1, 2, 5],
55:                              [1, 2, 3, 4, 5],
56:                              [9, 8, 7, 6, 5],
57:                              [17, 14, 11, 8, 5]])
58:         assert_array_equal(odd, expected)
59: 
60:         assert_raises(ValueError, odd_ext, a, 2, axis=0)
61:         assert_raises(ValueError, odd_ext, a, 5, axis=1)
62: 
63:     def test_even_ext(self):
64:         a = np.array([[1, 2, 3, 4, 5],
65:                       [9, 8, 7, 6, 5]])
66: 
67:         even = even_ext(a, 2, axis=1)
68:         expected = np.array([[3, 2, 1, 2, 3, 4, 5, 4, 3],
69:                              [7, 8, 9, 8, 7, 6, 5, 6, 7]])
70:         assert_array_equal(even, expected)
71: 
72:         even = even_ext(a, 1, axis=0)
73:         expected = np.array([[9, 8, 7, 6, 5],
74:                              [1, 2, 3, 4, 5],
75:                              [9, 8, 7, 6, 5],
76:                              [1, 2, 3, 4, 5]])
77:         assert_array_equal(even, expected)
78: 
79:         assert_raises(ValueError, even_ext, a, 2, axis=0)
80:         assert_raises(ValueError, even_ext, a, 5, axis=1)
81: 
82:     def test_const_ext(self):
83:         a = np.array([[1, 2, 3, 4, 5],
84:                       [9, 8, 7, 6, 5]])
85: 
86:         const = const_ext(a, 2, axis=1)
87:         expected = np.array([[1, 1, 1, 2, 3, 4, 5, 5, 5],
88:                              [9, 9, 9, 8, 7, 6, 5, 5, 5]])
89:         assert_array_equal(const, expected)
90: 
91:         const = const_ext(a, 1, axis=0)
92:         expected = np.array([[1, 2, 3, 4, 5],
93:                              [1, 2, 3, 4, 5],
94:                              [9, 8, 7, 6, 5],
95:                              [9, 8, 7, 6, 5]])
96:         assert_array_equal(const, expected)
97: 
98:     def test_zero_ext(self):
99:         a = np.array([[1, 2, 3, 4, 5],
100:                       [9, 8, 7, 6, 5]])
101: 
102:         zero = zero_ext(a, 2, axis=1)
103:         expected = np.array([[0, 0, 1, 2, 3, 4, 5, 0, 0],
104:                              [0, 0, 9, 8, 7, 6, 5, 0, 0]])
105:         assert_array_equal(zero, expected)
106: 
107:         zero = zero_ext(a, 1, axis=0)
108:         expected = np.array([[0, 0, 0, 0, 0],
109:                              [1, 2, 3, 4, 5],
110:                              [9, 8, 7, 6, 5],
111:                              [0, 0, 0, 0, 0]])
112:         assert_array_equal(zero, expected)
113: 
114: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_289233 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_289233) is not StypyTypeError):

    if (import_289233 != 'pyd_module'):
        __import__(import_289233)
        sys_modules_289234 = sys.modules[import_289233]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_289234.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_289233)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.testing import assert_array_equal' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_289235 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing')

if (type(import_289235) is not StypyTypeError):

    if (import_289235 != 'pyd_module'):
        __import__(import_289235)
        sys_modules_289236 = sys.modules[import_289235]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', sys_modules_289236.module_type_store, module_type_store, ['assert_array_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_289236, sys_modules_289236.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', None, module_type_store, ['assert_array_equal'], [assert_array_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', import_289235)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from pytest import assert_raises' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_289237 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest')

if (type(import_289237) is not StypyTypeError):

    if (import_289237 != 'pyd_module'):
        __import__(import_289237)
        sys_modules_289238 = sys.modules[import_289237]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', sys_modules_289238.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_289238, sys_modules_289238.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', import_289237)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.signal._arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext, zero_ext' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_289239 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal._arraytools')

if (type(import_289239) is not StypyTypeError):

    if (import_289239 != 'pyd_module'):
        __import__(import_289239)
        sys_modules_289240 = sys.modules[import_289239]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal._arraytools', sys_modules_289240.module_type_store, module_type_store, ['axis_slice', 'axis_reverse', 'odd_ext', 'even_ext', 'const_ext', 'zero_ext'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_289240, sys_modules_289240.module_type_store, module_type_store)
    else:
        from scipy.signal._arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext, zero_ext

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal._arraytools', None, module_type_store, ['axis_slice', 'axis_reverse', 'odd_ext', 'even_ext', 'const_ext', 'zero_ext'], [axis_slice, axis_reverse, odd_ext, even_ext, const_ext, zero_ext])

else:
    # Assigning a type to the variable 'scipy.signal._arraytools' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal._arraytools', import_289239)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

# Declaration of the 'TestArrayTools' class

class TestArrayTools(object, ):

    @norecursion
    def test_axis_slice(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_axis_slice'
        module_type_store = module_type_store.open_function_context('test_axis_slice', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayTools.test_axis_slice.__dict__.__setitem__('stypy_localization', localization)
        TestArrayTools.test_axis_slice.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayTools.test_axis_slice.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayTools.test_axis_slice.__dict__.__setitem__('stypy_function_name', 'TestArrayTools.test_axis_slice')
        TestArrayTools.test_axis_slice.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayTools.test_axis_slice.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayTools.test_axis_slice.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayTools.test_axis_slice.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayTools.test_axis_slice.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayTools.test_axis_slice.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayTools.test_axis_slice.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayTools.test_axis_slice', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_axis_slice', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_axis_slice(...)' code ##################

        
        # Assigning a Call to a Name (line 15):
        
        # Call to reshape(...): (line 15)
        # Processing the call arguments (line 15)
        int_289247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'int')
        int_289248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 37), 'int')
        # Processing the call keyword arguments (line 15)
        kwargs_289249 = {}
        
        # Call to arange(...): (line 15)
        # Processing the call arguments (line 15)
        int_289243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 22), 'int')
        # Processing the call keyword arguments (line 15)
        kwargs_289244 = {}
        # Getting the type of 'np' (line 15)
        np_289241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 15)
        arange_289242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 12), np_289241, 'arange')
        # Calling arange(args, kwargs) (line 15)
        arange_call_result_289245 = invoke(stypy.reporting.localization.Localization(__file__, 15, 12), arange_289242, *[int_289243], **kwargs_289244)
        
        # Obtaining the member 'reshape' of a type (line 15)
        reshape_289246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 12), arange_call_result_289245, 'reshape')
        # Calling reshape(args, kwargs) (line 15)
        reshape_call_result_289250 = invoke(stypy.reporting.localization.Localization(__file__, 15, 12), reshape_289246, *[int_289247, int_289248], **kwargs_289249)
        
        # Assigning a type to the variable 'a' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'a', reshape_call_result_289250)
        
        # Assigning a Call to a Name (line 17):
        
        # Call to axis_slice(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'a' (line 17)
        a_289252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 'a', False)
        # Processing the call keyword arguments (line 17)
        int_289253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'int')
        keyword_289254 = int_289253
        int_289255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 40), 'int')
        keyword_289256 = int_289255
        int_289257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 48), 'int')
        keyword_289258 = int_289257
        kwargs_289259 = {'start': keyword_289254, 'stop': keyword_289256, 'axis': keyword_289258}
        # Getting the type of 'axis_slice' (line 17)
        axis_slice_289251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'axis_slice', False)
        # Calling axis_slice(args, kwargs) (line 17)
        axis_slice_call_result_289260 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), axis_slice_289251, *[a_289252], **kwargs_289259)
        
        # Assigning a type to the variable 's' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 's', axis_slice_call_result_289260)
        
        # Call to assert_array_equal(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 's' (line 18)
        s_289262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 27), 's', False)
        
        # Obtaining the type of the subscript
        int_289263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 32), 'int')
        int_289264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'int')
        slice_289265 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 18, 30), int_289263, int_289264, None)
        slice_289266 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 18, 30), None, None, None)
        # Getting the type of 'a' (line 18)
        a_289267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 30), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___289268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 30), a_289267, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_289269 = invoke(stypy.reporting.localization.Localization(__file__, 18, 30), getitem___289268, (slice_289265, slice_289266))
        
        # Processing the call keyword arguments (line 18)
        kwargs_289270 = {}
        # Getting the type of 'assert_array_equal' (line 18)
        assert_array_equal_289261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 18)
        assert_array_equal_call_result_289271 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), assert_array_equal_289261, *[s_289262, subscript_call_result_289269], **kwargs_289270)
        
        
        # Assigning a Call to a Name (line 20):
        
        # Call to axis_slice(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'a' (line 20)
        a_289273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 23), 'a', False)
        # Processing the call keyword arguments (line 20)
        int_289274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'int')
        keyword_289275 = int_289274
        int_289276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 41), 'int')
        keyword_289277 = int_289276
        kwargs_289278 = {'start': keyword_289275, 'axis': keyword_289277}
        # Getting the type of 'axis_slice' (line 20)
        axis_slice_289272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'axis_slice', False)
        # Calling axis_slice(args, kwargs) (line 20)
        axis_slice_call_result_289279 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), axis_slice_289272, *[a_289273], **kwargs_289278)
        
        # Assigning a type to the variable 's' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 's', axis_slice_call_result_289279)
        
        # Call to assert_array_equal(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 's' (line 21)
        s_289281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 27), 's', False)
        
        # Obtaining the type of the subscript
        int_289282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 32), 'int')
        slice_289283 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 21, 30), int_289282, None, None)
        slice_289284 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 21, 30), None, None, None)
        # Getting the type of 'a' (line 21)
        a_289285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 30), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 21)
        getitem___289286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 30), a_289285, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 21)
        subscript_call_result_289287 = invoke(stypy.reporting.localization.Localization(__file__, 21, 30), getitem___289286, (slice_289283, slice_289284))
        
        # Processing the call keyword arguments (line 21)
        kwargs_289288 = {}
        # Getting the type of 'assert_array_equal' (line 21)
        assert_array_equal_289280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 21)
        assert_array_equal_call_result_289289 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assert_array_equal_289280, *[s_289281, subscript_call_result_289287], **kwargs_289288)
        
        
        # Assigning a Call to a Name (line 23):
        
        # Call to axis_slice(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'a' (line 23)
        a_289291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'a', False)
        # Processing the call keyword arguments (line 23)
        int_289292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 32), 'int')
        keyword_289293 = int_289292
        int_289294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 40), 'int')
        keyword_289295 = int_289294
        int_289296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 48), 'int')
        keyword_289297 = int_289296
        kwargs_289298 = {'start': keyword_289293, 'stop': keyword_289295, 'axis': keyword_289297}
        # Getting the type of 'axis_slice' (line 23)
        axis_slice_289290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'axis_slice', False)
        # Calling axis_slice(args, kwargs) (line 23)
        axis_slice_call_result_289299 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), axis_slice_289290, *[a_289291], **kwargs_289298)
        
        # Assigning a type to the variable 's' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 's', axis_slice_call_result_289299)
        
        # Call to assert_array_equal(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 's' (line 24)
        s_289301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 27), 's', False)
        
        # Obtaining the type of the subscript
        slice_289302 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 24, 30), None, None, None)
        int_289303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 35), 'int')
        int_289304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 37), 'int')
        slice_289305 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 24, 30), int_289303, int_289304, None)
        # Getting the type of 'a' (line 24)
        a_289306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 30), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 24)
        getitem___289307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 30), a_289306, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 24)
        subscript_call_result_289308 = invoke(stypy.reporting.localization.Localization(__file__, 24, 30), getitem___289307, (slice_289302, slice_289305))
        
        # Processing the call keyword arguments (line 24)
        kwargs_289309 = {}
        # Getting the type of 'assert_array_equal' (line 24)
        assert_array_equal_289300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 24)
        assert_array_equal_call_result_289310 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), assert_array_equal_289300, *[s_289301, subscript_call_result_289308], **kwargs_289309)
        
        
        # Assigning a Call to a Name (line 26):
        
        # Call to axis_slice(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'a' (line 26)
        a_289312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'a', False)
        # Processing the call keyword arguments (line 26)
        int_289313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'int')
        keyword_289314 = int_289313
        int_289315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 41), 'int')
        keyword_289316 = int_289315
        kwargs_289317 = {'start': keyword_289314, 'axis': keyword_289316}
        # Getting the type of 'axis_slice' (line 26)
        axis_slice_289311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'axis_slice', False)
        # Calling axis_slice(args, kwargs) (line 26)
        axis_slice_call_result_289318 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), axis_slice_289311, *[a_289312], **kwargs_289317)
        
        # Assigning a type to the variable 's' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 's', axis_slice_call_result_289318)
        
        # Call to assert_array_equal(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 's' (line 27)
        s_289320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 27), 's', False)
        
        # Obtaining the type of the subscript
        slice_289321 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 27, 30), None, None, None)
        int_289322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 35), 'int')
        slice_289323 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 27, 30), int_289322, None, None)
        # Getting the type of 'a' (line 27)
        a_289324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 30), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 27)
        getitem___289325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 30), a_289324, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 27)
        subscript_call_result_289326 = invoke(stypy.reporting.localization.Localization(__file__, 27, 30), getitem___289325, (slice_289321, slice_289323))
        
        # Processing the call keyword arguments (line 27)
        kwargs_289327 = {}
        # Getting the type of 'assert_array_equal' (line 27)
        assert_array_equal_289319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 27)
        assert_array_equal_call_result_289328 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assert_array_equal_289319, *[s_289320, subscript_call_result_289326], **kwargs_289327)
        
        
        # Assigning a Call to a Name (line 29):
        
        # Call to axis_slice(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'a' (line 29)
        a_289330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'a', False)
        # Processing the call keyword arguments (line 29)
        int_289331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 32), 'int')
        keyword_289332 = int_289331
        int_289333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 40), 'int')
        keyword_289334 = int_289333
        int_289335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 48), 'int')
        keyword_289336 = int_289335
        kwargs_289337 = {'start': keyword_289332, 'step': keyword_289334, 'axis': keyword_289336}
        # Getting the type of 'axis_slice' (line 29)
        axis_slice_289329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'axis_slice', False)
        # Calling axis_slice(args, kwargs) (line 29)
        axis_slice_call_result_289338 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), axis_slice_289329, *[a_289330], **kwargs_289337)
        
        # Assigning a type to the variable 's' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 's', axis_slice_call_result_289338)
        
        # Call to assert_array_equal(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 's' (line 30)
        s_289340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 27), 's', False)
        
        # Obtaining the type of the subscript
        int_289341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'int')
        slice_289342 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 30, 30), None, None, int_289341)
        slice_289343 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 30, 30), None, None, None)
        # Getting the type of 'a' (line 30)
        a_289344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___289345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 30), a_289344, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_289346 = invoke(stypy.reporting.localization.Localization(__file__, 30, 30), getitem___289345, (slice_289342, slice_289343))
        
        # Processing the call keyword arguments (line 30)
        kwargs_289347 = {}
        # Getting the type of 'assert_array_equal' (line 30)
        assert_array_equal_289339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 30)
        assert_array_equal_call_result_289348 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assert_array_equal_289339, *[s_289340, subscript_call_result_289346], **kwargs_289347)
        
        
        # Assigning a Call to a Name (line 32):
        
        # Call to axis_slice(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'a' (line 32)
        a_289350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'a', False)
        # Processing the call keyword arguments (line 32)
        int_289351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 32), 'int')
        keyword_289352 = int_289351
        int_289353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 40), 'int')
        keyword_289354 = int_289353
        int_289355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 48), 'int')
        keyword_289356 = int_289355
        kwargs_289357 = {'start': keyword_289352, 'step': keyword_289354, 'axis': keyword_289356}
        # Getting the type of 'axis_slice' (line 32)
        axis_slice_289349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'axis_slice', False)
        # Calling axis_slice(args, kwargs) (line 32)
        axis_slice_call_result_289358 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), axis_slice_289349, *[a_289350], **kwargs_289357)
        
        # Assigning a type to the variable 's' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 's', axis_slice_call_result_289358)
        
        # Call to assert_array_equal(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 's' (line 33)
        s_289360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 27), 's', False)
        
        # Obtaining the type of the subscript
        slice_289361 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 33, 30), None, None, None)
        int_289362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 37), 'int')
        slice_289363 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 33, 30), None, None, int_289362)
        # Getting the type of 'a' (line 33)
        a_289364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 30), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 33)
        getitem___289365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 30), a_289364, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 33)
        subscript_call_result_289366 = invoke(stypy.reporting.localization.Localization(__file__, 33, 30), getitem___289365, (slice_289361, slice_289363))
        
        # Processing the call keyword arguments (line 33)
        kwargs_289367 = {}
        # Getting the type of 'assert_array_equal' (line 33)
        assert_array_equal_289359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 33)
        assert_array_equal_call_result_289368 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), assert_array_equal_289359, *[s_289360, subscript_call_result_289366], **kwargs_289367)
        
        
        # ################# End of 'test_axis_slice(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_axis_slice' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_289369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_289369)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_axis_slice'
        return stypy_return_type_289369


    @norecursion
    def test_axis_reverse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_axis_reverse'
        module_type_store = module_type_store.open_function_context('test_axis_reverse', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayTools.test_axis_reverse.__dict__.__setitem__('stypy_localization', localization)
        TestArrayTools.test_axis_reverse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayTools.test_axis_reverse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayTools.test_axis_reverse.__dict__.__setitem__('stypy_function_name', 'TestArrayTools.test_axis_reverse')
        TestArrayTools.test_axis_reverse.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayTools.test_axis_reverse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayTools.test_axis_reverse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayTools.test_axis_reverse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayTools.test_axis_reverse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayTools.test_axis_reverse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayTools.test_axis_reverse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayTools.test_axis_reverse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_axis_reverse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_axis_reverse(...)' code ##################

        
        # Assigning a Call to a Name (line 36):
        
        # Call to reshape(...): (line 36)
        # Processing the call arguments (line 36)
        int_289376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 34), 'int')
        int_289377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 37), 'int')
        # Processing the call keyword arguments (line 36)
        kwargs_289378 = {}
        
        # Call to arange(...): (line 36)
        # Processing the call arguments (line 36)
        int_289372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 22), 'int')
        # Processing the call keyword arguments (line 36)
        kwargs_289373 = {}
        # Getting the type of 'np' (line 36)
        np_289370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 36)
        arange_289371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), np_289370, 'arange')
        # Calling arange(args, kwargs) (line 36)
        arange_call_result_289374 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), arange_289371, *[int_289372], **kwargs_289373)
        
        # Obtaining the member 'reshape' of a type (line 36)
        reshape_289375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), arange_call_result_289374, 'reshape')
        # Calling reshape(args, kwargs) (line 36)
        reshape_call_result_289379 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), reshape_289375, *[int_289376, int_289377], **kwargs_289378)
        
        # Assigning a type to the variable 'a' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'a', reshape_call_result_289379)
        
        # Assigning a Call to a Name (line 38):
        
        # Call to axis_reverse(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'a' (line 38)
        a_289381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'a', False)
        # Processing the call keyword arguments (line 38)
        int_289382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 33), 'int')
        keyword_289383 = int_289382
        kwargs_289384 = {'axis': keyword_289383}
        # Getting the type of 'axis_reverse' (line 38)
        axis_reverse_289380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'axis_reverse', False)
        # Calling axis_reverse(args, kwargs) (line 38)
        axis_reverse_call_result_289385 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), axis_reverse_289380, *[a_289381], **kwargs_289384)
        
        # Assigning a type to the variable 'r' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'r', axis_reverse_call_result_289385)
        
        # Call to assert_array_equal(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'r' (line 39)
        r_289387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'r', False)
        
        # Obtaining the type of the subscript
        int_289388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 34), 'int')
        slice_289389 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 39, 30), None, None, int_289388)
        slice_289390 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 39, 30), None, None, None)
        # Getting the type of 'a' (line 39)
        a_289391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 30), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___289392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 30), a_289391, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_289393 = invoke(stypy.reporting.localization.Localization(__file__, 39, 30), getitem___289392, (slice_289389, slice_289390))
        
        # Processing the call keyword arguments (line 39)
        kwargs_289394 = {}
        # Getting the type of 'assert_array_equal' (line 39)
        assert_array_equal_289386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 39)
        assert_array_equal_call_result_289395 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), assert_array_equal_289386, *[r_289387, subscript_call_result_289393], **kwargs_289394)
        
        
        # Assigning a Call to a Name (line 41):
        
        # Call to axis_reverse(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'a' (line 41)
        a_289397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'a', False)
        # Processing the call keyword arguments (line 41)
        int_289398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 33), 'int')
        keyword_289399 = int_289398
        kwargs_289400 = {'axis': keyword_289399}
        # Getting the type of 'axis_reverse' (line 41)
        axis_reverse_289396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'axis_reverse', False)
        # Calling axis_reverse(args, kwargs) (line 41)
        axis_reverse_call_result_289401 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), axis_reverse_289396, *[a_289397], **kwargs_289400)
        
        # Assigning a type to the variable 'r' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'r', axis_reverse_call_result_289401)
        
        # Call to assert_array_equal(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'r' (line 42)
        r_289403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'r', False)
        
        # Obtaining the type of the subscript
        slice_289404 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 30), None, None, None)
        int_289405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 37), 'int')
        slice_289406 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 30), None, None, int_289405)
        # Getting the type of 'a' (line 42)
        a_289407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 30), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___289408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 30), a_289407, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_289409 = invoke(stypy.reporting.localization.Localization(__file__, 42, 30), getitem___289408, (slice_289404, slice_289406))
        
        # Processing the call keyword arguments (line 42)
        kwargs_289410 = {}
        # Getting the type of 'assert_array_equal' (line 42)
        assert_array_equal_289402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 42)
        assert_array_equal_call_result_289411 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assert_array_equal_289402, *[r_289403, subscript_call_result_289409], **kwargs_289410)
        
        
        # ################# End of 'test_axis_reverse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_axis_reverse' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_289412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_289412)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_axis_reverse'
        return stypy_return_type_289412


    @norecursion
    def test_odd_ext(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_odd_ext'
        module_type_store = module_type_store.open_function_context('test_odd_ext', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayTools.test_odd_ext.__dict__.__setitem__('stypy_localization', localization)
        TestArrayTools.test_odd_ext.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayTools.test_odd_ext.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayTools.test_odd_ext.__dict__.__setitem__('stypy_function_name', 'TestArrayTools.test_odd_ext')
        TestArrayTools.test_odd_ext.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayTools.test_odd_ext.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayTools.test_odd_ext.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayTools.test_odd_ext.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayTools.test_odd_ext.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayTools.test_odd_ext.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayTools.test_odd_ext.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayTools.test_odd_ext', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_odd_ext', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_odd_ext(...)' code ##################

        
        # Assigning a Call to a Name (line 45):
        
        # Call to array(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Obtaining an instance of the builtin type 'list' (line 45)
        list_289415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 45)
        # Adding element type (line 45)
        
        # Obtaining an instance of the builtin type 'list' (line 45)
        list_289416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 45)
        # Adding element type (line 45)
        int_289417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 22), list_289416, int_289417)
        # Adding element type (line 45)
        int_289418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 22), list_289416, int_289418)
        # Adding element type (line 45)
        int_289419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 22), list_289416, int_289419)
        # Adding element type (line 45)
        int_289420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 22), list_289416, int_289420)
        # Adding element type (line 45)
        int_289421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 22), list_289416, int_289421)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 21), list_289415, list_289416)
        # Adding element type (line 45)
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_289422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        int_289423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_289422, int_289423)
        # Adding element type (line 46)
        int_289424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_289422, int_289424)
        # Adding element type (line 46)
        int_289425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_289422, int_289425)
        # Adding element type (line 46)
        int_289426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_289422, int_289426)
        # Adding element type (line 46)
        int_289427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), list_289422, int_289427)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 21), list_289415, list_289422)
        
        # Processing the call keyword arguments (line 45)
        kwargs_289428 = {}
        # Getting the type of 'np' (line 45)
        np_289413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 45)
        array_289414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), np_289413, 'array')
        # Calling array(args, kwargs) (line 45)
        array_call_result_289429 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), array_289414, *[list_289415], **kwargs_289428)
        
        # Assigning a type to the variable 'a' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'a', array_call_result_289429)
        
        # Assigning a Call to a Name (line 48):
        
        # Call to odd_ext(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'a' (line 48)
        a_289431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'a', False)
        int_289432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 25), 'int')
        # Processing the call keyword arguments (line 48)
        int_289433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 33), 'int')
        keyword_289434 = int_289433
        kwargs_289435 = {'axis': keyword_289434}
        # Getting the type of 'odd_ext' (line 48)
        odd_ext_289430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'odd_ext', False)
        # Calling odd_ext(args, kwargs) (line 48)
        odd_ext_call_result_289436 = invoke(stypy.reporting.localization.Localization(__file__, 48, 14), odd_ext_289430, *[a_289431, int_289432], **kwargs_289435)
        
        # Assigning a type to the variable 'odd' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'odd', odd_ext_call_result_289436)
        
        # Assigning a Call to a Name (line 49):
        
        # Call to array(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_289439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        # Adding element type (line 49)
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_289440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        # Adding element type (line 49)
        int_289441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), list_289440, int_289441)
        # Adding element type (line 49)
        int_289442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), list_289440, int_289442)
        # Adding element type (line 49)
        int_289443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), list_289440, int_289443)
        # Adding element type (line 49)
        int_289444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), list_289440, int_289444)
        # Adding element type (line 49)
        int_289445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), list_289440, int_289445)
        # Adding element type (line 49)
        int_289446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), list_289440, int_289446)
        # Adding element type (line 49)
        int_289447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), list_289440, int_289447)
        # Adding element type (line 49)
        int_289448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), list_289440, int_289448)
        # Adding element type (line 49)
        int_289449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), list_289440, int_289449)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 28), list_289439, list_289440)
        # Adding element type (line 49)
        
        # Obtaining an instance of the builtin type 'list' (line 50)
        list_289450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 50)
        # Adding element type (line 50)
        int_289451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), list_289450, int_289451)
        # Adding element type (line 50)
        int_289452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), list_289450, int_289452)
        # Adding element type (line 50)
        int_289453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), list_289450, int_289453)
        # Adding element type (line 50)
        int_289454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), list_289450, int_289454)
        # Adding element type (line 50)
        int_289455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), list_289450, int_289455)
        # Adding element type (line 50)
        int_289456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), list_289450, int_289456)
        # Adding element type (line 50)
        int_289457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), list_289450, int_289457)
        # Adding element type (line 50)
        int_289458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), list_289450, int_289458)
        # Adding element type (line 50)
        int_289459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), list_289450, int_289459)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 28), list_289439, list_289450)
        
        # Processing the call keyword arguments (line 49)
        kwargs_289460 = {}
        # Getting the type of 'np' (line 49)
        np_289437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 49)
        array_289438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 19), np_289437, 'array')
        # Calling array(args, kwargs) (line 49)
        array_call_result_289461 = invoke(stypy.reporting.localization.Localization(__file__, 49, 19), array_289438, *[list_289439], **kwargs_289460)
        
        # Assigning a type to the variable 'expected' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'expected', array_call_result_289461)
        
        # Call to assert_array_equal(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'odd' (line 51)
        odd_289463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 27), 'odd', False)
        # Getting the type of 'expected' (line 51)
        expected_289464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 32), 'expected', False)
        # Processing the call keyword arguments (line 51)
        kwargs_289465 = {}
        # Getting the type of 'assert_array_equal' (line 51)
        assert_array_equal_289462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 51)
        assert_array_equal_call_result_289466 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assert_array_equal_289462, *[odd_289463, expected_289464], **kwargs_289465)
        
        
        # Assigning a Call to a Name (line 53):
        
        # Call to odd_ext(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'a' (line 53)
        a_289468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'a', False)
        int_289469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'int')
        # Processing the call keyword arguments (line 53)
        int_289470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 33), 'int')
        keyword_289471 = int_289470
        kwargs_289472 = {'axis': keyword_289471}
        # Getting the type of 'odd_ext' (line 53)
        odd_ext_289467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'odd_ext', False)
        # Calling odd_ext(args, kwargs) (line 53)
        odd_ext_call_result_289473 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), odd_ext_289467, *[a_289468, int_289469], **kwargs_289472)
        
        # Assigning a type to the variable 'odd' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'odd', odd_ext_call_result_289473)
        
        # Assigning a Call to a Name (line 54):
        
        # Call to array(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_289476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_289477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_289478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 29), list_289477, int_289478)
        # Adding element type (line 54)
        int_289479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 29), list_289477, int_289479)
        # Adding element type (line 54)
        int_289480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 29), list_289477, int_289480)
        # Adding element type (line 54)
        int_289481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 29), list_289477, int_289481)
        # Adding element type (line 54)
        int_289482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 29), list_289477, int_289482)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 28), list_289476, list_289477)
        # Adding element type (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_289483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        int_289484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 29), list_289483, int_289484)
        # Adding element type (line 55)
        int_289485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 29), list_289483, int_289485)
        # Adding element type (line 55)
        int_289486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 29), list_289483, int_289486)
        # Adding element type (line 55)
        int_289487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 29), list_289483, int_289487)
        # Adding element type (line 55)
        int_289488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 29), list_289483, int_289488)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 28), list_289476, list_289483)
        # Adding element type (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_289489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        int_289490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 29), list_289489, int_289490)
        # Adding element type (line 56)
        int_289491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 29), list_289489, int_289491)
        # Adding element type (line 56)
        int_289492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 29), list_289489, int_289492)
        # Adding element type (line 56)
        int_289493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 29), list_289489, int_289493)
        # Adding element type (line 56)
        int_289494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 29), list_289489, int_289494)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 28), list_289476, list_289489)
        # Adding element type (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_289495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        int_289496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 29), list_289495, int_289496)
        # Adding element type (line 57)
        int_289497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 29), list_289495, int_289497)
        # Adding element type (line 57)
        int_289498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 29), list_289495, int_289498)
        # Adding element type (line 57)
        int_289499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 29), list_289495, int_289499)
        # Adding element type (line 57)
        int_289500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 29), list_289495, int_289500)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 28), list_289476, list_289495)
        
        # Processing the call keyword arguments (line 54)
        kwargs_289501 = {}
        # Getting the type of 'np' (line 54)
        np_289474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 54)
        array_289475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 19), np_289474, 'array')
        # Calling array(args, kwargs) (line 54)
        array_call_result_289502 = invoke(stypy.reporting.localization.Localization(__file__, 54, 19), array_289475, *[list_289476], **kwargs_289501)
        
        # Assigning a type to the variable 'expected' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'expected', array_call_result_289502)
        
        # Call to assert_array_equal(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'odd' (line 58)
        odd_289504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'odd', False)
        # Getting the type of 'expected' (line 58)
        expected_289505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'expected', False)
        # Processing the call keyword arguments (line 58)
        kwargs_289506 = {}
        # Getting the type of 'assert_array_equal' (line 58)
        assert_array_equal_289503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 58)
        assert_array_equal_call_result_289507 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), assert_array_equal_289503, *[odd_289504, expected_289505], **kwargs_289506)
        
        
        # Call to assert_raises(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'ValueError' (line 60)
        ValueError_289509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'ValueError', False)
        # Getting the type of 'odd_ext' (line 60)
        odd_ext_289510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'odd_ext', False)
        # Getting the type of 'a' (line 60)
        a_289511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 43), 'a', False)
        int_289512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 46), 'int')
        # Processing the call keyword arguments (line 60)
        int_289513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 54), 'int')
        keyword_289514 = int_289513
        kwargs_289515 = {'axis': keyword_289514}
        # Getting the type of 'assert_raises' (line 60)
        assert_raises_289508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 60)
        assert_raises_call_result_289516 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assert_raises_289508, *[ValueError_289509, odd_ext_289510, a_289511, int_289512], **kwargs_289515)
        
        
        # Call to assert_raises(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'ValueError' (line 61)
        ValueError_289518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 22), 'ValueError', False)
        # Getting the type of 'odd_ext' (line 61)
        odd_ext_289519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'odd_ext', False)
        # Getting the type of 'a' (line 61)
        a_289520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 43), 'a', False)
        int_289521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 46), 'int')
        # Processing the call keyword arguments (line 61)
        int_289522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 54), 'int')
        keyword_289523 = int_289522
        kwargs_289524 = {'axis': keyword_289523}
        # Getting the type of 'assert_raises' (line 61)
        assert_raises_289517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 61)
        assert_raises_call_result_289525 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assert_raises_289517, *[ValueError_289518, odd_ext_289519, a_289520, int_289521], **kwargs_289524)
        
        
        # ################# End of 'test_odd_ext(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_odd_ext' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_289526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_289526)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_odd_ext'
        return stypy_return_type_289526


    @norecursion
    def test_even_ext(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_even_ext'
        module_type_store = module_type_store.open_function_context('test_even_ext', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayTools.test_even_ext.__dict__.__setitem__('stypy_localization', localization)
        TestArrayTools.test_even_ext.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayTools.test_even_ext.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayTools.test_even_ext.__dict__.__setitem__('stypy_function_name', 'TestArrayTools.test_even_ext')
        TestArrayTools.test_even_ext.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayTools.test_even_ext.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayTools.test_even_ext.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayTools.test_even_ext.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayTools.test_even_ext.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayTools.test_even_ext.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayTools.test_even_ext.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayTools.test_even_ext', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_even_ext', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_even_ext(...)' code ##################

        
        # Assigning a Call to a Name (line 64):
        
        # Call to array(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_289529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_289530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        int_289531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 22), list_289530, int_289531)
        # Adding element type (line 64)
        int_289532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 22), list_289530, int_289532)
        # Adding element type (line 64)
        int_289533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 22), list_289530, int_289533)
        # Adding element type (line 64)
        int_289534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 22), list_289530, int_289534)
        # Adding element type (line 64)
        int_289535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 22), list_289530, int_289535)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 21), list_289529, list_289530)
        # Adding element type (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_289536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        int_289537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), list_289536, int_289537)
        # Adding element type (line 65)
        int_289538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), list_289536, int_289538)
        # Adding element type (line 65)
        int_289539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), list_289536, int_289539)
        # Adding element type (line 65)
        int_289540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), list_289536, int_289540)
        # Adding element type (line 65)
        int_289541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), list_289536, int_289541)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 21), list_289529, list_289536)
        
        # Processing the call keyword arguments (line 64)
        kwargs_289542 = {}
        # Getting the type of 'np' (line 64)
        np_289527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 64)
        array_289528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), np_289527, 'array')
        # Calling array(args, kwargs) (line 64)
        array_call_result_289543 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), array_289528, *[list_289529], **kwargs_289542)
        
        # Assigning a type to the variable 'a' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'a', array_call_result_289543)
        
        # Assigning a Call to a Name (line 67):
        
        # Call to even_ext(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'a' (line 67)
        a_289545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'a', False)
        int_289546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'int')
        # Processing the call keyword arguments (line 67)
        int_289547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 35), 'int')
        keyword_289548 = int_289547
        kwargs_289549 = {'axis': keyword_289548}
        # Getting the type of 'even_ext' (line 67)
        even_ext_289544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'even_ext', False)
        # Calling even_ext(args, kwargs) (line 67)
        even_ext_call_result_289550 = invoke(stypy.reporting.localization.Localization(__file__, 67, 15), even_ext_289544, *[a_289545, int_289546], **kwargs_289549)
        
        # Assigning a type to the variable 'even' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'even', even_ext_call_result_289550)
        
        # Assigning a Call to a Name (line 68):
        
        # Call to array(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_289553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_289554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        int_289555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 29), list_289554, int_289555)
        # Adding element type (line 68)
        int_289556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 29), list_289554, int_289556)
        # Adding element type (line 68)
        int_289557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 29), list_289554, int_289557)
        # Adding element type (line 68)
        int_289558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 29), list_289554, int_289558)
        # Adding element type (line 68)
        int_289559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 29), list_289554, int_289559)
        # Adding element type (line 68)
        int_289560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 29), list_289554, int_289560)
        # Adding element type (line 68)
        int_289561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 29), list_289554, int_289561)
        # Adding element type (line 68)
        int_289562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 29), list_289554, int_289562)
        # Adding element type (line 68)
        int_289563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 29), list_289554, int_289563)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), list_289553, list_289554)
        # Adding element type (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_289564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        int_289565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_289564, int_289565)
        # Adding element type (line 69)
        int_289566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_289564, int_289566)
        # Adding element type (line 69)
        int_289567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_289564, int_289567)
        # Adding element type (line 69)
        int_289568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_289564, int_289568)
        # Adding element type (line 69)
        int_289569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_289564, int_289569)
        # Adding element type (line 69)
        int_289570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_289564, int_289570)
        # Adding element type (line 69)
        int_289571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_289564, int_289571)
        # Adding element type (line 69)
        int_289572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_289564, int_289572)
        # Adding element type (line 69)
        int_289573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_289564, int_289573)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), list_289553, list_289564)
        
        # Processing the call keyword arguments (line 68)
        kwargs_289574 = {}
        # Getting the type of 'np' (line 68)
        np_289551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 68)
        array_289552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), np_289551, 'array')
        # Calling array(args, kwargs) (line 68)
        array_call_result_289575 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), array_289552, *[list_289553], **kwargs_289574)
        
        # Assigning a type to the variable 'expected' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'expected', array_call_result_289575)
        
        # Call to assert_array_equal(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'even' (line 70)
        even_289577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 27), 'even', False)
        # Getting the type of 'expected' (line 70)
        expected_289578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 33), 'expected', False)
        # Processing the call keyword arguments (line 70)
        kwargs_289579 = {}
        # Getting the type of 'assert_array_equal' (line 70)
        assert_array_equal_289576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 70)
        assert_array_equal_call_result_289580 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), assert_array_equal_289576, *[even_289577, expected_289578], **kwargs_289579)
        
        
        # Assigning a Call to a Name (line 72):
        
        # Call to even_ext(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'a' (line 72)
        a_289582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'a', False)
        int_289583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 27), 'int')
        # Processing the call keyword arguments (line 72)
        int_289584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 35), 'int')
        keyword_289585 = int_289584
        kwargs_289586 = {'axis': keyword_289585}
        # Getting the type of 'even_ext' (line 72)
        even_ext_289581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'even_ext', False)
        # Calling even_ext(args, kwargs) (line 72)
        even_ext_call_result_289587 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), even_ext_289581, *[a_289582, int_289583], **kwargs_289586)
        
        # Assigning a type to the variable 'even' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'even', even_ext_call_result_289587)
        
        # Assigning a Call to a Name (line 73):
        
        # Call to array(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_289590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_289591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        int_289592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 29), list_289591, int_289592)
        # Adding element type (line 73)
        int_289593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 29), list_289591, int_289593)
        # Adding element type (line 73)
        int_289594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 29), list_289591, int_289594)
        # Adding element type (line 73)
        int_289595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 29), list_289591, int_289595)
        # Adding element type (line 73)
        int_289596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 29), list_289591, int_289596)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), list_289590, list_289591)
        # Adding element type (line 73)
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_289597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        int_289598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 29), list_289597, int_289598)
        # Adding element type (line 74)
        int_289599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 29), list_289597, int_289599)
        # Adding element type (line 74)
        int_289600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 29), list_289597, int_289600)
        # Adding element type (line 74)
        int_289601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 29), list_289597, int_289601)
        # Adding element type (line 74)
        int_289602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 29), list_289597, int_289602)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), list_289590, list_289597)
        # Adding element type (line 73)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_289603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        int_289604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 29), list_289603, int_289604)
        # Adding element type (line 75)
        int_289605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 29), list_289603, int_289605)
        # Adding element type (line 75)
        int_289606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 29), list_289603, int_289606)
        # Adding element type (line 75)
        int_289607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 29), list_289603, int_289607)
        # Adding element type (line 75)
        int_289608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 29), list_289603, int_289608)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), list_289590, list_289603)
        # Adding element type (line 73)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_289609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        int_289610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 29), list_289609, int_289610)
        # Adding element type (line 76)
        int_289611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 29), list_289609, int_289611)
        # Adding element type (line 76)
        int_289612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 29), list_289609, int_289612)
        # Adding element type (line 76)
        int_289613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 29), list_289609, int_289613)
        # Adding element type (line 76)
        int_289614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 29), list_289609, int_289614)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), list_289590, list_289609)
        
        # Processing the call keyword arguments (line 73)
        kwargs_289615 = {}
        # Getting the type of 'np' (line 73)
        np_289588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 73)
        array_289589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 19), np_289588, 'array')
        # Calling array(args, kwargs) (line 73)
        array_call_result_289616 = invoke(stypy.reporting.localization.Localization(__file__, 73, 19), array_289589, *[list_289590], **kwargs_289615)
        
        # Assigning a type to the variable 'expected' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'expected', array_call_result_289616)
        
        # Call to assert_array_equal(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'even' (line 77)
        even_289618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'even', False)
        # Getting the type of 'expected' (line 77)
        expected_289619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 33), 'expected', False)
        # Processing the call keyword arguments (line 77)
        kwargs_289620 = {}
        # Getting the type of 'assert_array_equal' (line 77)
        assert_array_equal_289617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 77)
        assert_array_equal_call_result_289621 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), assert_array_equal_289617, *[even_289618, expected_289619], **kwargs_289620)
        
        
        # Call to assert_raises(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'ValueError' (line 79)
        ValueError_289623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 22), 'ValueError', False)
        # Getting the type of 'even_ext' (line 79)
        even_ext_289624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 34), 'even_ext', False)
        # Getting the type of 'a' (line 79)
        a_289625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 44), 'a', False)
        int_289626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 47), 'int')
        # Processing the call keyword arguments (line 79)
        int_289627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 55), 'int')
        keyword_289628 = int_289627
        kwargs_289629 = {'axis': keyword_289628}
        # Getting the type of 'assert_raises' (line 79)
        assert_raises_289622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 79)
        assert_raises_call_result_289630 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), assert_raises_289622, *[ValueError_289623, even_ext_289624, a_289625, int_289626], **kwargs_289629)
        
        
        # Call to assert_raises(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'ValueError' (line 80)
        ValueError_289632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'ValueError', False)
        # Getting the type of 'even_ext' (line 80)
        even_ext_289633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 'even_ext', False)
        # Getting the type of 'a' (line 80)
        a_289634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'a', False)
        int_289635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 47), 'int')
        # Processing the call keyword arguments (line 80)
        int_289636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 55), 'int')
        keyword_289637 = int_289636
        kwargs_289638 = {'axis': keyword_289637}
        # Getting the type of 'assert_raises' (line 80)
        assert_raises_289631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 80)
        assert_raises_call_result_289639 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assert_raises_289631, *[ValueError_289632, even_ext_289633, a_289634, int_289635], **kwargs_289638)
        
        
        # ################# End of 'test_even_ext(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_even_ext' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_289640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_289640)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_even_ext'
        return stypy_return_type_289640


    @norecursion
    def test_const_ext(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_const_ext'
        module_type_store = module_type_store.open_function_context('test_const_ext', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayTools.test_const_ext.__dict__.__setitem__('stypy_localization', localization)
        TestArrayTools.test_const_ext.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayTools.test_const_ext.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayTools.test_const_ext.__dict__.__setitem__('stypy_function_name', 'TestArrayTools.test_const_ext')
        TestArrayTools.test_const_ext.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayTools.test_const_ext.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayTools.test_const_ext.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayTools.test_const_ext.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayTools.test_const_ext.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayTools.test_const_ext.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayTools.test_const_ext.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayTools.test_const_ext', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_const_ext', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_const_ext(...)' code ##################

        
        # Assigning a Call to a Name (line 83):
        
        # Call to array(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_289643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_289644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        int_289645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 22), list_289644, int_289645)
        # Adding element type (line 83)
        int_289646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 22), list_289644, int_289646)
        # Adding element type (line 83)
        int_289647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 22), list_289644, int_289647)
        # Adding element type (line 83)
        int_289648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 22), list_289644, int_289648)
        # Adding element type (line 83)
        int_289649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 22), list_289644, int_289649)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_289643, list_289644)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_289650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        int_289651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), list_289650, int_289651)
        # Adding element type (line 84)
        int_289652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), list_289650, int_289652)
        # Adding element type (line 84)
        int_289653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), list_289650, int_289653)
        # Adding element type (line 84)
        int_289654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), list_289650, int_289654)
        # Adding element type (line 84)
        int_289655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), list_289650, int_289655)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_289643, list_289650)
        
        # Processing the call keyword arguments (line 83)
        kwargs_289656 = {}
        # Getting the type of 'np' (line 83)
        np_289641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 83)
        array_289642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), np_289641, 'array')
        # Calling array(args, kwargs) (line 83)
        array_call_result_289657 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), array_289642, *[list_289643], **kwargs_289656)
        
        # Assigning a type to the variable 'a' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'a', array_call_result_289657)
        
        # Assigning a Call to a Name (line 86):
        
        # Call to const_ext(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'a' (line 86)
        a_289659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 26), 'a', False)
        int_289660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'int')
        # Processing the call keyword arguments (line 86)
        int_289661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 37), 'int')
        keyword_289662 = int_289661
        kwargs_289663 = {'axis': keyword_289662}
        # Getting the type of 'const_ext' (line 86)
        const_ext_289658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'const_ext', False)
        # Calling const_ext(args, kwargs) (line 86)
        const_ext_call_result_289664 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), const_ext_289658, *[a_289659, int_289660], **kwargs_289663)
        
        # Assigning a type to the variable 'const' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'const', const_ext_call_result_289664)
        
        # Assigning a Call to a Name (line 87):
        
        # Call to array(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_289667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_289668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        int_289669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 29), list_289668, int_289669)
        # Adding element type (line 87)
        int_289670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 29), list_289668, int_289670)
        # Adding element type (line 87)
        int_289671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 29), list_289668, int_289671)
        # Adding element type (line 87)
        int_289672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 29), list_289668, int_289672)
        # Adding element type (line 87)
        int_289673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 29), list_289668, int_289673)
        # Adding element type (line 87)
        int_289674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 29), list_289668, int_289674)
        # Adding element type (line 87)
        int_289675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 29), list_289668, int_289675)
        # Adding element type (line 87)
        int_289676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 29), list_289668, int_289676)
        # Adding element type (line 87)
        int_289677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 29), list_289668, int_289677)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 28), list_289667, list_289668)
        # Adding element type (line 87)
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_289678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        # Adding element type (line 88)
        int_289679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 29), list_289678, int_289679)
        # Adding element type (line 88)
        int_289680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 29), list_289678, int_289680)
        # Adding element type (line 88)
        int_289681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 29), list_289678, int_289681)
        # Adding element type (line 88)
        int_289682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 29), list_289678, int_289682)
        # Adding element type (line 88)
        int_289683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 29), list_289678, int_289683)
        # Adding element type (line 88)
        int_289684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 29), list_289678, int_289684)
        # Adding element type (line 88)
        int_289685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 29), list_289678, int_289685)
        # Adding element type (line 88)
        int_289686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 29), list_289678, int_289686)
        # Adding element type (line 88)
        int_289687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 29), list_289678, int_289687)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 28), list_289667, list_289678)
        
        # Processing the call keyword arguments (line 87)
        kwargs_289688 = {}
        # Getting the type of 'np' (line 87)
        np_289665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 87)
        array_289666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 19), np_289665, 'array')
        # Calling array(args, kwargs) (line 87)
        array_call_result_289689 = invoke(stypy.reporting.localization.Localization(__file__, 87, 19), array_289666, *[list_289667], **kwargs_289688)
        
        # Assigning a type to the variable 'expected' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'expected', array_call_result_289689)
        
        # Call to assert_array_equal(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'const' (line 89)
        const_289691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'const', False)
        # Getting the type of 'expected' (line 89)
        expected_289692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 34), 'expected', False)
        # Processing the call keyword arguments (line 89)
        kwargs_289693 = {}
        # Getting the type of 'assert_array_equal' (line 89)
        assert_array_equal_289690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 89)
        assert_array_equal_call_result_289694 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assert_array_equal_289690, *[const_289691, expected_289692], **kwargs_289693)
        
        
        # Assigning a Call to a Name (line 91):
        
        # Call to const_ext(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'a' (line 91)
        a_289696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'a', False)
        int_289697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 29), 'int')
        # Processing the call keyword arguments (line 91)
        int_289698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 37), 'int')
        keyword_289699 = int_289698
        kwargs_289700 = {'axis': keyword_289699}
        # Getting the type of 'const_ext' (line 91)
        const_ext_289695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'const_ext', False)
        # Calling const_ext(args, kwargs) (line 91)
        const_ext_call_result_289701 = invoke(stypy.reporting.localization.Localization(__file__, 91, 16), const_ext_289695, *[a_289696, int_289697], **kwargs_289700)
        
        # Assigning a type to the variable 'const' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'const', const_ext_call_result_289701)
        
        # Assigning a Call to a Name (line 92):
        
        # Call to array(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_289704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_289705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        int_289706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 29), list_289705, int_289706)
        # Adding element type (line 92)
        int_289707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 29), list_289705, int_289707)
        # Adding element type (line 92)
        int_289708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 29), list_289705, int_289708)
        # Adding element type (line 92)
        int_289709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 29), list_289705, int_289709)
        # Adding element type (line 92)
        int_289710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 29), list_289705, int_289710)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 28), list_289704, list_289705)
        # Adding element type (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_289711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        int_289712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 29), list_289711, int_289712)
        # Adding element type (line 93)
        int_289713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 29), list_289711, int_289713)
        # Adding element type (line 93)
        int_289714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 29), list_289711, int_289714)
        # Adding element type (line 93)
        int_289715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 29), list_289711, int_289715)
        # Adding element type (line 93)
        int_289716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 29), list_289711, int_289716)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 28), list_289704, list_289711)
        # Adding element type (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_289717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        # Adding element type (line 94)
        int_289718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 29), list_289717, int_289718)
        # Adding element type (line 94)
        int_289719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 29), list_289717, int_289719)
        # Adding element type (line 94)
        int_289720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 29), list_289717, int_289720)
        # Adding element type (line 94)
        int_289721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 29), list_289717, int_289721)
        # Adding element type (line 94)
        int_289722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 29), list_289717, int_289722)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 28), list_289704, list_289717)
        # Adding element type (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_289723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        int_289724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 29), list_289723, int_289724)
        # Adding element type (line 95)
        int_289725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 29), list_289723, int_289725)
        # Adding element type (line 95)
        int_289726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 29), list_289723, int_289726)
        # Adding element type (line 95)
        int_289727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 29), list_289723, int_289727)
        # Adding element type (line 95)
        int_289728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 29), list_289723, int_289728)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 28), list_289704, list_289723)
        
        # Processing the call keyword arguments (line 92)
        kwargs_289729 = {}
        # Getting the type of 'np' (line 92)
        np_289702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 92)
        array_289703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 19), np_289702, 'array')
        # Calling array(args, kwargs) (line 92)
        array_call_result_289730 = invoke(stypy.reporting.localization.Localization(__file__, 92, 19), array_289703, *[list_289704], **kwargs_289729)
        
        # Assigning a type to the variable 'expected' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'expected', array_call_result_289730)
        
        # Call to assert_array_equal(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'const' (line 96)
        const_289732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'const', False)
        # Getting the type of 'expected' (line 96)
        expected_289733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 34), 'expected', False)
        # Processing the call keyword arguments (line 96)
        kwargs_289734 = {}
        # Getting the type of 'assert_array_equal' (line 96)
        assert_array_equal_289731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 96)
        assert_array_equal_call_result_289735 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), assert_array_equal_289731, *[const_289732, expected_289733], **kwargs_289734)
        
        
        # ################# End of 'test_const_ext(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_const_ext' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_289736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_289736)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_const_ext'
        return stypy_return_type_289736


    @norecursion
    def test_zero_ext(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_zero_ext'
        module_type_store = module_type_store.open_function_context('test_zero_ext', 98, 4, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayTools.test_zero_ext.__dict__.__setitem__('stypy_localization', localization)
        TestArrayTools.test_zero_ext.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayTools.test_zero_ext.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayTools.test_zero_ext.__dict__.__setitem__('stypy_function_name', 'TestArrayTools.test_zero_ext')
        TestArrayTools.test_zero_ext.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayTools.test_zero_ext.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayTools.test_zero_ext.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayTools.test_zero_ext.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayTools.test_zero_ext.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayTools.test_zero_ext.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayTools.test_zero_ext.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayTools.test_zero_ext', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_zero_ext', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_zero_ext(...)' code ##################

        
        # Assigning a Call to a Name (line 99):
        
        # Call to array(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Obtaining an instance of the builtin type 'list' (line 99)
        list_289739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 99)
        # Adding element type (line 99)
        
        # Obtaining an instance of the builtin type 'list' (line 99)
        list_289740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 99)
        # Adding element type (line 99)
        int_289741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 22), list_289740, int_289741)
        # Adding element type (line 99)
        int_289742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 22), list_289740, int_289742)
        # Adding element type (line 99)
        int_289743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 22), list_289740, int_289743)
        # Adding element type (line 99)
        int_289744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 22), list_289740, int_289744)
        # Adding element type (line 99)
        int_289745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 22), list_289740, int_289745)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 21), list_289739, list_289740)
        # Adding element type (line 99)
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_289746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        # Adding element type (line 100)
        int_289747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 22), list_289746, int_289747)
        # Adding element type (line 100)
        int_289748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 22), list_289746, int_289748)
        # Adding element type (line 100)
        int_289749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 22), list_289746, int_289749)
        # Adding element type (line 100)
        int_289750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 22), list_289746, int_289750)
        # Adding element type (line 100)
        int_289751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 22), list_289746, int_289751)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 21), list_289739, list_289746)
        
        # Processing the call keyword arguments (line 99)
        kwargs_289752 = {}
        # Getting the type of 'np' (line 99)
        np_289737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 99)
        array_289738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), np_289737, 'array')
        # Calling array(args, kwargs) (line 99)
        array_call_result_289753 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), array_289738, *[list_289739], **kwargs_289752)
        
        # Assigning a type to the variable 'a' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'a', array_call_result_289753)
        
        # Assigning a Call to a Name (line 102):
        
        # Call to zero_ext(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'a' (line 102)
        a_289755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'a', False)
        int_289756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 27), 'int')
        # Processing the call keyword arguments (line 102)
        int_289757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 35), 'int')
        keyword_289758 = int_289757
        kwargs_289759 = {'axis': keyword_289758}
        # Getting the type of 'zero_ext' (line 102)
        zero_ext_289754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'zero_ext', False)
        # Calling zero_ext(args, kwargs) (line 102)
        zero_ext_call_result_289760 = invoke(stypy.reporting.localization.Localization(__file__, 102, 15), zero_ext_289754, *[a_289755, int_289756], **kwargs_289759)
        
        # Assigning a type to the variable 'zero' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'zero', zero_ext_call_result_289760)
        
        # Assigning a Call to a Name (line 103):
        
        # Call to array(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining an instance of the builtin type 'list' (line 103)
        list_289763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 103)
        # Adding element type (line 103)
        
        # Obtaining an instance of the builtin type 'list' (line 103)
        list_289764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 103)
        # Adding element type (line 103)
        int_289765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 29), list_289764, int_289765)
        # Adding element type (line 103)
        int_289766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 29), list_289764, int_289766)
        # Adding element type (line 103)
        int_289767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 29), list_289764, int_289767)
        # Adding element type (line 103)
        int_289768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 29), list_289764, int_289768)
        # Adding element type (line 103)
        int_289769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 29), list_289764, int_289769)
        # Adding element type (line 103)
        int_289770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 29), list_289764, int_289770)
        # Adding element type (line 103)
        int_289771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 29), list_289764, int_289771)
        # Adding element type (line 103)
        int_289772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 29), list_289764, int_289772)
        # Adding element type (line 103)
        int_289773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 29), list_289764, int_289773)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 28), list_289763, list_289764)
        # Adding element type (line 103)
        
        # Obtaining an instance of the builtin type 'list' (line 104)
        list_289774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 104)
        # Adding element type (line 104)
        int_289775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 29), list_289774, int_289775)
        # Adding element type (line 104)
        int_289776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 29), list_289774, int_289776)
        # Adding element type (line 104)
        int_289777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 29), list_289774, int_289777)
        # Adding element type (line 104)
        int_289778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 29), list_289774, int_289778)
        # Adding element type (line 104)
        int_289779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 29), list_289774, int_289779)
        # Adding element type (line 104)
        int_289780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 29), list_289774, int_289780)
        # Adding element type (line 104)
        int_289781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 29), list_289774, int_289781)
        # Adding element type (line 104)
        int_289782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 29), list_289774, int_289782)
        # Adding element type (line 104)
        int_289783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 29), list_289774, int_289783)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 28), list_289763, list_289774)
        
        # Processing the call keyword arguments (line 103)
        kwargs_289784 = {}
        # Getting the type of 'np' (line 103)
        np_289761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 103)
        array_289762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 19), np_289761, 'array')
        # Calling array(args, kwargs) (line 103)
        array_call_result_289785 = invoke(stypy.reporting.localization.Localization(__file__, 103, 19), array_289762, *[list_289763], **kwargs_289784)
        
        # Assigning a type to the variable 'expected' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'expected', array_call_result_289785)
        
        # Call to assert_array_equal(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'zero' (line 105)
        zero_289787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'zero', False)
        # Getting the type of 'expected' (line 105)
        expected_289788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'expected', False)
        # Processing the call keyword arguments (line 105)
        kwargs_289789 = {}
        # Getting the type of 'assert_array_equal' (line 105)
        assert_array_equal_289786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 105)
        assert_array_equal_call_result_289790 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), assert_array_equal_289786, *[zero_289787, expected_289788], **kwargs_289789)
        
        
        # Assigning a Call to a Name (line 107):
        
        # Call to zero_ext(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'a' (line 107)
        a_289792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'a', False)
        int_289793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 27), 'int')
        # Processing the call keyword arguments (line 107)
        int_289794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 35), 'int')
        keyword_289795 = int_289794
        kwargs_289796 = {'axis': keyword_289795}
        # Getting the type of 'zero_ext' (line 107)
        zero_ext_289791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'zero_ext', False)
        # Calling zero_ext(args, kwargs) (line 107)
        zero_ext_call_result_289797 = invoke(stypy.reporting.localization.Localization(__file__, 107, 15), zero_ext_289791, *[a_289792, int_289793], **kwargs_289796)
        
        # Assigning a type to the variable 'zero' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'zero', zero_ext_call_result_289797)
        
        # Assigning a Call to a Name (line 108):
        
        # Call to array(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_289800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_289801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        int_289802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 29), list_289801, int_289802)
        # Adding element type (line 108)
        int_289803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 29), list_289801, int_289803)
        # Adding element type (line 108)
        int_289804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 29), list_289801, int_289804)
        # Adding element type (line 108)
        int_289805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 29), list_289801, int_289805)
        # Adding element type (line 108)
        int_289806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 29), list_289801, int_289806)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 28), list_289800, list_289801)
        # Adding element type (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_289807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        int_289808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 29), list_289807, int_289808)
        # Adding element type (line 109)
        int_289809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 29), list_289807, int_289809)
        # Adding element type (line 109)
        int_289810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 29), list_289807, int_289810)
        # Adding element type (line 109)
        int_289811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 29), list_289807, int_289811)
        # Adding element type (line 109)
        int_289812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 29), list_289807, int_289812)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 28), list_289800, list_289807)
        # Adding element type (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_289813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        int_289814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 29), list_289813, int_289814)
        # Adding element type (line 110)
        int_289815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 29), list_289813, int_289815)
        # Adding element type (line 110)
        int_289816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 29), list_289813, int_289816)
        # Adding element type (line 110)
        int_289817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 29), list_289813, int_289817)
        # Adding element type (line 110)
        int_289818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 29), list_289813, int_289818)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 28), list_289800, list_289813)
        # Adding element type (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_289819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        int_289820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 29), list_289819, int_289820)
        # Adding element type (line 111)
        int_289821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 29), list_289819, int_289821)
        # Adding element type (line 111)
        int_289822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 29), list_289819, int_289822)
        # Adding element type (line 111)
        int_289823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 29), list_289819, int_289823)
        # Adding element type (line 111)
        int_289824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 29), list_289819, int_289824)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 28), list_289800, list_289819)
        
        # Processing the call keyword arguments (line 108)
        kwargs_289825 = {}
        # Getting the type of 'np' (line 108)
        np_289798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 108)
        array_289799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 19), np_289798, 'array')
        # Calling array(args, kwargs) (line 108)
        array_call_result_289826 = invoke(stypy.reporting.localization.Localization(__file__, 108, 19), array_289799, *[list_289800], **kwargs_289825)
        
        # Assigning a type to the variable 'expected' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'expected', array_call_result_289826)
        
        # Call to assert_array_equal(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'zero' (line 112)
        zero_289828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), 'zero', False)
        # Getting the type of 'expected' (line 112)
        expected_289829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 33), 'expected', False)
        # Processing the call keyword arguments (line 112)
        kwargs_289830 = {}
        # Getting the type of 'assert_array_equal' (line 112)
        assert_array_equal_289827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 112)
        assert_array_equal_call_result_289831 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), assert_array_equal_289827, *[zero_289828, expected_289829], **kwargs_289830)
        
        
        # ################# End of 'test_zero_ext(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_zero_ext' in the type store
        # Getting the type of 'stypy_return_type' (line 98)
        stypy_return_type_289832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_289832)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_zero_ext'
        return stypy_return_type_289832


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 12, 0, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayTools.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestArrayTools' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'TestArrayTools', TestArrayTools)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
