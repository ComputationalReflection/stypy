
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''unit tests for sparse utility functions'''
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: import numpy as np
6: from numpy.testing import assert_equal, assert_raises
7: from pytest import raises as assert_raises
8: from scipy.sparse import sputils
9: 
10: 
11: class TestSparseUtils(object):
12: 
13:     def test_upcast(self):
14:         assert_equal(sputils.upcast('intc'), np.intc)
15:         assert_equal(sputils.upcast('int32', 'float32'), np.float64)
16:         assert_equal(sputils.upcast('bool', complex, float), np.complex128)
17:         assert_equal(sputils.upcast('i', 'd'), np.float64)
18: 
19:     def test_getdtype(self):
20:         A = np.array([1], dtype='int8')
21: 
22:         assert_equal(sputils.getdtype(None, default=float), float)
23:         assert_equal(sputils.getdtype(None, a=A), np.int8)
24: 
25:     def test_isscalarlike(self):
26:         assert_equal(sputils.isscalarlike(3.0), True)
27:         assert_equal(sputils.isscalarlike(-4), True)
28:         assert_equal(sputils.isscalarlike(2.5), True)
29:         assert_equal(sputils.isscalarlike(1 + 3j), True)
30:         assert_equal(sputils.isscalarlike(np.array(3)), True)
31:         assert_equal(sputils.isscalarlike("16"), True)
32: 
33:         assert_equal(sputils.isscalarlike(np.array([3])), False)
34:         assert_equal(sputils.isscalarlike([[3]]), False)
35:         assert_equal(sputils.isscalarlike((1,)), False)
36:         assert_equal(sputils.isscalarlike((1, 2)), False)
37: 
38:     def test_isintlike(self):
39:         assert_equal(sputils.isintlike(3.0), True)
40:         assert_equal(sputils.isintlike(-4), True)
41:         assert_equal(sputils.isintlike(np.array(3)), True)
42:         assert_equal(sputils.isintlike(np.array([3])), False)
43: 
44:         assert_equal(sputils.isintlike(2.5), False)
45:         assert_equal(sputils.isintlike(1 + 3j), False)
46:         assert_equal(sputils.isintlike((1,)), False)
47:         assert_equal(sputils.isintlike((1, 2)), False)
48: 
49:     def test_isshape(self):
50:         assert_equal(sputils.isshape((1, 2)), True)
51:         assert_equal(sputils.isshape((5, 2)), True)
52: 
53:         assert_equal(sputils.isshape((1.5, 2)), False)
54:         assert_equal(sputils.isshape((2, 2, 2)), False)
55:         assert_equal(sputils.isshape(([2], 2)), False)
56: 
57:     def test_issequence(self):
58:         assert_equal(sputils.issequence((1,)), True)
59:         assert_equal(sputils.issequence((1, 2, 3)), True)
60:         assert_equal(sputils.issequence([1]), True)
61:         assert_equal(sputils.issequence([1, 2, 3]), True)
62:         assert_equal(sputils.issequence(np.array([1, 2, 3])), True)
63: 
64:         assert_equal(sputils.issequence(np.array([[1], [2], [3]])), False)
65:         assert_equal(sputils.issequence(3), False)
66: 
67:     def test_ismatrix(self):
68:         assert_equal(sputils.ismatrix(((),)), True)
69:         assert_equal(sputils.ismatrix([[1], [2]]), True)
70:         assert_equal(sputils.ismatrix(np.arange(3)[None]), True)
71: 
72:         assert_equal(sputils.ismatrix([1, 2]), False)
73:         assert_equal(sputils.ismatrix(np.arange(3)), False)
74:         assert_equal(sputils.ismatrix([[[1]]]), False)
75:         assert_equal(sputils.ismatrix(3), False)
76: 
77:     def test_isdense(self):
78:         assert_equal(sputils.isdense(np.array([1])), True)
79:         assert_equal(sputils.isdense(np.matrix([1])), True)
80: 
81:     def test_validateaxis(self):
82:         func = sputils.validateaxis
83: 
84:         assert_raises(TypeError, func, (0, 1))
85:         assert_raises(TypeError, func, 1.5)
86:         assert_raises(ValueError, func, 3)
87: 
88:         # These function calls should not raise errors
89:         for axis in (-2, -1, 0, 1, None):
90:             func(axis)
91: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_462215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'unit tests for sparse utility functions')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_462216 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_462216) is not StypyTypeError):

    if (import_462216 != 'pyd_module'):
        __import__(import_462216)
        sys_modules_462217 = sys.modules[import_462216]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_462217.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_462216)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_equal, assert_raises' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_462218 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_462218) is not StypyTypeError):

    if (import_462218 != 'pyd_module'):
        __import__(import_462218)
        sys_modules_462219 = sys.modules[import_462218]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_462219.module_type_store, module_type_store, ['assert_equal', 'assert_raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_462219, sys_modules_462219.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_raises'], [assert_equal, assert_raises])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_462218)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from pytest import assert_raises' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_462220 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest')

if (type(import_462220) is not StypyTypeError):

    if (import_462220 != 'pyd_module'):
        __import__(import_462220)
        sys_modules_462221 = sys.modules[import_462220]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', sys_modules_462221.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_462221, sys_modules_462221.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', import_462220)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.sparse import sputils' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_462222 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse')

if (type(import_462222) is not StypyTypeError):

    if (import_462222 != 'pyd_module'):
        __import__(import_462222)
        sys_modules_462223 = sys.modules[import_462222]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse', sys_modules_462223.module_type_store, module_type_store, ['sputils'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_462223, sys_modules_462223.module_type_store, module_type_store)
    else:
        from scipy.sparse import sputils

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse', None, module_type_store, ['sputils'], [sputils])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse', import_462222)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

# Declaration of the 'TestSparseUtils' class

class TestSparseUtils(object, ):

    @norecursion
    def test_upcast(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_upcast'
        module_type_store = module_type_store.open_function_context('test_upcast', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSparseUtils.test_upcast.__dict__.__setitem__('stypy_localization', localization)
        TestSparseUtils.test_upcast.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSparseUtils.test_upcast.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSparseUtils.test_upcast.__dict__.__setitem__('stypy_function_name', 'TestSparseUtils.test_upcast')
        TestSparseUtils.test_upcast.__dict__.__setitem__('stypy_param_names_list', [])
        TestSparseUtils.test_upcast.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSparseUtils.test_upcast.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSparseUtils.test_upcast.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSparseUtils.test_upcast.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSparseUtils.test_upcast.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSparseUtils.test_upcast.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseUtils.test_upcast', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_upcast', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_upcast(...)' code ##################

        
        # Call to assert_equal(...): (line 14)
        # Processing the call arguments (line 14)
        
        # Call to upcast(...): (line 14)
        # Processing the call arguments (line 14)
        str_462227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 36), 'str', 'intc')
        # Processing the call keyword arguments (line 14)
        kwargs_462228 = {}
        # Getting the type of 'sputils' (line 14)
        sputils_462225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 21), 'sputils', False)
        # Obtaining the member 'upcast' of a type (line 14)
        upcast_462226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 21), sputils_462225, 'upcast')
        # Calling upcast(args, kwargs) (line 14)
        upcast_call_result_462229 = invoke(stypy.reporting.localization.Localization(__file__, 14, 21), upcast_462226, *[str_462227], **kwargs_462228)
        
        # Getting the type of 'np' (line 14)
        np_462230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 45), 'np', False)
        # Obtaining the member 'intc' of a type (line 14)
        intc_462231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 45), np_462230, 'intc')
        # Processing the call keyword arguments (line 14)
        kwargs_462232 = {}
        # Getting the type of 'assert_equal' (line 14)
        assert_equal_462224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 14)
        assert_equal_call_result_462233 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), assert_equal_462224, *[upcast_call_result_462229, intc_462231], **kwargs_462232)
        
        
        # Call to assert_equal(...): (line 15)
        # Processing the call arguments (line 15)
        
        # Call to upcast(...): (line 15)
        # Processing the call arguments (line 15)
        str_462237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 36), 'str', 'int32')
        str_462238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 45), 'str', 'float32')
        # Processing the call keyword arguments (line 15)
        kwargs_462239 = {}
        # Getting the type of 'sputils' (line 15)
        sputils_462235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'sputils', False)
        # Obtaining the member 'upcast' of a type (line 15)
        upcast_462236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 21), sputils_462235, 'upcast')
        # Calling upcast(args, kwargs) (line 15)
        upcast_call_result_462240 = invoke(stypy.reporting.localization.Localization(__file__, 15, 21), upcast_462236, *[str_462237, str_462238], **kwargs_462239)
        
        # Getting the type of 'np' (line 15)
        np_462241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 57), 'np', False)
        # Obtaining the member 'float64' of a type (line 15)
        float64_462242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 57), np_462241, 'float64')
        # Processing the call keyword arguments (line 15)
        kwargs_462243 = {}
        # Getting the type of 'assert_equal' (line 15)
        assert_equal_462234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 15)
        assert_equal_call_result_462244 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), assert_equal_462234, *[upcast_call_result_462240, float64_462242], **kwargs_462243)
        
        
        # Call to assert_equal(...): (line 16)
        # Processing the call arguments (line 16)
        
        # Call to upcast(...): (line 16)
        # Processing the call arguments (line 16)
        str_462248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 36), 'str', 'bool')
        # Getting the type of 'complex' (line 16)
        complex_462249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 44), 'complex', False)
        # Getting the type of 'float' (line 16)
        float_462250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 53), 'float', False)
        # Processing the call keyword arguments (line 16)
        kwargs_462251 = {}
        # Getting the type of 'sputils' (line 16)
        sputils_462246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 21), 'sputils', False)
        # Obtaining the member 'upcast' of a type (line 16)
        upcast_462247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 21), sputils_462246, 'upcast')
        # Calling upcast(args, kwargs) (line 16)
        upcast_call_result_462252 = invoke(stypy.reporting.localization.Localization(__file__, 16, 21), upcast_462247, *[str_462248, complex_462249, float_462250], **kwargs_462251)
        
        # Getting the type of 'np' (line 16)
        np_462253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 61), 'np', False)
        # Obtaining the member 'complex128' of a type (line 16)
        complex128_462254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 61), np_462253, 'complex128')
        # Processing the call keyword arguments (line 16)
        kwargs_462255 = {}
        # Getting the type of 'assert_equal' (line 16)
        assert_equal_462245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 16)
        assert_equal_call_result_462256 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), assert_equal_462245, *[upcast_call_result_462252, complex128_462254], **kwargs_462255)
        
        
        # Call to assert_equal(...): (line 17)
        # Processing the call arguments (line 17)
        
        # Call to upcast(...): (line 17)
        # Processing the call arguments (line 17)
        str_462260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 36), 'str', 'i')
        str_462261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 41), 'str', 'd')
        # Processing the call keyword arguments (line 17)
        kwargs_462262 = {}
        # Getting the type of 'sputils' (line 17)
        sputils_462258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 21), 'sputils', False)
        # Obtaining the member 'upcast' of a type (line 17)
        upcast_462259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 21), sputils_462258, 'upcast')
        # Calling upcast(args, kwargs) (line 17)
        upcast_call_result_462263 = invoke(stypy.reporting.localization.Localization(__file__, 17, 21), upcast_462259, *[str_462260, str_462261], **kwargs_462262)
        
        # Getting the type of 'np' (line 17)
        np_462264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 47), 'np', False)
        # Obtaining the member 'float64' of a type (line 17)
        float64_462265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 47), np_462264, 'float64')
        # Processing the call keyword arguments (line 17)
        kwargs_462266 = {}
        # Getting the type of 'assert_equal' (line 17)
        assert_equal_462257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 17)
        assert_equal_call_result_462267 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), assert_equal_462257, *[upcast_call_result_462263, float64_462265], **kwargs_462266)
        
        
        # ################# End of 'test_upcast(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_upcast' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_462268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_462268)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_upcast'
        return stypy_return_type_462268


    @norecursion
    def test_getdtype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_getdtype'
        module_type_store = module_type_store.open_function_context('test_getdtype', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSparseUtils.test_getdtype.__dict__.__setitem__('stypy_localization', localization)
        TestSparseUtils.test_getdtype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSparseUtils.test_getdtype.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSparseUtils.test_getdtype.__dict__.__setitem__('stypy_function_name', 'TestSparseUtils.test_getdtype')
        TestSparseUtils.test_getdtype.__dict__.__setitem__('stypy_param_names_list', [])
        TestSparseUtils.test_getdtype.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSparseUtils.test_getdtype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSparseUtils.test_getdtype.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSparseUtils.test_getdtype.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSparseUtils.test_getdtype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSparseUtils.test_getdtype.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseUtils.test_getdtype', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_getdtype', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_getdtype(...)' code ##################

        
        # Assigning a Call to a Name (line 20):
        
        # Call to array(...): (line 20)
        # Processing the call arguments (line 20)
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_462271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        # Adding element type (line 20)
        int_462272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 21), list_462271, int_462272)
        
        # Processing the call keyword arguments (line 20)
        str_462273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'str', 'int8')
        keyword_462274 = str_462273
        kwargs_462275 = {'dtype': keyword_462274}
        # Getting the type of 'np' (line 20)
        np_462269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 20)
        array_462270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), np_462269, 'array')
        # Calling array(args, kwargs) (line 20)
        array_call_result_462276 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), array_462270, *[list_462271], **kwargs_462275)
        
        # Assigning a type to the variable 'A' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'A', array_call_result_462276)
        
        # Call to assert_equal(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Call to getdtype(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'None' (line 22)
        None_462280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 38), 'None', False)
        # Processing the call keyword arguments (line 22)
        # Getting the type of 'float' (line 22)
        float_462281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 52), 'float', False)
        keyword_462282 = float_462281
        kwargs_462283 = {'default': keyword_462282}
        # Getting the type of 'sputils' (line 22)
        sputils_462278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 21), 'sputils', False)
        # Obtaining the member 'getdtype' of a type (line 22)
        getdtype_462279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 21), sputils_462278, 'getdtype')
        # Calling getdtype(args, kwargs) (line 22)
        getdtype_call_result_462284 = invoke(stypy.reporting.localization.Localization(__file__, 22, 21), getdtype_462279, *[None_462280], **kwargs_462283)
        
        # Getting the type of 'float' (line 22)
        float_462285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 60), 'float', False)
        # Processing the call keyword arguments (line 22)
        kwargs_462286 = {}
        # Getting the type of 'assert_equal' (line 22)
        assert_equal_462277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 22)
        assert_equal_call_result_462287 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), assert_equal_462277, *[getdtype_call_result_462284, float_462285], **kwargs_462286)
        
        
        # Call to assert_equal(...): (line 23)
        # Processing the call arguments (line 23)
        
        # Call to getdtype(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'None' (line 23)
        None_462291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 38), 'None', False)
        # Processing the call keyword arguments (line 23)
        # Getting the type of 'A' (line 23)
        A_462292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 46), 'A', False)
        keyword_462293 = A_462292
        kwargs_462294 = {'a': keyword_462293}
        # Getting the type of 'sputils' (line 23)
        sputils_462289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'sputils', False)
        # Obtaining the member 'getdtype' of a type (line 23)
        getdtype_462290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 21), sputils_462289, 'getdtype')
        # Calling getdtype(args, kwargs) (line 23)
        getdtype_call_result_462295 = invoke(stypy.reporting.localization.Localization(__file__, 23, 21), getdtype_462290, *[None_462291], **kwargs_462294)
        
        # Getting the type of 'np' (line 23)
        np_462296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 50), 'np', False)
        # Obtaining the member 'int8' of a type (line 23)
        int8_462297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 50), np_462296, 'int8')
        # Processing the call keyword arguments (line 23)
        kwargs_462298 = {}
        # Getting the type of 'assert_equal' (line 23)
        assert_equal_462288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 23)
        assert_equal_call_result_462299 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), assert_equal_462288, *[getdtype_call_result_462295, int8_462297], **kwargs_462298)
        
        
        # ################# End of 'test_getdtype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_getdtype' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_462300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_462300)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_getdtype'
        return stypy_return_type_462300


    @norecursion
    def test_isscalarlike(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_isscalarlike'
        module_type_store = module_type_store.open_function_context('test_isscalarlike', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSparseUtils.test_isscalarlike.__dict__.__setitem__('stypy_localization', localization)
        TestSparseUtils.test_isscalarlike.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSparseUtils.test_isscalarlike.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSparseUtils.test_isscalarlike.__dict__.__setitem__('stypy_function_name', 'TestSparseUtils.test_isscalarlike')
        TestSparseUtils.test_isscalarlike.__dict__.__setitem__('stypy_param_names_list', [])
        TestSparseUtils.test_isscalarlike.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSparseUtils.test_isscalarlike.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSparseUtils.test_isscalarlike.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSparseUtils.test_isscalarlike.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSparseUtils.test_isscalarlike.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSparseUtils.test_isscalarlike.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseUtils.test_isscalarlike', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_isscalarlike', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_isscalarlike(...)' code ##################

        
        # Call to assert_equal(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Call to isscalarlike(...): (line 26)
        # Processing the call arguments (line 26)
        float_462304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 42), 'float')
        # Processing the call keyword arguments (line 26)
        kwargs_462305 = {}
        # Getting the type of 'sputils' (line 26)
        sputils_462302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'sputils', False)
        # Obtaining the member 'isscalarlike' of a type (line 26)
        isscalarlike_462303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 21), sputils_462302, 'isscalarlike')
        # Calling isscalarlike(args, kwargs) (line 26)
        isscalarlike_call_result_462306 = invoke(stypy.reporting.localization.Localization(__file__, 26, 21), isscalarlike_462303, *[float_462304], **kwargs_462305)
        
        # Getting the type of 'True' (line 26)
        True_462307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 48), 'True', False)
        # Processing the call keyword arguments (line 26)
        kwargs_462308 = {}
        # Getting the type of 'assert_equal' (line 26)
        assert_equal_462301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 26)
        assert_equal_call_result_462309 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), assert_equal_462301, *[isscalarlike_call_result_462306, True_462307], **kwargs_462308)
        
        
        # Call to assert_equal(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Call to isscalarlike(...): (line 27)
        # Processing the call arguments (line 27)
        int_462313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 42), 'int')
        # Processing the call keyword arguments (line 27)
        kwargs_462314 = {}
        # Getting the type of 'sputils' (line 27)
        sputils_462311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'sputils', False)
        # Obtaining the member 'isscalarlike' of a type (line 27)
        isscalarlike_462312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 21), sputils_462311, 'isscalarlike')
        # Calling isscalarlike(args, kwargs) (line 27)
        isscalarlike_call_result_462315 = invoke(stypy.reporting.localization.Localization(__file__, 27, 21), isscalarlike_462312, *[int_462313], **kwargs_462314)
        
        # Getting the type of 'True' (line 27)
        True_462316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 47), 'True', False)
        # Processing the call keyword arguments (line 27)
        kwargs_462317 = {}
        # Getting the type of 'assert_equal' (line 27)
        assert_equal_462310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 27)
        assert_equal_call_result_462318 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assert_equal_462310, *[isscalarlike_call_result_462315, True_462316], **kwargs_462317)
        
        
        # Call to assert_equal(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Call to isscalarlike(...): (line 28)
        # Processing the call arguments (line 28)
        float_462322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 42), 'float')
        # Processing the call keyword arguments (line 28)
        kwargs_462323 = {}
        # Getting the type of 'sputils' (line 28)
        sputils_462320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 21), 'sputils', False)
        # Obtaining the member 'isscalarlike' of a type (line 28)
        isscalarlike_462321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 21), sputils_462320, 'isscalarlike')
        # Calling isscalarlike(args, kwargs) (line 28)
        isscalarlike_call_result_462324 = invoke(stypy.reporting.localization.Localization(__file__, 28, 21), isscalarlike_462321, *[float_462322], **kwargs_462323)
        
        # Getting the type of 'True' (line 28)
        True_462325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 48), 'True', False)
        # Processing the call keyword arguments (line 28)
        kwargs_462326 = {}
        # Getting the type of 'assert_equal' (line 28)
        assert_equal_462319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 28)
        assert_equal_call_result_462327 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), assert_equal_462319, *[isscalarlike_call_result_462324, True_462325], **kwargs_462326)
        
        
        # Call to assert_equal(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Call to isscalarlike(...): (line 29)
        # Processing the call arguments (line 29)
        int_462331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 42), 'int')
        complex_462332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 46), 'complex')
        # Applying the binary operator '+' (line 29)
        result_add_462333 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 42), '+', int_462331, complex_462332)
        
        # Processing the call keyword arguments (line 29)
        kwargs_462334 = {}
        # Getting the type of 'sputils' (line 29)
        sputils_462329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 21), 'sputils', False)
        # Obtaining the member 'isscalarlike' of a type (line 29)
        isscalarlike_462330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 21), sputils_462329, 'isscalarlike')
        # Calling isscalarlike(args, kwargs) (line 29)
        isscalarlike_call_result_462335 = invoke(stypy.reporting.localization.Localization(__file__, 29, 21), isscalarlike_462330, *[result_add_462333], **kwargs_462334)
        
        # Getting the type of 'True' (line 29)
        True_462336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 51), 'True', False)
        # Processing the call keyword arguments (line 29)
        kwargs_462337 = {}
        # Getting the type of 'assert_equal' (line 29)
        assert_equal_462328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 29)
        assert_equal_call_result_462338 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), assert_equal_462328, *[isscalarlike_call_result_462335, True_462336], **kwargs_462337)
        
        
        # Call to assert_equal(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Call to isscalarlike(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Call to array(...): (line 30)
        # Processing the call arguments (line 30)
        int_462344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 51), 'int')
        # Processing the call keyword arguments (line 30)
        kwargs_462345 = {}
        # Getting the type of 'np' (line 30)
        np_462342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 42), 'np', False)
        # Obtaining the member 'array' of a type (line 30)
        array_462343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 42), np_462342, 'array')
        # Calling array(args, kwargs) (line 30)
        array_call_result_462346 = invoke(stypy.reporting.localization.Localization(__file__, 30, 42), array_462343, *[int_462344], **kwargs_462345)
        
        # Processing the call keyword arguments (line 30)
        kwargs_462347 = {}
        # Getting the type of 'sputils' (line 30)
        sputils_462340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'sputils', False)
        # Obtaining the member 'isscalarlike' of a type (line 30)
        isscalarlike_462341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 21), sputils_462340, 'isscalarlike')
        # Calling isscalarlike(args, kwargs) (line 30)
        isscalarlike_call_result_462348 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), isscalarlike_462341, *[array_call_result_462346], **kwargs_462347)
        
        # Getting the type of 'True' (line 30)
        True_462349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 56), 'True', False)
        # Processing the call keyword arguments (line 30)
        kwargs_462350 = {}
        # Getting the type of 'assert_equal' (line 30)
        assert_equal_462339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 30)
        assert_equal_call_result_462351 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assert_equal_462339, *[isscalarlike_call_result_462348, True_462349], **kwargs_462350)
        
        
        # Call to assert_equal(...): (line 31)
        # Processing the call arguments (line 31)
        
        # Call to isscalarlike(...): (line 31)
        # Processing the call arguments (line 31)
        str_462355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 42), 'str', '16')
        # Processing the call keyword arguments (line 31)
        kwargs_462356 = {}
        # Getting the type of 'sputils' (line 31)
        sputils_462353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 21), 'sputils', False)
        # Obtaining the member 'isscalarlike' of a type (line 31)
        isscalarlike_462354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 21), sputils_462353, 'isscalarlike')
        # Calling isscalarlike(args, kwargs) (line 31)
        isscalarlike_call_result_462357 = invoke(stypy.reporting.localization.Localization(__file__, 31, 21), isscalarlike_462354, *[str_462355], **kwargs_462356)
        
        # Getting the type of 'True' (line 31)
        True_462358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 49), 'True', False)
        # Processing the call keyword arguments (line 31)
        kwargs_462359 = {}
        # Getting the type of 'assert_equal' (line 31)
        assert_equal_462352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 31)
        assert_equal_call_result_462360 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), assert_equal_462352, *[isscalarlike_call_result_462357, True_462358], **kwargs_462359)
        
        
        # Call to assert_equal(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Call to isscalarlike(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Call to array(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_462366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        # Adding element type (line 33)
        int_462367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 51), list_462366, int_462367)
        
        # Processing the call keyword arguments (line 33)
        kwargs_462368 = {}
        # Getting the type of 'np' (line 33)
        np_462364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 42), 'np', False)
        # Obtaining the member 'array' of a type (line 33)
        array_462365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 42), np_462364, 'array')
        # Calling array(args, kwargs) (line 33)
        array_call_result_462369 = invoke(stypy.reporting.localization.Localization(__file__, 33, 42), array_462365, *[list_462366], **kwargs_462368)
        
        # Processing the call keyword arguments (line 33)
        kwargs_462370 = {}
        # Getting the type of 'sputils' (line 33)
        sputils_462362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 21), 'sputils', False)
        # Obtaining the member 'isscalarlike' of a type (line 33)
        isscalarlike_462363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 21), sputils_462362, 'isscalarlike')
        # Calling isscalarlike(args, kwargs) (line 33)
        isscalarlike_call_result_462371 = invoke(stypy.reporting.localization.Localization(__file__, 33, 21), isscalarlike_462363, *[array_call_result_462369], **kwargs_462370)
        
        # Getting the type of 'False' (line 33)
        False_462372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 58), 'False', False)
        # Processing the call keyword arguments (line 33)
        kwargs_462373 = {}
        # Getting the type of 'assert_equal' (line 33)
        assert_equal_462361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 33)
        assert_equal_call_result_462374 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), assert_equal_462361, *[isscalarlike_call_result_462371, False_462372], **kwargs_462373)
        
        
        # Call to assert_equal(...): (line 34)
        # Processing the call arguments (line 34)
        
        # Call to isscalarlike(...): (line 34)
        # Processing the call arguments (line 34)
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_462378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        # Adding element type (line 34)
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_462379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        # Adding element type (line 34)
        int_462380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 43), list_462379, int_462380)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 42), list_462378, list_462379)
        
        # Processing the call keyword arguments (line 34)
        kwargs_462381 = {}
        # Getting the type of 'sputils' (line 34)
        sputils_462376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'sputils', False)
        # Obtaining the member 'isscalarlike' of a type (line 34)
        isscalarlike_462377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), sputils_462376, 'isscalarlike')
        # Calling isscalarlike(args, kwargs) (line 34)
        isscalarlike_call_result_462382 = invoke(stypy.reporting.localization.Localization(__file__, 34, 21), isscalarlike_462377, *[list_462378], **kwargs_462381)
        
        # Getting the type of 'False' (line 34)
        False_462383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 50), 'False', False)
        # Processing the call keyword arguments (line 34)
        kwargs_462384 = {}
        # Getting the type of 'assert_equal' (line 34)
        assert_equal_462375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 34)
        assert_equal_call_result_462385 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assert_equal_462375, *[isscalarlike_call_result_462382, False_462383], **kwargs_462384)
        
        
        # Call to assert_equal(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Call to isscalarlike(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Obtaining an instance of the builtin type 'tuple' (line 35)
        tuple_462389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 35)
        # Adding element type (line 35)
        int_462390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 43), tuple_462389, int_462390)
        
        # Processing the call keyword arguments (line 35)
        kwargs_462391 = {}
        # Getting the type of 'sputils' (line 35)
        sputils_462387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'sputils', False)
        # Obtaining the member 'isscalarlike' of a type (line 35)
        isscalarlike_462388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 21), sputils_462387, 'isscalarlike')
        # Calling isscalarlike(args, kwargs) (line 35)
        isscalarlike_call_result_462392 = invoke(stypy.reporting.localization.Localization(__file__, 35, 21), isscalarlike_462388, *[tuple_462389], **kwargs_462391)
        
        # Getting the type of 'False' (line 35)
        False_462393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 49), 'False', False)
        # Processing the call keyword arguments (line 35)
        kwargs_462394 = {}
        # Getting the type of 'assert_equal' (line 35)
        assert_equal_462386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 35)
        assert_equal_call_result_462395 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), assert_equal_462386, *[isscalarlike_call_result_462392, False_462393], **kwargs_462394)
        
        
        # Call to assert_equal(...): (line 36)
        # Processing the call arguments (line 36)
        
        # Call to isscalarlike(...): (line 36)
        # Processing the call arguments (line 36)
        
        # Obtaining an instance of the builtin type 'tuple' (line 36)
        tuple_462399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 36)
        # Adding element type (line 36)
        int_462400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 43), tuple_462399, int_462400)
        # Adding element type (line 36)
        int_462401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 43), tuple_462399, int_462401)
        
        # Processing the call keyword arguments (line 36)
        kwargs_462402 = {}
        # Getting the type of 'sputils' (line 36)
        sputils_462397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'sputils', False)
        # Obtaining the member 'isscalarlike' of a type (line 36)
        isscalarlike_462398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 21), sputils_462397, 'isscalarlike')
        # Calling isscalarlike(args, kwargs) (line 36)
        isscalarlike_call_result_462403 = invoke(stypy.reporting.localization.Localization(__file__, 36, 21), isscalarlike_462398, *[tuple_462399], **kwargs_462402)
        
        # Getting the type of 'False' (line 36)
        False_462404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 51), 'False', False)
        # Processing the call keyword arguments (line 36)
        kwargs_462405 = {}
        # Getting the type of 'assert_equal' (line 36)
        assert_equal_462396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 36)
        assert_equal_call_result_462406 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assert_equal_462396, *[isscalarlike_call_result_462403, False_462404], **kwargs_462405)
        
        
        # ################# End of 'test_isscalarlike(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_isscalarlike' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_462407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_462407)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_isscalarlike'
        return stypy_return_type_462407


    @norecursion
    def test_isintlike(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_isintlike'
        module_type_store = module_type_store.open_function_context('test_isintlike', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSparseUtils.test_isintlike.__dict__.__setitem__('stypy_localization', localization)
        TestSparseUtils.test_isintlike.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSparseUtils.test_isintlike.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSparseUtils.test_isintlike.__dict__.__setitem__('stypy_function_name', 'TestSparseUtils.test_isintlike')
        TestSparseUtils.test_isintlike.__dict__.__setitem__('stypy_param_names_list', [])
        TestSparseUtils.test_isintlike.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSparseUtils.test_isintlike.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSparseUtils.test_isintlike.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSparseUtils.test_isintlike.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSparseUtils.test_isintlike.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSparseUtils.test_isintlike.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseUtils.test_isintlike', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_isintlike', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_isintlike(...)' code ##################

        
        # Call to assert_equal(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Call to isintlike(...): (line 39)
        # Processing the call arguments (line 39)
        float_462411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 39), 'float')
        # Processing the call keyword arguments (line 39)
        kwargs_462412 = {}
        # Getting the type of 'sputils' (line 39)
        sputils_462409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'sputils', False)
        # Obtaining the member 'isintlike' of a type (line 39)
        isintlike_462410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 21), sputils_462409, 'isintlike')
        # Calling isintlike(args, kwargs) (line 39)
        isintlike_call_result_462413 = invoke(stypy.reporting.localization.Localization(__file__, 39, 21), isintlike_462410, *[float_462411], **kwargs_462412)
        
        # Getting the type of 'True' (line 39)
        True_462414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 45), 'True', False)
        # Processing the call keyword arguments (line 39)
        kwargs_462415 = {}
        # Getting the type of 'assert_equal' (line 39)
        assert_equal_462408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 39)
        assert_equal_call_result_462416 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), assert_equal_462408, *[isintlike_call_result_462413, True_462414], **kwargs_462415)
        
        
        # Call to assert_equal(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Call to isintlike(...): (line 40)
        # Processing the call arguments (line 40)
        int_462420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 39), 'int')
        # Processing the call keyword arguments (line 40)
        kwargs_462421 = {}
        # Getting the type of 'sputils' (line 40)
        sputils_462418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 21), 'sputils', False)
        # Obtaining the member 'isintlike' of a type (line 40)
        isintlike_462419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 21), sputils_462418, 'isintlike')
        # Calling isintlike(args, kwargs) (line 40)
        isintlike_call_result_462422 = invoke(stypy.reporting.localization.Localization(__file__, 40, 21), isintlike_462419, *[int_462420], **kwargs_462421)
        
        # Getting the type of 'True' (line 40)
        True_462423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 44), 'True', False)
        # Processing the call keyword arguments (line 40)
        kwargs_462424 = {}
        # Getting the type of 'assert_equal' (line 40)
        assert_equal_462417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 40)
        assert_equal_call_result_462425 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), assert_equal_462417, *[isintlike_call_result_462422, True_462423], **kwargs_462424)
        
        
        # Call to assert_equal(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Call to isintlike(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Call to array(...): (line 41)
        # Processing the call arguments (line 41)
        int_462431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 48), 'int')
        # Processing the call keyword arguments (line 41)
        kwargs_462432 = {}
        # Getting the type of 'np' (line 41)
        np_462429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 39), 'np', False)
        # Obtaining the member 'array' of a type (line 41)
        array_462430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 39), np_462429, 'array')
        # Calling array(args, kwargs) (line 41)
        array_call_result_462433 = invoke(stypy.reporting.localization.Localization(__file__, 41, 39), array_462430, *[int_462431], **kwargs_462432)
        
        # Processing the call keyword arguments (line 41)
        kwargs_462434 = {}
        # Getting the type of 'sputils' (line 41)
        sputils_462427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'sputils', False)
        # Obtaining the member 'isintlike' of a type (line 41)
        isintlike_462428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 21), sputils_462427, 'isintlike')
        # Calling isintlike(args, kwargs) (line 41)
        isintlike_call_result_462435 = invoke(stypy.reporting.localization.Localization(__file__, 41, 21), isintlike_462428, *[array_call_result_462433], **kwargs_462434)
        
        # Getting the type of 'True' (line 41)
        True_462436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 53), 'True', False)
        # Processing the call keyword arguments (line 41)
        kwargs_462437 = {}
        # Getting the type of 'assert_equal' (line 41)
        assert_equal_462426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 41)
        assert_equal_call_result_462438 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), assert_equal_462426, *[isintlike_call_result_462435, True_462436], **kwargs_462437)
        
        
        # Call to assert_equal(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to isintlike(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to array(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_462444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        int_462445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 48), list_462444, int_462445)
        
        # Processing the call keyword arguments (line 42)
        kwargs_462446 = {}
        # Getting the type of 'np' (line 42)
        np_462442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 'np', False)
        # Obtaining the member 'array' of a type (line 42)
        array_462443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 39), np_462442, 'array')
        # Calling array(args, kwargs) (line 42)
        array_call_result_462447 = invoke(stypy.reporting.localization.Localization(__file__, 42, 39), array_462443, *[list_462444], **kwargs_462446)
        
        # Processing the call keyword arguments (line 42)
        kwargs_462448 = {}
        # Getting the type of 'sputils' (line 42)
        sputils_462440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), 'sputils', False)
        # Obtaining the member 'isintlike' of a type (line 42)
        isintlike_462441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 21), sputils_462440, 'isintlike')
        # Calling isintlike(args, kwargs) (line 42)
        isintlike_call_result_462449 = invoke(stypy.reporting.localization.Localization(__file__, 42, 21), isintlike_462441, *[array_call_result_462447], **kwargs_462448)
        
        # Getting the type of 'False' (line 42)
        False_462450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 55), 'False', False)
        # Processing the call keyword arguments (line 42)
        kwargs_462451 = {}
        # Getting the type of 'assert_equal' (line 42)
        assert_equal_462439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 42)
        assert_equal_call_result_462452 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assert_equal_462439, *[isintlike_call_result_462449, False_462450], **kwargs_462451)
        
        
        # Call to assert_equal(...): (line 44)
        # Processing the call arguments (line 44)
        
        # Call to isintlike(...): (line 44)
        # Processing the call arguments (line 44)
        float_462456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 39), 'float')
        # Processing the call keyword arguments (line 44)
        kwargs_462457 = {}
        # Getting the type of 'sputils' (line 44)
        sputils_462454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), 'sputils', False)
        # Obtaining the member 'isintlike' of a type (line 44)
        isintlike_462455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 21), sputils_462454, 'isintlike')
        # Calling isintlike(args, kwargs) (line 44)
        isintlike_call_result_462458 = invoke(stypy.reporting.localization.Localization(__file__, 44, 21), isintlike_462455, *[float_462456], **kwargs_462457)
        
        # Getting the type of 'False' (line 44)
        False_462459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 45), 'False', False)
        # Processing the call keyword arguments (line 44)
        kwargs_462460 = {}
        # Getting the type of 'assert_equal' (line 44)
        assert_equal_462453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 44)
        assert_equal_call_result_462461 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), assert_equal_462453, *[isintlike_call_result_462458, False_462459], **kwargs_462460)
        
        
        # Call to assert_equal(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Call to isintlike(...): (line 45)
        # Processing the call arguments (line 45)
        int_462465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 39), 'int')
        complex_462466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 43), 'complex')
        # Applying the binary operator '+' (line 45)
        result_add_462467 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 39), '+', int_462465, complex_462466)
        
        # Processing the call keyword arguments (line 45)
        kwargs_462468 = {}
        # Getting the type of 'sputils' (line 45)
        sputils_462463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 21), 'sputils', False)
        # Obtaining the member 'isintlike' of a type (line 45)
        isintlike_462464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 21), sputils_462463, 'isintlike')
        # Calling isintlike(args, kwargs) (line 45)
        isintlike_call_result_462469 = invoke(stypy.reporting.localization.Localization(__file__, 45, 21), isintlike_462464, *[result_add_462467], **kwargs_462468)
        
        # Getting the type of 'False' (line 45)
        False_462470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 48), 'False', False)
        # Processing the call keyword arguments (line 45)
        kwargs_462471 = {}
        # Getting the type of 'assert_equal' (line 45)
        assert_equal_462462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 45)
        assert_equal_call_result_462472 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), assert_equal_462462, *[isintlike_call_result_462469, False_462470], **kwargs_462471)
        
        
        # Call to assert_equal(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Call to isintlike(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Obtaining an instance of the builtin type 'tuple' (line 46)
        tuple_462476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 46)
        # Adding element type (line 46)
        int_462477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 40), tuple_462476, int_462477)
        
        # Processing the call keyword arguments (line 46)
        kwargs_462478 = {}
        # Getting the type of 'sputils' (line 46)
        sputils_462474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'sputils', False)
        # Obtaining the member 'isintlike' of a type (line 46)
        isintlike_462475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 21), sputils_462474, 'isintlike')
        # Calling isintlike(args, kwargs) (line 46)
        isintlike_call_result_462479 = invoke(stypy.reporting.localization.Localization(__file__, 46, 21), isintlike_462475, *[tuple_462476], **kwargs_462478)
        
        # Getting the type of 'False' (line 46)
        False_462480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'False', False)
        # Processing the call keyword arguments (line 46)
        kwargs_462481 = {}
        # Getting the type of 'assert_equal' (line 46)
        assert_equal_462473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 46)
        assert_equal_call_result_462482 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), assert_equal_462473, *[isintlike_call_result_462479, False_462480], **kwargs_462481)
        
        
        # Call to assert_equal(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Call to isintlike(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_462486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        int_462487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 40), tuple_462486, int_462487)
        # Adding element type (line 47)
        int_462488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 40), tuple_462486, int_462488)
        
        # Processing the call keyword arguments (line 47)
        kwargs_462489 = {}
        # Getting the type of 'sputils' (line 47)
        sputils_462484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 21), 'sputils', False)
        # Obtaining the member 'isintlike' of a type (line 47)
        isintlike_462485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 21), sputils_462484, 'isintlike')
        # Calling isintlike(args, kwargs) (line 47)
        isintlike_call_result_462490 = invoke(stypy.reporting.localization.Localization(__file__, 47, 21), isintlike_462485, *[tuple_462486], **kwargs_462489)
        
        # Getting the type of 'False' (line 47)
        False_462491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 48), 'False', False)
        # Processing the call keyword arguments (line 47)
        kwargs_462492 = {}
        # Getting the type of 'assert_equal' (line 47)
        assert_equal_462483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 47)
        assert_equal_call_result_462493 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assert_equal_462483, *[isintlike_call_result_462490, False_462491], **kwargs_462492)
        
        
        # ################# End of 'test_isintlike(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_isintlike' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_462494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_462494)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_isintlike'
        return stypy_return_type_462494


    @norecursion
    def test_isshape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_isshape'
        module_type_store = module_type_store.open_function_context('test_isshape', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSparseUtils.test_isshape.__dict__.__setitem__('stypy_localization', localization)
        TestSparseUtils.test_isshape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSparseUtils.test_isshape.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSparseUtils.test_isshape.__dict__.__setitem__('stypy_function_name', 'TestSparseUtils.test_isshape')
        TestSparseUtils.test_isshape.__dict__.__setitem__('stypy_param_names_list', [])
        TestSparseUtils.test_isshape.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSparseUtils.test_isshape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSparseUtils.test_isshape.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSparseUtils.test_isshape.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSparseUtils.test_isshape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSparseUtils.test_isshape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseUtils.test_isshape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_isshape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_isshape(...)' code ##################

        
        # Call to assert_equal(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Call to isshape(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Obtaining an instance of the builtin type 'tuple' (line 50)
        tuple_462498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 50)
        # Adding element type (line 50)
        int_462499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 38), tuple_462498, int_462499)
        # Adding element type (line 50)
        int_462500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 38), tuple_462498, int_462500)
        
        # Processing the call keyword arguments (line 50)
        kwargs_462501 = {}
        # Getting the type of 'sputils' (line 50)
        sputils_462496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 21), 'sputils', False)
        # Obtaining the member 'isshape' of a type (line 50)
        isshape_462497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 21), sputils_462496, 'isshape')
        # Calling isshape(args, kwargs) (line 50)
        isshape_call_result_462502 = invoke(stypy.reporting.localization.Localization(__file__, 50, 21), isshape_462497, *[tuple_462498], **kwargs_462501)
        
        # Getting the type of 'True' (line 50)
        True_462503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 46), 'True', False)
        # Processing the call keyword arguments (line 50)
        kwargs_462504 = {}
        # Getting the type of 'assert_equal' (line 50)
        assert_equal_462495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 50)
        assert_equal_call_result_462505 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), assert_equal_462495, *[isshape_call_result_462502, True_462503], **kwargs_462504)
        
        
        # Call to assert_equal(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to isshape(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Obtaining an instance of the builtin type 'tuple' (line 51)
        tuple_462509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 51)
        # Adding element type (line 51)
        int_462510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 38), tuple_462509, int_462510)
        # Adding element type (line 51)
        int_462511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 38), tuple_462509, int_462511)
        
        # Processing the call keyword arguments (line 51)
        kwargs_462512 = {}
        # Getting the type of 'sputils' (line 51)
        sputils_462507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 21), 'sputils', False)
        # Obtaining the member 'isshape' of a type (line 51)
        isshape_462508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 21), sputils_462507, 'isshape')
        # Calling isshape(args, kwargs) (line 51)
        isshape_call_result_462513 = invoke(stypy.reporting.localization.Localization(__file__, 51, 21), isshape_462508, *[tuple_462509], **kwargs_462512)
        
        # Getting the type of 'True' (line 51)
        True_462514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 46), 'True', False)
        # Processing the call keyword arguments (line 51)
        kwargs_462515 = {}
        # Getting the type of 'assert_equal' (line 51)
        assert_equal_462506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 51)
        assert_equal_call_result_462516 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assert_equal_462506, *[isshape_call_result_462513, True_462514], **kwargs_462515)
        
        
        # Call to assert_equal(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to isshape(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_462520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        float_462521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 38), tuple_462520, float_462521)
        # Adding element type (line 53)
        int_462522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 38), tuple_462520, int_462522)
        
        # Processing the call keyword arguments (line 53)
        kwargs_462523 = {}
        # Getting the type of 'sputils' (line 53)
        sputils_462518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 21), 'sputils', False)
        # Obtaining the member 'isshape' of a type (line 53)
        isshape_462519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 21), sputils_462518, 'isshape')
        # Calling isshape(args, kwargs) (line 53)
        isshape_call_result_462524 = invoke(stypy.reporting.localization.Localization(__file__, 53, 21), isshape_462519, *[tuple_462520], **kwargs_462523)
        
        # Getting the type of 'False' (line 53)
        False_462525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 48), 'False', False)
        # Processing the call keyword arguments (line 53)
        kwargs_462526 = {}
        # Getting the type of 'assert_equal' (line 53)
        assert_equal_462517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 53)
        assert_equal_call_result_462527 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), assert_equal_462517, *[isshape_call_result_462524, False_462525], **kwargs_462526)
        
        
        # Call to assert_equal(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Call to isshape(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_462531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        int_462532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 38), tuple_462531, int_462532)
        # Adding element type (line 54)
        int_462533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 38), tuple_462531, int_462533)
        # Adding element type (line 54)
        int_462534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 38), tuple_462531, int_462534)
        
        # Processing the call keyword arguments (line 54)
        kwargs_462535 = {}
        # Getting the type of 'sputils' (line 54)
        sputils_462529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'sputils', False)
        # Obtaining the member 'isshape' of a type (line 54)
        isshape_462530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 21), sputils_462529, 'isshape')
        # Calling isshape(args, kwargs) (line 54)
        isshape_call_result_462536 = invoke(stypy.reporting.localization.Localization(__file__, 54, 21), isshape_462530, *[tuple_462531], **kwargs_462535)
        
        # Getting the type of 'False' (line 54)
        False_462537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 49), 'False', False)
        # Processing the call keyword arguments (line 54)
        kwargs_462538 = {}
        # Getting the type of 'assert_equal' (line 54)
        assert_equal_462528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 54)
        assert_equal_call_result_462539 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assert_equal_462528, *[isshape_call_result_462536, False_462537], **kwargs_462538)
        
        
        # Call to assert_equal(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Call to isshape(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Obtaining an instance of the builtin type 'tuple' (line 55)
        tuple_462543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 55)
        # Adding element type (line 55)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_462544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        int_462545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 38), list_462544, int_462545)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 38), tuple_462543, list_462544)
        # Adding element type (line 55)
        int_462546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 38), tuple_462543, int_462546)
        
        # Processing the call keyword arguments (line 55)
        kwargs_462547 = {}
        # Getting the type of 'sputils' (line 55)
        sputils_462541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'sputils', False)
        # Obtaining the member 'isshape' of a type (line 55)
        isshape_462542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 21), sputils_462541, 'isshape')
        # Calling isshape(args, kwargs) (line 55)
        isshape_call_result_462548 = invoke(stypy.reporting.localization.Localization(__file__, 55, 21), isshape_462542, *[tuple_462543], **kwargs_462547)
        
        # Getting the type of 'False' (line 55)
        False_462549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 48), 'False', False)
        # Processing the call keyword arguments (line 55)
        kwargs_462550 = {}
        # Getting the type of 'assert_equal' (line 55)
        assert_equal_462540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 55)
        assert_equal_call_result_462551 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), assert_equal_462540, *[isshape_call_result_462548, False_462549], **kwargs_462550)
        
        
        # ################# End of 'test_isshape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_isshape' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_462552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_462552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_isshape'
        return stypy_return_type_462552


    @norecursion
    def test_issequence(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_issequence'
        module_type_store = module_type_store.open_function_context('test_issequence', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSparseUtils.test_issequence.__dict__.__setitem__('stypy_localization', localization)
        TestSparseUtils.test_issequence.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSparseUtils.test_issequence.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSparseUtils.test_issequence.__dict__.__setitem__('stypy_function_name', 'TestSparseUtils.test_issequence')
        TestSparseUtils.test_issequence.__dict__.__setitem__('stypy_param_names_list', [])
        TestSparseUtils.test_issequence.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSparseUtils.test_issequence.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSparseUtils.test_issequence.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSparseUtils.test_issequence.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSparseUtils.test_issequence.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSparseUtils.test_issequence.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseUtils.test_issequence', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_issequence', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_issequence(...)' code ##################

        
        # Call to assert_equal(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to issequence(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Obtaining an instance of the builtin type 'tuple' (line 58)
        tuple_462556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 58)
        # Adding element type (line 58)
        int_462557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 41), tuple_462556, int_462557)
        
        # Processing the call keyword arguments (line 58)
        kwargs_462558 = {}
        # Getting the type of 'sputils' (line 58)
        sputils_462554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'sputils', False)
        # Obtaining the member 'issequence' of a type (line 58)
        issequence_462555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 21), sputils_462554, 'issequence')
        # Calling issequence(args, kwargs) (line 58)
        issequence_call_result_462559 = invoke(stypy.reporting.localization.Localization(__file__, 58, 21), issequence_462555, *[tuple_462556], **kwargs_462558)
        
        # Getting the type of 'True' (line 58)
        True_462560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 47), 'True', False)
        # Processing the call keyword arguments (line 58)
        kwargs_462561 = {}
        # Getting the type of 'assert_equal' (line 58)
        assert_equal_462553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 58)
        assert_equal_call_result_462562 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), assert_equal_462553, *[issequence_call_result_462559, True_462560], **kwargs_462561)
        
        
        # Call to assert_equal(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Call to issequence(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Obtaining an instance of the builtin type 'tuple' (line 59)
        tuple_462566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 59)
        # Adding element type (line 59)
        int_462567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 41), tuple_462566, int_462567)
        # Adding element type (line 59)
        int_462568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 41), tuple_462566, int_462568)
        # Adding element type (line 59)
        int_462569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 41), tuple_462566, int_462569)
        
        # Processing the call keyword arguments (line 59)
        kwargs_462570 = {}
        # Getting the type of 'sputils' (line 59)
        sputils_462564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'sputils', False)
        # Obtaining the member 'issequence' of a type (line 59)
        issequence_462565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 21), sputils_462564, 'issequence')
        # Calling issequence(args, kwargs) (line 59)
        issequence_call_result_462571 = invoke(stypy.reporting.localization.Localization(__file__, 59, 21), issequence_462565, *[tuple_462566], **kwargs_462570)
        
        # Getting the type of 'True' (line 59)
        True_462572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 52), 'True', False)
        # Processing the call keyword arguments (line 59)
        kwargs_462573 = {}
        # Getting the type of 'assert_equal' (line 59)
        assert_equal_462563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 59)
        assert_equal_call_result_462574 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), assert_equal_462563, *[issequence_call_result_462571, True_462572], **kwargs_462573)
        
        
        # Call to assert_equal(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to issequence(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_462578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        int_462579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 40), list_462578, int_462579)
        
        # Processing the call keyword arguments (line 60)
        kwargs_462580 = {}
        # Getting the type of 'sputils' (line 60)
        sputils_462576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 21), 'sputils', False)
        # Obtaining the member 'issequence' of a type (line 60)
        issequence_462577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 21), sputils_462576, 'issequence')
        # Calling issequence(args, kwargs) (line 60)
        issequence_call_result_462581 = invoke(stypy.reporting.localization.Localization(__file__, 60, 21), issequence_462577, *[list_462578], **kwargs_462580)
        
        # Getting the type of 'True' (line 60)
        True_462582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 46), 'True', False)
        # Processing the call keyword arguments (line 60)
        kwargs_462583 = {}
        # Getting the type of 'assert_equal' (line 60)
        assert_equal_462575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 60)
        assert_equal_call_result_462584 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assert_equal_462575, *[issequence_call_result_462581, True_462582], **kwargs_462583)
        
        
        # Call to assert_equal(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Call to issequence(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_462588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_462589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 40), list_462588, int_462589)
        # Adding element type (line 61)
        int_462590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 40), list_462588, int_462590)
        # Adding element type (line 61)
        int_462591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 40), list_462588, int_462591)
        
        # Processing the call keyword arguments (line 61)
        kwargs_462592 = {}
        # Getting the type of 'sputils' (line 61)
        sputils_462586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'sputils', False)
        # Obtaining the member 'issequence' of a type (line 61)
        issequence_462587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 21), sputils_462586, 'issequence')
        # Calling issequence(args, kwargs) (line 61)
        issequence_call_result_462593 = invoke(stypy.reporting.localization.Localization(__file__, 61, 21), issequence_462587, *[list_462588], **kwargs_462592)
        
        # Getting the type of 'True' (line 61)
        True_462594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 52), 'True', False)
        # Processing the call keyword arguments (line 61)
        kwargs_462595 = {}
        # Getting the type of 'assert_equal' (line 61)
        assert_equal_462585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 61)
        assert_equal_call_result_462596 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assert_equal_462585, *[issequence_call_result_462593, True_462594], **kwargs_462595)
        
        
        # Call to assert_equal(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to issequence(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to array(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_462602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        int_462603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 49), list_462602, int_462603)
        # Adding element type (line 62)
        int_462604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 49), list_462602, int_462604)
        # Adding element type (line 62)
        int_462605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 49), list_462602, int_462605)
        
        # Processing the call keyword arguments (line 62)
        kwargs_462606 = {}
        # Getting the type of 'np' (line 62)
        np_462600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 40), 'np', False)
        # Obtaining the member 'array' of a type (line 62)
        array_462601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 40), np_462600, 'array')
        # Calling array(args, kwargs) (line 62)
        array_call_result_462607 = invoke(stypy.reporting.localization.Localization(__file__, 62, 40), array_462601, *[list_462602], **kwargs_462606)
        
        # Processing the call keyword arguments (line 62)
        kwargs_462608 = {}
        # Getting the type of 'sputils' (line 62)
        sputils_462598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 21), 'sputils', False)
        # Obtaining the member 'issequence' of a type (line 62)
        issequence_462599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 21), sputils_462598, 'issequence')
        # Calling issequence(args, kwargs) (line 62)
        issequence_call_result_462609 = invoke(stypy.reporting.localization.Localization(__file__, 62, 21), issequence_462599, *[array_call_result_462607], **kwargs_462608)
        
        # Getting the type of 'True' (line 62)
        True_462610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 62), 'True', False)
        # Processing the call keyword arguments (line 62)
        kwargs_462611 = {}
        # Getting the type of 'assert_equal' (line 62)
        assert_equal_462597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 62)
        assert_equal_call_result_462612 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), assert_equal_462597, *[issequence_call_result_462609, True_462610], **kwargs_462611)
        
        
        # Call to assert_equal(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to issequence(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to array(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_462618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_462619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        int_462620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 50), list_462619, int_462620)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 49), list_462618, list_462619)
        # Adding element type (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_462621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        int_462622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 55), list_462621, int_462622)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 49), list_462618, list_462621)
        # Adding element type (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_462623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        int_462624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 60), list_462623, int_462624)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 49), list_462618, list_462623)
        
        # Processing the call keyword arguments (line 64)
        kwargs_462625 = {}
        # Getting the type of 'np' (line 64)
        np_462616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 40), 'np', False)
        # Obtaining the member 'array' of a type (line 64)
        array_462617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 40), np_462616, 'array')
        # Calling array(args, kwargs) (line 64)
        array_call_result_462626 = invoke(stypy.reporting.localization.Localization(__file__, 64, 40), array_462617, *[list_462618], **kwargs_462625)
        
        # Processing the call keyword arguments (line 64)
        kwargs_462627 = {}
        # Getting the type of 'sputils' (line 64)
        sputils_462614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'sputils', False)
        # Obtaining the member 'issequence' of a type (line 64)
        issequence_462615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 21), sputils_462614, 'issequence')
        # Calling issequence(args, kwargs) (line 64)
        issequence_call_result_462628 = invoke(stypy.reporting.localization.Localization(__file__, 64, 21), issequence_462615, *[array_call_result_462626], **kwargs_462627)
        
        # Getting the type of 'False' (line 64)
        False_462629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 68), 'False', False)
        # Processing the call keyword arguments (line 64)
        kwargs_462630 = {}
        # Getting the type of 'assert_equal' (line 64)
        assert_equal_462613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 64)
        assert_equal_call_result_462631 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assert_equal_462613, *[issequence_call_result_462628, False_462629], **kwargs_462630)
        
        
        # Call to assert_equal(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Call to issequence(...): (line 65)
        # Processing the call arguments (line 65)
        int_462635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 40), 'int')
        # Processing the call keyword arguments (line 65)
        kwargs_462636 = {}
        # Getting the type of 'sputils' (line 65)
        sputils_462633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'sputils', False)
        # Obtaining the member 'issequence' of a type (line 65)
        issequence_462634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 21), sputils_462633, 'issequence')
        # Calling issequence(args, kwargs) (line 65)
        issequence_call_result_462637 = invoke(stypy.reporting.localization.Localization(__file__, 65, 21), issequence_462634, *[int_462635], **kwargs_462636)
        
        # Getting the type of 'False' (line 65)
        False_462638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 44), 'False', False)
        # Processing the call keyword arguments (line 65)
        kwargs_462639 = {}
        # Getting the type of 'assert_equal' (line 65)
        assert_equal_462632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 65)
        assert_equal_call_result_462640 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), assert_equal_462632, *[issequence_call_result_462637, False_462638], **kwargs_462639)
        
        
        # ################# End of 'test_issequence(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_issequence' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_462641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_462641)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_issequence'
        return stypy_return_type_462641


    @norecursion
    def test_ismatrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ismatrix'
        module_type_store = module_type_store.open_function_context('test_ismatrix', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSparseUtils.test_ismatrix.__dict__.__setitem__('stypy_localization', localization)
        TestSparseUtils.test_ismatrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSparseUtils.test_ismatrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSparseUtils.test_ismatrix.__dict__.__setitem__('stypy_function_name', 'TestSparseUtils.test_ismatrix')
        TestSparseUtils.test_ismatrix.__dict__.__setitem__('stypy_param_names_list', [])
        TestSparseUtils.test_ismatrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSparseUtils.test_ismatrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSparseUtils.test_ismatrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSparseUtils.test_ismatrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSparseUtils.test_ismatrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSparseUtils.test_ismatrix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseUtils.test_ismatrix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ismatrix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ismatrix(...)' code ##################

        
        # Call to assert_equal(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Call to ismatrix(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Obtaining an instance of the builtin type 'tuple' (line 68)
        tuple_462645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 68)
        # Adding element type (line 68)
        
        # Obtaining an instance of the builtin type 'tuple' (line 68)
        tuple_462646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 68)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 39), tuple_462645, tuple_462646)
        
        # Processing the call keyword arguments (line 68)
        kwargs_462647 = {}
        # Getting the type of 'sputils' (line 68)
        sputils_462643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'sputils', False)
        # Obtaining the member 'ismatrix' of a type (line 68)
        ismatrix_462644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 21), sputils_462643, 'ismatrix')
        # Calling ismatrix(args, kwargs) (line 68)
        ismatrix_call_result_462648 = invoke(stypy.reporting.localization.Localization(__file__, 68, 21), ismatrix_462644, *[tuple_462645], **kwargs_462647)
        
        # Getting the type of 'True' (line 68)
        True_462649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 46), 'True', False)
        # Processing the call keyword arguments (line 68)
        kwargs_462650 = {}
        # Getting the type of 'assert_equal' (line 68)
        assert_equal_462642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 68)
        assert_equal_call_result_462651 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), assert_equal_462642, *[ismatrix_call_result_462648, True_462649], **kwargs_462650)
        
        
        # Call to assert_equal(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Call to ismatrix(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_462655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_462656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        int_462657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 39), list_462656, int_462657)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 38), list_462655, list_462656)
        # Adding element type (line 69)
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_462658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        int_462659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 44), list_462658, int_462659)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 38), list_462655, list_462658)
        
        # Processing the call keyword arguments (line 69)
        kwargs_462660 = {}
        # Getting the type of 'sputils' (line 69)
        sputils_462653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'sputils', False)
        # Obtaining the member 'ismatrix' of a type (line 69)
        ismatrix_462654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 21), sputils_462653, 'ismatrix')
        # Calling ismatrix(args, kwargs) (line 69)
        ismatrix_call_result_462661 = invoke(stypy.reporting.localization.Localization(__file__, 69, 21), ismatrix_462654, *[list_462655], **kwargs_462660)
        
        # Getting the type of 'True' (line 69)
        True_462662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 51), 'True', False)
        # Processing the call keyword arguments (line 69)
        kwargs_462663 = {}
        # Getting the type of 'assert_equal' (line 69)
        assert_equal_462652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 69)
        assert_equal_call_result_462664 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), assert_equal_462652, *[ismatrix_call_result_462661, True_462662], **kwargs_462663)
        
        
        # Call to assert_equal(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Call to ismatrix(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 70)
        None_462668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 51), 'None', False)
        
        # Call to arange(...): (line 70)
        # Processing the call arguments (line 70)
        int_462671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 48), 'int')
        # Processing the call keyword arguments (line 70)
        kwargs_462672 = {}
        # Getting the type of 'np' (line 70)
        np_462669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'np', False)
        # Obtaining the member 'arange' of a type (line 70)
        arange_462670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 38), np_462669, 'arange')
        # Calling arange(args, kwargs) (line 70)
        arange_call_result_462673 = invoke(stypy.reporting.localization.Localization(__file__, 70, 38), arange_462670, *[int_462671], **kwargs_462672)
        
        # Obtaining the member '__getitem__' of a type (line 70)
        getitem___462674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 38), arange_call_result_462673, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 70)
        subscript_call_result_462675 = invoke(stypy.reporting.localization.Localization(__file__, 70, 38), getitem___462674, None_462668)
        
        # Processing the call keyword arguments (line 70)
        kwargs_462676 = {}
        # Getting the type of 'sputils' (line 70)
        sputils_462666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'sputils', False)
        # Obtaining the member 'ismatrix' of a type (line 70)
        ismatrix_462667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 21), sputils_462666, 'ismatrix')
        # Calling ismatrix(args, kwargs) (line 70)
        ismatrix_call_result_462677 = invoke(stypy.reporting.localization.Localization(__file__, 70, 21), ismatrix_462667, *[subscript_call_result_462675], **kwargs_462676)
        
        # Getting the type of 'True' (line 70)
        True_462678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 59), 'True', False)
        # Processing the call keyword arguments (line 70)
        kwargs_462679 = {}
        # Getting the type of 'assert_equal' (line 70)
        assert_equal_462665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 70)
        assert_equal_call_result_462680 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), assert_equal_462665, *[ismatrix_call_result_462677, True_462678], **kwargs_462679)
        
        
        # Call to assert_equal(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Call to ismatrix(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_462684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        # Adding element type (line 72)
        int_462685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 38), list_462684, int_462685)
        # Adding element type (line 72)
        int_462686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 38), list_462684, int_462686)
        
        # Processing the call keyword arguments (line 72)
        kwargs_462687 = {}
        # Getting the type of 'sputils' (line 72)
        sputils_462682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'sputils', False)
        # Obtaining the member 'ismatrix' of a type (line 72)
        ismatrix_462683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 21), sputils_462682, 'ismatrix')
        # Calling ismatrix(args, kwargs) (line 72)
        ismatrix_call_result_462688 = invoke(stypy.reporting.localization.Localization(__file__, 72, 21), ismatrix_462683, *[list_462684], **kwargs_462687)
        
        # Getting the type of 'False' (line 72)
        False_462689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 47), 'False', False)
        # Processing the call keyword arguments (line 72)
        kwargs_462690 = {}
        # Getting the type of 'assert_equal' (line 72)
        assert_equal_462681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 72)
        assert_equal_call_result_462691 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), assert_equal_462681, *[ismatrix_call_result_462688, False_462689], **kwargs_462690)
        
        
        # Call to assert_equal(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to ismatrix(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to arange(...): (line 73)
        # Processing the call arguments (line 73)
        int_462697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 48), 'int')
        # Processing the call keyword arguments (line 73)
        kwargs_462698 = {}
        # Getting the type of 'np' (line 73)
        np_462695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 38), 'np', False)
        # Obtaining the member 'arange' of a type (line 73)
        arange_462696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 38), np_462695, 'arange')
        # Calling arange(args, kwargs) (line 73)
        arange_call_result_462699 = invoke(stypy.reporting.localization.Localization(__file__, 73, 38), arange_462696, *[int_462697], **kwargs_462698)
        
        # Processing the call keyword arguments (line 73)
        kwargs_462700 = {}
        # Getting the type of 'sputils' (line 73)
        sputils_462693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 21), 'sputils', False)
        # Obtaining the member 'ismatrix' of a type (line 73)
        ismatrix_462694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 21), sputils_462693, 'ismatrix')
        # Calling ismatrix(args, kwargs) (line 73)
        ismatrix_call_result_462701 = invoke(stypy.reporting.localization.Localization(__file__, 73, 21), ismatrix_462694, *[arange_call_result_462699], **kwargs_462700)
        
        # Getting the type of 'False' (line 73)
        False_462702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 53), 'False', False)
        # Processing the call keyword arguments (line 73)
        kwargs_462703 = {}
        # Getting the type of 'assert_equal' (line 73)
        assert_equal_462692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 73)
        assert_equal_call_result_462704 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), assert_equal_462692, *[ismatrix_call_result_462701, False_462702], **kwargs_462703)
        
        
        # Call to assert_equal(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Call to ismatrix(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_462708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_462709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_462710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        int_462711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 40), list_462710, int_462711)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 39), list_462709, list_462710)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 38), list_462708, list_462709)
        
        # Processing the call keyword arguments (line 74)
        kwargs_462712 = {}
        # Getting the type of 'sputils' (line 74)
        sputils_462706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 21), 'sputils', False)
        # Obtaining the member 'ismatrix' of a type (line 74)
        ismatrix_462707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 21), sputils_462706, 'ismatrix')
        # Calling ismatrix(args, kwargs) (line 74)
        ismatrix_call_result_462713 = invoke(stypy.reporting.localization.Localization(__file__, 74, 21), ismatrix_462707, *[list_462708], **kwargs_462712)
        
        # Getting the type of 'False' (line 74)
        False_462714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 48), 'False', False)
        # Processing the call keyword arguments (line 74)
        kwargs_462715 = {}
        # Getting the type of 'assert_equal' (line 74)
        assert_equal_462705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 74)
        assert_equal_call_result_462716 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), assert_equal_462705, *[ismatrix_call_result_462713, False_462714], **kwargs_462715)
        
        
        # Call to assert_equal(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Call to ismatrix(...): (line 75)
        # Processing the call arguments (line 75)
        int_462720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 38), 'int')
        # Processing the call keyword arguments (line 75)
        kwargs_462721 = {}
        # Getting the type of 'sputils' (line 75)
        sputils_462718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 21), 'sputils', False)
        # Obtaining the member 'ismatrix' of a type (line 75)
        ismatrix_462719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 21), sputils_462718, 'ismatrix')
        # Calling ismatrix(args, kwargs) (line 75)
        ismatrix_call_result_462722 = invoke(stypy.reporting.localization.Localization(__file__, 75, 21), ismatrix_462719, *[int_462720], **kwargs_462721)
        
        # Getting the type of 'False' (line 75)
        False_462723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 42), 'False', False)
        # Processing the call keyword arguments (line 75)
        kwargs_462724 = {}
        # Getting the type of 'assert_equal' (line 75)
        assert_equal_462717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 75)
        assert_equal_call_result_462725 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), assert_equal_462717, *[ismatrix_call_result_462722, False_462723], **kwargs_462724)
        
        
        # ################# End of 'test_ismatrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ismatrix' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_462726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_462726)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ismatrix'
        return stypy_return_type_462726


    @norecursion
    def test_isdense(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_isdense'
        module_type_store = module_type_store.open_function_context('test_isdense', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSparseUtils.test_isdense.__dict__.__setitem__('stypy_localization', localization)
        TestSparseUtils.test_isdense.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSparseUtils.test_isdense.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSparseUtils.test_isdense.__dict__.__setitem__('stypy_function_name', 'TestSparseUtils.test_isdense')
        TestSparseUtils.test_isdense.__dict__.__setitem__('stypy_param_names_list', [])
        TestSparseUtils.test_isdense.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSparseUtils.test_isdense.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSparseUtils.test_isdense.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSparseUtils.test_isdense.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSparseUtils.test_isdense.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSparseUtils.test_isdense.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseUtils.test_isdense', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_isdense', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_isdense(...)' code ##################

        
        # Call to assert_equal(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Call to isdense(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Call to array(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_462732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        int_462733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 46), list_462732, int_462733)
        
        # Processing the call keyword arguments (line 78)
        kwargs_462734 = {}
        # Getting the type of 'np' (line 78)
        np_462730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 37), 'np', False)
        # Obtaining the member 'array' of a type (line 78)
        array_462731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 37), np_462730, 'array')
        # Calling array(args, kwargs) (line 78)
        array_call_result_462735 = invoke(stypy.reporting.localization.Localization(__file__, 78, 37), array_462731, *[list_462732], **kwargs_462734)
        
        # Processing the call keyword arguments (line 78)
        kwargs_462736 = {}
        # Getting the type of 'sputils' (line 78)
        sputils_462728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'sputils', False)
        # Obtaining the member 'isdense' of a type (line 78)
        isdense_462729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 21), sputils_462728, 'isdense')
        # Calling isdense(args, kwargs) (line 78)
        isdense_call_result_462737 = invoke(stypy.reporting.localization.Localization(__file__, 78, 21), isdense_462729, *[array_call_result_462735], **kwargs_462736)
        
        # Getting the type of 'True' (line 78)
        True_462738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 53), 'True', False)
        # Processing the call keyword arguments (line 78)
        kwargs_462739 = {}
        # Getting the type of 'assert_equal' (line 78)
        assert_equal_462727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 78)
        assert_equal_call_result_462740 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), assert_equal_462727, *[isdense_call_result_462737, True_462738], **kwargs_462739)
        
        
        # Call to assert_equal(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Call to isdense(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Call to matrix(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_462746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        int_462747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 47), list_462746, int_462747)
        
        # Processing the call keyword arguments (line 79)
        kwargs_462748 = {}
        # Getting the type of 'np' (line 79)
        np_462744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 37), 'np', False)
        # Obtaining the member 'matrix' of a type (line 79)
        matrix_462745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 37), np_462744, 'matrix')
        # Calling matrix(args, kwargs) (line 79)
        matrix_call_result_462749 = invoke(stypy.reporting.localization.Localization(__file__, 79, 37), matrix_462745, *[list_462746], **kwargs_462748)
        
        # Processing the call keyword arguments (line 79)
        kwargs_462750 = {}
        # Getting the type of 'sputils' (line 79)
        sputils_462742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'sputils', False)
        # Obtaining the member 'isdense' of a type (line 79)
        isdense_462743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 21), sputils_462742, 'isdense')
        # Calling isdense(args, kwargs) (line 79)
        isdense_call_result_462751 = invoke(stypy.reporting.localization.Localization(__file__, 79, 21), isdense_462743, *[matrix_call_result_462749], **kwargs_462750)
        
        # Getting the type of 'True' (line 79)
        True_462752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 54), 'True', False)
        # Processing the call keyword arguments (line 79)
        kwargs_462753 = {}
        # Getting the type of 'assert_equal' (line 79)
        assert_equal_462741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 79)
        assert_equal_call_result_462754 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), assert_equal_462741, *[isdense_call_result_462751, True_462752], **kwargs_462753)
        
        
        # ################# End of 'test_isdense(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_isdense' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_462755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_462755)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_isdense'
        return stypy_return_type_462755


    @norecursion
    def test_validateaxis(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_validateaxis'
        module_type_store = module_type_store.open_function_context('test_validateaxis', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSparseUtils.test_validateaxis.__dict__.__setitem__('stypy_localization', localization)
        TestSparseUtils.test_validateaxis.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSparseUtils.test_validateaxis.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSparseUtils.test_validateaxis.__dict__.__setitem__('stypy_function_name', 'TestSparseUtils.test_validateaxis')
        TestSparseUtils.test_validateaxis.__dict__.__setitem__('stypy_param_names_list', [])
        TestSparseUtils.test_validateaxis.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSparseUtils.test_validateaxis.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSparseUtils.test_validateaxis.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSparseUtils.test_validateaxis.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSparseUtils.test_validateaxis.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSparseUtils.test_validateaxis.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseUtils.test_validateaxis', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_validateaxis', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_validateaxis(...)' code ##################

        
        # Assigning a Attribute to a Name (line 82):
        # Getting the type of 'sputils' (line 82)
        sputils_462756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'sputils')
        # Obtaining the member 'validateaxis' of a type (line 82)
        validateaxis_462757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), sputils_462756, 'validateaxis')
        # Assigning a type to the variable 'func' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'func', validateaxis_462757)
        
        # Call to assert_raises(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'TypeError' (line 84)
        TypeError_462759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 22), 'TypeError', False)
        # Getting the type of 'func' (line 84)
        func_462760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 33), 'func', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 84)
        tuple_462761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 84)
        # Adding element type (line 84)
        int_462762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 40), tuple_462761, int_462762)
        # Adding element type (line 84)
        int_462763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 40), tuple_462761, int_462763)
        
        # Processing the call keyword arguments (line 84)
        kwargs_462764 = {}
        # Getting the type of 'assert_raises' (line 84)
        assert_raises_462758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 84)
        assert_raises_call_result_462765 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), assert_raises_462758, *[TypeError_462759, func_462760, tuple_462761], **kwargs_462764)
        
        
        # Call to assert_raises(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'TypeError' (line 85)
        TypeError_462767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 22), 'TypeError', False)
        # Getting the type of 'func' (line 85)
        func_462768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 33), 'func', False)
        float_462769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 39), 'float')
        # Processing the call keyword arguments (line 85)
        kwargs_462770 = {}
        # Getting the type of 'assert_raises' (line 85)
        assert_raises_462766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 85)
        assert_raises_call_result_462771 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), assert_raises_462766, *[TypeError_462767, func_462768, float_462769], **kwargs_462770)
        
        
        # Call to assert_raises(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'ValueError' (line 86)
        ValueError_462773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 22), 'ValueError', False)
        # Getting the type of 'func' (line 86)
        func_462774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 34), 'func', False)
        int_462775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 40), 'int')
        # Processing the call keyword arguments (line 86)
        kwargs_462776 = {}
        # Getting the type of 'assert_raises' (line 86)
        assert_raises_462772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 86)
        assert_raises_call_result_462777 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), assert_raises_462772, *[ValueError_462773, func_462774, int_462775], **kwargs_462776)
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 89)
        tuple_462778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 89)
        # Adding element type (line 89)
        int_462779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 21), tuple_462778, int_462779)
        # Adding element type (line 89)
        int_462780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 21), tuple_462778, int_462780)
        # Adding element type (line 89)
        int_462781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 21), tuple_462778, int_462781)
        # Adding element type (line 89)
        int_462782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 21), tuple_462778, int_462782)
        # Adding element type (line 89)
        # Getting the type of 'None' (line 89)
        None_462783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 21), tuple_462778, None_462783)
        
        # Testing the type of a for loop iterable (line 89)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 89, 8), tuple_462778)
        # Getting the type of the for loop variable (line 89)
        for_loop_var_462784 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 89, 8), tuple_462778)
        # Assigning a type to the variable 'axis' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'axis', for_loop_var_462784)
        # SSA begins for a for statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to func(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'axis' (line 90)
        axis_462786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 17), 'axis', False)
        # Processing the call keyword arguments (line 90)
        kwargs_462787 = {}
        # Getting the type of 'func' (line 90)
        func_462785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'func', False)
        # Calling func(args, kwargs) (line 90)
        func_call_result_462788 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), func_462785, *[axis_462786], **kwargs_462787)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_validateaxis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_validateaxis' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_462789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_462789)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_validateaxis'
        return stypy_return_type_462789


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSparseUtils.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSparseUtils' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'TestSparseUtils', TestSparseUtils)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
