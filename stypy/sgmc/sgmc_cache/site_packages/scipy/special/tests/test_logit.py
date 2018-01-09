
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import (assert_equal, assert_almost_equal,
5:         assert_allclose)
6: from scipy.special import logit, expit
7: 
8: 
9: class TestLogit(object):
10:     def check_logit_out(self, dtype, expected):
11:         a = np.linspace(0,1,10)
12:         a = np.array(a, dtype=dtype)
13:         olderr = np.seterr(divide='ignore')
14:         try:
15:             actual = logit(a)
16:         finally:
17:             np.seterr(**olderr)
18: 
19:         assert_almost_equal(actual, expected)
20: 
21:         assert_equal(actual.dtype, np.dtype(dtype))
22: 
23:     def test_float32(self):
24:         expected = np.array([-np.inf, -2.07944155,
25:                             -1.25276291, -0.69314718,
26:                             -0.22314353, 0.22314365,
27:                             0.6931473, 1.25276303,
28:                             2.07944155, np.inf], dtype=np.float32)
29:         self.check_logit_out('f4', expected)
30: 
31:     def test_float64(self):
32:         expected = np.array([-np.inf, -2.07944154,
33:                             -1.25276297, -0.69314718,
34:                             -0.22314355, 0.22314355,
35:                             0.69314718, 1.25276297,
36:                             2.07944154, np.inf])
37:         self.check_logit_out('f8', expected)
38: 
39:     def test_nan(self):
40:         expected = np.array([np.nan]*4)
41:         olderr = np.seterr(invalid='ignore')
42:         try:
43:             actual = logit(np.array([-3., -2., 2., 3.]))
44:         finally:
45:             np.seterr(**olderr)
46: 
47:         assert_equal(expected, actual)
48: 
49: 
50: class TestExpit(object):
51:     def check_expit_out(self, dtype, expected):
52:         a = np.linspace(-4,4,10)
53:         a = np.array(a, dtype=dtype)
54:         actual = expit(a)
55:         assert_almost_equal(actual, expected)
56:         assert_equal(actual.dtype, np.dtype(dtype))
57: 
58:     def test_float32(self):
59:         expected = np.array([0.01798621, 0.04265125,
60:                             0.09777259, 0.20860852,
61:                             0.39068246, 0.60931754,
62:                             0.79139149, 0.9022274,
63:                             0.95734876, 0.98201376], dtype=np.float32)
64:         self.check_expit_out('f4',expected)
65: 
66:     def test_float64(self):
67:         expected = np.array([0.01798621, 0.04265125,
68:                             0.0977726, 0.20860853,
69:                             0.39068246, 0.60931754,
70:                             0.79139147, 0.9022274,
71:                             0.95734875, 0.98201379])
72:         self.check_expit_out('f8', expected)
73: 
74:     def test_large(self):
75:         for dtype in (np.float32, np.float64, np.longdouble):
76:             for n in (88, 89, 709, 710, 11356, 11357):
77:                 n = np.array(n, dtype=dtype)
78:                 assert_allclose(expit(n), 1.0, atol=1e-20)
79:                 assert_allclose(expit(-n), 0.0, atol=1e-20)
80:                 assert_equal(expit(n).dtype, dtype)
81:                 assert_equal(expit(-n).dtype, dtype)
82: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_541728 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_541728) is not StypyTypeError):

    if (import_541728 != 'pyd_module'):
        __import__(import_541728)
        sys_modules_541729 = sys.modules[import_541728]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_541729.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_541728)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_equal, assert_almost_equal, assert_allclose' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_541730 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_541730) is not StypyTypeError):

    if (import_541730 != 'pyd_module'):
        __import__(import_541730)
        sys_modules_541731 = sys.modules[import_541730]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_541731.module_type_store, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_541731, sys_modules_541731.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_almost_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_allclose'], [assert_equal, assert_almost_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_541730)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.special import logit, expit' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_541732 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special')

if (type(import_541732) is not StypyTypeError):

    if (import_541732 != 'pyd_module'):
        __import__(import_541732)
        sys_modules_541733 = sys.modules[import_541732]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special', sys_modules_541733.module_type_store, module_type_store, ['logit', 'expit'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_541733, sys_modules_541733.module_type_store, module_type_store)
    else:
        from scipy.special import logit, expit

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special', None, module_type_store, ['logit', 'expit'], [logit, expit])

else:
    # Assigning a type to the variable 'scipy.special' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special', import_541732)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

# Declaration of the 'TestLogit' class

class TestLogit(object, ):

    @norecursion
    def check_logit_out(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_logit_out'
        module_type_store = module_type_store.open_function_context('check_logit_out', 10, 4, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLogit.check_logit_out.__dict__.__setitem__('stypy_localization', localization)
        TestLogit.check_logit_out.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLogit.check_logit_out.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLogit.check_logit_out.__dict__.__setitem__('stypy_function_name', 'TestLogit.check_logit_out')
        TestLogit.check_logit_out.__dict__.__setitem__('stypy_param_names_list', ['dtype', 'expected'])
        TestLogit.check_logit_out.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLogit.check_logit_out.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLogit.check_logit_out.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLogit.check_logit_out.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLogit.check_logit_out.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLogit.check_logit_out.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLogit.check_logit_out', ['dtype', 'expected'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_logit_out', localization, ['dtype', 'expected'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_logit_out(...)' code ##################

        
        # Assigning a Call to a Name (line 11):
        
        # Call to linspace(...): (line 11)
        # Processing the call arguments (line 11)
        int_541736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 24), 'int')
        int_541737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'int')
        int_541738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 28), 'int')
        # Processing the call keyword arguments (line 11)
        kwargs_541739 = {}
        # Getting the type of 'np' (line 11)
        np_541734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 11)
        linspace_541735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 12), np_541734, 'linspace')
        # Calling linspace(args, kwargs) (line 11)
        linspace_call_result_541740 = invoke(stypy.reporting.localization.Localization(__file__, 11, 12), linspace_541735, *[int_541736, int_541737, int_541738], **kwargs_541739)
        
        # Assigning a type to the variable 'a' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'a', linspace_call_result_541740)
        
        # Assigning a Call to a Name (line 12):
        
        # Call to array(...): (line 12)
        # Processing the call arguments (line 12)
        # Getting the type of 'a' (line 12)
        a_541743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 'a', False)
        # Processing the call keyword arguments (line 12)
        # Getting the type of 'dtype' (line 12)
        dtype_541744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 30), 'dtype', False)
        keyword_541745 = dtype_541744
        kwargs_541746 = {'dtype': keyword_541745}
        # Getting the type of 'np' (line 12)
        np_541741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 12)
        array_541742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 12), np_541741, 'array')
        # Calling array(args, kwargs) (line 12)
        array_call_result_541747 = invoke(stypy.reporting.localization.Localization(__file__, 12, 12), array_541742, *[a_541743], **kwargs_541746)
        
        # Assigning a type to the variable 'a' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'a', array_call_result_541747)
        
        # Assigning a Call to a Name (line 13):
        
        # Call to seterr(...): (line 13)
        # Processing the call keyword arguments (line 13)
        str_541750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 34), 'str', 'ignore')
        keyword_541751 = str_541750
        kwargs_541752 = {'divide': keyword_541751}
        # Getting the type of 'np' (line 13)
        np_541748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 17), 'np', False)
        # Obtaining the member 'seterr' of a type (line 13)
        seterr_541749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 17), np_541748, 'seterr')
        # Calling seterr(args, kwargs) (line 13)
        seterr_call_result_541753 = invoke(stypy.reporting.localization.Localization(__file__, 13, 17), seterr_541749, *[], **kwargs_541752)
        
        # Assigning a type to the variable 'olderr' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'olderr', seterr_call_result_541753)
        
        # Try-finally block (line 14)
        
        # Assigning a Call to a Name (line 15):
        
        # Call to logit(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'a' (line 15)
        a_541755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 27), 'a', False)
        # Processing the call keyword arguments (line 15)
        kwargs_541756 = {}
        # Getting the type of 'logit' (line 15)
        logit_541754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'logit', False)
        # Calling logit(args, kwargs) (line 15)
        logit_call_result_541757 = invoke(stypy.reporting.localization.Localization(__file__, 15, 21), logit_541754, *[a_541755], **kwargs_541756)
        
        # Assigning a type to the variable 'actual' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'actual', logit_call_result_541757)
        
        # finally branch of the try-finally block (line 14)
        
        # Call to seterr(...): (line 17)
        # Processing the call keyword arguments (line 17)
        # Getting the type of 'olderr' (line 17)
        olderr_541760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'olderr', False)
        kwargs_541761 = {'olderr_541760': olderr_541760}
        # Getting the type of 'np' (line 17)
        np_541758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'np', False)
        # Obtaining the member 'seterr' of a type (line 17)
        seterr_541759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 12), np_541758, 'seterr')
        # Calling seterr(args, kwargs) (line 17)
        seterr_call_result_541762 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), seterr_541759, *[], **kwargs_541761)
        
        
        
        # Call to assert_almost_equal(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'actual' (line 19)
        actual_541764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 28), 'actual', False)
        # Getting the type of 'expected' (line 19)
        expected_541765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 36), 'expected', False)
        # Processing the call keyword arguments (line 19)
        kwargs_541766 = {}
        # Getting the type of 'assert_almost_equal' (line 19)
        assert_almost_equal_541763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 19)
        assert_almost_equal_call_result_541767 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), assert_almost_equal_541763, *[actual_541764, expected_541765], **kwargs_541766)
        
        
        # Call to assert_equal(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'actual' (line 21)
        actual_541769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'actual', False)
        # Obtaining the member 'dtype' of a type (line 21)
        dtype_541770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 21), actual_541769, 'dtype')
        
        # Call to dtype(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'dtype' (line 21)
        dtype_541773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 44), 'dtype', False)
        # Processing the call keyword arguments (line 21)
        kwargs_541774 = {}
        # Getting the type of 'np' (line 21)
        np_541771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 35), 'np', False)
        # Obtaining the member 'dtype' of a type (line 21)
        dtype_541772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 35), np_541771, 'dtype')
        # Calling dtype(args, kwargs) (line 21)
        dtype_call_result_541775 = invoke(stypy.reporting.localization.Localization(__file__, 21, 35), dtype_541772, *[dtype_541773], **kwargs_541774)
        
        # Processing the call keyword arguments (line 21)
        kwargs_541776 = {}
        # Getting the type of 'assert_equal' (line 21)
        assert_equal_541768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 21)
        assert_equal_call_result_541777 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assert_equal_541768, *[dtype_541770, dtype_call_result_541775], **kwargs_541776)
        
        
        # ################# End of 'check_logit_out(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_logit_out' in the type store
        # Getting the type of 'stypy_return_type' (line 10)
        stypy_return_type_541778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_541778)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_logit_out'
        return stypy_return_type_541778


    @norecursion
    def test_float32(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_float32'
        module_type_store = module_type_store.open_function_context('test_float32', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLogit.test_float32.__dict__.__setitem__('stypy_localization', localization)
        TestLogit.test_float32.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLogit.test_float32.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLogit.test_float32.__dict__.__setitem__('stypy_function_name', 'TestLogit.test_float32')
        TestLogit.test_float32.__dict__.__setitem__('stypy_param_names_list', [])
        TestLogit.test_float32.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLogit.test_float32.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLogit.test_float32.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLogit.test_float32.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLogit.test_float32.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLogit.test_float32.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLogit.test_float32', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_float32', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_float32(...)' code ##################

        
        # Assigning a Call to a Name (line 24):
        
        # Call to array(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Obtaining an instance of the builtin type 'list' (line 24)
        list_541781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 24)
        # Adding element type (line 24)
        
        # Getting the type of 'np' (line 24)
        np_541782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 30), 'np', False)
        # Obtaining the member 'inf' of a type (line 24)
        inf_541783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 30), np_541782, 'inf')
        # Applying the 'usub' unary operator (line 24)
        result___neg___541784 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 29), 'usub', inf_541783)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 28), list_541781, result___neg___541784)
        # Adding element type (line 24)
        float_541785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 28), list_541781, float_541785)
        # Adding element type (line 24)
        float_541786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 28), list_541781, float_541786)
        # Adding element type (line 24)
        float_541787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 28), list_541781, float_541787)
        # Adding element type (line 24)
        float_541788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 28), list_541781, float_541788)
        # Adding element type (line 24)
        float_541789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 28), list_541781, float_541789)
        # Adding element type (line 24)
        float_541790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 28), list_541781, float_541790)
        # Adding element type (line 24)
        float_541791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 28), list_541781, float_541791)
        # Adding element type (line 24)
        float_541792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 28), list_541781, float_541792)
        # Adding element type (line 24)
        # Getting the type of 'np' (line 28)
        np_541793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), 'np', False)
        # Obtaining the member 'inf' of a type (line 28)
        inf_541794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 40), np_541793, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 28), list_541781, inf_541794)
        
        # Processing the call keyword arguments (line 24)
        # Getting the type of 'np' (line 28)
        np_541795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 55), 'np', False)
        # Obtaining the member 'float32' of a type (line 28)
        float32_541796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 55), np_541795, 'float32')
        keyword_541797 = float32_541796
        kwargs_541798 = {'dtype': keyword_541797}
        # Getting the type of 'np' (line 24)
        np_541779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 24)
        array_541780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 19), np_541779, 'array')
        # Calling array(args, kwargs) (line 24)
        array_call_result_541799 = invoke(stypy.reporting.localization.Localization(__file__, 24, 19), array_541780, *[list_541781], **kwargs_541798)
        
        # Assigning a type to the variable 'expected' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'expected', array_call_result_541799)
        
        # Call to check_logit_out(...): (line 29)
        # Processing the call arguments (line 29)
        str_541802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'str', 'f4')
        # Getting the type of 'expected' (line 29)
        expected_541803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 35), 'expected', False)
        # Processing the call keyword arguments (line 29)
        kwargs_541804 = {}
        # Getting the type of 'self' (line 29)
        self_541800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self', False)
        # Obtaining the member 'check_logit_out' of a type (line 29)
        check_logit_out_541801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_541800, 'check_logit_out')
        # Calling check_logit_out(args, kwargs) (line 29)
        check_logit_out_call_result_541805 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), check_logit_out_541801, *[str_541802, expected_541803], **kwargs_541804)
        
        
        # ################# End of 'test_float32(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_float32' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_541806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_541806)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_float32'
        return stypy_return_type_541806


    @norecursion
    def test_float64(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_float64'
        module_type_store = module_type_store.open_function_context('test_float64', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLogit.test_float64.__dict__.__setitem__('stypy_localization', localization)
        TestLogit.test_float64.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLogit.test_float64.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLogit.test_float64.__dict__.__setitem__('stypy_function_name', 'TestLogit.test_float64')
        TestLogit.test_float64.__dict__.__setitem__('stypy_param_names_list', [])
        TestLogit.test_float64.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLogit.test_float64.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLogit.test_float64.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLogit.test_float64.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLogit.test_float64.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLogit.test_float64.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLogit.test_float64', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_float64', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_float64(...)' code ##################

        
        # Assigning a Call to a Name (line 32):
        
        # Call to array(...): (line 32)
        # Processing the call arguments (line 32)
        
        # Obtaining an instance of the builtin type 'list' (line 32)
        list_541809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 32)
        # Adding element type (line 32)
        
        # Getting the type of 'np' (line 32)
        np_541810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 30), 'np', False)
        # Obtaining the member 'inf' of a type (line 32)
        inf_541811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 30), np_541810, 'inf')
        # Applying the 'usub' unary operator (line 32)
        result___neg___541812 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 29), 'usub', inf_541811)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 28), list_541809, result___neg___541812)
        # Adding element type (line 32)
        float_541813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 28), list_541809, float_541813)
        # Adding element type (line 32)
        float_541814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 28), list_541809, float_541814)
        # Adding element type (line 32)
        float_541815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 28), list_541809, float_541815)
        # Adding element type (line 32)
        float_541816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 28), list_541809, float_541816)
        # Adding element type (line 32)
        float_541817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 28), list_541809, float_541817)
        # Adding element type (line 32)
        float_541818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 28), list_541809, float_541818)
        # Adding element type (line 32)
        float_541819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 28), list_541809, float_541819)
        # Adding element type (line 32)
        float_541820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 28), list_541809, float_541820)
        # Adding element type (line 32)
        # Getting the type of 'np' (line 36)
        np_541821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 40), 'np', False)
        # Obtaining the member 'inf' of a type (line 36)
        inf_541822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 40), np_541821, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 28), list_541809, inf_541822)
        
        # Processing the call keyword arguments (line 32)
        kwargs_541823 = {}
        # Getting the type of 'np' (line 32)
        np_541807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 32)
        array_541808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 19), np_541807, 'array')
        # Calling array(args, kwargs) (line 32)
        array_call_result_541824 = invoke(stypy.reporting.localization.Localization(__file__, 32, 19), array_541808, *[list_541809], **kwargs_541823)
        
        # Assigning a type to the variable 'expected' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'expected', array_call_result_541824)
        
        # Call to check_logit_out(...): (line 37)
        # Processing the call arguments (line 37)
        str_541827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 29), 'str', 'f8')
        # Getting the type of 'expected' (line 37)
        expected_541828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 35), 'expected', False)
        # Processing the call keyword arguments (line 37)
        kwargs_541829 = {}
        # Getting the type of 'self' (line 37)
        self_541825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self', False)
        # Obtaining the member 'check_logit_out' of a type (line 37)
        check_logit_out_541826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_541825, 'check_logit_out')
        # Calling check_logit_out(args, kwargs) (line 37)
        check_logit_out_call_result_541830 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), check_logit_out_541826, *[str_541827, expected_541828], **kwargs_541829)
        
        
        # ################# End of 'test_float64(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_float64' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_541831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_541831)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_float64'
        return stypy_return_type_541831


    @norecursion
    def test_nan(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_nan'
        module_type_store = module_type_store.open_function_context('test_nan', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLogit.test_nan.__dict__.__setitem__('stypy_localization', localization)
        TestLogit.test_nan.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLogit.test_nan.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLogit.test_nan.__dict__.__setitem__('stypy_function_name', 'TestLogit.test_nan')
        TestLogit.test_nan.__dict__.__setitem__('stypy_param_names_list', [])
        TestLogit.test_nan.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLogit.test_nan.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLogit.test_nan.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLogit.test_nan.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLogit.test_nan.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLogit.test_nan.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLogit.test_nan', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_nan', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_nan(...)' code ##################

        
        # Assigning a Call to a Name (line 40):
        
        # Call to array(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_541834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        # Getting the type of 'np' (line 40)
        np_541835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 29), 'np', False)
        # Obtaining the member 'nan' of a type (line 40)
        nan_541836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 29), np_541835, 'nan')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 28), list_541834, nan_541836)
        
        int_541837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 37), 'int')
        # Applying the binary operator '*' (line 40)
        result_mul_541838 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 28), '*', list_541834, int_541837)
        
        # Processing the call keyword arguments (line 40)
        kwargs_541839 = {}
        # Getting the type of 'np' (line 40)
        np_541832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 40)
        array_541833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), np_541832, 'array')
        # Calling array(args, kwargs) (line 40)
        array_call_result_541840 = invoke(stypy.reporting.localization.Localization(__file__, 40, 19), array_541833, *[result_mul_541838], **kwargs_541839)
        
        # Assigning a type to the variable 'expected' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'expected', array_call_result_541840)
        
        # Assigning a Call to a Name (line 41):
        
        # Call to seterr(...): (line 41)
        # Processing the call keyword arguments (line 41)
        str_541843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 35), 'str', 'ignore')
        keyword_541844 = str_541843
        kwargs_541845 = {'invalid': keyword_541844}
        # Getting the type of 'np' (line 41)
        np_541841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'np', False)
        # Obtaining the member 'seterr' of a type (line 41)
        seterr_541842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 17), np_541841, 'seterr')
        # Calling seterr(args, kwargs) (line 41)
        seterr_call_result_541846 = invoke(stypy.reporting.localization.Localization(__file__, 41, 17), seterr_541842, *[], **kwargs_541845)
        
        # Assigning a type to the variable 'olderr' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'olderr', seterr_call_result_541846)
        
        # Try-finally block (line 42)
        
        # Assigning a Call to a Name (line 43):
        
        # Call to logit(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Call to array(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_541850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        float_541851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 36), list_541850, float_541851)
        # Adding element type (line 43)
        float_541852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 36), list_541850, float_541852)
        # Adding element type (line 43)
        float_541853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 36), list_541850, float_541853)
        # Adding element type (line 43)
        float_541854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 36), list_541850, float_541854)
        
        # Processing the call keyword arguments (line 43)
        kwargs_541855 = {}
        # Getting the type of 'np' (line 43)
        np_541848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 27), 'np', False)
        # Obtaining the member 'array' of a type (line 43)
        array_541849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 27), np_541848, 'array')
        # Calling array(args, kwargs) (line 43)
        array_call_result_541856 = invoke(stypy.reporting.localization.Localization(__file__, 43, 27), array_541849, *[list_541850], **kwargs_541855)
        
        # Processing the call keyword arguments (line 43)
        kwargs_541857 = {}
        # Getting the type of 'logit' (line 43)
        logit_541847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'logit', False)
        # Calling logit(args, kwargs) (line 43)
        logit_call_result_541858 = invoke(stypy.reporting.localization.Localization(__file__, 43, 21), logit_541847, *[array_call_result_541856], **kwargs_541857)
        
        # Assigning a type to the variable 'actual' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'actual', logit_call_result_541858)
        
        # finally branch of the try-finally block (line 42)
        
        # Call to seterr(...): (line 45)
        # Processing the call keyword arguments (line 45)
        # Getting the type of 'olderr' (line 45)
        olderr_541861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'olderr', False)
        kwargs_541862 = {'olderr_541861': olderr_541861}
        # Getting the type of 'np' (line 45)
        np_541859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'np', False)
        # Obtaining the member 'seterr' of a type (line 45)
        seterr_541860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), np_541859, 'seterr')
        # Calling seterr(args, kwargs) (line 45)
        seterr_call_result_541863 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), seterr_541860, *[], **kwargs_541862)
        
        
        
        # Call to assert_equal(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'expected' (line 47)
        expected_541865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 21), 'expected', False)
        # Getting the type of 'actual' (line 47)
        actual_541866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'actual', False)
        # Processing the call keyword arguments (line 47)
        kwargs_541867 = {}
        # Getting the type of 'assert_equal' (line 47)
        assert_equal_541864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 47)
        assert_equal_call_result_541868 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assert_equal_541864, *[expected_541865, actual_541866], **kwargs_541867)
        
        
        # ################# End of 'test_nan(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_nan' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_541869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_541869)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_nan'
        return stypy_return_type_541869


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 0, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLogit.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLogit' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'TestLogit', TestLogit)
# Declaration of the 'TestExpit' class

class TestExpit(object, ):

    @norecursion
    def check_expit_out(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_expit_out'
        module_type_store = module_type_store.open_function_context('check_expit_out', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpit.check_expit_out.__dict__.__setitem__('stypy_localization', localization)
        TestExpit.check_expit_out.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpit.check_expit_out.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpit.check_expit_out.__dict__.__setitem__('stypy_function_name', 'TestExpit.check_expit_out')
        TestExpit.check_expit_out.__dict__.__setitem__('stypy_param_names_list', ['dtype', 'expected'])
        TestExpit.check_expit_out.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpit.check_expit_out.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpit.check_expit_out.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpit.check_expit_out.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpit.check_expit_out.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpit.check_expit_out.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpit.check_expit_out', ['dtype', 'expected'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_expit_out', localization, ['dtype', 'expected'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_expit_out(...)' code ##################

        
        # Assigning a Call to a Name (line 52):
        
        # Call to linspace(...): (line 52)
        # Processing the call arguments (line 52)
        int_541872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 24), 'int')
        int_541873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 27), 'int')
        int_541874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'int')
        # Processing the call keyword arguments (line 52)
        kwargs_541875 = {}
        # Getting the type of 'np' (line 52)
        np_541870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 52)
        linspace_541871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), np_541870, 'linspace')
        # Calling linspace(args, kwargs) (line 52)
        linspace_call_result_541876 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), linspace_541871, *[int_541872, int_541873, int_541874], **kwargs_541875)
        
        # Assigning a type to the variable 'a' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'a', linspace_call_result_541876)
        
        # Assigning a Call to a Name (line 53):
        
        # Call to array(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'a' (line 53)
        a_541879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 21), 'a', False)
        # Processing the call keyword arguments (line 53)
        # Getting the type of 'dtype' (line 53)
        dtype_541880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 30), 'dtype', False)
        keyword_541881 = dtype_541880
        kwargs_541882 = {'dtype': keyword_541881}
        # Getting the type of 'np' (line 53)
        np_541877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 53)
        array_541878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 12), np_541877, 'array')
        # Calling array(args, kwargs) (line 53)
        array_call_result_541883 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), array_541878, *[a_541879], **kwargs_541882)
        
        # Assigning a type to the variable 'a' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'a', array_call_result_541883)
        
        # Assigning a Call to a Name (line 54):
        
        # Call to expit(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'a' (line 54)
        a_541885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 'a', False)
        # Processing the call keyword arguments (line 54)
        kwargs_541886 = {}
        # Getting the type of 'expit' (line 54)
        expit_541884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'expit', False)
        # Calling expit(args, kwargs) (line 54)
        expit_call_result_541887 = invoke(stypy.reporting.localization.Localization(__file__, 54, 17), expit_541884, *[a_541885], **kwargs_541886)
        
        # Assigning a type to the variable 'actual' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'actual', expit_call_result_541887)
        
        # Call to assert_almost_equal(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'actual' (line 55)
        actual_541889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 28), 'actual', False)
        # Getting the type of 'expected' (line 55)
        expected_541890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 36), 'expected', False)
        # Processing the call keyword arguments (line 55)
        kwargs_541891 = {}
        # Getting the type of 'assert_almost_equal' (line 55)
        assert_almost_equal_541888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 55)
        assert_almost_equal_call_result_541892 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), assert_almost_equal_541888, *[actual_541889, expected_541890], **kwargs_541891)
        
        
        # Call to assert_equal(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'actual' (line 56)
        actual_541894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'actual', False)
        # Obtaining the member 'dtype' of a type (line 56)
        dtype_541895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 21), actual_541894, 'dtype')
        
        # Call to dtype(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'dtype' (line 56)
        dtype_541898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 44), 'dtype', False)
        # Processing the call keyword arguments (line 56)
        kwargs_541899 = {}
        # Getting the type of 'np' (line 56)
        np_541896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 35), 'np', False)
        # Obtaining the member 'dtype' of a type (line 56)
        dtype_541897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 35), np_541896, 'dtype')
        # Calling dtype(args, kwargs) (line 56)
        dtype_call_result_541900 = invoke(stypy.reporting.localization.Localization(__file__, 56, 35), dtype_541897, *[dtype_541898], **kwargs_541899)
        
        # Processing the call keyword arguments (line 56)
        kwargs_541901 = {}
        # Getting the type of 'assert_equal' (line 56)
        assert_equal_541893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 56)
        assert_equal_call_result_541902 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), assert_equal_541893, *[dtype_541895, dtype_call_result_541900], **kwargs_541901)
        
        
        # ################# End of 'check_expit_out(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_expit_out' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_541903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_541903)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_expit_out'
        return stypy_return_type_541903


    @norecursion
    def test_float32(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_float32'
        module_type_store = module_type_store.open_function_context('test_float32', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpit.test_float32.__dict__.__setitem__('stypy_localization', localization)
        TestExpit.test_float32.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpit.test_float32.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpit.test_float32.__dict__.__setitem__('stypy_function_name', 'TestExpit.test_float32')
        TestExpit.test_float32.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpit.test_float32.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpit.test_float32.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpit.test_float32.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpit.test_float32.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpit.test_float32.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpit.test_float32.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpit.test_float32', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_float32', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_float32(...)' code ##################

        
        # Assigning a Call to a Name (line 59):
        
        # Call to array(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_541906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        # Adding element type (line 59)
        float_541907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 28), list_541906, float_541907)
        # Adding element type (line 59)
        float_541908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 28), list_541906, float_541908)
        # Adding element type (line 59)
        float_541909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 28), list_541906, float_541909)
        # Adding element type (line 59)
        float_541910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 28), list_541906, float_541910)
        # Adding element type (line 59)
        float_541911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 28), list_541906, float_541911)
        # Adding element type (line 59)
        float_541912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 28), list_541906, float_541912)
        # Adding element type (line 59)
        float_541913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 28), list_541906, float_541913)
        # Adding element type (line 59)
        float_541914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 28), list_541906, float_541914)
        # Adding element type (line 59)
        float_541915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 28), list_541906, float_541915)
        # Adding element type (line 59)
        float_541916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 28), list_541906, float_541916)
        
        # Processing the call keyword arguments (line 59)
        # Getting the type of 'np' (line 63)
        np_541917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 59), 'np', False)
        # Obtaining the member 'float32' of a type (line 63)
        float32_541918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 59), np_541917, 'float32')
        keyword_541919 = float32_541918
        kwargs_541920 = {'dtype': keyword_541919}
        # Getting the type of 'np' (line 59)
        np_541904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 59)
        array_541905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 19), np_541904, 'array')
        # Calling array(args, kwargs) (line 59)
        array_call_result_541921 = invoke(stypy.reporting.localization.Localization(__file__, 59, 19), array_541905, *[list_541906], **kwargs_541920)
        
        # Assigning a type to the variable 'expected' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'expected', array_call_result_541921)
        
        # Call to check_expit_out(...): (line 64)
        # Processing the call arguments (line 64)
        str_541924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 29), 'str', 'f4')
        # Getting the type of 'expected' (line 64)
        expected_541925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'expected', False)
        # Processing the call keyword arguments (line 64)
        kwargs_541926 = {}
        # Getting the type of 'self' (line 64)
        self_541922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self', False)
        # Obtaining the member 'check_expit_out' of a type (line 64)
        check_expit_out_541923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_541922, 'check_expit_out')
        # Calling check_expit_out(args, kwargs) (line 64)
        check_expit_out_call_result_541927 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), check_expit_out_541923, *[str_541924, expected_541925], **kwargs_541926)
        
        
        # ################# End of 'test_float32(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_float32' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_541928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_541928)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_float32'
        return stypy_return_type_541928


    @norecursion
    def test_float64(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_float64'
        module_type_store = module_type_store.open_function_context('test_float64', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpit.test_float64.__dict__.__setitem__('stypy_localization', localization)
        TestExpit.test_float64.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpit.test_float64.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpit.test_float64.__dict__.__setitem__('stypy_function_name', 'TestExpit.test_float64')
        TestExpit.test_float64.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpit.test_float64.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpit.test_float64.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpit.test_float64.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpit.test_float64.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpit.test_float64.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpit.test_float64.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpit.test_float64', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_float64', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_float64(...)' code ##################

        
        # Assigning a Call to a Name (line 67):
        
        # Call to array(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_541931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        float_541932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 28), list_541931, float_541932)
        # Adding element type (line 67)
        float_541933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 28), list_541931, float_541933)
        # Adding element type (line 67)
        float_541934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 28), list_541931, float_541934)
        # Adding element type (line 67)
        float_541935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 28), list_541931, float_541935)
        # Adding element type (line 67)
        float_541936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 28), list_541931, float_541936)
        # Adding element type (line 67)
        float_541937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 28), list_541931, float_541937)
        # Adding element type (line 67)
        float_541938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 28), list_541931, float_541938)
        # Adding element type (line 67)
        float_541939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 28), list_541931, float_541939)
        # Adding element type (line 67)
        float_541940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 28), list_541931, float_541940)
        # Adding element type (line 67)
        float_541941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 28), list_541931, float_541941)
        
        # Processing the call keyword arguments (line 67)
        kwargs_541942 = {}
        # Getting the type of 'np' (line 67)
        np_541929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 67)
        array_541930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 19), np_541929, 'array')
        # Calling array(args, kwargs) (line 67)
        array_call_result_541943 = invoke(stypy.reporting.localization.Localization(__file__, 67, 19), array_541930, *[list_541931], **kwargs_541942)
        
        # Assigning a type to the variable 'expected' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'expected', array_call_result_541943)
        
        # Call to check_expit_out(...): (line 72)
        # Processing the call arguments (line 72)
        str_541946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 29), 'str', 'f8')
        # Getting the type of 'expected' (line 72)
        expected_541947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 35), 'expected', False)
        # Processing the call keyword arguments (line 72)
        kwargs_541948 = {}
        # Getting the type of 'self' (line 72)
        self_541944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self', False)
        # Obtaining the member 'check_expit_out' of a type (line 72)
        check_expit_out_541945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_541944, 'check_expit_out')
        # Calling check_expit_out(args, kwargs) (line 72)
        check_expit_out_call_result_541949 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), check_expit_out_541945, *[str_541946, expected_541947], **kwargs_541948)
        
        
        # ################# End of 'test_float64(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_float64' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_541950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_541950)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_float64'
        return stypy_return_type_541950


    @norecursion
    def test_large(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_large'
        module_type_store = module_type_store.open_function_context('test_large', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpit.test_large.__dict__.__setitem__('stypy_localization', localization)
        TestExpit.test_large.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpit.test_large.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpit.test_large.__dict__.__setitem__('stypy_function_name', 'TestExpit.test_large')
        TestExpit.test_large.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpit.test_large.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpit.test_large.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpit.test_large.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpit.test_large.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpit.test_large.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpit.test_large.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpit.test_large', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_large', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_large(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_541951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        # Getting the type of 'np' (line 75)
        np_541952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'np')
        # Obtaining the member 'float32' of a type (line 75)
        float32_541953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 22), np_541952, 'float32')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), tuple_541951, float32_541953)
        # Adding element type (line 75)
        # Getting the type of 'np' (line 75)
        np_541954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 34), 'np')
        # Obtaining the member 'float64' of a type (line 75)
        float64_541955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 34), np_541954, 'float64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), tuple_541951, float64_541955)
        # Adding element type (line 75)
        # Getting the type of 'np' (line 75)
        np_541956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 46), 'np')
        # Obtaining the member 'longdouble' of a type (line 75)
        longdouble_541957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 46), np_541956, 'longdouble')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 22), tuple_541951, longdouble_541957)
        
        # Testing the type of a for loop iterable (line 75)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 75, 8), tuple_541951)
        # Getting the type of the for loop variable (line 75)
        for_loop_var_541958 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 75, 8), tuple_541951)
        # Assigning a type to the variable 'dtype' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'dtype', for_loop_var_541958)
        # SSA begins for a for statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_541959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        int_541960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 22), tuple_541959, int_541960)
        # Adding element type (line 76)
        int_541961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 22), tuple_541959, int_541961)
        # Adding element type (line 76)
        int_541962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 22), tuple_541959, int_541962)
        # Adding element type (line 76)
        int_541963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 22), tuple_541959, int_541963)
        # Adding element type (line 76)
        int_541964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 22), tuple_541959, int_541964)
        # Adding element type (line 76)
        int_541965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 22), tuple_541959, int_541965)
        
        # Testing the type of a for loop iterable (line 76)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 76, 12), tuple_541959)
        # Getting the type of the for loop variable (line 76)
        for_loop_var_541966 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 76, 12), tuple_541959)
        # Assigning a type to the variable 'n' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'n', for_loop_var_541966)
        # SSA begins for a for statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 77):
        
        # Call to array(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'n' (line 77)
        n_541969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'n', False)
        # Processing the call keyword arguments (line 77)
        # Getting the type of 'dtype' (line 77)
        dtype_541970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'dtype', False)
        keyword_541971 = dtype_541970
        kwargs_541972 = {'dtype': keyword_541971}
        # Getting the type of 'np' (line 77)
        np_541967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'np', False)
        # Obtaining the member 'array' of a type (line 77)
        array_541968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 20), np_541967, 'array')
        # Calling array(args, kwargs) (line 77)
        array_call_result_541973 = invoke(stypy.reporting.localization.Localization(__file__, 77, 20), array_541968, *[n_541969], **kwargs_541972)
        
        # Assigning a type to the variable 'n' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'n', array_call_result_541973)
        
        # Call to assert_allclose(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Call to expit(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'n' (line 78)
        n_541976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 38), 'n', False)
        # Processing the call keyword arguments (line 78)
        kwargs_541977 = {}
        # Getting the type of 'expit' (line 78)
        expit_541975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 32), 'expit', False)
        # Calling expit(args, kwargs) (line 78)
        expit_call_result_541978 = invoke(stypy.reporting.localization.Localization(__file__, 78, 32), expit_541975, *[n_541976], **kwargs_541977)
        
        float_541979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 42), 'float')
        # Processing the call keyword arguments (line 78)
        float_541980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 52), 'float')
        keyword_541981 = float_541980
        kwargs_541982 = {'atol': keyword_541981}
        # Getting the type of 'assert_allclose' (line 78)
        assert_allclose_541974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 78)
        assert_allclose_call_result_541983 = invoke(stypy.reporting.localization.Localization(__file__, 78, 16), assert_allclose_541974, *[expit_call_result_541978, float_541979], **kwargs_541982)
        
        
        # Call to assert_allclose(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Call to expit(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Getting the type of 'n' (line 79)
        n_541986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 39), 'n', False)
        # Applying the 'usub' unary operator (line 79)
        result___neg___541987 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 38), 'usub', n_541986)
        
        # Processing the call keyword arguments (line 79)
        kwargs_541988 = {}
        # Getting the type of 'expit' (line 79)
        expit_541985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 32), 'expit', False)
        # Calling expit(args, kwargs) (line 79)
        expit_call_result_541989 = invoke(stypy.reporting.localization.Localization(__file__, 79, 32), expit_541985, *[result___neg___541987], **kwargs_541988)
        
        float_541990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 43), 'float')
        # Processing the call keyword arguments (line 79)
        float_541991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 53), 'float')
        keyword_541992 = float_541991
        kwargs_541993 = {'atol': keyword_541992}
        # Getting the type of 'assert_allclose' (line 79)
        assert_allclose_541984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 79)
        assert_allclose_call_result_541994 = invoke(stypy.reporting.localization.Localization(__file__, 79, 16), assert_allclose_541984, *[expit_call_result_541989, float_541990], **kwargs_541993)
        
        
        # Call to assert_equal(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Call to expit(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'n' (line 80)
        n_541997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'n', False)
        # Processing the call keyword arguments (line 80)
        kwargs_541998 = {}
        # Getting the type of 'expit' (line 80)
        expit_541996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'expit', False)
        # Calling expit(args, kwargs) (line 80)
        expit_call_result_541999 = invoke(stypy.reporting.localization.Localization(__file__, 80, 29), expit_541996, *[n_541997], **kwargs_541998)
        
        # Obtaining the member 'dtype' of a type (line 80)
        dtype_542000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 29), expit_call_result_541999, 'dtype')
        # Getting the type of 'dtype' (line 80)
        dtype_542001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 45), 'dtype', False)
        # Processing the call keyword arguments (line 80)
        kwargs_542002 = {}
        # Getting the type of 'assert_equal' (line 80)
        assert_equal_541995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 80)
        assert_equal_call_result_542003 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), assert_equal_541995, *[dtype_542000, dtype_542001], **kwargs_542002)
        
        
        # Call to assert_equal(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Call to expit(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Getting the type of 'n' (line 81)
        n_542006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 36), 'n', False)
        # Applying the 'usub' unary operator (line 81)
        result___neg___542007 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 35), 'usub', n_542006)
        
        # Processing the call keyword arguments (line 81)
        kwargs_542008 = {}
        # Getting the type of 'expit' (line 81)
        expit_542005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 29), 'expit', False)
        # Calling expit(args, kwargs) (line 81)
        expit_call_result_542009 = invoke(stypy.reporting.localization.Localization(__file__, 81, 29), expit_542005, *[result___neg___542007], **kwargs_542008)
        
        # Obtaining the member 'dtype' of a type (line 81)
        dtype_542010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 29), expit_call_result_542009, 'dtype')
        # Getting the type of 'dtype' (line 81)
        dtype_542011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 46), 'dtype', False)
        # Processing the call keyword arguments (line 81)
        kwargs_542012 = {}
        # Getting the type of 'assert_equal' (line 81)
        assert_equal_542004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 81)
        assert_equal_call_result_542013 = invoke(stypy.reporting.localization.Localization(__file__, 81, 16), assert_equal_542004, *[dtype_542010, dtype_542011], **kwargs_542012)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_large(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_large' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_542014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_542014)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_large'
        return stypy_return_type_542014


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 50, 0, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpit.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestExpit' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'TestExpit', TestExpit)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
