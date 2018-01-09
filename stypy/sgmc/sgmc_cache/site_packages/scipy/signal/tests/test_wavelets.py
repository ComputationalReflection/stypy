
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_equal, \
5:     assert_array_equal, assert_array_almost_equal, assert_array_less, assert_
6: from scipy._lib.six import xrange
7: 
8: from scipy.signal import wavelets
9: 
10: 
11: class TestWavelets(object):
12:     def test_qmf(self):
13:         assert_array_equal(wavelets.qmf([1, 1]), [1, -1])
14: 
15:     def test_daub(self):
16:         for i in xrange(1, 15):
17:             assert_equal(len(wavelets.daub(i)), i * 2)
18: 
19:     def test_cascade(self):
20:         for J in xrange(1, 7):
21:             for i in xrange(1, 5):
22:                 lpcoef = wavelets.daub(i)
23:                 k = len(lpcoef)
24:                 x, phi, psi = wavelets.cascade(lpcoef, J)
25:                 assert_(len(x) == len(phi) == len(psi))
26:                 assert_equal(len(x), (k - 1) * 2 ** J)
27: 
28:     def test_morlet(self):
29:         x = wavelets.morlet(50, 4.1, complete=True)
30:         y = wavelets.morlet(50, 4.1, complete=False)
31:         # Test if complete and incomplete wavelet have same lengths:
32:         assert_equal(len(x), len(y))
33:         # Test if complete wavelet is less than incomplete wavelet:
34:         assert_array_less(x, y)
35: 
36:         x = wavelets.morlet(10, 50, complete=False)
37:         y = wavelets.morlet(10, 50, complete=True)
38:         # For large widths complete and incomplete wavelets should be
39:         # identical within numerical precision:
40:         assert_equal(x, y)
41: 
42:         # miscellaneous tests:
43:         x = np.array([1.73752399e-09 + 9.84327394e-25j,
44:                       6.49471756e-01 + 0.00000000e+00j,
45:                       1.73752399e-09 - 9.84327394e-25j])
46:         y = wavelets.morlet(3, w=2, complete=True)
47:         assert_array_almost_equal(x, y)
48: 
49:         x = np.array([2.00947715e-09 + 9.84327394e-25j,
50:                       7.51125544e-01 + 0.00000000e+00j,
51:                       2.00947715e-09 - 9.84327394e-25j])
52:         y = wavelets.morlet(3, w=2, complete=False)
53:         assert_array_almost_equal(x, y, decimal=2)
54: 
55:         x = wavelets.morlet(10000, s=4, complete=True)
56:         y = wavelets.morlet(20000, s=8, complete=True)[5000:15000]
57:         assert_array_almost_equal(x, y, decimal=2)
58: 
59:         x = wavelets.morlet(10000, s=4, complete=False)
60:         assert_array_almost_equal(y, x, decimal=2)
61:         y = wavelets.morlet(20000, s=8, complete=False)[5000:15000]
62:         assert_array_almost_equal(x, y, decimal=2)
63: 
64:         x = wavelets.morlet(10000, w=3, s=5, complete=True)
65:         y = wavelets.morlet(20000, w=3, s=10, complete=True)[5000:15000]
66:         assert_array_almost_equal(x, y, decimal=2)
67: 
68:         x = wavelets.morlet(10000, w=3, s=5, complete=False)
69:         assert_array_almost_equal(y, x, decimal=2)
70:         y = wavelets.morlet(20000, w=3, s=10, complete=False)[5000:15000]
71:         assert_array_almost_equal(x, y, decimal=2)
72: 
73:         x = wavelets.morlet(10000, w=7, s=10, complete=True)
74:         y = wavelets.morlet(20000, w=7, s=20, complete=True)[5000:15000]
75:         assert_array_almost_equal(x, y, decimal=2)
76: 
77:         x = wavelets.morlet(10000, w=7, s=10, complete=False)
78:         assert_array_almost_equal(x, y, decimal=2)
79:         y = wavelets.morlet(20000, w=7, s=20, complete=False)[5000:15000]
80:         assert_array_almost_equal(x, y, decimal=2)
81: 
82:     def test_ricker(self):
83:         w = wavelets.ricker(1.0, 1)
84:         expected = 2 / (np.sqrt(3 * 1.0) * (np.pi ** 0.25))
85:         assert_array_equal(w, expected)
86: 
87:         lengths = [5, 11, 15, 51, 101]
88:         for length in lengths:
89:             w = wavelets.ricker(length, 1.0)
90:             assert_(len(w) == length)
91:             max_loc = np.argmax(w)
92:             assert_(max_loc == (length // 2))
93: 
94:         points = 100
95:         w = wavelets.ricker(points, 2.0)
96:         half_vec = np.arange(0, points // 2)
97:         #Wavelet should be symmetric
98:         assert_array_almost_equal(w[half_vec], w[-(half_vec + 1)])
99: 
100:         #Check zeros
101:         aas = [5, 10, 15, 20, 30]
102:         points = 99
103:         for a in aas:
104:             w = wavelets.ricker(points, a)
105:             vec = np.arange(0, points) - (points - 1.0) / 2
106:             exp_zero1 = np.argmin(np.abs(vec - a))
107:             exp_zero2 = np.argmin(np.abs(vec + a))
108:             assert_array_almost_equal(w[exp_zero1], 0)
109:             assert_array_almost_equal(w[exp_zero2], 0)
110: 
111:     def test_cwt(self):
112:         widths = [1.0]
113:         delta_wavelet = lambda s, t: np.array([1])
114:         len_data = 100
115:         test_data = np.sin(np.pi * np.arange(0, len_data) / 10.0)
116: 
117:         #Test delta function input gives same data as output
118:         cwt_dat = wavelets.cwt(test_data, delta_wavelet, widths)
119:         assert_(cwt_dat.shape == (len(widths), len_data))
120:         assert_array_almost_equal(test_data, cwt_dat.flatten())
121: 
122:         #Check proper shape on output
123:         widths = [1, 3, 4, 5, 10]
124:         cwt_dat = wavelets.cwt(test_data, wavelets.ricker, widths)
125:         assert_(cwt_dat.shape == (len(widths), len_data))
126: 
127:         widths = [len_data * 10]
128:         #Note: this wavelet isn't defined quite right, but is fine for this test
129:         flat_wavelet = lambda l, w: np.ones(w) / w
130:         cwt_dat = wavelets.cwt(test_data, flat_wavelet, widths)
131:         assert_array_almost_equal(cwt_dat, np.mean(test_data))
132: 
133: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_353124 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_353124) is not StypyTypeError):

    if (import_353124 != 'pyd_module'):
        __import__(import_353124)
        sys_modules_353125 = sys.modules[import_353124]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_353125.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_353124)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal, assert_array_less, assert_' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_353126 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_353126) is not StypyTypeError):

    if (import_353126 != 'pyd_module'):
        __import__(import_353126)
        sys_modules_353127 = sys.modules[import_353126]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_353127.module_type_store, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_array_almost_equal', 'assert_array_less', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_353127, sys_modules_353127.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal, assert_array_less, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_array_almost_equal', 'assert_array_less', 'assert_'], [assert_equal, assert_array_equal, assert_array_almost_equal, assert_array_less, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_353126)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy._lib.six import xrange' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_353128 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib.six')

if (type(import_353128) is not StypyTypeError):

    if (import_353128 != 'pyd_module'):
        __import__(import_353128)
        sys_modules_353129 = sys.modules[import_353128]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib.six', sys_modules_353129.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_353129, sys_modules_353129.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib.six', import_353128)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.signal import wavelets' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_353130 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal')

if (type(import_353130) is not StypyTypeError):

    if (import_353130 != 'pyd_module'):
        __import__(import_353130)
        sys_modules_353131 = sys.modules[import_353130]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal', sys_modules_353131.module_type_store, module_type_store, ['wavelets'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_353131, sys_modules_353131.module_type_store, module_type_store)
    else:
        from scipy.signal import wavelets

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal', None, module_type_store, ['wavelets'], [wavelets])

else:
    # Assigning a type to the variable 'scipy.signal' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal', import_353130)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

# Declaration of the 'TestWavelets' class

class TestWavelets(object, ):

    @norecursion
    def test_qmf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_qmf'
        module_type_store = module_type_store.open_function_context('test_qmf', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestWavelets.test_qmf.__dict__.__setitem__('stypy_localization', localization)
        TestWavelets.test_qmf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestWavelets.test_qmf.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestWavelets.test_qmf.__dict__.__setitem__('stypy_function_name', 'TestWavelets.test_qmf')
        TestWavelets.test_qmf.__dict__.__setitem__('stypy_param_names_list', [])
        TestWavelets.test_qmf.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestWavelets.test_qmf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestWavelets.test_qmf.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestWavelets.test_qmf.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestWavelets.test_qmf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestWavelets.test_qmf.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWavelets.test_qmf', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_qmf', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_qmf(...)' code ##################

        
        # Call to assert_array_equal(...): (line 13)
        # Processing the call arguments (line 13)
        
        # Call to qmf(...): (line 13)
        # Processing the call arguments (line 13)
        
        # Obtaining an instance of the builtin type 'list' (line 13)
        list_353135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 13)
        # Adding element type (line 13)
        int_353136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 40), list_353135, int_353136)
        # Adding element type (line 13)
        int_353137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 40), list_353135, int_353137)
        
        # Processing the call keyword arguments (line 13)
        kwargs_353138 = {}
        # Getting the type of 'wavelets' (line 13)
        wavelets_353133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 27), 'wavelets', False)
        # Obtaining the member 'qmf' of a type (line 13)
        qmf_353134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 27), wavelets_353133, 'qmf')
        # Calling qmf(args, kwargs) (line 13)
        qmf_call_result_353139 = invoke(stypy.reporting.localization.Localization(__file__, 13, 27), qmf_353134, *[list_353135], **kwargs_353138)
        
        
        # Obtaining an instance of the builtin type 'list' (line 13)
        list_353140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 13)
        # Adding element type (line 13)
        int_353141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 49), list_353140, int_353141)
        # Adding element type (line 13)
        int_353142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 49), list_353140, int_353142)
        
        # Processing the call keyword arguments (line 13)
        kwargs_353143 = {}
        # Getting the type of 'assert_array_equal' (line 13)
        assert_array_equal_353132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 13)
        assert_array_equal_call_result_353144 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), assert_array_equal_353132, *[qmf_call_result_353139, list_353140], **kwargs_353143)
        
        
        # ################# End of 'test_qmf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_qmf' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_353145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_353145)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_qmf'
        return stypy_return_type_353145


    @norecursion
    def test_daub(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_daub'
        module_type_store = module_type_store.open_function_context('test_daub', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestWavelets.test_daub.__dict__.__setitem__('stypy_localization', localization)
        TestWavelets.test_daub.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestWavelets.test_daub.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestWavelets.test_daub.__dict__.__setitem__('stypy_function_name', 'TestWavelets.test_daub')
        TestWavelets.test_daub.__dict__.__setitem__('stypy_param_names_list', [])
        TestWavelets.test_daub.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestWavelets.test_daub.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestWavelets.test_daub.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestWavelets.test_daub.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestWavelets.test_daub.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestWavelets.test_daub.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWavelets.test_daub', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_daub', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_daub(...)' code ##################

        
        
        # Call to xrange(...): (line 16)
        # Processing the call arguments (line 16)
        int_353147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'int')
        int_353148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 27), 'int')
        # Processing the call keyword arguments (line 16)
        kwargs_353149 = {}
        # Getting the type of 'xrange' (line 16)
        xrange_353146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 16)
        xrange_call_result_353150 = invoke(stypy.reporting.localization.Localization(__file__, 16, 17), xrange_353146, *[int_353147, int_353148], **kwargs_353149)
        
        # Testing the type of a for loop iterable (line 16)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 16, 8), xrange_call_result_353150)
        # Getting the type of the for loop variable (line 16)
        for_loop_var_353151 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 16, 8), xrange_call_result_353150)
        # Assigning a type to the variable 'i' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'i', for_loop_var_353151)
        # SSA begins for a for statement (line 16)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_equal(...): (line 17)
        # Processing the call arguments (line 17)
        
        # Call to len(...): (line 17)
        # Processing the call arguments (line 17)
        
        # Call to daub(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'i' (line 17)
        i_353156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 43), 'i', False)
        # Processing the call keyword arguments (line 17)
        kwargs_353157 = {}
        # Getting the type of 'wavelets' (line 17)
        wavelets_353154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 29), 'wavelets', False)
        # Obtaining the member 'daub' of a type (line 17)
        daub_353155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 29), wavelets_353154, 'daub')
        # Calling daub(args, kwargs) (line 17)
        daub_call_result_353158 = invoke(stypy.reporting.localization.Localization(__file__, 17, 29), daub_353155, *[i_353156], **kwargs_353157)
        
        # Processing the call keyword arguments (line 17)
        kwargs_353159 = {}
        # Getting the type of 'len' (line 17)
        len_353153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), 'len', False)
        # Calling len(args, kwargs) (line 17)
        len_call_result_353160 = invoke(stypy.reporting.localization.Localization(__file__, 17, 25), len_353153, *[daub_call_result_353158], **kwargs_353159)
        
        # Getting the type of 'i' (line 17)
        i_353161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 48), 'i', False)
        int_353162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 52), 'int')
        # Applying the binary operator '*' (line 17)
        result_mul_353163 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 48), '*', i_353161, int_353162)
        
        # Processing the call keyword arguments (line 17)
        kwargs_353164 = {}
        # Getting the type of 'assert_equal' (line 17)
        assert_equal_353152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 17)
        assert_equal_call_result_353165 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), assert_equal_353152, *[len_call_result_353160, result_mul_353163], **kwargs_353164)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_daub(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_daub' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_353166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_353166)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_daub'
        return stypy_return_type_353166


    @norecursion
    def test_cascade(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cascade'
        module_type_store = module_type_store.open_function_context('test_cascade', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestWavelets.test_cascade.__dict__.__setitem__('stypy_localization', localization)
        TestWavelets.test_cascade.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestWavelets.test_cascade.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestWavelets.test_cascade.__dict__.__setitem__('stypy_function_name', 'TestWavelets.test_cascade')
        TestWavelets.test_cascade.__dict__.__setitem__('stypy_param_names_list', [])
        TestWavelets.test_cascade.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestWavelets.test_cascade.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestWavelets.test_cascade.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestWavelets.test_cascade.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestWavelets.test_cascade.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestWavelets.test_cascade.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWavelets.test_cascade', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cascade', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cascade(...)' code ##################

        
        
        # Call to xrange(...): (line 20)
        # Processing the call arguments (line 20)
        int_353168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'int')
        int_353169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 27), 'int')
        # Processing the call keyword arguments (line 20)
        kwargs_353170 = {}
        # Getting the type of 'xrange' (line 20)
        xrange_353167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 20)
        xrange_call_result_353171 = invoke(stypy.reporting.localization.Localization(__file__, 20, 17), xrange_353167, *[int_353168, int_353169], **kwargs_353170)
        
        # Testing the type of a for loop iterable (line 20)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 20, 8), xrange_call_result_353171)
        # Getting the type of the for loop variable (line 20)
        for_loop_var_353172 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 20, 8), xrange_call_result_353171)
        # Assigning a type to the variable 'J' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'J', for_loop_var_353172)
        # SSA begins for a for statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to xrange(...): (line 21)
        # Processing the call arguments (line 21)
        int_353174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'int')
        int_353175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 31), 'int')
        # Processing the call keyword arguments (line 21)
        kwargs_353176 = {}
        # Getting the type of 'xrange' (line 21)
        xrange_353173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'xrange', False)
        # Calling xrange(args, kwargs) (line 21)
        xrange_call_result_353177 = invoke(stypy.reporting.localization.Localization(__file__, 21, 21), xrange_353173, *[int_353174, int_353175], **kwargs_353176)
        
        # Testing the type of a for loop iterable (line 21)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 21, 12), xrange_call_result_353177)
        # Getting the type of the for loop variable (line 21)
        for_loop_var_353178 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 21, 12), xrange_call_result_353177)
        # Assigning a type to the variable 'i' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'i', for_loop_var_353178)
        # SSA begins for a for statement (line 21)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 22):
        
        # Assigning a Call to a Name (line 22):
        
        # Call to daub(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'i' (line 22)
        i_353181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 39), 'i', False)
        # Processing the call keyword arguments (line 22)
        kwargs_353182 = {}
        # Getting the type of 'wavelets' (line 22)
        wavelets_353179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'wavelets', False)
        # Obtaining the member 'daub' of a type (line 22)
        daub_353180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 25), wavelets_353179, 'daub')
        # Calling daub(args, kwargs) (line 22)
        daub_call_result_353183 = invoke(stypy.reporting.localization.Localization(__file__, 22, 25), daub_353180, *[i_353181], **kwargs_353182)
        
        # Assigning a type to the variable 'lpcoef' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'lpcoef', daub_call_result_353183)
        
        # Assigning a Call to a Name (line 23):
        
        # Assigning a Call to a Name (line 23):
        
        # Call to len(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'lpcoef' (line 23)
        lpcoef_353185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'lpcoef', False)
        # Processing the call keyword arguments (line 23)
        kwargs_353186 = {}
        # Getting the type of 'len' (line 23)
        len_353184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'len', False)
        # Calling len(args, kwargs) (line 23)
        len_call_result_353187 = invoke(stypy.reporting.localization.Localization(__file__, 23, 20), len_353184, *[lpcoef_353185], **kwargs_353186)
        
        # Assigning a type to the variable 'k' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'k', len_call_result_353187)
        
        # Assigning a Call to a Tuple (line 24):
        
        # Assigning a Subscript to a Name (line 24):
        
        # Obtaining the type of the subscript
        int_353188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'int')
        
        # Call to cascade(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'lpcoef' (line 24)
        lpcoef_353191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 47), 'lpcoef', False)
        # Getting the type of 'J' (line 24)
        J_353192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 55), 'J', False)
        # Processing the call keyword arguments (line 24)
        kwargs_353193 = {}
        # Getting the type of 'wavelets' (line 24)
        wavelets_353189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 30), 'wavelets', False)
        # Obtaining the member 'cascade' of a type (line 24)
        cascade_353190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 30), wavelets_353189, 'cascade')
        # Calling cascade(args, kwargs) (line 24)
        cascade_call_result_353194 = invoke(stypy.reporting.localization.Localization(__file__, 24, 30), cascade_353190, *[lpcoef_353191, J_353192], **kwargs_353193)
        
        # Obtaining the member '__getitem__' of a type (line 24)
        getitem___353195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), cascade_call_result_353194, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 24)
        subscript_call_result_353196 = invoke(stypy.reporting.localization.Localization(__file__, 24, 16), getitem___353195, int_353188)
        
        # Assigning a type to the variable 'tuple_var_assignment_353121' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'tuple_var_assignment_353121', subscript_call_result_353196)
        
        # Assigning a Subscript to a Name (line 24):
        
        # Obtaining the type of the subscript
        int_353197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'int')
        
        # Call to cascade(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'lpcoef' (line 24)
        lpcoef_353200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 47), 'lpcoef', False)
        # Getting the type of 'J' (line 24)
        J_353201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 55), 'J', False)
        # Processing the call keyword arguments (line 24)
        kwargs_353202 = {}
        # Getting the type of 'wavelets' (line 24)
        wavelets_353198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 30), 'wavelets', False)
        # Obtaining the member 'cascade' of a type (line 24)
        cascade_353199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 30), wavelets_353198, 'cascade')
        # Calling cascade(args, kwargs) (line 24)
        cascade_call_result_353203 = invoke(stypy.reporting.localization.Localization(__file__, 24, 30), cascade_353199, *[lpcoef_353200, J_353201], **kwargs_353202)
        
        # Obtaining the member '__getitem__' of a type (line 24)
        getitem___353204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), cascade_call_result_353203, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 24)
        subscript_call_result_353205 = invoke(stypy.reporting.localization.Localization(__file__, 24, 16), getitem___353204, int_353197)
        
        # Assigning a type to the variable 'tuple_var_assignment_353122' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'tuple_var_assignment_353122', subscript_call_result_353205)
        
        # Assigning a Subscript to a Name (line 24):
        
        # Obtaining the type of the subscript
        int_353206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'int')
        
        # Call to cascade(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'lpcoef' (line 24)
        lpcoef_353209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 47), 'lpcoef', False)
        # Getting the type of 'J' (line 24)
        J_353210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 55), 'J', False)
        # Processing the call keyword arguments (line 24)
        kwargs_353211 = {}
        # Getting the type of 'wavelets' (line 24)
        wavelets_353207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 30), 'wavelets', False)
        # Obtaining the member 'cascade' of a type (line 24)
        cascade_353208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 30), wavelets_353207, 'cascade')
        # Calling cascade(args, kwargs) (line 24)
        cascade_call_result_353212 = invoke(stypy.reporting.localization.Localization(__file__, 24, 30), cascade_353208, *[lpcoef_353209, J_353210], **kwargs_353211)
        
        # Obtaining the member '__getitem__' of a type (line 24)
        getitem___353213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), cascade_call_result_353212, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 24)
        subscript_call_result_353214 = invoke(stypy.reporting.localization.Localization(__file__, 24, 16), getitem___353213, int_353206)
        
        # Assigning a type to the variable 'tuple_var_assignment_353123' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'tuple_var_assignment_353123', subscript_call_result_353214)
        
        # Assigning a Name to a Name (line 24):
        # Getting the type of 'tuple_var_assignment_353121' (line 24)
        tuple_var_assignment_353121_353215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'tuple_var_assignment_353121')
        # Assigning a type to the variable 'x' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'x', tuple_var_assignment_353121_353215)
        
        # Assigning a Name to a Name (line 24):
        # Getting the type of 'tuple_var_assignment_353122' (line 24)
        tuple_var_assignment_353122_353216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'tuple_var_assignment_353122')
        # Assigning a type to the variable 'phi' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'phi', tuple_var_assignment_353122_353216)
        
        # Assigning a Name to a Name (line 24):
        # Getting the type of 'tuple_var_assignment_353123' (line 24)
        tuple_var_assignment_353123_353217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'tuple_var_assignment_353123')
        # Assigning a type to the variable 'psi' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'psi', tuple_var_assignment_353123_353217)
        
        # Call to assert_(...): (line 25)
        # Processing the call arguments (line 25)
        
        
        # Call to len(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'x' (line 25)
        x_353220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 28), 'x', False)
        # Processing the call keyword arguments (line 25)
        kwargs_353221 = {}
        # Getting the type of 'len' (line 25)
        len_353219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'len', False)
        # Calling len(args, kwargs) (line 25)
        len_call_result_353222 = invoke(stypy.reporting.localization.Localization(__file__, 25, 24), len_353219, *[x_353220], **kwargs_353221)
        
        
        # Call to len(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'phi' (line 25)
        phi_353224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 38), 'phi', False)
        # Processing the call keyword arguments (line 25)
        kwargs_353225 = {}
        # Getting the type of 'len' (line 25)
        len_353223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 34), 'len', False)
        # Calling len(args, kwargs) (line 25)
        len_call_result_353226 = invoke(stypy.reporting.localization.Localization(__file__, 25, 34), len_353223, *[phi_353224], **kwargs_353225)
        
        # Applying the binary operator '==' (line 25)
        result_eq_353227 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 24), '==', len_call_result_353222, len_call_result_353226)
        
        # Call to len(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'psi' (line 25)
        psi_353229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 50), 'psi', False)
        # Processing the call keyword arguments (line 25)
        kwargs_353230 = {}
        # Getting the type of 'len' (line 25)
        len_353228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 46), 'len', False)
        # Calling len(args, kwargs) (line 25)
        len_call_result_353231 = invoke(stypy.reporting.localization.Localization(__file__, 25, 46), len_353228, *[psi_353229], **kwargs_353230)
        
        # Applying the binary operator '==' (line 25)
        result_eq_353232 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 24), '==', len_call_result_353226, len_call_result_353231)
        # Applying the binary operator '&' (line 25)
        result_and__353233 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 24), '&', result_eq_353227, result_eq_353232)
        
        # Processing the call keyword arguments (line 25)
        kwargs_353234 = {}
        # Getting the type of 'assert_' (line 25)
        assert__353218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'assert_', False)
        # Calling assert_(args, kwargs) (line 25)
        assert__call_result_353235 = invoke(stypy.reporting.localization.Localization(__file__, 25, 16), assert__353218, *[result_and__353233], **kwargs_353234)
        
        
        # Call to assert_equal(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Call to len(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'x' (line 26)
        x_353238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 33), 'x', False)
        # Processing the call keyword arguments (line 26)
        kwargs_353239 = {}
        # Getting the type of 'len' (line 26)
        len_353237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'len', False)
        # Calling len(args, kwargs) (line 26)
        len_call_result_353240 = invoke(stypy.reporting.localization.Localization(__file__, 26, 29), len_353237, *[x_353238], **kwargs_353239)
        
        # Getting the type of 'k' (line 26)
        k_353241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 38), 'k', False)
        int_353242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 42), 'int')
        # Applying the binary operator '-' (line 26)
        result_sub_353243 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 38), '-', k_353241, int_353242)
        
        int_353244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 47), 'int')
        # Getting the type of 'J' (line 26)
        J_353245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 52), 'J', False)
        # Applying the binary operator '**' (line 26)
        result_pow_353246 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 47), '**', int_353244, J_353245)
        
        # Applying the binary operator '*' (line 26)
        result_mul_353247 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 37), '*', result_sub_353243, result_pow_353246)
        
        # Processing the call keyword arguments (line 26)
        kwargs_353248 = {}
        # Getting the type of 'assert_equal' (line 26)
        assert_equal_353236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 26)
        assert_equal_call_result_353249 = invoke(stypy.reporting.localization.Localization(__file__, 26, 16), assert_equal_353236, *[len_call_result_353240, result_mul_353247], **kwargs_353248)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_cascade(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cascade' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_353250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_353250)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cascade'
        return stypy_return_type_353250


    @norecursion
    def test_morlet(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_morlet'
        module_type_store = module_type_store.open_function_context('test_morlet', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestWavelets.test_morlet.__dict__.__setitem__('stypy_localization', localization)
        TestWavelets.test_morlet.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestWavelets.test_morlet.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestWavelets.test_morlet.__dict__.__setitem__('stypy_function_name', 'TestWavelets.test_morlet')
        TestWavelets.test_morlet.__dict__.__setitem__('stypy_param_names_list', [])
        TestWavelets.test_morlet.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestWavelets.test_morlet.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestWavelets.test_morlet.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestWavelets.test_morlet.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestWavelets.test_morlet.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestWavelets.test_morlet.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWavelets.test_morlet', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_morlet', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_morlet(...)' code ##################

        
        # Assigning a Call to a Name (line 29):
        
        # Assigning a Call to a Name (line 29):
        
        # Call to morlet(...): (line 29)
        # Processing the call arguments (line 29)
        int_353253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'int')
        float_353254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 32), 'float')
        # Processing the call keyword arguments (line 29)
        # Getting the type of 'True' (line 29)
        True_353255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 46), 'True', False)
        keyword_353256 = True_353255
        kwargs_353257 = {'complete': keyword_353256}
        # Getting the type of 'wavelets' (line 29)
        wavelets_353251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 29)
        morlet_353252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), wavelets_353251, 'morlet')
        # Calling morlet(args, kwargs) (line 29)
        morlet_call_result_353258 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), morlet_353252, *[int_353253, float_353254], **kwargs_353257)
        
        # Assigning a type to the variable 'x' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'x', morlet_call_result_353258)
        
        # Assigning a Call to a Name (line 30):
        
        # Assigning a Call to a Name (line 30):
        
        # Call to morlet(...): (line 30)
        # Processing the call arguments (line 30)
        int_353261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 28), 'int')
        float_353262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 32), 'float')
        # Processing the call keyword arguments (line 30)
        # Getting the type of 'False' (line 30)
        False_353263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 46), 'False', False)
        keyword_353264 = False_353263
        kwargs_353265 = {'complete': keyword_353264}
        # Getting the type of 'wavelets' (line 30)
        wavelets_353259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 30)
        morlet_353260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 12), wavelets_353259, 'morlet')
        # Calling morlet(args, kwargs) (line 30)
        morlet_call_result_353266 = invoke(stypy.reporting.localization.Localization(__file__, 30, 12), morlet_353260, *[int_353261, float_353262], **kwargs_353265)
        
        # Assigning a type to the variable 'y' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'y', morlet_call_result_353266)
        
        # Call to assert_equal(...): (line 32)
        # Processing the call arguments (line 32)
        
        # Call to len(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'x' (line 32)
        x_353269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'x', False)
        # Processing the call keyword arguments (line 32)
        kwargs_353270 = {}
        # Getting the type of 'len' (line 32)
        len_353268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 21), 'len', False)
        # Calling len(args, kwargs) (line 32)
        len_call_result_353271 = invoke(stypy.reporting.localization.Localization(__file__, 32, 21), len_353268, *[x_353269], **kwargs_353270)
        
        
        # Call to len(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'y' (line 32)
        y_353273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'y', False)
        # Processing the call keyword arguments (line 32)
        kwargs_353274 = {}
        # Getting the type of 'len' (line 32)
        len_353272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 29), 'len', False)
        # Calling len(args, kwargs) (line 32)
        len_call_result_353275 = invoke(stypy.reporting.localization.Localization(__file__, 32, 29), len_353272, *[y_353273], **kwargs_353274)
        
        # Processing the call keyword arguments (line 32)
        kwargs_353276 = {}
        # Getting the type of 'assert_equal' (line 32)
        assert_equal_353267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 32)
        assert_equal_call_result_353277 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), assert_equal_353267, *[len_call_result_353271, len_call_result_353275], **kwargs_353276)
        
        
        # Call to assert_array_less(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'x' (line 34)
        x_353279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'x', False)
        # Getting the type of 'y' (line 34)
        y_353280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 29), 'y', False)
        # Processing the call keyword arguments (line 34)
        kwargs_353281 = {}
        # Getting the type of 'assert_array_less' (line 34)
        assert_array_less_353278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'assert_array_less', False)
        # Calling assert_array_less(args, kwargs) (line 34)
        assert_array_less_call_result_353282 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assert_array_less_353278, *[x_353279, y_353280], **kwargs_353281)
        
        
        # Assigning a Call to a Name (line 36):
        
        # Assigning a Call to a Name (line 36):
        
        # Call to morlet(...): (line 36)
        # Processing the call arguments (line 36)
        int_353285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 28), 'int')
        int_353286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 32), 'int')
        # Processing the call keyword arguments (line 36)
        # Getting the type of 'False' (line 36)
        False_353287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 45), 'False', False)
        keyword_353288 = False_353287
        kwargs_353289 = {'complete': keyword_353288}
        # Getting the type of 'wavelets' (line 36)
        wavelets_353283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 36)
        morlet_353284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), wavelets_353283, 'morlet')
        # Calling morlet(args, kwargs) (line 36)
        morlet_call_result_353290 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), morlet_353284, *[int_353285, int_353286], **kwargs_353289)
        
        # Assigning a type to the variable 'x' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'x', morlet_call_result_353290)
        
        # Assigning a Call to a Name (line 37):
        
        # Assigning a Call to a Name (line 37):
        
        # Call to morlet(...): (line 37)
        # Processing the call arguments (line 37)
        int_353293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 28), 'int')
        int_353294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 32), 'int')
        # Processing the call keyword arguments (line 37)
        # Getting the type of 'True' (line 37)
        True_353295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 45), 'True', False)
        keyword_353296 = True_353295
        kwargs_353297 = {'complete': keyword_353296}
        # Getting the type of 'wavelets' (line 37)
        wavelets_353291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 37)
        morlet_353292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), wavelets_353291, 'morlet')
        # Calling morlet(args, kwargs) (line 37)
        morlet_call_result_353298 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), morlet_353292, *[int_353293, int_353294], **kwargs_353297)
        
        # Assigning a type to the variable 'y' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'y', morlet_call_result_353298)
        
        # Call to assert_equal(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'x' (line 40)
        x_353300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 21), 'x', False)
        # Getting the type of 'y' (line 40)
        y_353301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'y', False)
        # Processing the call keyword arguments (line 40)
        kwargs_353302 = {}
        # Getting the type of 'assert_equal' (line 40)
        assert_equal_353299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 40)
        assert_equal_call_result_353303 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), assert_equal_353299, *[x_353300, y_353301], **kwargs_353302)
        
        
        # Assigning a Call to a Name (line 43):
        
        # Assigning a Call to a Name (line 43):
        
        # Call to array(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_353306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        float_353307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 22), 'float')
        complex_353308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 39), 'complex')
        # Applying the binary operator '+' (line 43)
        result_add_353309 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 22), '+', float_353307, complex_353308)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), list_353306, result_add_353309)
        # Adding element type (line 43)
        float_353310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 22), 'float')
        complex_353311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 39), 'complex')
        # Applying the binary operator '+' (line 44)
        result_add_353312 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 22), '+', float_353310, complex_353311)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), list_353306, result_add_353312)
        # Adding element type (line 43)
        float_353313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 22), 'float')
        complex_353314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 39), 'complex')
        # Applying the binary operator '-' (line 45)
        result_sub_353315 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 22), '-', float_353313, complex_353314)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), list_353306, result_sub_353315)
        
        # Processing the call keyword arguments (line 43)
        kwargs_353316 = {}
        # Getting the type of 'np' (line 43)
        np_353304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 43)
        array_353305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), np_353304, 'array')
        # Calling array(args, kwargs) (line 43)
        array_call_result_353317 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), array_353305, *[list_353306], **kwargs_353316)
        
        # Assigning a type to the variable 'x' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'x', array_call_result_353317)
        
        # Assigning a Call to a Name (line 46):
        
        # Assigning a Call to a Name (line 46):
        
        # Call to morlet(...): (line 46)
        # Processing the call arguments (line 46)
        int_353320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 28), 'int')
        # Processing the call keyword arguments (line 46)
        int_353321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 33), 'int')
        keyword_353322 = int_353321
        # Getting the type of 'True' (line 46)
        True_353323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 45), 'True', False)
        keyword_353324 = True_353323
        kwargs_353325 = {'complete': keyword_353324, 'w': keyword_353322}
        # Getting the type of 'wavelets' (line 46)
        wavelets_353318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 46)
        morlet_353319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), wavelets_353318, 'morlet')
        # Calling morlet(args, kwargs) (line 46)
        morlet_call_result_353326 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), morlet_353319, *[int_353320], **kwargs_353325)
        
        # Assigning a type to the variable 'y' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'y', morlet_call_result_353326)
        
        # Call to assert_array_almost_equal(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'x' (line 47)
        x_353328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 34), 'x', False)
        # Getting the type of 'y' (line 47)
        y_353329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'y', False)
        # Processing the call keyword arguments (line 47)
        kwargs_353330 = {}
        # Getting the type of 'assert_array_almost_equal' (line 47)
        assert_array_almost_equal_353327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 47)
        assert_array_almost_equal_call_result_353331 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assert_array_almost_equal_353327, *[x_353328, y_353329], **kwargs_353330)
        
        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to array(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_353334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        # Adding element type (line 49)
        float_353335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 22), 'float')
        complex_353336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 39), 'complex')
        # Applying the binary operator '+' (line 49)
        result_add_353337 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 22), '+', float_353335, complex_353336)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 21), list_353334, result_add_353337)
        # Adding element type (line 49)
        float_353338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 22), 'float')
        complex_353339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 39), 'complex')
        # Applying the binary operator '+' (line 50)
        result_add_353340 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 22), '+', float_353338, complex_353339)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 21), list_353334, result_add_353340)
        # Adding element type (line 49)
        float_353341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 22), 'float')
        complex_353342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 39), 'complex')
        # Applying the binary operator '-' (line 51)
        result_sub_353343 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 22), '-', float_353341, complex_353342)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 21), list_353334, result_sub_353343)
        
        # Processing the call keyword arguments (line 49)
        kwargs_353344 = {}
        # Getting the type of 'np' (line 49)
        np_353332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 49)
        array_353333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), np_353332, 'array')
        # Calling array(args, kwargs) (line 49)
        array_call_result_353345 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), array_353333, *[list_353334], **kwargs_353344)
        
        # Assigning a type to the variable 'x' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'x', array_call_result_353345)
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to morlet(...): (line 52)
        # Processing the call arguments (line 52)
        int_353348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'int')
        # Processing the call keyword arguments (line 52)
        int_353349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 33), 'int')
        keyword_353350 = int_353349
        # Getting the type of 'False' (line 52)
        False_353351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 45), 'False', False)
        keyword_353352 = False_353351
        kwargs_353353 = {'complete': keyword_353352, 'w': keyword_353350}
        # Getting the type of 'wavelets' (line 52)
        wavelets_353346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 52)
        morlet_353347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), wavelets_353346, 'morlet')
        # Calling morlet(args, kwargs) (line 52)
        morlet_call_result_353354 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), morlet_353347, *[int_353348], **kwargs_353353)
        
        # Assigning a type to the variable 'y' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'y', morlet_call_result_353354)
        
        # Call to assert_array_almost_equal(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'x' (line 53)
        x_353356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'x', False)
        # Getting the type of 'y' (line 53)
        y_353357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 37), 'y', False)
        # Processing the call keyword arguments (line 53)
        int_353358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 48), 'int')
        keyword_353359 = int_353358
        kwargs_353360 = {'decimal': keyword_353359}
        # Getting the type of 'assert_array_almost_equal' (line 53)
        assert_array_almost_equal_353355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 53)
        assert_array_almost_equal_call_result_353361 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), assert_array_almost_equal_353355, *[x_353356, y_353357], **kwargs_353360)
        
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to morlet(...): (line 55)
        # Processing the call arguments (line 55)
        int_353364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 28), 'int')
        # Processing the call keyword arguments (line 55)
        int_353365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 37), 'int')
        keyword_353366 = int_353365
        # Getting the type of 'True' (line 55)
        True_353367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 49), 'True', False)
        keyword_353368 = True_353367
        kwargs_353369 = {'s': keyword_353366, 'complete': keyword_353368}
        # Getting the type of 'wavelets' (line 55)
        wavelets_353362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 55)
        morlet_353363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 12), wavelets_353362, 'morlet')
        # Calling morlet(args, kwargs) (line 55)
        morlet_call_result_353370 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), morlet_353363, *[int_353364], **kwargs_353369)
        
        # Assigning a type to the variable 'x' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'x', morlet_call_result_353370)
        
        # Assigning a Subscript to a Name (line 56):
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_353371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 55), 'int')
        int_353372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 60), 'int')
        slice_353373 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 12), int_353371, int_353372, None)
        
        # Call to morlet(...): (line 56)
        # Processing the call arguments (line 56)
        int_353376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 28), 'int')
        # Processing the call keyword arguments (line 56)
        int_353377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 37), 'int')
        keyword_353378 = int_353377
        # Getting the type of 'True' (line 56)
        True_353379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 49), 'True', False)
        keyword_353380 = True_353379
        kwargs_353381 = {'s': keyword_353378, 'complete': keyword_353380}
        # Getting the type of 'wavelets' (line 56)
        wavelets_353374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 56)
        morlet_353375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), wavelets_353374, 'morlet')
        # Calling morlet(args, kwargs) (line 56)
        morlet_call_result_353382 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), morlet_353375, *[int_353376], **kwargs_353381)
        
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___353383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), morlet_call_result_353382, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_353384 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), getitem___353383, slice_353373)
        
        # Assigning a type to the variable 'y' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'y', subscript_call_result_353384)
        
        # Call to assert_array_almost_equal(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'x' (line 57)
        x_353386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'x', False)
        # Getting the type of 'y' (line 57)
        y_353387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 37), 'y', False)
        # Processing the call keyword arguments (line 57)
        int_353388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 48), 'int')
        keyword_353389 = int_353388
        kwargs_353390 = {'decimal': keyword_353389}
        # Getting the type of 'assert_array_almost_equal' (line 57)
        assert_array_almost_equal_353385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 57)
        assert_array_almost_equal_call_result_353391 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), assert_array_almost_equal_353385, *[x_353386, y_353387], **kwargs_353390)
        
        
        # Assigning a Call to a Name (line 59):
        
        # Assigning a Call to a Name (line 59):
        
        # Call to morlet(...): (line 59)
        # Processing the call arguments (line 59)
        int_353394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 28), 'int')
        # Processing the call keyword arguments (line 59)
        int_353395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 37), 'int')
        keyword_353396 = int_353395
        # Getting the type of 'False' (line 59)
        False_353397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 49), 'False', False)
        keyword_353398 = False_353397
        kwargs_353399 = {'s': keyword_353396, 'complete': keyword_353398}
        # Getting the type of 'wavelets' (line 59)
        wavelets_353392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 59)
        morlet_353393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 12), wavelets_353392, 'morlet')
        # Calling morlet(args, kwargs) (line 59)
        morlet_call_result_353400 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), morlet_353393, *[int_353394], **kwargs_353399)
        
        # Assigning a type to the variable 'x' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'x', morlet_call_result_353400)
        
        # Call to assert_array_almost_equal(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'y' (line 60)
        y_353402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'y', False)
        # Getting the type of 'x' (line 60)
        x_353403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'x', False)
        # Processing the call keyword arguments (line 60)
        int_353404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 48), 'int')
        keyword_353405 = int_353404
        kwargs_353406 = {'decimal': keyword_353405}
        # Getting the type of 'assert_array_almost_equal' (line 60)
        assert_array_almost_equal_353401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 60)
        assert_array_almost_equal_call_result_353407 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assert_array_almost_equal_353401, *[y_353402, x_353403], **kwargs_353406)
        
        
        # Assigning a Subscript to a Name (line 61):
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        int_353408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 56), 'int')
        int_353409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 61), 'int')
        slice_353410 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 61, 12), int_353408, int_353409, None)
        
        # Call to morlet(...): (line 61)
        # Processing the call arguments (line 61)
        int_353413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 28), 'int')
        # Processing the call keyword arguments (line 61)
        int_353414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 37), 'int')
        keyword_353415 = int_353414
        # Getting the type of 'False' (line 61)
        False_353416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 49), 'False', False)
        keyword_353417 = False_353416
        kwargs_353418 = {'s': keyword_353415, 'complete': keyword_353417}
        # Getting the type of 'wavelets' (line 61)
        wavelets_353411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 61)
        morlet_353412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), wavelets_353411, 'morlet')
        # Calling morlet(args, kwargs) (line 61)
        morlet_call_result_353419 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), morlet_353412, *[int_353413], **kwargs_353418)
        
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___353420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), morlet_call_result_353419, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_353421 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), getitem___353420, slice_353410)
        
        # Assigning a type to the variable 'y' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'y', subscript_call_result_353421)
        
        # Call to assert_array_almost_equal(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'x' (line 62)
        x_353423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 34), 'x', False)
        # Getting the type of 'y' (line 62)
        y_353424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 37), 'y', False)
        # Processing the call keyword arguments (line 62)
        int_353425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 48), 'int')
        keyword_353426 = int_353425
        kwargs_353427 = {'decimal': keyword_353426}
        # Getting the type of 'assert_array_almost_equal' (line 62)
        assert_array_almost_equal_353422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 62)
        assert_array_almost_equal_call_result_353428 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), assert_array_almost_equal_353422, *[x_353423, y_353424], **kwargs_353427)
        
        
        # Assigning a Call to a Name (line 64):
        
        # Assigning a Call to a Name (line 64):
        
        # Call to morlet(...): (line 64)
        # Processing the call arguments (line 64)
        int_353431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 28), 'int')
        # Processing the call keyword arguments (line 64)
        int_353432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 37), 'int')
        keyword_353433 = int_353432
        int_353434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 42), 'int')
        keyword_353435 = int_353434
        # Getting the type of 'True' (line 64)
        True_353436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 54), 'True', False)
        keyword_353437 = True_353436
        kwargs_353438 = {'s': keyword_353435, 'complete': keyword_353437, 'w': keyword_353433}
        # Getting the type of 'wavelets' (line 64)
        wavelets_353429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 64)
        morlet_353430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), wavelets_353429, 'morlet')
        # Calling morlet(args, kwargs) (line 64)
        morlet_call_result_353439 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), morlet_353430, *[int_353431], **kwargs_353438)
        
        # Assigning a type to the variable 'x' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'x', morlet_call_result_353439)
        
        # Assigning a Subscript to a Name (line 65):
        
        # Assigning a Subscript to a Name (line 65):
        
        # Obtaining the type of the subscript
        int_353440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 61), 'int')
        int_353441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 66), 'int')
        slice_353442 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 65, 12), int_353440, int_353441, None)
        
        # Call to morlet(...): (line 65)
        # Processing the call arguments (line 65)
        int_353445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 28), 'int')
        # Processing the call keyword arguments (line 65)
        int_353446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 37), 'int')
        keyword_353447 = int_353446
        int_353448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 42), 'int')
        keyword_353449 = int_353448
        # Getting the type of 'True' (line 65)
        True_353450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 55), 'True', False)
        keyword_353451 = True_353450
        kwargs_353452 = {'s': keyword_353449, 'complete': keyword_353451, 'w': keyword_353447}
        # Getting the type of 'wavelets' (line 65)
        wavelets_353443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 65)
        morlet_353444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), wavelets_353443, 'morlet')
        # Calling morlet(args, kwargs) (line 65)
        morlet_call_result_353453 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), morlet_353444, *[int_353445], **kwargs_353452)
        
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___353454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), morlet_call_result_353453, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_353455 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), getitem___353454, slice_353442)
        
        # Assigning a type to the variable 'y' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'y', subscript_call_result_353455)
        
        # Call to assert_array_almost_equal(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'x' (line 66)
        x_353457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'x', False)
        # Getting the type of 'y' (line 66)
        y_353458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 37), 'y', False)
        # Processing the call keyword arguments (line 66)
        int_353459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 48), 'int')
        keyword_353460 = int_353459
        kwargs_353461 = {'decimal': keyword_353460}
        # Getting the type of 'assert_array_almost_equal' (line 66)
        assert_array_almost_equal_353456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 66)
        assert_array_almost_equal_call_result_353462 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assert_array_almost_equal_353456, *[x_353457, y_353458], **kwargs_353461)
        
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to morlet(...): (line 68)
        # Processing the call arguments (line 68)
        int_353465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'int')
        # Processing the call keyword arguments (line 68)
        int_353466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 37), 'int')
        keyword_353467 = int_353466
        int_353468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 42), 'int')
        keyword_353469 = int_353468
        # Getting the type of 'False' (line 68)
        False_353470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 54), 'False', False)
        keyword_353471 = False_353470
        kwargs_353472 = {'s': keyword_353469, 'complete': keyword_353471, 'w': keyword_353467}
        # Getting the type of 'wavelets' (line 68)
        wavelets_353463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 68)
        morlet_353464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), wavelets_353463, 'morlet')
        # Calling morlet(args, kwargs) (line 68)
        morlet_call_result_353473 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), morlet_353464, *[int_353465], **kwargs_353472)
        
        # Assigning a type to the variable 'x' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'x', morlet_call_result_353473)
        
        # Call to assert_array_almost_equal(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'y' (line 69)
        y_353475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 34), 'y', False)
        # Getting the type of 'x' (line 69)
        x_353476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 37), 'x', False)
        # Processing the call keyword arguments (line 69)
        int_353477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 48), 'int')
        keyword_353478 = int_353477
        kwargs_353479 = {'decimal': keyword_353478}
        # Getting the type of 'assert_array_almost_equal' (line 69)
        assert_array_almost_equal_353474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 69)
        assert_array_almost_equal_call_result_353480 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), assert_array_almost_equal_353474, *[y_353475, x_353476], **kwargs_353479)
        
        
        # Assigning a Subscript to a Name (line 70):
        
        # Assigning a Subscript to a Name (line 70):
        
        # Obtaining the type of the subscript
        int_353481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 62), 'int')
        int_353482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 67), 'int')
        slice_353483 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 70, 12), int_353481, int_353482, None)
        
        # Call to morlet(...): (line 70)
        # Processing the call arguments (line 70)
        int_353486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'int')
        # Processing the call keyword arguments (line 70)
        int_353487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 37), 'int')
        keyword_353488 = int_353487
        int_353489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 42), 'int')
        keyword_353490 = int_353489
        # Getting the type of 'False' (line 70)
        False_353491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 55), 'False', False)
        keyword_353492 = False_353491
        kwargs_353493 = {'s': keyword_353490, 'complete': keyword_353492, 'w': keyword_353488}
        # Getting the type of 'wavelets' (line 70)
        wavelets_353484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 70)
        morlet_353485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), wavelets_353484, 'morlet')
        # Calling morlet(args, kwargs) (line 70)
        morlet_call_result_353494 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), morlet_353485, *[int_353486], **kwargs_353493)
        
        # Obtaining the member '__getitem__' of a type (line 70)
        getitem___353495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), morlet_call_result_353494, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 70)
        subscript_call_result_353496 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), getitem___353495, slice_353483)
        
        # Assigning a type to the variable 'y' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'y', subscript_call_result_353496)
        
        # Call to assert_array_almost_equal(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'x' (line 71)
        x_353498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 34), 'x', False)
        # Getting the type of 'y' (line 71)
        y_353499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 37), 'y', False)
        # Processing the call keyword arguments (line 71)
        int_353500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 48), 'int')
        keyword_353501 = int_353500
        kwargs_353502 = {'decimal': keyword_353501}
        # Getting the type of 'assert_array_almost_equal' (line 71)
        assert_array_almost_equal_353497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 71)
        assert_array_almost_equal_call_result_353503 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), assert_array_almost_equal_353497, *[x_353498, y_353499], **kwargs_353502)
        
        
        # Assigning a Call to a Name (line 73):
        
        # Assigning a Call to a Name (line 73):
        
        # Call to morlet(...): (line 73)
        # Processing the call arguments (line 73)
        int_353506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'int')
        # Processing the call keyword arguments (line 73)
        int_353507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 37), 'int')
        keyword_353508 = int_353507
        int_353509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 42), 'int')
        keyword_353510 = int_353509
        # Getting the type of 'True' (line 73)
        True_353511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 55), 'True', False)
        keyword_353512 = True_353511
        kwargs_353513 = {'s': keyword_353510, 'complete': keyword_353512, 'w': keyword_353508}
        # Getting the type of 'wavelets' (line 73)
        wavelets_353504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 73)
        morlet_353505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 12), wavelets_353504, 'morlet')
        # Calling morlet(args, kwargs) (line 73)
        morlet_call_result_353514 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), morlet_353505, *[int_353506], **kwargs_353513)
        
        # Assigning a type to the variable 'x' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'x', morlet_call_result_353514)
        
        # Assigning a Subscript to a Name (line 74):
        
        # Assigning a Subscript to a Name (line 74):
        
        # Obtaining the type of the subscript
        int_353515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 61), 'int')
        int_353516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 66), 'int')
        slice_353517 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 74, 12), int_353515, int_353516, None)
        
        # Call to morlet(...): (line 74)
        # Processing the call arguments (line 74)
        int_353520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 28), 'int')
        # Processing the call keyword arguments (line 74)
        int_353521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 37), 'int')
        keyword_353522 = int_353521
        int_353523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 42), 'int')
        keyword_353524 = int_353523
        # Getting the type of 'True' (line 74)
        True_353525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 55), 'True', False)
        keyword_353526 = True_353525
        kwargs_353527 = {'s': keyword_353524, 'complete': keyword_353526, 'w': keyword_353522}
        # Getting the type of 'wavelets' (line 74)
        wavelets_353518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 74)
        morlet_353519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), wavelets_353518, 'morlet')
        # Calling morlet(args, kwargs) (line 74)
        morlet_call_result_353528 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), morlet_353519, *[int_353520], **kwargs_353527)
        
        # Obtaining the member '__getitem__' of a type (line 74)
        getitem___353529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), morlet_call_result_353528, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 74)
        subscript_call_result_353530 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), getitem___353529, slice_353517)
        
        # Assigning a type to the variable 'y' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'y', subscript_call_result_353530)
        
        # Call to assert_array_almost_equal(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'x' (line 75)
        x_353532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 34), 'x', False)
        # Getting the type of 'y' (line 75)
        y_353533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 37), 'y', False)
        # Processing the call keyword arguments (line 75)
        int_353534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 48), 'int')
        keyword_353535 = int_353534
        kwargs_353536 = {'decimal': keyword_353535}
        # Getting the type of 'assert_array_almost_equal' (line 75)
        assert_array_almost_equal_353531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 75)
        assert_array_almost_equal_call_result_353537 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), assert_array_almost_equal_353531, *[x_353532, y_353533], **kwargs_353536)
        
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to morlet(...): (line 77)
        # Processing the call arguments (line 77)
        int_353540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 28), 'int')
        # Processing the call keyword arguments (line 77)
        int_353541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 37), 'int')
        keyword_353542 = int_353541
        int_353543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 42), 'int')
        keyword_353544 = int_353543
        # Getting the type of 'False' (line 77)
        False_353545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 55), 'False', False)
        keyword_353546 = False_353545
        kwargs_353547 = {'s': keyword_353544, 'complete': keyword_353546, 'w': keyword_353542}
        # Getting the type of 'wavelets' (line 77)
        wavelets_353538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 77)
        morlet_353539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), wavelets_353538, 'morlet')
        # Calling morlet(args, kwargs) (line 77)
        morlet_call_result_353548 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), morlet_353539, *[int_353540], **kwargs_353547)
        
        # Assigning a type to the variable 'x' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'x', morlet_call_result_353548)
        
        # Call to assert_array_almost_equal(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'x' (line 78)
        x_353550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 34), 'x', False)
        # Getting the type of 'y' (line 78)
        y_353551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 37), 'y', False)
        # Processing the call keyword arguments (line 78)
        int_353552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 48), 'int')
        keyword_353553 = int_353552
        kwargs_353554 = {'decimal': keyword_353553}
        # Getting the type of 'assert_array_almost_equal' (line 78)
        assert_array_almost_equal_353549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 78)
        assert_array_almost_equal_call_result_353555 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), assert_array_almost_equal_353549, *[x_353550, y_353551], **kwargs_353554)
        
        
        # Assigning a Subscript to a Name (line 79):
        
        # Assigning a Subscript to a Name (line 79):
        
        # Obtaining the type of the subscript
        int_353556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 62), 'int')
        int_353557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 67), 'int')
        slice_353558 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 79, 12), int_353556, int_353557, None)
        
        # Call to morlet(...): (line 79)
        # Processing the call arguments (line 79)
        int_353561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'int')
        # Processing the call keyword arguments (line 79)
        int_353562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 37), 'int')
        keyword_353563 = int_353562
        int_353564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 42), 'int')
        keyword_353565 = int_353564
        # Getting the type of 'False' (line 79)
        False_353566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 55), 'False', False)
        keyword_353567 = False_353566
        kwargs_353568 = {'s': keyword_353565, 'complete': keyword_353567, 'w': keyword_353563}
        # Getting the type of 'wavelets' (line 79)
        wavelets_353559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'wavelets', False)
        # Obtaining the member 'morlet' of a type (line 79)
        morlet_353560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), wavelets_353559, 'morlet')
        # Calling morlet(args, kwargs) (line 79)
        morlet_call_result_353569 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), morlet_353560, *[int_353561], **kwargs_353568)
        
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___353570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), morlet_call_result_353569, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_353571 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), getitem___353570, slice_353558)
        
        # Assigning a type to the variable 'y' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'y', subscript_call_result_353571)
        
        # Call to assert_array_almost_equal(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'x' (line 80)
        x_353573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 'x', False)
        # Getting the type of 'y' (line 80)
        y_353574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 37), 'y', False)
        # Processing the call keyword arguments (line 80)
        int_353575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 48), 'int')
        keyword_353576 = int_353575
        kwargs_353577 = {'decimal': keyword_353576}
        # Getting the type of 'assert_array_almost_equal' (line 80)
        assert_array_almost_equal_353572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 80)
        assert_array_almost_equal_call_result_353578 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assert_array_almost_equal_353572, *[x_353573, y_353574], **kwargs_353577)
        
        
        # ################# End of 'test_morlet(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_morlet' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_353579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_353579)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_morlet'
        return stypy_return_type_353579


    @norecursion
    def test_ricker(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ricker'
        module_type_store = module_type_store.open_function_context('test_ricker', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestWavelets.test_ricker.__dict__.__setitem__('stypy_localization', localization)
        TestWavelets.test_ricker.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestWavelets.test_ricker.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestWavelets.test_ricker.__dict__.__setitem__('stypy_function_name', 'TestWavelets.test_ricker')
        TestWavelets.test_ricker.__dict__.__setitem__('stypy_param_names_list', [])
        TestWavelets.test_ricker.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestWavelets.test_ricker.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestWavelets.test_ricker.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestWavelets.test_ricker.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestWavelets.test_ricker.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestWavelets.test_ricker.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWavelets.test_ricker', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ricker', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ricker(...)' code ##################

        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to ricker(...): (line 83)
        # Processing the call arguments (line 83)
        float_353582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 28), 'float')
        int_353583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 33), 'int')
        # Processing the call keyword arguments (line 83)
        kwargs_353584 = {}
        # Getting the type of 'wavelets' (line 83)
        wavelets_353580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'wavelets', False)
        # Obtaining the member 'ricker' of a type (line 83)
        ricker_353581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), wavelets_353580, 'ricker')
        # Calling ricker(args, kwargs) (line 83)
        ricker_call_result_353585 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), ricker_353581, *[float_353582, int_353583], **kwargs_353584)
        
        # Assigning a type to the variable 'w' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'w', ricker_call_result_353585)
        
        # Assigning a BinOp to a Name (line 84):
        
        # Assigning a BinOp to a Name (line 84):
        int_353586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 19), 'int')
        
        # Call to sqrt(...): (line 84)
        # Processing the call arguments (line 84)
        int_353589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 32), 'int')
        float_353590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 36), 'float')
        # Applying the binary operator '*' (line 84)
        result_mul_353591 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 32), '*', int_353589, float_353590)
        
        # Processing the call keyword arguments (line 84)
        kwargs_353592 = {}
        # Getting the type of 'np' (line 84)
        np_353587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 24), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 84)
        sqrt_353588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 24), np_353587, 'sqrt')
        # Calling sqrt(args, kwargs) (line 84)
        sqrt_call_result_353593 = invoke(stypy.reporting.localization.Localization(__file__, 84, 24), sqrt_353588, *[result_mul_353591], **kwargs_353592)
        
        # Getting the type of 'np' (line 84)
        np_353594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 44), 'np')
        # Obtaining the member 'pi' of a type (line 84)
        pi_353595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 44), np_353594, 'pi')
        float_353596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 53), 'float')
        # Applying the binary operator '**' (line 84)
        result_pow_353597 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 44), '**', pi_353595, float_353596)
        
        # Applying the binary operator '*' (line 84)
        result_mul_353598 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 24), '*', sqrt_call_result_353593, result_pow_353597)
        
        # Applying the binary operator 'div' (line 84)
        result_div_353599 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 19), 'div', int_353586, result_mul_353598)
        
        # Assigning a type to the variable 'expected' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'expected', result_div_353599)
        
        # Call to assert_array_equal(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'w' (line 85)
        w_353601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'w', False)
        # Getting the type of 'expected' (line 85)
        expected_353602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 30), 'expected', False)
        # Processing the call keyword arguments (line 85)
        kwargs_353603 = {}
        # Getting the type of 'assert_array_equal' (line 85)
        assert_array_equal_353600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 85)
        assert_array_equal_call_result_353604 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), assert_array_equal_353600, *[w_353601, expected_353602], **kwargs_353603)
        
        
        # Assigning a List to a Name (line 87):
        
        # Assigning a List to a Name (line 87):
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_353605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        int_353606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 18), list_353605, int_353606)
        # Adding element type (line 87)
        int_353607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 18), list_353605, int_353607)
        # Adding element type (line 87)
        int_353608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 18), list_353605, int_353608)
        # Adding element type (line 87)
        int_353609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 18), list_353605, int_353609)
        # Adding element type (line 87)
        int_353610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 18), list_353605, int_353610)
        
        # Assigning a type to the variable 'lengths' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'lengths', list_353605)
        
        # Getting the type of 'lengths' (line 88)
        lengths_353611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'lengths')
        # Testing the type of a for loop iterable (line 88)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 8), lengths_353611)
        # Getting the type of the for loop variable (line 88)
        for_loop_var_353612 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 8), lengths_353611)
        # Assigning a type to the variable 'length' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'length', for_loop_var_353612)
        # SSA begins for a for statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to ricker(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'length' (line 89)
        length_353615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 32), 'length', False)
        float_353616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 40), 'float')
        # Processing the call keyword arguments (line 89)
        kwargs_353617 = {}
        # Getting the type of 'wavelets' (line 89)
        wavelets_353613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'wavelets', False)
        # Obtaining the member 'ricker' of a type (line 89)
        ricker_353614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 16), wavelets_353613, 'ricker')
        # Calling ricker(args, kwargs) (line 89)
        ricker_call_result_353618 = invoke(stypy.reporting.localization.Localization(__file__, 89, 16), ricker_353614, *[length_353615, float_353616], **kwargs_353617)
        
        # Assigning a type to the variable 'w' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'w', ricker_call_result_353618)
        
        # Call to assert_(...): (line 90)
        # Processing the call arguments (line 90)
        
        
        # Call to len(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'w' (line 90)
        w_353621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'w', False)
        # Processing the call keyword arguments (line 90)
        kwargs_353622 = {}
        # Getting the type of 'len' (line 90)
        len_353620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'len', False)
        # Calling len(args, kwargs) (line 90)
        len_call_result_353623 = invoke(stypy.reporting.localization.Localization(__file__, 90, 20), len_353620, *[w_353621], **kwargs_353622)
        
        # Getting the type of 'length' (line 90)
        length_353624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'length', False)
        # Applying the binary operator '==' (line 90)
        result_eq_353625 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 20), '==', len_call_result_353623, length_353624)
        
        # Processing the call keyword arguments (line 90)
        kwargs_353626 = {}
        # Getting the type of 'assert_' (line 90)
        assert__353619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 90)
        assert__call_result_353627 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), assert__353619, *[result_eq_353625], **kwargs_353626)
        
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to argmax(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'w' (line 91)
        w_353630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'w', False)
        # Processing the call keyword arguments (line 91)
        kwargs_353631 = {}
        # Getting the type of 'np' (line 91)
        np_353628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'np', False)
        # Obtaining the member 'argmax' of a type (line 91)
        argmax_353629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 22), np_353628, 'argmax')
        # Calling argmax(args, kwargs) (line 91)
        argmax_call_result_353632 = invoke(stypy.reporting.localization.Localization(__file__, 91, 22), argmax_353629, *[w_353630], **kwargs_353631)
        
        # Assigning a type to the variable 'max_loc' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'max_loc', argmax_call_result_353632)
        
        # Call to assert_(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Getting the type of 'max_loc' (line 92)
        max_loc_353634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'max_loc', False)
        # Getting the type of 'length' (line 92)
        length_353635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'length', False)
        int_353636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 42), 'int')
        # Applying the binary operator '//' (line 92)
        result_floordiv_353637 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 32), '//', length_353635, int_353636)
        
        # Applying the binary operator '==' (line 92)
        result_eq_353638 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 20), '==', max_loc_353634, result_floordiv_353637)
        
        # Processing the call keyword arguments (line 92)
        kwargs_353639 = {}
        # Getting the type of 'assert_' (line 92)
        assert__353633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 92)
        assert__call_result_353640 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), assert__353633, *[result_eq_353638], **kwargs_353639)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 94):
        
        # Assigning a Num to a Name (line 94):
        int_353641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 17), 'int')
        # Assigning a type to the variable 'points' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'points', int_353641)
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to ricker(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'points' (line 95)
        points_353644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'points', False)
        float_353645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 36), 'float')
        # Processing the call keyword arguments (line 95)
        kwargs_353646 = {}
        # Getting the type of 'wavelets' (line 95)
        wavelets_353642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'wavelets', False)
        # Obtaining the member 'ricker' of a type (line 95)
        ricker_353643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), wavelets_353642, 'ricker')
        # Calling ricker(args, kwargs) (line 95)
        ricker_call_result_353647 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), ricker_353643, *[points_353644, float_353645], **kwargs_353646)
        
        # Assigning a type to the variable 'w' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'w', ricker_call_result_353647)
        
        # Assigning a Call to a Name (line 96):
        
        # Assigning a Call to a Name (line 96):
        
        # Call to arange(...): (line 96)
        # Processing the call arguments (line 96)
        int_353650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 29), 'int')
        # Getting the type of 'points' (line 96)
        points_353651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 32), 'points', False)
        int_353652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 42), 'int')
        # Applying the binary operator '//' (line 96)
        result_floordiv_353653 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 32), '//', points_353651, int_353652)
        
        # Processing the call keyword arguments (line 96)
        kwargs_353654 = {}
        # Getting the type of 'np' (line 96)
        np_353648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 19), 'np', False)
        # Obtaining the member 'arange' of a type (line 96)
        arange_353649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 19), np_353648, 'arange')
        # Calling arange(args, kwargs) (line 96)
        arange_call_result_353655 = invoke(stypy.reporting.localization.Localization(__file__, 96, 19), arange_353649, *[int_353650, result_floordiv_353653], **kwargs_353654)
        
        # Assigning a type to the variable 'half_vec' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'half_vec', arange_call_result_353655)
        
        # Call to assert_array_almost_equal(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Obtaining the type of the subscript
        # Getting the type of 'half_vec' (line 98)
        half_vec_353657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 36), 'half_vec', False)
        # Getting the type of 'w' (line 98)
        w_353658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 34), 'w', False)
        # Obtaining the member '__getitem__' of a type (line 98)
        getitem___353659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 34), w_353658, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 98)
        subscript_call_result_353660 = invoke(stypy.reporting.localization.Localization(__file__, 98, 34), getitem___353659, half_vec_353657)
        
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'half_vec' (line 98)
        half_vec_353661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 51), 'half_vec', False)
        int_353662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 62), 'int')
        # Applying the binary operator '+' (line 98)
        result_add_353663 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 51), '+', half_vec_353661, int_353662)
        
        # Applying the 'usub' unary operator (line 98)
        result___neg___353664 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 49), 'usub', result_add_353663)
        
        # Getting the type of 'w' (line 98)
        w_353665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 47), 'w', False)
        # Obtaining the member '__getitem__' of a type (line 98)
        getitem___353666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 47), w_353665, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 98)
        subscript_call_result_353667 = invoke(stypy.reporting.localization.Localization(__file__, 98, 47), getitem___353666, result___neg___353664)
        
        # Processing the call keyword arguments (line 98)
        kwargs_353668 = {}
        # Getting the type of 'assert_array_almost_equal' (line 98)
        assert_array_almost_equal_353656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 98)
        assert_array_almost_equal_call_result_353669 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), assert_array_almost_equal_353656, *[subscript_call_result_353660, subscript_call_result_353667], **kwargs_353668)
        
        
        # Assigning a List to a Name (line 101):
        
        # Assigning a List to a Name (line 101):
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_353670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        int_353671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 14), list_353670, int_353671)
        # Adding element type (line 101)
        int_353672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 14), list_353670, int_353672)
        # Adding element type (line 101)
        int_353673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 14), list_353670, int_353673)
        # Adding element type (line 101)
        int_353674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 14), list_353670, int_353674)
        # Adding element type (line 101)
        int_353675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 14), list_353670, int_353675)
        
        # Assigning a type to the variable 'aas' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'aas', list_353670)
        
        # Assigning a Num to a Name (line 102):
        
        # Assigning a Num to a Name (line 102):
        int_353676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 17), 'int')
        # Assigning a type to the variable 'points' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'points', int_353676)
        
        # Getting the type of 'aas' (line 103)
        aas_353677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 17), 'aas')
        # Testing the type of a for loop iterable (line 103)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 103, 8), aas_353677)
        # Getting the type of the for loop variable (line 103)
        for_loop_var_353678 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 103, 8), aas_353677)
        # Assigning a type to the variable 'a' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'a', for_loop_var_353678)
        # SSA begins for a for statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 104):
        
        # Assigning a Call to a Name (line 104):
        
        # Call to ricker(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'points' (line 104)
        points_353681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 32), 'points', False)
        # Getting the type of 'a' (line 104)
        a_353682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 40), 'a', False)
        # Processing the call keyword arguments (line 104)
        kwargs_353683 = {}
        # Getting the type of 'wavelets' (line 104)
        wavelets_353679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'wavelets', False)
        # Obtaining the member 'ricker' of a type (line 104)
        ricker_353680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 16), wavelets_353679, 'ricker')
        # Calling ricker(args, kwargs) (line 104)
        ricker_call_result_353684 = invoke(stypy.reporting.localization.Localization(__file__, 104, 16), ricker_353680, *[points_353681, a_353682], **kwargs_353683)
        
        # Assigning a type to the variable 'w' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'w', ricker_call_result_353684)
        
        # Assigning a BinOp to a Name (line 105):
        
        # Assigning a BinOp to a Name (line 105):
        
        # Call to arange(...): (line 105)
        # Processing the call arguments (line 105)
        int_353687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 28), 'int')
        # Getting the type of 'points' (line 105)
        points_353688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 31), 'points', False)
        # Processing the call keyword arguments (line 105)
        kwargs_353689 = {}
        # Getting the type of 'np' (line 105)
        np_353685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'np', False)
        # Obtaining the member 'arange' of a type (line 105)
        arange_353686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 18), np_353685, 'arange')
        # Calling arange(args, kwargs) (line 105)
        arange_call_result_353690 = invoke(stypy.reporting.localization.Localization(__file__, 105, 18), arange_353686, *[int_353687, points_353688], **kwargs_353689)
        
        # Getting the type of 'points' (line 105)
        points_353691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 42), 'points')
        float_353692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 51), 'float')
        # Applying the binary operator '-' (line 105)
        result_sub_353693 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 42), '-', points_353691, float_353692)
        
        int_353694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 58), 'int')
        # Applying the binary operator 'div' (line 105)
        result_div_353695 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 41), 'div', result_sub_353693, int_353694)
        
        # Applying the binary operator '-' (line 105)
        result_sub_353696 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 18), '-', arange_call_result_353690, result_div_353695)
        
        # Assigning a type to the variable 'vec' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'vec', result_sub_353696)
        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to argmin(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Call to abs(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'vec' (line 106)
        vec_353701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 41), 'vec', False)
        # Getting the type of 'a' (line 106)
        a_353702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 47), 'a', False)
        # Applying the binary operator '-' (line 106)
        result_sub_353703 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 41), '-', vec_353701, a_353702)
        
        # Processing the call keyword arguments (line 106)
        kwargs_353704 = {}
        # Getting the type of 'np' (line 106)
        np_353699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 34), 'np', False)
        # Obtaining the member 'abs' of a type (line 106)
        abs_353700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 34), np_353699, 'abs')
        # Calling abs(args, kwargs) (line 106)
        abs_call_result_353705 = invoke(stypy.reporting.localization.Localization(__file__, 106, 34), abs_353700, *[result_sub_353703], **kwargs_353704)
        
        # Processing the call keyword arguments (line 106)
        kwargs_353706 = {}
        # Getting the type of 'np' (line 106)
        np_353697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'np', False)
        # Obtaining the member 'argmin' of a type (line 106)
        argmin_353698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 24), np_353697, 'argmin')
        # Calling argmin(args, kwargs) (line 106)
        argmin_call_result_353707 = invoke(stypy.reporting.localization.Localization(__file__, 106, 24), argmin_353698, *[abs_call_result_353705], **kwargs_353706)
        
        # Assigning a type to the variable 'exp_zero1' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'exp_zero1', argmin_call_result_353707)
        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to argmin(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Call to abs(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'vec' (line 107)
        vec_353712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 41), 'vec', False)
        # Getting the type of 'a' (line 107)
        a_353713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 47), 'a', False)
        # Applying the binary operator '+' (line 107)
        result_add_353714 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 41), '+', vec_353712, a_353713)
        
        # Processing the call keyword arguments (line 107)
        kwargs_353715 = {}
        # Getting the type of 'np' (line 107)
        np_353710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 34), 'np', False)
        # Obtaining the member 'abs' of a type (line 107)
        abs_353711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 34), np_353710, 'abs')
        # Calling abs(args, kwargs) (line 107)
        abs_call_result_353716 = invoke(stypy.reporting.localization.Localization(__file__, 107, 34), abs_353711, *[result_add_353714], **kwargs_353715)
        
        # Processing the call keyword arguments (line 107)
        kwargs_353717 = {}
        # Getting the type of 'np' (line 107)
        np_353708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'np', False)
        # Obtaining the member 'argmin' of a type (line 107)
        argmin_353709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 24), np_353708, 'argmin')
        # Calling argmin(args, kwargs) (line 107)
        argmin_call_result_353718 = invoke(stypy.reporting.localization.Localization(__file__, 107, 24), argmin_353709, *[abs_call_result_353716], **kwargs_353717)
        
        # Assigning a type to the variable 'exp_zero2' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'exp_zero2', argmin_call_result_353718)
        
        # Call to assert_array_almost_equal(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining the type of the subscript
        # Getting the type of 'exp_zero1' (line 108)
        exp_zero1_353720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), 'exp_zero1', False)
        # Getting the type of 'w' (line 108)
        w_353721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 38), 'w', False)
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___353722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 38), w_353721, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_353723 = invoke(stypy.reporting.localization.Localization(__file__, 108, 38), getitem___353722, exp_zero1_353720)
        
        int_353724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 52), 'int')
        # Processing the call keyword arguments (line 108)
        kwargs_353725 = {}
        # Getting the type of 'assert_array_almost_equal' (line 108)
        assert_array_almost_equal_353719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 108)
        assert_array_almost_equal_call_result_353726 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), assert_array_almost_equal_353719, *[subscript_call_result_353723, int_353724], **kwargs_353725)
        
        
        # Call to assert_array_almost_equal(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining the type of the subscript
        # Getting the type of 'exp_zero2' (line 109)
        exp_zero2_353728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 40), 'exp_zero2', False)
        # Getting the type of 'w' (line 109)
        w_353729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 38), 'w', False)
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___353730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 38), w_353729, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_353731 = invoke(stypy.reporting.localization.Localization(__file__, 109, 38), getitem___353730, exp_zero2_353728)
        
        int_353732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 52), 'int')
        # Processing the call keyword arguments (line 109)
        kwargs_353733 = {}
        # Getting the type of 'assert_array_almost_equal' (line 109)
        assert_array_almost_equal_353727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 109)
        assert_array_almost_equal_call_result_353734 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), assert_array_almost_equal_353727, *[subscript_call_result_353731, int_353732], **kwargs_353733)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_ricker(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ricker' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_353735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_353735)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ricker'
        return stypy_return_type_353735


    @norecursion
    def test_cwt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cwt'
        module_type_store = module_type_store.open_function_context('test_cwt', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestWavelets.test_cwt.__dict__.__setitem__('stypy_localization', localization)
        TestWavelets.test_cwt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestWavelets.test_cwt.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestWavelets.test_cwt.__dict__.__setitem__('stypy_function_name', 'TestWavelets.test_cwt')
        TestWavelets.test_cwt.__dict__.__setitem__('stypy_param_names_list', [])
        TestWavelets.test_cwt.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestWavelets.test_cwt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestWavelets.test_cwt.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestWavelets.test_cwt.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestWavelets.test_cwt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestWavelets.test_cwt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWavelets.test_cwt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cwt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cwt(...)' code ##################

        
        # Assigning a List to a Name (line 112):
        
        # Assigning a List to a Name (line 112):
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_353736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        float_353737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 17), list_353736, float_353737)
        
        # Assigning a type to the variable 'widths' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'widths', list_353736)
        
        # Assigning a Lambda to a Name (line 113):
        
        # Assigning a Lambda to a Name (line 113):

        @norecursion
        def _stypy_temp_lambda_199(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_199'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_199', 113, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_199.stypy_localization = localization
            _stypy_temp_lambda_199.stypy_type_of_self = None
            _stypy_temp_lambda_199.stypy_type_store = module_type_store
            _stypy_temp_lambda_199.stypy_function_name = '_stypy_temp_lambda_199'
            _stypy_temp_lambda_199.stypy_param_names_list = ['s', 't']
            _stypy_temp_lambda_199.stypy_varargs_param_name = None
            _stypy_temp_lambda_199.stypy_kwargs_param_name = None
            _stypy_temp_lambda_199.stypy_call_defaults = defaults
            _stypy_temp_lambda_199.stypy_call_varargs = varargs
            _stypy_temp_lambda_199.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_199', ['s', 't'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_199', ['s', 't'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to array(...): (line 113)
            # Processing the call arguments (line 113)
            
            # Obtaining an instance of the builtin type 'list' (line 113)
            list_353740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 46), 'list')
            # Adding type elements to the builtin type 'list' instance (line 113)
            # Adding element type (line 113)
            int_353741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 47), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 46), list_353740, int_353741)
            
            # Processing the call keyword arguments (line 113)
            kwargs_353742 = {}
            # Getting the type of 'np' (line 113)
            np_353738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 37), 'np', False)
            # Obtaining the member 'array' of a type (line 113)
            array_353739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 37), np_353738, 'array')
            # Calling array(args, kwargs) (line 113)
            array_call_result_353743 = invoke(stypy.reporting.localization.Localization(__file__, 113, 37), array_353739, *[list_353740], **kwargs_353742)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 113)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'stypy_return_type', array_call_result_353743)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_199' in the type store
            # Getting the type of 'stypy_return_type' (line 113)
            stypy_return_type_353744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_353744)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_199'
            return stypy_return_type_353744

        # Assigning a type to the variable '_stypy_temp_lambda_199' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), '_stypy_temp_lambda_199', _stypy_temp_lambda_199)
        # Getting the type of '_stypy_temp_lambda_199' (line 113)
        _stypy_temp_lambda_199_353745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), '_stypy_temp_lambda_199')
        # Assigning a type to the variable 'delta_wavelet' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'delta_wavelet', _stypy_temp_lambda_199_353745)
        
        # Assigning a Num to a Name (line 114):
        
        # Assigning a Num to a Name (line 114):
        int_353746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 19), 'int')
        # Assigning a type to the variable 'len_data' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'len_data', int_353746)
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to sin(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'np' (line 115)
        np_353749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'np', False)
        # Obtaining the member 'pi' of a type (line 115)
        pi_353750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 27), np_353749, 'pi')
        
        # Call to arange(...): (line 115)
        # Processing the call arguments (line 115)
        int_353753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 45), 'int')
        # Getting the type of 'len_data' (line 115)
        len_data_353754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 48), 'len_data', False)
        # Processing the call keyword arguments (line 115)
        kwargs_353755 = {}
        # Getting the type of 'np' (line 115)
        np_353751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 35), 'np', False)
        # Obtaining the member 'arange' of a type (line 115)
        arange_353752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 35), np_353751, 'arange')
        # Calling arange(args, kwargs) (line 115)
        arange_call_result_353756 = invoke(stypy.reporting.localization.Localization(__file__, 115, 35), arange_353752, *[int_353753, len_data_353754], **kwargs_353755)
        
        # Applying the binary operator '*' (line 115)
        result_mul_353757 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 27), '*', pi_353750, arange_call_result_353756)
        
        float_353758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 60), 'float')
        # Applying the binary operator 'div' (line 115)
        result_div_353759 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 58), 'div', result_mul_353757, float_353758)
        
        # Processing the call keyword arguments (line 115)
        kwargs_353760 = {}
        # Getting the type of 'np' (line 115)
        np_353747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'np', False)
        # Obtaining the member 'sin' of a type (line 115)
        sin_353748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 20), np_353747, 'sin')
        # Calling sin(args, kwargs) (line 115)
        sin_call_result_353761 = invoke(stypy.reporting.localization.Localization(__file__, 115, 20), sin_353748, *[result_div_353759], **kwargs_353760)
        
        # Assigning a type to the variable 'test_data' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'test_data', sin_call_result_353761)
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to cwt(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'test_data' (line 118)
        test_data_353764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'test_data', False)
        # Getting the type of 'delta_wavelet' (line 118)
        delta_wavelet_353765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 42), 'delta_wavelet', False)
        # Getting the type of 'widths' (line 118)
        widths_353766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 57), 'widths', False)
        # Processing the call keyword arguments (line 118)
        kwargs_353767 = {}
        # Getting the type of 'wavelets' (line 118)
        wavelets_353762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 18), 'wavelets', False)
        # Obtaining the member 'cwt' of a type (line 118)
        cwt_353763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 18), wavelets_353762, 'cwt')
        # Calling cwt(args, kwargs) (line 118)
        cwt_call_result_353768 = invoke(stypy.reporting.localization.Localization(__file__, 118, 18), cwt_353763, *[test_data_353764, delta_wavelet_353765, widths_353766], **kwargs_353767)
        
        # Assigning a type to the variable 'cwt_dat' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'cwt_dat', cwt_call_result_353768)
        
        # Call to assert_(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Getting the type of 'cwt_dat' (line 119)
        cwt_dat_353770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'cwt_dat', False)
        # Obtaining the member 'shape' of a type (line 119)
        shape_353771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 16), cwt_dat_353770, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 119)
        tuple_353772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 119)
        # Adding element type (line 119)
        
        # Call to len(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'widths' (line 119)
        widths_353774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'widths', False)
        # Processing the call keyword arguments (line 119)
        kwargs_353775 = {}
        # Getting the type of 'len' (line 119)
        len_353773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 34), 'len', False)
        # Calling len(args, kwargs) (line 119)
        len_call_result_353776 = invoke(stypy.reporting.localization.Localization(__file__, 119, 34), len_353773, *[widths_353774], **kwargs_353775)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 34), tuple_353772, len_call_result_353776)
        # Adding element type (line 119)
        # Getting the type of 'len_data' (line 119)
        len_data_353777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 47), 'len_data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 34), tuple_353772, len_data_353777)
        
        # Applying the binary operator '==' (line 119)
        result_eq_353778 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 16), '==', shape_353771, tuple_353772)
        
        # Processing the call keyword arguments (line 119)
        kwargs_353779 = {}
        # Getting the type of 'assert_' (line 119)
        assert__353769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 119)
        assert__call_result_353780 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), assert__353769, *[result_eq_353778], **kwargs_353779)
        
        
        # Call to assert_array_almost_equal(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'test_data' (line 120)
        test_data_353782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 34), 'test_data', False)
        
        # Call to flatten(...): (line 120)
        # Processing the call keyword arguments (line 120)
        kwargs_353785 = {}
        # Getting the type of 'cwt_dat' (line 120)
        cwt_dat_353783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 45), 'cwt_dat', False)
        # Obtaining the member 'flatten' of a type (line 120)
        flatten_353784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 45), cwt_dat_353783, 'flatten')
        # Calling flatten(args, kwargs) (line 120)
        flatten_call_result_353786 = invoke(stypy.reporting.localization.Localization(__file__, 120, 45), flatten_353784, *[], **kwargs_353785)
        
        # Processing the call keyword arguments (line 120)
        kwargs_353787 = {}
        # Getting the type of 'assert_array_almost_equal' (line 120)
        assert_array_almost_equal_353781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 120)
        assert_array_almost_equal_call_result_353788 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), assert_array_almost_equal_353781, *[test_data_353782, flatten_call_result_353786], **kwargs_353787)
        
        
        # Assigning a List to a Name (line 123):
        
        # Assigning a List to a Name (line 123):
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_353789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        int_353790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 17), list_353789, int_353790)
        # Adding element type (line 123)
        int_353791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 17), list_353789, int_353791)
        # Adding element type (line 123)
        int_353792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 17), list_353789, int_353792)
        # Adding element type (line 123)
        int_353793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 17), list_353789, int_353793)
        # Adding element type (line 123)
        int_353794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 17), list_353789, int_353794)
        
        # Assigning a type to the variable 'widths' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'widths', list_353789)
        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to cwt(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'test_data' (line 124)
        test_data_353797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 31), 'test_data', False)
        # Getting the type of 'wavelets' (line 124)
        wavelets_353798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 42), 'wavelets', False)
        # Obtaining the member 'ricker' of a type (line 124)
        ricker_353799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 42), wavelets_353798, 'ricker')
        # Getting the type of 'widths' (line 124)
        widths_353800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 59), 'widths', False)
        # Processing the call keyword arguments (line 124)
        kwargs_353801 = {}
        # Getting the type of 'wavelets' (line 124)
        wavelets_353795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'wavelets', False)
        # Obtaining the member 'cwt' of a type (line 124)
        cwt_353796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 18), wavelets_353795, 'cwt')
        # Calling cwt(args, kwargs) (line 124)
        cwt_call_result_353802 = invoke(stypy.reporting.localization.Localization(__file__, 124, 18), cwt_353796, *[test_data_353797, ricker_353799, widths_353800], **kwargs_353801)
        
        # Assigning a type to the variable 'cwt_dat' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'cwt_dat', cwt_call_result_353802)
        
        # Call to assert_(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Getting the type of 'cwt_dat' (line 125)
        cwt_dat_353804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'cwt_dat', False)
        # Obtaining the member 'shape' of a type (line 125)
        shape_353805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 16), cwt_dat_353804, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 125)
        tuple_353806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 125)
        # Adding element type (line 125)
        
        # Call to len(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'widths' (line 125)
        widths_353808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 38), 'widths', False)
        # Processing the call keyword arguments (line 125)
        kwargs_353809 = {}
        # Getting the type of 'len' (line 125)
        len_353807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 34), 'len', False)
        # Calling len(args, kwargs) (line 125)
        len_call_result_353810 = invoke(stypy.reporting.localization.Localization(__file__, 125, 34), len_353807, *[widths_353808], **kwargs_353809)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 34), tuple_353806, len_call_result_353810)
        # Adding element type (line 125)
        # Getting the type of 'len_data' (line 125)
        len_data_353811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 47), 'len_data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 34), tuple_353806, len_data_353811)
        
        # Applying the binary operator '==' (line 125)
        result_eq_353812 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 16), '==', shape_353805, tuple_353806)
        
        # Processing the call keyword arguments (line 125)
        kwargs_353813 = {}
        # Getting the type of 'assert_' (line 125)
        assert__353803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 125)
        assert__call_result_353814 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), assert__353803, *[result_eq_353812], **kwargs_353813)
        
        
        # Assigning a List to a Name (line 127):
        
        # Assigning a List to a Name (line 127):
        
        # Obtaining an instance of the builtin type 'list' (line 127)
        list_353815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 127)
        # Adding element type (line 127)
        # Getting the type of 'len_data' (line 127)
        len_data_353816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 18), 'len_data')
        int_353817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 29), 'int')
        # Applying the binary operator '*' (line 127)
        result_mul_353818 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 18), '*', len_data_353816, int_353817)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 17), list_353815, result_mul_353818)
        
        # Assigning a type to the variable 'widths' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'widths', list_353815)
        
        # Assigning a Lambda to a Name (line 129):
        
        # Assigning a Lambda to a Name (line 129):

        @norecursion
        def _stypy_temp_lambda_200(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_200'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_200', 129, 23, True)
            # Passed parameters checking function
            _stypy_temp_lambda_200.stypy_localization = localization
            _stypy_temp_lambda_200.stypy_type_of_self = None
            _stypy_temp_lambda_200.stypy_type_store = module_type_store
            _stypy_temp_lambda_200.stypy_function_name = '_stypy_temp_lambda_200'
            _stypy_temp_lambda_200.stypy_param_names_list = ['l', 'w']
            _stypy_temp_lambda_200.stypy_varargs_param_name = None
            _stypy_temp_lambda_200.stypy_kwargs_param_name = None
            _stypy_temp_lambda_200.stypy_call_defaults = defaults
            _stypy_temp_lambda_200.stypy_call_varargs = varargs
            _stypy_temp_lambda_200.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_200', ['l', 'w'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_200', ['l', 'w'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to ones(...): (line 129)
            # Processing the call arguments (line 129)
            # Getting the type of 'w' (line 129)
            w_353821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 44), 'w', False)
            # Processing the call keyword arguments (line 129)
            kwargs_353822 = {}
            # Getting the type of 'np' (line 129)
            np_353819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 36), 'np', False)
            # Obtaining the member 'ones' of a type (line 129)
            ones_353820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 36), np_353819, 'ones')
            # Calling ones(args, kwargs) (line 129)
            ones_call_result_353823 = invoke(stypy.reporting.localization.Localization(__file__, 129, 36), ones_353820, *[w_353821], **kwargs_353822)
            
            # Getting the type of 'w' (line 129)
            w_353824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'w')
            # Applying the binary operator 'div' (line 129)
            result_div_353825 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 36), 'div', ones_call_result_353823, w_353824)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'stypy_return_type', result_div_353825)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_200' in the type store
            # Getting the type of 'stypy_return_type' (line 129)
            stypy_return_type_353826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_353826)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_200'
            return stypy_return_type_353826

        # Assigning a type to the variable '_stypy_temp_lambda_200' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), '_stypy_temp_lambda_200', _stypy_temp_lambda_200)
        # Getting the type of '_stypy_temp_lambda_200' (line 129)
        _stypy_temp_lambda_200_353827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), '_stypy_temp_lambda_200')
        # Assigning a type to the variable 'flat_wavelet' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'flat_wavelet', _stypy_temp_lambda_200_353827)
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to cwt(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'test_data' (line 130)
        test_data_353830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 31), 'test_data', False)
        # Getting the type of 'flat_wavelet' (line 130)
        flat_wavelet_353831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 42), 'flat_wavelet', False)
        # Getting the type of 'widths' (line 130)
        widths_353832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 56), 'widths', False)
        # Processing the call keyword arguments (line 130)
        kwargs_353833 = {}
        # Getting the type of 'wavelets' (line 130)
        wavelets_353828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'wavelets', False)
        # Obtaining the member 'cwt' of a type (line 130)
        cwt_353829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 18), wavelets_353828, 'cwt')
        # Calling cwt(args, kwargs) (line 130)
        cwt_call_result_353834 = invoke(stypy.reporting.localization.Localization(__file__, 130, 18), cwt_353829, *[test_data_353830, flat_wavelet_353831, widths_353832], **kwargs_353833)
        
        # Assigning a type to the variable 'cwt_dat' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'cwt_dat', cwt_call_result_353834)
        
        # Call to assert_array_almost_equal(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'cwt_dat' (line 131)
        cwt_dat_353836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 34), 'cwt_dat', False)
        
        # Call to mean(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'test_data' (line 131)
        test_data_353839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 51), 'test_data', False)
        # Processing the call keyword arguments (line 131)
        kwargs_353840 = {}
        # Getting the type of 'np' (line 131)
        np_353837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 43), 'np', False)
        # Obtaining the member 'mean' of a type (line 131)
        mean_353838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 43), np_353837, 'mean')
        # Calling mean(args, kwargs) (line 131)
        mean_call_result_353841 = invoke(stypy.reporting.localization.Localization(__file__, 131, 43), mean_353838, *[test_data_353839], **kwargs_353840)
        
        # Processing the call keyword arguments (line 131)
        kwargs_353842 = {}
        # Getting the type of 'assert_array_almost_equal' (line 131)
        assert_array_almost_equal_353835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 131)
        assert_array_almost_equal_call_result_353843 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), assert_array_almost_equal_353835, *[cwt_dat_353836, mean_call_result_353841], **kwargs_353842)
        
        
        # ################# End of 'test_cwt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cwt' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_353844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_353844)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cwt'
        return stypy_return_type_353844


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWavelets.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestWavelets' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'TestWavelets', TestWavelets)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
