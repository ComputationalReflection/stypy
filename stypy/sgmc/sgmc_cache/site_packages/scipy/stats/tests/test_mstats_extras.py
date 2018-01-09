
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: import numpy.ma as ma
5: import scipy.stats.mstats as ms
6: 
7: from numpy.testing import (assert_equal, assert_almost_equal, assert_,
8:     assert_allclose)
9: 
10: 
11: def test_compare_medians_ms():
12:     x = np.arange(7)
13:     y = x + 10
14:     assert_almost_equal(ms.compare_medians_ms(x, y), 0)
15: 
16:     y2 = np.linspace(0, 1, num=10)
17:     assert_almost_equal(ms.compare_medians_ms(x, y2), 0.017116406778)
18: 
19: 
20: def test_hdmedian():
21:     # 1-D array
22:     x = ma.arange(11)
23:     assert_equal(ms.hdmedian(x), 5)
24:     x.mask = ma.make_mask(x)
25:     x.mask[:7] = False
26:     assert_equal(ms.hdmedian(x), 3)
27: 
28:     # Check that `var` keyword returns a value.  TODO: check whether returned
29:     # value is actually correct.
30:     assert_(ms.hdmedian(x, var=True).size == 2)
31: 
32:     # 2-D array
33:     x2 = ma.arange(22).reshape((11, 2))
34:     assert_allclose(ms.hdmedian(x2, axis=0), [10, 11])
35:     x2.mask = ma.make_mask(x2)
36:     x2.mask[:7, :] = False
37:     assert_allclose(ms.hdmedian(x2, axis=0), [6, 7])
38: 
39: 
40: def test_rsh():
41:     np.random.seed(132345)
42:     x = np.random.randn(100)
43:     res = ms.rsh(x)
44:     # Just a sanity check that the code runs and output shape is correct.
45:     # TODO: check that implementation is correct.
46:     assert_(res.shape == x.shape)
47: 
48:     # Check points keyword
49:     res = ms.rsh(x, points=[0, 1.])
50:     assert_(res.size == 2)
51: 
52: 
53: def test_mjci():
54:     # Tests the Marits-Jarrett estimator
55:     data = ma.array([77, 87, 88,114,151,210,219,246,253,262,
56:                       296,299,306,376,428,515,666,1310,2611])
57:     assert_almost_equal(ms.mjci(data),[55.76819,45.84028,198.87875],5)
58: 
59: 
60: def test_trimmed_mean_ci():
61:     # Tests the confidence intervals of the trimmed mean.
62:     data = ma.array([545,555,558,572,575,576,578,580,
63:                      594,605,635,651,653,661,666])
64:     assert_almost_equal(ms.trimmed_mean(data,0.2), 596.2, 1)
65:     assert_equal(np.round(ms.trimmed_mean_ci(data,(0.2,0.2)),1),
66:                  [561.8, 630.6])
67: 
68: 
69: def test_idealfourths():
70:     # Tests ideal-fourths
71:     test = np.arange(100)
72:     assert_almost_equal(np.asarray(ms.idealfourths(test)),
73:                         [24.416667,74.583333],6)
74:     test_2D = test.repeat(3).reshape(-1,3)
75:     assert_almost_equal(ms.idealfourths(test_2D, axis=0),
76:                         [[24.416667,24.416667,24.416667],
77:                          [74.583333,74.583333,74.583333]],6)
78:     assert_almost_equal(ms.idealfourths(test_2D, axis=1),
79:                         test.repeat(2).reshape(-1,2))
80:     test = [0, 0]
81:     _result = ms.idealfourths(test)
82:     assert_(np.isnan(_result).all())
83: 
84: 
85: class TestQuantiles(object):
86:     data = [0.706560797,0.727229578,0.990399276,0.927065621,0.158953014,
87:         0.887764025,0.239407086,0.349638551,0.972791145,0.149789972,
88:         0.936947700,0.132359948,0.046041972,0.641675031,0.945530547,
89:         0.224218684,0.771450991,0.820257774,0.336458052,0.589113496,
90:         0.509736129,0.696838829,0.491323573,0.622767425,0.775189248,
91:         0.641461450,0.118455200,0.773029450,0.319280007,0.752229111,
92:         0.047841438,0.466295911,0.583850781,0.840581845,0.550086491,
93:         0.466470062,0.504765074,0.226855960,0.362641207,0.891620942,
94:         0.127898691,0.490094097,0.044882048,0.041441695,0.317976349,
95:         0.504135618,0.567353033,0.434617473,0.636243375,0.231803616,
96:         0.230154113,0.160011327,0.819464108,0.854706985,0.438809221,
97:         0.487427267,0.786907310,0.408367937,0.405534192,0.250444460,
98:         0.995309248,0.144389588,0.739947527,0.953543606,0.680051621,
99:         0.388382017,0.863530727,0.006514031,0.118007779,0.924024803,
100:         0.384236354,0.893687694,0.626534881,0.473051932,0.750134705,
101:         0.241843555,0.432947602,0.689538104,0.136934797,0.150206859,
102:         0.474335206,0.907775349,0.525869295,0.189184225,0.854284286,
103:         0.831089744,0.251637345,0.587038213,0.254475554,0.237781276,
104:         0.827928620,0.480283781,0.594514455,0.213641488,0.024194386,
105:         0.536668589,0.699497811,0.892804071,0.093835427,0.731107772]
106: 
107:     def test_hdquantiles(self):
108:         data = self.data
109:         assert_almost_equal(ms.hdquantiles(data,[0., 1.]),
110:                             [0.006514031, 0.995309248])
111:         hdq = ms.hdquantiles(data,[0.25, 0.5, 0.75])
112:         assert_almost_equal(hdq, [0.253210762, 0.512847491, 0.762232442,])
113:         hdq = ms.hdquantiles_sd(data,[0.25, 0.5, 0.75])
114:         assert_almost_equal(hdq, [0.03786954, 0.03805389, 0.03800152,], 4)
115: 
116:         data = np.array(data).reshape(10,10)
117:         hdq = ms.hdquantiles(data,[0.25,0.5,0.75],axis=0)
118:         assert_almost_equal(hdq[:,0], ms.hdquantiles(data[:,0],[0.25,0.5,0.75]))
119:         assert_almost_equal(hdq[:,-1], ms.hdquantiles(data[:,-1],[0.25,0.5,0.75]))
120:         hdq = ms.hdquantiles(data,[0.25,0.5,0.75],axis=0,var=True)
121:         assert_almost_equal(hdq[...,0],
122:                             ms.hdquantiles(data[:,0],[0.25,0.5,0.75],var=True))
123:         assert_almost_equal(hdq[...,-1],
124:                             ms.hdquantiles(data[:,-1],[0.25,0.5,0.75], var=True))
125: 
126:     def test_hdquantiles_sd(self):
127:         # Only test that code runs, implementation not checked for correctness
128:         res = ms.hdquantiles_sd(self.data)
129:         assert_(res.size == 3)
130: 
131:     def test_mquantiles_cimj(self):
132:         # Only test that code runs, implementation not checked for correctness
133:         ci_lower, ci_upper = ms.mquantiles_cimj(self.data)
134:         assert_(ci_lower.size == ci_upper.size == 3)
135: 
136: 
137: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_670849 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_670849) is not StypyTypeError):

    if (import_670849 != 'pyd_module'):
        __import__(import_670849)
        sys_modules_670850 = sys.modules[import_670849]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_670850.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_670849)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy.ma' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_670851 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.ma')

if (type(import_670851) is not StypyTypeError):

    if (import_670851 != 'pyd_module'):
        __import__(import_670851)
        sys_modules_670852 = sys.modules[import_670851]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'ma', sys_modules_670852.module_type_store, module_type_store)
    else:
        import numpy.ma as ma

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'ma', numpy.ma, module_type_store)

else:
    # Assigning a type to the variable 'numpy.ma' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.ma', import_670851)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import scipy.stats.mstats' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_670853 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.stats.mstats')

if (type(import_670853) is not StypyTypeError):

    if (import_670853 != 'pyd_module'):
        __import__(import_670853)
        sys_modules_670854 = sys.modules[import_670853]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'ms', sys_modules_670854.module_type_store, module_type_store)
    else:
        import scipy.stats.mstats as ms

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'ms', scipy.stats.mstats, module_type_store)

else:
    # Assigning a type to the variable 'scipy.stats.mstats' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.stats.mstats', import_670853)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_equal, assert_almost_equal, assert_, assert_allclose' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_670855 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_670855) is not StypyTypeError):

    if (import_670855 != 'pyd_module'):
        __import__(import_670855)
        sys_modules_670856 = sys.modules[import_670855]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_670856.module_type_store, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_670856, sys_modules_670856.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_almost_equal, assert_, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_', 'assert_allclose'], [assert_equal, assert_almost_equal, assert_, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_670855)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')


@norecursion
def test_compare_medians_ms(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_compare_medians_ms'
    module_type_store = module_type_store.open_function_context('test_compare_medians_ms', 11, 0, False)
    
    # Passed parameters checking function
    test_compare_medians_ms.stypy_localization = localization
    test_compare_medians_ms.stypy_type_of_self = None
    test_compare_medians_ms.stypy_type_store = module_type_store
    test_compare_medians_ms.stypy_function_name = 'test_compare_medians_ms'
    test_compare_medians_ms.stypy_param_names_list = []
    test_compare_medians_ms.stypy_varargs_param_name = None
    test_compare_medians_ms.stypy_kwargs_param_name = None
    test_compare_medians_ms.stypy_call_defaults = defaults
    test_compare_medians_ms.stypy_call_varargs = varargs
    test_compare_medians_ms.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_compare_medians_ms', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_compare_medians_ms', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_compare_medians_ms(...)' code ##################

    
    # Assigning a Call to a Name (line 12):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to arange(...): (line 12)
    # Processing the call arguments (line 12)
    int_670859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'int')
    # Processing the call keyword arguments (line 12)
    kwargs_670860 = {}
    # Getting the type of 'np' (line 12)
    np_670857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 12)
    arange_670858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), np_670857, 'arange')
    # Calling arange(args, kwargs) (line 12)
    arange_call_result_670861 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), arange_670858, *[int_670859], **kwargs_670860)
    
    # Assigning a type to the variable 'x' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'x', arange_call_result_670861)
    
    # Assigning a BinOp to a Name (line 13):
    
    # Assigning a BinOp to a Name (line 13):
    # Getting the type of 'x' (line 13)
    x_670862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'x')
    int_670863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'int')
    # Applying the binary operator '+' (line 13)
    result_add_670864 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 8), '+', x_670862, int_670863)
    
    # Assigning a type to the variable 'y' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'y', result_add_670864)
    
    # Call to assert_almost_equal(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Call to compare_medians_ms(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'x' (line 14)
    x_670868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 46), 'x', False)
    # Getting the type of 'y' (line 14)
    y_670869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 49), 'y', False)
    # Processing the call keyword arguments (line 14)
    kwargs_670870 = {}
    # Getting the type of 'ms' (line 14)
    ms_670866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 24), 'ms', False)
    # Obtaining the member 'compare_medians_ms' of a type (line 14)
    compare_medians_ms_670867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 24), ms_670866, 'compare_medians_ms')
    # Calling compare_medians_ms(args, kwargs) (line 14)
    compare_medians_ms_call_result_670871 = invoke(stypy.reporting.localization.Localization(__file__, 14, 24), compare_medians_ms_670867, *[x_670868, y_670869], **kwargs_670870)
    
    int_670872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 53), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_670873 = {}
    # Getting the type of 'assert_almost_equal' (line 14)
    assert_almost_equal_670865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 14)
    assert_almost_equal_call_result_670874 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), assert_almost_equal_670865, *[compare_medians_ms_call_result_670871, int_670872], **kwargs_670873)
    
    
    # Assigning a Call to a Name (line 16):
    
    # Assigning a Call to a Name (line 16):
    
    # Call to linspace(...): (line 16)
    # Processing the call arguments (line 16)
    int_670877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'int')
    int_670878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'int')
    # Processing the call keyword arguments (line 16)
    int_670879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 31), 'int')
    keyword_670880 = int_670879
    kwargs_670881 = {'num': keyword_670880}
    # Getting the type of 'np' (line 16)
    np_670875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'np', False)
    # Obtaining the member 'linspace' of a type (line 16)
    linspace_670876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 9), np_670875, 'linspace')
    # Calling linspace(args, kwargs) (line 16)
    linspace_call_result_670882 = invoke(stypy.reporting.localization.Localization(__file__, 16, 9), linspace_670876, *[int_670877, int_670878], **kwargs_670881)
    
    # Assigning a type to the variable 'y2' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'y2', linspace_call_result_670882)
    
    # Call to assert_almost_equal(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to compare_medians_ms(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'x' (line 17)
    x_670886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 46), 'x', False)
    # Getting the type of 'y2' (line 17)
    y2_670887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 49), 'y2', False)
    # Processing the call keyword arguments (line 17)
    kwargs_670888 = {}
    # Getting the type of 'ms' (line 17)
    ms_670884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'ms', False)
    # Obtaining the member 'compare_medians_ms' of a type (line 17)
    compare_medians_ms_670885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 24), ms_670884, 'compare_medians_ms')
    # Calling compare_medians_ms(args, kwargs) (line 17)
    compare_medians_ms_call_result_670889 = invoke(stypy.reporting.localization.Localization(__file__, 17, 24), compare_medians_ms_670885, *[x_670886, y2_670887], **kwargs_670888)
    
    float_670890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 54), 'float')
    # Processing the call keyword arguments (line 17)
    kwargs_670891 = {}
    # Getting the type of 'assert_almost_equal' (line 17)
    assert_almost_equal_670883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 17)
    assert_almost_equal_call_result_670892 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), assert_almost_equal_670883, *[compare_medians_ms_call_result_670889, float_670890], **kwargs_670891)
    
    
    # ################# End of 'test_compare_medians_ms(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_compare_medians_ms' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_670893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_670893)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_compare_medians_ms'
    return stypy_return_type_670893

# Assigning a type to the variable 'test_compare_medians_ms' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'test_compare_medians_ms', test_compare_medians_ms)

@norecursion
def test_hdmedian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_hdmedian'
    module_type_store = module_type_store.open_function_context('test_hdmedian', 20, 0, False)
    
    # Passed parameters checking function
    test_hdmedian.stypy_localization = localization
    test_hdmedian.stypy_type_of_self = None
    test_hdmedian.stypy_type_store = module_type_store
    test_hdmedian.stypy_function_name = 'test_hdmedian'
    test_hdmedian.stypy_param_names_list = []
    test_hdmedian.stypy_varargs_param_name = None
    test_hdmedian.stypy_kwargs_param_name = None
    test_hdmedian.stypy_call_defaults = defaults
    test_hdmedian.stypy_call_varargs = varargs
    test_hdmedian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_hdmedian', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_hdmedian', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_hdmedian(...)' code ##################

    
    # Assigning a Call to a Name (line 22):
    
    # Assigning a Call to a Name (line 22):
    
    # Call to arange(...): (line 22)
    # Processing the call arguments (line 22)
    int_670896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'int')
    # Processing the call keyword arguments (line 22)
    kwargs_670897 = {}
    # Getting the type of 'ma' (line 22)
    ma_670894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'ma', False)
    # Obtaining the member 'arange' of a type (line 22)
    arange_670895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), ma_670894, 'arange')
    # Calling arange(args, kwargs) (line 22)
    arange_call_result_670898 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), arange_670895, *[int_670896], **kwargs_670897)
    
    # Assigning a type to the variable 'x' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'x', arange_call_result_670898)
    
    # Call to assert_equal(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to hdmedian(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'x' (line 23)
    x_670902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 29), 'x', False)
    # Processing the call keyword arguments (line 23)
    kwargs_670903 = {}
    # Getting the type of 'ms' (line 23)
    ms_670900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 17), 'ms', False)
    # Obtaining the member 'hdmedian' of a type (line 23)
    hdmedian_670901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 17), ms_670900, 'hdmedian')
    # Calling hdmedian(args, kwargs) (line 23)
    hdmedian_call_result_670904 = invoke(stypy.reporting.localization.Localization(__file__, 23, 17), hdmedian_670901, *[x_670902], **kwargs_670903)
    
    int_670905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 33), 'int')
    # Processing the call keyword arguments (line 23)
    kwargs_670906 = {}
    # Getting the type of 'assert_equal' (line 23)
    assert_equal_670899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 23)
    assert_equal_call_result_670907 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), assert_equal_670899, *[hdmedian_call_result_670904, int_670905], **kwargs_670906)
    
    
    # Assigning a Call to a Attribute (line 24):
    
    # Assigning a Call to a Attribute (line 24):
    
    # Call to make_mask(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'x' (line 24)
    x_670910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 26), 'x', False)
    # Processing the call keyword arguments (line 24)
    kwargs_670911 = {}
    # Getting the type of 'ma' (line 24)
    ma_670908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'ma', False)
    # Obtaining the member 'make_mask' of a type (line 24)
    make_mask_670909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 13), ma_670908, 'make_mask')
    # Calling make_mask(args, kwargs) (line 24)
    make_mask_call_result_670912 = invoke(stypy.reporting.localization.Localization(__file__, 24, 13), make_mask_670909, *[x_670910], **kwargs_670911)
    
    # Getting the type of 'x' (line 24)
    x_670913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'x')
    # Setting the type of the member 'mask' of a type (line 24)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), x_670913, 'mask', make_mask_call_result_670912)
    
    # Assigning a Name to a Subscript (line 25):
    
    # Assigning a Name to a Subscript (line 25):
    # Getting the type of 'False' (line 25)
    False_670914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'False')
    # Getting the type of 'x' (line 25)
    x_670915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'x')
    # Obtaining the member 'mask' of a type (line 25)
    mask_670916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), x_670915, 'mask')
    int_670917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 12), 'int')
    slice_670918 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 4), None, int_670917, None)
    # Storing an element on a container (line 25)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 4), mask_670916, (slice_670918, False_670914))
    
    # Call to assert_equal(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Call to hdmedian(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'x' (line 26)
    x_670922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'x', False)
    # Processing the call keyword arguments (line 26)
    kwargs_670923 = {}
    # Getting the type of 'ms' (line 26)
    ms_670920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'ms', False)
    # Obtaining the member 'hdmedian' of a type (line 26)
    hdmedian_670921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 17), ms_670920, 'hdmedian')
    # Calling hdmedian(args, kwargs) (line 26)
    hdmedian_call_result_670924 = invoke(stypy.reporting.localization.Localization(__file__, 26, 17), hdmedian_670921, *[x_670922], **kwargs_670923)
    
    int_670925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 33), 'int')
    # Processing the call keyword arguments (line 26)
    kwargs_670926 = {}
    # Getting the type of 'assert_equal' (line 26)
    assert_equal_670919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 26)
    assert_equal_call_result_670927 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), assert_equal_670919, *[hdmedian_call_result_670924, int_670925], **kwargs_670926)
    
    
    # Call to assert_(...): (line 30)
    # Processing the call arguments (line 30)
    
    
    # Call to hdmedian(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'x' (line 30)
    x_670931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'x', False)
    # Processing the call keyword arguments (line 30)
    # Getting the type of 'True' (line 30)
    True_670932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 31), 'True', False)
    keyword_670933 = True_670932
    kwargs_670934 = {'var': keyword_670933}
    # Getting the type of 'ms' (line 30)
    ms_670929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'ms', False)
    # Obtaining the member 'hdmedian' of a type (line 30)
    hdmedian_670930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 12), ms_670929, 'hdmedian')
    # Calling hdmedian(args, kwargs) (line 30)
    hdmedian_call_result_670935 = invoke(stypy.reporting.localization.Localization(__file__, 30, 12), hdmedian_670930, *[x_670931], **kwargs_670934)
    
    # Obtaining the member 'size' of a type (line 30)
    size_670936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 12), hdmedian_call_result_670935, 'size')
    int_670937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 45), 'int')
    # Applying the binary operator '==' (line 30)
    result_eq_670938 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 12), '==', size_670936, int_670937)
    
    # Processing the call keyword arguments (line 30)
    kwargs_670939 = {}
    # Getting the type of 'assert_' (line 30)
    assert__670928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 30)
    assert__call_result_670940 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), assert__670928, *[result_eq_670938], **kwargs_670939)
    
    
    # Assigning a Call to a Name (line 33):
    
    # Assigning a Call to a Name (line 33):
    
    # Call to reshape(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Obtaining an instance of the builtin type 'tuple' (line 33)
    tuple_670947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 33)
    # Adding element type (line 33)
    int_670948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 32), tuple_670947, int_670948)
    # Adding element type (line 33)
    int_670949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 32), tuple_670947, int_670949)
    
    # Processing the call keyword arguments (line 33)
    kwargs_670950 = {}
    
    # Call to arange(...): (line 33)
    # Processing the call arguments (line 33)
    int_670943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 19), 'int')
    # Processing the call keyword arguments (line 33)
    kwargs_670944 = {}
    # Getting the type of 'ma' (line 33)
    ma_670941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 9), 'ma', False)
    # Obtaining the member 'arange' of a type (line 33)
    arange_670942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 9), ma_670941, 'arange')
    # Calling arange(args, kwargs) (line 33)
    arange_call_result_670945 = invoke(stypy.reporting.localization.Localization(__file__, 33, 9), arange_670942, *[int_670943], **kwargs_670944)
    
    # Obtaining the member 'reshape' of a type (line 33)
    reshape_670946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 9), arange_call_result_670945, 'reshape')
    # Calling reshape(args, kwargs) (line 33)
    reshape_call_result_670951 = invoke(stypy.reporting.localization.Localization(__file__, 33, 9), reshape_670946, *[tuple_670947], **kwargs_670950)
    
    # Assigning a type to the variable 'x2' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'x2', reshape_call_result_670951)
    
    # Call to assert_allclose(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Call to hdmedian(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'x2' (line 34)
    x2_670955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'x2', False)
    # Processing the call keyword arguments (line 34)
    int_670956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 41), 'int')
    keyword_670957 = int_670956
    kwargs_670958 = {'axis': keyword_670957}
    # Getting the type of 'ms' (line 34)
    ms_670953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'ms', False)
    # Obtaining the member 'hdmedian' of a type (line 34)
    hdmedian_670954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 20), ms_670953, 'hdmedian')
    # Calling hdmedian(args, kwargs) (line 34)
    hdmedian_call_result_670959 = invoke(stypy.reporting.localization.Localization(__file__, 34, 20), hdmedian_670954, *[x2_670955], **kwargs_670958)
    
    
    # Obtaining an instance of the builtin type 'list' (line 34)
    list_670960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 45), 'list')
    # Adding type elements to the builtin type 'list' instance (line 34)
    # Adding element type (line 34)
    int_670961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 45), list_670960, int_670961)
    # Adding element type (line 34)
    int_670962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 45), list_670960, int_670962)
    
    # Processing the call keyword arguments (line 34)
    kwargs_670963 = {}
    # Getting the type of 'assert_allclose' (line 34)
    assert_allclose_670952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 34)
    assert_allclose_call_result_670964 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), assert_allclose_670952, *[hdmedian_call_result_670959, list_670960], **kwargs_670963)
    
    
    # Assigning a Call to a Attribute (line 35):
    
    # Assigning a Call to a Attribute (line 35):
    
    # Call to make_mask(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'x2' (line 35)
    x2_670967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 27), 'x2', False)
    # Processing the call keyword arguments (line 35)
    kwargs_670968 = {}
    # Getting the type of 'ma' (line 35)
    ma_670965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'ma', False)
    # Obtaining the member 'make_mask' of a type (line 35)
    make_mask_670966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 14), ma_670965, 'make_mask')
    # Calling make_mask(args, kwargs) (line 35)
    make_mask_call_result_670969 = invoke(stypy.reporting.localization.Localization(__file__, 35, 14), make_mask_670966, *[x2_670967], **kwargs_670968)
    
    # Getting the type of 'x2' (line 35)
    x2_670970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'x2')
    # Setting the type of the member 'mask' of a type (line 35)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), x2_670970, 'mask', make_mask_call_result_670969)
    
    # Assigning a Name to a Subscript (line 36):
    
    # Assigning a Name to a Subscript (line 36):
    # Getting the type of 'False' (line 36)
    False_670971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'False')
    # Getting the type of 'x2' (line 36)
    x2_670972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'x2')
    # Obtaining the member 'mask' of a type (line 36)
    mask_670973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 4), x2_670972, 'mask')
    int_670974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'int')
    slice_670975 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 36, 4), None, int_670974, None)
    slice_670976 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 36, 4), None, None, None)
    # Storing an element on a container (line 36)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 4), mask_670973, ((slice_670975, slice_670976), False_670971))
    
    # Call to assert_allclose(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Call to hdmedian(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'x2' (line 37)
    x2_670980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 32), 'x2', False)
    # Processing the call keyword arguments (line 37)
    int_670981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 41), 'int')
    keyword_670982 = int_670981
    kwargs_670983 = {'axis': keyword_670982}
    # Getting the type of 'ms' (line 37)
    ms_670978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'ms', False)
    # Obtaining the member 'hdmedian' of a type (line 37)
    hdmedian_670979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 20), ms_670978, 'hdmedian')
    # Calling hdmedian(args, kwargs) (line 37)
    hdmedian_call_result_670984 = invoke(stypy.reporting.localization.Localization(__file__, 37, 20), hdmedian_670979, *[x2_670980], **kwargs_670983)
    
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_670985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 45), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    int_670986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 45), list_670985, int_670986)
    # Adding element type (line 37)
    int_670987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 45), list_670985, int_670987)
    
    # Processing the call keyword arguments (line 37)
    kwargs_670988 = {}
    # Getting the type of 'assert_allclose' (line 37)
    assert_allclose_670977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 37)
    assert_allclose_call_result_670989 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), assert_allclose_670977, *[hdmedian_call_result_670984, list_670985], **kwargs_670988)
    
    
    # ################# End of 'test_hdmedian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_hdmedian' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_670990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_670990)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_hdmedian'
    return stypy_return_type_670990

# Assigning a type to the variable 'test_hdmedian' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'test_hdmedian', test_hdmedian)

@norecursion
def test_rsh(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_rsh'
    module_type_store = module_type_store.open_function_context('test_rsh', 40, 0, False)
    
    # Passed parameters checking function
    test_rsh.stypy_localization = localization
    test_rsh.stypy_type_of_self = None
    test_rsh.stypy_type_store = module_type_store
    test_rsh.stypy_function_name = 'test_rsh'
    test_rsh.stypy_param_names_list = []
    test_rsh.stypy_varargs_param_name = None
    test_rsh.stypy_kwargs_param_name = None
    test_rsh.stypy_call_defaults = defaults
    test_rsh.stypy_call_varargs = varargs
    test_rsh.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_rsh', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_rsh', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_rsh(...)' code ##################

    
    # Call to seed(...): (line 41)
    # Processing the call arguments (line 41)
    int_670994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 19), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_670995 = {}
    # Getting the type of 'np' (line 41)
    np_670991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 41)
    random_670992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), np_670991, 'random')
    # Obtaining the member 'seed' of a type (line 41)
    seed_670993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), random_670992, 'seed')
    # Calling seed(args, kwargs) (line 41)
    seed_call_result_670996 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), seed_670993, *[int_670994], **kwargs_670995)
    
    
    # Assigning a Call to a Name (line 42):
    
    # Assigning a Call to a Name (line 42):
    
    # Call to randn(...): (line 42)
    # Processing the call arguments (line 42)
    int_671000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 24), 'int')
    # Processing the call keyword arguments (line 42)
    kwargs_671001 = {}
    # Getting the type of 'np' (line 42)
    np_670997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 42)
    random_670998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), np_670997, 'random')
    # Obtaining the member 'randn' of a type (line 42)
    randn_670999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), random_670998, 'randn')
    # Calling randn(args, kwargs) (line 42)
    randn_call_result_671002 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), randn_670999, *[int_671000], **kwargs_671001)
    
    # Assigning a type to the variable 'x' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'x', randn_call_result_671002)
    
    # Assigning a Call to a Name (line 43):
    
    # Assigning a Call to a Name (line 43):
    
    # Call to rsh(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'x' (line 43)
    x_671005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'x', False)
    # Processing the call keyword arguments (line 43)
    kwargs_671006 = {}
    # Getting the type of 'ms' (line 43)
    ms_671003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'ms', False)
    # Obtaining the member 'rsh' of a type (line 43)
    rsh_671004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 10), ms_671003, 'rsh')
    # Calling rsh(args, kwargs) (line 43)
    rsh_call_result_671007 = invoke(stypy.reporting.localization.Localization(__file__, 43, 10), rsh_671004, *[x_671005], **kwargs_671006)
    
    # Assigning a type to the variable 'res' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'res', rsh_call_result_671007)
    
    # Call to assert_(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Getting the type of 'res' (line 46)
    res_671009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'res', False)
    # Obtaining the member 'shape' of a type (line 46)
    shape_671010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), res_671009, 'shape')
    # Getting the type of 'x' (line 46)
    x_671011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'x', False)
    # Obtaining the member 'shape' of a type (line 46)
    shape_671012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 25), x_671011, 'shape')
    # Applying the binary operator '==' (line 46)
    result_eq_671013 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 12), '==', shape_671010, shape_671012)
    
    # Processing the call keyword arguments (line 46)
    kwargs_671014 = {}
    # Getting the type of 'assert_' (line 46)
    assert__671008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 46)
    assert__call_result_671015 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), assert__671008, *[result_eq_671013], **kwargs_671014)
    
    
    # Assigning a Call to a Name (line 49):
    
    # Assigning a Call to a Name (line 49):
    
    # Call to rsh(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'x' (line 49)
    x_671018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'x', False)
    # Processing the call keyword arguments (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 49)
    list_671019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 49)
    # Adding element type (line 49)
    int_671020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 27), list_671019, int_671020)
    # Adding element type (line 49)
    float_671021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 27), list_671019, float_671021)
    
    keyword_671022 = list_671019
    kwargs_671023 = {'points': keyword_671022}
    # Getting the type of 'ms' (line 49)
    ms_671016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 10), 'ms', False)
    # Obtaining the member 'rsh' of a type (line 49)
    rsh_671017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 10), ms_671016, 'rsh')
    # Calling rsh(args, kwargs) (line 49)
    rsh_call_result_671024 = invoke(stypy.reporting.localization.Localization(__file__, 49, 10), rsh_671017, *[x_671018], **kwargs_671023)
    
    # Assigning a type to the variable 'res' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'res', rsh_call_result_671024)
    
    # Call to assert_(...): (line 50)
    # Processing the call arguments (line 50)
    
    # Getting the type of 'res' (line 50)
    res_671026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'res', False)
    # Obtaining the member 'size' of a type (line 50)
    size_671027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), res_671026, 'size')
    int_671028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 24), 'int')
    # Applying the binary operator '==' (line 50)
    result_eq_671029 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 12), '==', size_671027, int_671028)
    
    # Processing the call keyword arguments (line 50)
    kwargs_671030 = {}
    # Getting the type of 'assert_' (line 50)
    assert__671025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 50)
    assert__call_result_671031 = invoke(stypy.reporting.localization.Localization(__file__, 50, 4), assert__671025, *[result_eq_671029], **kwargs_671030)
    
    
    # ################# End of 'test_rsh(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_rsh' in the type store
    # Getting the type of 'stypy_return_type' (line 40)
    stypy_return_type_671032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_671032)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_rsh'
    return stypy_return_type_671032

# Assigning a type to the variable 'test_rsh' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'test_rsh', test_rsh)

@norecursion
def test_mjci(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_mjci'
    module_type_store = module_type_store.open_function_context('test_mjci', 53, 0, False)
    
    # Passed parameters checking function
    test_mjci.stypy_localization = localization
    test_mjci.stypy_type_of_self = None
    test_mjci.stypy_type_store = module_type_store
    test_mjci.stypy_function_name = 'test_mjci'
    test_mjci.stypy_param_names_list = []
    test_mjci.stypy_varargs_param_name = None
    test_mjci.stypy_kwargs_param_name = None
    test_mjci.stypy_call_defaults = defaults
    test_mjci.stypy_call_varargs = varargs
    test_mjci.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_mjci', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_mjci', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_mjci(...)' code ##################

    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to array(...): (line 55)
    # Processing the call arguments (line 55)
    
    # Obtaining an instance of the builtin type 'list' (line 55)
    list_671035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 55)
    # Adding element type (line 55)
    int_671036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671036)
    # Adding element type (line 55)
    int_671037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671037)
    # Adding element type (line 55)
    int_671038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671038)
    # Adding element type (line 55)
    int_671039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671039)
    # Adding element type (line 55)
    int_671040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671040)
    # Adding element type (line 55)
    int_671041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671041)
    # Adding element type (line 55)
    int_671042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671042)
    # Adding element type (line 55)
    int_671043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671043)
    # Adding element type (line 55)
    int_671044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671044)
    # Adding element type (line 55)
    int_671045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 56), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671045)
    # Adding element type (line 55)
    int_671046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671046)
    # Adding element type (line 55)
    int_671047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671047)
    # Adding element type (line 55)
    int_671048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671048)
    # Adding element type (line 55)
    int_671049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671049)
    # Adding element type (line 55)
    int_671050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671050)
    # Adding element type (line 55)
    int_671051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671051)
    # Adding element type (line 55)
    int_671052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671052)
    # Adding element type (line 55)
    int_671053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671053)
    # Adding element type (line 55)
    int_671054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_671035, int_671054)
    
    # Processing the call keyword arguments (line 55)
    kwargs_671055 = {}
    # Getting the type of 'ma' (line 55)
    ma_671033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'ma', False)
    # Obtaining the member 'array' of a type (line 55)
    array_671034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 11), ma_671033, 'array')
    # Calling array(args, kwargs) (line 55)
    array_call_result_671056 = invoke(stypy.reporting.localization.Localization(__file__, 55, 11), array_671034, *[list_671035], **kwargs_671055)
    
    # Assigning a type to the variable 'data' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'data', array_call_result_671056)
    
    # Call to assert_almost_equal(...): (line 57)
    # Processing the call arguments (line 57)
    
    # Call to mjci(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'data' (line 57)
    data_671060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 32), 'data', False)
    # Processing the call keyword arguments (line 57)
    kwargs_671061 = {}
    # Getting the type of 'ms' (line 57)
    ms_671058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'ms', False)
    # Obtaining the member 'mjci' of a type (line 57)
    mjci_671059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 24), ms_671058, 'mjci')
    # Calling mjci(args, kwargs) (line 57)
    mjci_call_result_671062 = invoke(stypy.reporting.localization.Localization(__file__, 57, 24), mjci_671059, *[data_671060], **kwargs_671061)
    
    
    # Obtaining an instance of the builtin type 'list' (line 57)
    list_671063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 57)
    # Adding element type (line 57)
    float_671064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 38), list_671063, float_671064)
    # Adding element type (line 57)
    float_671065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 48), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 38), list_671063, float_671065)
    # Adding element type (line 57)
    float_671066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 57), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 38), list_671063, float_671066)
    
    int_671067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 68), 'int')
    # Processing the call keyword arguments (line 57)
    kwargs_671068 = {}
    # Getting the type of 'assert_almost_equal' (line 57)
    assert_almost_equal_671057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 57)
    assert_almost_equal_call_result_671069 = invoke(stypy.reporting.localization.Localization(__file__, 57, 4), assert_almost_equal_671057, *[mjci_call_result_671062, list_671063, int_671067], **kwargs_671068)
    
    
    # ################# End of 'test_mjci(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_mjci' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_671070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_671070)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_mjci'
    return stypy_return_type_671070

# Assigning a type to the variable 'test_mjci' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'test_mjci', test_mjci)

@norecursion
def test_trimmed_mean_ci(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_trimmed_mean_ci'
    module_type_store = module_type_store.open_function_context('test_trimmed_mean_ci', 60, 0, False)
    
    # Passed parameters checking function
    test_trimmed_mean_ci.stypy_localization = localization
    test_trimmed_mean_ci.stypy_type_of_self = None
    test_trimmed_mean_ci.stypy_type_store = module_type_store
    test_trimmed_mean_ci.stypy_function_name = 'test_trimmed_mean_ci'
    test_trimmed_mean_ci.stypy_param_names_list = []
    test_trimmed_mean_ci.stypy_varargs_param_name = None
    test_trimmed_mean_ci.stypy_kwargs_param_name = None
    test_trimmed_mean_ci.stypy_call_defaults = defaults
    test_trimmed_mean_ci.stypy_call_varargs = varargs
    test_trimmed_mean_ci.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_trimmed_mean_ci', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_trimmed_mean_ci', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_trimmed_mean_ci(...)' code ##################

    
    # Assigning a Call to a Name (line 62):
    
    # Assigning a Call to a Name (line 62):
    
    # Call to array(...): (line 62)
    # Processing the call arguments (line 62)
    
    # Obtaining an instance of the builtin type 'list' (line 62)
    list_671073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 62)
    # Adding element type (line 62)
    int_671074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671074)
    # Adding element type (line 62)
    int_671075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671075)
    # Adding element type (line 62)
    int_671076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671076)
    # Adding element type (line 62)
    int_671077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671077)
    # Adding element type (line 62)
    int_671078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671078)
    # Adding element type (line 62)
    int_671079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671079)
    # Adding element type (line 62)
    int_671080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671080)
    # Adding element type (line 62)
    int_671081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671081)
    # Adding element type (line 62)
    int_671082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671082)
    # Adding element type (line 62)
    int_671083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671083)
    # Adding element type (line 62)
    int_671084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671084)
    # Adding element type (line 62)
    int_671085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671085)
    # Adding element type (line 62)
    int_671086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671086)
    # Adding element type (line 62)
    int_671087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671087)
    # Adding element type (line 62)
    int_671088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_671073, int_671088)
    
    # Processing the call keyword arguments (line 62)
    kwargs_671089 = {}
    # Getting the type of 'ma' (line 62)
    ma_671071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'ma', False)
    # Obtaining the member 'array' of a type (line 62)
    array_671072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 11), ma_671071, 'array')
    # Calling array(args, kwargs) (line 62)
    array_call_result_671090 = invoke(stypy.reporting.localization.Localization(__file__, 62, 11), array_671072, *[list_671073], **kwargs_671089)
    
    # Assigning a type to the variable 'data' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'data', array_call_result_671090)
    
    # Call to assert_almost_equal(...): (line 64)
    # Processing the call arguments (line 64)
    
    # Call to trimmed_mean(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'data' (line 64)
    data_671094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 40), 'data', False)
    float_671095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 45), 'float')
    # Processing the call keyword arguments (line 64)
    kwargs_671096 = {}
    # Getting the type of 'ms' (line 64)
    ms_671092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'ms', False)
    # Obtaining the member 'trimmed_mean' of a type (line 64)
    trimmed_mean_671093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 24), ms_671092, 'trimmed_mean')
    # Calling trimmed_mean(args, kwargs) (line 64)
    trimmed_mean_call_result_671097 = invoke(stypy.reporting.localization.Localization(__file__, 64, 24), trimmed_mean_671093, *[data_671094, float_671095], **kwargs_671096)
    
    float_671098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 51), 'float')
    int_671099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 58), 'int')
    # Processing the call keyword arguments (line 64)
    kwargs_671100 = {}
    # Getting the type of 'assert_almost_equal' (line 64)
    assert_almost_equal_671091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 64)
    assert_almost_equal_call_result_671101 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), assert_almost_equal_671091, *[trimmed_mean_call_result_671097, float_671098, int_671099], **kwargs_671100)
    
    
    # Call to assert_equal(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Call to round(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Call to trimmed_mean_ci(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'data' (line 65)
    data_671107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 45), 'data', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 65)
    tuple_671108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 65)
    # Adding element type (line 65)
    float_671109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 51), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 51), tuple_671108, float_671109)
    # Adding element type (line 65)
    float_671110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 55), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 51), tuple_671108, float_671110)
    
    # Processing the call keyword arguments (line 65)
    kwargs_671111 = {}
    # Getting the type of 'ms' (line 65)
    ms_671105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'ms', False)
    # Obtaining the member 'trimmed_mean_ci' of a type (line 65)
    trimmed_mean_ci_671106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 26), ms_671105, 'trimmed_mean_ci')
    # Calling trimmed_mean_ci(args, kwargs) (line 65)
    trimmed_mean_ci_call_result_671112 = invoke(stypy.reporting.localization.Localization(__file__, 65, 26), trimmed_mean_ci_671106, *[data_671107, tuple_671108], **kwargs_671111)
    
    int_671113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 61), 'int')
    # Processing the call keyword arguments (line 65)
    kwargs_671114 = {}
    # Getting the type of 'np' (line 65)
    np_671103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 17), 'np', False)
    # Obtaining the member 'round' of a type (line 65)
    round_671104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 17), np_671103, 'round')
    # Calling round(args, kwargs) (line 65)
    round_call_result_671115 = invoke(stypy.reporting.localization.Localization(__file__, 65, 17), round_671104, *[trimmed_mean_ci_call_result_671112, int_671113], **kwargs_671114)
    
    
    # Obtaining an instance of the builtin type 'list' (line 66)
    list_671116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 66)
    # Adding element type (line 66)
    float_671117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 17), list_671116, float_671117)
    # Adding element type (line 66)
    float_671118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 17), list_671116, float_671118)
    
    # Processing the call keyword arguments (line 65)
    kwargs_671119 = {}
    # Getting the type of 'assert_equal' (line 65)
    assert_equal_671102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 65)
    assert_equal_call_result_671120 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), assert_equal_671102, *[round_call_result_671115, list_671116], **kwargs_671119)
    
    
    # ################# End of 'test_trimmed_mean_ci(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_trimmed_mean_ci' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_671121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_671121)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_trimmed_mean_ci'
    return stypy_return_type_671121

# Assigning a type to the variable 'test_trimmed_mean_ci' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'test_trimmed_mean_ci', test_trimmed_mean_ci)

@norecursion
def test_idealfourths(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_idealfourths'
    module_type_store = module_type_store.open_function_context('test_idealfourths', 69, 0, False)
    
    # Passed parameters checking function
    test_idealfourths.stypy_localization = localization
    test_idealfourths.stypy_type_of_self = None
    test_idealfourths.stypy_type_store = module_type_store
    test_idealfourths.stypy_function_name = 'test_idealfourths'
    test_idealfourths.stypy_param_names_list = []
    test_idealfourths.stypy_varargs_param_name = None
    test_idealfourths.stypy_kwargs_param_name = None
    test_idealfourths.stypy_call_defaults = defaults
    test_idealfourths.stypy_call_varargs = varargs
    test_idealfourths.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_idealfourths', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_idealfourths', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_idealfourths(...)' code ##################

    
    # Assigning a Call to a Name (line 71):
    
    # Assigning a Call to a Name (line 71):
    
    # Call to arange(...): (line 71)
    # Processing the call arguments (line 71)
    int_671124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'int')
    # Processing the call keyword arguments (line 71)
    kwargs_671125 = {}
    # Getting the type of 'np' (line 71)
    np_671122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 71)
    arange_671123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 11), np_671122, 'arange')
    # Calling arange(args, kwargs) (line 71)
    arange_call_result_671126 = invoke(stypy.reporting.localization.Localization(__file__, 71, 11), arange_671123, *[int_671124], **kwargs_671125)
    
    # Assigning a type to the variable 'test' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'test', arange_call_result_671126)
    
    # Call to assert_almost_equal(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Call to asarray(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Call to idealfourths(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'test' (line 72)
    test_671132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 51), 'test', False)
    # Processing the call keyword arguments (line 72)
    kwargs_671133 = {}
    # Getting the type of 'ms' (line 72)
    ms_671130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 35), 'ms', False)
    # Obtaining the member 'idealfourths' of a type (line 72)
    idealfourths_671131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 35), ms_671130, 'idealfourths')
    # Calling idealfourths(args, kwargs) (line 72)
    idealfourths_call_result_671134 = invoke(stypy.reporting.localization.Localization(__file__, 72, 35), idealfourths_671131, *[test_671132], **kwargs_671133)
    
    # Processing the call keyword arguments (line 72)
    kwargs_671135 = {}
    # Getting the type of 'np' (line 72)
    np_671128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'np', False)
    # Obtaining the member 'asarray' of a type (line 72)
    asarray_671129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 24), np_671128, 'asarray')
    # Calling asarray(args, kwargs) (line 72)
    asarray_call_result_671136 = invoke(stypy.reporting.localization.Localization(__file__, 72, 24), asarray_671129, *[idealfourths_call_result_671134], **kwargs_671135)
    
    
    # Obtaining an instance of the builtin type 'list' (line 73)
    list_671137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 73)
    # Adding element type (line 73)
    float_671138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 24), list_671137, float_671138)
    # Adding element type (line 73)
    float_671139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 24), list_671137, float_671139)
    
    int_671140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 46), 'int')
    # Processing the call keyword arguments (line 72)
    kwargs_671141 = {}
    # Getting the type of 'assert_almost_equal' (line 72)
    assert_almost_equal_671127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 72)
    assert_almost_equal_call_result_671142 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), assert_almost_equal_671127, *[asarray_call_result_671136, list_671137, int_671140], **kwargs_671141)
    
    
    # Assigning a Call to a Name (line 74):
    
    # Assigning a Call to a Name (line 74):
    
    # Call to reshape(...): (line 74)
    # Processing the call arguments (line 74)
    int_671149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 37), 'int')
    int_671150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 40), 'int')
    # Processing the call keyword arguments (line 74)
    kwargs_671151 = {}
    
    # Call to repeat(...): (line 74)
    # Processing the call arguments (line 74)
    int_671145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 26), 'int')
    # Processing the call keyword arguments (line 74)
    kwargs_671146 = {}
    # Getting the type of 'test' (line 74)
    test_671143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 14), 'test', False)
    # Obtaining the member 'repeat' of a type (line 74)
    repeat_671144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 14), test_671143, 'repeat')
    # Calling repeat(args, kwargs) (line 74)
    repeat_call_result_671147 = invoke(stypy.reporting.localization.Localization(__file__, 74, 14), repeat_671144, *[int_671145], **kwargs_671146)
    
    # Obtaining the member 'reshape' of a type (line 74)
    reshape_671148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 14), repeat_call_result_671147, 'reshape')
    # Calling reshape(args, kwargs) (line 74)
    reshape_call_result_671152 = invoke(stypy.reporting.localization.Localization(__file__, 74, 14), reshape_671148, *[int_671149, int_671150], **kwargs_671151)
    
    # Assigning a type to the variable 'test_2D' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'test_2D', reshape_call_result_671152)
    
    # Call to assert_almost_equal(...): (line 75)
    # Processing the call arguments (line 75)
    
    # Call to idealfourths(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'test_2D' (line 75)
    test_2D_671156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 40), 'test_2D', False)
    # Processing the call keyword arguments (line 75)
    int_671157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 54), 'int')
    keyword_671158 = int_671157
    kwargs_671159 = {'axis': keyword_671158}
    # Getting the type of 'ms' (line 75)
    ms_671154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'ms', False)
    # Obtaining the member 'idealfourths' of a type (line 75)
    idealfourths_671155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 24), ms_671154, 'idealfourths')
    # Calling idealfourths(args, kwargs) (line 75)
    idealfourths_call_result_671160 = invoke(stypy.reporting.localization.Localization(__file__, 75, 24), idealfourths_671155, *[test_2D_671156], **kwargs_671159)
    
    
    # Obtaining an instance of the builtin type 'list' (line 76)
    list_671161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 76)
    # Adding element type (line 76)
    
    # Obtaining an instance of the builtin type 'list' (line 76)
    list_671162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 76)
    # Adding element type (line 76)
    float_671163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 25), list_671162, float_671163)
    # Adding element type (line 76)
    float_671164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 36), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 25), list_671162, float_671164)
    # Adding element type (line 76)
    float_671165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 46), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 25), list_671162, float_671165)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 24), list_671161, list_671162)
    # Adding element type (line 76)
    
    # Obtaining an instance of the builtin type 'list' (line 77)
    list_671166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 77)
    # Adding element type (line 77)
    float_671167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 25), list_671166, float_671167)
    # Adding element type (line 77)
    float_671168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 36), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 25), list_671166, float_671168)
    # Adding element type (line 77)
    float_671169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 46), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 25), list_671166, float_671169)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 24), list_671161, list_671166)
    
    int_671170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 58), 'int')
    # Processing the call keyword arguments (line 75)
    kwargs_671171 = {}
    # Getting the type of 'assert_almost_equal' (line 75)
    assert_almost_equal_671153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 75)
    assert_almost_equal_call_result_671172 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), assert_almost_equal_671153, *[idealfourths_call_result_671160, list_671161, int_671170], **kwargs_671171)
    
    
    # Call to assert_almost_equal(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Call to idealfourths(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'test_2D' (line 78)
    test_2D_671176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'test_2D', False)
    # Processing the call keyword arguments (line 78)
    int_671177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 54), 'int')
    keyword_671178 = int_671177
    kwargs_671179 = {'axis': keyword_671178}
    # Getting the type of 'ms' (line 78)
    ms_671174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'ms', False)
    # Obtaining the member 'idealfourths' of a type (line 78)
    idealfourths_671175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 24), ms_671174, 'idealfourths')
    # Calling idealfourths(args, kwargs) (line 78)
    idealfourths_call_result_671180 = invoke(stypy.reporting.localization.Localization(__file__, 78, 24), idealfourths_671175, *[test_2D_671176], **kwargs_671179)
    
    
    # Call to reshape(...): (line 79)
    # Processing the call arguments (line 79)
    int_671187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 47), 'int')
    int_671188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 50), 'int')
    # Processing the call keyword arguments (line 79)
    kwargs_671189 = {}
    
    # Call to repeat(...): (line 79)
    # Processing the call arguments (line 79)
    int_671183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 36), 'int')
    # Processing the call keyword arguments (line 79)
    kwargs_671184 = {}
    # Getting the type of 'test' (line 79)
    test_671181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'test', False)
    # Obtaining the member 'repeat' of a type (line 79)
    repeat_671182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 24), test_671181, 'repeat')
    # Calling repeat(args, kwargs) (line 79)
    repeat_call_result_671185 = invoke(stypy.reporting.localization.Localization(__file__, 79, 24), repeat_671182, *[int_671183], **kwargs_671184)
    
    # Obtaining the member 'reshape' of a type (line 79)
    reshape_671186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 24), repeat_call_result_671185, 'reshape')
    # Calling reshape(args, kwargs) (line 79)
    reshape_call_result_671190 = invoke(stypy.reporting.localization.Localization(__file__, 79, 24), reshape_671186, *[int_671187, int_671188], **kwargs_671189)
    
    # Processing the call keyword arguments (line 78)
    kwargs_671191 = {}
    # Getting the type of 'assert_almost_equal' (line 78)
    assert_almost_equal_671173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 78)
    assert_almost_equal_call_result_671192 = invoke(stypy.reporting.localization.Localization(__file__, 78, 4), assert_almost_equal_671173, *[idealfourths_call_result_671180, reshape_call_result_671190], **kwargs_671191)
    
    
    # Assigning a List to a Name (line 80):
    
    # Assigning a List to a Name (line 80):
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_671193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    # Adding element type (line 80)
    int_671194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 11), list_671193, int_671194)
    # Adding element type (line 80)
    int_671195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 11), list_671193, int_671195)
    
    # Assigning a type to the variable 'test' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'test', list_671193)
    
    # Assigning a Call to a Name (line 81):
    
    # Assigning a Call to a Name (line 81):
    
    # Call to idealfourths(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'test' (line 81)
    test_671198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'test', False)
    # Processing the call keyword arguments (line 81)
    kwargs_671199 = {}
    # Getting the type of 'ms' (line 81)
    ms_671196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 14), 'ms', False)
    # Obtaining the member 'idealfourths' of a type (line 81)
    idealfourths_671197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 14), ms_671196, 'idealfourths')
    # Calling idealfourths(args, kwargs) (line 81)
    idealfourths_call_result_671200 = invoke(stypy.reporting.localization.Localization(__file__, 81, 14), idealfourths_671197, *[test_671198], **kwargs_671199)
    
    # Assigning a type to the variable '_result' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), '_result', idealfourths_call_result_671200)
    
    # Call to assert_(...): (line 82)
    # Processing the call arguments (line 82)
    
    # Call to all(...): (line 82)
    # Processing the call keyword arguments (line 82)
    kwargs_671208 = {}
    
    # Call to isnan(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of '_result' (line 82)
    _result_671204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), '_result', False)
    # Processing the call keyword arguments (line 82)
    kwargs_671205 = {}
    # Getting the type of 'np' (line 82)
    np_671202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'np', False)
    # Obtaining the member 'isnan' of a type (line 82)
    isnan_671203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), np_671202, 'isnan')
    # Calling isnan(args, kwargs) (line 82)
    isnan_call_result_671206 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), isnan_671203, *[_result_671204], **kwargs_671205)
    
    # Obtaining the member 'all' of a type (line 82)
    all_671207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), isnan_call_result_671206, 'all')
    # Calling all(args, kwargs) (line 82)
    all_call_result_671209 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), all_671207, *[], **kwargs_671208)
    
    # Processing the call keyword arguments (line 82)
    kwargs_671210 = {}
    # Getting the type of 'assert_' (line 82)
    assert__671201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 82)
    assert__call_result_671211 = invoke(stypy.reporting.localization.Localization(__file__, 82, 4), assert__671201, *[all_call_result_671209], **kwargs_671210)
    
    
    # ################# End of 'test_idealfourths(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_idealfourths' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_671212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_671212)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_idealfourths'
    return stypy_return_type_671212

# Assigning a type to the variable 'test_idealfourths' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'test_idealfourths', test_idealfourths)
# Declaration of the 'TestQuantiles' class

class TestQuantiles(object, ):
    
    # Assigning a List to a Name (line 86):

    @norecursion
    def test_hdquantiles(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hdquantiles'
        module_type_store = module_type_store.open_function_context('test_hdquantiles', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuantiles.test_hdquantiles.__dict__.__setitem__('stypy_localization', localization)
        TestQuantiles.test_hdquantiles.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuantiles.test_hdquantiles.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuantiles.test_hdquantiles.__dict__.__setitem__('stypy_function_name', 'TestQuantiles.test_hdquantiles')
        TestQuantiles.test_hdquantiles.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuantiles.test_hdquantiles.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuantiles.test_hdquantiles.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuantiles.test_hdquantiles.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuantiles.test_hdquantiles.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuantiles.test_hdquantiles.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuantiles.test_hdquantiles.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuantiles.test_hdquantiles', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hdquantiles', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hdquantiles(...)' code ##################

        
        # Assigning a Attribute to a Name (line 108):
        
        # Assigning a Attribute to a Name (line 108):
        # Getting the type of 'self' (line 108)
        self_671213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'self')
        # Obtaining the member 'data' of a type (line 108)
        data_671214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), self_671213, 'data')
        # Assigning a type to the variable 'data' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'data', data_671214)
        
        # Call to assert_almost_equal(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Call to hdquantiles(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'data' (line 109)
        data_671218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 43), 'data', False)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_671219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        float_671220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 48), list_671219, float_671220)
        # Adding element type (line 109)
        float_671221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 48), list_671219, float_671221)
        
        # Processing the call keyword arguments (line 109)
        kwargs_671222 = {}
        # Getting the type of 'ms' (line 109)
        ms_671216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'ms', False)
        # Obtaining the member 'hdquantiles' of a type (line 109)
        hdquantiles_671217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 28), ms_671216, 'hdquantiles')
        # Calling hdquantiles(args, kwargs) (line 109)
        hdquantiles_call_result_671223 = invoke(stypy.reporting.localization.Localization(__file__, 109, 28), hdquantiles_671217, *[data_671218, list_671219], **kwargs_671222)
        
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_671224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        float_671225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 28), list_671224, float_671225)
        # Adding element type (line 110)
        float_671226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 28), list_671224, float_671226)
        
        # Processing the call keyword arguments (line 109)
        kwargs_671227 = {}
        # Getting the type of 'assert_almost_equal' (line 109)
        assert_almost_equal_671215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 109)
        assert_almost_equal_call_result_671228 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), assert_almost_equal_671215, *[hdquantiles_call_result_671223, list_671224], **kwargs_671227)
        
        
        # Assigning a Call to a Name (line 111):
        
        # Assigning a Call to a Name (line 111):
        
        # Call to hdquantiles(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'data' (line 111)
        data_671231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 29), 'data', False)
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_671232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        float_671233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 34), list_671232, float_671233)
        # Adding element type (line 111)
        float_671234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 34), list_671232, float_671234)
        # Adding element type (line 111)
        float_671235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 34), list_671232, float_671235)
        
        # Processing the call keyword arguments (line 111)
        kwargs_671236 = {}
        # Getting the type of 'ms' (line 111)
        ms_671229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 14), 'ms', False)
        # Obtaining the member 'hdquantiles' of a type (line 111)
        hdquantiles_671230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 14), ms_671229, 'hdquantiles')
        # Calling hdquantiles(args, kwargs) (line 111)
        hdquantiles_call_result_671237 = invoke(stypy.reporting.localization.Localization(__file__, 111, 14), hdquantiles_671230, *[data_671231, list_671232], **kwargs_671236)
        
        # Assigning a type to the variable 'hdq' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'hdq', hdquantiles_call_result_671237)
        
        # Call to assert_almost_equal(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'hdq' (line 112)
        hdq_671239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'hdq', False)
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_671240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        float_671241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 33), list_671240, float_671241)
        # Adding element type (line 112)
        float_671242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 33), list_671240, float_671242)
        # Adding element type (line 112)
        float_671243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 33), list_671240, float_671243)
        
        # Processing the call keyword arguments (line 112)
        kwargs_671244 = {}
        # Getting the type of 'assert_almost_equal' (line 112)
        assert_almost_equal_671238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 112)
        assert_almost_equal_call_result_671245 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), assert_almost_equal_671238, *[hdq_671239, list_671240], **kwargs_671244)
        
        
        # Assigning a Call to a Name (line 113):
        
        # Assigning a Call to a Name (line 113):
        
        # Call to hdquantiles_sd(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'data' (line 113)
        data_671248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 32), 'data', False)
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_671249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        float_671250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 37), list_671249, float_671250)
        # Adding element type (line 113)
        float_671251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 37), list_671249, float_671251)
        # Adding element type (line 113)
        float_671252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 37), list_671249, float_671252)
        
        # Processing the call keyword arguments (line 113)
        kwargs_671253 = {}
        # Getting the type of 'ms' (line 113)
        ms_671246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 14), 'ms', False)
        # Obtaining the member 'hdquantiles_sd' of a type (line 113)
        hdquantiles_sd_671247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 14), ms_671246, 'hdquantiles_sd')
        # Calling hdquantiles_sd(args, kwargs) (line 113)
        hdquantiles_sd_call_result_671254 = invoke(stypy.reporting.localization.Localization(__file__, 113, 14), hdquantiles_sd_671247, *[data_671248, list_671249], **kwargs_671253)
        
        # Assigning a type to the variable 'hdq' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'hdq', hdquantiles_sd_call_result_671254)
        
        # Call to assert_almost_equal(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'hdq' (line 114)
        hdq_671256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'hdq', False)
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_671257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        # Adding element type (line 114)
        float_671258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 33), list_671257, float_671258)
        # Adding element type (line 114)
        float_671259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 33), list_671257, float_671259)
        # Adding element type (line 114)
        float_671260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 33), list_671257, float_671260)
        
        int_671261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 72), 'int')
        # Processing the call keyword arguments (line 114)
        kwargs_671262 = {}
        # Getting the type of 'assert_almost_equal' (line 114)
        assert_almost_equal_671255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 114)
        assert_almost_equal_call_result_671263 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), assert_almost_equal_671255, *[hdq_671256, list_671257, int_671261], **kwargs_671262)
        
        
        # Assigning a Call to a Name (line 116):
        
        # Assigning a Call to a Name (line 116):
        
        # Call to reshape(...): (line 116)
        # Processing the call arguments (line 116)
        int_671270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 38), 'int')
        int_671271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 41), 'int')
        # Processing the call keyword arguments (line 116)
        kwargs_671272 = {}
        
        # Call to array(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'data' (line 116)
        data_671266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'data', False)
        # Processing the call keyword arguments (line 116)
        kwargs_671267 = {}
        # Getting the type of 'np' (line 116)
        np_671264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 116)
        array_671265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 15), np_671264, 'array')
        # Calling array(args, kwargs) (line 116)
        array_call_result_671268 = invoke(stypy.reporting.localization.Localization(__file__, 116, 15), array_671265, *[data_671266], **kwargs_671267)
        
        # Obtaining the member 'reshape' of a type (line 116)
        reshape_671269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 15), array_call_result_671268, 'reshape')
        # Calling reshape(args, kwargs) (line 116)
        reshape_call_result_671273 = invoke(stypy.reporting.localization.Localization(__file__, 116, 15), reshape_671269, *[int_671270, int_671271], **kwargs_671272)
        
        # Assigning a type to the variable 'data' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'data', reshape_call_result_671273)
        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to hdquantiles(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'data' (line 117)
        data_671276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'data', False)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_671277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        float_671278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 34), list_671277, float_671278)
        # Adding element type (line 117)
        float_671279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 34), list_671277, float_671279)
        # Adding element type (line 117)
        float_671280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 34), list_671277, float_671280)
        
        # Processing the call keyword arguments (line 117)
        int_671281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 55), 'int')
        keyword_671282 = int_671281
        kwargs_671283 = {'axis': keyword_671282}
        # Getting the type of 'ms' (line 117)
        ms_671274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 14), 'ms', False)
        # Obtaining the member 'hdquantiles' of a type (line 117)
        hdquantiles_671275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 14), ms_671274, 'hdquantiles')
        # Calling hdquantiles(args, kwargs) (line 117)
        hdquantiles_call_result_671284 = invoke(stypy.reporting.localization.Localization(__file__, 117, 14), hdquantiles_671275, *[data_671276, list_671277], **kwargs_671283)
        
        # Assigning a type to the variable 'hdq' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'hdq', hdquantiles_call_result_671284)
        
        # Call to assert_almost_equal(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Obtaining the type of the subscript
        slice_671286 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 118, 28), None, None, None)
        int_671287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 34), 'int')
        # Getting the type of 'hdq' (line 118)
        hdq_671288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'hdq', False)
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___671289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 28), hdq_671288, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_671290 = invoke(stypy.reporting.localization.Localization(__file__, 118, 28), getitem___671289, (slice_671286, int_671287))
        
        
        # Call to hdquantiles(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Obtaining the type of the subscript
        slice_671293 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 118, 53), None, None, None)
        int_671294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 60), 'int')
        # Getting the type of 'data' (line 118)
        data_671295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 53), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___671296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 53), data_671295, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_671297 = invoke(stypy.reporting.localization.Localization(__file__, 118, 53), getitem___671296, (slice_671293, int_671294))
        
        
        # Obtaining an instance of the builtin type 'list' (line 118)
        list_671298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 63), 'list')
        # Adding type elements to the builtin type 'list' instance (line 118)
        # Adding element type (line 118)
        float_671299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 63), list_671298, float_671299)
        # Adding element type (line 118)
        float_671300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 69), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 63), list_671298, float_671300)
        # Adding element type (line 118)
        float_671301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 63), list_671298, float_671301)
        
        # Processing the call keyword arguments (line 118)
        kwargs_671302 = {}
        # Getting the type of 'ms' (line 118)
        ms_671291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 38), 'ms', False)
        # Obtaining the member 'hdquantiles' of a type (line 118)
        hdquantiles_671292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 38), ms_671291, 'hdquantiles')
        # Calling hdquantiles(args, kwargs) (line 118)
        hdquantiles_call_result_671303 = invoke(stypy.reporting.localization.Localization(__file__, 118, 38), hdquantiles_671292, *[subscript_call_result_671297, list_671298], **kwargs_671302)
        
        # Processing the call keyword arguments (line 118)
        kwargs_671304 = {}
        # Getting the type of 'assert_almost_equal' (line 118)
        assert_almost_equal_671285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 118)
        assert_almost_equal_call_result_671305 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), assert_almost_equal_671285, *[subscript_call_result_671290, hdquantiles_call_result_671303], **kwargs_671304)
        
        
        # Call to assert_almost_equal(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Obtaining the type of the subscript
        slice_671307 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 119, 28), None, None, None)
        int_671308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 34), 'int')
        # Getting the type of 'hdq' (line 119)
        hdq_671309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 28), 'hdq', False)
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___671310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 28), hdq_671309, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_671311 = invoke(stypy.reporting.localization.Localization(__file__, 119, 28), getitem___671310, (slice_671307, int_671308))
        
        
        # Call to hdquantiles(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Obtaining the type of the subscript
        slice_671314 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 119, 54), None, None, None)
        int_671315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 61), 'int')
        # Getting the type of 'data' (line 119)
        data_671316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 54), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___671317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 54), data_671316, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_671318 = invoke(stypy.reporting.localization.Localization(__file__, 119, 54), getitem___671317, (slice_671314, int_671315))
        
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_671319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 65), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        float_671320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 65), list_671319, float_671320)
        # Adding element type (line 119)
        float_671321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 71), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 65), list_671319, float_671321)
        # Adding element type (line 119)
        float_671322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 75), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 65), list_671319, float_671322)
        
        # Processing the call keyword arguments (line 119)
        kwargs_671323 = {}
        # Getting the type of 'ms' (line 119)
        ms_671312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 39), 'ms', False)
        # Obtaining the member 'hdquantiles' of a type (line 119)
        hdquantiles_671313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 39), ms_671312, 'hdquantiles')
        # Calling hdquantiles(args, kwargs) (line 119)
        hdquantiles_call_result_671324 = invoke(stypy.reporting.localization.Localization(__file__, 119, 39), hdquantiles_671313, *[subscript_call_result_671318, list_671319], **kwargs_671323)
        
        # Processing the call keyword arguments (line 119)
        kwargs_671325 = {}
        # Getting the type of 'assert_almost_equal' (line 119)
        assert_almost_equal_671306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 119)
        assert_almost_equal_call_result_671326 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), assert_almost_equal_671306, *[subscript_call_result_671311, hdquantiles_call_result_671324], **kwargs_671325)
        
        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to hdquantiles(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'data' (line 120)
        data_671329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'data', False)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_671330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        float_671331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 34), list_671330, float_671331)
        # Adding element type (line 120)
        float_671332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 34), list_671330, float_671332)
        # Adding element type (line 120)
        float_671333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 34), list_671330, float_671333)
        
        # Processing the call keyword arguments (line 120)
        int_671334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 55), 'int')
        keyword_671335 = int_671334
        # Getting the type of 'True' (line 120)
        True_671336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 61), 'True', False)
        keyword_671337 = True_671336
        kwargs_671338 = {'var': keyword_671337, 'axis': keyword_671335}
        # Getting the type of 'ms' (line 120)
        ms_671327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), 'ms', False)
        # Obtaining the member 'hdquantiles' of a type (line 120)
        hdquantiles_671328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 14), ms_671327, 'hdquantiles')
        # Calling hdquantiles(args, kwargs) (line 120)
        hdquantiles_call_result_671339 = invoke(stypy.reporting.localization.Localization(__file__, 120, 14), hdquantiles_671328, *[data_671329, list_671330], **kwargs_671338)
        
        # Assigning a type to the variable 'hdq' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'hdq', hdquantiles_call_result_671339)
        
        # Call to assert_almost_equal(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Obtaining the type of the subscript
        Ellipsis_671341 = Ellipsis
        int_671342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 36), 'int')
        # Getting the type of 'hdq' (line 121)
        hdq_671343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'hdq', False)
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___671344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 28), hdq_671343, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_671345 = invoke(stypy.reporting.localization.Localization(__file__, 121, 28), getitem___671344, (Ellipsis_671341, int_671342))
        
        
        # Call to hdquantiles(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining the type of the subscript
        slice_671348 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 122, 43), None, None, None)
        int_671349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 50), 'int')
        # Getting the type of 'data' (line 122)
        data_671350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 43), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___671351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 43), data_671350, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_671352 = invoke(stypy.reporting.localization.Localization(__file__, 122, 43), getitem___671351, (slice_671348, int_671349))
        
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_671353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        # Adding element type (line 122)
        float_671354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 53), list_671353, float_671354)
        # Adding element type (line 122)
        float_671355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 53), list_671353, float_671355)
        # Adding element type (line 122)
        float_671356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 53), list_671353, float_671356)
        
        # Processing the call keyword arguments (line 122)
        # Getting the type of 'True' (line 122)
        True_671357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 73), 'True', False)
        keyword_671358 = True_671357
        kwargs_671359 = {'var': keyword_671358}
        # Getting the type of 'ms' (line 122)
        ms_671346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'ms', False)
        # Obtaining the member 'hdquantiles' of a type (line 122)
        hdquantiles_671347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 28), ms_671346, 'hdquantiles')
        # Calling hdquantiles(args, kwargs) (line 122)
        hdquantiles_call_result_671360 = invoke(stypy.reporting.localization.Localization(__file__, 122, 28), hdquantiles_671347, *[subscript_call_result_671352, list_671353], **kwargs_671359)
        
        # Processing the call keyword arguments (line 121)
        kwargs_671361 = {}
        # Getting the type of 'assert_almost_equal' (line 121)
        assert_almost_equal_671340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 121)
        assert_almost_equal_call_result_671362 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), assert_almost_equal_671340, *[subscript_call_result_671345, hdquantiles_call_result_671360], **kwargs_671361)
        
        
        # Call to assert_almost_equal(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Obtaining the type of the subscript
        Ellipsis_671364 = Ellipsis
        int_671365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 36), 'int')
        # Getting the type of 'hdq' (line 123)
        hdq_671366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'hdq', False)
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___671367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 28), hdq_671366, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_671368 = invoke(stypy.reporting.localization.Localization(__file__, 123, 28), getitem___671367, (Ellipsis_671364, int_671365))
        
        
        # Call to hdquantiles(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Obtaining the type of the subscript
        slice_671371 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 124, 43), None, None, None)
        int_671372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 50), 'int')
        # Getting the type of 'data' (line 124)
        data_671373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 43), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___671374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 43), data_671373, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_671375 = invoke(stypy.reporting.localization.Localization(__file__, 124, 43), getitem___671374, (slice_671371, int_671372))
        
        
        # Obtaining an instance of the builtin type 'list' (line 124)
        list_671376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 124)
        # Adding element type (line 124)
        float_671377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 54), list_671376, float_671377)
        # Adding element type (line 124)
        float_671378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 54), list_671376, float_671378)
        # Adding element type (line 124)
        float_671379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 54), list_671376, float_671379)
        
        # Processing the call keyword arguments (line 124)
        # Getting the type of 'True' (line 124)
        True_671380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 75), 'True', False)
        keyword_671381 = True_671380
        kwargs_671382 = {'var': keyword_671381}
        # Getting the type of 'ms' (line 124)
        ms_671369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'ms', False)
        # Obtaining the member 'hdquantiles' of a type (line 124)
        hdquantiles_671370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 28), ms_671369, 'hdquantiles')
        # Calling hdquantiles(args, kwargs) (line 124)
        hdquantiles_call_result_671383 = invoke(stypy.reporting.localization.Localization(__file__, 124, 28), hdquantiles_671370, *[subscript_call_result_671375, list_671376], **kwargs_671382)
        
        # Processing the call keyword arguments (line 123)
        kwargs_671384 = {}
        # Getting the type of 'assert_almost_equal' (line 123)
        assert_almost_equal_671363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 123)
        assert_almost_equal_call_result_671385 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), assert_almost_equal_671363, *[subscript_call_result_671368, hdquantiles_call_result_671383], **kwargs_671384)
        
        
        # ################# End of 'test_hdquantiles(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hdquantiles' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_671386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_671386)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hdquantiles'
        return stypy_return_type_671386


    @norecursion
    def test_hdquantiles_sd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hdquantiles_sd'
        module_type_store = module_type_store.open_function_context('test_hdquantiles_sd', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuantiles.test_hdquantiles_sd.__dict__.__setitem__('stypy_localization', localization)
        TestQuantiles.test_hdquantiles_sd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuantiles.test_hdquantiles_sd.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuantiles.test_hdquantiles_sd.__dict__.__setitem__('stypy_function_name', 'TestQuantiles.test_hdquantiles_sd')
        TestQuantiles.test_hdquantiles_sd.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuantiles.test_hdquantiles_sd.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuantiles.test_hdquantiles_sd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuantiles.test_hdquantiles_sd.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuantiles.test_hdquantiles_sd.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuantiles.test_hdquantiles_sd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuantiles.test_hdquantiles_sd.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuantiles.test_hdquantiles_sd', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hdquantiles_sd', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hdquantiles_sd(...)' code ##################

        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to hdquantiles_sd(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'self' (line 128)
        self_671389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'self', False)
        # Obtaining the member 'data' of a type (line 128)
        data_671390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 32), self_671389, 'data')
        # Processing the call keyword arguments (line 128)
        kwargs_671391 = {}
        # Getting the type of 'ms' (line 128)
        ms_671387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 14), 'ms', False)
        # Obtaining the member 'hdquantiles_sd' of a type (line 128)
        hdquantiles_sd_671388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 14), ms_671387, 'hdquantiles_sd')
        # Calling hdquantiles_sd(args, kwargs) (line 128)
        hdquantiles_sd_call_result_671392 = invoke(stypy.reporting.localization.Localization(__file__, 128, 14), hdquantiles_sd_671388, *[data_671390], **kwargs_671391)
        
        # Assigning a type to the variable 'res' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'res', hdquantiles_sd_call_result_671392)
        
        # Call to assert_(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Getting the type of 'res' (line 129)
        res_671394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'res', False)
        # Obtaining the member 'size' of a type (line 129)
        size_671395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), res_671394, 'size')
        int_671396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 28), 'int')
        # Applying the binary operator '==' (line 129)
        result_eq_671397 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 16), '==', size_671395, int_671396)
        
        # Processing the call keyword arguments (line 129)
        kwargs_671398 = {}
        # Getting the type of 'assert_' (line 129)
        assert__671393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 129)
        assert__call_result_671399 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), assert__671393, *[result_eq_671397], **kwargs_671398)
        
        
        # ################# End of 'test_hdquantiles_sd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hdquantiles_sd' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_671400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_671400)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hdquantiles_sd'
        return stypy_return_type_671400


    @norecursion
    def test_mquantiles_cimj(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_mquantiles_cimj'
        module_type_store = module_type_store.open_function_context('test_mquantiles_cimj', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuantiles.test_mquantiles_cimj.__dict__.__setitem__('stypy_localization', localization)
        TestQuantiles.test_mquantiles_cimj.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuantiles.test_mquantiles_cimj.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuantiles.test_mquantiles_cimj.__dict__.__setitem__('stypy_function_name', 'TestQuantiles.test_mquantiles_cimj')
        TestQuantiles.test_mquantiles_cimj.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuantiles.test_mquantiles_cimj.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuantiles.test_mquantiles_cimj.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuantiles.test_mquantiles_cimj.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuantiles.test_mquantiles_cimj.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuantiles.test_mquantiles_cimj.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuantiles.test_mquantiles_cimj.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuantiles.test_mquantiles_cimj', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_mquantiles_cimj', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_mquantiles_cimj(...)' code ##################

        
        # Assigning a Call to a Tuple (line 133):
        
        # Assigning a Subscript to a Name (line 133):
        
        # Obtaining the type of the subscript
        int_671401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'int')
        
        # Call to mquantiles_cimj(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'self' (line 133)
        self_671404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 48), 'self', False)
        # Obtaining the member 'data' of a type (line 133)
        data_671405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 48), self_671404, 'data')
        # Processing the call keyword arguments (line 133)
        kwargs_671406 = {}
        # Getting the type of 'ms' (line 133)
        ms_671402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 29), 'ms', False)
        # Obtaining the member 'mquantiles_cimj' of a type (line 133)
        mquantiles_cimj_671403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 29), ms_671402, 'mquantiles_cimj')
        # Calling mquantiles_cimj(args, kwargs) (line 133)
        mquantiles_cimj_call_result_671407 = invoke(stypy.reporting.localization.Localization(__file__, 133, 29), mquantiles_cimj_671403, *[data_671405], **kwargs_671406)
        
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___671408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), mquantiles_cimj_call_result_671407, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_671409 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), getitem___671408, int_671401)
        
        # Assigning a type to the variable 'tuple_var_assignment_670847' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_670847', subscript_call_result_671409)
        
        # Assigning a Subscript to a Name (line 133):
        
        # Obtaining the type of the subscript
        int_671410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'int')
        
        # Call to mquantiles_cimj(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'self' (line 133)
        self_671413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 48), 'self', False)
        # Obtaining the member 'data' of a type (line 133)
        data_671414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 48), self_671413, 'data')
        # Processing the call keyword arguments (line 133)
        kwargs_671415 = {}
        # Getting the type of 'ms' (line 133)
        ms_671411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 29), 'ms', False)
        # Obtaining the member 'mquantiles_cimj' of a type (line 133)
        mquantiles_cimj_671412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 29), ms_671411, 'mquantiles_cimj')
        # Calling mquantiles_cimj(args, kwargs) (line 133)
        mquantiles_cimj_call_result_671416 = invoke(stypy.reporting.localization.Localization(__file__, 133, 29), mquantiles_cimj_671412, *[data_671414], **kwargs_671415)
        
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___671417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), mquantiles_cimj_call_result_671416, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_671418 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), getitem___671417, int_671410)
        
        # Assigning a type to the variable 'tuple_var_assignment_670848' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_670848', subscript_call_result_671418)
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'tuple_var_assignment_670847' (line 133)
        tuple_var_assignment_670847_671419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_670847')
        # Assigning a type to the variable 'ci_lower' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'ci_lower', tuple_var_assignment_670847_671419)
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'tuple_var_assignment_670848' (line 133)
        tuple_var_assignment_670848_671420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_670848')
        # Assigning a type to the variable 'ci_upper' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'ci_upper', tuple_var_assignment_670848_671420)
        
        # Call to assert_(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Getting the type of 'ci_lower' (line 134)
        ci_lower_671422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'ci_lower', False)
        # Obtaining the member 'size' of a type (line 134)
        size_671423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 16), ci_lower_671422, 'size')
        # Getting the type of 'ci_upper' (line 134)
        ci_upper_671424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 33), 'ci_upper', False)
        # Obtaining the member 'size' of a type (line 134)
        size_671425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 33), ci_upper_671424, 'size')
        # Applying the binary operator '==' (line 134)
        result_eq_671426 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 16), '==', size_671423, size_671425)
        int_671427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 50), 'int')
        # Applying the binary operator '==' (line 134)
        result_eq_671428 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 16), '==', size_671425, int_671427)
        # Applying the binary operator '&' (line 134)
        result_and__671429 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 16), '&', result_eq_671426, result_eq_671428)
        
        # Processing the call keyword arguments (line 134)
        kwargs_671430 = {}
        # Getting the type of 'assert_' (line 134)
        assert__671421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 134)
        assert__call_result_671431 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), assert__671421, *[result_and__671429], **kwargs_671430)
        
        
        # ################# End of 'test_mquantiles_cimj(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_mquantiles_cimj' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_671432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_671432)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_mquantiles_cimj'
        return stypy_return_type_671432


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 85, 0, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuantiles.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestQuantiles' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'TestQuantiles', TestQuantiles)

# Assigning a List to a Name (line 86):

# Obtaining an instance of the builtin type 'list' (line 86)
list_671433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 86)
# Adding element type (line 86)
float_671434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 12), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671434)
# Adding element type (line 86)
float_671435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671435)
# Adding element type (line 86)
float_671436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671436)
# Adding element type (line 86)
float_671437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 48), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671437)
# Adding element type (line 86)
float_671438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 60), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671438)
# Adding element type (line 86)
float_671439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671439)
# Adding element type (line 86)
float_671440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671440)
# Adding element type (line 86)
float_671441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671441)
# Adding element type (line 86)
float_671442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671442)
# Adding element type (line 86)
float_671443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671443)
# Adding element type (line 86)
float_671444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671444)
# Adding element type (line 86)
float_671445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671445)
# Adding element type (line 86)
float_671446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671446)
# Adding element type (line 86)
float_671447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671447)
# Adding element type (line 86)
float_671448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671448)
# Adding element type (line 86)
float_671449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671449)
# Adding element type (line 86)
float_671450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671450)
# Adding element type (line 86)
float_671451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671451)
# Adding element type (line 86)
float_671452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671452)
# Adding element type (line 86)
float_671453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671453)
# Adding element type (line 86)
float_671454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671454)
# Adding element type (line 86)
float_671455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671455)
# Adding element type (line 86)
float_671456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671456)
# Adding element type (line 86)
float_671457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671457)
# Adding element type (line 86)
float_671458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671458)
# Adding element type (line 86)
float_671459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671459)
# Adding element type (line 86)
float_671460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671460)
# Adding element type (line 86)
float_671461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671461)
# Adding element type (line 86)
float_671462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671462)
# Adding element type (line 86)
float_671463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671463)
# Adding element type (line 86)
float_671464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671464)
# Adding element type (line 86)
float_671465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671465)
# Adding element type (line 86)
float_671466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671466)
# Adding element type (line 86)
float_671467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671467)
# Adding element type (line 86)
float_671468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671468)
# Adding element type (line 86)
float_671469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671469)
# Adding element type (line 86)
float_671470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671470)
# Adding element type (line 86)
float_671471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671471)
# Adding element type (line 86)
float_671472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671472)
# Adding element type (line 86)
float_671473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671473)
# Adding element type (line 86)
float_671474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671474)
# Adding element type (line 86)
float_671475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671475)
# Adding element type (line 86)
float_671476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671476)
# Adding element type (line 86)
float_671477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671477)
# Adding element type (line 86)
float_671478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671478)
# Adding element type (line 86)
float_671479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671479)
# Adding element type (line 86)
float_671480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671480)
# Adding element type (line 86)
float_671481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671481)
# Adding element type (line 86)
float_671482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671482)
# Adding element type (line 86)
float_671483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671483)
# Adding element type (line 86)
float_671484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671484)
# Adding element type (line 86)
float_671485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671485)
# Adding element type (line 86)
float_671486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671486)
# Adding element type (line 86)
float_671487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671487)
# Adding element type (line 86)
float_671488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671488)
# Adding element type (line 86)
float_671489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671489)
# Adding element type (line 86)
float_671490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671490)
# Adding element type (line 86)
float_671491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671491)
# Adding element type (line 86)
float_671492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671492)
# Adding element type (line 86)
float_671493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671493)
# Adding element type (line 86)
float_671494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671494)
# Adding element type (line 86)
float_671495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671495)
# Adding element type (line 86)
float_671496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671496)
# Adding element type (line 86)
float_671497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671497)
# Adding element type (line 86)
float_671498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671498)
# Adding element type (line 86)
float_671499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671499)
# Adding element type (line 86)
float_671500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671500)
# Adding element type (line 86)
float_671501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671501)
# Adding element type (line 86)
float_671502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671502)
# Adding element type (line 86)
float_671503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671503)
# Adding element type (line 86)
float_671504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671504)
# Adding element type (line 86)
float_671505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671505)
# Adding element type (line 86)
float_671506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671506)
# Adding element type (line 86)
float_671507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671507)
# Adding element type (line 86)
float_671508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671508)
# Adding element type (line 86)
float_671509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671509)
# Adding element type (line 86)
float_671510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671510)
# Adding element type (line 86)
float_671511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671511)
# Adding element type (line 86)
float_671512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671512)
# Adding element type (line 86)
float_671513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671513)
# Adding element type (line 86)
float_671514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671514)
# Adding element type (line 86)
float_671515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671515)
# Adding element type (line 86)
float_671516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671516)
# Adding element type (line 86)
float_671517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671517)
# Adding element type (line 86)
float_671518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671518)
# Adding element type (line 86)
float_671519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671519)
# Adding element type (line 86)
float_671520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671520)
# Adding element type (line 86)
float_671521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671521)
# Adding element type (line 86)
float_671522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671522)
# Adding element type (line 86)
float_671523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671523)
# Adding element type (line 86)
float_671524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671524)
# Adding element type (line 86)
float_671525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671525)
# Adding element type (line 86)
float_671526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671526)
# Adding element type (line 86)
float_671527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671527)
# Adding element type (line 86)
float_671528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671528)
# Adding element type (line 86)
float_671529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671529)
# Adding element type (line 86)
float_671530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671530)
# Adding element type (line 86)
float_671531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671531)
# Adding element type (line 86)
float_671532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671532)
# Adding element type (line 86)
float_671533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), list_671433, float_671533)

# Getting the type of 'TestQuantiles'
TestQuantiles_671534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestQuantiles')
# Setting the type of the member 'data' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestQuantiles_671534, 'data', list_671433)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
