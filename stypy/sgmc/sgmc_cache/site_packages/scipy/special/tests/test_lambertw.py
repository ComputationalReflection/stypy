
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #
2: # Tests for the lambertw function,
3: # Adapted from the MPMath tests [1] by Yosef Meller, mellerf@netvision.net.il
4: # Distributed under the same license as SciPy itself.
5: #
6: # [1] mpmath source code, Subversion revision 992
7: #     http://code.google.com/p/mpmath/source/browse/trunk/mpmath/tests/test_functions2.py?spec=svn994&r=992
8: 
9: from __future__ import division, print_function, absolute_import
10: 
11: import numpy as np
12: from numpy.testing import assert_, assert_equal, assert_array_almost_equal
13: from scipy.special import lambertw
14: from numpy import nan, inf, pi, e, isnan, log, r_, array, complex_
15: 
16: from scipy.special._testutils import FuncData
17: 
18: 
19: def test_values():
20:     assert_(isnan(lambertw(nan)))
21:     assert_equal(lambertw(inf,1).real, inf)
22:     assert_equal(lambertw(inf,1).imag, 2*pi)
23:     assert_equal(lambertw(-inf,1).real, inf)
24:     assert_equal(lambertw(-inf,1).imag, 3*pi)
25: 
26:     assert_equal(lambertw(1.), lambertw(1., 0))
27: 
28:     data = [
29:         (0,0, 0),
30:         (0+0j,0, 0),
31:         (inf,0, inf),
32:         (0,-1, -inf),
33:         (0,1, -inf),
34:         (0,3, -inf),
35:         (e,0, 1),
36:         (1,0, 0.567143290409783873),
37:         (-pi/2,0, 1j*pi/2),
38:         (-log(2)/2,0, -log(2)),
39:         (0.25,0, 0.203888354702240164),
40:         (-0.25,0, -0.357402956181388903),
41:         (-1./10000,0, -0.000100010001500266719),
42:         (-0.25,-1, -2.15329236411034965),
43:         (0.25,-1, -3.00899800997004620-4.07652978899159763j),
44:         (-0.25,-1, -2.15329236411034965),
45:         (0.25,1, -3.00899800997004620+4.07652978899159763j),
46:         (-0.25,1, -3.48973228422959210+7.41405453009603664j),
47:         (-4,0, 0.67881197132094523+1.91195078174339937j),
48:         (-4,1, -0.66743107129800988+7.76827456802783084j),
49:         (-4,-1, 0.67881197132094523-1.91195078174339937j),
50:         (1000,0, 5.24960285240159623),
51:         (1000,1, 4.91492239981054535+5.44652615979447070j),
52:         (1000,-1, 4.91492239981054535-5.44652615979447070j),
53:         (1000,5, 3.5010625305312892+29.9614548941181328j),
54:         (3+4j,0, 1.281561806123775878+0.533095222020971071j),
55:         (-0.4+0.4j,0, -0.10396515323290657+0.61899273315171632j),
56:         (3+4j,1, -0.11691092896595324+5.61888039871282334j),
57:         (3+4j,-1, 0.25856740686699742-3.85211668616143559j),
58:         (-0.5,-1, -0.794023632344689368-0.770111750510379110j),
59:         (-1./10000,1, -11.82350837248724344+6.80546081842002101j),
60:         (-1./10000,-1, -11.6671145325663544),
61:         (-1./10000,-2, -11.82350837248724344-6.80546081842002101j),
62:         (-1./100000,4, -14.9186890769540539+26.1856750178782046j),
63:         (-1./100000,5, -15.0931437726379218666+32.5525721210262290086j),
64:         ((2+1j)/10,0, 0.173704503762911669+0.071781336752835511j),
65:         ((2+1j)/10,1, -3.21746028349820063+4.56175438896292539j),
66:         ((2+1j)/10,-1, -3.03781405002993088-3.53946629633505737j),
67:         ((2+1j)/10,4, -4.6878509692773249+23.8313630697683291j),
68:         (-(2+1j)/10,0, -0.226933772515757933-0.164986470020154580j),
69:         (-(2+1j)/10,1, -2.43569517046110001+0.76974067544756289j),
70:         (-(2+1j)/10,-1, -3.54858738151989450-6.91627921869943589j),
71:         (-(2+1j)/10,4, -4.5500846928118151+20.6672982215434637j),
72:         (pi,0, 1.073658194796149172092178407024821347547745350410314531),
73: 
74:         # Former bug in generated branch,
75:         (-0.5+0.002j,0, -0.78917138132659918344 + 0.76743539379990327749j),
76:         (-0.5-0.002j,0, -0.78917138132659918344 - 0.76743539379990327749j),
77:         (-0.448+0.4j,0, -0.11855133765652382241 + 0.66570534313583423116j),
78:         (-0.448-0.4j,0, -0.11855133765652382241 - 0.66570534313583423116j),
79:     ]
80:     data = array(data, dtype=complex_)
81: 
82:     def w(x, y):
83:         return lambertw(x, y.real.astype(int))
84:     olderr = np.seterr(all='ignore')
85:     try:
86:         FuncData(w, data, (0,1), 2, rtol=1e-10, atol=1e-13).check()
87:     finally:
88:         np.seterr(**olderr)
89: 
90: 
91: def test_ufunc():
92:     assert_array_almost_equal(
93:         lambertw(r_[0., e, 1.]), r_[0., 1., 0.567143290409783873])
94: 
95: 
96: def test_lambertw_ufunc_loop_selection():
97:     # see https://github.com/scipy/scipy/issues/4895
98:     dt = np.dtype(np.complex128)
99:     assert_equal(lambertw(0, 0, 0).dtype, dt)
100:     assert_equal(lambertw([0], 0, 0).dtype, dt)
101:     assert_equal(lambertw(0, [0], 0).dtype, dt)
102:     assert_equal(lambertw(0, 0, [0]).dtype, dt)
103:     assert_equal(lambertw([0], [0], [0]).dtype, dt)
104: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_540878 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_540878) is not StypyTypeError):

    if (import_540878 != 'pyd_module'):
        __import__(import_540878)
        sys_modules_540879 = sys.modules[import_540878]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_540879.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_540878)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.testing import assert_, assert_equal, assert_array_almost_equal' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_540880 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing')

if (type(import_540880) is not StypyTypeError):

    if (import_540880 != 'pyd_module'):
        __import__(import_540880)
        sys_modules_540881 = sys.modules[import_540880]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing', sys_modules_540881.module_type_store, module_type_store, ['assert_', 'assert_equal', 'assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_540881, sys_modules_540881.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_equal, assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_equal', 'assert_array_almost_equal'], [assert_, assert_equal, assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing', import_540880)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.special import lambertw' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_540882 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.special')

if (type(import_540882) is not StypyTypeError):

    if (import_540882 != 'pyd_module'):
        __import__(import_540882)
        sys_modules_540883 = sys.modules[import_540882]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.special', sys_modules_540883.module_type_store, module_type_store, ['lambertw'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_540883, sys_modules_540883.module_type_store, module_type_store)
    else:
        from scipy.special import lambertw

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.special', None, module_type_store, ['lambertw'], [lambertw])

else:
    # Assigning a type to the variable 'scipy.special' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.special', import_540882)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy import nan, inf, pi, e, isnan, log, r_, array, complex_' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_540884 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy')

if (type(import_540884) is not StypyTypeError):

    if (import_540884 != 'pyd_module'):
        __import__(import_540884)
        sys_modules_540885 = sys.modules[import_540884]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', sys_modules_540885.module_type_store, module_type_store, ['nan', 'inf', 'pi', 'e', 'isnan', 'log', 'r_', 'array', 'complex_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_540885, sys_modules_540885.module_type_store, module_type_store)
    else:
        from numpy import nan, inf, pi, e, isnan, log, r_, array, complex_

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', None, module_type_store, ['nan', 'inf', 'pi', 'e', 'isnan', 'log', 'r_', 'array', 'complex_'], [nan, inf, pi, e, isnan, log, r_, array, complex_])

else:
    # Assigning a type to the variable 'numpy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', import_540884)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.special._testutils import FuncData' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_540886 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.special._testutils')

if (type(import_540886) is not StypyTypeError):

    if (import_540886 != 'pyd_module'):
        __import__(import_540886)
        sys_modules_540887 = sys.modules[import_540886]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.special._testutils', sys_modules_540887.module_type_store, module_type_store, ['FuncData'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_540887, sys_modules_540887.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import FuncData

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.special._testutils', None, module_type_store, ['FuncData'], [FuncData])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.special._testutils', import_540886)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


@norecursion
def test_values(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_values'
    module_type_store = module_type_store.open_function_context('test_values', 19, 0, False)
    
    # Passed parameters checking function
    test_values.stypy_localization = localization
    test_values.stypy_type_of_self = None
    test_values.stypy_type_store = module_type_store
    test_values.stypy_function_name = 'test_values'
    test_values.stypy_param_names_list = []
    test_values.stypy_varargs_param_name = None
    test_values.stypy_kwargs_param_name = None
    test_values.stypy_call_defaults = defaults
    test_values.stypy_call_varargs = varargs
    test_values.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_values', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_values', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_values(...)' code ##################

    
    # Call to assert_(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Call to isnan(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Call to lambertw(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'nan' (line 20)
    nan_540891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 27), 'nan', False)
    # Processing the call keyword arguments (line 20)
    kwargs_540892 = {}
    # Getting the type of 'lambertw' (line 20)
    lambertw_540890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 18), 'lambertw', False)
    # Calling lambertw(args, kwargs) (line 20)
    lambertw_call_result_540893 = invoke(stypy.reporting.localization.Localization(__file__, 20, 18), lambertw_540890, *[nan_540891], **kwargs_540892)
    
    # Processing the call keyword arguments (line 20)
    kwargs_540894 = {}
    # Getting the type of 'isnan' (line 20)
    isnan_540889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'isnan', False)
    # Calling isnan(args, kwargs) (line 20)
    isnan_call_result_540895 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), isnan_540889, *[lambertw_call_result_540893], **kwargs_540894)
    
    # Processing the call keyword arguments (line 20)
    kwargs_540896 = {}
    # Getting the type of 'assert_' (line 20)
    assert__540888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 20)
    assert__call_result_540897 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), assert__540888, *[isnan_call_result_540895], **kwargs_540896)
    
    
    # Call to assert_equal(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Call to lambertw(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'inf' (line 21)
    inf_540900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 26), 'inf', False)
    int_540901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'int')
    # Processing the call keyword arguments (line 21)
    kwargs_540902 = {}
    # Getting the type of 'lambertw' (line 21)
    lambertw_540899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'lambertw', False)
    # Calling lambertw(args, kwargs) (line 21)
    lambertw_call_result_540903 = invoke(stypy.reporting.localization.Localization(__file__, 21, 17), lambertw_540899, *[inf_540900, int_540901], **kwargs_540902)
    
    # Obtaining the member 'real' of a type (line 21)
    real_540904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 17), lambertw_call_result_540903, 'real')
    # Getting the type of 'inf' (line 21)
    inf_540905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 39), 'inf', False)
    # Processing the call keyword arguments (line 21)
    kwargs_540906 = {}
    # Getting the type of 'assert_equal' (line 21)
    assert_equal_540898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 21)
    assert_equal_call_result_540907 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), assert_equal_540898, *[real_540904, inf_540905], **kwargs_540906)
    
    
    # Call to assert_equal(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Call to lambertw(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'inf' (line 22)
    inf_540910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 26), 'inf', False)
    int_540911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'int')
    # Processing the call keyword arguments (line 22)
    kwargs_540912 = {}
    # Getting the type of 'lambertw' (line 22)
    lambertw_540909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'lambertw', False)
    # Calling lambertw(args, kwargs) (line 22)
    lambertw_call_result_540913 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), lambertw_540909, *[inf_540910, int_540911], **kwargs_540912)
    
    # Obtaining the member 'imag' of a type (line 22)
    imag_540914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), lambertw_call_result_540913, 'imag')
    int_540915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 39), 'int')
    # Getting the type of 'pi' (line 22)
    pi_540916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 41), 'pi', False)
    # Applying the binary operator '*' (line 22)
    result_mul_540917 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 39), '*', int_540915, pi_540916)
    
    # Processing the call keyword arguments (line 22)
    kwargs_540918 = {}
    # Getting the type of 'assert_equal' (line 22)
    assert_equal_540908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 22)
    assert_equal_call_result_540919 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), assert_equal_540908, *[imag_540914, result_mul_540917], **kwargs_540918)
    
    
    # Call to assert_equal(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to lambertw(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Getting the type of 'inf' (line 23)
    inf_540922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 27), 'inf', False)
    # Applying the 'usub' unary operator (line 23)
    result___neg___540923 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 26), 'usub', inf_540922)
    
    int_540924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 31), 'int')
    # Processing the call keyword arguments (line 23)
    kwargs_540925 = {}
    # Getting the type of 'lambertw' (line 23)
    lambertw_540921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 17), 'lambertw', False)
    # Calling lambertw(args, kwargs) (line 23)
    lambertw_call_result_540926 = invoke(stypy.reporting.localization.Localization(__file__, 23, 17), lambertw_540921, *[result___neg___540923, int_540924], **kwargs_540925)
    
    # Obtaining the member 'real' of a type (line 23)
    real_540927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 17), lambertw_call_result_540926, 'real')
    # Getting the type of 'inf' (line 23)
    inf_540928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 40), 'inf', False)
    # Processing the call keyword arguments (line 23)
    kwargs_540929 = {}
    # Getting the type of 'assert_equal' (line 23)
    assert_equal_540920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 23)
    assert_equal_call_result_540930 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), assert_equal_540920, *[real_540927, inf_540928], **kwargs_540929)
    
    
    # Call to assert_equal(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to lambertw(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Getting the type of 'inf' (line 24)
    inf_540933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 27), 'inf', False)
    # Applying the 'usub' unary operator (line 24)
    result___neg___540934 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 26), 'usub', inf_540933)
    
    int_540935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 31), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_540936 = {}
    # Getting the type of 'lambertw' (line 24)
    lambertw_540932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'lambertw', False)
    # Calling lambertw(args, kwargs) (line 24)
    lambertw_call_result_540937 = invoke(stypy.reporting.localization.Localization(__file__, 24, 17), lambertw_540932, *[result___neg___540934, int_540935], **kwargs_540936)
    
    # Obtaining the member 'imag' of a type (line 24)
    imag_540938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 17), lambertw_call_result_540937, 'imag')
    int_540939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 40), 'int')
    # Getting the type of 'pi' (line 24)
    pi_540940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 42), 'pi', False)
    # Applying the binary operator '*' (line 24)
    result_mul_540941 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 40), '*', int_540939, pi_540940)
    
    # Processing the call keyword arguments (line 24)
    kwargs_540942 = {}
    # Getting the type of 'assert_equal' (line 24)
    assert_equal_540931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 24)
    assert_equal_call_result_540943 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), assert_equal_540931, *[imag_540938, result_mul_540941], **kwargs_540942)
    
    
    # Call to assert_equal(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Call to lambertw(...): (line 26)
    # Processing the call arguments (line 26)
    float_540946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'float')
    # Processing the call keyword arguments (line 26)
    kwargs_540947 = {}
    # Getting the type of 'lambertw' (line 26)
    lambertw_540945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'lambertw', False)
    # Calling lambertw(args, kwargs) (line 26)
    lambertw_call_result_540948 = invoke(stypy.reporting.localization.Localization(__file__, 26, 17), lambertw_540945, *[float_540946], **kwargs_540947)
    
    
    # Call to lambertw(...): (line 26)
    # Processing the call arguments (line 26)
    float_540950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 40), 'float')
    int_540951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 44), 'int')
    # Processing the call keyword arguments (line 26)
    kwargs_540952 = {}
    # Getting the type of 'lambertw' (line 26)
    lambertw_540949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'lambertw', False)
    # Calling lambertw(args, kwargs) (line 26)
    lambertw_call_result_540953 = invoke(stypy.reporting.localization.Localization(__file__, 26, 31), lambertw_540949, *[float_540950, int_540951], **kwargs_540952)
    
    # Processing the call keyword arguments (line 26)
    kwargs_540954 = {}
    # Getting the type of 'assert_equal' (line 26)
    assert_equal_540944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 26)
    assert_equal_call_result_540955 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), assert_equal_540944, *[lambertw_call_result_540948, lambertw_call_result_540953], **kwargs_540954)
    
    
    # Assigning a List to a Name (line 28):
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_540956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 29)
    tuple_540957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 29)
    # Adding element type (line 29)
    int_540958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), tuple_540957, int_540958)
    # Adding element type (line 29)
    int_540959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), tuple_540957, int_540959)
    # Adding element type (line 29)
    int_540960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), tuple_540957, int_540960)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_540957)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 30)
    tuple_540961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 30)
    # Adding element type (line 30)
    int_540962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'int')
    complex_540963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 11), 'complex')
    # Applying the binary operator '+' (line 30)
    result_add_540964 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 9), '+', int_540962, complex_540963)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_540961, result_add_540964)
    # Adding element type (line 30)
    int_540965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_540961, int_540965)
    # Adding element type (line 30)
    int_540966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_540961, int_540966)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_540961)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 31)
    tuple_540967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 31)
    # Adding element type (line 31)
    # Getting the type of 'inf' (line 31)
    inf_540968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 9), 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_540967, inf_540968)
    # Adding element type (line 31)
    int_540969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_540967, int_540969)
    # Adding element type (line 31)
    # Getting the type of 'inf' (line 31)
    inf_540970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_540967, inf_540970)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_540967)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 32)
    tuple_540971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 32)
    # Adding element type (line 32)
    int_540972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 9), tuple_540971, int_540972)
    # Adding element type (line 32)
    int_540973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 9), tuple_540971, int_540973)
    # Adding element type (line 32)
    
    # Getting the type of 'inf' (line 32)
    inf_540974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'inf')
    # Applying the 'usub' unary operator (line 32)
    result___neg___540975 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 15), 'usub', inf_540974)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 9), tuple_540971, result___neg___540975)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_540971)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 33)
    tuple_540976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 33)
    # Adding element type (line 33)
    int_540977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_540976, int_540977)
    # Adding element type (line 33)
    int_540978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_540976, int_540978)
    # Adding element type (line 33)
    
    # Getting the type of 'inf' (line 33)
    inf_540979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'inf')
    # Applying the 'usub' unary operator (line 33)
    result___neg___540980 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 14), 'usub', inf_540979)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_540976, result___neg___540980)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_540976)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 34)
    tuple_540981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 34)
    # Adding element type (line 34)
    int_540982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 9), tuple_540981, int_540982)
    # Adding element type (line 34)
    int_540983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 9), tuple_540981, int_540983)
    # Adding element type (line 34)
    
    # Getting the type of 'inf' (line 34)
    inf_540984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'inf')
    # Applying the 'usub' unary operator (line 34)
    result___neg___540985 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 14), 'usub', inf_540984)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 9), tuple_540981, result___neg___540985)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_540981)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 35)
    tuple_540986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 35)
    # Adding element type (line 35)
    # Getting the type of 'e' (line 35)
    e_540987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 9), 'e')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_540986, e_540987)
    # Adding element type (line 35)
    int_540988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_540986, int_540988)
    # Adding element type (line 35)
    int_540989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_540986, int_540989)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_540986)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_540990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    int_540991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 9), tuple_540990, int_540991)
    # Adding element type (line 36)
    int_540992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 9), tuple_540990, int_540992)
    # Adding element type (line 36)
    float_540993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 9), tuple_540990, float_540993)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_540990)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 37)
    tuple_540994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 37)
    # Adding element type (line 37)
    
    # Getting the type of 'pi' (line 37)
    pi_540995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 'pi')
    # Applying the 'usub' unary operator (line 37)
    result___neg___540996 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 9), 'usub', pi_540995)
    
    int_540997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 13), 'int')
    # Applying the binary operator 'div' (line 37)
    result_div_540998 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 9), 'div', result___neg___540996, int_540997)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), tuple_540994, result_div_540998)
    # Adding element type (line 37)
    int_540999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), tuple_540994, int_540999)
    # Adding element type (line 37)
    complex_541000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 18), 'complex')
    # Getting the type of 'pi' (line 37)
    pi_541001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'pi')
    # Applying the binary operator '*' (line 37)
    result_mul_541002 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 18), '*', complex_541000, pi_541001)
    
    int_541003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 24), 'int')
    # Applying the binary operator 'div' (line 37)
    result_div_541004 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 23), 'div', result_mul_541002, int_541003)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), tuple_540994, result_div_541004)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_540994)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 38)
    tuple_541005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 38)
    # Adding element type (line 38)
    
    
    # Call to log(...): (line 38)
    # Processing the call arguments (line 38)
    int_541007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 14), 'int')
    # Processing the call keyword arguments (line 38)
    kwargs_541008 = {}
    # Getting the type of 'log' (line 38)
    log_541006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 10), 'log', False)
    # Calling log(args, kwargs) (line 38)
    log_call_result_541009 = invoke(stypy.reporting.localization.Localization(__file__, 38, 10), log_541006, *[int_541007], **kwargs_541008)
    
    # Applying the 'usub' unary operator (line 38)
    result___neg___541010 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 9), 'usub', log_call_result_541009)
    
    int_541011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'int')
    # Applying the binary operator 'div' (line 38)
    result_div_541012 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 9), 'div', result___neg___541010, int_541011)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 9), tuple_541005, result_div_541012)
    # Adding element type (line 38)
    int_541013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 9), tuple_541005, int_541013)
    # Adding element type (line 38)
    
    
    # Call to log(...): (line 38)
    # Processing the call arguments (line 38)
    int_541015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 27), 'int')
    # Processing the call keyword arguments (line 38)
    kwargs_541016 = {}
    # Getting the type of 'log' (line 38)
    log_541014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'log', False)
    # Calling log(args, kwargs) (line 38)
    log_call_result_541017 = invoke(stypy.reporting.localization.Localization(__file__, 38, 23), log_541014, *[int_541015], **kwargs_541016)
    
    # Applying the 'usub' unary operator (line 38)
    result___neg___541018 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 22), 'usub', log_call_result_541017)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 9), tuple_541005, result___neg___541018)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541005)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 39)
    tuple_541019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 39)
    # Adding element type (line 39)
    float_541020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), tuple_541019, float_541020)
    # Adding element type (line 39)
    int_541021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), tuple_541019, int_541021)
    # Adding element type (line 39)
    float_541022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), tuple_541019, float_541022)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541019)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 40)
    tuple_541023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 40)
    # Adding element type (line 40)
    float_541024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), tuple_541023, float_541024)
    # Adding element type (line 40)
    int_541025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), tuple_541023, int_541025)
    # Adding element type (line 40)
    float_541026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), tuple_541023, float_541026)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541023)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 41)
    tuple_541027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 41)
    # Adding element type (line 41)
    float_541028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'float')
    int_541029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 13), 'int')
    # Applying the binary operator 'div' (line 41)
    result_div_541030 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 9), 'div', float_541028, int_541029)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_541027, result_div_541030)
    # Adding element type (line 41)
    int_541031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_541027, int_541031)
    # Adding element type (line 41)
    float_541032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_541027, float_541032)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541027)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_541033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    float_541034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), tuple_541033, float_541034)
    # Adding element type (line 42)
    int_541035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), tuple_541033, int_541035)
    # Adding element type (line 42)
    float_541036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), tuple_541033, float_541036)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541033)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 43)
    tuple_541037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 43)
    # Adding element type (line 43)
    float_541038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 9), tuple_541037, float_541038)
    # Adding element type (line 43)
    int_541039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 9), tuple_541037, int_541039)
    # Adding element type (line 43)
    float_541040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 18), 'float')
    complex_541041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 39), 'complex')
    # Applying the binary operator '-' (line 43)
    result_sub_541042 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 18), '-', float_541040, complex_541041)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 9), tuple_541037, result_sub_541042)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541037)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_541043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    float_541044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 9), tuple_541043, float_541044)
    # Adding element type (line 44)
    int_541045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 9), tuple_541043, int_541045)
    # Adding element type (line 44)
    float_541046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 9), tuple_541043, float_541046)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541043)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 45)
    tuple_541047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 45)
    # Adding element type (line 45)
    float_541048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 9), tuple_541047, float_541048)
    # Adding element type (line 45)
    int_541049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 9), tuple_541047, int_541049)
    # Adding element type (line 45)
    float_541050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 17), 'float')
    complex_541051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 38), 'complex')
    # Applying the binary operator '+' (line 45)
    result_add_541052 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 17), '+', float_541050, complex_541051)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 9), tuple_541047, result_add_541052)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541047)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_541053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    float_541054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 9), tuple_541053, float_541054)
    # Adding element type (line 46)
    int_541055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 9), tuple_541053, int_541055)
    # Adding element type (line 46)
    float_541056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 18), 'float')
    complex_541057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 39), 'complex')
    # Applying the binary operator '+' (line 46)
    result_add_541058 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 18), '+', float_541056, complex_541057)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 9), tuple_541053, result_add_541058)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541053)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 47)
    tuple_541059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 47)
    # Adding element type (line 47)
    int_541060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 9), tuple_541059, int_541060)
    # Adding element type (line 47)
    int_541061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 9), tuple_541059, int_541061)
    # Adding element type (line 47)
    float_541062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 15), 'float')
    complex_541063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 35), 'complex')
    # Applying the binary operator '+' (line 47)
    result_add_541064 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 15), '+', float_541062, complex_541063)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 9), tuple_541059, result_add_541064)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541059)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 48)
    tuple_541065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 48)
    # Adding element type (line 48)
    int_541066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 9), tuple_541065, int_541066)
    # Adding element type (line 48)
    int_541067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 9), tuple_541065, int_541067)
    # Adding element type (line 48)
    float_541068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 15), 'float')
    complex_541069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 36), 'complex')
    # Applying the binary operator '+' (line 48)
    result_add_541070 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 15), '+', float_541068, complex_541069)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 9), tuple_541065, result_add_541070)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541065)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 49)
    tuple_541071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 49)
    # Adding element type (line 49)
    int_541072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 9), tuple_541071, int_541072)
    # Adding element type (line 49)
    int_541073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 9), tuple_541071, int_541073)
    # Adding element type (line 49)
    float_541074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 16), 'float')
    complex_541075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 36), 'complex')
    # Applying the binary operator '-' (line 49)
    result_sub_541076 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 16), '-', float_541074, complex_541075)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 9), tuple_541071, result_sub_541076)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541071)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 50)
    tuple_541077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 50)
    # Adding element type (line 50)
    int_541078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 9), tuple_541077, int_541078)
    # Adding element type (line 50)
    int_541079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 9), tuple_541077, int_541079)
    # Adding element type (line 50)
    float_541080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 9), tuple_541077, float_541080)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541077)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 51)
    tuple_541081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 51)
    # Adding element type (line 51)
    int_541082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), tuple_541081, int_541082)
    # Adding element type (line 51)
    int_541083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), tuple_541081, int_541083)
    # Adding element type (line 51)
    float_541084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 17), 'float')
    complex_541085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 37), 'complex')
    # Applying the binary operator '+' (line 51)
    result_add_541086 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 17), '+', float_541084, complex_541085)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), tuple_541081, result_add_541086)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541081)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 52)
    tuple_541087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 52)
    # Adding element type (line 52)
    int_541088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 9), tuple_541087, int_541088)
    # Adding element type (line 52)
    int_541089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 9), tuple_541087, int_541089)
    # Adding element type (line 52)
    float_541090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 18), 'float')
    complex_541091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 38), 'complex')
    # Applying the binary operator '-' (line 52)
    result_sub_541092 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 18), '-', float_541090, complex_541091)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 9), tuple_541087, result_sub_541092)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541087)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 53)
    tuple_541093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 53)
    # Adding element type (line 53)
    int_541094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 9), tuple_541093, int_541094)
    # Adding element type (line 53)
    int_541095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 9), tuple_541093, int_541095)
    # Adding element type (line 53)
    float_541096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 17), 'float')
    complex_541097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 36), 'complex')
    # Applying the binary operator '+' (line 53)
    result_add_541098 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 17), '+', float_541096, complex_541097)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 9), tuple_541093, result_add_541098)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541093)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 54)
    tuple_541099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 54)
    # Adding element type (line 54)
    int_541100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'int')
    complex_541101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 11), 'complex')
    # Applying the binary operator '+' (line 54)
    result_add_541102 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 9), '+', int_541100, complex_541101)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_541099, result_add_541102)
    # Adding element type (line 54)
    int_541103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_541099, int_541103)
    # Adding element type (line 54)
    float_541104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 17), 'float')
    complex_541105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'complex')
    # Applying the binary operator '+' (line 54)
    result_add_541106 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 17), '+', float_541104, complex_541105)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_541099, result_add_541106)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541099)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 55)
    tuple_541107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 55)
    # Adding element type (line 55)
    float_541108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 9), 'float')
    complex_541109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 14), 'complex')
    # Applying the binary operator '+' (line 55)
    result_add_541110 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 9), '+', float_541108, complex_541109)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_541107, result_add_541110)
    # Adding element type (line 55)
    int_541111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_541107, int_541111)
    # Adding element type (line 55)
    float_541112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'float')
    complex_541113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 43), 'complex')
    # Applying the binary operator '+' (line 55)
    result_add_541114 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 22), '+', float_541112, complex_541113)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_541107, result_add_541114)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541107)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 56)
    tuple_541115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 56)
    # Adding element type (line 56)
    int_541116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 9), 'int')
    complex_541117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 11), 'complex')
    # Applying the binary operator '+' (line 56)
    result_add_541118 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 9), '+', int_541116, complex_541117)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_541115, result_add_541118)
    # Adding element type (line 56)
    int_541119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_541115, int_541119)
    # Adding element type (line 56)
    float_541120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 17), 'float')
    complex_541121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 38), 'complex')
    # Applying the binary operator '+' (line 56)
    result_add_541122 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 17), '+', float_541120, complex_541121)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_541115, result_add_541122)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541115)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 57)
    tuple_541123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 57)
    # Adding element type (line 57)
    int_541124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 9), 'int')
    complex_541125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 11), 'complex')
    # Applying the binary operator '+' (line 57)
    result_add_541126 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 9), '+', int_541124, complex_541125)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 9), tuple_541123, result_add_541126)
    # Adding element type (line 57)
    int_541127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 9), tuple_541123, int_541127)
    # Adding element type (line 57)
    float_541128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 18), 'float')
    complex_541129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 38), 'complex')
    # Applying the binary operator '-' (line 57)
    result_sub_541130 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 18), '-', float_541128, complex_541129)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 9), tuple_541123, result_sub_541130)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541123)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 58)
    tuple_541131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 58)
    # Adding element type (line 58)
    float_541132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), tuple_541131, float_541132)
    # Adding element type (line 58)
    int_541133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), tuple_541131, int_541133)
    # Adding element type (line 58)
    float_541134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 18), 'float')
    complex_541135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 40), 'complex')
    # Applying the binary operator '-' (line 58)
    result_sub_541136 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 18), '-', float_541134, complex_541135)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), tuple_541131, result_sub_541136)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541131)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 59)
    tuple_541137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 59)
    # Adding element type (line 59)
    float_541138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 9), 'float')
    int_541139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 13), 'int')
    # Applying the binary operator 'div' (line 59)
    result_div_541140 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 9), 'div', float_541138, int_541139)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 9), tuple_541137, result_div_541140)
    # Adding element type (line 59)
    int_541141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 9), tuple_541137, int_541141)
    # Adding element type (line 59)
    float_541142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'float')
    complex_541143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 44), 'complex')
    # Applying the binary operator '+' (line 59)
    result_add_541144 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 22), '+', float_541142, complex_541143)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 9), tuple_541137, result_add_541144)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541137)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 60)
    tuple_541145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 60)
    # Adding element type (line 60)
    float_541146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 9), 'float')
    int_541147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 13), 'int')
    # Applying the binary operator 'div' (line 60)
    result_div_541148 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 9), 'div', float_541146, int_541147)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 9), tuple_541145, result_div_541148)
    # Adding element type (line 60)
    int_541149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 9), tuple_541145, int_541149)
    # Adding element type (line 60)
    float_541150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 23), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 9), tuple_541145, float_541150)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541145)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 61)
    tuple_541151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 61)
    # Adding element type (line 61)
    float_541152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 9), 'float')
    int_541153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 13), 'int')
    # Applying the binary operator 'div' (line 61)
    result_div_541154 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 9), 'div', float_541152, int_541153)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 9), tuple_541151, result_div_541154)
    # Adding element type (line 61)
    int_541155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 9), tuple_541151, int_541155)
    # Adding element type (line 61)
    float_541156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'float')
    complex_541157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 45), 'complex')
    # Applying the binary operator '-' (line 61)
    result_sub_541158 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 23), '-', float_541156, complex_541157)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 9), tuple_541151, result_sub_541158)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541151)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 62)
    tuple_541159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 62)
    # Adding element type (line 62)
    float_541160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 9), 'float')
    int_541161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 13), 'int')
    # Applying the binary operator 'div' (line 62)
    result_div_541162 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 9), 'div', float_541160, int_541161)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 9), tuple_541159, result_div_541162)
    # Adding element type (line 62)
    int_541163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 9), tuple_541159, int_541163)
    # Adding element type (line 62)
    float_541164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'float')
    complex_541165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 44), 'complex')
    # Applying the binary operator '+' (line 62)
    result_add_541166 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 23), '+', float_541164, complex_541165)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 9), tuple_541159, result_add_541166)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541159)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_541167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    float_541168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 9), 'float')
    int_541169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 13), 'int')
    # Applying the binary operator 'div' (line 63)
    result_div_541170 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 9), 'div', float_541168, int_541169)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 9), tuple_541167, result_div_541170)
    # Adding element type (line 63)
    int_541171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 9), tuple_541167, int_541171)
    # Adding element type (line 63)
    float_541172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 23), 'float')
    complex_541173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 47), 'complex')
    # Applying the binary operator '+' (line 63)
    result_add_541174 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 23), '+', float_541172, complex_541173)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 9), tuple_541167, result_add_541174)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541167)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 64)
    tuple_541175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 64)
    # Adding element type (line 64)
    int_541176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 10), 'int')
    complex_541177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 12), 'complex')
    # Applying the binary operator '+' (line 64)
    result_add_541178 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 10), '+', int_541176, complex_541177)
    
    int_541179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'int')
    # Applying the binary operator 'div' (line 64)
    result_div_541180 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 9), 'div', result_add_541178, int_541179)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 9), tuple_541175, result_div_541180)
    # Adding element type (line 64)
    int_541181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 9), tuple_541175, int_541181)
    # Adding element type (line 64)
    float_541182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 22), 'float')
    complex_541183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 43), 'complex')
    # Applying the binary operator '+' (line 64)
    result_add_541184 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 22), '+', float_541182, complex_541183)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 9), tuple_541175, result_add_541184)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541175)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 65)
    tuple_541185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 65)
    # Adding element type (line 65)
    int_541186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 10), 'int')
    complex_541187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 12), 'complex')
    # Applying the binary operator '+' (line 65)
    result_add_541188 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 10), '+', int_541186, complex_541187)
    
    int_541189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 16), 'int')
    # Applying the binary operator 'div' (line 65)
    result_div_541190 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 9), 'div', result_add_541188, int_541189)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_541185, result_div_541190)
    # Adding element type (line 65)
    int_541191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_541185, int_541191)
    # Adding element type (line 65)
    float_541192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'float')
    complex_541193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 43), 'complex')
    # Applying the binary operator '+' (line 65)
    result_add_541194 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 22), '+', float_541192, complex_541193)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_541185, result_add_541194)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541185)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 66)
    tuple_541195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 66)
    # Adding element type (line 66)
    int_541196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 10), 'int')
    complex_541197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 12), 'complex')
    # Applying the binary operator '+' (line 66)
    result_add_541198 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 10), '+', int_541196, complex_541197)
    
    int_541199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 16), 'int')
    # Applying the binary operator 'div' (line 66)
    result_div_541200 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 9), 'div', result_add_541198, int_541199)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 9), tuple_541195, result_div_541200)
    # Adding element type (line 66)
    int_541201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 9), tuple_541195, int_541201)
    # Adding element type (line 66)
    float_541202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 23), 'float')
    complex_541203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 44), 'complex')
    # Applying the binary operator '-' (line 66)
    result_sub_541204 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 23), '-', float_541202, complex_541203)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 9), tuple_541195, result_sub_541204)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541195)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 67)
    tuple_541205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 67)
    # Adding element type (line 67)
    int_541206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 10), 'int')
    complex_541207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 12), 'complex')
    # Applying the binary operator '+' (line 67)
    result_add_541208 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 10), '+', int_541206, complex_541207)
    
    int_541209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 16), 'int')
    # Applying the binary operator 'div' (line 67)
    result_div_541210 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 9), 'div', result_add_541208, int_541209)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 9), tuple_541205, result_div_541210)
    # Adding element type (line 67)
    int_541211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 9), tuple_541205, int_541211)
    # Adding element type (line 67)
    float_541212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 22), 'float')
    complex_541213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 42), 'complex')
    # Applying the binary operator '+' (line 67)
    result_add_541214 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 22), '+', float_541212, complex_541213)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 9), tuple_541205, result_add_541214)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541205)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 68)
    tuple_541215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 68)
    # Adding element type (line 68)
    
    int_541216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 11), 'int')
    complex_541217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 13), 'complex')
    # Applying the binary operator '+' (line 68)
    result_add_541218 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 11), '+', int_541216, complex_541217)
    
    # Applying the 'usub' unary operator (line 68)
    result___neg___541219 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 9), 'usub', result_add_541218)
    
    int_541220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 17), 'int')
    # Applying the binary operator 'div' (line 68)
    result_div_541221 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 9), 'div', result___neg___541219, int_541220)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 9), tuple_541215, result_div_541221)
    # Adding element type (line 68)
    int_541222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 9), tuple_541215, int_541222)
    # Adding element type (line 68)
    float_541223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'float')
    complex_541224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 45), 'complex')
    # Applying the binary operator '-' (line 68)
    result_sub_541225 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 23), '-', float_541223, complex_541224)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 9), tuple_541215, result_sub_541225)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541215)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 69)
    tuple_541226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 69)
    # Adding element type (line 69)
    
    int_541227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 11), 'int')
    complex_541228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 13), 'complex')
    # Applying the binary operator '+' (line 69)
    result_add_541229 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 11), '+', int_541227, complex_541228)
    
    # Applying the 'usub' unary operator (line 69)
    result___neg___541230 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 9), 'usub', result_add_541229)
    
    int_541231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 17), 'int')
    # Applying the binary operator 'div' (line 69)
    result_div_541232 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 9), 'div', result___neg___541230, int_541231)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 9), tuple_541226, result_div_541232)
    # Adding element type (line 69)
    int_541233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 9), tuple_541226, int_541233)
    # Adding element type (line 69)
    float_541234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'float')
    complex_541235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 44), 'complex')
    # Applying the binary operator '+' (line 69)
    result_add_541236 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 23), '+', float_541234, complex_541235)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 9), tuple_541226, result_add_541236)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541226)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 70)
    tuple_541237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 70)
    # Adding element type (line 70)
    
    int_541238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 11), 'int')
    complex_541239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 13), 'complex')
    # Applying the binary operator '+' (line 70)
    result_add_541240 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 11), '+', int_541238, complex_541239)
    
    # Applying the 'usub' unary operator (line 70)
    result___neg___541241 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 9), 'usub', result_add_541240)
    
    int_541242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 17), 'int')
    # Applying the binary operator 'div' (line 70)
    result_div_541243 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 9), 'div', result___neg___541241, int_541242)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 9), tuple_541237, result_div_541243)
    # Adding element type (line 70)
    int_541244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 9), tuple_541237, int_541244)
    # Adding element type (line 70)
    float_541245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 24), 'float')
    complex_541246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 45), 'complex')
    # Applying the binary operator '-' (line 70)
    result_sub_541247 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 24), '-', float_541245, complex_541246)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 9), tuple_541237, result_sub_541247)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541237)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 71)
    tuple_541248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 71)
    # Adding element type (line 71)
    
    int_541249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 11), 'int')
    complex_541250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 13), 'complex')
    # Applying the binary operator '+' (line 71)
    result_add_541251 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 11), '+', int_541249, complex_541250)
    
    # Applying the 'usub' unary operator (line 71)
    result___neg___541252 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 9), 'usub', result_add_541251)
    
    int_541253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 17), 'int')
    # Applying the binary operator 'div' (line 71)
    result_div_541254 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 9), 'div', result___neg___541252, int_541253)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 9), tuple_541248, result_div_541254)
    # Adding element type (line 71)
    int_541255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 9), tuple_541248, int_541255)
    # Adding element type (line 71)
    float_541256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 23), 'float')
    complex_541257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 43), 'complex')
    # Applying the binary operator '+' (line 71)
    result_add_541258 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 23), '+', float_541256, complex_541257)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 9), tuple_541248, result_add_541258)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541248)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 72)
    tuple_541259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 72)
    # Adding element type (line 72)
    # Getting the type of 'pi' (line 72)
    pi_541260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 9), 'pi')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 9), tuple_541259, pi_541260)
    # Adding element type (line 72)
    int_541261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 9), tuple_541259, int_541261)
    # Adding element type (line 72)
    float_541262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 9), tuple_541259, float_541262)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541259)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 75)
    tuple_541263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 75)
    # Adding element type (line 75)
    float_541264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 9), 'float')
    complex_541265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 14), 'complex')
    # Applying the binary operator '+' (line 75)
    result_add_541266 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 9), '+', float_541264, complex_541265)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 9), tuple_541263, result_add_541266)
    # Adding element type (line 75)
    int_541267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 9), tuple_541263, int_541267)
    # Adding element type (line 75)
    float_541268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 24), 'float')
    complex_541269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 50), 'complex')
    # Applying the binary operator '+' (line 75)
    result_add_541270 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 24), '+', float_541268, complex_541269)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 9), tuple_541263, result_add_541270)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541263)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 76)
    tuple_541271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 76)
    # Adding element type (line 76)
    float_541272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 9), 'float')
    complex_541273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 14), 'complex')
    # Applying the binary operator '-' (line 76)
    result_sub_541274 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 9), '-', float_541272, complex_541273)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 9), tuple_541271, result_sub_541274)
    # Adding element type (line 76)
    int_541275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 9), tuple_541271, int_541275)
    # Adding element type (line 76)
    float_541276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 24), 'float')
    complex_541277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 50), 'complex')
    # Applying the binary operator '-' (line 76)
    result_sub_541278 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 24), '-', float_541276, complex_541277)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 9), tuple_541271, result_sub_541278)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541271)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 77)
    tuple_541279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 77)
    # Adding element type (line 77)
    float_541280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 9), 'float')
    complex_541281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 16), 'complex')
    # Applying the binary operator '+' (line 77)
    result_add_541282 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 9), '+', float_541280, complex_541281)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 9), tuple_541279, result_add_541282)
    # Adding element type (line 77)
    int_541283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 9), tuple_541279, int_541283)
    # Adding element type (line 77)
    float_541284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 24), 'float')
    complex_541285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 50), 'complex')
    # Applying the binary operator '+' (line 77)
    result_add_541286 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 24), '+', float_541284, complex_541285)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 9), tuple_541279, result_add_541286)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541279)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_541287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    float_541288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 9), 'float')
    complex_541289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 16), 'complex')
    # Applying the binary operator '-' (line 78)
    result_sub_541290 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 9), '-', float_541288, complex_541289)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 9), tuple_541287, result_sub_541290)
    # Adding element type (line 78)
    int_541291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 9), tuple_541287, int_541291)
    # Adding element type (line 78)
    float_541292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 24), 'float')
    complex_541293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 50), 'complex')
    # Applying the binary operator '-' (line 78)
    result_sub_541294 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 24), '-', float_541292, complex_541293)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 9), tuple_541287, result_sub_541294)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_540956, tuple_541287)
    
    # Assigning a type to the variable 'data' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'data', list_540956)
    
    # Assigning a Call to a Name (line 80):
    
    # Call to array(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'data' (line 80)
    data_541296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'data', False)
    # Processing the call keyword arguments (line 80)
    # Getting the type of 'complex_' (line 80)
    complex__541297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'complex_', False)
    keyword_541298 = complex__541297
    kwargs_541299 = {'dtype': keyword_541298}
    # Getting the type of 'array' (line 80)
    array_541295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'array', False)
    # Calling array(args, kwargs) (line 80)
    array_call_result_541300 = invoke(stypy.reporting.localization.Localization(__file__, 80, 11), array_541295, *[data_541296], **kwargs_541299)
    
    # Assigning a type to the variable 'data' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'data', array_call_result_541300)

    @norecursion
    def w(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'w'
        module_type_store = module_type_store.open_function_context('w', 82, 4, False)
        
        # Passed parameters checking function
        w.stypy_localization = localization
        w.stypy_type_of_self = None
        w.stypy_type_store = module_type_store
        w.stypy_function_name = 'w'
        w.stypy_param_names_list = ['x', 'y']
        w.stypy_varargs_param_name = None
        w.stypy_kwargs_param_name = None
        w.stypy_call_defaults = defaults
        w.stypy_call_varargs = varargs
        w.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'w', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'w', localization, ['x', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'w(...)' code ##################

        
        # Call to lambertw(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'x' (line 83)
        x_541302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'x', False)
        
        # Call to astype(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'int' (line 83)
        int_541306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 41), 'int', False)
        # Processing the call keyword arguments (line 83)
        kwargs_541307 = {}
        # Getting the type of 'y' (line 83)
        y_541303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'y', False)
        # Obtaining the member 'real' of a type (line 83)
        real_541304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 27), y_541303, 'real')
        # Obtaining the member 'astype' of a type (line 83)
        astype_541305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 27), real_541304, 'astype')
        # Calling astype(args, kwargs) (line 83)
        astype_call_result_541308 = invoke(stypy.reporting.localization.Localization(__file__, 83, 27), astype_541305, *[int_541306], **kwargs_541307)
        
        # Processing the call keyword arguments (line 83)
        kwargs_541309 = {}
        # Getting the type of 'lambertw' (line 83)
        lambertw_541301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'lambertw', False)
        # Calling lambertw(args, kwargs) (line 83)
        lambertw_call_result_541310 = invoke(stypy.reporting.localization.Localization(__file__, 83, 15), lambertw_541301, *[x_541302, astype_call_result_541308], **kwargs_541309)
        
        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', lambertw_call_result_541310)
        
        # ################# End of 'w(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'w' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_541311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_541311)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'w'
        return stypy_return_type_541311

    # Assigning a type to the variable 'w' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'w', w)
    
    # Assigning a Call to a Name (line 84):
    
    # Call to seterr(...): (line 84)
    # Processing the call keyword arguments (line 84)
    str_541314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 27), 'str', 'ignore')
    keyword_541315 = str_541314
    kwargs_541316 = {'all': keyword_541315}
    # Getting the type of 'np' (line 84)
    np_541312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'np', False)
    # Obtaining the member 'seterr' of a type (line 84)
    seterr_541313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 13), np_541312, 'seterr')
    # Calling seterr(args, kwargs) (line 84)
    seterr_call_result_541317 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), seterr_541313, *[], **kwargs_541316)
    
    # Assigning a type to the variable 'olderr' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'olderr', seterr_call_result_541317)
    
    # Try-finally block (line 85)
    
    # Call to check(...): (line 86)
    # Processing the call keyword arguments (line 86)
    kwargs_541332 = {}
    
    # Call to FuncData(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'w' (line 86)
    w_541319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 17), 'w', False)
    # Getting the type of 'data' (line 86)
    data_541320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'data', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 86)
    tuple_541321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 86)
    # Adding element type (line 86)
    int_541322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 27), tuple_541321, int_541322)
    # Adding element type (line 86)
    int_541323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 27), tuple_541321, int_541323)
    
    int_541324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 33), 'int')
    # Processing the call keyword arguments (line 86)
    float_541325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 41), 'float')
    keyword_541326 = float_541325
    float_541327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 53), 'float')
    keyword_541328 = float_541327
    kwargs_541329 = {'rtol': keyword_541326, 'atol': keyword_541328}
    # Getting the type of 'FuncData' (line 86)
    FuncData_541318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 86)
    FuncData_call_result_541330 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), FuncData_541318, *[w_541319, data_541320, tuple_541321, int_541324], **kwargs_541329)
    
    # Obtaining the member 'check' of a type (line 86)
    check_541331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), FuncData_call_result_541330, 'check')
    # Calling check(args, kwargs) (line 86)
    check_call_result_541333 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), check_541331, *[], **kwargs_541332)
    
    
    # finally branch of the try-finally block (line 85)
    
    # Call to seterr(...): (line 88)
    # Processing the call keyword arguments (line 88)
    # Getting the type of 'olderr' (line 88)
    olderr_541336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'olderr', False)
    kwargs_541337 = {'olderr_541336': olderr_541336}
    # Getting the type of 'np' (line 88)
    np_541334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'np', False)
    # Obtaining the member 'seterr' of a type (line 88)
    seterr_541335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), np_541334, 'seterr')
    # Calling seterr(args, kwargs) (line 88)
    seterr_call_result_541338 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), seterr_541335, *[], **kwargs_541337)
    
    
    
    # ################# End of 'test_values(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_values' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_541339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_541339)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_values'
    return stypy_return_type_541339

# Assigning a type to the variable 'test_values' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'test_values', test_values)

@norecursion
def test_ufunc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_ufunc'
    module_type_store = module_type_store.open_function_context('test_ufunc', 91, 0, False)
    
    # Passed parameters checking function
    test_ufunc.stypy_localization = localization
    test_ufunc.stypy_type_of_self = None
    test_ufunc.stypy_type_store = module_type_store
    test_ufunc.stypy_function_name = 'test_ufunc'
    test_ufunc.stypy_param_names_list = []
    test_ufunc.stypy_varargs_param_name = None
    test_ufunc.stypy_kwargs_param_name = None
    test_ufunc.stypy_call_defaults = defaults
    test_ufunc.stypy_call_varargs = varargs
    test_ufunc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_ufunc', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_ufunc', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_ufunc(...)' code ##################

    
    # Call to assert_array_almost_equal(...): (line 92)
    # Processing the call arguments (line 92)
    
    # Call to lambertw(...): (line 93)
    # Processing the call arguments (line 93)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 93)
    tuple_541342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 93)
    # Adding element type (line 93)
    float_541343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 20), tuple_541342, float_541343)
    # Adding element type (line 93)
    # Getting the type of 'e' (line 93)
    e_541344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'e', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 20), tuple_541342, e_541344)
    # Adding element type (line 93)
    float_541345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 27), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 20), tuple_541342, float_541345)
    
    # Getting the type of 'r_' (line 93)
    r__541346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'r_', False)
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___541347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 17), r__541346, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_541348 = invoke(stypy.reporting.localization.Localization(__file__, 93, 17), getitem___541347, tuple_541342)
    
    # Processing the call keyword arguments (line 93)
    kwargs_541349 = {}
    # Getting the type of 'lambertw' (line 93)
    lambertw_541341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'lambertw', False)
    # Calling lambertw(args, kwargs) (line 93)
    lambertw_call_result_541350 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), lambertw_541341, *[subscript_call_result_541348], **kwargs_541349)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 93)
    tuple_541351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 93)
    # Adding element type (line 93)
    float_541352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 36), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 36), tuple_541351, float_541352)
    # Adding element type (line 93)
    float_541353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 36), tuple_541351, float_541353)
    # Adding element type (line 93)
    float_541354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 44), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 36), tuple_541351, float_541354)
    
    # Getting the type of 'r_' (line 93)
    r__541355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 33), 'r_', False)
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___541356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 33), r__541355, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_541357 = invoke(stypy.reporting.localization.Localization(__file__, 93, 33), getitem___541356, tuple_541351)
    
    # Processing the call keyword arguments (line 92)
    kwargs_541358 = {}
    # Getting the type of 'assert_array_almost_equal' (line 92)
    assert_array_almost_equal_541340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 92)
    assert_array_almost_equal_call_result_541359 = invoke(stypy.reporting.localization.Localization(__file__, 92, 4), assert_array_almost_equal_541340, *[lambertw_call_result_541350, subscript_call_result_541357], **kwargs_541358)
    
    
    # ################# End of 'test_ufunc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_ufunc' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_541360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_541360)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_ufunc'
    return stypy_return_type_541360

# Assigning a type to the variable 'test_ufunc' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'test_ufunc', test_ufunc)

@norecursion
def test_lambertw_ufunc_loop_selection(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_lambertw_ufunc_loop_selection'
    module_type_store = module_type_store.open_function_context('test_lambertw_ufunc_loop_selection', 96, 0, False)
    
    # Passed parameters checking function
    test_lambertw_ufunc_loop_selection.stypy_localization = localization
    test_lambertw_ufunc_loop_selection.stypy_type_of_self = None
    test_lambertw_ufunc_loop_selection.stypy_type_store = module_type_store
    test_lambertw_ufunc_loop_selection.stypy_function_name = 'test_lambertw_ufunc_loop_selection'
    test_lambertw_ufunc_loop_selection.stypy_param_names_list = []
    test_lambertw_ufunc_loop_selection.stypy_varargs_param_name = None
    test_lambertw_ufunc_loop_selection.stypy_kwargs_param_name = None
    test_lambertw_ufunc_loop_selection.stypy_call_defaults = defaults
    test_lambertw_ufunc_loop_selection.stypy_call_varargs = varargs
    test_lambertw_ufunc_loop_selection.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_lambertw_ufunc_loop_selection', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_lambertw_ufunc_loop_selection', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_lambertw_ufunc_loop_selection(...)' code ##################

    
    # Assigning a Call to a Name (line 98):
    
    # Call to dtype(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'np' (line 98)
    np_541363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'np', False)
    # Obtaining the member 'complex128' of a type (line 98)
    complex128_541364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 18), np_541363, 'complex128')
    # Processing the call keyword arguments (line 98)
    kwargs_541365 = {}
    # Getting the type of 'np' (line 98)
    np_541361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 9), 'np', False)
    # Obtaining the member 'dtype' of a type (line 98)
    dtype_541362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 9), np_541361, 'dtype')
    # Calling dtype(args, kwargs) (line 98)
    dtype_call_result_541366 = invoke(stypy.reporting.localization.Localization(__file__, 98, 9), dtype_541362, *[complex128_541364], **kwargs_541365)
    
    # Assigning a type to the variable 'dt' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'dt', dtype_call_result_541366)
    
    # Call to assert_equal(...): (line 99)
    # Processing the call arguments (line 99)
    
    # Call to lambertw(...): (line 99)
    # Processing the call arguments (line 99)
    int_541369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 26), 'int')
    int_541370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 29), 'int')
    int_541371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 32), 'int')
    # Processing the call keyword arguments (line 99)
    kwargs_541372 = {}
    # Getting the type of 'lambertw' (line 99)
    lambertw_541368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'lambertw', False)
    # Calling lambertw(args, kwargs) (line 99)
    lambertw_call_result_541373 = invoke(stypy.reporting.localization.Localization(__file__, 99, 17), lambertw_541368, *[int_541369, int_541370, int_541371], **kwargs_541372)
    
    # Obtaining the member 'dtype' of a type (line 99)
    dtype_541374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 17), lambertw_call_result_541373, 'dtype')
    # Getting the type of 'dt' (line 99)
    dt_541375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 42), 'dt', False)
    # Processing the call keyword arguments (line 99)
    kwargs_541376 = {}
    # Getting the type of 'assert_equal' (line 99)
    assert_equal_541367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 99)
    assert_equal_call_result_541377 = invoke(stypy.reporting.localization.Localization(__file__, 99, 4), assert_equal_541367, *[dtype_541374, dt_541375], **kwargs_541376)
    
    
    # Call to assert_equal(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Call to lambertw(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Obtaining an instance of the builtin type 'list' (line 100)
    list_541380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 100)
    # Adding element type (line 100)
    int_541381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 26), list_541380, int_541381)
    
    int_541382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 31), 'int')
    int_541383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 34), 'int')
    # Processing the call keyword arguments (line 100)
    kwargs_541384 = {}
    # Getting the type of 'lambertw' (line 100)
    lambertw_541379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 17), 'lambertw', False)
    # Calling lambertw(args, kwargs) (line 100)
    lambertw_call_result_541385 = invoke(stypy.reporting.localization.Localization(__file__, 100, 17), lambertw_541379, *[list_541380, int_541382, int_541383], **kwargs_541384)
    
    # Obtaining the member 'dtype' of a type (line 100)
    dtype_541386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 17), lambertw_call_result_541385, 'dtype')
    # Getting the type of 'dt' (line 100)
    dt_541387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 44), 'dt', False)
    # Processing the call keyword arguments (line 100)
    kwargs_541388 = {}
    # Getting the type of 'assert_equal' (line 100)
    assert_equal_541378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 100)
    assert_equal_call_result_541389 = invoke(stypy.reporting.localization.Localization(__file__, 100, 4), assert_equal_541378, *[dtype_541386, dt_541387], **kwargs_541388)
    
    
    # Call to assert_equal(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Call to lambertw(...): (line 101)
    # Processing the call arguments (line 101)
    int_541392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 101)
    list_541393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 101)
    # Adding element type (line 101)
    int_541394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 29), list_541393, int_541394)
    
    int_541395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 34), 'int')
    # Processing the call keyword arguments (line 101)
    kwargs_541396 = {}
    # Getting the type of 'lambertw' (line 101)
    lambertw_541391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'lambertw', False)
    # Calling lambertw(args, kwargs) (line 101)
    lambertw_call_result_541397 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), lambertw_541391, *[int_541392, list_541393, int_541395], **kwargs_541396)
    
    # Obtaining the member 'dtype' of a type (line 101)
    dtype_541398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 17), lambertw_call_result_541397, 'dtype')
    # Getting the type of 'dt' (line 101)
    dt_541399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'dt', False)
    # Processing the call keyword arguments (line 101)
    kwargs_541400 = {}
    # Getting the type of 'assert_equal' (line 101)
    assert_equal_541390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 101)
    assert_equal_call_result_541401 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), assert_equal_541390, *[dtype_541398, dt_541399], **kwargs_541400)
    
    
    # Call to assert_equal(...): (line 102)
    # Processing the call arguments (line 102)
    
    # Call to lambertw(...): (line 102)
    # Processing the call arguments (line 102)
    int_541404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 26), 'int')
    int_541405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 29), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 102)
    list_541406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 102)
    # Adding element type (line 102)
    int_541407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 32), list_541406, int_541407)
    
    # Processing the call keyword arguments (line 102)
    kwargs_541408 = {}
    # Getting the type of 'lambertw' (line 102)
    lambertw_541403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 'lambertw', False)
    # Calling lambertw(args, kwargs) (line 102)
    lambertw_call_result_541409 = invoke(stypy.reporting.localization.Localization(__file__, 102, 17), lambertw_541403, *[int_541404, int_541405, list_541406], **kwargs_541408)
    
    # Obtaining the member 'dtype' of a type (line 102)
    dtype_541410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 17), lambertw_call_result_541409, 'dtype')
    # Getting the type of 'dt' (line 102)
    dt_541411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 44), 'dt', False)
    # Processing the call keyword arguments (line 102)
    kwargs_541412 = {}
    # Getting the type of 'assert_equal' (line 102)
    assert_equal_541402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 102)
    assert_equal_call_result_541413 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), assert_equal_541402, *[dtype_541410, dt_541411], **kwargs_541412)
    
    
    # Call to assert_equal(...): (line 103)
    # Processing the call arguments (line 103)
    
    # Call to lambertw(...): (line 103)
    # Processing the call arguments (line 103)
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_541416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    int_541417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 26), list_541416, int_541417)
    
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_541418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    int_541419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 31), list_541418, int_541419)
    
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_541420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    int_541421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 36), list_541420, int_541421)
    
    # Processing the call keyword arguments (line 103)
    kwargs_541422 = {}
    # Getting the type of 'lambertw' (line 103)
    lambertw_541415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 17), 'lambertw', False)
    # Calling lambertw(args, kwargs) (line 103)
    lambertw_call_result_541423 = invoke(stypy.reporting.localization.Localization(__file__, 103, 17), lambertw_541415, *[list_541416, list_541418, list_541420], **kwargs_541422)
    
    # Obtaining the member 'dtype' of a type (line 103)
    dtype_541424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 17), lambertw_call_result_541423, 'dtype')
    # Getting the type of 'dt' (line 103)
    dt_541425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 48), 'dt', False)
    # Processing the call keyword arguments (line 103)
    kwargs_541426 = {}
    # Getting the type of 'assert_equal' (line 103)
    assert_equal_541414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 103)
    assert_equal_call_result_541427 = invoke(stypy.reporting.localization.Localization(__file__, 103, 4), assert_equal_541414, *[dtype_541424, dt_541425], **kwargs_541426)
    
    
    # ################# End of 'test_lambertw_ufunc_loop_selection(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_lambertw_ufunc_loop_selection' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_541428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_541428)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_lambertw_ufunc_loop_selection'
    return stypy_return_type_541428

# Assigning a type to the variable 'test_lambertw_ufunc_loop_selection' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'test_lambertw_ufunc_loop_selection', test_lambertw_ufunc_loop_selection)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
