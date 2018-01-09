
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_allclose
5: 
6: from scipy import ndimage
7: from scipy.ndimage import _ctest
8: from scipy.ndimage import _ctest_oldapi
9: from scipy.ndimage import _cytest
10: from scipy._lib._ccallback import LowLevelCallable
11: 
12: FILTER1D_FUNCTIONS = [
13:     lambda filter_size: _ctest.filter1d(filter_size),
14:     lambda filter_size: _ctest_oldapi.filter1d(filter_size),
15:     lambda filter_size: _cytest.filter1d(filter_size, with_signature=False),
16:     lambda filter_size: LowLevelCallable(_cytest.filter1d(filter_size, with_signature=True)),
17:     lambda filter_size: LowLevelCallable.from_cython(_cytest, "_filter1d",
18:                                                      _cytest.filter1d_capsule(filter_size)),
19: ]
20: 
21: FILTER2D_FUNCTIONS = [
22:     lambda weights: _ctest.filter2d(weights),
23:     lambda weights: _ctest_oldapi.filter2d(weights),
24:     lambda weights: _cytest.filter2d(weights, with_signature=False),
25:     lambda weights: LowLevelCallable(_cytest.filter2d(weights, with_signature=True)),
26:     lambda weights: LowLevelCallable.from_cython(_cytest, "_filter2d", _cytest.filter2d_capsule(weights)),
27: ]
28: 
29: TRANSFORM_FUNCTIONS = [
30:     lambda shift: _ctest.transform(shift),
31:     lambda shift: _ctest_oldapi.transform(shift),
32:     lambda shift: _cytest.transform(shift, with_signature=False),
33:     lambda shift: LowLevelCallable(_cytest.transform(shift, with_signature=True)),
34:     lambda shift: LowLevelCallable.from_cython(_cytest, "_transform", _cytest.transform_capsule(shift)),
35: ]
36: 
37: 
38: def test_generic_filter():
39:     def filter2d(footprint_elements, weights):
40:         return (weights*footprint_elements).sum()
41: 
42:     def check(j):
43:         func = FILTER2D_FUNCTIONS[j]
44: 
45:         im = np.ones((20, 20))
46:         im[:10,:10] = 0
47:         footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
48:         footprint_size = np.count_nonzero(footprint)
49:         weights = np.ones(footprint_size)/footprint_size
50: 
51:         res = ndimage.generic_filter(im, func(weights),
52:                                      footprint=footprint)
53:         std = ndimage.generic_filter(im, filter2d, footprint=footprint,
54:                                      extra_arguments=(weights,))
55:         assert_allclose(res, std, err_msg="#{} failed".format(j))
56: 
57:     for j, func in enumerate(FILTER2D_FUNCTIONS):
58:         check(j)
59: 
60: 
61: def test_generic_filter1d():
62:     def filter1d(input_line, output_line, filter_size):
63:         for i in range(output_line.size):
64:             output_line[i] = 0
65:             for j in range(filter_size):
66:                 output_line[i] += input_line[i+j]
67:         output_line /= filter_size
68: 
69:     def check(j):
70:         func = FILTER1D_FUNCTIONS[j]
71: 
72:         im = np.tile(np.hstack((np.zeros(10), np.ones(10))), (10, 1))
73:         filter_size = 3
74: 
75:         res = ndimage.generic_filter1d(im, func(filter_size),
76:                                        filter_size)
77:         std = ndimage.generic_filter1d(im, filter1d, filter_size,
78:                                        extra_arguments=(filter_size,))
79:         assert_allclose(res, std, err_msg="#{} failed".format(j))
80: 
81:     for j, func in enumerate(FILTER1D_FUNCTIONS):
82:         check(j)
83: 
84: 
85: def test_geometric_transform():
86:     def transform(output_coordinates, shift):
87:         return output_coordinates[0] - shift, output_coordinates[1] - shift
88: 
89:     def check(j):
90:         func = TRANSFORM_FUNCTIONS[j]
91: 
92:         im = np.arange(12).reshape(4, 3).astype(np.float64)
93:         shift = 0.5
94: 
95:         res = ndimage.geometric_transform(im, func(shift))
96:         std = ndimage.geometric_transform(im, transform, extra_arguments=(shift,))
97:         assert_allclose(res, std, err_msg="#{} failed".format(j))
98: 
99:     for j, func in enumerate(TRANSFORM_FUNCTIONS):
100:         check(j)
101: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_126846 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_126846) is not StypyTypeError):

    if (import_126846 != 'pyd_module'):
        __import__(import_126846)
        sys_modules_126847 = sys.modules[import_126846]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_126847.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_126846)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_allclose' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_126848 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_126848) is not StypyTypeError):

    if (import_126848 != 'pyd_module'):
        __import__(import_126848)
        sys_modules_126849 = sys.modules[import_126848]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_126849.module_type_store, module_type_store, ['assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_126849, sys_modules_126849.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_allclose'], [assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_126848)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy import ndimage' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_126850 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy')

if (type(import_126850) is not StypyTypeError):

    if (import_126850 != 'pyd_module'):
        __import__(import_126850)
        sys_modules_126851 = sys.modules[import_126850]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy', sys_modules_126851.module_type_store, module_type_store, ['ndimage'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_126851, sys_modules_126851.module_type_store, module_type_store)
    else:
        from scipy import ndimage

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy', None, module_type_store, ['ndimage'], [ndimage])

else:
    # Assigning a type to the variable 'scipy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy', import_126850)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.ndimage import _ctest' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_126852 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.ndimage')

if (type(import_126852) is not StypyTypeError):

    if (import_126852 != 'pyd_module'):
        __import__(import_126852)
        sys_modules_126853 = sys.modules[import_126852]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.ndimage', sys_modules_126853.module_type_store, module_type_store, ['_ctest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_126853, sys_modules_126853.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _ctest

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.ndimage', None, module_type_store, ['_ctest'], [_ctest])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.ndimage', import_126852)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.ndimage import _ctest_oldapi' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_126854 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.ndimage')

if (type(import_126854) is not StypyTypeError):

    if (import_126854 != 'pyd_module'):
        __import__(import_126854)
        sys_modules_126855 = sys.modules[import_126854]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.ndimage', sys_modules_126855.module_type_store, module_type_store, ['_ctest_oldapi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_126855, sys_modules_126855.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _ctest_oldapi

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.ndimage', None, module_type_store, ['_ctest_oldapi'], [_ctest_oldapi])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.ndimage', import_126854)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.ndimage import _cytest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_126856 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.ndimage')

if (type(import_126856) is not StypyTypeError):

    if (import_126856 != 'pyd_module'):
        __import__(import_126856)
        sys_modules_126857 = sys.modules[import_126856]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.ndimage', sys_modules_126857.module_type_store, module_type_store, ['_cytest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_126857, sys_modules_126857.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _cytest

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.ndimage', None, module_type_store, ['_cytest'], [_cytest])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.ndimage', import_126856)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib._ccallback import LowLevelCallable' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_126858 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._ccallback')

if (type(import_126858) is not StypyTypeError):

    if (import_126858 != 'pyd_module'):
        __import__(import_126858)
        sys_modules_126859 = sys.modules[import_126858]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._ccallback', sys_modules_126859.module_type_store, module_type_store, ['LowLevelCallable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_126859, sys_modules_126859.module_type_store, module_type_store)
    else:
        from scipy._lib._ccallback import LowLevelCallable

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._ccallback', None, module_type_store, ['LowLevelCallable'], [LowLevelCallable])

else:
    # Assigning a type to the variable 'scipy._lib._ccallback' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._ccallback', import_126858)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')


# Assigning a List to a Name (line 12):

# Obtaining an instance of the builtin type 'list' (line 12)
list_126860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)

@norecursion
def _stypy_temp_lambda_30(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_30'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_30', 13, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_30.stypy_localization = localization
    _stypy_temp_lambda_30.stypy_type_of_self = None
    _stypy_temp_lambda_30.stypy_type_store = module_type_store
    _stypy_temp_lambda_30.stypy_function_name = '_stypy_temp_lambda_30'
    _stypy_temp_lambda_30.stypy_param_names_list = ['filter_size']
    _stypy_temp_lambda_30.stypy_varargs_param_name = None
    _stypy_temp_lambda_30.stypy_kwargs_param_name = None
    _stypy_temp_lambda_30.stypy_call_defaults = defaults
    _stypy_temp_lambda_30.stypy_call_varargs = varargs
    _stypy_temp_lambda_30.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_30', ['filter_size'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_30', ['filter_size'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to filter1d(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'filter_size' (line 13)
    filter_size_126863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 40), 'filter_size', False)
    # Processing the call keyword arguments (line 13)
    kwargs_126864 = {}
    # Getting the type of '_ctest' (line 13)
    _ctest_126861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 24), '_ctest', False)
    # Obtaining the member 'filter1d' of a type (line 13)
    filter1d_126862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 24), _ctest_126861, 'filter1d')
    # Calling filter1d(args, kwargs) (line 13)
    filter1d_call_result_126865 = invoke(stypy.reporting.localization.Localization(__file__, 13, 24), filter1d_126862, *[filter_size_126863], **kwargs_126864)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type', filter1d_call_result_126865)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_30' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_126866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126866)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_30'
    return stypy_return_type_126866

# Assigning a type to the variable '_stypy_temp_lambda_30' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), '_stypy_temp_lambda_30', _stypy_temp_lambda_30)
# Getting the type of '_stypy_temp_lambda_30' (line 13)
_stypy_temp_lambda_30_126867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), '_stypy_temp_lambda_30')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), list_126860, _stypy_temp_lambda_30_126867)
# Adding element type (line 12)

@norecursion
def _stypy_temp_lambda_31(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_31'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_31', 14, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_31.stypy_localization = localization
    _stypy_temp_lambda_31.stypy_type_of_self = None
    _stypy_temp_lambda_31.stypy_type_store = module_type_store
    _stypy_temp_lambda_31.stypy_function_name = '_stypy_temp_lambda_31'
    _stypy_temp_lambda_31.stypy_param_names_list = ['filter_size']
    _stypy_temp_lambda_31.stypy_varargs_param_name = None
    _stypy_temp_lambda_31.stypy_kwargs_param_name = None
    _stypy_temp_lambda_31.stypy_call_defaults = defaults
    _stypy_temp_lambda_31.stypy_call_varargs = varargs
    _stypy_temp_lambda_31.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_31', ['filter_size'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_31', ['filter_size'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to filter1d(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'filter_size' (line 14)
    filter_size_126870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 47), 'filter_size', False)
    # Processing the call keyword arguments (line 14)
    kwargs_126871 = {}
    # Getting the type of '_ctest_oldapi' (line 14)
    _ctest_oldapi_126868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 24), '_ctest_oldapi', False)
    # Obtaining the member 'filter1d' of a type (line 14)
    filter1d_126869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 24), _ctest_oldapi_126868, 'filter1d')
    # Calling filter1d(args, kwargs) (line 14)
    filter1d_call_result_126872 = invoke(stypy.reporting.localization.Localization(__file__, 14, 24), filter1d_126869, *[filter_size_126870], **kwargs_126871)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type', filter1d_call_result_126872)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_31' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_126873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126873)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_31'
    return stypy_return_type_126873

# Assigning a type to the variable '_stypy_temp_lambda_31' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), '_stypy_temp_lambda_31', _stypy_temp_lambda_31)
# Getting the type of '_stypy_temp_lambda_31' (line 14)
_stypy_temp_lambda_31_126874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), '_stypy_temp_lambda_31')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), list_126860, _stypy_temp_lambda_31_126874)
# Adding element type (line 12)

@norecursion
def _stypy_temp_lambda_32(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_32'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_32', 15, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_32.stypy_localization = localization
    _stypy_temp_lambda_32.stypy_type_of_self = None
    _stypy_temp_lambda_32.stypy_type_store = module_type_store
    _stypy_temp_lambda_32.stypy_function_name = '_stypy_temp_lambda_32'
    _stypy_temp_lambda_32.stypy_param_names_list = ['filter_size']
    _stypy_temp_lambda_32.stypy_varargs_param_name = None
    _stypy_temp_lambda_32.stypy_kwargs_param_name = None
    _stypy_temp_lambda_32.stypy_call_defaults = defaults
    _stypy_temp_lambda_32.stypy_call_varargs = varargs
    _stypy_temp_lambda_32.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_32', ['filter_size'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_32', ['filter_size'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to filter1d(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'filter_size' (line 15)
    filter_size_126877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 41), 'filter_size', False)
    # Processing the call keyword arguments (line 15)
    # Getting the type of 'False' (line 15)
    False_126878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 69), 'False', False)
    keyword_126879 = False_126878
    kwargs_126880 = {'with_signature': keyword_126879}
    # Getting the type of '_cytest' (line 15)
    _cytest_126875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), '_cytest', False)
    # Obtaining the member 'filter1d' of a type (line 15)
    filter1d_126876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 24), _cytest_126875, 'filter1d')
    # Calling filter1d(args, kwargs) (line 15)
    filter1d_call_result_126881 = invoke(stypy.reporting.localization.Localization(__file__, 15, 24), filter1d_126876, *[filter_size_126877], **kwargs_126880)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type', filter1d_call_result_126881)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_32' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_126882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126882)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_32'
    return stypy_return_type_126882

# Assigning a type to the variable '_stypy_temp_lambda_32' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), '_stypy_temp_lambda_32', _stypy_temp_lambda_32)
# Getting the type of '_stypy_temp_lambda_32' (line 15)
_stypy_temp_lambda_32_126883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), '_stypy_temp_lambda_32')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), list_126860, _stypy_temp_lambda_32_126883)
# Adding element type (line 12)

@norecursion
def _stypy_temp_lambda_33(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_33'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_33', 16, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_33.stypy_localization = localization
    _stypy_temp_lambda_33.stypy_type_of_self = None
    _stypy_temp_lambda_33.stypy_type_store = module_type_store
    _stypy_temp_lambda_33.stypy_function_name = '_stypy_temp_lambda_33'
    _stypy_temp_lambda_33.stypy_param_names_list = ['filter_size']
    _stypy_temp_lambda_33.stypy_varargs_param_name = None
    _stypy_temp_lambda_33.stypy_kwargs_param_name = None
    _stypy_temp_lambda_33.stypy_call_defaults = defaults
    _stypy_temp_lambda_33.stypy_call_varargs = varargs
    _stypy_temp_lambda_33.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_33', ['filter_size'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_33', ['filter_size'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to LowLevelCallable(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Call to filter1d(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'filter_size' (line 16)
    filter_size_126887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 58), 'filter_size', False)
    # Processing the call keyword arguments (line 16)
    # Getting the type of 'True' (line 16)
    True_126888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 86), 'True', False)
    keyword_126889 = True_126888
    kwargs_126890 = {'with_signature': keyword_126889}
    # Getting the type of '_cytest' (line 16)
    _cytest_126885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 41), '_cytest', False)
    # Obtaining the member 'filter1d' of a type (line 16)
    filter1d_126886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 41), _cytest_126885, 'filter1d')
    # Calling filter1d(args, kwargs) (line 16)
    filter1d_call_result_126891 = invoke(stypy.reporting.localization.Localization(__file__, 16, 41), filter1d_126886, *[filter_size_126887], **kwargs_126890)
    
    # Processing the call keyword arguments (line 16)
    kwargs_126892 = {}
    # Getting the type of 'LowLevelCallable' (line 16)
    LowLevelCallable_126884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 24), 'LowLevelCallable', False)
    # Calling LowLevelCallable(args, kwargs) (line 16)
    LowLevelCallable_call_result_126893 = invoke(stypy.reporting.localization.Localization(__file__, 16, 24), LowLevelCallable_126884, *[filter1d_call_result_126891], **kwargs_126892)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type', LowLevelCallable_call_result_126893)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_33' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_126894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126894)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_33'
    return stypy_return_type_126894

# Assigning a type to the variable '_stypy_temp_lambda_33' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), '_stypy_temp_lambda_33', _stypy_temp_lambda_33)
# Getting the type of '_stypy_temp_lambda_33' (line 16)
_stypy_temp_lambda_33_126895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), '_stypy_temp_lambda_33')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), list_126860, _stypy_temp_lambda_33_126895)
# Adding element type (line 12)

@norecursion
def _stypy_temp_lambda_34(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_34'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_34', 17, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_34.stypy_localization = localization
    _stypy_temp_lambda_34.stypy_type_of_self = None
    _stypy_temp_lambda_34.stypy_type_store = module_type_store
    _stypy_temp_lambda_34.stypy_function_name = '_stypy_temp_lambda_34'
    _stypy_temp_lambda_34.stypy_param_names_list = ['filter_size']
    _stypy_temp_lambda_34.stypy_varargs_param_name = None
    _stypy_temp_lambda_34.stypy_kwargs_param_name = None
    _stypy_temp_lambda_34.stypy_call_defaults = defaults
    _stypy_temp_lambda_34.stypy_call_varargs = varargs
    _stypy_temp_lambda_34.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_34', ['filter_size'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_34', ['filter_size'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to from_cython(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of '_cytest' (line 17)
    _cytest_126898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 53), '_cytest', False)
    str_126899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 62), 'str', '_filter1d')
    
    # Call to filter1d_capsule(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'filter_size' (line 18)
    filter_size_126902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 78), 'filter_size', False)
    # Processing the call keyword arguments (line 18)
    kwargs_126903 = {}
    # Getting the type of '_cytest' (line 18)
    _cytest_126900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 53), '_cytest', False)
    # Obtaining the member 'filter1d_capsule' of a type (line 18)
    filter1d_capsule_126901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 53), _cytest_126900, 'filter1d_capsule')
    # Calling filter1d_capsule(args, kwargs) (line 18)
    filter1d_capsule_call_result_126904 = invoke(stypy.reporting.localization.Localization(__file__, 18, 53), filter1d_capsule_126901, *[filter_size_126902], **kwargs_126903)
    
    # Processing the call keyword arguments (line 17)
    kwargs_126905 = {}
    # Getting the type of 'LowLevelCallable' (line 17)
    LowLevelCallable_126896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'LowLevelCallable', False)
    # Obtaining the member 'from_cython' of a type (line 17)
    from_cython_126897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 24), LowLevelCallable_126896, 'from_cython')
    # Calling from_cython(args, kwargs) (line 17)
    from_cython_call_result_126906 = invoke(stypy.reporting.localization.Localization(__file__, 17, 24), from_cython_126897, *[_cytest_126898, str_126899, filter1d_capsule_call_result_126904], **kwargs_126905)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type', from_cython_call_result_126906)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_34' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_126907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126907)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_34'
    return stypy_return_type_126907

# Assigning a type to the variable '_stypy_temp_lambda_34' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), '_stypy_temp_lambda_34', _stypy_temp_lambda_34)
# Getting the type of '_stypy_temp_lambda_34' (line 17)
_stypy_temp_lambda_34_126908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), '_stypy_temp_lambda_34')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), list_126860, _stypy_temp_lambda_34_126908)

# Assigning a type to the variable 'FILTER1D_FUNCTIONS' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'FILTER1D_FUNCTIONS', list_126860)

# Assigning a List to a Name (line 21):

# Obtaining an instance of the builtin type 'list' (line 21)
list_126909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)

@norecursion
def _stypy_temp_lambda_35(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_35'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_35', 22, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_35.stypy_localization = localization
    _stypy_temp_lambda_35.stypy_type_of_self = None
    _stypy_temp_lambda_35.stypy_type_store = module_type_store
    _stypy_temp_lambda_35.stypy_function_name = '_stypy_temp_lambda_35'
    _stypy_temp_lambda_35.stypy_param_names_list = ['weights']
    _stypy_temp_lambda_35.stypy_varargs_param_name = None
    _stypy_temp_lambda_35.stypy_kwargs_param_name = None
    _stypy_temp_lambda_35.stypy_call_defaults = defaults
    _stypy_temp_lambda_35.stypy_call_varargs = varargs
    _stypy_temp_lambda_35.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_35', ['weights'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_35', ['weights'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to filter2d(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'weights' (line 22)
    weights_126912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 36), 'weights', False)
    # Processing the call keyword arguments (line 22)
    kwargs_126913 = {}
    # Getting the type of '_ctest' (line 22)
    _ctest_126910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), '_ctest', False)
    # Obtaining the member 'filter2d' of a type (line 22)
    filter2d_126911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 20), _ctest_126910, 'filter2d')
    # Calling filter2d(args, kwargs) (line 22)
    filter2d_call_result_126914 = invoke(stypy.reporting.localization.Localization(__file__, 22, 20), filter2d_126911, *[weights_126912], **kwargs_126913)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type', filter2d_call_result_126914)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_35' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_126915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126915)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_35'
    return stypy_return_type_126915

# Assigning a type to the variable '_stypy_temp_lambda_35' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), '_stypy_temp_lambda_35', _stypy_temp_lambda_35)
# Getting the type of '_stypy_temp_lambda_35' (line 22)
_stypy_temp_lambda_35_126916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), '_stypy_temp_lambda_35')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 21), list_126909, _stypy_temp_lambda_35_126916)
# Adding element type (line 21)

@norecursion
def _stypy_temp_lambda_36(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_36'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_36', 23, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_36.stypy_localization = localization
    _stypy_temp_lambda_36.stypy_type_of_self = None
    _stypy_temp_lambda_36.stypy_type_store = module_type_store
    _stypy_temp_lambda_36.stypy_function_name = '_stypy_temp_lambda_36'
    _stypy_temp_lambda_36.stypy_param_names_list = ['weights']
    _stypy_temp_lambda_36.stypy_varargs_param_name = None
    _stypy_temp_lambda_36.stypy_kwargs_param_name = None
    _stypy_temp_lambda_36.stypy_call_defaults = defaults
    _stypy_temp_lambda_36.stypy_call_varargs = varargs
    _stypy_temp_lambda_36.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_36', ['weights'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_36', ['weights'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to filter2d(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'weights' (line 23)
    weights_126919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 43), 'weights', False)
    # Processing the call keyword arguments (line 23)
    kwargs_126920 = {}
    # Getting the type of '_ctest_oldapi' (line 23)
    _ctest_oldapi_126917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), '_ctest_oldapi', False)
    # Obtaining the member 'filter2d' of a type (line 23)
    filter2d_126918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 20), _ctest_oldapi_126917, 'filter2d')
    # Calling filter2d(args, kwargs) (line 23)
    filter2d_call_result_126921 = invoke(stypy.reporting.localization.Localization(__file__, 23, 20), filter2d_126918, *[weights_126919], **kwargs_126920)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type', filter2d_call_result_126921)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_36' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_126922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126922)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_36'
    return stypy_return_type_126922

# Assigning a type to the variable '_stypy_temp_lambda_36' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), '_stypy_temp_lambda_36', _stypy_temp_lambda_36)
# Getting the type of '_stypy_temp_lambda_36' (line 23)
_stypy_temp_lambda_36_126923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), '_stypy_temp_lambda_36')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 21), list_126909, _stypy_temp_lambda_36_126923)
# Adding element type (line 21)

@norecursion
def _stypy_temp_lambda_37(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_37'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_37', 24, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_37.stypy_localization = localization
    _stypy_temp_lambda_37.stypy_type_of_self = None
    _stypy_temp_lambda_37.stypy_type_store = module_type_store
    _stypy_temp_lambda_37.stypy_function_name = '_stypy_temp_lambda_37'
    _stypy_temp_lambda_37.stypy_param_names_list = ['weights']
    _stypy_temp_lambda_37.stypy_varargs_param_name = None
    _stypy_temp_lambda_37.stypy_kwargs_param_name = None
    _stypy_temp_lambda_37.stypy_call_defaults = defaults
    _stypy_temp_lambda_37.stypy_call_varargs = varargs
    _stypy_temp_lambda_37.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_37', ['weights'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_37', ['weights'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to filter2d(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'weights' (line 24)
    weights_126926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 37), 'weights', False)
    # Processing the call keyword arguments (line 24)
    # Getting the type of 'False' (line 24)
    False_126927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 61), 'False', False)
    keyword_126928 = False_126927
    kwargs_126929 = {'with_signature': keyword_126928}
    # Getting the type of '_cytest' (line 24)
    _cytest_126924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), '_cytest', False)
    # Obtaining the member 'filter2d' of a type (line 24)
    filter2d_126925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 20), _cytest_126924, 'filter2d')
    # Calling filter2d(args, kwargs) (line 24)
    filter2d_call_result_126930 = invoke(stypy.reporting.localization.Localization(__file__, 24, 20), filter2d_126925, *[weights_126926], **kwargs_126929)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type', filter2d_call_result_126930)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_37' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_126931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126931)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_37'
    return stypy_return_type_126931

# Assigning a type to the variable '_stypy_temp_lambda_37' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), '_stypy_temp_lambda_37', _stypy_temp_lambda_37)
# Getting the type of '_stypy_temp_lambda_37' (line 24)
_stypy_temp_lambda_37_126932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), '_stypy_temp_lambda_37')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 21), list_126909, _stypy_temp_lambda_37_126932)
# Adding element type (line 21)

@norecursion
def _stypy_temp_lambda_38(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_38'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_38', 25, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_38.stypy_localization = localization
    _stypy_temp_lambda_38.stypy_type_of_self = None
    _stypy_temp_lambda_38.stypy_type_store = module_type_store
    _stypy_temp_lambda_38.stypy_function_name = '_stypy_temp_lambda_38'
    _stypy_temp_lambda_38.stypy_param_names_list = ['weights']
    _stypy_temp_lambda_38.stypy_varargs_param_name = None
    _stypy_temp_lambda_38.stypy_kwargs_param_name = None
    _stypy_temp_lambda_38.stypy_call_defaults = defaults
    _stypy_temp_lambda_38.stypy_call_varargs = varargs
    _stypy_temp_lambda_38.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_38', ['weights'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_38', ['weights'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to LowLevelCallable(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Call to filter2d(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'weights' (line 25)
    weights_126936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 54), 'weights', False)
    # Processing the call keyword arguments (line 25)
    # Getting the type of 'True' (line 25)
    True_126937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 78), 'True', False)
    keyword_126938 = True_126937
    kwargs_126939 = {'with_signature': keyword_126938}
    # Getting the type of '_cytest' (line 25)
    _cytest_126934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 37), '_cytest', False)
    # Obtaining the member 'filter2d' of a type (line 25)
    filter2d_126935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 37), _cytest_126934, 'filter2d')
    # Calling filter2d(args, kwargs) (line 25)
    filter2d_call_result_126940 = invoke(stypy.reporting.localization.Localization(__file__, 25, 37), filter2d_126935, *[weights_126936], **kwargs_126939)
    
    # Processing the call keyword arguments (line 25)
    kwargs_126941 = {}
    # Getting the type of 'LowLevelCallable' (line 25)
    LowLevelCallable_126933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'LowLevelCallable', False)
    # Calling LowLevelCallable(args, kwargs) (line 25)
    LowLevelCallable_call_result_126942 = invoke(stypy.reporting.localization.Localization(__file__, 25, 20), LowLevelCallable_126933, *[filter2d_call_result_126940], **kwargs_126941)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type', LowLevelCallable_call_result_126942)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_38' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_126943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126943)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_38'
    return stypy_return_type_126943

# Assigning a type to the variable '_stypy_temp_lambda_38' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), '_stypy_temp_lambda_38', _stypy_temp_lambda_38)
# Getting the type of '_stypy_temp_lambda_38' (line 25)
_stypy_temp_lambda_38_126944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), '_stypy_temp_lambda_38')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 21), list_126909, _stypy_temp_lambda_38_126944)
# Adding element type (line 21)

@norecursion
def _stypy_temp_lambda_39(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_39'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_39', 26, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_39.stypy_localization = localization
    _stypy_temp_lambda_39.stypy_type_of_self = None
    _stypy_temp_lambda_39.stypy_type_store = module_type_store
    _stypy_temp_lambda_39.stypy_function_name = '_stypy_temp_lambda_39'
    _stypy_temp_lambda_39.stypy_param_names_list = ['weights']
    _stypy_temp_lambda_39.stypy_varargs_param_name = None
    _stypy_temp_lambda_39.stypy_kwargs_param_name = None
    _stypy_temp_lambda_39.stypy_call_defaults = defaults
    _stypy_temp_lambda_39.stypy_call_varargs = varargs
    _stypy_temp_lambda_39.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_39', ['weights'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_39', ['weights'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to from_cython(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of '_cytest' (line 26)
    _cytest_126947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 49), '_cytest', False)
    str_126948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 58), 'str', '_filter2d')
    
    # Call to filter2d_capsule(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'weights' (line 26)
    weights_126951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 96), 'weights', False)
    # Processing the call keyword arguments (line 26)
    kwargs_126952 = {}
    # Getting the type of '_cytest' (line 26)
    _cytest_126949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 71), '_cytest', False)
    # Obtaining the member 'filter2d_capsule' of a type (line 26)
    filter2d_capsule_126950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 71), _cytest_126949, 'filter2d_capsule')
    # Calling filter2d_capsule(args, kwargs) (line 26)
    filter2d_capsule_call_result_126953 = invoke(stypy.reporting.localization.Localization(__file__, 26, 71), filter2d_capsule_126950, *[weights_126951], **kwargs_126952)
    
    # Processing the call keyword arguments (line 26)
    kwargs_126954 = {}
    # Getting the type of 'LowLevelCallable' (line 26)
    LowLevelCallable_126945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'LowLevelCallable', False)
    # Obtaining the member 'from_cython' of a type (line 26)
    from_cython_126946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 20), LowLevelCallable_126945, 'from_cython')
    # Calling from_cython(args, kwargs) (line 26)
    from_cython_call_result_126955 = invoke(stypy.reporting.localization.Localization(__file__, 26, 20), from_cython_126946, *[_cytest_126947, str_126948, filter2d_capsule_call_result_126953], **kwargs_126954)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', from_cython_call_result_126955)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_39' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_126956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126956)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_39'
    return stypy_return_type_126956

# Assigning a type to the variable '_stypy_temp_lambda_39' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), '_stypy_temp_lambda_39', _stypy_temp_lambda_39)
# Getting the type of '_stypy_temp_lambda_39' (line 26)
_stypy_temp_lambda_39_126957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), '_stypy_temp_lambda_39')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 21), list_126909, _stypy_temp_lambda_39_126957)

# Assigning a type to the variable 'FILTER2D_FUNCTIONS' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'FILTER2D_FUNCTIONS', list_126909)

# Assigning a List to a Name (line 29):

# Obtaining an instance of the builtin type 'list' (line 29)
list_126958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)

@norecursion
def _stypy_temp_lambda_40(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_40'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_40', 30, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_40.stypy_localization = localization
    _stypy_temp_lambda_40.stypy_type_of_self = None
    _stypy_temp_lambda_40.stypy_type_store = module_type_store
    _stypy_temp_lambda_40.stypy_function_name = '_stypy_temp_lambda_40'
    _stypy_temp_lambda_40.stypy_param_names_list = ['shift']
    _stypy_temp_lambda_40.stypy_varargs_param_name = None
    _stypy_temp_lambda_40.stypy_kwargs_param_name = None
    _stypy_temp_lambda_40.stypy_call_defaults = defaults
    _stypy_temp_lambda_40.stypy_call_varargs = varargs
    _stypy_temp_lambda_40.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_40', ['shift'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_40', ['shift'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to transform(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'shift' (line 30)
    shift_126961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 35), 'shift', False)
    # Processing the call keyword arguments (line 30)
    kwargs_126962 = {}
    # Getting the type of '_ctest' (line 30)
    _ctest_126959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 18), '_ctest', False)
    # Obtaining the member 'transform' of a type (line 30)
    transform_126960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 18), _ctest_126959, 'transform')
    # Calling transform(args, kwargs) (line 30)
    transform_call_result_126963 = invoke(stypy.reporting.localization.Localization(__file__, 30, 18), transform_126960, *[shift_126961], **kwargs_126962)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type', transform_call_result_126963)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_40' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_126964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126964)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_40'
    return stypy_return_type_126964

# Assigning a type to the variable '_stypy_temp_lambda_40' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), '_stypy_temp_lambda_40', _stypy_temp_lambda_40)
# Getting the type of '_stypy_temp_lambda_40' (line 30)
_stypy_temp_lambda_40_126965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), '_stypy_temp_lambda_40')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 22), list_126958, _stypy_temp_lambda_40_126965)
# Adding element type (line 29)

@norecursion
def _stypy_temp_lambda_41(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_41'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_41', 31, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_41.stypy_localization = localization
    _stypy_temp_lambda_41.stypy_type_of_self = None
    _stypy_temp_lambda_41.stypy_type_store = module_type_store
    _stypy_temp_lambda_41.stypy_function_name = '_stypy_temp_lambda_41'
    _stypy_temp_lambda_41.stypy_param_names_list = ['shift']
    _stypy_temp_lambda_41.stypy_varargs_param_name = None
    _stypy_temp_lambda_41.stypy_kwargs_param_name = None
    _stypy_temp_lambda_41.stypy_call_defaults = defaults
    _stypy_temp_lambda_41.stypy_call_varargs = varargs
    _stypy_temp_lambda_41.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_41', ['shift'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_41', ['shift'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to transform(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'shift' (line 31)
    shift_126968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 42), 'shift', False)
    # Processing the call keyword arguments (line 31)
    kwargs_126969 = {}
    # Getting the type of '_ctest_oldapi' (line 31)
    _ctest_oldapi_126966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), '_ctest_oldapi', False)
    # Obtaining the member 'transform' of a type (line 31)
    transform_126967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 18), _ctest_oldapi_126966, 'transform')
    # Calling transform(args, kwargs) (line 31)
    transform_call_result_126970 = invoke(stypy.reporting.localization.Localization(__file__, 31, 18), transform_126967, *[shift_126968], **kwargs_126969)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type', transform_call_result_126970)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_41' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_126971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126971)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_41'
    return stypy_return_type_126971

# Assigning a type to the variable '_stypy_temp_lambda_41' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), '_stypy_temp_lambda_41', _stypy_temp_lambda_41)
# Getting the type of '_stypy_temp_lambda_41' (line 31)
_stypy_temp_lambda_41_126972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), '_stypy_temp_lambda_41')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 22), list_126958, _stypy_temp_lambda_41_126972)
# Adding element type (line 29)

@norecursion
def _stypy_temp_lambda_42(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_42'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_42', 32, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_42.stypy_localization = localization
    _stypy_temp_lambda_42.stypy_type_of_self = None
    _stypy_temp_lambda_42.stypy_type_store = module_type_store
    _stypy_temp_lambda_42.stypy_function_name = '_stypy_temp_lambda_42'
    _stypy_temp_lambda_42.stypy_param_names_list = ['shift']
    _stypy_temp_lambda_42.stypy_varargs_param_name = None
    _stypy_temp_lambda_42.stypy_kwargs_param_name = None
    _stypy_temp_lambda_42.stypy_call_defaults = defaults
    _stypy_temp_lambda_42.stypy_call_varargs = varargs
    _stypy_temp_lambda_42.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_42', ['shift'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_42', ['shift'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to transform(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'shift' (line 32)
    shift_126975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 36), 'shift', False)
    # Processing the call keyword arguments (line 32)
    # Getting the type of 'False' (line 32)
    False_126976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 58), 'False', False)
    keyword_126977 = False_126976
    kwargs_126978 = {'with_signature': keyword_126977}
    # Getting the type of '_cytest' (line 32)
    _cytest_126973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 18), '_cytest', False)
    # Obtaining the member 'transform' of a type (line 32)
    transform_126974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 18), _cytest_126973, 'transform')
    # Calling transform(args, kwargs) (line 32)
    transform_call_result_126979 = invoke(stypy.reporting.localization.Localization(__file__, 32, 18), transform_126974, *[shift_126975], **kwargs_126978)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type', transform_call_result_126979)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_42' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_126980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126980)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_42'
    return stypy_return_type_126980

# Assigning a type to the variable '_stypy_temp_lambda_42' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), '_stypy_temp_lambda_42', _stypy_temp_lambda_42)
# Getting the type of '_stypy_temp_lambda_42' (line 32)
_stypy_temp_lambda_42_126981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), '_stypy_temp_lambda_42')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 22), list_126958, _stypy_temp_lambda_42_126981)
# Adding element type (line 29)

@norecursion
def _stypy_temp_lambda_43(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_43'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_43', 33, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_43.stypy_localization = localization
    _stypy_temp_lambda_43.stypy_type_of_self = None
    _stypy_temp_lambda_43.stypy_type_store = module_type_store
    _stypy_temp_lambda_43.stypy_function_name = '_stypy_temp_lambda_43'
    _stypy_temp_lambda_43.stypy_param_names_list = ['shift']
    _stypy_temp_lambda_43.stypy_varargs_param_name = None
    _stypy_temp_lambda_43.stypy_kwargs_param_name = None
    _stypy_temp_lambda_43.stypy_call_defaults = defaults
    _stypy_temp_lambda_43.stypy_call_varargs = varargs
    _stypy_temp_lambda_43.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_43', ['shift'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_43', ['shift'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to LowLevelCallable(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Call to transform(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'shift' (line 33)
    shift_126985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 53), 'shift', False)
    # Processing the call keyword arguments (line 33)
    # Getting the type of 'True' (line 33)
    True_126986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 75), 'True', False)
    keyword_126987 = True_126986
    kwargs_126988 = {'with_signature': keyword_126987}
    # Getting the type of '_cytest' (line 33)
    _cytest_126983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 35), '_cytest', False)
    # Obtaining the member 'transform' of a type (line 33)
    transform_126984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 35), _cytest_126983, 'transform')
    # Calling transform(args, kwargs) (line 33)
    transform_call_result_126989 = invoke(stypy.reporting.localization.Localization(__file__, 33, 35), transform_126984, *[shift_126985], **kwargs_126988)
    
    # Processing the call keyword arguments (line 33)
    kwargs_126990 = {}
    # Getting the type of 'LowLevelCallable' (line 33)
    LowLevelCallable_126982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 18), 'LowLevelCallable', False)
    # Calling LowLevelCallable(args, kwargs) (line 33)
    LowLevelCallable_call_result_126991 = invoke(stypy.reporting.localization.Localization(__file__, 33, 18), LowLevelCallable_126982, *[transform_call_result_126989], **kwargs_126990)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type', LowLevelCallable_call_result_126991)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_43' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_126992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126992)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_43'
    return stypy_return_type_126992

# Assigning a type to the variable '_stypy_temp_lambda_43' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), '_stypy_temp_lambda_43', _stypy_temp_lambda_43)
# Getting the type of '_stypy_temp_lambda_43' (line 33)
_stypy_temp_lambda_43_126993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), '_stypy_temp_lambda_43')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 22), list_126958, _stypy_temp_lambda_43_126993)
# Adding element type (line 29)

@norecursion
def _stypy_temp_lambda_44(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_44'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_44', 34, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_44.stypy_localization = localization
    _stypy_temp_lambda_44.stypy_type_of_self = None
    _stypy_temp_lambda_44.stypy_type_store = module_type_store
    _stypy_temp_lambda_44.stypy_function_name = '_stypy_temp_lambda_44'
    _stypy_temp_lambda_44.stypy_param_names_list = ['shift']
    _stypy_temp_lambda_44.stypy_varargs_param_name = None
    _stypy_temp_lambda_44.stypy_kwargs_param_name = None
    _stypy_temp_lambda_44.stypy_call_defaults = defaults
    _stypy_temp_lambda_44.stypy_call_varargs = varargs
    _stypy_temp_lambda_44.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_44', ['shift'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_44', ['shift'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to from_cython(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of '_cytest' (line 34)
    _cytest_126996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 47), '_cytest', False)
    str_126997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 56), 'str', '_transform')
    
    # Call to transform_capsule(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'shift' (line 34)
    shift_127000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 96), 'shift', False)
    # Processing the call keyword arguments (line 34)
    kwargs_127001 = {}
    # Getting the type of '_cytest' (line 34)
    _cytest_126998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 70), '_cytest', False)
    # Obtaining the member 'transform_capsule' of a type (line 34)
    transform_capsule_126999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 70), _cytest_126998, 'transform_capsule')
    # Calling transform_capsule(args, kwargs) (line 34)
    transform_capsule_call_result_127002 = invoke(stypy.reporting.localization.Localization(__file__, 34, 70), transform_capsule_126999, *[shift_127000], **kwargs_127001)
    
    # Processing the call keyword arguments (line 34)
    kwargs_127003 = {}
    # Getting the type of 'LowLevelCallable' (line 34)
    LowLevelCallable_126994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'LowLevelCallable', False)
    # Obtaining the member 'from_cython' of a type (line 34)
    from_cython_126995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 18), LowLevelCallable_126994, 'from_cython')
    # Calling from_cython(args, kwargs) (line 34)
    from_cython_call_result_127004 = invoke(stypy.reporting.localization.Localization(__file__, 34, 18), from_cython_126995, *[_cytest_126996, str_126997, transform_capsule_call_result_127002], **kwargs_127003)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type', from_cython_call_result_127004)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_44' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_127005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127005)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_44'
    return stypy_return_type_127005

# Assigning a type to the variable '_stypy_temp_lambda_44' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), '_stypy_temp_lambda_44', _stypy_temp_lambda_44)
# Getting the type of '_stypy_temp_lambda_44' (line 34)
_stypy_temp_lambda_44_127006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), '_stypy_temp_lambda_44')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 22), list_126958, _stypy_temp_lambda_44_127006)

# Assigning a type to the variable 'TRANSFORM_FUNCTIONS' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'TRANSFORM_FUNCTIONS', list_126958)

@norecursion
def test_generic_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_generic_filter'
    module_type_store = module_type_store.open_function_context('test_generic_filter', 38, 0, False)
    
    # Passed parameters checking function
    test_generic_filter.stypy_localization = localization
    test_generic_filter.stypy_type_of_self = None
    test_generic_filter.stypy_type_store = module_type_store
    test_generic_filter.stypy_function_name = 'test_generic_filter'
    test_generic_filter.stypy_param_names_list = []
    test_generic_filter.stypy_varargs_param_name = None
    test_generic_filter.stypy_kwargs_param_name = None
    test_generic_filter.stypy_call_defaults = defaults
    test_generic_filter.stypy_call_varargs = varargs
    test_generic_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_generic_filter', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_generic_filter', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_generic_filter(...)' code ##################


    @norecursion
    def filter2d(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'filter2d'
        module_type_store = module_type_store.open_function_context('filter2d', 39, 4, False)
        
        # Passed parameters checking function
        filter2d.stypy_localization = localization
        filter2d.stypy_type_of_self = None
        filter2d.stypy_type_store = module_type_store
        filter2d.stypy_function_name = 'filter2d'
        filter2d.stypy_param_names_list = ['footprint_elements', 'weights']
        filter2d.stypy_varargs_param_name = None
        filter2d.stypy_kwargs_param_name = None
        filter2d.stypy_call_defaults = defaults
        filter2d.stypy_call_varargs = varargs
        filter2d.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'filter2d', ['footprint_elements', 'weights'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'filter2d', localization, ['footprint_elements', 'weights'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'filter2d(...)' code ##################

        
        # Call to sum(...): (line 40)
        # Processing the call keyword arguments (line 40)
        kwargs_127011 = {}
        # Getting the type of 'weights' (line 40)
        weights_127007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'weights', False)
        # Getting the type of 'footprint_elements' (line 40)
        footprint_elements_127008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'footprint_elements', False)
        # Applying the binary operator '*' (line 40)
        result_mul_127009 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 16), '*', weights_127007, footprint_elements_127008)
        
        # Obtaining the member 'sum' of a type (line 40)
        sum_127010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 16), result_mul_127009, 'sum')
        # Calling sum(args, kwargs) (line 40)
        sum_call_result_127012 = invoke(stypy.reporting.localization.Localization(__file__, 40, 16), sum_127010, *[], **kwargs_127011)
        
        # Assigning a type to the variable 'stypy_return_type' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'stypy_return_type', sum_call_result_127012)
        
        # ################# End of 'filter2d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'filter2d' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_127013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_127013)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'filter2d'
        return stypy_return_type_127013

    # Assigning a type to the variable 'filter2d' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'filter2d', filter2d)

    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 42, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['j']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['j'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['j'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Subscript to a Name (line 43):
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 43)
        j_127014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), 'j')
        # Getting the type of 'FILTER2D_FUNCTIONS' (line 43)
        FILTER2D_FUNCTIONS_127015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'FILTER2D_FUNCTIONS')
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___127016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 15), FILTER2D_FUNCTIONS_127015, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_127017 = invoke(stypy.reporting.localization.Localization(__file__, 43, 15), getitem___127016, j_127014)
        
        # Assigning a type to the variable 'func' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'func', subscript_call_result_127017)
        
        # Assigning a Call to a Name (line 45):
        
        # Call to ones(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Obtaining an instance of the builtin type 'tuple' (line 45)
        tuple_127020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 45)
        # Adding element type (line 45)
        int_127021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 22), tuple_127020, int_127021)
        # Adding element type (line 45)
        int_127022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 22), tuple_127020, int_127022)
        
        # Processing the call keyword arguments (line 45)
        kwargs_127023 = {}
        # Getting the type of 'np' (line 45)
        np_127018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'np', False)
        # Obtaining the member 'ones' of a type (line 45)
        ones_127019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 13), np_127018, 'ones')
        # Calling ones(args, kwargs) (line 45)
        ones_call_result_127024 = invoke(stypy.reporting.localization.Localization(__file__, 45, 13), ones_127019, *[tuple_127020], **kwargs_127023)
        
        # Assigning a type to the variable 'im' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'im', ones_call_result_127024)
        
        # Assigning a Num to a Subscript (line 46):
        int_127025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'int')
        # Getting the type of 'im' (line 46)
        im_127026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'im')
        int_127027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 12), 'int')
        slice_127028 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 46, 8), None, int_127027, None)
        int_127029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 16), 'int')
        slice_127030 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 46, 8), None, int_127029, None)
        # Storing an element on a container (line 46)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 8), im_127026, ((slice_127028, slice_127030), int_127025))
        
        # Assigning a Call to a Name (line 47):
        
        # Call to array(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_127033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_127034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        int_127035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 30), list_127034, int_127035)
        # Adding element type (line 47)
        int_127036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 30), list_127034, int_127036)
        # Adding element type (line 47)
        int_127037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 30), list_127034, int_127037)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 29), list_127033, list_127034)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_127038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        int_127039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 41), list_127038, int_127039)
        # Adding element type (line 47)
        int_127040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 41), list_127038, int_127040)
        # Adding element type (line 47)
        int_127041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 41), list_127038, int_127041)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 29), list_127033, list_127038)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_127042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        int_127043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 52), list_127042, int_127043)
        # Adding element type (line 47)
        int_127044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 52), list_127042, int_127044)
        # Adding element type (line 47)
        int_127045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 52), list_127042, int_127045)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 29), list_127033, list_127042)
        
        # Processing the call keyword arguments (line 47)
        kwargs_127046 = {}
        # Getting the type of 'np' (line 47)
        np_127031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 20), 'np', False)
        # Obtaining the member 'array' of a type (line 47)
        array_127032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 20), np_127031, 'array')
        # Calling array(args, kwargs) (line 47)
        array_call_result_127047 = invoke(stypy.reporting.localization.Localization(__file__, 47, 20), array_127032, *[list_127033], **kwargs_127046)
        
        # Assigning a type to the variable 'footprint' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'footprint', array_call_result_127047)
        
        # Assigning a Call to a Name (line 48):
        
        # Call to count_nonzero(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'footprint' (line 48)
        footprint_127050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 42), 'footprint', False)
        # Processing the call keyword arguments (line 48)
        kwargs_127051 = {}
        # Getting the type of 'np' (line 48)
        np_127048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'np', False)
        # Obtaining the member 'count_nonzero' of a type (line 48)
        count_nonzero_127049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 25), np_127048, 'count_nonzero')
        # Calling count_nonzero(args, kwargs) (line 48)
        count_nonzero_call_result_127052 = invoke(stypy.reporting.localization.Localization(__file__, 48, 25), count_nonzero_127049, *[footprint_127050], **kwargs_127051)
        
        # Assigning a type to the variable 'footprint_size' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'footprint_size', count_nonzero_call_result_127052)
        
        # Assigning a BinOp to a Name (line 49):
        
        # Call to ones(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'footprint_size' (line 49)
        footprint_size_127055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'footprint_size', False)
        # Processing the call keyword arguments (line 49)
        kwargs_127056 = {}
        # Getting the type of 'np' (line 49)
        np_127053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'np', False)
        # Obtaining the member 'ones' of a type (line 49)
        ones_127054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 18), np_127053, 'ones')
        # Calling ones(args, kwargs) (line 49)
        ones_call_result_127057 = invoke(stypy.reporting.localization.Localization(__file__, 49, 18), ones_127054, *[footprint_size_127055], **kwargs_127056)
        
        # Getting the type of 'footprint_size' (line 49)
        footprint_size_127058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 42), 'footprint_size')
        # Applying the binary operator 'div' (line 49)
        result_div_127059 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 18), 'div', ones_call_result_127057, footprint_size_127058)
        
        # Assigning a type to the variable 'weights' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'weights', result_div_127059)
        
        # Assigning a Call to a Name (line 51):
        
        # Call to generic_filter(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'im' (line 51)
        im_127062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 37), 'im', False)
        
        # Call to func(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'weights' (line 51)
        weights_127064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 46), 'weights', False)
        # Processing the call keyword arguments (line 51)
        kwargs_127065 = {}
        # Getting the type of 'func' (line 51)
        func_127063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 41), 'func', False)
        # Calling func(args, kwargs) (line 51)
        func_call_result_127066 = invoke(stypy.reporting.localization.Localization(__file__, 51, 41), func_127063, *[weights_127064], **kwargs_127065)
        
        # Processing the call keyword arguments (line 51)
        # Getting the type of 'footprint' (line 52)
        footprint_127067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 47), 'footprint', False)
        keyword_127068 = footprint_127067
        kwargs_127069 = {'footprint': keyword_127068}
        # Getting the type of 'ndimage' (line 51)
        ndimage_127060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'ndimage', False)
        # Obtaining the member 'generic_filter' of a type (line 51)
        generic_filter_127061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 14), ndimage_127060, 'generic_filter')
        # Calling generic_filter(args, kwargs) (line 51)
        generic_filter_call_result_127070 = invoke(stypy.reporting.localization.Localization(__file__, 51, 14), generic_filter_127061, *[im_127062, func_call_result_127066], **kwargs_127069)
        
        # Assigning a type to the variable 'res' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'res', generic_filter_call_result_127070)
        
        # Assigning a Call to a Name (line 53):
        
        # Call to generic_filter(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'im' (line 53)
        im_127073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 37), 'im', False)
        # Getting the type of 'filter2d' (line 53)
        filter2d_127074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 41), 'filter2d', False)
        # Processing the call keyword arguments (line 53)
        # Getting the type of 'footprint' (line 53)
        footprint_127075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 61), 'footprint', False)
        keyword_127076 = footprint_127075
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_127077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        # Getting the type of 'weights' (line 54)
        weights_127078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 54), 'weights', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 54), tuple_127077, weights_127078)
        
        keyword_127079 = tuple_127077
        kwargs_127080 = {'footprint': keyword_127076, 'extra_arguments': keyword_127079}
        # Getting the type of 'ndimage' (line 53)
        ndimage_127071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'ndimage', False)
        # Obtaining the member 'generic_filter' of a type (line 53)
        generic_filter_127072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 14), ndimage_127071, 'generic_filter')
        # Calling generic_filter(args, kwargs) (line 53)
        generic_filter_call_result_127081 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), generic_filter_127072, *[im_127073, filter2d_127074], **kwargs_127080)
        
        # Assigning a type to the variable 'std' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'std', generic_filter_call_result_127081)
        
        # Call to assert_allclose(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'res' (line 55)
        res_127083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'res', False)
        # Getting the type of 'std' (line 55)
        std_127084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 29), 'std', False)
        # Processing the call keyword arguments (line 55)
        
        # Call to format(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'j' (line 55)
        j_127087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 62), 'j', False)
        # Processing the call keyword arguments (line 55)
        kwargs_127088 = {}
        str_127085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 42), 'str', '#{} failed')
        # Obtaining the member 'format' of a type (line 55)
        format_127086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 42), str_127085, 'format')
        # Calling format(args, kwargs) (line 55)
        format_call_result_127089 = invoke(stypy.reporting.localization.Localization(__file__, 55, 42), format_127086, *[j_127087], **kwargs_127088)
        
        keyword_127090 = format_call_result_127089
        kwargs_127091 = {'err_msg': keyword_127090}
        # Getting the type of 'assert_allclose' (line 55)
        assert_allclose_127082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 55)
        assert_allclose_call_result_127092 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), assert_allclose_127082, *[res_127083, std_127084], **kwargs_127091)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_127093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_127093)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_127093

    # Assigning a type to the variable 'check' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'check', check)
    
    
    # Call to enumerate(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'FILTER2D_FUNCTIONS' (line 57)
    FILTER2D_FUNCTIONS_127095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 'FILTER2D_FUNCTIONS', False)
    # Processing the call keyword arguments (line 57)
    kwargs_127096 = {}
    # Getting the type of 'enumerate' (line 57)
    enumerate_127094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 57)
    enumerate_call_result_127097 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), enumerate_127094, *[FILTER2D_FUNCTIONS_127095], **kwargs_127096)
    
    # Testing the type of a for loop iterable (line 57)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 57, 4), enumerate_call_result_127097)
    # Getting the type of the for loop variable (line 57)
    for_loop_var_127098 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 57, 4), enumerate_call_result_127097)
    # Assigning a type to the variable 'j' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 4), for_loop_var_127098))
    # Assigning a type to the variable 'func' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 4), for_loop_var_127098))
    # SSA begins for a for statement (line 57)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'j' (line 58)
    j_127100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 14), 'j', False)
    # Processing the call keyword arguments (line 58)
    kwargs_127101 = {}
    # Getting the type of 'check' (line 58)
    check_127099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'check', False)
    # Calling check(args, kwargs) (line 58)
    check_call_result_127102 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), check_127099, *[j_127100], **kwargs_127101)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_generic_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_generic_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_127103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127103)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_generic_filter'
    return stypy_return_type_127103

# Assigning a type to the variable 'test_generic_filter' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'test_generic_filter', test_generic_filter)

@norecursion
def test_generic_filter1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_generic_filter1d'
    module_type_store = module_type_store.open_function_context('test_generic_filter1d', 61, 0, False)
    
    # Passed parameters checking function
    test_generic_filter1d.stypy_localization = localization
    test_generic_filter1d.stypy_type_of_self = None
    test_generic_filter1d.stypy_type_store = module_type_store
    test_generic_filter1d.stypy_function_name = 'test_generic_filter1d'
    test_generic_filter1d.stypy_param_names_list = []
    test_generic_filter1d.stypy_varargs_param_name = None
    test_generic_filter1d.stypy_kwargs_param_name = None
    test_generic_filter1d.stypy_call_defaults = defaults
    test_generic_filter1d.stypy_call_varargs = varargs
    test_generic_filter1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_generic_filter1d', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_generic_filter1d', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_generic_filter1d(...)' code ##################


    @norecursion
    def filter1d(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'filter1d'
        module_type_store = module_type_store.open_function_context('filter1d', 62, 4, False)
        
        # Passed parameters checking function
        filter1d.stypy_localization = localization
        filter1d.stypy_type_of_self = None
        filter1d.stypy_type_store = module_type_store
        filter1d.stypy_function_name = 'filter1d'
        filter1d.stypy_param_names_list = ['input_line', 'output_line', 'filter_size']
        filter1d.stypy_varargs_param_name = None
        filter1d.stypy_kwargs_param_name = None
        filter1d.stypy_call_defaults = defaults
        filter1d.stypy_call_varargs = varargs
        filter1d.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'filter1d', ['input_line', 'output_line', 'filter_size'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'filter1d', localization, ['input_line', 'output_line', 'filter_size'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'filter1d(...)' code ##################

        
        
        # Call to range(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'output_line' (line 63)
        output_line_127105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 23), 'output_line', False)
        # Obtaining the member 'size' of a type (line 63)
        size_127106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 23), output_line_127105, 'size')
        # Processing the call keyword arguments (line 63)
        kwargs_127107 = {}
        # Getting the type of 'range' (line 63)
        range_127104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'range', False)
        # Calling range(args, kwargs) (line 63)
        range_call_result_127108 = invoke(stypy.reporting.localization.Localization(__file__, 63, 17), range_127104, *[size_127106], **kwargs_127107)
        
        # Testing the type of a for loop iterable (line 63)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 8), range_call_result_127108)
        # Getting the type of the for loop variable (line 63)
        for_loop_var_127109 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 8), range_call_result_127108)
        # Assigning a type to the variable 'i' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'i', for_loop_var_127109)
        # SSA begins for a for statement (line 63)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 64):
        int_127110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 29), 'int')
        # Getting the type of 'output_line' (line 64)
        output_line_127111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'output_line')
        # Getting the type of 'i' (line 64)
        i_127112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'i')
        # Storing an element on a container (line 64)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 12), output_line_127111, (i_127112, int_127110))
        
        
        # Call to range(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'filter_size' (line 65)
        filter_size_127114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'filter_size', False)
        # Processing the call keyword arguments (line 65)
        kwargs_127115 = {}
        # Getting the type of 'range' (line 65)
        range_127113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'range', False)
        # Calling range(args, kwargs) (line 65)
        range_call_result_127116 = invoke(stypy.reporting.localization.Localization(__file__, 65, 21), range_127113, *[filter_size_127114], **kwargs_127115)
        
        # Testing the type of a for loop iterable (line 65)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 65, 12), range_call_result_127116)
        # Getting the type of the for loop variable (line 65)
        for_loop_var_127117 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 65, 12), range_call_result_127116)
        # Assigning a type to the variable 'j' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'j', for_loop_var_127117)
        # SSA begins for a for statement (line 65)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'output_line' (line 66)
        output_line_127118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'output_line')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 66)
        i_127119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'i')
        # Getting the type of 'output_line' (line 66)
        output_line_127120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'output_line')
        # Obtaining the member '__getitem__' of a type (line 66)
        getitem___127121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 16), output_line_127120, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 66)
        subscript_call_result_127122 = invoke(stypy.reporting.localization.Localization(__file__, 66, 16), getitem___127121, i_127119)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 66)
        i_127123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 45), 'i')
        # Getting the type of 'j' (line 66)
        j_127124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 47), 'j')
        # Applying the binary operator '+' (line 66)
        result_add_127125 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 45), '+', i_127123, j_127124)
        
        # Getting the type of 'input_line' (line 66)
        input_line_127126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'input_line')
        # Obtaining the member '__getitem__' of a type (line 66)
        getitem___127127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 34), input_line_127126, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 66)
        subscript_call_result_127128 = invoke(stypy.reporting.localization.Localization(__file__, 66, 34), getitem___127127, result_add_127125)
        
        # Applying the binary operator '+=' (line 66)
        result_iadd_127129 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 16), '+=', subscript_call_result_127122, subscript_call_result_127128)
        # Getting the type of 'output_line' (line 66)
        output_line_127130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'output_line')
        # Getting the type of 'i' (line 66)
        i_127131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'i')
        # Storing an element on a container (line 66)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 16), output_line_127130, (i_127131, result_iadd_127129))
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'output_line' (line 67)
        output_line_127132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'output_line')
        # Getting the type of 'filter_size' (line 67)
        filter_size_127133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'filter_size')
        # Applying the binary operator 'div=' (line 67)
        result_div_127134 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 8), 'div=', output_line_127132, filter_size_127133)
        # Assigning a type to the variable 'output_line' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'output_line', result_div_127134)
        
        
        # ################# End of 'filter1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'filter1d' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_127135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_127135)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'filter1d'
        return stypy_return_type_127135

    # Assigning a type to the variable 'filter1d' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'filter1d', filter1d)

    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 69, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['j']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['j'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['j'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Subscript to a Name (line 70):
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 70)
        j_127136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 34), 'j')
        # Getting the type of 'FILTER1D_FUNCTIONS' (line 70)
        FILTER1D_FUNCTIONS_127137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'FILTER1D_FUNCTIONS')
        # Obtaining the member '__getitem__' of a type (line 70)
        getitem___127138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 15), FILTER1D_FUNCTIONS_127137, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 70)
        subscript_call_result_127139 = invoke(stypy.reporting.localization.Localization(__file__, 70, 15), getitem___127138, j_127136)
        
        # Assigning a type to the variable 'func' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'func', subscript_call_result_127139)
        
        # Assigning a Call to a Name (line 72):
        
        # Call to tile(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Call to hstack(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_127144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        
        # Call to zeros(...): (line 72)
        # Processing the call arguments (line 72)
        int_127147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 41), 'int')
        # Processing the call keyword arguments (line 72)
        kwargs_127148 = {}
        # Getting the type of 'np' (line 72)
        np_127145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'np', False)
        # Obtaining the member 'zeros' of a type (line 72)
        zeros_127146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 32), np_127145, 'zeros')
        # Calling zeros(args, kwargs) (line 72)
        zeros_call_result_127149 = invoke(stypy.reporting.localization.Localization(__file__, 72, 32), zeros_127146, *[int_127147], **kwargs_127148)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 32), tuple_127144, zeros_call_result_127149)
        # Adding element type (line 72)
        
        # Call to ones(...): (line 72)
        # Processing the call arguments (line 72)
        int_127152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 54), 'int')
        # Processing the call keyword arguments (line 72)
        kwargs_127153 = {}
        # Getting the type of 'np' (line 72)
        np_127150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 46), 'np', False)
        # Obtaining the member 'ones' of a type (line 72)
        ones_127151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 46), np_127150, 'ones')
        # Calling ones(args, kwargs) (line 72)
        ones_call_result_127154 = invoke(stypy.reporting.localization.Localization(__file__, 72, 46), ones_127151, *[int_127152], **kwargs_127153)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 32), tuple_127144, ones_call_result_127154)
        
        # Processing the call keyword arguments (line 72)
        kwargs_127155 = {}
        # Getting the type of 'np' (line 72)
        np_127142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'np', False)
        # Obtaining the member 'hstack' of a type (line 72)
        hstack_127143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 21), np_127142, 'hstack')
        # Calling hstack(args, kwargs) (line 72)
        hstack_call_result_127156 = invoke(stypy.reporting.localization.Localization(__file__, 72, 21), hstack_127143, *[tuple_127144], **kwargs_127155)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_127157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 62), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        int_127158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 62), tuple_127157, int_127158)
        # Adding element type (line 72)
        int_127159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 62), tuple_127157, int_127159)
        
        # Processing the call keyword arguments (line 72)
        kwargs_127160 = {}
        # Getting the type of 'np' (line 72)
        np_127140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'np', False)
        # Obtaining the member 'tile' of a type (line 72)
        tile_127141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), np_127140, 'tile')
        # Calling tile(args, kwargs) (line 72)
        tile_call_result_127161 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), tile_127141, *[hstack_call_result_127156, tuple_127157], **kwargs_127160)
        
        # Assigning a type to the variable 'im' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'im', tile_call_result_127161)
        
        # Assigning a Num to a Name (line 73):
        int_127162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 22), 'int')
        # Assigning a type to the variable 'filter_size' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'filter_size', int_127162)
        
        # Assigning a Call to a Name (line 75):
        
        # Call to generic_filter1d(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'im' (line 75)
        im_127165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 39), 'im', False)
        
        # Call to func(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'filter_size' (line 75)
        filter_size_127167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 48), 'filter_size', False)
        # Processing the call keyword arguments (line 75)
        kwargs_127168 = {}
        # Getting the type of 'func' (line 75)
        func_127166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 43), 'func', False)
        # Calling func(args, kwargs) (line 75)
        func_call_result_127169 = invoke(stypy.reporting.localization.Localization(__file__, 75, 43), func_127166, *[filter_size_127167], **kwargs_127168)
        
        # Getting the type of 'filter_size' (line 76)
        filter_size_127170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 39), 'filter_size', False)
        # Processing the call keyword arguments (line 75)
        kwargs_127171 = {}
        # Getting the type of 'ndimage' (line 75)
        ndimage_127163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'ndimage', False)
        # Obtaining the member 'generic_filter1d' of a type (line 75)
        generic_filter1d_127164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 14), ndimage_127163, 'generic_filter1d')
        # Calling generic_filter1d(args, kwargs) (line 75)
        generic_filter1d_call_result_127172 = invoke(stypy.reporting.localization.Localization(__file__, 75, 14), generic_filter1d_127164, *[im_127165, func_call_result_127169, filter_size_127170], **kwargs_127171)
        
        # Assigning a type to the variable 'res' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'res', generic_filter1d_call_result_127172)
        
        # Assigning a Call to a Name (line 77):
        
        # Call to generic_filter1d(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'im' (line 77)
        im_127175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 39), 'im', False)
        # Getting the type of 'filter1d' (line 77)
        filter1d_127176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 43), 'filter1d', False)
        # Getting the type of 'filter_size' (line 77)
        filter_size_127177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 53), 'filter_size', False)
        # Processing the call keyword arguments (line 77)
        
        # Obtaining an instance of the builtin type 'tuple' (line 78)
        tuple_127178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 78)
        # Adding element type (line 78)
        # Getting the type of 'filter_size' (line 78)
        filter_size_127179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 56), 'filter_size', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 56), tuple_127178, filter_size_127179)
        
        keyword_127180 = tuple_127178
        kwargs_127181 = {'extra_arguments': keyword_127180}
        # Getting the type of 'ndimage' (line 77)
        ndimage_127173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), 'ndimage', False)
        # Obtaining the member 'generic_filter1d' of a type (line 77)
        generic_filter1d_127174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 14), ndimage_127173, 'generic_filter1d')
        # Calling generic_filter1d(args, kwargs) (line 77)
        generic_filter1d_call_result_127182 = invoke(stypy.reporting.localization.Localization(__file__, 77, 14), generic_filter1d_127174, *[im_127175, filter1d_127176, filter_size_127177], **kwargs_127181)
        
        # Assigning a type to the variable 'std' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'std', generic_filter1d_call_result_127182)
        
        # Call to assert_allclose(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'res' (line 79)
        res_127184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'res', False)
        # Getting the type of 'std' (line 79)
        std_127185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'std', False)
        # Processing the call keyword arguments (line 79)
        
        # Call to format(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'j' (line 79)
        j_127188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 62), 'j', False)
        # Processing the call keyword arguments (line 79)
        kwargs_127189 = {}
        str_127186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 42), 'str', '#{} failed')
        # Obtaining the member 'format' of a type (line 79)
        format_127187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 42), str_127186, 'format')
        # Calling format(args, kwargs) (line 79)
        format_call_result_127190 = invoke(stypy.reporting.localization.Localization(__file__, 79, 42), format_127187, *[j_127188], **kwargs_127189)
        
        keyword_127191 = format_call_result_127190
        kwargs_127192 = {'err_msg': keyword_127191}
        # Getting the type of 'assert_allclose' (line 79)
        assert_allclose_127183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 79)
        assert_allclose_call_result_127193 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), assert_allclose_127183, *[res_127184, std_127185], **kwargs_127192)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_127194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_127194)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_127194

    # Assigning a type to the variable 'check' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'check', check)
    
    
    # Call to enumerate(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'FILTER1D_FUNCTIONS' (line 81)
    FILTER1D_FUNCTIONS_127196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 29), 'FILTER1D_FUNCTIONS', False)
    # Processing the call keyword arguments (line 81)
    kwargs_127197 = {}
    # Getting the type of 'enumerate' (line 81)
    enumerate_127195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 81)
    enumerate_call_result_127198 = invoke(stypy.reporting.localization.Localization(__file__, 81, 19), enumerate_127195, *[FILTER1D_FUNCTIONS_127196], **kwargs_127197)
    
    # Testing the type of a for loop iterable (line 81)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 81, 4), enumerate_call_result_127198)
    # Getting the type of the for loop variable (line 81)
    for_loop_var_127199 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 81, 4), enumerate_call_result_127198)
    # Assigning a type to the variable 'j' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 4), for_loop_var_127199))
    # Assigning a type to the variable 'func' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 4), for_loop_var_127199))
    # SSA begins for a for statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'j' (line 82)
    j_127201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 14), 'j', False)
    # Processing the call keyword arguments (line 82)
    kwargs_127202 = {}
    # Getting the type of 'check' (line 82)
    check_127200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'check', False)
    # Calling check(args, kwargs) (line 82)
    check_call_result_127203 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), check_127200, *[j_127201], **kwargs_127202)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_generic_filter1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_generic_filter1d' in the type store
    # Getting the type of 'stypy_return_type' (line 61)
    stypy_return_type_127204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127204)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_generic_filter1d'
    return stypy_return_type_127204

# Assigning a type to the variable 'test_generic_filter1d' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'test_generic_filter1d', test_generic_filter1d)

@norecursion
def test_geometric_transform(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_geometric_transform'
    module_type_store = module_type_store.open_function_context('test_geometric_transform', 85, 0, False)
    
    # Passed parameters checking function
    test_geometric_transform.stypy_localization = localization
    test_geometric_transform.stypy_type_of_self = None
    test_geometric_transform.stypy_type_store = module_type_store
    test_geometric_transform.stypy_function_name = 'test_geometric_transform'
    test_geometric_transform.stypy_param_names_list = []
    test_geometric_transform.stypy_varargs_param_name = None
    test_geometric_transform.stypy_kwargs_param_name = None
    test_geometric_transform.stypy_call_defaults = defaults
    test_geometric_transform.stypy_call_varargs = varargs
    test_geometric_transform.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_geometric_transform', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_geometric_transform', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_geometric_transform(...)' code ##################


    @norecursion
    def transform(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'transform'
        module_type_store = module_type_store.open_function_context('transform', 86, 4, False)
        
        # Passed parameters checking function
        transform.stypy_localization = localization
        transform.stypy_type_of_self = None
        transform.stypy_type_store = module_type_store
        transform.stypy_function_name = 'transform'
        transform.stypy_param_names_list = ['output_coordinates', 'shift']
        transform.stypy_varargs_param_name = None
        transform.stypy_kwargs_param_name = None
        transform.stypy_call_defaults = defaults
        transform.stypy_call_varargs = varargs
        transform.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'transform', ['output_coordinates', 'shift'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transform', localization, ['output_coordinates', 'shift'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transform(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 87)
        tuple_127205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 87)
        # Adding element type (line 87)
        
        # Obtaining the type of the subscript
        int_127206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 34), 'int')
        # Getting the type of 'output_coordinates' (line 87)
        output_coordinates_127207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'output_coordinates')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___127208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 15), output_coordinates_127207, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_127209 = invoke(stypy.reporting.localization.Localization(__file__, 87, 15), getitem___127208, int_127206)
        
        # Getting the type of 'shift' (line 87)
        shift_127210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 39), 'shift')
        # Applying the binary operator '-' (line 87)
        result_sub_127211 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 15), '-', subscript_call_result_127209, shift_127210)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 15), tuple_127205, result_sub_127211)
        # Adding element type (line 87)
        
        # Obtaining the type of the subscript
        int_127212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 65), 'int')
        # Getting the type of 'output_coordinates' (line 87)
        output_coordinates_127213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 46), 'output_coordinates')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___127214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 46), output_coordinates_127213, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_127215 = invoke(stypy.reporting.localization.Localization(__file__, 87, 46), getitem___127214, int_127212)
        
        # Getting the type of 'shift' (line 87)
        shift_127216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 70), 'shift')
        # Applying the binary operator '-' (line 87)
        result_sub_127217 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 46), '-', subscript_call_result_127215, shift_127216)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 15), tuple_127205, result_sub_127217)
        
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', tuple_127205)
        
        # ################# End of 'transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transform' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_127218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_127218)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transform'
        return stypy_return_type_127218

    # Assigning a type to the variable 'transform' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'transform', transform)

    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 89, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['j']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['j'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['j'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Subscript to a Name (line 90):
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 90)
        j_127219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 35), 'j')
        # Getting the type of 'TRANSFORM_FUNCTIONS' (line 90)
        TRANSFORM_FUNCTIONS_127220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'TRANSFORM_FUNCTIONS')
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___127221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 15), TRANSFORM_FUNCTIONS_127220, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_127222 = invoke(stypy.reporting.localization.Localization(__file__, 90, 15), getitem___127221, j_127219)
        
        # Assigning a type to the variable 'func' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'func', subscript_call_result_127222)
        
        # Assigning a Call to a Name (line 92):
        
        # Call to astype(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'np' (line 92)
        np_127234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 48), 'np', False)
        # Obtaining the member 'float64' of a type (line 92)
        float64_127235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 48), np_127234, 'float64')
        # Processing the call keyword arguments (line 92)
        kwargs_127236 = {}
        
        # Call to reshape(...): (line 92)
        # Processing the call arguments (line 92)
        int_127229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 35), 'int')
        int_127230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 38), 'int')
        # Processing the call keyword arguments (line 92)
        kwargs_127231 = {}
        
        # Call to arange(...): (line 92)
        # Processing the call arguments (line 92)
        int_127225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 23), 'int')
        # Processing the call keyword arguments (line 92)
        kwargs_127226 = {}
        # Getting the type of 'np' (line 92)
        np_127223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'np', False)
        # Obtaining the member 'arange' of a type (line 92)
        arange_127224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), np_127223, 'arange')
        # Calling arange(args, kwargs) (line 92)
        arange_call_result_127227 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), arange_127224, *[int_127225], **kwargs_127226)
        
        # Obtaining the member 'reshape' of a type (line 92)
        reshape_127228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), arange_call_result_127227, 'reshape')
        # Calling reshape(args, kwargs) (line 92)
        reshape_call_result_127232 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), reshape_127228, *[int_127229, int_127230], **kwargs_127231)
        
        # Obtaining the member 'astype' of a type (line 92)
        astype_127233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), reshape_call_result_127232, 'astype')
        # Calling astype(args, kwargs) (line 92)
        astype_call_result_127237 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), astype_127233, *[float64_127235], **kwargs_127236)
        
        # Assigning a type to the variable 'im' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'im', astype_call_result_127237)
        
        # Assigning a Num to a Name (line 93):
        float_127238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 16), 'float')
        # Assigning a type to the variable 'shift' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'shift', float_127238)
        
        # Assigning a Call to a Name (line 95):
        
        # Call to geometric_transform(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'im' (line 95)
        im_127241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 42), 'im', False)
        
        # Call to func(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'shift' (line 95)
        shift_127243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 51), 'shift', False)
        # Processing the call keyword arguments (line 95)
        kwargs_127244 = {}
        # Getting the type of 'func' (line 95)
        func_127242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 46), 'func', False)
        # Calling func(args, kwargs) (line 95)
        func_call_result_127245 = invoke(stypy.reporting.localization.Localization(__file__, 95, 46), func_127242, *[shift_127243], **kwargs_127244)
        
        # Processing the call keyword arguments (line 95)
        kwargs_127246 = {}
        # Getting the type of 'ndimage' (line 95)
        ndimage_127239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 14), 'ndimage', False)
        # Obtaining the member 'geometric_transform' of a type (line 95)
        geometric_transform_127240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 14), ndimage_127239, 'geometric_transform')
        # Calling geometric_transform(args, kwargs) (line 95)
        geometric_transform_call_result_127247 = invoke(stypy.reporting.localization.Localization(__file__, 95, 14), geometric_transform_127240, *[im_127241, func_call_result_127245], **kwargs_127246)
        
        # Assigning a type to the variable 'res' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'res', geometric_transform_call_result_127247)
        
        # Assigning a Call to a Name (line 96):
        
        # Call to geometric_transform(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'im' (line 96)
        im_127250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 42), 'im', False)
        # Getting the type of 'transform' (line 96)
        transform_127251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 46), 'transform', False)
        # Processing the call keyword arguments (line 96)
        
        # Obtaining an instance of the builtin type 'tuple' (line 96)
        tuple_127252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 74), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 96)
        # Adding element type (line 96)
        # Getting the type of 'shift' (line 96)
        shift_127253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 74), 'shift', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 74), tuple_127252, shift_127253)
        
        keyword_127254 = tuple_127252
        kwargs_127255 = {'extra_arguments': keyword_127254}
        # Getting the type of 'ndimage' (line 96)
        ndimage_127248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 14), 'ndimage', False)
        # Obtaining the member 'geometric_transform' of a type (line 96)
        geometric_transform_127249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 14), ndimage_127248, 'geometric_transform')
        # Calling geometric_transform(args, kwargs) (line 96)
        geometric_transform_call_result_127256 = invoke(stypy.reporting.localization.Localization(__file__, 96, 14), geometric_transform_127249, *[im_127250, transform_127251], **kwargs_127255)
        
        # Assigning a type to the variable 'std' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'std', geometric_transform_call_result_127256)
        
        # Call to assert_allclose(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'res' (line 97)
        res_127258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'res', False)
        # Getting the type of 'std' (line 97)
        std_127259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 29), 'std', False)
        # Processing the call keyword arguments (line 97)
        
        # Call to format(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'j' (line 97)
        j_127262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 62), 'j', False)
        # Processing the call keyword arguments (line 97)
        kwargs_127263 = {}
        str_127260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 42), 'str', '#{} failed')
        # Obtaining the member 'format' of a type (line 97)
        format_127261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 42), str_127260, 'format')
        # Calling format(args, kwargs) (line 97)
        format_call_result_127264 = invoke(stypy.reporting.localization.Localization(__file__, 97, 42), format_127261, *[j_127262], **kwargs_127263)
        
        keyword_127265 = format_call_result_127264
        kwargs_127266 = {'err_msg': keyword_127265}
        # Getting the type of 'assert_allclose' (line 97)
        assert_allclose_127257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 97)
        assert_allclose_call_result_127267 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assert_allclose_127257, *[res_127258, std_127259], **kwargs_127266)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_127268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_127268)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_127268

    # Assigning a type to the variable 'check' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'check', check)
    
    
    # Call to enumerate(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'TRANSFORM_FUNCTIONS' (line 99)
    TRANSFORM_FUNCTIONS_127270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'TRANSFORM_FUNCTIONS', False)
    # Processing the call keyword arguments (line 99)
    kwargs_127271 = {}
    # Getting the type of 'enumerate' (line 99)
    enumerate_127269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 99)
    enumerate_call_result_127272 = invoke(stypy.reporting.localization.Localization(__file__, 99, 19), enumerate_127269, *[TRANSFORM_FUNCTIONS_127270], **kwargs_127271)
    
    # Testing the type of a for loop iterable (line 99)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 4), enumerate_call_result_127272)
    # Getting the type of the for loop variable (line 99)
    for_loop_var_127273 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 4), enumerate_call_result_127272)
    # Assigning a type to the variable 'j' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 4), for_loop_var_127273))
    # Assigning a type to the variable 'func' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 4), for_loop_var_127273))
    # SSA begins for a for statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'j' (line 100)
    j_127275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 'j', False)
    # Processing the call keyword arguments (line 100)
    kwargs_127276 = {}
    # Getting the type of 'check' (line 100)
    check_127274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'check', False)
    # Calling check(args, kwargs) (line 100)
    check_call_result_127277 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), check_127274, *[j_127275], **kwargs_127276)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_geometric_transform(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_geometric_transform' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_127278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127278)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_geometric_transform'
    return stypy_return_type_127278

# Assigning a type to the variable 'test_geometric_transform' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'test_geometric_transform', test_geometric_transform)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
